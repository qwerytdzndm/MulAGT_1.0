from __future__ import annotations

import copy
import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import BoundedSemaphore, Condition, Lock, RLock
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .agents import AgentSuite
from .config import Settings
from .models import ApprovalDecision, IssueRequest, RunCreated
from .state import STATE_SCHEMA_VERSION
from .workflow import build_workflow


TERMINAL_STATUSES = {
    "completed",
    "cancelled",
    "review_rejected",
    "failed_rolled_back",
    "failed",
}
HUMAN_CHECKPOINT_STATUSES = {"awaiting_approval"}
WAITABLE_STATUSES = TERMINAL_STATUSES | HUMAN_CHECKPOINT_STATUSES


@dataclass
class RunRuntime:
    run_id: str
    status: str
    state: dict[str, Any]
    interrupt: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False
    condition: Condition = field(default_factory=Condition)
    operation_lock: Lock = field(default_factory=Lock)
    future: Future | None = None


class MulagtService:
    """Local runner for the retained repository-maintenance workflow."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.database = self.settings.runtime_dir / "checkpoints.sqlite3"
        self.suite = AgentSuite(self.settings)
        self._runtimes: dict[str, RunRuntime] = {}
        self._runtimes_lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.max_workers,
            thread_name_prefix="mul-run",
        )
        self._run_slots = BoundedSemaphore(self.settings.max_pending_runs)
        self._closed = False
        self._setup_database()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.suite.close()

    def __enter__(self) -> "MulagtService":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def start(self, request: IssueRequest) -> RunCreated:
        self._ensure_open()
        mul_path = Path(request.mul_path).resolve()
        if self.settings.allowed_workspace_roots:
            allowed = False
            for workspace_root in self.settings.allowed_workspace_roots:
                try:
                    mul_path.relative_to(workspace_root.resolve())
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed:
                raise PermissionError(
                    "repository is outside MUL_WORKSPACE_ROOTS"
                )
        mode = request.llm_mode or self.settings.llm_mode
        if mode not in {"mock", "deepseek"}:
            raise ValueError("llm_mode must be 'mock' or 'deepseek'")

        run_id = uuid.uuid4().hex
        initial: dict[str, Any] = {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "run_id": run_id,
            "mul_path": str(mul_path),
            "issue": request.issue,
            "constraints": request.constraints,
            "test_command": request.test_command,
            "llm_mode": mode,
            "status": "queued",
            "approved": False,
            "traces": [],
        }
        runtime = RunRuntime(run_id=run_id, status="queued", state=initial)
        with self._runtimes_lock:
            self._runtimes[run_id] = runtime
        self._publish(
            runtime,
            "run_status",
            {
                "status": "queued",
                "summary": "Run accepted and queued for background execution.",
            },
        )
        try:
            runtime.future = self._submit_graph(runtime, initial, False)
        except Exception:
            with self._runtimes_lock:
                self._runtimes.pop(run_id, None)
            raise
        return self._runtime_response(runtime)

    def decide(self, run_id: str, decision: ApprovalDecision) -> RunCreated:
        self._ensure_open()
        checkpoint_response, next_nodes = self._checkpoint_response(run_id)
        if not checkpoint_response.state:
            raise KeyError(f"Unknown run_id: {run_id}")
        if "approval" not in next_nodes:
            raise RuntimeError(
                "Run is not awaiting approval; current status is "
                f"{checkpoint_response.status}"
            )
        expected_digest = str(
            (checkpoint_response.interrupt or {}).get("artifact_digest") or ""
        )
        if expected_digest and decision.artifact_digest != expected_digest:
            raise RuntimeError(
                "Approval artifact digest is missing or stale; refresh the "
                "checkpoint before deciding"
            )
        runtime = self._ensure_runtime(checkpoint_response)
        with runtime.condition:
            if runtime.status == "resuming":
                raise RuntimeError("A decision is already resuming this run")
            runtime.status = "resuming"
            runtime.state["status"] = "resuming"
            runtime.interrupt = None
        self._publish(
            runtime,
            "human_decision",
            {
                "approved": decision.approved,
                "comment": decision.comment,
                "actor_id": decision.actor_id,
                "artifact_digest": decision.artifact_digest,
            },
        )
        self._publish(
            runtime,
            "run_status",
            {
                "status": "resuming",
                "summary": "Human decision accepted; workflow is resuming.",
            },
        )
        try:
            runtime.future = self._submit_graph(
                runtime,
                Command(resume=decision.model_dump()),
                True,
            )
        except Exception:
            with runtime.condition:
                runtime.status = checkpoint_response.status
                runtime.state = copy.deepcopy(checkpoint_response.state)
                runtime.interrupt = copy.deepcopy(
                    checkpoint_response.interrupt
                )
            raise
        return self._runtime_response(runtime)

    def get(self, run_id: str) -> RunCreated:
        runtime = self._get_runtime(run_id)
        if runtime is not None:
            return self._runtime_response(runtime)
        response, _ = self._checkpoint_response(run_id)
        if response.state:
            return response
        artifact = self._artifact_response(run_id)
        if artifact is not None:
            return artifact
        raise KeyError(f"Unknown run_id: {run_id}")

    def wait(
        self,
        run_id: str,
        statuses: set[str] | None = None,
        timeout_seconds: float = 30,
    ) -> RunCreated:
        targets = statuses or WAITABLE_STATUSES
        deadline = time.monotonic() + timeout_seconds
        while True:
            response = self.get(run_id)
            ready_for_approval = not (
                response.status in HUMAN_CHECKPOINT_STATUSES
                and response.interrupt is None
            )
            runtime = self._get_runtime(run_id)
            terminal_ready = not (
                response.status in TERMINAL_STATUSES
                and runtime is not None
                and not runtime.closed
            )
            if response.status in targets and ready_for_approval and terminal_ready:
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Run {run_id} did not reach {sorted(targets)} in time"
                )
            if runtime is None:
                time.sleep(min(0.05, remaining))
                continue
            with runtime.condition:
                runtime.condition.wait(timeout=min(0.25, remaining))

    def iter_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        heartbeat_seconds: float = 10,
    ) -> Iterator[dict[str, Any] | None]:
        response = self.get(run_id)
        runtime = self._ensure_runtime(response)
        cursor = max(0, after_sequence)
        while True:
            with runtime.condition:
                pending = [
                    copy.deepcopy(event)
                    for event in runtime.events
                    if event["sequence"] > cursor
                ]
                if not pending and not runtime.closed:
                    runtime.condition.wait(timeout=heartbeat_seconds)
                    pending = [
                        copy.deepcopy(event)
                        for event in runtime.events
                        if event["sequence"] > cursor
                    ]
                closed = runtime.closed
            if pending:
                for event in pending:
                    cursor = event["sequence"]
                    yield event
                continue
            if closed:
                return
            yield None

    def _run_graph(
        self,
        runtime: RunRuntime,
        graph_input: dict[str, Any] | Command,
        is_resume: bool,
    ) -> None:
        try:
            with runtime.operation_lock:
                try:
                    if not is_resume:
                        self._publish(
                            runtime,
                            "run_status",
                            {
                                "status": "running",
                                "summary": "Background graph execution started.",
                            },
                        )
                        with runtime.condition:
                            runtime.status = "running"
                            runtime.state["status"] = "running"
                    with self._graph_session() as graph:
                        for chunk in graph.stream(
                            graph_input,
                            config=self._config(runtime.run_id),
                            stream_mode="updates",
                        ):
                            self._consume_chunk(runtime, chunk)
                        response = self._response_from_graph(
                            graph, runtime.run_id
                        )
                    self._replace_runtime_state(runtime, response)
                    self._publish(
                        runtime,
                        "run_status",
                        {
                            "status": response.status,
                            "summary": (
                                "Checkpoint reached; human decision required."
                                if response.status in HUMAN_CHECKPOINT_STATUSES
                                else "Checkpoint committed."
                            ),
                            "checkpointed": True,
                        },
                    )
                    self._persist_artifact(response)
                    if response.status in TERMINAL_STATUSES:
                        self._close_runtime(runtime)
                except Exception as exc:
                    self._handle_framework_failure(runtime, exc)
        finally:
            self._run_slots.release()

    def _submit_graph(
        self,
        runtime: RunRuntime,
        graph_input: dict[str, Any] | Command,
        is_resume: bool,
    ) -> Future:
        if not self._run_slots.acquire(blocking=False):
            raise RuntimeError(
                "Run queue is full; retry after active work reaches a checkpoint"
            )
        try:
            return self._executor.submit(
                self._run_graph,
                runtime,
                graph_input,
                is_resume,
            )
        except Exception:
            self._run_slots.release()
            raise

    def _consume_chunk(self, runtime: RunRuntime, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        for node_name, update in chunk.items():
            if node_name == "__interrupt__" or not isinstance(update, dict):
                continue
            traces = list(update.get("traces") or [])
            with runtime.condition:
                for key, value in update.items():
                    if key == "traces":
                        runtime.state.setdefault("traces", []).extend(
                            copy.deepcopy(value)
                        )
                    else:
                        runtime.state[key] = copy.deepcopy(value)
                if "status" in update:
                    runtime.status = str(update["status"])
                runtime.condition.notify_all()
            for trace in traces:
                self._publish(
                    runtime,
                    "agent_trace",
                    {"node": node_name, **copy.deepcopy(trace)},
                )
            if update.get("error"):
                self._publish(
                    runtime,
                    "run_error",
                    copy.deepcopy(update["error"]),
                )
            if "status" in update:
                summary = traces[-1].get("summary") if traces else None
                self._publish(
                    runtime,
                    "run_status",
                    {
                        "status": update["status"],
                        "node": node_name,
                        "summary": summary,
                    },
                )

    def _replace_runtime_state(
        self, runtime: RunRuntime, response: RunCreated
    ) -> None:
        with runtime.condition:
            runtime.status = response.status
            runtime.state = copy.deepcopy(response.state)
            runtime.interrupt = copy.deepcopy(response.interrupt)
            runtime.closed = response.status in TERMINAL_STATUSES
            runtime.condition.notify_all()

    def _handle_framework_failure(
        self, runtime: RunRuntime, exc: Exception
    ) -> None:
        message = self.suite.redact_secrets(
            str(exc).strip() or exc.__class__.__name__
        )
        rollback: dict[str, Any] | None = None
        state = copy.deepcopy(runtime.state)
        changed_files = state.get("apply_result", {}).get("changed_files", [])
        already_rolled_back = state.get("verification", {}).get("rollback")
        verification_succeeded = bool(state.get("verification", {}).get("success"))
        if changed_files and not already_rolled_back and not verification_succeeded:
            try:
                rollback = self.suite.applier.rollback(
                    state["mul_path"],
                    runtime.run_id,
                    changed_files,
                    expected_applied_sha256=state.get(
                        "apply_result", {}
                    ).get("applied_sha256"),
                )
            except Exception as rollback_error:
                rollback = {
                    "success": False,
                    "error": self.suite.redact_secrets(str(rollback_error))[:1000],
                }
        error = {
            "agent": "WorkflowRuntime",
            "type": exc.__class__.__name__,
            "message": message[:1000],
        }
        if rollback is not None:
            error["rollback"] = rollback
        trace = {
            "agent": "FailureReporterAgent",
            "status": "failed",
            "summary": f"Runtime failed safely: {error['message']}",
            "latency_ms": 0,
            "metadata": {
                "failed_agent": "WorkflowRuntime",
                "rollback": rollback,
            },
        }
        with runtime.condition:
            runtime.status = "failed"
            runtime.state["status"] = "failed"
            runtime.state["error"] = error
            runtime.state.setdefault("traces", []).append(trace)
            runtime.state["report"] = (
                f"# Mulagt Run {runtime.run_id} - Failed safely\n\n"
                f"- Error type: `{error['type']}`\n"
                f"- Message: {error['message']}\n"
            )
            runtime.interrupt = None
        self._publish(runtime, "agent_trace", trace)
        self._publish(runtime, "run_error", error)
        self._publish(
            runtime,
            "run_status",
            {
                "status": "failed",
                "summary": "Workflow runtime ended through the failure boundary.",
            },
        )
        self._persist_artifact(self._runtime_response(runtime))
        self._close_runtime(runtime)

    def _publish(
        self,
        runtime: RunRuntime,
        event_type: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        with runtime.condition:
            occurred_at = datetime.now(UTC).isoformat()
            next_memory_sequence = (
                max(
                    (int(item.get("sequence", 0)) for item in runtime.events),
                    default=0,
                )
                + 1
            )
            try:
                sequence = self._append_durable_event(
                    runtime.run_id,
                    event_type,
                    data,
                    occurred_at,
                    minimum_sequence=next_memory_sequence,
                )
            except sqlite3.Error:
                sequence = next_memory_sequence
            event = {
                "sequence": sequence,
                "type": event_type,
                "occurred_at": occurred_at,
                "data": copy.deepcopy(data),
            }
            runtime.events.append(event)
            runtime.condition.notify_all()
            return copy.deepcopy(event)

    def _close_runtime(self, runtime: RunRuntime) -> None:
        with runtime.condition:
            runtime.closed = True
            runtime.condition.notify_all()
        with self._runtimes_lock:
            if self._runtimes.get(runtime.run_id) is runtime:
                self._runtimes.pop(runtime.run_id, None)

    def _runtime_response(self, runtime: RunRuntime) -> RunCreated:
        with runtime.condition:
            return RunCreated(
                run_id=runtime.run_id,
                status=runtime.status,
                state=copy.deepcopy(runtime.state),
                interrupt=copy.deepcopy(runtime.interrupt),
            )

    def _ensure_runtime(self, response: RunCreated) -> RunRuntime:
        existing = self._get_runtime(response.run_id)
        if existing is not None:
            return existing
        runtime = RunRuntime(
            run_id=response.run_id,
            status=response.status,
            state=copy.deepcopy(response.state),
            interrupt=copy.deepcopy(response.interrupt),
            closed=response.status in TERMINAL_STATUSES,
        )
        runtime.events = self._load_durable_events(response.run_id)
        with self._runtimes_lock:
            return self._runtimes.setdefault(response.run_id, runtime)

    def _get_runtime(self, run_id: str) -> RunRuntime | None:
        with self._runtimes_lock:
            return self._runtimes.get(run_id)

    def _checkpoint_response(
        self, run_id: str
    ) -> tuple[RunCreated, tuple[str, ...]]:
        with self._graph_session() as graph:
            snapshot = graph.get_state(self._config(run_id))
            response = self._response_from_snapshot(run_id, snapshot)
            return response, tuple(snapshot.next or ())

    def _response_from_graph(self, graph, run_id: str) -> RunCreated:
        snapshot = graph.get_state(self._config(run_id))
        return self._response_from_snapshot(run_id, snapshot)

    @staticmethod
    def _response_from_snapshot(run_id: str, snapshot) -> RunCreated:
        values = dict(snapshot.values or {})
        interrupt_payload = None
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        if interrupts:
            interrupt_payload = interrupts[0].value
        return RunCreated(
            run_id=run_id,
            status=values.get("status", "unknown"),
            state=values,
            interrupt=interrupt_payload,
        )

    def _artifact_response(self, run_id: str) -> RunCreated | None:
        state_file = self.settings.runtime_dir / "runs" / run_id / "state.json"
        if not state_file.is_file():
            return None
        try:
            return RunCreated.model_validate_json(
                state_file.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

    def _persist_artifact(self, response: RunCreated) -> None:
        output_dir = self.settings.runtime_dir / "runs" / response.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        state_file = output_dir / "state.json"
        temporary = output_dir / "state.json.tmp"
        temporary.write_text(
            json.dumps(response.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(state_file)
        report = response.state.get("report")
        if report:
            report_file = output_dir / "report.md"
            report_temporary = output_dir / "report.md.tmp"
            report_temporary.write_text(report, encoding="utf-8")
            report_temporary.replace(report_file)

    def _setup_database(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            SqliteSaver(connection).setup()
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_run_events_run_sequence
                ON run_events (run_id, sequence)
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _append_durable_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
        occurred_at: str,
        *,
        minimum_sequence: int = 1,
    ) -> int:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1
                FROM run_events
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            sequence = max(int(row[0]), minimum_sequence)
            connection.execute(
                """
                INSERT INTO run_events
                    (run_id, sequence, event_type, occurred_at, data_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    sequence,
                    event_type,
                    occurred_at,
                    json.dumps(data, ensure_ascii=False),
                ),
            )
            connection.commit()
            return sequence
        finally:
            connection.close()

    def _load_durable_events(self, run_id: str) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT sequence, event_type, occurred_at, data_json
                FROM run_events
                WHERE run_id = ?
                ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()
        finally:
            connection.close()
        result = []
        for sequence, event_type, occurred_at, data_json in rows:
            try:
                data = json.loads(data_json)
            except (TypeError, ValueError):
                data = {"error": "event payload could not be decoded"}
            result.append(
                {
                    "sequence": int(sequence),
                    "type": str(event_type),
                    "occurred_at": str(occurred_at),
                    "data": data,
                }
            )
        return result

    @contextmanager
    def _graph_session(self):
        connection = self._connect()
        try:
            yield build_workflow(self.suite, SqliteSaver(connection))
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database,
            timeout=30,
            check_same_thread=False,
        )
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MulagtService is closed")

    @staticmethod
    def _config(run_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": run_id}}
