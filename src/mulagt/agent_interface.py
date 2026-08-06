from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agents import AgentSuite
from .config import Settings
from .models import IssueRequest
from .state import STATE_SCHEMA_VERSION, MulagtState


SCHEMA_AGENT_NAMES = {
    "change_plan": "PlannerAgent",
    "code_edits": "CoderAgent",
    "review_decision": "ReviewerAgent",
    "reflection": "ReflectionAgent",
}


class TranscriptAgentSuite(AgentSuite):
    """AgentSuite variant that records model-facing messages and responses."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.transcript: list[dict[str, Any]] = []

    def _record(
        self,
        *,
        agent: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.transcript.append(
            {
                "agent": agent,
                "role": role,
                "content": content,
                "metadata": metadata or {},
            }
        )

    def _complete_json(
        self,
        state: MulagtState,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        agent = SCHEMA_AGENT_NAMES.get(schema_name, schema_name)
        compressed, context_stats = self.context_compressor.compress(
            schema_name,
            payload,
        )
        self._record(
            agent=agent,
            role="system",
            content=system_prompt,
            metadata={"schema": schema_name},
        )
        self._record(
            agent=agent,
            role="user",
            content=json.dumps(compressed, ensure_ascii=False, indent=2),
            metadata={"schema": schema_name, "compressed": True},
        )

        llm = self._llm(state, schema_name)
        result = llm.complete_json(
            system_prompt=system_prompt,
            payload=compressed,
            schema_name=schema_name,
        )
        self._record(
            agent=agent,
            role="assistant",
            content=json.dumps(result, ensure_ascii=False, indent=2),
            metadata={"schema": schema_name, "raw": True},
        )

        repair_attempts = 0
        schema_model = self._SCHEMA_MODELS.get(schema_name)
        if schema_model is not None:
            try:
                result = schema_model.model_validate(result).model_dump()
            except Exception as exc:
                repair_attempts = 1
                validation_error = str(exc)[:4000]
                repair_payload = {
                    "original_task_input": compressed,
                    "invalid_response": result,
                    "validation_error": validation_error,
                    "repair_rule": (
                        "Preserve valid facts. Fix only schema/type/required-field "
                        "errors. Return one JSON object and no markdown."
                    ),
                }
                repaired, repair_context = self.context_compressor.compress(
                    "schema_repair",
                    repair_payload,
                )
                repair_prompt = (
                    system_prompt
                    + "\n\nYour previous response failed deterministic schema "
                    "validation. Repair it once using the supplied validation "
                    "error. Do not invent facts to fill optional fields."
                )
                self._record(
                    agent=agent,
                    role="system",
                    content=repair_prompt,
                    metadata={"schema": schema_name, "repair": True},
                )
                self._record(
                    agent=agent,
                    role="user",
                    content=json.dumps(repaired, ensure_ascii=False, indent=2),
                    metadata={"schema": schema_name, "repair": True},
                )
                result = llm.complete_json(
                    system_prompt=repair_prompt,
                    payload=repaired,
                    schema_name=schema_name,
                )
                self._record(
                    agent=agent,
                    role="assistant",
                    content=json.dumps(result, ensure_ascii=False, indent=2),
                    metadata={"schema": schema_name, "repair": True},
                )
                result = schema_model.model_validate(result).model_dump()
                context_stats["repair_context"] = repair_context

        api_usage = getattr(llm, "last_usage", None)
        if api_usage:
            context_stats["api_usage"] = dict(api_usage)
        route_metadata = getattr(llm, "route_metadata", None)
        if route_metadata:
            context_stats["model_route"] = dict(route_metadata)
        context_stats["repair_attempts"] = repair_attempts
        self._record(
            agent=agent,
            role="validator",
            content=json.dumps(result, ensure_ascii=False, indent=2),
            metadata={"schema": schema_name, "context": context_stats},
        )
        return result, context_stats


def _merge_state(state: MulagtState, update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key == "traces":
            state.setdefault("traces", []).extend(value)
        else:
            state[key] = value


def run_three_agent_code_generation(
    request: IssueRequest,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run RAG plus Planner/Coder/Reviewer and return the full transcript.

    This interface stops before human approval, patch application, and tests.
    It is intended for inspecting the three-agent code-generation conversation.
    """

    suite = TranscriptAgentSuite(settings or Settings.from_env())
    state: MulagtState = {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "mul_path": str(Path(request.mul_path).resolve()),
        "issue": request.issue,
        "constraints": request.constraints,
        "test_command": request.test_command,
        "llm_mode": request.llm_mode or suite.settings.llm_mode,
        "status": "queued",
        "approved": False,
        "traces": [],
    }
    try:
        for agent_name, node in (
            ("IntakeAgent", suite.intake_agent),
            ("MulMapperAgent", suite.mul_mapper_agent),
            ("RetrievalAgent", suite.retrieval_agent),
            ("PlannerAgent", suite.planner_agent),
            ("CoderAgent", suite.coder_agent),
            ("ReviewerAgent", suite.reviewer_agent),
        ):
            suite._record(
                agent=agent_name,
                role="runtime",
                content=f"Calling {agent_name}.",
                metadata={"status_before": state.get("status")},
            )
            update = node(state)
            _merge_state(state, update)
            suite._record(
                agent=agent_name,
                role="runtime",
                content=f"{agent_name} returned status {state.get('status')}.",
                metadata={
                    "status_after": state.get("status"),
                    "trace": (update.get("traces") or [{}])[-1],
                },
            )

        return {
            "run_id": state["run_id"],
            "status": state.get("status"),
            "plan": state.get("plan"),
            "proposals": state.get("proposals", []),
            "review": state.get("review"),
            "transcript": suite.transcript,
            "state": dict(state),
        }
    finally:
        suite.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Mulagt's Planner, Coder, and Reviewer Agents."
    )
    parser.add_argument("--mul-path", required=True)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument(
        "--test-command",
        default="python -m unittest discover -s tests",
    )
    parser.add_argument("--llm-mode", choices=["mock", "deepseek"], default=None)
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument(
        "--transcript",
        default="mulagt_three_agent_transcript.json",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.runtime_dir:
        settings = replace(settings, runtime_dir=Path(args.runtime_dir).resolve())
    result = run_three_agent_code_generation(
        IssueRequest(
            mul_path=args.mul_path,
            issue=args.issue,
            constraints=args.constraint,
            test_command=args.test_command,
            llm_mode=args.llm_mode,
        ),
        settings=settings,
    )
    output = Path(args.transcript)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"status: {result['status']}")
    print(f"proposals: {len(result['proposals'])}")
    print(f"review approved: {result.get('review', {}).get('approved')}")
    print(f"transcript: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
