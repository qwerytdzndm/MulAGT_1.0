from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from threading import local
from typing import Any

from openai import OpenAI

from .config import Settings


class LLMConfigurationError(RuntimeError):
    pass


class StructuredLLM(ABC):
    @abstractmethod
    def complete_json(
        self, *, system_prompt: str, payload: dict[str, Any], schema_name: str
    ) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def route_metadata(self) -> dict[str, Any]:
        return {}

    @property
    def last_usage(self) -> dict[str, int]:
        return {}

    def close(self) -> None:
        """Release provider transports when the implementation owns one."""


class DeepSeekLLM(StructuredLLM):
    def __init__(self, settings: Settings) -> None:
        if not settings.deepseek_api_key:
            raise LLMConfigurationError(
                "DEEPSEEK_API_KEY is required when llm_mode=deepseek"
            )
        self.model = settings.deepseek_model
        self.max_output_tokens = settings.llm_max_output_tokens
        self.client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=90.0,
            max_retries=0,
        )
        self._request_local = local()

    @property
    def last_usage(self) -> dict[str, int]:
        return dict(getattr(self._request_local, "usage", {}))

    @property
    def route_metadata(self) -> dict[str, Any]:
        return {
            "provider": "deepseek",
            "model": self.model,
            "credential_source": "environment",
        }

    def close(self) -> None:
        self.client.close()

    def complete_json(
        self, *, system_prompt: str, payload: dict[str, Any], schema_name: str
    ) -> dict[str, Any]:
        self._request_local.usage = {}
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Produce JSON for schema '{schema_name}'. Input:\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=self.max_output_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("DeepSeek returned empty JSON content")
                if response.usage:
                    raw_usage = response.usage.model_dump()
                    self._request_local.usage = {
                        key: int(value)
                        for key, value in raw_usage.items()
                        if key
                        in {
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                            "prompt_cache_hit_tokens",
                            "prompt_cache_miss_tokens",
                        }
                        and value is not None
                    }
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("DeepSeek JSON response must be an object")
                return parsed
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                retryable = status_code is None or status_code == 429 or (
                    isinstance(status_code, int) and status_code >= 500
                )
                if attempt < 2 and retryable:
                    time.sleep(
                        1.5 * (attempt + 1) + random.uniform(0.0, 0.35)
                    )
                    continue
                break
        raise RuntimeError(
            "DeepSeek structured request failed after retries: "
            f"{type(last_error).__name__}"
        ) from last_error


class MockLLM(StructuredLLM):
    """Deterministic test double for the retained coding workflow."""

    def complete_json(
        self, *, system_prompt: str, payload: dict[str, Any], schema_name: str
    ) -> dict[str, Any]:
        if schema_name == "change_plan":
            mul_map = payload.get("mul_map", [])
            target = next(
                (
                    item["path"]
                    for item in mul_map
                    if "calculator" in item["path"].lower()
                ),
                mul_map[0]["path"] if mul_map else "src/calculator.py",
            )
            evidence_ids = [
                item["evidence_id"] for item in payload.get("evidence", [])[:2]
            ]
            return {
                "summary": "Add an explicit zero-divisor guard and verify behavior.",
                "assumptions": [
                    "The public function signature must remain unchanged."
                ],
                "steps": [
                    {
                        "step_id": "P1",
                        "description": "Add a ValueError guard before division.",
                        "target_files": [target],
                        "evidence_ids": evidence_ids,
                        "risk": "low",
                        "verification": "Run the repository unit test suite.",
                    }
                ],
            }
        if schema_name == "code_edits":
            for item in payload.get("files", []):
                content = "\n".join(item.get("excerpts", []))
                if "return a / b" in content:
                    return {
                        "edits": [
                            {
                                "path": item["path"],
                                "reasoning": (
                                    "Convert the implicit ZeroDivisionError into "
                                    "the explicit ValueError required by the issue."
                                ),
                                "search": "    return a / b",
                                "replace": (
                                    '    if b == 0:\n'
                                    '        raise ValueError("divisor must not be zero")\n'
                                    "    return a / b"
                                ),
                            }
                        ]
                    }
            raise ValueError("MockLLM demo pattern was not found in target files")
        if schema_name == "review_decision":
            diffs = str(payload.get("diffs", ""))
            dangerous = any(
                marker in diffs
                for marker in ("eval(", "exec(", "shell=True", "sk-", "AKIA")
            )
            return {
                "approved": not dangerous,
                "summary": (
                    "The patch is minimal and matches the requested behavior."
                    if not dangerous
                    else "The patch contains a dangerous pattern."
                ),
                "findings": (
                    []
                    if not dangerous
                    else [
                        {
                            "severity": "critical",
                            "message": "Dangerous code or credential pattern detected.",
                            "path": None,
                        }
                    ]
                ),
            }
        if schema_name == "reflection":
            return {
                "root_cause": "The applied change did not satisfy the allowlisted tests.",
                "next_action": (
                    "Inspect the failing assertion, retrieve additional context, "
                    "and create a new reviewed proposal."
                ),
                "retry_recommended": True,
            }
        raise ValueError(f"Unsupported mock schema: {schema_name}")


def build_llm(mode: str, settings: Settings) -> StructuredLLM:
    if mode == "deepseek":
        return DeepSeekLLM(settings)
    if mode == "mock":
        return MockLLM()
    raise ValueError(f"Unsupported LLM mode: {mode}")
