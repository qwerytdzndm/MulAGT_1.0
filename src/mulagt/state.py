from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


STATE_SCHEMA_VERSION = 1


class MulagtState(TypedDict, total=False):
    state_schema_version: int
    run_id: str
    mul_path: str
    issue: str
    constraints: list[str]
    test_command: str
    llm_mode: str
    status: str
    mul_map: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    plan: dict[str, Any]
    proposals: list[dict[str, Any]]
    review: dict[str, Any]
    approved: bool
    approval_comment: str
    apply_result: dict[str, Any]
    test_result: dict[str, Any]
    verification: dict[str, Any]
    reflection: dict[str, Any]
    report: str
    error: dict[str, Any]
    traces: Annotated[list[dict[str, Any]], operator.add]
