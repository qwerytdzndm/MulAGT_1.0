from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LLMMode = Literal["mock", "deepseek"]


class IssueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mul_path: str
    issue: str = Field(min_length=5, max_length=10_000)
    constraints: list[str] = Field(default_factory=list, max_length=50)
    test_command: str = "python -m unittest discover -s tests"
    llm_mode: LLMMode | None = None

    @field_validator("mul_path", "test_command")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    comment: str = Field(default="", max_length=5_000)
    actor_id: str = Field(default="human", min_length=1, max_length=120)
    artifact_digest: str = Field(min_length=64, max_length=64)


class RunCreated(BaseModel):
    run_id: str
    status: str
    state: dict = Field(default_factory=dict)
    interrupt: dict | None = None


class Evidence(BaseModel):
    evidence_id: str
    chunk_id: str
    document_id: str
    path: str
    text: str
    source_type: str = "document"
    trust_level: str = "repository_source"
    document_version: str | None = None
    language: str = "text"
    start_line: int | None = None
    end_line: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    symbol: str | None = None
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    score: float = 0.0


class ChangeStep(BaseModel):
    step_id: str
    description: str
    target_files: list[str] = Field(min_length=1, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    risk: Literal["low", "medium", "high"] = "medium"
    verification: str


class ChangePlan(BaseModel):
    summary: str
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    steps: list[ChangeStep] = Field(min_length=1, max_length=20)


class CodeEdit(BaseModel):
    path: str
    reasoning: str
    search: str = Field(min_length=1)
    replace: str


class CodeEditSet(BaseModel):
    edits: list[CodeEdit] = Field(min_length=1, max_length=50)


class ChangeProposal(BaseModel):
    path: str
    reasoning: str
    original_sha256: str
    new_content: str
    diff: str


class ReviewFinding(BaseModel):
    severity: Literal["info", "low", "medium", "high", "critical"] = "medium"
    message: str
    path: str | None = None


class ReviewDecision(BaseModel):
    approved: bool
    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list, max_length=100)


class Reflection(BaseModel):
    root_cause: str
    next_action: str
    retry_recommended: bool = True
