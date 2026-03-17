from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PromptRecord(BaseModel):
    id: str
    prompt: str
    answer: str


class TraceRecord(BaseModel):
    id: str
    prompt: str
    answer: str
    trace_id: int
    reasoning: str
    steps: list[str]
    final_answer: str


class ScoredTraceRecord(BaseModel):
    id: str
    prompt: str
    answer: str
    trace_id: int
    steps: list[str]
    step_scores: list[float]
    reasoning: str
    final_answer: str
    final_correct: bool


class PairRecord(BaseModel):
    id: str
    prompt: str
    prefix_steps: list[str]
    preferred_steps: list[str]
    dispreferred_steps: list[str]
    weight: float
    features: dict[str, float]
    meta: dict[str, Any] = Field(default_factory=dict)


class ProcessGroundTruthRecord(BaseModel):
    id: str
    gold_earliest_error_step: int
