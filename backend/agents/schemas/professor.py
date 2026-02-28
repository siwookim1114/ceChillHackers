"""Strict Professor agent schemas (Pydantic v2)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProfessorMode(str, Enum):
    STRICT = "strict"
    CONVENIENCE = "convenience"


class ProfessorTurnStrategy(str, Enum):
    SOCRATIC_QUESTION = "socratic_question"
    HINT = "hint"
    CONCEPT_EXPLAIN = "concept_explain"
    ENCOURAGEMENT = "encouragement"


class ProfessorNextAction(str, Enum):
    CONTINUE = "continue"
    ROUTE_PROBLEM_TA = "route_problem_ta"
    ROUTE_PLANNER = "route_planner"


class ProfessorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["beginner", "intermediate", "advanced"]
    learning_style: Literal["visual", "textual", "example_first", "mixed"]
    pace: Literal["slow", "medium", "fast"] = "medium"


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    url: str | None = None


class ProfessorTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    mode: ProfessorMode = ProfessorMode.STRICT
    profile: ProfessorProfile

    @property
    def student_message(self) -> str:
        return self.message


class ProfessorTurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_response: str = Field(min_length=1)
    strategy: ProfessorTurnStrategy
    revealed_final_answer: Literal[False] = False
    next_action: ProfessorNextAction = ProfessorNextAction.CONTINUE
    citations: list[Citation] = Field(default_factory=list)
