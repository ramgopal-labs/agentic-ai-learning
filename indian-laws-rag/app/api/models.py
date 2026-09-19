from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        description="The legal question asked by the user.",
    )
    session_id: str = Field(
        default="default-session",
        description="Groups multiple questions into one conversation.",
    )
    act_filter: str | None = Field(
        default=None,
        description="Optional exact Act title used to narrow retrieval.",
    )
    provider: Literal["openai", "groq", "gemini"] = "openai"


class Citation(BaseModel):
    act_title: str | None = None
    section: str | None = None
    source_url: str | None = None


class AskResponse(BaseModel):
    answer: str
    confidence: float
    source_path: Literal["vector_store", "web_search", "not_ready"]
    citations: list[Citation] = []
    rewritten_question: str
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Sub-questions the query was decomposed into before retrieval.",
    )
