"""Request/response schemas for /chat. The assignment says the schema is non-negotiable;
keep this file the single source of truth."""

from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    # The assignment example uses single-letter test_type codes ("K", "P", ...).
    # Items with multiple keys are emitted comma-joined, mirroring the conversation
    # traces ("K,S", "P,C") so the evaluator can split if it wants to.
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[List[Recommendation]] = None
    end_of_conversation: bool = False


class Health(BaseModel):
    status: Literal["ok"] = "ok"
