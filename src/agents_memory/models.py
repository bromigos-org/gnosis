from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class MemoryVisibility(StrEnum):
    PRIVATE_USER = "private_user"
    AGENT_PRIVATE = "agent_private"
    AGENT_SHARED = "agent_shared"
    CHANNEL = "channel"
    GUILD = "guild"
    TENANT = "tenant"
    GLOBAL = "global"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MemoryScope(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    space_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    visibility: MemoryVisibility
    guild_id: str | None = Field(default=None, min_length=1)
    channel_id: str | None = Field(default=None, min_length=1)


class MessageWriteRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    scope: MemoryScope
    role: MessageRole
    content: str = Field(min_length=1)


class MessageWriteResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    accepted: bool


class ContextRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    scope: MemoryScope
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=30)


class ContextResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    context: str


class HealthResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    status: str
