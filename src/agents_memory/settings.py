from os import environ
from typing import ClassVar, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _required_operator_token(name: str) -> str:
    return environ.get(name, "")


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )

    agents_memory_token: str = Field(default="", min_length=1)
    agents_memory_read_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "AGENTS_MEMORY_READ_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    agents_memory_export_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "AGENTS_MEMORY_EXPORT_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    agents_memory_write_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "AGENTS_MEMORY_WRITE_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    agents_memory_admin_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "AGENTS_MEMORY_ADMIN_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    agents_memory_tenant_id: str = Field(default="bromigos", min_length=1)
    neo4j_uri: str = Field(default="", min_length=1)
    neo4j_username: str = Field(default="neo4j", min_length=1)
    neo4j_password: str = Field(default="", min_length=1)
    litellm_base_url: str = Field(default="", min_length=1)
    litellm_api_key: str = Field(default="", min_length=1)
    memory_llm: str = Field(default="openai/gemma4", min_length=1)
    memory_embedding: str = Field(default="local-qwen3-embedding-0.6b", min_length=1)
    memory_embedding_dimensions: int = Field(default=1024, gt=0)
    memory_audit_read: bool = False
    memory_conversation_ttl_days: int | None = Field(default=None, ge=1)
    memory_write_mode: Literal["sync", "buffered"] = "sync"
    memory_max_pending: int = Field(default=200, ge=1)
    memory_fact_deduplication_enabled: bool = True
    memory_trace_embedding_enabled: bool = True
    memory_extract_entities_enabled: bool = False
    memory_extract_relations_enabled: bool = False
    memory_extraction_preview_enabled: bool = False
    memory_extraction_batch_size: int = Field(default=25, ge=1)
    memory_extraction_max_concurrency: int = Field(default=1, ge=1)
    memory_extraction_chunk_size: int = Field(default=4000, ge=1)
    memory_extraction_chunk_overlap: int = Field(default=200, ge=0)
    memory_ocr_enabled: bool = False
    memory_ocr_model: str = ""
    memory_ocr_max_image_bytes: int = Field(default=0, ge=0)
    memory_rustfs_enabled: bool = False
    memory_rustfs_bucket: str = ""
    memory_rustfs_prefix: str = ""
    memory_rustfs_endpoint: str = ""
    memory_rustfs_retention_days: int | None = Field(default=None, ge=1)
    memory_prompt_entities_enabled: bool = False
    memory_prompt_preferences_enabled: bool = False
    memory_prompt_reasoning_enabled: bool = False
    memory_consolidation_schedule_enabled: bool = False


def load_settings() -> Settings:
    return Settings()
