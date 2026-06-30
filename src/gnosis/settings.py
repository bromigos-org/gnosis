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

    gnosis_token: str = Field(default="", min_length=1)
    gnosis_read_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "GNOSIS_READ_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    gnosis_export_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "GNOSIS_EXPORT_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    gnosis_write_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "GNOSIS_WRITE_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    gnosis_admin_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "GNOSIS_ADMIN_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    gnosis_tenant_id: str = Field(default="bromigos", min_length=1)
    neo4j_uri: str = Field(default="", min_length=1)
    neo4j_username: str = Field(default="neo4j", min_length=1)
    neo4j_password: str = Field(default="", min_length=1)
    litellm_base_url: str = Field(default="", min_length=1)
    litellm_api_key: str = Field(default="", min_length=1)
    gnosis_llm: str = Field(default="openai/gemma4", min_length=1)
    gnosis_embedding: str = Field(default="local-qwen3-embedding-0.6b", min_length=1)
    gnosis_embedding_dimensions: int = Field(default=1024, gt=0)
    gnosis_audit_read: bool = False
    gnosis_conversation_ttl_days: int | None = Field(default=None, ge=1)
    gnosis_write_mode: Literal["sync", "buffered"] = "sync"
    gnosis_max_pending: int = Field(default=200, ge=1)
    gnosis_fact_deduplication_enabled: bool = True
    gnosis_trace_embedding_enabled: bool = True
    gnosis_extract_entities_enabled: bool = False
    gnosis_extract_relations_enabled: bool = False
    gnosis_extraction_preview_enabled: bool = False
    gnosis_extraction_batch_size: int = Field(default=25, ge=1)
    gnosis_extraction_max_concurrency: int = Field(default=1, ge=1)
    gnosis_extraction_chunk_size: int = Field(default=4000, ge=1)
    gnosis_extraction_chunk_overlap: int = Field(default=200, ge=0)
    gnosis_ocr_enabled: bool = False
    gnosis_ocr_model: str = ""
    gnosis_ocr_max_image_bytes: int = Field(default=0, ge=0)
    gnosis_rustfs_enabled: bool = False
    gnosis_rustfs_bucket: str = ""
    gnosis_rustfs_prefix: str = ""
    gnosis_rustfs_endpoint: str = ""
    gnosis_rustfs_retention_days: int | None = Field(default=None, ge=1)
    gnosis_prompt_entities_enabled: bool = False
    gnosis_prompt_preferences_enabled: bool = False
    gnosis_prompt_reasoning_enabled: bool = False
    gnosis_consolidation_schedule_enabled: bool = False


def load_settings() -> Settings:
    return Settings()
