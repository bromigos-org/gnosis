from os import environ
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type PeerDirection = Literal["both", "push", "pull"]

_PEER_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"


def _required_operator_token(name: str) -> str:
    return environ.get(name, "")


class PeerConfig(BaseModel):
    """One remote gnosis deployment this instance may federate with."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=_PEER_NAME_PATTERN)
    base_url: str = Field(min_length=1)
    direction: PeerDirection = "both"
    remote_tenant_id: str = Field(min_length=1)

    @property
    def token_env_var(self) -> str:
        return f"GNOSIS_PEER_{self.name.upper().replace('-', '_')}_TOKEN"

    def allows_push(self) -> bool:
        return self.direction in {"both", "push"}

    def allows_pull(self) -> bool:
        return self.direction in {"both", "pull"}


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
    gnosis_recall_filter_enabled: bool = False
    gnosis_recall_filter_candidates: int = Field(default=30, ge=1)
    gnosis_hybrid_retrieval_enabled: bool = False
    gnosis_graphqa_fusion_enabled: bool = False
    # The graph-QA planner is an LLM call that commonly takes ~10s on a
    # frontier model; a 5s budget timed out on every fusion request. 20s
    # leaves headroom while still bounding a stalled planner.
    gnosis_graphqa_fusion_timeout_seconds: float = Field(default=20.0, gt=0)
    gnosis_read_supersession_enabled: bool = False
    gnosis_entity_graph_enabled: bool = False
    gnosis_graph_traversal_enabled: bool = False
    gnosis_adaptive_routing_enabled: bool = False
    gnosis_routing_model: str = ""
    gnosis_sufficiency_check_enabled: bool = False
    gnosis_sufficiency_model: str = ""
    gnosis_abstention_prompt_enabled: bool = False
    gnosis_chain_of_note_enabled: bool = False
    gnosis_fact_verbatim_expansion_enabled: bool = False
    gnosis_fact_verbatim_expansion_max: int = Field(default=5, ge=1)
    gnosis_fact_extraction_enabled: bool = False
    gnosis_fact_extraction_model: str = ""
    gnosis_fact_extraction_context_turns: int = Field(default=10, ge=0)
    gnosis_fact_extraction_mode: Literal["sync", "background"] = "sync"
    gnosis_fact_extraction_max_concurrency: int = Field(default=2, ge=1)
    gnosis_fact_extraction_max_pending: int = Field(default=200, ge=1)
    gnosis_prompt_entities_enabled: bool = False
    gnosis_prompt_preferences_enabled: bool = False
    gnosis_prompt_reasoning_enabled: bool = False
    gnosis_consolidation_schedule_enabled: bool = False
    gnosis_memory_edit_enabled: bool = False
    gnosis_mcp_enabled: bool = False
    gnosis_mcp_agent_id: str = Field(default="mcp-client", min_length=1)
    gnosis_federation_token: str = ""
    gnosis_peers: list[PeerConfig] = Field(default_factory=list)

    @field_validator("gnosis_peers")
    @classmethod
    def _require_unique_peer_names(
        cls,
        peers: list[PeerConfig],
    ) -> list[PeerConfig]:
        seen: set[str] = set()
        for peer in peers:
            normalized = peer.name.casefold()
            if normalized in seen:
                detail = f"duplicate peer name: {peer.name}"
                raise ValueError(detail)
            seen.add(normalized)
        return peers


def load_settings() -> Settings:
    return Settings()
