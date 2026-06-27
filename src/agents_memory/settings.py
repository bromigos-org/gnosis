from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )

    agents_memory_token: str = Field(default="", min_length=1)
    agents_memory_tenant_id: str = Field(default="bromigos", min_length=1)
    neo4j_uri: str = Field(default="", min_length=1)
    neo4j_username: str = Field(default="neo4j", min_length=1)
    neo4j_password: str = Field(default="", min_length=1)
    litellm_base_url: str = Field(default="", min_length=1)
    litellm_api_key: str = Field(default="", min_length=1)
    memory_llm: str = Field(default="openai/gemma4", min_length=1)
    memory_embedding: str = Field(
        default="openai/copilot-text-embedding-3-small",
        min_length=1,
    )
    memory_embedding_dimensions: int = Field(default=1536, gt=0)


def load_settings() -> Settings:
    return Settings()
