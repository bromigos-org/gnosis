# agents-memory

`agents-memory` is the shared homelab memory service for PC Principal and future agents. It exposes policy-scoped HTTP endpoints instead of giving agents direct Neo4j access.

## API

- `GET /health` returns service status.
- `POST /v1/messages` records a scoped message for extraction into memory.
- `POST /v1/context` retrieves scoped memory context for a query.

All non-health endpoints require `Authorization: Bearer <AGENTS_MEMORY_TOKEN>`.

## Local development

```bash
uv sync
uv run uvicorn agents_memory.main:app --host 0.0.0.0 --port 8080 --reload
uv run pytest
```

## Configuration

- `AGENTS_MEMORY_TOKEN`: bearer token required by API clients.
- `AGENTS_MEMORY_TENANT_ID`: default tenant, usually `bromigos`.
- `NEO4J_URI`: in-cluster Bolt URI, for example `bolt://neo4j.neo4j.svc.cluster.local:7687`.
- `NEO4J_USERNAME`: Neo4j username, usually `neo4j`.
- `NEO4J_PASSWORD`: Neo4j password from Vault/ESO.
- `LITELLM_BASE_URL`: OpenAI-compatible LiteLLM endpoint.
- `LITELLM_API_KEY`: LiteLLM master key or service token.
- `MEMORY_LLM`: extraction/reasoning model alias.
- `MEMORY_EMBEDDING`: embedding model alias.

The first API layer is intentionally small: agents pass scope metadata with every request, and future backend adapters enforce that scope before touching graph/vector memory.
