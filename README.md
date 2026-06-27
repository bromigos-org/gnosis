# agents-memory

`agents-memory` is the shared homelab memory service for PC Principal and future agents. It exposes policy-scoped HTTP endpoints instead of giving agents direct Neo4j access.

## API

- `GET /health` returns service status.
- `POST /v1/messages` records a scoped message for extraction into memory.
- `POST /v1/context` retrieves scoped memory context for a query.
- `POST /v1/events` and `POST /v1/events/batch` ingest scoped structured events such as Discord messages, topology updates, reactions, links, and attachment metadata.
- `POST /v1/graph/context` returns scoped graph facts for a query.
- `POST /v1/skills` lists reviewed skill context for one tenant and agent.
- `POST /v1/skills/proposals` stores a proposed skill for human review.
- `POST /v1/skills/usage` records usage only for approved reviewed skills.

All non-health endpoints require `Authorization: Bearer <AGENTS_MEMORY_TOKEN>`.

## Discord memory scope and privacy

- `agents-memory` is policy-scoped first. Clients must send tenant, agent, session, user, and visibility metadata on every message, event, or query.
- PC Principal uses tenant `bromigos` and agent `pc-principal` by default, so recall stays inside that tenant and agent boundary.
- Visibility is enforced in the backend. `private_user` stays tied to the matching user, `channel` stays tied to the matching guild and channel, `guild` stays inside that guild, `agent_shared` stays within the same agent, and `global` is the only broad scope.
- Channel-scoped graph recall does not cross into sibling channels. This prevents cross-channel graph recall even when two channels live in the same guild.
- Topology deletes and renames are preserved as tombstones or event history, not hard-deleted facts. That keeps the audit trail while still reflecting current state.

## Reviewed skill workflow

- Skills are not self-modifying executable behavior. The intended workflow is observe, propose, ask for approval, save an approved reviewed skill, then expose that approved record as non-executable context.
- `list_skills` only returns approved skills whose metadata marks them as reviewed.
- Proposals are stored for review, but they are not returned as runnable context.
- Usage recording is rejected for unapproved skills, with the backend returning `skill is not approved`.

## Attachments and Discord event posture

- The current Discord rollout is metadata-first. PC Principal sends attachment filename, content type, size, dimensions, spoiler status, and sanitized URLs, plus sanitized link discoveries.
- Attachment bytes are not copied by default. If an operator later enables a copy policy in PC Principal, that is a separate rollout decision from the default memory service behavior.
- RustFS is only relevant for a future intentional copy path. The current shared-memory contract assumes metadata-only attachment ingestion.

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
- `MEMORY_EMBEDDING`: embedding model alias. Homelab deployments use the local-only LiteLLM alias `local-bge-m3`, backed by `BAAI/bge-m3` with 1024 dimensions through an in-cluster OpenAI-compatible `/v1` endpoint.
- `MEMORY_EMBEDDING_DIMENSIONS`: embedding vector dimensions. This must match the selected embedding model; `local-bge-m3` uses `1024`.

Neo4j agent memory embeddings must stay behind LiteLLM. Do not point PC Principal or `agents-memory` directly at an embedding runtime, and do not use OpenAI/Copilot aliases for memory embeddings. Before rolling out `local-bge-m3`, ensure the local embedding runtime serves `BAAI/bge-m3` at the in-cluster endpoint configured in the homelab LiteLLM wrapper chart.

The first API layer is intentionally small: agents pass scope metadata with every request, and future backend adapters enforce that scope before touching graph/vector memory.

For operators, keep the deployment path GitOps-only. Update the calling service's Helm values or manifests in Git, push to the tracked branch, and let ArgoCD reconcile. Don't bypass tracked services with manual cluster apply steps or manual chart install or upgrade steps.
