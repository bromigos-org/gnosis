# agents-memory

`agents-memory` is the shared homelab memory service for PC Principal and future agents. It exposes policy-scoped HTTP endpoints instead of giving agents direct Neo4j access.

## API

- `GET /health` returns shallow liveness as `{"status":"ok"}` without checking dependencies.
- `GET /ready` is unauthenticated Kubernetes readiness. It returns `{"status":"ready"}` only after the structured graph backend can connect and bootstrap schema.
- `GET /v1/diagnostics` returns bearer-protected, non-secret tenant, config, and backend readiness details.
- `POST /v1/messages` records a scoped message for extraction into memory.
- `POST /v1/context` remains the legacy short-term context endpoint for backward compatibility. It still returns the historical `{"context": ...}` shape.
- `POST /v1/memory/context` is the combined official-style context endpoint for prompt-facing retrieval. It returns labeled sections for short-term memory, explicit long-term facts, upstream long-term preferences and entities, reasoning trace context, and optional local graph recall.
- `POST /v1/events` and `POST /v1/events/batch` ingest scoped structured events such as Discord messages, topology updates, reactions, links, and attachment metadata.
- `POST /v1/graph/context` returns scoped graph facts for a query. This is a custom local `GraphNode` and vector recall extension, using the configured embedding model when Neo4j vector schema is available.
- `POST /v1/skills` lists reviewed skill context for one tenant and agent.
- `POST /v1/skills/proposals` stores a proposed skill for human review.
- `POST /v1/skills/usage` records usage only for approved reviewed skills.

All endpoints except `/health` and `/ready` require `Authorization: Bearer <AGENTS_MEMORY_TOKEN>`.

## Memory layers and local extensions

- `POST /v1/memory/context` is the operator-facing combined endpoint. It keeps the official-style layered contract in one response so callers do not have to stitch memory pieces together themselves.
- `POST /v1/context` stays available as the legacy short-term-only path. Existing callers can keep using it, but new prompt assembly should prefer `/v1/memory/context`.
- Long-term facts are included explicitly in the combined response because the upstream long-term formatter separates preferences and entities from fact formatting. Local event promotion already writes scoped facts, so the combined endpoint surfaces them as their own labeled section.
- Reasoning trace context is limited to high-level action, observation, and tool records. It must not store raw chain-of-thought, raw prompt internals, or secrets.
- `POST /v1/graph/context` stays separate as a local extension. It returns scoped `GraphNode` recall and vector-backed graph facts without changing the graph endpoint's own response shape.
- `POST /v1/skills`, `POST /v1/skills/proposals`, and `POST /v1/skills/usage` stay separate from reasoning. Reviewed skills remain reviewed guidance, not hidden memory traces or self-modifying behavior.

## Discord memory scope and privacy

- `agents-memory` is policy-scoped first. Clients must send tenant, agent, session, user, and visibility metadata on every message, event, or query.
- PC Principal uses tenant `bromigos` and agent `pc-principal` by default, so recall stays inside that tenant and agent boundary.
- Visibility is enforced in the backend. `private_user` stays tied to the matching user, `channel` stays tied to the matching guild and channel, `guild` stays inside that guild, `agent_shared` stays within the same agent, and `global` is the only broad scope.
- Channel-scoped graph recall does not cross into sibling channels. This prevents cross-channel graph recall even when two channels live in the same guild.
- Topology deletes and renames are preserved as tombstones or event history, not hard-deleted facts. That keeps the audit trail while still reflecting current state.
- Structured event ingestion bootstraps Neo4j constraints, scalar indexes, and the `GraphNode.embedding` vector index before graph writes. Accepted events are written to graph state and promoted to embedded long-term facts through `MemoryClient.long_term.add_fact(..., generate_embedding=True)` with event, tenant, agent, session, user, visibility, guild, and channel metadata. Duplicate event deliveries repair current graph state but are not promoted to long-term facts again.

## Reviewed skill workflow

- Skills are not self-modifying executable behavior. The intended workflow is observe, propose, ask for approval, save an approved reviewed skill, then expose that approved record as non-executable context.
- `/v1/skills*` remains intentionally separate from `/v1/memory/context` and reasoning trace data. Operators should review it as approved guidance, not as an automatic memory layer.
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
- `MEMORY_EMBEDDING`: embedding model alias. Homelab deployments use the local-only LiteLLM alias `local-qwen3-embedding-0.6b`, backed by `Qwen/Qwen3-Embedding-0.6B` with 1024 dimensions through an in-cluster OpenAI-compatible `/v1` endpoint.
- `MEMORY_EMBEDDING_DIMENSIONS`: embedding vector dimensions. This must match the selected embedding model; `local-qwen3-embedding-0.6b` uses `1024`.

Neo4j agent memory embeddings must stay behind LiteLLM. Do not point PC Principal or `agents-memory` directly at an embedding runtime, and do not use OpenAI/Copilot aliases for memory embeddings. Before rolling out `local-qwen3-embedding-0.6b`, ensure the local embedding runtime serves `Qwen/Qwen3-Embedding-0.6B` at the in-cluster endpoint configured in the homelab LiteLLM wrapper chart.

The first API layer is intentionally small: agents pass scope metadata with every request, and future backend adapters enforce that scope before touching graph/vector memory.

For operators, keep the deployment path GitOps-only. Update the calling service's Helm values or manifests in Git, push to the tracked branch, and let ArgoCD reconcile. Don't bypass tracked services with manual cluster apply steps or manual chart install or upgrade steps.

## Rollout and rollback posture

- Treat `/v1/memory/context` as the new prompt-facing contract. Keep `/v1/context` available during rollout so older callers can continue using the short-term legacy response.
- Roll out combined context first, then any optional features that depend on it, such as graph recall or reasoning trace consumption in callers.
- Keep the service private. The intended homelab posture remains `ClusterIP` plus the existing private Traefik and GitOps flow.
- If operators need to back out, revert the smallest Git change or chart value that introduced the behavior, push the change, and let ArgoCD reconcile.
- A rollback should preserve endpoint compatibility. `/v1/context` stays available for old clients, `/v1/memory/context` can be disabled at the caller side, and `/v1/graph/context` plus `/v1/skills*` remain separate surfaces.
