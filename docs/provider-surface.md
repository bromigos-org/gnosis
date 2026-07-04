# Memory provider surface

This document records the agreed contract for the `/v1/memories` provider surface and the `/mcp` server, plus the implementation decisions and deviations that fell out of the installed `neo4j-agent-memory==0.5.0` SDK.

## Auth and scope

- All routes use the existing bearer service token (`GNOSIS_TOKEN`).
- Requests carry the existing `MemoryScope` (`tenant_id`, `space_id`, `agent_id`, `session_id`, `user_id`, `visibility`, optional `guild_id`, `channel_id`). Tenant enforcement matches the existing routes: a scope for another tenant is rejected with `403` before the backend runs.
- Scoping semantics: `scope.user_id` (with the tenant) is the read filter for search and list. `agent_id` and caller `metadata` are write-side tags stored on records.
- **Federation token class.** `POST /v1/memories`, `POST /v1/memories/search`, and `POST /v1/memories/list` additionally accept `GNOSIS_FEDERATION_TOKEN` (constant-time compared like every token class; the empty default disables it). Federated callers get the mandatory `metadata.shareable == true` filter conjunct injected on reads, must carry `metadata.promoted_from` on adds (`403` otherwise), cannot name `peers` in a search (`403`, prevents federation loops), and receive `403` on every other route - including `PATCH`/`DELETE /v1/memories/{memory_id}` and `POST /v1/memories/promote`.

## Endpoints

### `POST /v1/memories` - add

Body: `{scope, messages?: [{role: "user"|"assistant", content}], content?, infer: bool = true, metadata?}`

- `messages` + `infer=true`: extraction-mode add (turn sync). Each message flows through the SDK conversation path (`short_term.add_message`) and the long-term fact add path, producing one durable memory per message.
- `content` + `infer=false`: verbatim add as a durable long-term memory.
- Any other combination (both, neither, `messages` with `infer=false`, `content` with `infer=true`) is a `400`.

Returns `{results: [{memory_id, content, event: "ADD"|"UPDATE"|"NONE", metadata?}]}`.

Memory ids are the SDK `Fact.id` UUIDs persisted as the `id` property on the `Fact` node, so they are stable across reads, updates, and deletes. When the SDK deduplicates an add into an existing fact, the surviving record's id is returned with `event: "UPDATE"`. If an SDK result ever lacks an id, the gateway falls back to a parameterized lookup query keyed on the written triple and scope fragments; if that also fails the request errors rather than returning an id-less result.

### `POST /v1/memories/search`

Body: `{scope, query, filters?: FilterDSL, limit: int = 8, min_score?: float, peers?: [str] = []}`

Returns `{results: [{memory_id, content, score, metadata, created_at, updated_at}]}`, relevance-ranked by the SDK's vector similarity over long-term memories, scope-filtered, filter-evaluated, and redacted like other outbound payloads.

The federation extension is contract-additive; existing clients are unaffected:

- `peers` names federation peers to fan the same query out to (each must exist in `GNOSIS_PEERS` - `400` otherwise - and allow `pull` - `403` otherwise). The remote query maps `scope.tenant_id` to the peer's `remote_tenant_id`, keeps `user_id`, and never forwards `peers`, so fan-out is not transitive. Per-peer timeout is ~10s.
- When `peers` is non-empty, local and remote results merge by score descending (missing scores sort as 0), capped at `limit`, and every result gains `origin: "local" | "<peer name>"`. `origin` is omitted for plain non-federated searches.
- A failed, timed-out, or token-less peer never causes a 5xx; it is reported in `peer_errors: [{peer, error}]` (omitted when empty).

### `POST /v1/memories/list`

Body: `{scope, filters?: FilterDSL, page: int = 1, page_size: int = 50}`

Returns `{results: [...], total, page, page_size}` with deterministic ordering: `created_at` descending, then `id` ascending as a tiebreaker.

### `POST /v1/memories/promote`

Body: `{peer, scope, filters?: FilterDSL, limit: int = 50 (max 200), dry_run: bool = true}`

Pushes the caller's shareable memories to a federation peer. The peer must exist (`400`) and allow `push` (`403`). Candidates come from the same list internals as `/v1/memories/list`, using the caller's scope plus the mandatory `metadata.shareable == true` conjunct - unshared memories can never be promoted, regardless of caller filters.

- `dry_run=true` (the default - review-first, matching the dedup/consolidation posture) returns `{peer, count, dry_run, candidates: [{memory_id, content, metadata}]}` with no side effects.
- `dry_run=false` posts each candidate to the peer's `/v1/memories` as a verbatim add (`content` + `infer=false`) with `Authorization: Bearer <GNOSIS_PEER_<NAME>_TOKEN>`, scope `{tenant_id: <peer remote_tenant_id>, space_id: "federation", agent_id: "gnosis:<local tenant>", session_id: "promote", user_id: <same>, visibility: "private_user"}`, redaction applied to outbound content and metadata, and provenance metadata `{promoted_from: <local tenant>, source_memory_id, promoted_at}` merged in (`shareable` is stripped so sharing is never transitive). Returns a manifest `{peer, count, dry_run, promoted: [{source_memory_id, peer_memory_id, event}], failed: [{source_memory_id, error}]}`; partial failure is tolerated and reported per memory. Pushes use bounded concurrency (4) with a ~15s per-call timeout. A missing outbound peer token is a `503`.

Auth decision: the route requires the normal service token (`GNOSIS_TOKEN`), because callers promote their own scope; operator token classes and the federation token are not accepted (`403`).

### `PATCH /v1/memories/{memory_id}`

Body: `{scope, content?, metadata?}` -> `{memory_id, content, event: "UPDATE"}`. At least one of `content`/`metadata` is required (`400` otherwise). Metadata merges over the stored metadata; scope tags on the record cannot be overwritten by the caller.

### `DELETE /v1/memories/{memory_id}`

Body: `{scope}` -> `{memory_id, event: "DELETE"}`.

Both edit routes verify that the memory belongs to the request scope (tenant + `user_id`) before touching it and answer `404 memory not found in scope` otherwise - the same answer for "missing" and "owned by someone else", so cross-scope existence never leaks. Both are gated behind `GNOSIS_MEMORY_EDIT_ENABLED` (default `false`); while disabled they return `403` with `Memory editing is disabled by service policy.`. Every applied edit emits a structured audit log entry (`memory update applied` / `memory delete applied`) with tenant, agent, user, and memory id.

## Filter DSL

mem0-v2-style JSON:

- Logical: `{"AND": [...]}`, `{"OR": [...]}`, `{"NOT": {...}}`. A leaf object with several fields is an implicit `AND`; a field with several operators is an implicit `AND` of those operators.
- Leaf: `{"field": value}` (implicit `eq`) or `{"field": {"op": value}}`.
- Operators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `contains`, `icontains`.
- Fields: `user_id`, `agent_id`, `created_at`, `metadata.<key>`.
- Unknown fields or operators are rejected with `400`. `created_at` values must be ISO-8601 timestamps and do not accept `contains`/`icontains`; `gt`/`gte`/`lt`/`lte` on metadata fields expect numbers.

The parser (`gnosis/memory_filters.py`) validates the DSL into a typed tree and enforces it in two phases, following the `graph_query_validation.py` philosophy that the validator, not the query author, is the enforcement boundary:

1. **Parameterized Cypher narrowing.** The tree is translated into a `WHERE` fragment; every value is bound as a query parameter - values are never string-interpolated into Cypher. Because record metadata is stored by the SDK as a JSON string property, metadata conditions narrow via `CONTAINS` on parameterized JSON fragments. The translation is polarity-aware: a fragment is only emitted where it is provably a superset of the true matches (or exact, for `created_at` property comparisons, which is why they may appear under `NOT`). Anything not provably safe degrades to `true` rather than risking a wrong exclusion.
2. **Exact in-gateway evaluation.** Every row that survives narrowing is re-checked against the full DSL semantics on the deserialized record before it can leave the service. This phase is authoritative.

## Storage model and SDK notes

Provider memories are SDK long-term `Fact` nodes:

- Verbatim adds use the predicate `memory`; turn-sync adds use `said_user` / `said_assistant`, matching the existing `/v1/messages` write path.
- Scope fields are write-side tags inside the fact's metadata JSON, alongside caller metadata (redacted before storage).
- `created_at` is a real node property set by the SDK; list ordering and `created_at` filters use it directly.

## Deviations from an "ideal" provider backend

These are consequences of the installed SDK and are the closest safe equivalents:

1. **No SDK update/delete for single memories.** `neo4j-agent-memory==0.5.0` exposes no per-fact update or delete API. Update and delete are implemented as gateway-owned, parameterized Cypher (`SET .../coalesce` and `MATCH ... DETACH DELETE`) executed through the SDK client's graph write handle after the scope-ownership check. Ids are preserved across updates. If the SDK client exposes no graph write surface, the routes answer `501 capability_unavailable`. Note that in 0.5.0 `client.graph` is a deprecation proxy that exposes `execute_write` only through dynamic `__getattr__` delegation, which the runtime-protocol `isinstance` check cannot see on Python 3.12+ (it resolves members with `inspect.getattr_static`); the write-handle acquisition therefore falls back to a duck-typed `execute_write` check so update/delete ride the same driver/session the read routes use instead of returning a spurious `501`.
2. **Embedding refresh on content update.** Content updates re-embed through the SDK client's embedder when available so semantic search stays correct; if no embedder is reachable the previous embedding is left in place (the memory stays findable, possibly with slightly stale ranking) rather than being dropped from the vector index.
3. **Extraction-mode adds are the turn-sync path.** The SDK has entity/relation extraction but no separate fact-extraction batch API; `messages` + `infer=true` therefore rides the same short-term + long-term path as `/v1/messages`, with entity/relation extraction governed by the existing `GNOSIS_EXTRACT_*` flags (off by default). `event: "NONE"` is reserved in the response schema but not currently emitted.
4. **List scan cap.** Because metadata lives in a JSON string, exact filtering happens in the gateway after Cypher narrowing; list pagination therefore scans at most 2000 scoped rows per request (newest first) before paging. `total` is exact within that window.
5. **Search candidate pool.** Search fetches up to 100 vector candidates from the SDK before scope/filter/score pruning, so heavily filtered queries can return fewer than `limit` results even when older matches exist.
6. **Scores.** `score` is the SDK vector similarity surfaced by `search_facts`, clamped to `[0, 1]`. `min_score` is applied after scope and filter checks.

## MCP server

- Official `mcp` Python SDK, streamable-HTTP transport, mounted in the FastAPI app at `/mcp`.
- Feature-flagged by `GNOSIS_MCP_ENABLED` (default `false`); requires the same bearer token, enforced by an ASGI wrapper in front of the MCP transport. DNS-rebinding protection in the MCP transport is disabled because gnosis terminates auth itself and is served behind ingress hostnames.
- Exactly six tools:
  - `add_memory(content, user_id, metadata?, infer=false)` - `infer=true` routes the content through the extraction-mode add as a user message.
  - `search_memory(query, user_id, limit=8)`
  - `get_context(query, user_id, max_items=8)` - wraps the existing combined memory-context assembly.
  - `list_memories(user_id, page=1)`
  - `delete_memory(memory_id, user_id)` - honors `GNOSIS_MEMORY_EDIT_ENABLED`.
  - `get_status()` - service readiness plus redacted backend diagnostics.
- Tools construct the scope server-side: tenant from settings, `space_id="mcp"`, `agent_id` from `GNOSIS_MCP_AGENT_ID` (default `mcp-client`), `session_id="mcp:<user_id>"`, visibility `private_user`. The module stays thin; all logic lives in the backend layer shared with the HTTP routes.

## Settings added by this surface

- `GNOSIS_MEMORY_EDIT_ENABLED` (default `false`)
- `GNOSIS_MCP_ENABLED` (default `false`)
- `GNOSIS_MCP_AGENT_ID` (default `mcp-client`)
- `GNOSIS_PEERS` (default `[]`) - JSON federation peer registry: `[{"name", "base_url", "direction": "both"|"push"|"pull", "remote_tenant_id"}]`, validated at startup with unique names.
- `GNOSIS_PEER_<NAME>_TOKEN` - outbound bearer token per peer (peer name uppercased, `-` mapped to `_`); the value is the remote instance's `GNOSIS_FEDERATION_TOKEN`.
- `GNOSIS_FEDERATION_TOKEN` (default empty = inbound federation disabled) - the inbound federation token class described under Auth and scope.
