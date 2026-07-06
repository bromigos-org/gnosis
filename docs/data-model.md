# Data model

gnosis stores memory in Neo4j as a hybrid **graph + vector** store: episodic
messages and extracted facts are nodes with embeddings, and an entity knowledge
graph links them. Every record is scoped, dated, and append-only. This page is
the reference for the node labels, relationships, properties, and the scope
spine. See [CAPABILITIES.md](CAPABILITIES.md) for how reads use this model and
[provider-surface.md](provider-surface.md) for the HTTP contract.

## The scope spine

Every record carries the six-field scope that the gateway enforces on every read
and write:

| Field | Role |
|---|---|
| `tenant_id` | top-level isolation boundary (one business/deployment) |
| `space_id` | a namespace within a tenant |
| `agent_id` | which agent wrote it — a **write-side audit tag**, not a recall filter |
| `session_id` | write provenance (a conversation/session) — not a read filter |
| `user_id` | the subject the memory is about |
| `visibility` | e.g. `private_user`, `channel` |

**Long-term recall is keyed by `tenant_id` + `user_id`.** Two agents on the same
deployment asking about the same user see the same memories; `agent_id`,
`session_id`, and caller metadata are stored for audit and filtered views but do
**not** partition recall (and are redacted out of prompt-facing content).
Different business entities run separate deployments with separate tenants.

## Bi-temporal + append-only

Facts carry two timestamps — `event_date` (when the thing happened, in metadata)
and `created_at` (when written). Storage is append-only: contradictions are
resolved at **read time** (deterministic newest-wins), never by mutating or
deleting stored records. See supersession in [CAPABILITIES.md](CAPABILITIES.md).

## Node labels

23 labels, grouped by role.

**Memory core**
| Label | Contents |
|---|---|
| `Fact` | extracted atomic fact unit (`subject`, `predicate`, `object`, embedding, metadata incl. `event_date`, `confidence`) |
| `Message` | a verbatim conversation turn (episodic layer) |
| `Conversation` | a session container for messages |
| `Preference` | a stored user preference |
| `Entity` | a knowledge-graph entity (`name`, `normalized`, deduped within `tenant_id`+`user_id`) |

**Reasoning** (audit-only; kept out of prompt recall)
| Label | Contents |
|---|---|
| `ReasoningTrace` / `ReasoningStep` | stored reasoning for audit and reuse |
| `Tool` / `ToolCall` | tool-call records |

**Identity & scope**
`Tenant`, `Agent`, `Client`, `User` — the scope/identity nodes.

**Source modules** (populated by the writing client's domain, e.g. a Discord
bot): `Guild`, `Channel`, `Role`, `Category`, `Link`, `Attachment`,
`Event`. These are source-specific and generalize behind the scope spine.

**Operations / audit**
`ConsolidationRun` (dedup/consolidation runs), `MemoryReadAudit` (read audit when
`GNOSIS_AUDIT_READ` is on), `GraphNode` (generic graph node).

## Relationships

6 relationship types.

| Relationship | Pattern | Meaning |
|---|---|---|
| `HAS_CONVERSATION` | scope → `Conversation` | a session belongs to a scope |
| `HAS_MESSAGE` | `Conversation` → `Message` | a turn belongs to a session |
| `FIRST_MESSAGE` | `Conversation` → `Message` | the session's first turn |
| `NEXT_MESSAGE` | `Message` → `Message` | turn ordering within a session |
| `MENTIONS` | `Fact` → `Entity` | provenance: which entities a fact names |
| `RELATES` | `Entity` → `Entity` | a directed `{relation, fact_id, event_date}` edge from an extracted `(head, relation, tail)` triple |

`MENTIONS` and `RELATES` form the **entity knowledge graph**
(`GNOSIS_ENTITY_GRAPH_ENABLED`) that the graph-QA and traversal read paths walk;
they only exist when entity-graph materialization was on at ingest.

## Key node properties

**`Fact`**: `id`, `subject`, `predicate`, `object`, `created_at`, `confidence`,
`embedding` (vector), `metadata` (JSON string — carries `event_date`, `entities`,
`source_memory_ids`, `session_date`, plus the scope fields for the in-query
scope re-check).

**`Entity`**: `id`, `name`, `normalized` (dedup key), `tenant_id`, `user_id`,
`created_at`.

Metadata is stored as a JSON string on nodes (an SDK deviation); the gateway
parses, scope-re-checks, and redacts it before it reaches a prompt.

## Predicates

The `predicate` field distinguishes memory kinds and drives rendering and
supersession:

- `fact` — an extracted EDU-v1 fact unit (participates in the entity graph and
  slot-based supersession).
- `memory` — a verbatim stored memory.
- `said_user` / `said_assistant` (turn-prefixed) — verbatim turns; rendered
  without subject/predicate and never superseded.
