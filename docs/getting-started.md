# Getting started

Install gnosis, bring it up, and make your first write and read. For the full
API contract see [provider-surface.md](provider-surface.md); for production
deployment see [operations.md](operations.md).

## What gnosis needs

gnosis is a stateless gateway; it depends on two backing services:

- **Neo4j 5.26+** — the graph + vector store. gnosis holds the only credentials
  to it; clients never touch Bolt.
- **An OpenAI-compatible LLM + embedding endpoint** — [LiteLLM](https://litellm.ai),
  or ollama's `/v1` for a fully local stack. Embeddings are always used; the
  optional read/write LLM features need a *capable* chat model.

Everything else (tokens, config) is environment. gnosis ships as the
`ghcr.io/bromigos-org/gnosis` image (uvicorn on `:8080`) and bundles `configs/`,
so it auto-loads the preferred config on start.

## Fastest path — the local stack

The sibling [`gnosis-membench`](https://github.com/bromigos-org/gnosis-membench)
harness ships a compose file that brings up **Neo4j + gnosis** wired together
(point it at ollama or a LiteLLM you already run). This is the quickest way to a
running instance:

```bash
git clone https://github.com/bromigos-org/gnosis-membench
cd gnosis-membench/stack
# set tokens + LLM endpoint in the environment or an .env beside compose.yaml
docker compose up
```

gnosis comes up on `http://localhost:8080` (override with `GNOSIS_PORT`). Skip to
[Verify it's up](#verify-its-up).

## Running the image directly

If you already have Neo4j and an LLM endpoint, run the image with the minimum
environment. Five token classes are required (least-privilege — see
[security.md](security.md)); generate real random secrets for each:

```bash
docker run --rm -p 8080:8080 \
  -e GNOSIS_TOKEN=$(openssl rand -hex 32) \
  -e GNOSIS_READ_OPERATOR_TOKEN=$(openssl rand -hex 32) \
  -e GNOSIS_WRITE_OPERATOR_TOKEN=$(openssl rand -hex 32) \
  -e GNOSIS_EXPORT_OPERATOR_TOKEN=$(openssl rand -hex 32) \
  -e GNOSIS_ADMIN_OPERATOR_TOKEN=$(openssl rand -hex 32) \
  -e GNOSIS_TENANT_ID=bromigos \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e NEO4J_USERNAME=neo4j -e NEO4J_PASSWORD=... \
  -e LITELLM_BASE_URL=http://litellm:4000/v1 -e LITELLM_API_KEY=... \
  -e GNOSIS_LLM=openai/<capable-chat-model> \
  -e GNOSIS_EMBEDDING=<embedding-model> \
  -e GNOSIS_EMBEDDING_DIMENSIONS=<dim> \
  ghcr.io/bromigos-org/gnosis
```

`GNOSIS_TENANT_ID` (default `bromigos`) is the deployment's isolation boundary:
every request's `scope.tenant_id` **must match it** or the gateway returns `403`
before the backend runs. The examples below use `bromigos`; change both together.

### From source (for development)

gnosis targets **Python 3.13** and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/bromigos-org/gnosis && cd gnosis
uv sync
# export the same env vars as above, then:
uv run uvicorn gnosis.main:app --host 0.0.0.0 --port 8080
```

See [development.md](development.md) for the test suite and the CI gates.

## Verify it's up

Two unauthenticated probes:

```bash
curl -s localhost:8080/health   # {"status":"ok"}       — liveness
curl -s localhost:8080/ready    # {"status":"ready"}    — deps reachable (503 until they are)
```

Use `/ready` as your orchestrator's readiness probe so traffic only arrives once
Neo4j and the LLM endpoint answer. Everything past this point authenticates with
the service token:

```bash
export GNOSIS_URL=http://localhost:8080
export TOKEN=<your GNOSIS_TOKEN>
```

## The scope, once

Every read and write carries a six-field **scope**. It is the whole access model,
so it's worth understanding before the first call:

| Field | Role |
|---|---|
| `tenant_id` | isolation boundary — must equal `GNOSIS_TENANT_ID` |
| `space_id` | a namespace within the tenant |
| `agent_id` | which agent wrote it — a write-side audit tag |
| `session_id` | write provenance (a conversation) |
| `user_id` | the subject the memory is *about* |
| `visibility` | `private_user`, `channel`, `guild`, `tenant`, `global`, … |

**Long-term recall is keyed by `tenant_id` + `user_id`.** Two agents on the same
deployment asking about the same user see the same memories; `agent_id` and
`session_id` are stored for audit but do not partition recall. Full model:
[data-model.md](data-model.md).

## First write

`POST /v1/memories` in extraction mode (`infer: true`) turns conversation turns
into durable memory. Each message becomes a memory; with fact extraction on (it is
in the preferred config) the turns are also decomposed into atomic facts.

```bash
curl -s $GNOSIS_URL/v1/memories \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{
    "scope": {
      "tenant_id": "bromigos", "space_id": "demo", "agent_id": "assistant",
      "session_id": "sess-1", "user_id": "alice", "visibility": "private_user"
    },
    "messages": [
      {"role": "user", "content": "I moved from Seattle to Austin last March."},
      {"role": "assistant", "content": "Got it — you’re in Austin now."}
    ],
    "infer": true
  }'
```

Response — one result per durable memory (plus extracted facts when extraction is
on), each with a stable `memory_id`:

```json
{"results": [{"memory_id": "…", "content": "…", "event": "ADD", "metadata": {}}]}
```

To store a fact verbatim instead of extracting it, send `content` with
`infer: false`:

```bash
curl -s $GNOSIS_URL/v1/memories \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"scope": {"tenant_id":"bromigos","space_id":"demo","agent_id":"assistant",
       "session_id":"sess-1","user_id":"alice","visibility":"private_user"},
       "content": "Alice lives in Austin.", "infer": false}'
```

## First read

There are two read surfaces. **`/v1/memory/context`** is the primary one: it
assembles prompt-ready, scope-checked, redacted `sections[]` for a query — drop it
straight into your model's context window.

```bash
curl -s $GNOSIS_URL/v1/memory/context \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{
    "scope": {"tenant_id":"bromigos","space_id":"demo","agent_id":"assistant",
              "session_id":"sess-2","user_id":"alice","visibility":"private_user"},
    "query": "Where does Alice live?",
    "max_items": 8
  }'
```

```json
{"sections": [{"source": "long_term", "content": "…Alice lives in Austin…", "facts": [ … ]}]}
```

Use the same `tenant_id` + `user_id` you wrote under — that pair is the recall
key, so the reading session/agent need not match the writing one.

For raw ranked hits instead of assembled context, use **`/v1/memories/search`**:

```bash
curl -s $GNOSIS_URL/v1/memories/search \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"scope": {"tenant_id":"bromigos","space_id":"demo","agent_id":"assistant",
       "session_id":"sess-2","user_id":"alice","visibility":"private_user"},
       "query": "where does alice live", "limit": 8}'
```

```json
{"results": [{"memory_id": "…", "content": "Alice lives in Austin.", "score": 0.83, "metadata": {}}]}
```

> Extracted facts and the entity graph are a **write-path** capability: the payoff
> shows up on data ingested with the preferred config on, not retroactively. If
> your first read looks thin, confirm the write used `infer: true` and give
> background extraction a moment to flush.

## Choosing a config

gnosis auto-loads **`configs/default.yaml`** — the preferred, benchmark-tuned
config — on start. To pick another, point `GNOSIS_CONFIG_FILE` at a
`configs/runs/runN.yaml` or your own file; set it to `""` for the minimal safe
defaults (all optional features off). Precedence is
**env vars → `.env` → the YAML file → code defaults**, so a single env var still
overrides one key without editing the file. The catalogue of runs and what each
enables is in [`configs/`](../configs/README.md); the full setting reference is
[configuration.md](configuration.md).

## Where to next

- [CAPABILITIES.md](CAPABILITIES.md) — what each read/write technique does and why.
- [provider-surface.md](provider-surface.md) — the complete HTTP contract:
  list, edit/delete, promote, the filter DSL, and the `/mcp` server.
- [operations.md](operations.md) — running it for real: health, the extraction
  worker, backup, scale.
- [security.md](security.md) — the token classes, scope enforcement, redaction,
  and federation safety.
