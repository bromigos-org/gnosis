# gnosis

**gnosis** is a self-hosted memory service for AI agents. It puts an authenticated,
tenant-scoped HTTP gateway in front of a Neo4j graph/vector store and an
OpenAI-compatible LLM/embedding endpoint. Clients use HTTP (or the optional MCP
mount); they do not connect to Neo4j, Bolt, or the Python SDK directly.

The gateway keeps long-term recall keyed by `tenant_id` + `user_id`, while
retaining agent and session fields as write-side provenance. Responses are
scope-checked, redacted, and rendered as prompt-safe sections.

## Documentation

- [Getting started](docs/getting-started.md) — a longer integration walkthrough.
- [Architecture](docs/architecture.md) — request flow and module map.
- [Data model](docs/data-model.md) — graph schema and scope spine.
- [Provider surface](docs/provider-surface.md) — HTTP and MCP contracts.
- [Configuration](docs/configuration.md) — every environment variable and YAML key.
- [Security](docs/security.md) — token classes, scope, redaction, and federation.
- [Operations](docs/operations.md) — health, workers, backup, and scaling notes.
- [Capabilities](docs/CAPABILITIES.md) — feature behavior and measured tradeoffs.
- [Development](docs/development.md) — contribution and measurement workflow.
- [Benchmarks](docs/BENCHMARKS.md) — the maintained benchmark ledger.

## Five-minute local stack

The tracked [`compose.yaml`](compose.yaml) starts Neo4j 5.26+ and the published
`ghcr.io/nolgiainc/gnosis:latest` image. You need Docker Compose v2 and an
OpenAI-compatible chat/embedding endpoint. By default the compose file points at
Ollama on the host; use the variables in the file (or a `.env` next to it) for
LiteLLM, OpenAI, or another endpoint. For the Ollama default, pull the models
before starting the stack:

```bash
git clone https://github.com/nolgiainc/gnosis.git
cd gnosis
ollama pull llama3.2:latest
ollama pull nomic-embed-text
docker compose up -d
```

The compose file supplies development-only placeholder tokens and enables
memory editing for this local example. Replace them with secret-backed values
before sharing a deployment. Check liveness and backend readiness:

```bash
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:8080/ready
```

Compose also publishes Neo4j's browser and Bolt ports (`7474` and `7687`) for
local development. Do not expose this default binding on an untrusted network;
use a reviewed override or deployment network policy for shared environments.

`/health` is a shallow liveness response. `/ready` is `200` only after the
Neo4j graph, schema bootstrap, and write buffer report ready; it does not prove
that an LLM request will succeed. Stop this disposable stack with
`docker compose down -v` when finished (the `-v` removes its named Neo4j volume).

### Write and read one memory

The provider surface accepts either `content` with `infer: false` (a verbatim
write) or `messages` with `infer: true` (a request for the extraction path when
that feature is enabled). The following verbatim smoke test avoids an extra
extraction LLM call while still exercising the real embedding and read paths:

```bash
export GNOSIS_URL=http://localhost:8080
export GNOSIS_TOKEN=dev-token

curl -fsS "$GNOSIS_URL/v1/memories" \
  -H "Authorization: Bearer $GNOSIS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "scope": {"tenant_id":"nolgia","space_id":"demo","agent_id":"assistant",
              "session_id":"session-1","user_id":"alice","visibility":"private_user"},
    "content": "Alice moved from Seattle to Austin in March.",
    "infer": false
  }'

curl -fsS "$GNOSIS_URL/v1/memory/context" \
  -H "Authorization: Bearer $GNOSIS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "scope": {"tenant_id":"nolgia","space_id":"demo","agent_id":"assistant",
              "session_id":"session-2","user_id":"alice","visibility":"private_user"},
    "query": "Where does Alice live?",
    "max_items": 8
  }'
```

The response contains `sections[]` with the scoped, redacted context. To use
the extraction path instead, send a `messages` array and `infer: true`; enable a
capable `GNOSIS_LLM`, because extraction and adaptive routing make LLM calls.
`POST /v1/memories/search` returns ranked records instead of assembled context.

## What the service provides

### Write and read paths

- `POST /v1/memories` stores a verbatim memory and can extract dated fact units
  from a turn pair when extraction is enabled. `POST /v1/messages` is the
  message-oriented equivalent.
- `POST /v1/memory/context` assembles prompt-ready short-term, long-term,
  graph, and reasoning sections. `POST /v1/memories/search` and
  `POST /v1/memories/list` provide ranked and deterministic record views.
- Optional write features include fact extraction, entity/relationship graph
  materialization, document extraction preview, buffered writes, and a bounded
  background extraction worker. Write-path features affect newly ingested data;
  enabling them does not rebuild existing records.
- Optional read features include BM25+dense fusion, scope-narrowed dense
  retrieval, read-time newest-wins supersession, graph-QA fusion, entity/bridge
  traversal, Chain-of-Note grounding, facts-to-verbatim expansion, sufficiency
  assessment, and adaptive per-query routing.

### Reranking

`GNOSIS_RERANK_ENABLED` adds a listwise LLM reranker over the fused candidate
pool before the item-budget cut. It reorders (never drops) the first
`GNOSIS_RERANK_CANDIDATE_CAP` candidates, which defaults to `50`; the model is
`GNOSIS_RERANK_MODEL`, falling back to `GNOSIS_LLM`. A failed or malformed
rerank response keeps retrieval order and does not fail the context request.
The flag is default-off and its production benchmark result is not yet claimed.

Graph-QA fusion is similarly best-effort: generated Cypher is passed through the
repository's graph-query validator and the contract requires parameterized,
read-only plans with tenant scope plus user, guild, or channel scope as
applicable. `GNOSIS_GRAPHQA_FUSION_TIMEOUT_SECONDS` defaults to **20 seconds**;
a timeout, validation rejection, or backend error degrades to dense retrieval
rather than failing the read.

### Trust boundary and safety

- Every stored memory and memory-context request is scoped by `tenant_id`,
  `space_id`, `agent_id`, `session_id`, `user_id`, and `visibility`. Tenant
  mismatches are rejected before backend access, and long-term recall is shared
  across sessions for the same tenant and user.
- `GNOSIS_TOKEN` authenticates normal callers. Read, write, export, admin, and
  federation operations use separate least-privilege token classes:
  `GNOSIS_READ_OPERATOR_TOKEN`, `GNOSIS_WRITE_OPERATOR_TOKEN`,
  `GNOSIS_EXPORT_OPERATOR_TOKEN`, `GNOSIS_ADMIN_OPERATOR_TOKEN`, and
  `GNOSIS_FEDERATION_TOKEN`.
- Prompt-facing output is redacted. Deduplication and consolidation are
  review-first (dry run before apply), and graph-QA never accepts caller-supplied
  Cypher. Federation is off by default and only promotes memories explicitly
  tagged `metadata.shareable: true`.

## API surface

Application data and operator routes require `Authorization: Bearer <token>`;
`/health` and `/ready` are unauthenticated. FastAPI's generated `/docs`,
`/redoc`, and `/openapi.json` are also public unless the deployment disables
them. The provider schemas and MCP contract are in
[docs/provider-surface.md](docs/provider-surface.md); broader context, event,
reasoning, skills, and operator routes are implemented under `src/gnosis/routes/`
and outlined in [architecture.md](docs/architecture.md) and
[operations.md](docs/operations.md).

| Common surface | Routes |
| --- | --- |
| Health | `GET /health`, `GET /ready`, authenticated `GET /v1/diagnostics` |
| Memory | `POST /v1/memories`, `/v1/memories/search`, `/v1/memories/list`, `/v1/memories/promote` |
| Context | `POST /v1/memory/context`, `/v1/graph/context`, `/v1/reasoning/context` |
| Ingestion | `POST /v1/messages`, `/v1/events`, `/v1/events/batch`, `/v1/memory/extraction/preview` |
| Editing | `PATCH`/`DELETE /v1/memories/{memory_id}` when `GNOSIS_MEMORY_EDIT_ENABLED=true` |
| MCP | Streamable HTTP at `/mcp` when `GNOSIS_MCP_ENABLED=true` |

`POST /v1/context` remains as a deprecated short-term compatibility route; new
clients should use `/v1/memory/context`.

## Configuration and runtime commands

Minimum service inputs are `GNOSIS_TOKEN`, the operator/federation token
variables used by your routes, `NEO4J_URI`, `NEO4J_USERNAME`,
`NEO4J_PASSWORD`, `LITELLM_BASE_URL`, and `LITELLM_API_KEY`. The code defaults
`GNOSIS_TENANT_ID` to `nolgia`, `GNOSIS_LLM` to `openai/gemma4`,
`GNOSIS_EMBEDDING` to `local-qwen3-embedding-0.6b`, and embedding dimensions to
`1024`; set model and tenant values explicitly for a deployment, especially
when using the preferred config. See [configuration.md](docs/configuration.md)
for the complete matrix.

Configuration precedence is **explicit environment variables → `.env` → YAML
config → code defaults**. When `GNOSIS_CONFIG_FILE` is unset, the service
auto-loads [`configs/default.yaml`](configs/default.yaml), the preferred
benchmark configuration. It enables fact extraction, entity graph materialization,
adaptive routing, and route-aware Chain-of-Note. Set `GNOSIS_CONFIG_FILE` to a
different file (for example `configs/runs/run11.yaml`) or set it to an empty
string to use the minimal code defaults, where optional quality features are
off. The preferred config requires a capable chat model; `GNOSIS_LLM`'s code
default is `openai/gemma4` and is not adequate for extraction.

Run from a checkout with Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run uvicorn gnosis.main:app --host localhost --port 8080
```

The source command expects the same Neo4j, token, LLM, and embedding variables
as the container. There is no standalone `gnosis` executable in this package;
the container and the source command both start `uvicorn gnosis.main:app`.

## Deployment boundary

The repository owns the container and local Compose contract:

- [`Dockerfile`](Dockerfile) builds from the pinned `uv.lock`, copies `src/`
  and `configs/`, and starts Uvicorn on port `8080`.
- [`compose.yaml`](compose.yaml) is a minimal Neo4j + service stack for local
  use. It is not a production topology or secret-management system.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs tests for pull
  requests. A push to `main` runs the same test job, then builds and publishes
  `ghcr.io/nolgiainc/gnosis:latest` and `ghcr.io/nolgiainc/gnosis:sha-<commit>`.

Kubernetes, ingress, secret managers, and rollout controllers are deployment
concerns owned by the environment that runs this image; this repository does
not define or promise a particular orchestrator. Keep backend credentials and
all token classes in that environment's secret-backed configuration.

## Development and CI

Install the locked runtime and development dependencies, then run the same four
gates used by the CI test job:

```bash
uv sync --locked
uv run ruff check
uv run ruff format --check
uv run basedpyright
uv run pytest -q
```

The Docker build is a separate `main`-push job and does not run for pull
requests. For feature work, keep optional flags default-off and make LLM-backed
read enhancements degrade to the simpler path. Measure changes with the
[gnosis-membench](https://github.com/nolgiainc/gnosis-membench) harness before
making quality claims; update the maintained capability and benchmark docs with
both positive and negative results.

## Benchmark headline

The preferred `configs/default.yaml` is Run 18. On the fast LOCOMO subset-3
development gate it scored **74.8 J excluding adversarial questions** (76.7
overall). Run 23 remeasured the same configuration on all ten LOCOMO
conversations at **66.9–68.9 J excluding adversarial questions**, using GPT-5.5
and GPT-5.4-mini judges. That full-LOCOMO result is directional against
published systems because judge, model, and protocol details differ; it is not a
claim of universal comparability or a leaderboard position. The subset-3 score
is a regression signal, not a competitive comparison.

The README intentionally does not reproduce every run, category table, or cost
assumption. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and the
[gnosis-membench RESULTS.md](https://github.com/nolgiainc/gnosis-membench/blob/main/RESULTS.md)
for the canonical ledger, frozen protocols, deviations, and negative results.

## Related projects

- [hermes-gnosis](https://github.com/nolgiainc/hermes-gnosis) — memory-provider
  plugin for NousResearch hermes agents.
- [gnosis-membench](https://github.com/nolgiainc/gnosis-membench) — benchmark
  harness for LOCOMO and LongMemEval experiments.

The service builds on `neo4j-agent-memory==0.5.0`; the Nolgia-specific layer is
the HTTP contract, authentication, scope policy, rollout controls, and
redaction boundary.
