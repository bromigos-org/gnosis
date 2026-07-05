# Getting started

gnosis gives your agents durable, cross-session memory behind an auth + scope +
redaction gateway. This guide brings gnosis up, then wires a **real project** to
it — a [NousResearch **hermes**](https://github.com/NousResearch/hermes-agent)
agent, via the [hermes-gnosis](https://github.com/bromigos-org/hermes-gnosis)
memory-provider plugin — so you finish with an agent that remembers across turns
and sessions. If you're integrating a different client, the raw HTTP path is at
the end and the full contract is in [provider-surface.md](provider-surface.md).

## What gnosis needs

gnosis is a stateless gateway with two backing services:

- **Neo4j 5.26+** — the graph + vector store. gnosis holds the only credentials
  to it; clients never touch Bolt.
- **An OpenAI-compatible LLM + embedding endpoint** — [LiteLLM](https://litellm.ai),
  or [ollama](https://ollama.com)'s `/v1` for a fully local setup. Embeddings are
  always used; the write/read LLM features (extraction, routing) want a *capable*
  chat model.

The `ghcr.io/bromigos-org/gnosis` image runs uvicorn on `:8080` and bundles
`configs/`, so it auto-loads the preferred config on start.

## 1 — Run gnosis

The repo ships a minimal [`compose.yaml`](../compose.yaml) (Neo4j + gnosis) — the
fastest way up. It defaults to ollama on the host; edit the `OPENAI_BASE_URL` /
`GNOSIS_LLM` / `GNOSIS_EMBEDDING` vars for any other endpoint.

```bash
git clone https://github.com/bromigos-org/gnosis && cd gnosis
# for the local ollama default, pull the models first:
#   ollama pull llama3.2:latest && ollama pull nomic-embed-text
docker compose up -d
```

<details>
<summary>Run the image directly, or from source</summary>

**Image, against your own Neo4j + LLM.** Five token classes are required
(least-privilege — [security.md](security.md)); generate real secrets:

```bash
docker run --rm -p 8080:8080 \
  -e GNOSIS_TOKEN=$(openssl rand -hex 32) \
  -e GNOSIS_READ_OPERATOR_TOKEN=$(openssl rand -hex 32) \
  -e GNOSIS_WRITE_OPERATOR_TOKEN=$(openssl rand -hex 32) \
  -e GNOSIS_EXPORT_OPERATOR_TOKEN=$(openssl rand -hex 32) \
  -e GNOSIS_ADMIN_OPERATOR_TOKEN=$(openssl rand -hex 32) \
  -e GNOSIS_TENANT_ID=bromigos \
  -e NEO4J_URI=bolt://neo4j:7687 -e NEO4J_USERNAME=neo4j -e NEO4J_PASSWORD=... \
  -e LITELLM_BASE_URL=http://litellm:4000/v1 -e LITELLM_API_KEY=... \
  -e GNOSIS_LLM=openai/<capable-chat-model> \
  -e GNOSIS_EMBEDDING=<embedding-model> -e GNOSIS_EMBEDDING_DIMENSIONS=<dim> \
  ghcr.io/bromigos-org/gnosis
```

**From source** (Python 3.13 + [uv](https://docs.astral.sh/uv/); see
[development.md](development.md)):

```bash
uv sync
uv run uvicorn gnosis.main:app --host 0.0.0.0 --port 8080   # export the env vars first
```
</details>

`GNOSIS_TENANT_ID` (default `bromigos`) is the deployment's isolation boundary:
every request's `scope.tenant_id` **must match it** or the gateway returns `403`
before the backend runs. Keep the tenant consistent between gnosis and every
client below.

## 2 — Verify it's up

Two unauthenticated probes:

```bash
curl -s localhost:8080/health   # {"status":"ok"}      — liveness
curl -s localhost:8080/ready    # {"status":"ready"}   — deps reachable (503 until they are)
```

Use `/ready` as your orchestrator's readiness probe. Everything past this point
authenticates with the service token (`GNOSIS_TOKEN`).

## 3 — The scope, once

Every read and write carries a six-field **scope** — the whole access model:

| Field | Role |
|---|---|
| `tenant_id` | isolation boundary — must equal `GNOSIS_TENANT_ID` |
| `space_id` | a namespace within the tenant (hermes uses `hermes`) |
| `agent_id` | which agent wrote it — a write-side audit tag |
| `session_id` | write provenance (a conversation) |
| `user_id` | the subject the memory is *about* |
| `visibility` | `private_user`, `channel`, `guild`, `tenant`, `global`, … |

**Long-term recall is keyed by `tenant_id` + `user_id`.** Two agents on the same
deployment asking about the same user see the same memories; `agent_id` and
`session_id` are stored for audit but do not partition recall. Full model:
[data-model.md](data-model.md).

## 4 — Use it in a real project: a hermes agent

[hermes-gnosis](https://github.com/bromigos-org/hermes-gnosis) is a drop-in
memory provider for [NousResearch hermes-agent](https://github.com/NousResearch/hermes-agent).
Point a hermes agent at your gnosis instance and it gains long-term memory with
no code changes — this is the intended integration path.

**Install the plugin** (hermes discovers providers from `$HERMES_HOME/plugins/`,
default `~/.hermes/plugins/`):

```bash
pip install git+https://github.com/bromigos-org/hermes-gnosis   # or a local checkout path
hermes-gnosis-install                                           # copies it into $HERMES_HOME/plugins/gnosis/
```

(Or skip pip and symlink the package:
`ln -s /path/to/hermes-gnosis/hermes_gnosis ~/.hermes/plugins/gnosis`.)

**Point it at gnosis** and drop in the service token:

```bash
hermes config set memory.provider gnosis
echo 'GNOSIS_SERVICE_TOKEN=<your GNOSIS_TOKEN>' >> ~/.hermes/.env
```

Then set the connection in `$HERMES_HOME/gnosis.json` (or run
`hermes memory setup` and pick `gnosis`). `gnosis_url` is required; `tenant_id`
**must match** the server's `GNOSIS_TENANT_ID`:

```json
{
  "gnosis_url": "http://localhost:8080",
  "tenant_id": "bromigos",
  "user_id": "hermes-user",
  "agent_id": "hermes"
}
```

**That's it — the agent now remembers.** Per turn the plugin:

- **prefetches** a semantic search of past memory before the model runs (bounded
  ~1.5 s, never blocks the loop) and injects the top hits into the system prompt;
- **syncs the turn** — after each `(user, assistant)` exchange it sends the pair
  to `POST /v1/memories` with `infer=true`, so gnosis extracts durable facts
  server-side (non-blocking);
- exposes five model-facing tools — `gnosis_list`, `gnosis_search`, `gnosis_add`,
  `gnosis_update`, `gnosis_delete` — so the agent can manage memory itself.

`gnosis_update` / `gnosis_delete` need **`GNOSIS_MEMORY_EDIT_ENABLED=true` on the
server** (the bundled `compose.yaml` sets it; off by default otherwise — the
tools then report that editing is disabled rather than erroring). Failures degrade
to empty results, and a circuit breaker pauses calls for two minutes after five
consecutive errors, so a gnosis blip never takes the agent down.

Have a conversation, start a **new session**, and ask about something from the
first — recall spans sessions because it's keyed by `tenant_id` + `user_id`, not
by session.

## 5 — Or talk to gnosis directly

Any HTTP or MCP client can use the same surface. A first write (extraction mode —
turns become durable memory and, with the preferred config, extracted facts):

```bash
export GNOSIS_URL=http://localhost:8080 TOKEN=<your GNOSIS_TOKEN>

curl -s $GNOSIS_URL/v1/memories \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{
    "scope": {"tenant_id":"bromigos","space_id":"demo","agent_id":"assistant",
              "session_id":"sess-1","user_id":"alice","visibility":"private_user"},
    "messages": [
      {"role":"user","content":"I moved from Seattle to Austin last March."},
      {"role":"assistant","content":"Got it — you’re in Austin now."}
    ],
    "infer": true
  }'
# {"results": [{"memory_id":"…","content":"…","event":"ADD","metadata":{}}]}
```

Then read prompt-ready, scope-checked, redacted `sections[]` for a query with
`POST /v1/memory/context` (use the same `tenant_id` + `user_id`; session/agent
need not match):

```bash
curl -s $GNOSIS_URL/v1/memory/context \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{
    "scope": {"tenant_id":"bromigos","space_id":"demo","agent_id":"assistant",
              "session_id":"sess-2","user_id":"alice","visibility":"private_user"},
    "query": "Where does Alice live?", "max_items": 8
  }'
# {"sections": [{"source":"long_term_facts","content":"…Alice lives in Austin…","facts":[…]}]}
```

`POST /v1/memories/search` returns raw ranked hits instead of assembled context.
Full contract — list, edit/delete, promote, the filter DSL, the `/mcp` server:
[provider-surface.md](provider-surface.md).

> Extracted facts and the entity graph are a **write-path** capability: the payoff
> shows up on data ingested with the preferred config on, not retroactively. If a
> first read looks thin, confirm the write used `infer=true` and give background
> extraction a moment to flush.

## Choosing a config

gnosis auto-loads **`configs/default.yaml`** — the benchmark-tuned preferred
config — on start. To pick another, point `GNOSIS_CONFIG_FILE` at a
`configs/runs/runN.yaml` or your own file; set it to `""` for the minimal safe
defaults (all optional features off — a good choice on a small local model).
Precedence is **env vars → `.env` → the YAML file → code defaults**, so one env
var still overrides a single key. Catalogue: [`configs/`](../configs/README.md);
full setting reference: [configuration.md](configuration.md).

## Where to next

- [hermes-gnosis](https://github.com/bromigos-org/hermes-gnosis) — the plugin's
  own README: tool behavior, prefetch/turn-sync internals, and every setting.
- [CAPABILITIES.md](CAPABILITIES.md) — what each read/write technique does and why.
- [provider-surface.md](provider-surface.md) — the complete HTTP contract + MCP.
- [operations.md](operations.md) — running it for real: health, the extraction
  worker, backup, scale.
- [security.md](security.md) — token classes, scope enforcement, redaction,
  federation safety.

> Benchmarking rather than integrating? The
> [gnosis-membench](https://github.com/bromigos-org/gnosis-membench) harness
> ships a build-from-source variant of this stack that A/Bs feature flags against
> the official LOCOMO / LongMemEval judges.
