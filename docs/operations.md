# Operations

Deploying and running gnosis. For every tunable, see
[configuration.md](configuration.md); for the auth model, [security.md](security.md).

## Requirements

- **Neo4j** (5.26+). gnosis holds the only graph credentials.
- **An OpenAI-compatible LLM + embedding endpoint** (LiteLLM, or ollama's `/v1`
  for local). Extraction and routing need a *capable* chat model — the default
  `gemma4` is fine for embeddings-adjacent use but not for extraction.

## Run

The image (`ghcr.io/nolgiainc/gnosis`) runs `uvicorn gnosis.main:app` on
`:8080` and **ships `configs/`**, so it auto-loads the preferred config
(`configs/default.yaml`) unless `GNOSIS_CONFIG_FILE` says otherwise. Minimum
environment:

```bash
GNOSIS_TOKEN=... GNOSIS_READ_OPERATOR_TOKEN=... GNOSIS_WRITE_OPERATOR_TOKEN=... \
GNOSIS_EXPORT_OPERATOR_TOKEN=... GNOSIS_ADMIN_OPERATOR_TOKEN=... \
NEO4J_URI=bolt://neo4j:7687 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=... \
LITELLM_BASE_URL=http://litellm:4000/v1 LITELLM_API_KEY=... \
GNOSIS_LLM=openai/<capable-model> \
GNOSIS_EMBEDDING=<embedding-model> GNOSIS_EMBEDDING_DIMENSIONS=<dim> \
  gnosis
```

A local stack (Neo4j + gnosis, wired to ollama or LiteLLM) is in the sibling
[`gnosis-membench`](https://github.com/nolgiainc/gnosis-membench) harness's
`stack/compose.yaml`.

## Configuration

Precedence: env vars → `.env` → the YAML config file → code defaults. Select a
config by pointing `GNOSIS_CONFIG_FILE` at `configs/default.yaml` (the default),
another `configs/runs/runN.yaml`, or your own file; set it to `""` for the
minimal safe defaults. Full reference: [configuration.md](configuration.md).

## Health & readiness

- `GET /health` — liveness.
- `GET /ready` — readiness (dependencies reachable). Use it as the k8s readiness
  probe so traffic only arrives once Neo4j and the LLM endpoint are up.

## Write path & the extraction worker

- `GNOSIS_WRITE_MODE=sync` returns after the write; `buffered` acks fast and
  flushes a bounded queue (`GNOSIS_MAX_PENDING`).
- Extraction with `GNOSIS_FACT_EXTRACTION_MODE=background` runs on a bounded async
  queue (`GNOSIS_FACT_EXTRACTION_MAX_CONCURRENCY` /`_MAX_PENDING`, drop-not-block)
  so the write path never blocks on the LLM; extracted facts appear shortly after
  the verbatim write. Malformed extractor JSON is re-sampled, not dropped.
- Enabling the entity graph or the read-path features that read it is a
  **write-path** change — the payoff appears on newly-ingested data, not
  retroactively.

## Operator workflows

Operator endpoints (in `routes/operator.py`) require the matching operator token
class ([security.md](security.md)). Maintenance is review-first:

- **Dedup / consolidation** return a candidate manifest under a read-operator
  token (`dry_run` default); applying requires a write-operator token.
- **Export** requires the export-operator token; output is redacted.
- Stats and entity/fact/preference/reasoning inspection are read-operator.

## Backup & durability

State lives in Neo4j (append-only) plus, optionally, media blobs in an
S3-compatible store (`GNOSIS_RUSTFS_*`). Back up the Neo4j volume; gnosis itself
is stateless (config + code) and safe to redeploy. `GNOSIS_CONVERSATION_TTL_DAYS`
can bound episodic message retention.

## Observability

- Structured logs for every optional read step that degrades (routing, rerank,
  filter, graph-QA, sufficiency) — a `WARNING` with the failure type, never a
  failed read.
- `GNOSIS_AUDIT_READ=true` writes a `MemoryReadAudit` node per read for audit.

## Notes on scale

- gnosis is horizontally stateless; scale replicas behind the gateway and point
  them at one Neo4j.
- The read-path LLM features (routing, rerank, sufficiency, graph-QA) each add an
  LLM call per read — trade latency/cost for quality per your route table.
- For multi-user single-store deployments, enable
  `GNOSIS_SCOPED_DENSE_RETRIEVAL_ENABLED` (a correctness requirement, not just an
  optimization — see [CAPABILITIES.md](CAPABILITIES.md)).
