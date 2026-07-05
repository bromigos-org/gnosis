# gnosis documentation

| Doc | What it covers |
|---|---|
| [getting-started.md](getting-started.md) | Run gnosis (the bundled `compose.yaml`) and wire a real project to it — a hermes agent via [hermes-gnosis](https://github.com/bromigos-org/hermes-gnosis) — plus the direct HTTP path. **Start here to use gnosis.** |
| [CAPABILITIES.md](CAPABILITIES.md) | Every technique/algorithm — write path, read path, federation, policy — each with its approach, peer-reviewed basis, flag, and measured status. **Start here for "what does gnosis do and how."** |
| [architecture.md](architecture.md) | Layers, the read/write request flow, graph-QA planning, and the module map. |
| [data-model.md](data-model.md) | The Neo4j graph schema: node labels, relationships, properties, the scope spine, and the bi-temporal/append-only model. |
| [configuration.md](configuration.md) | Complete settings reference — every `GNOSIS_*` env var and YAML key, grouped, with defaults and the preferred config. |
| [security.md](security.md) | Trust boundary, the six token classes, scope enforcement, redaction, review-first ops, and federation safety. |
| [operations.md](operations.md) | Deploying and running gnosis — requirements, config, health probes, the extraction worker, operator workflows, backup, and scale. |
| [development.md](development.md) | Contributing: setup, the four CI gates, tests, running locally, the feature-flag pattern, measuring changes, and conventions. |
| [provider-surface.md](provider-surface.md) | The HTTP contract: `/v1/memories` add/search/list/promote/edit, the filter DSL, storage model, and MCP. |
| [BENCHMARKS.md](BENCHMARKS.md) | Every measured run, per-category history, configs, and honest deviations (LOCOMO + LongMemEval). |

Configs (the preferred config and one file per measured run) live in
[`../configs/`](../configs/README.md).
