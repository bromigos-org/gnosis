# gnosis documentation

Start with the guide that matches your job.

- **To run gnosis and connect an agent,** read [getting-started.md](getting-started.md).
- **To learn what gnosis does and why,** read [CAPABILITIES.md](CAPABILITIES.md).
- **To deploy and operate it,** read [operations.md](operations.md), then
  [security.md](security.md).
- **To change the code,** read [development.md](development.md).

## Guides

| Doc | What it covers |
|---|---|
| [getting-started.md](getting-started.md) | Bring gnosis up and connect a hermes agent or any HTTP client |
| [operations.md](operations.md) | Requirements, health probes, the extraction worker, backup and scale |
| [development.md](development.md) | Setup, the four CI gates, tests and measuring a change |

## Reference

| Doc | What it covers |
|---|---|
| [configuration.md](configuration.md) | Every setting, its default and the preferred config |
| [provider-surface.md](provider-surface.md) | The `/v1/memories` contract, the filter DSL and the MCP server |
| [security.md](security.md) | Token classes, scope enforcement, redaction and federation |
| [architecture.md](architecture.md) | Layers, request flow and the module map |
| [data-model.md](data-model.md) | Neo4j labels, relationships, properties and the scope fields |
| [CAPABILITIES.md](CAPABILITIES.md) | Each technique, its flag, its research basis and its measured status |
| [BENCHMARKS.md](BENCHMARKS.md) | Every measured LOCOMO and LongMemEval run |

The config files live in [`../configs/`](../configs/README.md). There is one
preferred config, plus one file per measured run.
