# Development & contributing

## Setup

gnosis targets **Python 3.13** and uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # install runtime + dev dependencies from uv.lock
uv run pytest -q        # 500+ tests, ~4s
```

There is no Makefile — every task is a `uv run` command.

## The four CI gates — run all four before pushing

CI (`.github/workflows/ci.yml`) runs these in order, then builds the Docker
image. **All four must pass.** They are independent — `ruff check` passing does
**not** mean `ruff format --check` passes (lint rules vs. code formatting are
separate gates, and this has bitten PRs before):

```bash
uv run ruff check            # 1. lint
uv run ruff format --check   # 2. formatting  (SEPARATE from ruff check)
uv run basedpyright          # 3. types — must be 0 errors / 0 warnings
uv run pytest -q             # 4. tests
```

One-liner before every push:

```bash
uv run ruff check && uv run ruff format --check && uv run basedpyright && uv run pytest -q
```

To auto-fix the first two: `uv run ruff check --fix` and `uv run ruff format`.

## Tests

`uv run pytest -q` runs the full suite green with no extra environment. Running a
**single file in isolation** may fail on the required operator tokens (the suite
relies on them being present); supply them for isolated runs:

```bash
GNOSIS_READ_OPERATOR_TOKEN=x GNOSIS_EXPORT_OPERATOR_TOKEN=x \
GNOSIS_WRITE_OPERATOR_TOKEN=x GNOSIS_ADMIN_OPERATOR_TOKEN=x \
  uv run pytest tests/test_long_term_context.py -q
```

`tests/conftest.py` opts the suite out of the auto-loaded default config so tests
exercise features over the clean, code-default baseline; a test that needs a
config sets `GNOSIS_CONFIG_FILE` itself.

## Running gnosis locally

gnosis needs Neo4j and an OpenAI-compatible LLM + embedding endpoint. The sibling
[`gnosis-membench`](https://github.com/bromigos-org/gnosis-membench) harness ships
a `stack/compose.yaml` (Neo4j + gnosis, wired to ollama or LiteLLM) — the fastest
way to bring the whole thing up. Otherwise see [operations.md](operations.md).

## Adding a feature

gnosis's read/write quality features follow one pattern — the reason changes
compose cleanly and stay measurable:

1. **Add a flag, default-off, in `settings.py`** (`gnosis_x_enabled: bool = False`).
   Off must be **byte-identical** to the prior behavior.
2. **Wire it in gated**, in the write path (`fact_extraction`/`entity_graph`/…) or
   read path (`context_assembly`/`backend`/a new module like `reranker.py` or
   `sufficiency.py` — mirror an existing one). Read-path features that make an LLM
   call must **degrade gracefully** (fall back to the simpler path, log a warning,
   never block the read).
3. **Route it, don't stack it.** Features that only help some queries belong in the
   route table (`query_router` / `RouteDecision`), not enabled globally —
   enabling every good feature at once *regressed* the score (see the composition
   principle in [CAPABILITIES.md](CAPABILITIES.md)).
4. **Test it** — a pure unit test for any algebra, a gating test (off = no-op), and
   an end-to-end test through the read/write path. Keep basedpyright at 0 warnings.
5. **Measure it** before claiming anything (below), and document it in
   [CAPABILITIES.md](CAPABILITIES.md) + [configuration.md](configuration.md).
6. If it ships to the preferred config, add it to `configs/default.yaml`; either
   way it gets a `configs/runs/runN.yaml`.

## Measuring a change

Quality is measured, not asserted. Run the change through
[`gnosis-membench`](https://github.com/bromigos-org/gnosis-membench) on LOCOMO or
LongMemEval with the official judges, flag-gated and A/B'd against the frozen
gate, and record the result — **kept and rejected alike** — in
[BENCHMARKS.md](BENCHMARKS.md). Guardrails learned the hard way:

- **Measure at comparable n and judge/protocol** before any "beats X" claim (a
  subset headline that looks like a win can vanish at full scale).
- **Watch the noise floor** — small deltas on a saturated benchmark are noise.
- **Report negative results.** The record keeps rejected features (recall filter,
  bridge traversal, coverage budget, entity-grouped rendering) on purpose.

## Conventions

- **Formatting & types are enforced** — `ruff format` and `basedpyright` in strict
  mode (0 warnings). Fully type-annotated; no implicit `Any`.
- **Docstrings explain *why*.** Match the surrounding code's density and idiom;
  cite the paper or the measured finding behind a non-obvious choice.
- **Safe by default.** New features are flags, off by default, byte-identical when
  off. Nothing that costs money or latency (extraction, LLM read steps) is on by
  default.
- **Redaction and scope are non-negotiable.** Anything reaching a prompt is
  scope-re-checked and redacted; provenance and hidden reasoning never leak into
  recall.

## Pull requests

- Branch off `main`; keep the working tree on the main checkout.
- All four gates green locally first.
- **Docs are part of done** — update the README and the relevant `docs/` page in
  the same PR as the change.
- End commit messages with the project's `Co-Authored-By` trailer.

## Project layout

See [architecture.md](architecture.md) for the layer diagram and the full module
map (gateway → policy → orchestration → SDK/store).
