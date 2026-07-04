# Official benchmark results

Mirror of the canonical log in [gnosis-membench/RESULTS.md](https://github.com/bromigos-org/gnosis-membench/blob/main/RESULTS.md); the harness repo is the source of truth for new runs.

Canonical record of all gnosis memory-quality benchmark runs. Every run uses the
frozen comparison config unless a deviation is noted. Raw artifacts
(`answers_*.jsonl`, `graded_*.jsonl`, `results.json`, `report.md`) live in the
gitignored `results/` tree on the machine that ran them; this file is the
durable summary.

**Frozen config**: LOCOMO subset 3 (conv-26, conv-30, conv-41), 1,451 turns
ingested, 497 questions; retrieval depth `max_items`/`limit` = 20; answering
and judging on GPT-5.5 via homelab LiteLLM at judge temperature = provider
default (gpt-5.5 hard-rejects the `temperature` param — deviation from the
official protocol, constant across all runs); LOCOMO adversarial rows scored
by the official substring rule; headline J excludes adversarial (matches the
mem0 paper's convention). gnosis embeddings: `local-qwen3-embedding-0.6b`
(1024-dim). Graph QA off (`MEMBENCH_INCLUDE_GRAPH=false`).

## Trajectory (headline: context condition, J excl. adversarial)

**37.4 → 41.0 → 59.5** in one day (2026-07-03), against a raw-search
reference of 61.3.

| Category (n) | Run 1 baseline: context | Run 1 baseline: search | Run 2 (PR #6): context | Run 3 (PR #7): context |
|---|---|---|---|---|
| single-hop (200) | 55.0 | 75.0 | 57.0 | **76.5** |
| multi-hop (74) | 10.8 | 44.6 | 14.9 | 40.5 |
| temporal (90) | 24.4 | 48.9 | 30.0 | 42.2 |
| open-domain (21) | 19.1 | 42.9 | 28.6 | 38.1 |
| adversarial (112) | 74.1 | 68.8 | 67.9 | 67.9 |
| **overall excl. adversarial (385)** | **37.4** | **61.3** | **41.0** | **59.5** |
| overall (497) | 45.7 | 63.0 | 47.1 | 61.4 |

Retrieval mechanism stats (context condition):

| Run | avg retrieved chars | % retrievals dated | % "no information" answers |
|---|---|---|---|
| Run 1 baseline | 947 | 0% | 36.0% |
| Run 2 (PR #6) | 10,018 | 100% | 31.4% |
| Run 3 (PR #7) | 5,218 | 100% | 21.9% |
| (search reference) | 4,231 | 100% | 22.3% |

## Run details

### Run 1 — `results/locomo/baseline-20260703/`

- gnosis: pre-fix main (before gnosis PR #6); `GNOSIS_LLM=openai/gemma4`;
  fresh Neo4j; write mode sync. Note: with extraction flags off (production
  default), "extraction-mode" add stores verbatim dated `said_user`/
  `said_assistant` facts and makes zero LLM calls at ingest — this measured
  gnosis as a dated-RAG store.
- Answering: `copilot-gpt-5.5`; judging: `gpt-5.5` via the chatgpt-route
  responses shim. Judge-route sensitivity check: both routes judged the
  context condition with **96.9% agreement** (J 39.0 vs 37.4) — route choice
  is not score-material.
- Full-condition scores (F1 / BLEU-1 / J, excl. adversarial):
  context 28.6 / 24.0 / 37.4 · search 43.8 / 37.8 / 61.3.
- Wall-clock: ~1h28m effective (ingest 3.2 min at ~7.4 turns/s, answering
  48 min, judging 36 min).
- Key finding: `/v1/memory/context` subtracted value vs gnosis's own raw
  search — session-pinned long-term reads (a launch-era production bug),
  zero dates in rendered facts, starved item budget.

### Run 2 — `results/locomo/context-fix-20260703/` (measures gnosis PR #6)

- gnosis main @ `24b7ac1` (cross-session reads + dates on every fact +
  honored item budget). Same ingested data reused (read-path-only change).
- Deviation: answering switched to `gpt-5.5` via the responses shim
  mid-experiment — the Copilot account hit a hard 402 monthly quota (all
  `copilot-*` models). Judge identical to Run 1.
- Context condition: **41.0 J excl. adversarial** (+3.6). Gains landed where
  the fix predicted (temporal +5.6, open-domain +9.5, multi-hop +4.1);
  adversarial dipped to search parity (richer retrieval → less abstention).

### Run 3 — `results/locomo/context-relevance-20260703/` (measures gnosis PR #7)

- gnosis main @ `480c707` (long-term facts relevance-ranked via the search
  candidate path when a query is present + compact `- [7 May 2023] ...`
  one-line rendering). Same ingested data reused.
- Answering + judging both `gpt-5.5` via the responses shim.
- Context condition: **59.5 J excl. adversarial** (+18.5 over Run 2), now
  within 1.8 of the search reference and **beating search on single-hop**
  (76.5 vs 75.0).

## Published comparison targets

LOCOMO overall J as published (gpt-4o-mini judge — different judge and
backbone, so directional only; cross-vendor numbers in this space are
actively disputed): OpenAI memory 52.9 · LangMem 58.1 · Zep 66.0 ·
mem0 66.9 · mem0-graph 68.4 · full-context 72.9 · Letta (blog) 74.0.

## Known limitations of the current record

- Subset 3 of 10 LOCOMO conversations; LongMemEval_S not yet run at scale.
- Answerer route changed between Run 1 and Runs 2/3 (Copilot quota) — the
  judge was held constant, but the context-vs-search comparison within Run 1
  is the cleanest same-route pair.
- gnosis ingests verbatim (no LLM extraction) in all runs to date; enabling
  extraction is the next measured experiment.
- Weekly regression runs (subset 2, this same frozen judge) execute in-cluster
  via the homelab `membench` CronJob and upload to RustFS `membench/results/`.
