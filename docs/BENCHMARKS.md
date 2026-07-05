# Official benchmark results

Mirror of the canonical log in [gnosis-membench/RESULTS.md](https://github.com/bromigos-org/gnosis-membench/blob/main/RESULTS.md); the harness repo is the source of truth for new runs.

> **Full-LOCOMO standing (Run 23, 2026-07-04).** The Run 1–22 trajectory
> below was measured on **subset 3** (3 of 10 conversations, 497 Q) — a fast
> dev-loop gate, not comparable-n to published systems. **Run 23** is the
> first full-10 measurement of the production (Run 18) config and the
> apples-to-apples competitor comparison:
>
> | Category (full n) | our F1 | J gpt-5.5 | J gpt-5.4-mini | best published |
> |---|---|---|---|---|
> | single-hop (841) | 60.9 | 77.0 | 77.8 | mem0 J 67.13 / F1 38.72 — **lead** |
> | multi-hop (282) | 34.3 | 41.5 | 49.6 | mem0 J 51.15 / F1 28.64 — **lead F1; J = judge generosity** |
> | temporal (321) | 32.5 | 73.8 | 73.8 | mem0^g J 58.13 — **lead J (F1 = date artifact)** |
> | open-domain (96) | 18.0 | 29.2 | 31.2 | Zep J 76.60 — **trail (weakness)** |
> | adversarial (446) | 83.9 | 83.9 | 83.9 | unpublished |
> | **excl-adv (1540)** | **47.5** | **66.9** | **68.9** | mem0 66.88 / mem0^g 68.44 / full-ctx 72.90 |
>
> Excl-adv J **66.9–68.9** is **at parity with mem0 (66.88)**, ties mem0^g
> (68.44), and sits **below the full-context ceiling (72.90)**. The prior
> "74.8, above the ceiling, best overall" headline was a subset-3 artifact
> (and 74.8 reproduces at ~71 on re-ingest); it does **not** hold at full n.
> Defensible full-n leads: **single-hop** (both axes), **temporal** and
> **adversarial** (judge-robust J), **multi-hop on the judge-independent F1**.
> Genuine weakness: **open-domain**. Full detail in the Run 23 section of
> RESULTS.md.


Canonical record of all gnosis memory-quality benchmark runs. Every run uses the
frozen comparison config unless a deviation is noted. Raw artifacts
(`answers_*.jsonl`, `graded_*.jsonl`, `results.json`, `report.md`) live in the
gitignored `results/` tree on the machine that ran them; this file is the
durable summary.

**Frozen config (LOCOMO — regression gate)**: LOCOMO subset 3 (conv-26,
conv-30, conv-41), 1,451 turns
ingested, 497 questions; retrieval depth `max_items`/`limit` = 20; answering
and judging on GPT-5.5 via homelab LiteLLM at judge temperature = provider
default (gpt-5.5 hard-rejects the `temperature` param — deviation from the
official protocol, constant across all runs); LOCOMO adversarial rows scored
by the official substring rule; headline J excludes adversarial (matches the
mem0 paper's convention). gnosis embeddings: `local-qwen3-embedding-0.6b`
(1024-dim). Graph QA off (`MEMBENCH_INCLUDE_GRAPH=false`).

**Embedder note (2026-07-04)**: the LongMemEval_S campaign (new primary
target) switches gnosis to cloud embeddings — `gemini-embedding-001`
(3072-dim, via the homelab LiteLLM; homelab PR #14 exposed the route).
Chosen over `text-embedding-3-small`/`-large` because it outranks both on
MTEB retrieval and the goal is highest scores, not published-system
comparability; verified live through gnosis's config path (3072-dim Fact
vectors in Neo4j) before any measured run. The LOCOMO gate above KEEPS
qwen3 embeddings — its entire 19-run history was measured there, and
changing the embedder would invalidate the comparison. Answering and
judging stay on the chatgpt-sub gpt-5.5 route; only `/v1/embeddings`
traffic hits the paid keys (cents at these volumes).

## Trajectory (headline: J excl. adversarial, LOCOMO subset 3)

One row per run, newest last. "Context" = assembled `/v1/memory/context`;
"search" = raw `/v1/memories/search`. A dash means that condition was not
re-run (read-path change measured on context only).

**Each run's flag config is a YAML file in the gnosis repo:
[`configs/runs/runN.yaml`](../configs/runs/), indexed in
[`configs/README.md`](../configs/README.md). gnosis auto-loads
[`configs/default.yaml`](../configs/default.yaml) (= Run 18, the preferred
config) out of the box; set `GNOSIS_CONFIG_FILE` to load a different run.** Runs
that share flags but differ by Chain-of-Note *code* (14/15/17/18) are noted in
each file.

| Run | Change under test | Context J | Search J | Verdict |
|---|---|---|---|---|
| 1 (baseline) | verbatim RAG, gemma4 | 37.4 | 61.3 | starting line |
| 2 (PR #6) | cross-session reads + dates + budget | 41.0 | — | kept (+3.6) |
| 3 (PR #7) | relevance ranking + compact rendering | 59.5 | — | kept (+18.5) |
| 4 (PR #13) | LLM recall filter | 58.7 | 59.0 | **rejected** — flat, +6s/read |
| 5 (PR #14) | **fact extraction at ingest** | **71.2** | 67.3 | **kept** (+11.7, temporal 42→84) |
| 6 (PR #15) | + hybrid BM25 retrieval | 71.4 | 69.1 | wash — temporal +8, multi-hop −5 |

Read-path experiments on the Run 5 extraction store (Runs 7–8 isolate one
flag alone — "does this feature help?", not "new best"; Run 9 stacks them):

| Run | Feature isolated | Context J | Search J | Finding |
|---|---|---|---|---|
| 7 (PR #19) | abstention prompt | 69.6 | 68.1 | **adversarial +8.9** but −1.6 excl-adv (over-abstains on answerable); overall-J +0.8. Tunable. |
| 8 (PR #20) | facts→verbatim expansion | 71.2 | 68.8 | flat headline; **multi-hop +2.7** (only thing to nudge multi-hop up), open-domain −4.8 |
| 9 | hybrid + verbatim + supersession STACKED | **57.9** | 29.1 | **CRASHED −13.3** — features do not compose; verbatim `said_*` turns displace dated extracted facts. Proves per-query routing is required. |

Write-path change (fresh ingest, new store):

| Run | Change under test | Context J | Search J | Verdict |
|---|---|---|---|---|
| 10 (PR #29) | entity graph + graph-QA fusion (+ extraction) | 70.9 | 67.5 | **multi-hop FLAT at 39.2** — the graph alone is inert; needs a decomposition driver (T1). Headline −0.3 = noise. |

Read-path changes on the Run 10 entity-graph store:

| Run | Change under test | Context J | Search J | Verdict |
|---|---|---|---|---|
| 11 (PR #30+#33) | adaptive per-query routing (T3) | **74.3** | 68.8 | **NEW BEST +2.9.** Single-hop 81.0 and multi-hop 44.6 both best-ever, temporal 92.2 peak kept — routing composes the per-category winners. Cost: adversarial −5.4 (router under-fires `unanswerable_risk`: 2/497). |
| 12 (PR #34+#35) | entity-anchored graph traversal (T1) alone | 70.7 | 67.3 | **rejected as a global flag** — multi-hop went *down* (39.2→36.5): neighborhood facts displace 5 ranked-dense slots on every query for no bridge gain; adversarial −3.6 (extra facts manufacture false support). |
| 13 (PR #31) | Chain-of-Note reading instruction (T2) alone | 72.0 | — | **kept** — **adversarial 79.5, best ever (+11.6)** AND excl-adv +1.1 (multi-hop +4.0): strictly dominates the Run 7 abstention prompt, which bought adversarial by *costing* answerable. |
| 14 | routing (T3) + Chain-of-Note (T2) combined | 71.4 | — | **does not compose** — worse than either alone on its own front (excl-adv 74.3→71.4, adv 79.5→75.9). Temporal 92.2→83.3: CoN's "state what the memory says" makes the reader parrot relative dates from hybrid's raw turns. |
| 15 (PR #37) | routing + **route-aware** CoN (skip temporal) | 73.2 | — | **composes** — temporal repaired 83.3→91.1, adversarial 78.6, **overall-incl-adv 74.5 = NEW BEST**. Excl-adv 73.2 sits 1.1 under Run 11's peak but buys +16.1 adversarial; best production config. |
| 16 (PR #39) | + directed bridge-entity traversal (T1-directed) | 72.5 | — | **rejected as measured — fires too rarely to matter.** The mechanism works (one textbook repair: "Which city have both Jean and John visited?" → Rome via the bridge fetch) but retrieval changed on only 35/497 questions; every category net-flips within noise (multi-hop 43.2→41.9, adversarial −1.8). LOCOMO's multi-hop misses are mostly cross-session enumerations, not bridge chains. |
| 17 (PR #40) | hardened CoN (attribution + never-guess clauses) | 72.2 | — | **adversarial 83.0 = BEST EVER (+4.4, 5 repairs / 0 regressions), overall 74.7 = new best**, multi-hop 44.6 ties best. Cost: the never-guess rule over-abstains on open-domain "would X likely..." inference questions (42.9→28.6, 3 abstention regressions on n=21). Carve-out measured next. |
| 18 (PR #41) | + likelihood carve-out in the never-guess rule | **74.8** | — | **NEW BEST on both headlines: excl-adv 74.8, overall 76.7.** The carve-out recovered open-domain 28.6→42.9 (3/3 abstention regressions repaired) AND single-hop 78.5→82.0 (best ever, 8 repairs / 1 regression) while adversarial held 83.0 with zero flips. Every category at or within noise of its historic peak — the production config. |
| 19 (PR #43+#44) | 2x coverage item budget on multi-hop/aggregative routes | 72.5 | — | **rejected — retrieval coverage improved, answers did not.** Gold-item presence on the 27-question enumeration cohort rose 50%→60% yet **0/27 repaired**: even fully-covered questions still answer with a subset. The residual failure is the reader/judge (exact multi-item list golds), not retrieval. Also quantified the noise floor: 20 flips on 437 byte-identical-retrieval questions (±2.3 J between identical configs). |

**Current best on the subset-3 dev gate: Run 18 — excl-adv 74.8 AND overall
76.7** (both subset-3 numbers; **superseded for competitor comparison by Run
23** — see the banner at top — and ~4 pts optimistic: reproducible level ~71).
**On subset 3, every category is at or within noise of its historic peak
simultaneously** (single-hop 82.0 peak, multi-hop 44.6 ties peak,
temporal 91.1, open-domain 42.9 ties peak, adversarial 83.0 peak).
The path there: Run 9 proved stacking the measured winners globally
*destroys* the score, Run 11 proved routing each query to its
category's measured-best feature set *composes* them, Run 12 killed
radial graph traversal, Run 13 measured Chain-of-Note as the first
strict win since extraction, Run 14 showed routing + CoN do NOT stack
blindly (CoN parrots hybrid's relative dates on temporal), Run 15
fixed that seam by making CoN route-aware (skip temporal; overall
74.5), Run 16 killed *directed* bridge traversal too (mechanism works,
fires on too few LOCOMO questions to matter — multi-hop misses are
enumerations, not bridge chains), Run 17 hardened the CoN instruction
against the two residual adversarial patterns (attribution +
never-guess; adversarial 83.0 best ever, but open-domain inference
questions over-abstained), and Run 18 carved likelihood questions out
of the never-guess rule — recovering open-domain AND unlocking
single-hop's 82.0 peak while adversarial held. The reading instruction
is now the highest-leverage seam in the system: three consecutive
prompt-only changes moved the totals more than any retrieval change
since extraction. Run 19 then closed the last known retrieval lever:
doubling the item budget on the enumeration-bearing routes raised
gold-item coverage 50%→60% and repaired **zero** of the 27 target
questions — multi-hop's residual gap is exact-list grading of
ambiguous enumerations, not retrieval. Run 19 also quantified the
benchmark's noise floor: 20 correctness flips across 437 questions
with byte-identical retrieval (±2.3 J excl-adv between effectively
identical configs), larger than any remaining candidate lever.
**LOCOMO subset 3 is measured out for this system** — see the
saturation note below.

### Saturation note (2026-07-04, after Run 19)

Three independent measurements say the benchmark, not the memory
system, is now the binding constraint: (1) the pure noise floor
between identical configs is ±2.3 J excl-adv, wider than every
post-Run-15 delta; (2) the largest remaining category gap (multi-hop
~44) is capped by exact-list grading — full gold coverage in context
does not flip answers; (3) temporal's 1.1 gap to peak is one
question. **[Superseded by Run 23 — see the banner at the top.]** Run 18's
subset-3 74.8 was originally read as above the full-context ceiling (72.9);
the full-LOCOMO re-measure shows it does **not** hold — full-n excl-adv J is
66.9–68.9 (two judges), at parity with mem0 and below the 72.9 ceiling (and
74.8 itself reproduces at ~71). This saturation note describes subset 3 as a
dev-loop gate, not the competitive standing.

**Status (2026-07-04): LOCOMO subset 3 is FROZEN as the regression
gate at the Run 18 config** (extraction + entity graph at write;
adaptive routing + route-aware hardened CoN v3 with the likelihood
carve-out at read; embeddings `local-qwen3-embedding-0.6b` /
1024-dim; judge gpt-5.5; `max_items` 20; context condition).
Reference scores for the gate: excl-adv **~71** (the reproducible level —
the recorded 74.8 was a ~4 pt favorable-extraction outlier; two independent
re-ingests land at 70.9 and 71.4), with a ±2.3 J noise band on excl-adv
between identical configs. **This gate is an internal dev-loop signal on 3
easy conversations, not a competitive claim — see the Run 23 full-LOCOMO
standing at the top.** Any future
gnosis change should re-run this gate and is a regression only if it
lands below the noise band, judged per-category. The gate keeps the
qwen3 embedder its whole history was measured with, so its recorded
scores stay comparable; re-running it requires a re-ingest at that
embedder (the store is not kept warm). The primary optimization
target is now LongMemEval_S (500 questions, 5 ability axes including
abstention and knowledge updates, ~115k-token haystacks), whose
frozen config switches to cloud embeddings (see the LongMemEval_S
section); its noise floor must be re-baselined there before
believing any new lever.

### Full per-category history — context condition (`/v1/memory/context`)

Every run, so each category's progression is visible (e.g. temporal
24→30→42→43→84→92, multi-hop's stubborn plateau). Runs 1–6 are
cumulative; Runs 7–9 are read-path experiments on the Run 5 store
(7–8 one flag alone, 9 stacked) so they compare against Run 5, not each
other; Run 10 is a fresh ingest (new store: extraction + entity graph)
with graph-QA fusion on at read time; Runs 11–15 are read-path
experiments on the Run 10 store (11: adaptive routing on, all global
read flags off; 12: graph traversal alone; 13: Chain-of-Note alone;
14: routing + Chain-of-Note together; 15: routing + route-aware
Chain-of-Note, skipped on the temporal route; 16: Run 15 config plus
directed bridge-entity traversal on the multi-hop route; 17: Run 15
config with the hardened Chain-of-Note instruction; 18: Run 17 plus
the likelihood carve-out clause; 19: Run 18 plus a 2x item budget on
the multi-hop and aggregative routes).

| Category (n) | Run 1 | Run 2 (#6) | Run 3 (#7) | Run 4 (#13) | Run 5 (#14) | Run 6 (#15) | Run 7 abst (#19) | Run 8 verb (#20) | Run 9 stacked | Run 10 graph (#29) | Run 11 routed (#30) | Run 12 traversal (#35) | Run 13 CoN (#31) | Run 14 routed+CoN | Run 15 route-aware (#37) | Run 16 bridge (#39) | Run 17 hardened (#40) | Run 18 likelihood (#41) | Run 19 coverage (#43+#44) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single-hop (200) | 55.0 | 57.0 | 76.5 | 74.5 | 80.5 | 79.5 | 78.5 | 80.0 | 75.5 | 79.0 | 81.0 | 79.5 | 79.5 | 80.0 | 79.5 | 79.5 | 78.5 | **82.0** | 78.5 |
| multi-hop (74) | 10.8 | 14.9 | 40.5 | 40.5 | 39.2 | 33.8 | 37.8 | 41.9 | 28.4 | 39.2 | **44.6** | 36.5 | 43.2 | 43.2 | 43.2 | 41.9 | **44.6** | **44.6** | 43.2 |
| temporal (90) | 24.4 | 30.0 | 42.2 | 43.3 | 84.4 | **92.2** | 85.6 | 84.4 | 47.8 | 85.6 | **92.2** | 85.6 | 85.6 | 83.3 | 91.1 | 90.0 | 91.1 | 91.1 | 90.0 |
| open-domain (21) | 19.1 | 28.6 | 38.1 | 38.1 | 38.1 | 38.1 | 28.6 | 33.3 | 38.1 | **42.9** | 38.1 | **42.9** | **42.9** | 38.1 | **42.9** | 38.1 | 28.6 | **42.9** | **42.9** |
| adversarial (112) | 74.1 | 67.9 | 67.9 | 67.9 | 67.9 | 71.4 | 76.8 | 68.8 | 64.3 | 67.9 | 62.5 | 64.3 | 79.5 | 75.9 | 78.6 | 76.8 | **83.0** | **83.0** | **83.0** |
| **overall excl. adv. (385)** | **37.4** | **41.0** | **59.5** | **58.7** | **71.2** | **71.4** | **69.6** | **71.2** | **57.9** | **70.9** | **74.3** | **70.7** | **72.0** | **71.4** | **73.2** | **72.5** | **72.2** | **74.8** | **72.5** |
| overall (497) | 45.7 | 47.1 | 61.4 | 60.8 | 70.4 | 71.4 | 71.2 | 70.7 | 59.4 | 70.2 | 71.6 | 69.2 | 73.6 | 72.4 | 74.5 | 73.4 | 74.7 | **76.7** | 74.9 |

### Full per-category history — search condition (`/v1/memories/search`)

Runs where search was re-run (read-path changes measured on context only
are omitted).

| Category (n) | Run 1 | Run 4 (#13) | Run 5 (#14) | Run 6 (#15) | Run 7 abst (#19) | Run 8 verb (#20) | Run 9 stacked | Run 10 graph (#29) | Run 11 routed (#30) | Run 12 traversal (#35) |
|---|---|---|---|---|---|---|---|---|---|---|
| single-hop (200) | 75.0 | 73.5 | 73.5 | **75.5** | 74.0 | 75.0 | 40.5 | 74.0 | 74.5 | 73.5 |
| multi-hop (74) | **44.6** | 46.0 | 37.8 | 32.4 | 39.2 | 39.2 | 4.0 | 35.1 | 35.1 | 33.8 |
| temporal (90) | 48.9 | 43.3 | 84.4 | **92.2** | 86.7 | 85.6 | 27.8 | 86.7 | 90.0 | 87.8 |
| open-domain (21) | 42.9 | 33.3 | 38.1 | 38.1 | 33.3 | **42.9** | 14.3 | 38.1 | **42.9** | 38.1 |
| adversarial (112) | 68.8 | 72.3 | 71.4 | 73.2 | 70.5 | 72.3 | **81.2** | 75.0 | 74.1 | 76.8 |
| **overall excl. adv. (385)** | **61.3** | **59.0** | **67.3** | **69.1** | **68.1** | **68.8** | **29.1** | **67.5** | **68.8** | **67.3** |
| overall (497) | 63.0 | 62.0 | 68.2 | 70.0 | 68.6 | 69.6 | 40.8 | 69.2 | 70.0 | 69.4 |

Retrieval mechanism stats (context condition unless noted):

| Run | avg retrieved chars | % "no information" answers |
|---|---|---|
| Run 1 baseline | 947 | 36.0% |
| Run 3 (PR #7) | 5,218 | 21.9% |
| Run 4 (PR #13) | 2,093 | 20.5% |
| Run 5 (PR #14) context | 4,349 | 20.5% |
| Run 5 (PR #14) search | 2,175 | 22.7% |
| Run 9 stacked context | 4,833 | 22.7% |
| Run 9 stacked search | 503 | 55.5% |
| Run 10 (PR #29) context | 4,367 | 20.5% |
| Run 10 (PR #29) search | 2,189 | 23.5% |
| Run 11 (PR #30) context | 4,392 | 19.3% |
| Run 11 (PR #30) search | 2,210 | 23.1% |
| Run 12 (PR #35) context | 4,345 | 19.3% |
| Run 13 (PR #31) context | 4,726 | 23.7% |
| Run 14 routed+CoN context | 4,753 | 22.5% |
| Run 15 route-aware CoN context | 4,670 | 23.1% |
| Run 16 bridge traversal context | 4,681 | 23.1% |
| Run 17 hardened CoN context | 4,809 | 25.6% |
| Run 18 likelihood carve-out context | 4,932 | 24.7% |
| Run 19 coverage budget context | 5,191 | 24.1% |

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

### Run 4 — `results/locomo/recall-filter-20260703/` (measures gnosis PR #13)

- gnosis main @ `d490b83` (EMem-style LLM recall filter behind
  `GNOSIS_RECALL_FILTER_ENABLED`, candidates cap 30). Same ingested data
  reused (read-path-only change). Deviations from the frozen config:
  `GNOSIS_RECALL_FILTER_ENABLED=true` (the feature under test) and gnosis-side
  `GNOSIS_LLM=openai/gpt-5.5` via the homelab LiteLLM (matches production; the
  filter needs a real model — reads make no other `GNOSIS_LLM` calls, so this
  only powers the filter). A logging-only compose overlay
  (`stack/compose.recall-logging.yaml`) surfaced the filter's structured log
  extras; app behavior unchanged.
- Answering + judging both `gpt-5.5` via the responses shim. **Both**
  conditions rerun — the filter sits in `/v1/memory/context` and
  `/v1/memories/search`.
- Scores: context **58.7** J excl. adversarial (−0.8 vs Run 3, ≈3 questions —
  noise-level), search **59.0** (−2.3 vs the Run 1 search reference; note the
  answerer route differs from Run 1, copilot vs shim). The hoped-for
  multi-hop/temporal gains did not materialize on context (multi-hop flat at
  40.5, temporal +1.1); search moved multi-hop +1.4 and adversarial +3.5 but
  gave back temporal −5.6 and open-domain −9.5 (n=21).
- Filter mechanism (from gnosis logs over the run window): avg 29.5
  candidates in → **4.7 kept** (~84% pruned) in both conditions;
  fallback-to-unfiltered rate 0.5% (5 of ~995 calls); filter LLM latency
  mean 7.2 s / p50 5.7 s / p95 12.1 s added to every read. Retrieved payload
  dropped 60% (context, 5,218 → 2,093 chars) and 73% (search,
  4,231 → 1,122 chars). The filter's empty-selection fallback returns the
  *unfiltered* list, so adversarial questions (nothing relevant) mostly still
  see full context.
- Wall-clock: ~38 min for both conditions end-to-end (context answer+grade
  17m46s; search ~20m including one httpx-timeout crash at 496/497 answers
  and a resumable-driver restart).
- Verdict: **keep default-off in production.** Accuracy is
  flat-to-slightly-negative on LOCOMO (EMem's ablation gains did not
  reproduce here), while every read gains one gpt-5.5 call and ~6-7 s median
  latency. The 60-73% retrieval-payload cut is real and would matter under
  tight context budgets or expensive answer models; reconsider with a
  fast/cheap filter model.

### Run 5 — `results/locomo/extraction-20260703/` (measures gnosis PR #14) — CURRENT BEST

- gnosis main @ `a4a9254` (edu-v1 LLM fact extraction at ingest behind
  `GNOSIS_FACT_EXTRACTION_ENABLED`). Recall filter OFF (Run 4 showed it flat),
  so this isolates extraction. Deviations from frozen config:
  `GNOSIS_FACT_EXTRACTION_ENABLED=true`, gnosis-side `GNOSIS_LLM=openai/gpt-5.5`
  (the extractor needs a real model). **Fresh ingest required** (write-path
  change): neo4j wiped, LOCOMO subset 3 re-ingested as turn-pair adds so
  extraction fires per pair.
- Answering + judging both `gpt-5.5` via the responses shim. Both conditions run.
- **Scores: context 71.2 J excl. adversarial (+11.7 over Run 3), search 67.3
  (+6.0 over the Run 1 search reference).** The win is almost entirely
  temporal: context temporal 42.2 → **84.4 (+42.2)**, search 48.9 → 84.4
  (+35.5) — dated, self-contained fact units make "when" questions answerable.
  single-hop also up on context (76.5 → 80.5). multi-hop did NOT improve
  (context 40.5 → 39.2, search 44.6 → 37.8) — extraction makes facts
  answerable but does not connect them across hops; that is the next target
  (hybrid retrieval PR #15 + graph-QA fusion).
- Extraction mechanism: 3,365 extracted `fact`-predicate units created from
  the turn-pairs (~3.3 per pair), stored alongside 1,037 verbatim `said_*`
  facts (non-compressive). Context retrieved 4,349 avg chars / 20.5% no-info;
  search 2,175 chars / 22.7% no-info.
- Cost: one gpt-5.5 extraction call per turn-pair at ingest (inline in PR #14
  — ~2-5 s each; a background-mode worker is required before production
  enablement so Discord/hermes writes are not blocked in the hot path).
- Ops note: the run survived a Docker Desktop crash mid-search-answering; the
  resumable driver reprocessed only the remaining answers after the stack (and
  its LiteLLM key env) was restored. No data lost.
- **Verdict: extraction is the biggest single lever measured. Ship it to
  production** (behind a background-extraction worker for latency). gnosis
  context 71.2 now exceeds published mem0 (66.9), mem0-graph (68.4), Zep
  (66.0) and sits 1.7 under the full-context ceiling (72.9) — while sending
  ~4.3k chars, not the whole conversation.

### Run 6 — `results/locomo/hybrid-extraction-20260703/` (measures gnosis PR #15)

- Same extracted store as Run 5 (read-path-only), `GNOSIS_HYBRID_RETRIEVAL_ENABLED=true`
  added (BM25 full-text fused with dense via RRF). gnosis-side gpt-5.5.
- **Scores: context 71.4 (+0.2 vs Run 5), search 69.1 (+1.8).** A genuine
  tradeoff, consistent across both conditions: temporal **+7.8** (84.4→92.2,
  BM25 nails exact dates/names) and adversarial +1.8–3.5 (lexical mismatch →
  nothing retrieved → correct abstention), but **multi-hop −5.4** (39.2→33.8
  context, 37.8→32.4 search) — lexical matching surfaces similar-but-wrong
  facts that displace the intermediate-fact chain multi-hop needs.
- Verdict: net-neutral on context, mildly positive on search. NOT the
  multi-hop fix — multi-hop needs graph traversal, not lexical matching
  (→ graph-QA fusion, gnosis PR #21). Hybrid's temporal/abstention gains are
  real; best used alongside a multi-hop route rather than alone. Leave
  default-off pending a combined graph-QA + hybrid run.

### Run 9 — `results/locomo/combined-quality-20260704/` (stacked read-path flags)

- Same Run 5 extraction store (read-path-only). Flags stacked together:
  `GNOSIS_HYBRID_RETRIEVAL_ENABLED` + `GNOSIS_FACT_VERBATIM_EXPANSION_ENABLED`
  + `GNOSIS_READ_SUPERSESSION_ENABLED` (abstention prompt off — 0/497
  retrievals carry an `[instructions]` section; 431/497 carry verbatim
  `quote:` lines). gnosis-side gpt-5.5. Both conditions rerun.
- **Scores: context 57.9 J excl. adversarial (−13.3 vs Run 5), search 29.1
  (−38.2).** Temporal 84.4→47.8 context / 84.4→27.8 search; multi-hop
  39.2→28.4 / 37.8→4.0; single-hop 80.5→75.5 / 73.5→40.5. Search
  adversarial "improved" to 81.2 only because retrieval collapsed —
  55.5% of answers were "no information" (vs 22.7% in Run 5) and the
  search payload shrank 2,175→503 chars.
- Failure mechanism (verified per-question): the stacked flags push
  verbatim `said_*` turn facts above the dated extracted `fact` units in
  the ranked candidates, so the answer model reads raw turns with relative
  dates ("I went to a support group *yesterday*") instead of the resolved
  dated fact, and answers "yesterday" where Run 5 answered "2023-05-07".
  85 context questions flipped correct→wrong vs Run 5 (38 temporal,
  20 single-hop, 15 multi-hop, 12 adversarial). Hybrid's BM25 leg matches
  the verbose raw turns lexically, RRF boosts them, supersession and the
  item budget then cut the extracted facts that carried the resolved dates.
- **Verdict: features measured as individual wins do NOT compose.** Each
  global flag applies its failure mode to every category (hybrid's
  similar-but-wrong matches, verbatim's raw-turn duplication). This run
  is the direct motivation for per-query adaptive routing (gnosis PR #30):
  apply each feature only to the query class where it measured as a win.
  The verbatim-over-extracted ranking inversion is additionally a
  standalone bug (composability fix, tracked).

### Run 10 — `results/locomo/entity-graph-20260704/` (measures gnosis PR #29)

- gnosis main @ `072e893` (entity graph: edu-v1 extractor emits
  `(head, relation, tail)` triples; each extracted fact MERGEs scope-keyed
  `(:Entity)` nodes, `MENTIONS` edges, and dated `RELATES` edges).
  **Fresh ingest required** (write-path change): neo4j wiped, subset 3
  re-ingested with `GNOSIS_FACT_EXTRACTION_ENABLED` +
  `GNOSIS_ENTITY_GRAPH_ENABLED` + `GNOSIS_GRAPHQA_FUSION_ENABLED`,
  gnosis-side gpt-5.5. Both conditions run.
- Store shape after ingest: 4,330 Fact nodes, **139 Entity nodes, 667
  RELATES edges, 4,899 MENTIONS edges** — the graph materialized and
  relative dates resolved onto RELATES `event_date` (spot-checked).
- **Scores: context 70.9 J excl. adversarial (−0.3 vs Run 5 = noise),
  search 67.5 (+0.2).** The target metric — **multi-hop — is exactly flat
  at 39.2** (Run 5: 39.2); search multi-hop 35.1 (−2.7). temporal 85.6 /
  86.7, single-hop 79.0 / 74.0, open-domain 42.9 (+4.8, n=21 noise-band),
  adversarial 67.9 / 75.0.
- Fusion mechanism: the graph-QA leg ran on all 497 context queries; 55
  degraded on planner timeout/failure (~11%, logged, dense-only fallback).
  The planned Cypher route returned nodes on some queries but the fused
  candidates did not convert into multi-hop answers.
- **Verdict: a materialized entity graph with only an LLM-planned Cypher
  query driving it does NOT move multi-hop** — consistent with the
  docs/multihop-techniques.md prior that "a graph with nothing driving the
  second hop is inert". The graph itself is healthy infrastructure (and
  open-domain's small bump suggests the fused nodes add some breadth); the
  missing piece is sequential query decomposition (T1): resolve the bridge
  entity first, then traverse `RELATES` from that pinned node. Keep
  `GNOSIS_ENTITY_GRAPH_ENABLED` default-off in prod until T1 lands.
- Ops notes: two ingest restarts were needed (an early launch died to
  process-group cleanup killing the nohup'd run; later one 300s-timeout
  crash on a conv-41 extraction call — raised `MEMBENCH_TIMEOUT` to 600s
  and resumed; partial conv-30/conv-41 data wiped by scope before each
  re-ingest so no duplicates). Ingest wall-clock ~15 min effective at
  concurrency 4–8; answer+grade ~55 min for both conditions.

### Run 11 — `results/locomo/adaptive-routing-20260704/` (measures gnosis PR #30 + #33)

- Same Run 10 store (read-path-only; extraction + entity graph in the
  data). gnosis main @ PR #33: `GNOSIS_ADAPTIVE_ROUTING_ENABLED=true`,
  every global read-path flag off — the router's decision is the only
  source of per-query features. One gpt-5.5 classification call per
  query tags it `temporal` / `multi_hop` / `single_hop` /
  `unanswerable_risk` / `aggregative`; the route applies that category's
  measured-best set (temporal→hybrid BM25; multi-hop→graph-QA fusion +
  verbatim expansion, no hybrid; unanswerable-risk→abstention prompt;
  single-hop/aggregative→plain dense). Both conditions rerun.
- **Scores: context 74.3 J excl. adversarial — NEW BEST (+2.9 vs Run 6's
  71.4; +3.4 vs Run 10 on the same store). Search 68.8 (ties Run 8's
  best-since-extraction).** Per category (context): single-hop **81.0**
  (best ever), multi-hop **44.6** (best ever — first move off the ~40
  plateau), temporal **92.2** (ties the Run 6 hybrid peak), open-domain
  38.1, adversarial 62.5 (−5.4 vs Run 10). Overall including adversarial
  71.6 — also a best.
- Mechanism verified per-question vs Run 10: 19 flips wrong→correct
  (8 temporal, 6 single-hop, 5 multi-hop) against 12 correct→wrong
  (6 of them adversarial). Temporal's +6.6 is hybrid quarantined to
  temporal queries; multi-hop's +5.4 came from verbatim expansion
  alone — the graph-QA fusion leg the route also enables was silently
  dead (its candidates were always cut by the item budget, the PR #34
  bug, found *after* this run) — so routed multi-hop still has headroom.
- The adversarial give-back is a router blind spot, not an abstention
  failure: only 2 of 497 context payloads carried the abstention
  instruction (both open-domain questions), i.e. the classifier almost
  never emits `unanswerable_risk` on LOCOMO's presupposition-style
  adversarial questions ("What kind of counseling workshop did Melanie
  attend?" → routed single-hop → answered "LGBTQ+ counseling workshop"
  from adjacent facts instead of refusing). 70/112 adversarial context
  answers still abstained (down from 76 in Run 10). Fixes to try: an
  explicit presupposition-check line in the router guide, or handing
  abstention to Chain-of-Note (T2) globally instead of routing it.
- Ops notes: the first smoke test of routing 500'd every read — the
  LiteLLM gpt-5.5 route answers a single-enum JSON schema with the bare
  enum value, which the SDK's strict parser rejects, and `ValidationError`
  escaped the router's failure fallback. Fixed in gnosis PR #33 (lenient
  parse + fallback catches ValidationError) before the run. Answer+grade
  ~19 min for both conditions at concurrency 8 (routing adds one cheap
  LLM call per query; no visible latency impact at this concurrency).

### Run 12 — `results/locomo/graph-traversal-20260704/` (measures gnosis PR #34 + #35)

- Same Run 10 store (read-path-only). Two merged changes under test:
  the **PR #34 budget fix** (graph-derived candidates get up to a
  quarter of `max_items` reserved instead of always being cut — found
  by code inspection after Run 11: `_fuse_graph_facts` appended graph
  candidates after the ~100-deep dense ranking, so the `max_items=20`
  cut silently dropped every one; the graph-QA fusion leg had **never
  rendered a single node in any prior run**), and **PR #35 entity
  traversal (T1)**: `GNOSIS_GRAPH_TRAVERSAL_ENABLED=true` alone,
  routing/fusion/hybrid off. Traversal pins the query's entity mentions
  as `:Entity` seeds (deterministic n-gram normalization, zero extra
  LLM calls), expands 1–2 `RELATES` hops, and fuses the edge-provenance
  facts into the reserved slots. Both conditions rerun.
- **Scores: context 70.7 (−0.2 vs Run 10 = noise), search 67.3 (−0.2).
  The target metric went the WRONG way: multi-hop 39.2→36.5 context /
  35.1→33.8 search.** temporal 85.6 flat, single-hop 79.5 (−0.5 noise),
  open-domain 42.9 flat-at-its-best, adversarial 64.3 (−3.6).
- Failure mechanism (13 correct→wrong vs Run 10, spread across all
  categories, vs 8 repairs): with the flag global, *every* query pins
  seeds (LOCOMO questions almost always name a speaker), so every
  context gives up to 5 ranked-dense slots to RELATES-neighborhood
  facts. Those neighbors are topically adjacent but rarely the bridge
  evidence — dense retrieval already surfaces the seed's own facts; the
  expansion adds the seed's *other* facts, displacing the ranked tail
  where multi-hop chain links actually lived. Adversarial −3.6 is the
  familiar false-support pattern: more adjacent facts, more material to
  answer presupposed questions from.
- **Verdict: rejected as a standalone/global flag.** Blind 2-hop
  neighborhood expansion is not decomposition — the graph needs
  *directed* hop-2 (resolve the bridge entity first, then expand only
  through it), not radial flooding. The flag stays merged (default-off)
  and routed-multi-hop-only wiring exists (`RouteDecision`), but it is
  NOT added to the route table without a measured multi-hop win. The
  PR #34 budget fix is kept — it is a correctness fix and its effect
  here is bounded by what fills the reserved slots.
- Ops: answer+grade ~15.5 min, both conditions, concurrency 8.

### Run 13 — `results/locomo/chain-of-note-20260704/` (measures gnosis PR #31)

- Same Run 10 store (read-path-only, prompt-only change):
  `GNOSIS_CHAIN_OF_NOTE_ENABLED=true` alone — a standing read-then-reason
  instruction prepended as a leading `instructions` section (note each
  memory's relevance and contradictions, ignore similar-but-irrelevant
  memories, answer only from relevant ones or say you don't know; arXiv
  2311.09210, LongMemEval 2410.10813). Context condition only — the
  instruction never appears in `/v1/memories/search` payloads.
- **Scores: context 72.0 J excl. adversarial (+1.1 vs Run 10) with
  adversarial 79.5 — best ever, +11.6 vs Run 10 and +2.7 over the Run 7
  abstention prompt's previous record.** Overall including adversarial
  **73.6, the best overall of any run.** multi-hop 43.2 (+4.0),
  open-domain 42.9, single-hop 79.5, temporal 85.6.
- Mechanism (25 wrong→correct vs Run 10 against only 8 losses):
  14 adversarial repairs — 89/112 adversarial answers abstained (vs
  76 in Run 10) — *plus* 3 multi-hop and net gains on temporal and
  single-hop. The note step rejects retrieved similar-but-wrong facts
  instead of answering from them, which is exactly the dual profile the
  papers promised (higher rejection AND higher answer quality under
  noisy retrieval). Contrast Run 7's bare abstention prompt: adversarial
  +8.9 but −1.6 on answerable. Chain-of-Note strictly dominates it.
- **Verdict: kept; the first candidate for a production default-on among
  the read-path prompts.** Also the natural partner for routing: Run 11
  won retrieval (74.3 excl-adv) but gave back adversarial (62.5);
  Chain-of-Note repairs exactly that seam from the reading side. The
  combined config (routing + CoN) is measured next as Run 14.

### Run 14 — `results/locomo/routing-chain-of-note-20260704/` (T3 + T2 combined)

- Same Run 10 store: `GNOSIS_ADAPTIVE_ROUTING_ENABLED=true` +
  `GNOSIS_CHAIN_OF_NOTE_ENABLED=true` (CoN takes precedence over the
  routed abstention instruction by design — it subsumes the grounding
  line). Context only. Hypothesis: routing's retrieval wins (74.3) plus
  CoN's adversarial repair (79.5) compose, since they occupy different
  seams (retrieval vs reading).
- **Scores: context 71.4 excl-adv, adversarial 75.9 — worse than either
  parent on its own front** (Run 11: 74.3 excl-adv; Run 13: 79.5 adv).
  temporal **92.2→83.3** is the damage. multi-hop 43.2, single-hop 80.0,
  open-domain 38.1.
- Failure mechanism (verified per-question vs Run 11: 10 temporal flips
  correct→wrong): on temporal-routed queries, hybrid BM25 surfaces the
  raw verbatim turns with relative dates alongside the resolved
  extracted facts, and CoN's "state what each memory says" step makes
  the reader faithfully report the raw turn's phrasing — Run 11 answered
  "2023-05-20", Run 14 answers "last Saturday". The abstention side
  works as intended (adversarial +13.4 over Run 11, 16 repairs), but the
  reading instruction interacts with hybrid's retrieval texture.
- **Verdict: do not stack blindly — the composability lesson repeats at
  the prompt level.** The obvious fix is route-aware reading: apply CoN
  on every route *except* temporal (whose hybrid+dated-facts pipeline
  needs no note step), i.e. make the reading instruction part of the
  route table rather than a global flag. Candidate next change, needs
  its own run.

### Run 15 — `results/locomo/routing-con-route-aware-20260704/` (route-aware CoN, gnosis PR #37)

- Same Run 10 store and the same two flags as Run 14
  (`GNOSIS_ADAPTIVE_ROUTING_ENABLED=true` +
  `GNOSIS_CHAIN_OF_NOTE_ENABLED=true`), but Chain-of-Note moved into
  the route table: the reading instruction is applied on every route
  *except* temporal. Context only. Hypothesis: Run 14's only damage was
  temporal (CoN parroting hybrid's relative-dated raw turns), so
  removing the note step from exactly that route recovers the peak
  while keeping CoN's adversarial repair everywhere else.
- **Scores: context 73.2 excl-adv, adversarial 78.6, overall-incl-adv
  74.5 — NEW BEST overall.** temporal **83.3→91.1** (repaired),
  multi-hop 43.2, single-hop 79.5, open-domain 42.9 (ties best).
- Per-question verification: vs Run 14, 8 temporal repairs / 1
  regression (exactly the parroting cohort). vs Run 11 (routing alone),
  adversarial has **18 repairs / 0 regressions** (+16.1) while temporal
  differs by only 3 up / 4 down and single-hop 4 up / 7 down — both
  within per-category noise. vs Run 13 (CoN alone), adversarial is a
  wash (1 up / 2 down) — route-aware CoN loses none of the reading
  lever's abstention power.
- **Verdict: the two winners now compose — this is the best production
  config measured** (routing + route-aware CoN). Run 11 nominally keeps
  the excl-adv headline (74.3 vs 73.2) but pays 16.1 adversarial for
  1.1 excl-adv of noise-level flips; Run 15 dominates on overall J
  (74.5 vs 71.6). The composability bug's general fix is confirmed:
  feature interactions live in the route table, where each is scoped to
  the categories it measurably helps.

### Run 16 — `results/locomo/routing-con-bridge-20260704/` (directed bridge traversal, gnosis PR #39)

- Run 15 config plus `GNOSIS_BRIDGE_TRAVERSAL_ENABLED=true` — the
  *directed* replacement for Run 12's rejected radial traversal
  (self-ask, arXiv 2210.03350): after dense retrieval, one LLM call
  reads the query plus hop-1's dense facts and names up to three bridge
  entities the facts reveal but the question never names; a fixed
  Cypher fetches the facts `MENTIONS`-linked to them, fused into the
  reserved graph budget slots. Routed to multi-hop-classified queries
  only. Context only.
- **Scores: context 72.5 excl-adv, adversarial 76.8, overall 73.4 —
  all slightly below Run 15** (73.2 / 78.6 / 74.5). multi-hop
  43.2→41.9, temporal 91.1→90.0, open-domain 42.9→38.1.
- Per-question verification: retrieval changed on only **35/497**
  questions — the router + namer gate fires rarely, and when it fires
  it usually adds nothing. Exactly one textbook bridge repair ("Which
  city have both Jean and John visited?" → Rome, previously abstained;
  the bridge fetch surfaced Jean's Rome fact). Every category's flips
  are 1-3 questions, i.e. noise; the -0.7 excl-adv delta is judge
  variance on unchanged retrievals (e.g. "transgender" vs "Transgender
  woman" re-judged wrong).
- **Verdict: rejected as measured — the mechanism works but the
  addressable population is too small.** Run 15's multi-hop misses are
  dominated by cross-session *enumerations* ("What activities has
  Melanie done with her family?" — needs 6 facts across sessions), not
  bridge chains; a directed hop cannot fix incomplete enumeration. The
  multi-hop lever is retrieval *coverage* (e.g. a larger item budget or
  per-entity fan-out on the multi-hop route), not deeper traversal.
  Flag stays merged but off.

### Run 17 — `results/locomo/routing-con-hardened-20260704/` (hardened Chain-of-Note, gnosis PR #40)

- Run 15 config (routing + route-aware CoN, no bridge flag) with the
  Chain-of-Note instruction hardened against the two residual
  adversarial patterns found by per-question analysis of Run 15's 24
  misses: answering from a *different person's* similar fact (LOCOMO
  adversarial swaps the speaker — "What did Jon want his customers to
  feel in her store?" answered from Gina's store facts) and answering
  yes/no about things the memories never mention ("Is Oscar Melanie's
  pet?"). The note step now asks who each memory is about, discards
  other-person memories explicitly, and never guesses or answers
  yes/no about unmentioned things. Note: the router lever the Run 11
  analysis suggested (under-fired `unanswerable_risk`, 2/497) is a
  no-op under this config — that route differs from single-hop only by
  the abstention line, which CoN subsumes — so the fix belongs in the
  reading instruction itself. Context only.
- **Scores: adversarial 83.0 — BEST EVER (+4.4); overall-incl-adv
  74.7 = new best.** multi-hop 44.6 (ties Run 11's best), temporal
  91.1 held, single-hop 78.5, open-domain 28.6 (the cost), excl-adv
  72.2.
- Per-question verification vs Run 15: adversarial **5 repairs / 0
  regressions** — both targeted patterns fixed. But the bare
  never-guess rule over-fires on open-domain *inference* questions
  ("Would Caroline likely have Dr. Seuss books on her bookshelf?",
  "What might John's degree be in?") — 3 regressions, all abstentions
  (open-domain 42.9→28.6 on n=21). Abstain rate 23.1%→25.6%.
- **Verdict: keep the attribution + never-guess hardening, carve out
  likelihood questions.** The Run 7 lesson repeats in miniature: any
  abstention pressure needs an escape hatch for questions that *ask*
  for an inference. PR #41 adds exactly that clause ("only when the
  question itself asks what is likely or probable, infer the most
  plausible answer"), measured next.

### Run 18 — `results/locomo/routing-con-likelihood-20260704/` (likelihood carve-out, gnosis PR #41)

- Run 17 config with one added Chain-of-Note clause: only when the
  question itself asks what is likely or probable, infer the most
  plausible answer from the relevant memories instead of abstaining.
  Context only.
- **Scores: excl-adv 74.8 AND overall 76.7 — NEW BEST on both
  headlines.** single-hop **82.0 (best ever)**, multi-hop 44.6 (ties
  best), temporal 91.1, open-domain 42.9 (ties best), adversarial
  **83.0 (ties best, zero flips vs Run 17)**.
- Per-question verification vs Run 17: open-domain 3 repairs / 0
  regressions (exactly the three "would X likely..." abstentions the
  carve-out targeted), single-hop 8 repairs / 1 regression (the
  inference permission also unlocked implicit single-hop answers the
  never-guess rule had suppressed), adversarial untouched. Abstain
  rate 25.6%→24.7%.
- **Verdict: the production config.** Every category is at or within
  noise of its historic peak simultaneously — the first config to do
  that: single-hop 82.0 (peak), multi-hop 44.6 (ties peak), temporal
  91.1 (1.1 under the Run 6/11 peak), open-domain 42.9 (ties peak),
  adversarial 83.0 (peak). Flags: extraction + entity graph at write;
  adaptive routing + route-aware hardened Chain-of-Note at read.

### Run 19 — `results/locomo/routing-coverage-budget-20260704/` (coverage item budget, gnosis PR #43+#44)

- Run 18 config plus `GNOSIS_COVERAGE_BUDGET_MULTIPLIER=2`: the
  request's `max_items` fact cut doubles (20→40) on multi-hop- and
  aggregative-routed reads. Built from the Run 18 miss analysis (27 of
  41 multi-hop-category misses are cross-session enumerations whose
  facts rank in the 100-deep dense pool but below the cut). A live
  router probe redirected the design mid-flight: the classifier files
  LOCOMO's enumerations under *aggregative*, not multi_hop, so the
  multiplier covers both routes (PR #44). Smoke-verified the mechanism
  before launch: "What desserts has Maria made?" rendered 40 facts
  including BOTH gold desserts, where Run 18's 20 facts held one.
  Context only.
- **Scores: 72.5 excl-adv / 74.9 overall — below Run 18 (74.8/76.7).
  Rejected.** multi-hop 43.2, single-hop 78.5, temporal 90.0,
  open-domain 42.9, adversarial 83.0.
- Per-question verification, the decisive part: retrieval changed on
  60/497 questions and gold-item presence on the 27-question
  enumeration cohort rose **50%→60%** (fully-covered questions 6→7) —
  the coverage mechanism *worked* — yet **0 of 27 were repaired**.
  Even fully-covered questions still answer with a subset (the
  desserts question had both items in context and still answered
  "peach cobbler"). The residual enumeration failure is the
  reader/judge seam: LOCOMO's list golds demand the exact multi-item
  enumeration, and a reader over 40 facts picks a defensible different
  subset. Meanwhile the runs' 437 byte-identical-retrieval questions
  flipped 20 times anyway (6 up / 14 down — router + reader sampling
  noise), which fully accounts for the headline drop.
- **Two conclusions.** (1) Multi-hop's remaining ~55 J gap is NOT a
  retrieval problem — retrieval now covers 60% of gold items and
  answers don't move; the category is capped by exact-list grading of
  genuinely ambiguous enumerations. (2) The noise floor is now the
  dominant term between configs: ±2.3 J excl-adv between two runs of
  effectively identical retrieval. Flag stays merged, default 1 (off);
  Run 18 remains the production config.

## LongMemEval_S — primary optimization target (from 2026-07-04)

**Frozen config (LongMemEval_S)**: 100-instance stratified subset of the
500-question benchmark ([arXiv 2410.10813](https://arxiv.org/abs/2410.10813)):
all 30 abstention (`*_abs`) instances plus 70 non-abstention instances
sampled proportionally per question type (largest-remainder allocation,
seed 42). Resulting per-type composition (abstention instances also
carry a question type): 30 multi-session, 25 temporal-reasoning, 17
knowledge-update, 16 single-session-user, 8 single-session-assistant,
4 single-session-preference. 49,437 turns across 4,783 haystack
sessions. The question-id list is
committed as `data/longmemeval_s_subset100.txt` and the data file is
regenerated deterministically by `membench/scripts/make_lme_subset100.py`.
Rationale: extraction-enabled adds on LongMemEval-sized turns cost ~30s
each, so the full 500 (or 150) is not ingestable in one campaign window;
abstention is over-sampled deliberately because it is a new axis LOCOMO
lacks and n=6 proportional would be unmeasurable.

Stack config: Run 18 production flags (extraction + entity graph at
write; adaptive routing + route-aware hardened CoN v3 with the
likelihood carve-out at read) **plus two LongMemEval-specific
deviations**:

- **Embeddings: `gemini-embedding-001` at 3072 dims** (vs the LOCOMO
  gate's local qwen3/1024) — see the embedder note at the top.
- **`GNOSIS_SCOPED_DENSE_RETRIEVAL_ENABLED=true`** (gnosis PR #46,
  pool 10,000): LongMemEval instances share haystack sessions, so the
  100 instances live as ~100 users in one store whose fact vectors are
  near-duplicates across users. The SDK's dense path ranks the vector
  index *globally* and scope-filters afterwards, so with ~100 users'
  near-duplicate facts in one store the requesting user's facts get
  crowded out of the global top-k. The flag narrows the
  vector query to the request scope in-query. This is a correctness
  requirement for multi-user single-store benchmarking, not an
  optimization; single-user LOCOMO is unaffected (flag-off path is
  byte-identical).

Ingest protocol: one gnosis `user_id` per instance
(`longmemeval_s:<question_id>`), one gnosis session per haystack
session with the dataset's per-session timestamp as `session_date`,
turns paired user/assistant with inline date prefixes on.
Grading: the official LongMemEval judge prompts per question type
(abstention judged by the official abstention prompt), judge gpt-5.5
frozen as on LOCOMO. Headline metric: overall accuracy plus the
per-question-type breakdown (the paper's 5 ability axes report is
derived from question types).

Smoke run (3 instances, one per protocol-critical type —
`results/longmemeval_s/smoke3/`): temporal-reasoning and abstention
passed end-to-end; the knowledge-update instance failed by answering
with the *superseded* older value — expected at baseline (supersession
is a known unshipped lever) and confirms the axis measures what it
claims. The smoke run also flushed out and fixed three pipeline bugs
(membench PRs #13/#14: add retries + transport-error retries; gnosis
PR #47: extraction re-samples malformed LLM JSON instead of 500ing).

| Run | Change under test | Overall | Verdict |
|---|---|---|---|
| L-0 (baseline) | Run 18 config + gemini embeddings + scoped dense retrieval | *ingesting* | — |

## Published comparison targets

LOCOMO overall J as published (gpt-4o-mini judge — different judge and
backbone, so directional only; cross-vendor numbers in this space are
actively disputed): OpenAI memory 52.9 · LangMem 58.1 · Zep 66.0 ·
mem0 66.9 · mem0-graph 68.4 · full-context 72.9 · Letta (blog) 74.0.

## Known limitations of the current record

- Subset 3 of 10 LOCOMO conversations; LongMemEval_S not yet run at scale.
- Answerer route changed between Run 1 and Runs 2-4 (Copilot quota) — the
  judge was held constant, but the context-vs-search comparison within Run 1
  is the cleanest same-route pair.
- Runs 1-4 ingest verbatim (no LLM extraction); Run 5 onward ingests with
  edu-v1 fact extraction.
- Weekly regression runs (subset 2, this same frozen judge) execute in-cluster
  via the homelab `membench` CronJob and upload to RustFS `membench/results/`.

## Research sources behind the measured changes

The changes tested above were not guesses — each traces to specific literature,
dissected in this repo's research docs: [docs/extraction-design.md](docs/extraction-design.md),
[docs/frontier-2026.md](docs/frontier-2026.md),
[docs/gaps-abstention-maintenance.md](docs/gaps-abstention-maintenance.md).

**Sources that directly shaped shipped changes:**

- **EMem** ([arXiv 2511.17208](https://arxiv.org/abs/2511.17208)) — enriched
  elementary discourse units: self-contained, dated, entity-normalized fact
  statements. Basis of Run 5's edu-v1 extraction (+11.7 J, temporal +42).
  Its recall-filter ablation motivated Run 4 — which did NOT reproduce on our
  stack (a useful negative result: component gains are stack-dependent).
- **Zep / Graphiti** ([arXiv 2501.13956](https://arxiv.org/abs/2501.13956)) —
  temporal knowledge-graph design; inspired dating every rendered fact
  (Run 2) and the event_date/created_at bi-temporal seam in extracted facts.
  Graphiti's never-hallucinate-dates prompt rules are embedded in edu-v1.
- **mem0** ([arXiv 2504.19413](https://arxiv.org/abs/2504.19413), ECAI 2025) —
  calibration evidence that extraction quality dominates graph structure
  (their graph variant adds only ~1.6 J); their OSS extraction prompts
  informed edu-v1's guardrails. Also the source of the published comparison
  table and the headline-J-excluding-adversarial convention.
- **Mnemis / frontier hybrid-retrieval consensus** (dissected in
  docs/frontier-2026.md) — BM25+dense with RRF fusion appears in all three
  strongest 2026 systems; Mnemis's ablation attributes its largest gain to it.
  Basis of Run 6 (gnosis PR #15).

**Foundational papers (first verified research pass):**

- MemGPT ([arXiv 2310.08560](https://arxiv.org/abs/2310.08560)) — layered
  memory, LLM-directed memory operations.
- Generative Agents ([arXiv 2304.03442](https://arxiv.org/abs/2304.03442),
  UIST 2023) — importance/recency/relevance retrieval scoring, reflection.
- HippoRAG 2 ([arXiv 2502.14802](https://arxiv.org/abs/2502.14802), ICML
  2025) — passages-as-graph-nodes + Personalized PageRank; the strongest
  peer-reviewed hybrid-retrieval evidence; queued behind Run 6.
- MemoryBank ([arXiv 2305.10250](https://arxiv.org/abs/2305.10250), AAAI
  2024) — Ebbinghaus decay with recall reinforcement (unproven for QA
  accuracy; parked).

**Benchmarks:**

- LOCOMO ([arXiv 2402.17753](https://arxiv.org/abs/2402.17753)) — this log's
  primary benchmark; known ceiling: ~6.4% erroneous gold answers.
- LongMemEval ([arXiv 2410.10813](https://arxiv.org/abs/2410.10813)) —
  primary target from 2026-07-04 (knowledge-update + abstention categories
  LOCOMO lacks); see the LongMemEval_S section.

**Open-gap sources (abstention + maintenance, see gaps doc):**

- Sufficient Context ([arXiv 2411.06037](https://arxiv.org/abs/2411.06037)) —
  explains our adversarial drop (richer context suppresses abstention even
  when insufficient); sufficiency autorater is the planned fix.
- AbstentionBench ([arXiv 2506.09038](https://arxiv.org/abs/2506.09038)) —
  evidence-grounded abstention prompting raises abstention without precision
  loss.
- "Don't Ask the LLM to Track Freshness" ([arXiv 2606.01435](https://arxiv.org/abs/2606.01435)) —
  deterministic read-time newest-wins (78–94.8%) crushes LLM/bi-temporal
  invalidation (Zep: 7.0%) on FactConsolidation — this REVERSED our original
  plan to build write-time bi-temporal invalidation, and validates gnosis's
  append-only + read-time recency design.
- Selective memory addition ([arXiv 2505.16067](https://arxiv.org/abs/2505.16067)) —
  add-all degrades accuracy over time (67.5→55.5); store-time selectivity
  matters for the maintenance roadmap.
