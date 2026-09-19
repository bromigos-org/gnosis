"""Adaptive per-query retrieval routing (Adaptive-RAG, arXiv 2403.14403).

Run 9 (combined-config benchmark, 2026-07-04) proved gnosis's measured
per-category peaks do not stack: enabling every read-path feature at once
crashed the headline score 13 points because the features interfere (hybrid
BM25 wins temporal but displaces the multi-hop fact chain; the abstention
prompt wins adversarial but over-abstains on answerable queries). The
literature answer is routed retrieval: classify each query and apply only the
strategy that won that query's category in ablation, instead of one global
pipeline.

Behind ``GNOSIS_ADAPTIVE_ROUTING_ENABLED`` (default off), one cheap
structured-output LLM call tags each context/search query with a route and the
backend applies that route's measured-best feature set for the request:

======================  =====================================================
route                   read-path features applied (measured source)
======================  =====================================================
``temporal``            hybrid BM25+RRF (Run 6: temporal 84.4 -> 92.2)
``multi_hop``           graph-QA fusion over the entity graph + facts-to-
                        verbatim expansion, dense-only ranking (Run 6 showed
                        hybrid costs multi-hop -5.4; Run 8 verbatim was the
                        only multi-hop gain, +2.7)
``single_hop``          plain dense ranking (Run 5 extraction store: 80.5)
``unanswerable_risk``   abstention standing instruction (Run 7: adversarial
                        +8.9, quarantined here so it cannot over-abstain on
                        answerable queries, its measured -1.6 failure mode)
``aggregative``         plain dense ranking (Run 8 verbatim expansion hurt
                        open-domain -4.8; no measured winner yet)
======================  =====================================================

While the flag is off, every request uses the globally configured feature
flags unchanged (byte-identical behavior). While the flag is on, the router's
decision replaces the global read-path toggles for that request; any
classifier failure degrades to the globally configured flags with a
structured warning, so routing can never fail a read.
"""

import logging
import time
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, Protocol

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ConfigDict

from gnosis.graph_query_qa import proxy_model_name
from gnosis.settings import Settings

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

type QueryRoute = Literal[
    "single_hop",
    "multi_hop",
    "temporal",
    "unanswerable_risk",
    "aggregative",
    "knowledge_update",
]

_ROUTER_GUIDE: Final[str] = """
You classify one memory-retrieval query into exactly one route.
Routes:
- temporal: asks WHEN something happened, a specific DATE, elapsed time
  ("how long ago", "how many days/months ago", "how many days/months since"),
  or ordering in time (e.g. "When did Maria adopt the cat?", "How long has Tom
  worked at the bakery?", "How many months have passed since my last visit?",
  "How long had I been running when I finished my first 5K?").
  IMPORTANT: Do NOT classify here if the question asks "how many TIMES" or
  "how many DIFFERENT things" — those are frequency counts, not time offsets;
  route them to aggregative instead.
- multi_hop: needs two or more distinct remembered facts chained through a
  bridge entity to answer (e.g. "What instrument does the sister of John's
  coworker play?", "Which city is the company that Ana joined based in?").
- aggregative: asks for a count, frequency, or synthesis across many memories
  (e.g. "What do they usually talk about?", "List all the hobbies mentioned.",
  "How many times did I do X?", "How many different Y did I attend?").
- knowledge_update: asks for the CURRENT value of a fact the user has
  changed or updated over time (e.g. "What is my current job?", "Where do I
  live now?", "What's my marathon PR?", "How many bikes do I have?", "Which
  car do I drive?"). Use when the correct answer is the user's MOST RECENTLY
  mentioned value of something that plausibly changes. Do NOT use for stable
  biographical facts ("what is my daughter's name?") or when the question
  explicitly asks about a past state.
- unanswerable_risk: presupposes or asks about something personal memories
  definitively do not contain — a private/obscure fact that could not
  realistically appear in casual personal conversation
  (e.g. "What brand of toothpaste does Bob's dentist recommend?",
  "What is my coworker's mother's maiden name?").
  CRITICAL: When the query says "I mentioned X", "I told you X", "I discussed
  X", or otherwise signals the user KNOWS the topic was recorded, that fact IS
  in memory — choose single_hop, temporal, multi_hop, or aggregative as
  appropriate, NEVER unanswerable_risk. Also prefer temporal over
  unanswerable_risk whenever the query asks how long, when, what date, or how
  many days/months (elapsed time) — those are duration/time questions, not
  absence questions.
- single_hop: everything else - one remembered fact answers it directly.
Choose the single best route. Respond with only the route name.
""".strip()

# The response is one route token, so a small cap holds the latency budget.
_MAX_COMPLETION_TOKENS: Final[int] = 100

_ROUTES: Final[tuple[QueryRoute, ...]] = (
    "single_hop",
    "multi_hop",
    "temporal",
    "unanswerable_risk",
    "aggregative",
    "knowledge_update",
)


class RouteVerdict(BaseModel):
    """Structured router output: the single chosen route."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    route: QueryRoute = "single_hop"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """The effective read-path feature set for one routed request."""

    route: QueryRoute | None
    hybrid_retrieval: bool
    graphqa_fusion: bool
    verbatim_expansion: bool
    abstention_prompt: bool
    graph_traversal: bool
    bridge_traversal: bool
    chain_of_note: bool
    budget_multiplier: int
    supersession_enabled: bool
    sufficiency_check_enabled: bool
    recency_injection_enabled: bool = False
    filter_superseded: bool = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "RouteDecision":
        """The unrouted decision: the globally configured feature flags."""
        return cls(
            route=None,
            hybrid_retrieval=settings.gnosis_hybrid_retrieval_enabled,
            graphqa_fusion=settings.gnosis_graphqa_fusion_enabled,
            verbatim_expansion=settings.gnosis_fact_verbatim_expansion_enabled,
            abstention_prompt=settings.gnosis_abstention_prompt_enabled,
            graph_traversal=settings.gnosis_graph_traversal_enabled,
            bridge_traversal=settings.gnosis_bridge_traversal_enabled,
            chain_of_note=settings.gnosis_chain_of_note_enabled,
            budget_multiplier=settings.gnosis_coverage_budget_multiplier,
            supersession_enabled=settings.gnosis_read_supersession_enabled,
            sufficiency_check_enabled=settings.gnosis_sufficiency_check_enabled,
            recency_injection_enabled=False,
            filter_superseded=False,
        )

    @classmethod
    def for_route(cls, route: QueryRoute, settings: Settings) -> "RouteDecision":
        """The measured-best feature set for one classified route.

        Entity traversal is not yet in any route's measured-best set, so a
        routed request runs it only where it *could* win - multi-hop - and
        only when its own flag is on; the flag alone (routing off) applies
        it to every query for standalone measurement. Directed bridge
        traversal follows the same contract: multi-hop only, and only when
        its own flag is on.

        Chain-of-Note is route-aware by measurement: stacked globally with
        routing it *cost* temporal 8.9 points (Run 14, 2026-07-04) because
        the note step makes the reader faithfully report the relative dates
        in hybrid's raw verbatim turns ("last Saturday") instead of the
        resolved dated facts - so a routed request reads with Chain-of-Note
        on every route except temporal.
        """
        return cls(
            route=route,
            # Temporal gets hybrid BM25 (Run 6: exact dates/names in verbatim
            # turns). Aggregative also gets it: LME_S analysis shows exact
            # keyword matching surfaces scattered per-event facts (art events,
            # camping trips) that fall below the dense-only top-20 cut.
            # Single_hop added (LME_S L-20): two stable-wrong questions
            # (0bc8ad93 museum companion, a96c20ee_abs university poster) are
            # likely single_hop with specific entity/keyword facts that fall
            # below dense top-20; BM25 should surface them. LOCOMO evidence:
            # global BM25 was neutral for single_hop (80.5→79.5, Run 6).
            # Knowledge_update added (L-29): entity/value names in the update
            # conversation are exact-match candidates; BM25 surfaces them.
            hybrid_retrieval=route in (
                "temporal", "aggregative", "single_hop", "knowledge_update"
            ),
            graphqa_fusion=route == "multi_hop",
            verbatim_expansion=route == "multi_hop",
            abstention_prompt=route == "unanswerable_risk",
            graph_traversal=(
                route == "multi_hop" and settings.gnosis_graph_traversal_enabled
            ),
            bridge_traversal=(
                route == "multi_hop" and settings.gnosis_bridge_traversal_enabled
            ),
            chain_of_note=(
                route != "temporal" and settings.gnosis_chain_of_note_enabled
            ),
            # Coverage, not traversal, is the measured multi-hop gap (Run 18
            # miss analysis: 27/41 misses are cross-session enumerations cut
            # by the item budget), and the router classifies enumerations as
            # aggregative or multi_hop - so exactly those two coverage-hungry
            # routes read with an expanded budget.
            budget_multiplier=(
                settings.gnosis_coverage_budget_multiplier
                if route in ("multi_hop", "aggregative")
                else 1
            ),
            # Temporal queries ask for historical first-occurrence or
            # timestamped events. Supersession's "newest-wins" aggregation
            # discards earlier facts for the same slot, removing the dated
            # history that temporal retrieval needs (L-4 result: -21pp vs
            # L-0 baseline when supersession was applied globally).
            # Unanswerable-risk also opts out: it needs the full fact set
            # to correctly conclude "this topic was never mentioned" rather
            # than being misled by a thin slice of the most-recent context.
            # Aggregative also opts out: summing/counting across sessions
            # requires all occurrences; "newest-wins" discards older values
            # that are addends (e.g. combined reading time across two books).
            supersession_enabled=(
                route not in ("temporal", "unanswerable_risk", "aggregative")
                and settings.gnosis_read_supersession_enabled
            ),
            # Sufficiency check signals "not enough info" — useful only for
            # unanswerable_risk, where detecting genuine absence is the goal.
            # Applied globally it marks temporal/SSU facts as insufficient and
            # the model abstains on answerable questions (L-6: -16pp temporal).
            sufficiency_check_enabled=(
                route == "unanswerable_risk"
                and settings.gnosis_sufficiency_check_enabled
            ),
            # Recency injection fetches the top-N most recently created facts
            # (by ingest timestamp) and merges them into the candidate set
            # before the item budget cut. Knowledge-update questions target
            # the current state of a changing fact; the newer value is often
            # buried below the dense top-20 because its embedding similarity
            # to the question is lower than the earlier, more directly-stated
            # original fact. Injecting by recency guarantees the latest ingest
            # is always a candidate regardless of embedding rank.
            recency_injection_enabled=(route == "knowledge_update"),
            # Structural supersession filter: exclude facts with valid_to set
            # (marked superseded at write time via SUPERSEDES edges) from the
            # vector and lexical candidate pools. For knowledge_update queries,
            # this ensures the dense top-20 contains only current (non-
            # superseded) facts so the newest value naturally ranks first
            # without fighting stale facts in embedding space.
            filter_superseded=(route == "knowledge_update"),
        )


class QueryRouter(Protocol):
    async def classify(self, query: str) -> RouteVerdict | None: ...


@dataclass(frozen=True, slots=True)
class LiteLLMQueryRouter:
    model: str
    base_url: str
    api_key: str

    async def classify(self, query: str) -> RouteVerdict | None:
        start = time.perf_counter()
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            # A plain completion parsed leniently, NOT structured output: the
            # LiteLLM gpt-5.5 route answers a single-enum-property JSON schema
            # with the bare enum value ("temporal"), which the strict SDK
            # parser rejects. gpt-5.x endpoints also reject `temperature` and
            # `max_tokens`, so this call sends neither and caps via
            # max_completion_tokens.
            response = await client.chat.completions.create(
                messages=_messages(query),
                model=proxy_model_name(self.model),
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
            )
        content = response.choices[0].message.content
        route = parse_route(content)
        if route is None:
            _LOGGER.info(
                "query router returned no recognizable route",
                extra={"model": self.model, "content": (content or "")[:120]},
            )
            return None
        _LOGGER.info(
            "query router classified query",
            extra={
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "model": self.model,
                "route": route,
            },
        )
        return RouteVerdict(route=route)


def parse_route(content: str | None) -> QueryRoute | None:
    """Extract the chosen route from a model reply, leniently.

    Accepts the bare route token (what the prompt asks for), a quoted or
    JSON-wrapped variant (``{"route": "temporal"}``), and hyphen/case noise
    (``Multi-Hop``). Ambiguous replies naming several routes, or replies
    naming none, return ``None`` so the caller can fall back.
    """
    if not content:
        return None
    normalized = content.strip().casefold().replace("-", "_")
    found: list[QueryRoute] = [route for route in _ROUTES if route in normalized]
    if len(found) != 1:
        return None
    return found[0]


def _messages(query: str) -> tuple[ChatCompletionMessageParam, ...]:
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": _ROUTER_GUIDE,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": f"Query: {query}",
    }
    return (system_message, user_message)
