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
]

_ROUTER_GUIDE: Final[str] = """
You classify one memory-retrieval query into exactly one route.
Routes:
- temporal: asks when something happened, a date, a duration, an ordering in
  time, or "how long ago / how many days" (e.g. "When did Maria adopt the
  cat?", "How long has Tom worked at the bakery?").
- multi_hop: needs two or more distinct remembered facts chained through a
  bridge entity to answer (e.g. "What instrument does the sister of John's
  coworker play?", "Which city is the company that Ana joined based in?").
- aggregative: asks for a broad summary, list, or synthesis across many
  conversations or topics (e.g. "What do they usually talk about?", "List all
  the hobbies mentioned.").
- unanswerable_risk: presupposes or asks about something personal memories
  likely never contain, fishing for a fact that was probably never said
  (e.g. "What brand of toothpaste does Bob's dentist recommend?").
- single_hop: everything else - one remembered fact answers it directly.
Choose the single best route.
""".strip()

# The response is one enum field, so a small cap holds the latency budget
# while leaving room for structured-output framing.
_MAX_COMPLETION_TOKENS: Final[int] = 100


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

    @classmethod
    def from_settings(cls, settings: Settings) -> "RouteDecision":
        """The unrouted decision: the globally configured feature flags."""
        return cls(
            route=None,
            hybrid_retrieval=settings.gnosis_hybrid_retrieval_enabled,
            graphqa_fusion=settings.gnosis_graphqa_fusion_enabled,
            verbatim_expansion=settings.gnosis_fact_verbatim_expansion_enabled,
            abstention_prompt=settings.gnosis_abstention_prompt_enabled,
        )

    @classmethod
    def for_route(cls, route: QueryRoute) -> "RouteDecision":
        """The measured-best feature set for one classified route."""
        return cls(
            route=route,
            hybrid_retrieval=route == "temporal",
            graphqa_fusion=route == "multi_hop",
            verbatim_expansion=route == "multi_hop",
            abstention_prompt=route == "unanswerable_risk",
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
            response = await client.beta.chat.completions.parse(
                messages=_messages(query),
                model=proxy_model_name(self.model),
                # gpt-5.x endpoints reject `temperature` and `max_tokens`, so
                # this call sends neither and caps via max_completion_tokens.
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                response_format=RouteVerdict,
            )
        verdict = response.choices[0].message.parsed
        if verdict is None:
            _LOGGER.info(
                "query router returned no content",
                extra={"model": self.model},
            )
            return None
        _LOGGER.info(
            "query router classified query",
            extra={
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "model": self.model,
                "route": verdict.route,
            },
        )
        return verdict


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
