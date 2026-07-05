"""Listwise LLM reranker over fused long-term fact candidates.

Retrieval is the bottleneck on long-haystack memory benchmarks (LongMemEval:
full-context 0.606 vs oracle retrieval 0.870 - the whole gap is which facts
reach the reader). Dense/hybrid similarity ranks candidates by vector
proximity, which mis-orders near-duplicate long-haystack facts; a cross-encoder
or LLM reranker over the fused pool is the single lever present in every
strongest 2026 system (e.g. Mnemis' hybrid+reranker ablation 73.8 -> 89.1).
No cross-encoder is exposed on the deployment's LiteLLM, so this is a listwise
LLM reranker (RankGPT-style): one structured-output call scores the top
candidates by query relevance and returns their ranked order.

Mirrors ``sufficiency.py``: one structured-output LLM call per query, no
temperature, a small completion cap, and every failure degrades to the
input order so reranking can never drop a candidate or block a context read.
Reordering happens BEFORE the item-budget cut, so the reranker decides which
candidates survive into the prompt.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Final, Protocol

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ConfigDict, Field

from gnosis.graph_query_qa import proxy_model_name
from gnosis.models import JsonObject
from gnosis.settings import Settings

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_RERANK_GUIDE: Final[str] = """
You reorder retrieved memories by how useful each is for answering one query.
You get the query and a numbered list of candidate memories.
Return "order": the candidate numbers ranked from most to least relevant to the
query. Include every candidate number exactly once. Judge relevance to the
specific query - a memory that directly answers or narrows the query outranks a
memory that merely shares a topic. Do not invent numbers.
""".strip()

# The candidates carry the retrieval signal; capping how many the LLM reorders
# holds the prompt and latency bounded while covering the pool that feeds a
# typical item budget. Candidates past the cap keep their retrieval order after
# the reranked head.
_DEFAULT_CANDIDATE_CAP: Final[int] = 50
# Order-only output (a permutation of small ints), so a modest cap suffices.
_MAX_COMPLETION_TOKENS: Final[int] = 600
# Reranking has no effect below two candidates - nothing to reorder.
MIN_RERANK_CANDIDATES: Final[int] = 2


class RerankResult(BaseModel):
    """Structured reranker output: candidate indices, most relevant first."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    order: list[int] = Field(default_factory=list)


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: Sequence[str],
    ) -> RerankResult | None: ...


@dataclass(frozen=True, slots=True)
class LiteLLMReranker:
    model: str
    base_url: str
    api_key: str

    async def rerank(
        self,
        query: str,
        candidates: Sequence[str],
    ) -> RerankResult | None:
        start = time.perf_counter()
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            response = await client.beta.chat.completions.parse(
                messages=_messages(query, candidates),
                model=proxy_model_name(self.model),
                # gpt-5.x endpoints reject `temperature`/`max_tokens`; cap via
                # max_completion_tokens only, as the sufficiency check does.
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                response_format=RerankResult,
            )
        result = response.choices[0].message.parsed
        if result is None:
            _LOGGER.info("rerank returned no content", extra={"model": self.model})
            return None
        _LOGGER.info(
            "rerank produced order",
            extra={
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "model": self.model,
                "candidates": len(candidates),
                "ranked": len(result.order),
            },
        )
        return result


def rerank_model(settings: Settings) -> str:
    """The reranker model: its own override, else the main gnosis LLM."""
    return settings.gnosis_rerank_model or settings.gnosis_llm


def rerank_candidate_cap(settings: Settings) -> int:
    cap = settings.gnosis_rerank_candidate_cap
    return cap if cap > 0 else _DEFAULT_CANDIDATE_CAP


def apply_rerank(
    facts: list[JsonObject],
    order: Sequence[int],
    cap: int,
) -> list[JsonObject]:
    """Reorder the top ``cap`` candidates by ``order``; never drop a candidate.

    Only the reranked head (first ``cap`` facts) is reordered. Indices out of
    range or repeated are ignored; any head candidate the model omitted is
    appended in its original retrieval order, so the set of facts is preserved
    exactly and only their order changes. Candidates past the cap keep their
    retrieval order after the reranked head.
    """
    head = facts[:cap]
    tail = facts[cap:]
    seen: set[int] = set()
    reordered: list[JsonObject] = []
    for index in order:
        if 0 <= index < len(head) and index not in seen:
            seen.add(index)
            reordered.append(head[index])
    for index, fact in enumerate(head):
        if index not in seen:
            reordered.append(fact)
    return [*reordered, *tail]


def _messages(
    query: str,
    candidates: Sequence[str],
) -> tuple[ChatCompletionMessageParam, ...]:
    numbered = "\n".join(f"{index}. {line}" for index, line in enumerate(candidates))
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": _RERANK_GUIDE,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": f"Query: {query}\n\nCandidate memories:\n{numbered}",
    }
    return (system_message, user_message)
