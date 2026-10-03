"""Bounded multi-query rewrite for insufficient retrieval context.

EverMemOS (arXiv:2601.02163) and MemCog (arXiv:2605.28046) demonstrate that when
initial retrieval is insufficient, generating 2-3 complementary queries and fusing
their results with RRF closes the multi-hop and open-domain gaps without
unacceptably inflating latency. EverMemOS fires this rewrite loop on 31% of
LoCoMo queries.

This is distinct from the LLM recall filter (Run 4, rejected): the filter
*discarded* candidates from a single retrieval (a precision move that hurt recall);
this module *adds* reformulated queries targeting different aspects of the question
(a recall move). The evidence base is specifically on the recall side.

Four rewrite strategies, per EverMemOS (§3.5 + App. A.1):
- ``entity_pivot``: extract bridge entities from the question and current context,
  query for facts about them directly — the primary multi-hop repair.
- ``temporal_calculation``: anchor relative date expressions against retrieved
  session dates ("last Tuesday" → absolute date), then re-query.
- ``concept_expansion``: paraphrase the query using synonyms and related concepts
  from the retrieved context — open-domain broadening.
- ``hyde``: generate a hypothetical answer and use it as the retrieval query
  (Hypothetical Document Embeddings, Gao et al. 2022).

Hard cap: one extra retrieval round (max 3 reformulated queries). The sufficiency
check (``sufficiency.py``) decides whether to fire; if it returns ``sufficient=True``
or is disabled, this module does nothing.

Failure modes are fully degraded: any rewrite or retrieval failure returns an
empty list of extra candidates, never blocking the original context assembly.
"""

import logging
from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ConfigDict, Field

from gnosis.graph_query_qa import proxy_model_name
from gnosis.llm_clients import shared_openai_client

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

type RewriteStrategy = Literal[
    "entity_pivot",
    "temporal_calculation",
    "concept_expansion",
    "hyde",
]

# Structured output cap: a JSON list of up to 3 short queries.
_MAX_REWRITE_TOKENS: Final[int] = 400
# Never issue more than this many complementary retrieval queries.
MAX_REWRITE_QUERIES: Final[int] = 3

_REWRITE_GUIDE: Final[str] = """
You generate complementary retrieval queries for a memory system.
The original query was not fully answerable from the retrieved context.
Generate up to 3 alternative retrieval queries that approach the question from
different angles to find missing information. Use these strategies as needed:

- entity_pivot: name a specific person, place, or thing mentioned or implied by
  the question and context, and ask about them directly.
- temporal_calculation: if the query involves time or sequence, convert any
  relative expressions to specific periods based on session dates in the context.
- concept_expansion: rephrase the query using synonyms or related concepts that
  might match differently-worded memories.
- hyde: write a short hypothetical answer as if the memory exists, then turn it
  into a search query.

Return a JSON object with "queries": a list of alternative query strings
(1-3 items, each a concise natural-language memory-retrieval query). Return
fewer if only 1-2 are genuinely complementary. Do not repeat the original query.
""".strip()


class RewriteResult(BaseModel):
    """Structured rewrite output: alternative queries to retrieve."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    queries: list[str] = Field(default_factory=list, max_length=MAX_REWRITE_QUERIES)


@dataclass(frozen=True, slots=True)
class LiteLLMQueryRewriter:
    """Generates complementary retrieval queries when context is insufficient."""

    model: str
    base_url: str
    api_key: str

    async def rewrite(
        self,
        original_query: str,
        retrieved_context: str,
        insufficiency_reason: str | None = None,
    ) -> RewriteResult | None:
        """Generate complementary queries for the original query.

        Returns ``None`` on any failure so callers can skip and use original
        retrieval only.
        """
        try:
            client = shared_openai_client(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            response = await client.beta.chat.completions.parse(
                messages=_rewrite_messages(
                    original_query, retrieved_context, insufficiency_reason
                ),
                model=proxy_model_name(self.model),
                max_completion_tokens=_MAX_REWRITE_TOKENS,
                response_format=RewriteResult,
            )
            result = response.choices[0].message.parsed
            if result is None:
                _LOGGER.info(
                    "query rewriter returned no content",
                    extra={"model": self.model},
                )
                return None
            valid_queries = [
                q.strip()
                for q in result.queries
                if q.strip() and q.strip() != original_query
            ][:MAX_REWRITE_QUERIES]
            _LOGGER.info(
                "query rewriter produced alternatives",
                extra={
                    "model": self.model,
                    "n_queries": len(valid_queries),
                },
            )
            return RewriteResult(queries=valid_queries)
        except Exception:
            _LOGGER.exception(
                "query rewrite failed",
                extra={"model": self.model},
            )
            return None


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[str]:
    """Reciprocal Rank Fusion over multiple ranked lists of fact IDs.

    Standard RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each list
    where d appears. k=60 is the EverMemOS default. Returns a single deduplicated
    ranked list, highest score first.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def _rewrite_messages(
    original_query: str,
    retrieved_context: str,
    insufficiency_reason: str | None,
) -> tuple[ChatCompletionMessageParam, ...]:
    context_snippet = retrieved_context[:2000] if retrieved_context else "(none)"
    reason_line = (
        f"\nReason the context was insufficient: {insufficiency_reason}"
        if insufficiency_reason
        else ""
    )
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": _REWRITE_GUIDE,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": (
            f"Original query: {original_query}"
            f"{reason_line}"
            f"\n\nRetrieved context so far:\n{context_snippet}"
        ),
    }
    return (system_message, user_message)
