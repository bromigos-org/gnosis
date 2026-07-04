"""EMem-style LLM recall filter over retrieved long-term memory candidates.

After vector ranking, one LLM call screens the top candidates against the
query and keeps only those that could help answer it (arXiv 2511.17208 shows
this post-retrieval filter is the single most valuable retrieval component).
The filter only ever removes or keeps candidates - it can never add items, so
scope enforcement stays untouched - and every failure mode degrades to the
unfiltered ranking.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Final, Protocol

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ConfigDict, Field

from gnosis.graph_query_qa import proxy_model_name

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_RECALL_FILTER_GUIDE: Final[str] = """
You screen retrieved long-term memory candidates for one query.
You get the query and a numbered candidate list, one dated memory per line.
Select every candidate that could help answer the query, directly or as
partial evidence: dates, people, places, counts, or intermediate facts that a
multi-hop or temporal answer may need. Keep a candidate when unsure.
Return the selected candidate numbers in kept_indices, exactly as numbered.
Never return a number that is not in the list.
""".strip()

# The response is just indices, so a small completion cap holds the latency
# budget while leaving room for structured-output framing.
_MAX_COMPLETION_TOKENS: Final[int] = 500


class RecallSelection(BaseModel):
    """Structured filter output: 1-based candidate numbers worth keeping."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kept_indices: list[int] = Field(default_factory=list)


class RecallFilter(Protocol):
    async def select_candidates(
        self,
        query: str,
        candidates: Sequence[str],
    ) -> RecallSelection | None: ...


@dataclass(frozen=True, slots=True)
class LiteLLMRecallFilter:
    model: str
    base_url: str
    api_key: str

    async def select_candidates(
        self,
        query: str,
        candidates: Sequence[str],
    ) -> RecallSelection | None:
        start = time.perf_counter()
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            response = await client.beta.chat.completions.parse(
                messages=_messages(query, candidates),
                model=proxy_model_name(self.model),
                # gpt-5.x endpoints reject `temperature` and `max_tokens`, so
                # this call sends neither and caps via max_completion_tokens.
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                response_format=RecallSelection,
            )
        selection = response.choices[0].message.parsed
        if selection is None:
            _LOGGER.info(
                "recall filter returned no content",
                extra={"model": self.model},
            )
            return None
        _LOGGER.info(
            "recall filter produced selection",
            extra={
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "model": self.model,
                "selected": len(selection.kept_indices),
            },
        )
        return selection


async def keep_relevant_candidates[ItemT](
    recall_filter: RecallFilter,
    *,
    query: str,
    items: Sequence[ItemT],
    render: Callable[[ItemT], str],
    max_candidates: int,
) -> list[ItemT]:
    """Keep the top-ranked items the filter judges useful for the query.

    Only the first ``max_candidates`` items go to the filter; kept items
    preserve their original rank order. Every failure mode - transport or
    model errors, an empty parse, or an empty/out-of-range selection - falls
    back to the unfiltered ranking, so the filter alone can never empty a
    result set.
    """
    candidates = list(items[:max_candidates])
    if not candidates:
        return list(items)
    try:
        selection = await recall_filter.select_candidates(
            query,
            [render(item) for item in candidates],
        )
    except (RuntimeError, OSError, OpenAIError) as error:
        _LOGGER.warning(
            "recall filter failed; keeping unfiltered ranking",
            extra={
                "candidates_in": len(candidates),
                "error_type": type(error).__name__,
            },
        )
        return list(items)
    if selection is None:
        _LOGGER.warning(
            "recall filter returned no selection; keeping unfiltered ranking",
            extra={"candidates_in": len(candidates)},
        )
        return list(items)
    positions = _valid_positions(selection.kept_indices, len(candidates))
    if not positions:
        _LOGGER.warning(
            "recall filter kept no valid candidates; keeping unfiltered ranking",
            extra={
                "candidates_in": len(candidates),
                "selected": len(selection.kept_indices),
            },
        )
        return list(items)
    kept = [candidates[position] for position in positions]
    _LOGGER.info(
        "recall filter applied",
        extra={"candidates_in": len(candidates), "kept": len(kept)},
    )
    return kept


def _valid_positions(kept_indices: Sequence[int], count: int) -> list[int]:
    """Map 1-based selection numbers to deduplicated in-range rank positions."""
    return sorted({index - 1 for index in kept_indices if 1 <= index <= count})


def _messages(
    query: str,
    candidates: Sequence[str],
) -> tuple[ChatCompletionMessageParam, ...]:
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": _RECALL_FILTER_GUIDE,
    }
    numbered = "\n".join(
        f"{position}. {line.removeprefix('- ')}"
        for position, line in enumerate(candidates, start=1)
    )
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": f"Query: {query}\n\nCandidates:\n{numbered}",
    }
    return (system_message, user_message)
