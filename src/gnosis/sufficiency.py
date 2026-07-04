"""Cheap evidence-sufficiency autorater over assembled memory context.

Google's "Sufficient Context" study (arXiv 2411.06037) shows that richer
retrieval makes answer models *more* confident and *less* likely to abstain,
even when the context does not determine the answer - the same effect behind
our measured adversarial/abstention regression (74.1 -> 67.9). Their fix is a
prompted autorater that judges "does the retrieved context fully determine the
answer?" (>=93% agreement with human experts). gnosis is the memory service,
not the answering model, so it exposes that signal to clients rather than
deciding to abstain itself.

One structured-output LLM call per query, mirroring ``recall_filter.py``: no
temperature, a small completion cap. Every failure degrades to "not assessed"
so the sufficiency check can never block a context response.
"""

import logging
import time
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

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_SUFFICIENCY_GUIDE: Final[str] = """
You judge whether retrieved memories are sufficient to answer one query.
You get the query and the assembled memory context.
Context is sufficient only when the memories, on their own, fully determine a
correct answer to the query. If the answer is missing, only partially covered,
or the query presupposes something the memories do not support, it is not
sufficient. Return sufficient true/false and a short reason (one sentence).
""".strip()

# The response is a boolean plus one short sentence, so a small cap holds the
# latency budget while leaving room for structured-output framing.
_MAX_COMPLETION_TOKENS: Final[int] = 200
_MAX_REASON_LENGTH: Final[int] = 300


class SufficiencyVerdict(BaseModel):
    """Structured autorater output: sufficiency judgement plus a short reason."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    sufficient: bool = False
    reason: str = Field(default="")


class SufficiencyAssessor(Protocol):
    async def assess(
        self,
        query: str,
        context: str,
    ) -> SufficiencyVerdict | None: ...


@dataclass(frozen=True, slots=True)
class LiteLLMSufficiencyAssessor:
    model: str
    base_url: str
    api_key: str

    async def assess(
        self,
        query: str,
        context: str,
    ) -> SufficiencyVerdict | None:
        start = time.perf_counter()
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            response = await client.beta.chat.completions.parse(
                messages=_messages(query, context),
                model=proxy_model_name(self.model),
                # gpt-5.x endpoints reject `temperature` and `max_tokens`, so
                # this call sends neither and caps via max_completion_tokens.
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                response_format=SufficiencyVerdict,
            )
        verdict = response.choices[0].message.parsed
        if verdict is None:
            _LOGGER.info(
                "sufficiency check returned no content",
                extra={"model": self.model},
            )
            return None
        _LOGGER.info(
            "sufficiency check produced verdict",
            extra={
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "model": self.model,
                "sufficient": verdict.sufficient,
            },
        )
        return verdict


def bounded_reason(reason: str) -> str | None:
    """Trim the model reason to a short, single line, or ``None`` when empty."""
    collapsed = " ".join(reason.split())
    if not collapsed:
        return None
    return collapsed[:_MAX_REASON_LENGTH]


def _messages(
    query: str,
    context: str,
) -> tuple[ChatCompletionMessageParam, ...]:
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": _SUFFICIENCY_GUIDE,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": f"Query: {query}\n\nMemory context:\n{context}",
    }
    return (system_message, user_message)
