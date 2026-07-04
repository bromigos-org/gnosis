"""Directed bridge-entity traversal for multi-hop context retrieval (T1-directed).

Run 12 (graph-traversal benchmark, 2026-07-04) rejected *radial* traversal:
blindly expanding 1-2 RELATES hops from every entity the query names floods
the reserved graph budget with neighborhood noise and multi-hop went down.
The self-ask prior (arXiv 2210.03350, IRCoT arXiv 2212.10509) says the hop
must be *directed*: first resolve hop 1 from the retrieved evidence, then
fetch hop 2's facts about the specific bridge entity hop 1 revealed.

Behind ``GNOSIS_BRIDGE_TRAVERSAL_ENABLED`` (default off), this module runs
that loop with one extra LLM call, only on queries the router already
classified multi-hop:

1. **Name the bridge.** The dense-ranked facts (hop 1's evidence) and the
   query go to one cheap completion that names up to three entities the
   facts reveal are needed but the question never names - "who did John go
   to yoga with?" plus a fact naming the colleague Rob yields ``Rob``.
2. **Fetch hop 2.** A fixed parameterized Cypher reads the dated extracted
   facts that ``MENTIONS`` those bridge entities - the evidence a dense or
   lexical ranking of the *query* can never surface, because the bridge
   entity does not appear in the query text.
3. **Fuse.** The bridge facts join the candidate pool as graph-derived
   candidates, holding the reserved graph slots of the item budget.

Entities the query itself names are filtered out of the bridge list (dense
retrieval already covers them; they are hop 1, not the bridge), so a namer
that parrots the query degrades to a no-op instead of double-fetching.
"""

import logging
import time
from dataclasses import dataclass
from typing import Final, Protocol

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from gnosis.graph_query_qa import proxy_model_name
from gnosis.graph_types import CypherParameters
from gnosis.models import JsonValue

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

# At most this many bridge entities per query keeps the hop-2 read narrow -
# a directed hop resolves one or two bridges, never a neighborhood.
MAX_BRIDGE_ENTITIES: Final[int] = 3

# Hop-1 evidence shown to the namer. The reserved-budget cut renders at most
# ~20 facts, so the namer sees what the reader will see.
MAX_EVIDENCE_LINES: Final[int] = 12

_NAMER_GUIDE: Final[str] = """
You resolve bridge entities for one multi-hop memory question.
You get the question and the top retrieved memory facts. Name up to three
entities (people, places, organizations, events, things) that the facts
reveal are needed to answer the question but that the question itself does
not name. Example: the question asks who "the colleague" is and a fact
names the colleague Rob - answer: Rob.
Respond with only the entity names, one per line, nothing else.
If the facts reveal no such bridge entity, respond with exactly: NONE
""".strip()

# The response is at most three short names.
_MAX_COMPLETION_TOKENS: Final[int] = 100

# Fetch the dated extracted facts that mention the named bridge entities.
# Scope is pinned twice: entities match only within tenant_id + user_id (the
# entity dedup scope) and every fact re-checks the caller's metadata scope
# fragments in-query, with the gateway re-checking scope on the rows again.
BRIDGE_MENTION_CYPHER: Final[str] = """
MATCH (e:Entity {tenant_id: $tenant_id, user_id: $user_id})
WHERE e.normalized IN $bridges
MATCH (f:Fact)-[:MENTIONS]->(e)
WHERE f.metadata IS NOT NULL
  AND all(fragment IN $scope_fragments WHERE f.metadata CONTAINS fragment)
RETURN DISTINCT f.id AS id,
       f.subject AS subject,
       f.predicate AS predicate,
       f.object AS object,
       f.metadata AS metadata,
       toString(f.created_at) AS created_at,
       toString(f.updated_at) AS updated_at
ORDER BY created_at DESC, id ASC
LIMIT $limit
"""

_LINE_PREFIXES: Final[tuple[str, ...]] = ("-", "*", "\u2022")


class BridgeNamer(Protocol):
    async def name_bridges(self, query: str, evidence: list[str]) -> str | None: ...


@dataclass(frozen=True, slots=True)
class LiteLLMBridgeNamer:
    model: str
    base_url: str
    api_key: str

    async def name_bridges(self, query: str, evidence: list[str]) -> str | None:
        """One plain completion naming the bridge entities, or None when empty.

        Plain text parsed leniently, not structured output, for the same
        reason as the query router: the LiteLLM gpt-5.x routes answer simple
        schemas with bare values the strict SDK parser rejects, and reject
        ``temperature``/``max_tokens``.
        """
        start = time.perf_counter()
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            response = await client.chat.completions.create(
                messages=_messages(query, evidence),
                model=proxy_model_name(self.model),
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
            )
        content = response.choices[0].message.content
        _LOGGER.info(
            "bridge namer answered",
            extra={
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "model": self.model,
                "content": (content or "")[:120],
            },
        )
        return content


def parse_bridge_names(content: str | None) -> list[str]:
    """Entity display names from a namer reply, leniently.

    Accepts one name per line (what the prompt asks for), comma-separated
    names, and bullet/numbering noise. ``NONE`` and empty replies parse to
    no names. Deduplicated case-insensitively in reply order and capped at
    ``MAX_BRIDGE_ENTITIES``.
    """
    if not content:
        return []
    names: dict[str, str] = {}
    for line in content.splitlines():
        for piece in line.split(","):
            name = _cleaned_name(piece)
            if not name or name.casefold() == "none":
                continue
            _ = names.setdefault(name.casefold(), name)
            if len(names) >= MAX_BRIDGE_ENTITIES:
                return list(names.values())
    return list(names.values())


def _cleaned_name(piece: str) -> str:
    """One reply token with bullet, numbering, and quote noise removed."""
    cleaned = piece.strip()
    while cleaned and (
        cleaned[0] in "\"'\u2018\u2019\u201c\u201d"
        or cleaned.startswith(_LINE_PREFIXES)
    ):
        cleaned = cleaned[1:].lstrip()
    prefix, separator, rest = cleaned.partition(". ")
    if separator and prefix.isdigit():
        cleaned = rest.strip()
    return cleaned.strip("\"'\u2018\u2019\u201c\u201d").strip()


def bridge_parameters(
    *,
    tenant_id: str,
    user_id: str,
    bridges: list[str],
    scope_fragments: list[JsonValue],
    limit: int,
) -> CypherParameters:
    """Parameters for one scope-pinned ``BRIDGE_MENTION_CYPHER`` read."""
    return {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "bridges": list(bridges),
        "scope_fragments": scope_fragments,
        "limit": limit,
    }


def _messages(
    query: str, evidence: list[str]
) -> tuple[ChatCompletionMessageParam, ...]:
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": _NAMER_GUIDE,
    }
    rendered = "\n".join(evidence[:MAX_EVIDENCE_LINES])
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": f"Question: {query}\n\nRetrieved memory facts:\n{rendered}",
    }
    return (system_message, user_message)
