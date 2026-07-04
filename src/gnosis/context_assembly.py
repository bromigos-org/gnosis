"""Long-term fact candidates: rendering, fusion, scope re-checks, budgets.

The read path assembles prompt-facing context from several candidate legs
(dense ranking, recency fallback, graph-QA fusion, entity/bridge traversal).
This module holds the pure pieces those legs share: deserializing rows and
stored memories into fact candidates, re-checking scope on every
deserialized fact, rendering the compact dated context lines, read-time
freshness signals for supersession, fusing and budget-cutting graph-derived
candidates, verbatim-expansion targeting, and the legacy ``/v1/context``
request/response adapters. Orchestration (which legs run, with which client)
stays on :class:`gnosis.backend.Neo4jAgentMemoryBackend`.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import ValidationError

from gnosis.json_redaction import (
    JSON_OBJECT_ADAPTER,
    metadata_from_json,
    redacted_text,
    string_metadata,
)
from gnosis.memory_provider import (
    EXTRACTED_FACT_PREDICATE,
    TURN_MEMORY_PREDICATE_PREFIX,
    VERBATIM_MEMORY_PREDICATE,
    StoredMemory,
)
from gnosis.models import (
    ContextRequest,
    ContextResponse,
    GraphContextResponse,
    JsonObject,
    JsonValue,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSection,
    MemoryRecord,
)
from gnosis.sdk_client import MemoryClientContext
from gnosis.settings import Settings
from gnosis.supersession import FactFreshness, slot_key

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

MEMORY_SEARCH_CANDIDATE_LIMIT: Final[int] = 100

# At most this fraction of the context item budget is reserved for
# graph-derived candidates, so dense relevance keeps the bulk of the slots.
_GRAPH_RESERVE_DIVISOR: Final[int] = 4

# Long-term memory is durable across conversations, so reads never narrow by
# session_id: pinning recall to the requesting session starves the context
# budget while /v1/memories/search over the same data spans sessions.
_FACT_READ_EXCLUDED_FIELDS = {
    "space_id",
    "session_id",
}

_FACT_SCOPE_FIELDS = {
    "tenant_id",
    "agent_id",
    "user_id",
    "visibility",
    "guild_id",
    "channel_id",
}

# Temporal anchor for rendered facts: a stored date tag wins over created_at.
_FACT_DATE_METADATA_KEYS = ("session_date", "date")


async def query_recent_facts(
    client: MemoryClientContext,
    scope_metadata: Mapping[str, JsonValue],
) -> list[JsonObject]:
    """Most recently written scoped facts, newest first.

    Scans a candidate pool sized like /v1/memories/search and re-checks scope
    on the deserialized rows so the item budget only sees in-scope facts.
    """
    params: JsonObject = {
        "metadata_fragments": metadata_fragments(scope_metadata),
        "candidate_limit": MEMORY_SEARCH_CANDIDATE_LIMIT,
    }
    rows = await client.query.cypher(
        """
        MATCH (f:Fact)
        WHERE f.metadata IS NOT NULL
          AND all(
            fragment IN $metadata_fragments WHERE f.metadata CONTAINS fragment
          )
        WITH f
        ORDER BY f.created_at DESC, f.subject ASC, f.predicate ASC, f.object ASC
        LIMIT $candidate_limit
        RETURN f{
            .id, .subject, .predicate, .object, .metadata,
            created_at: toString(f.created_at)
        } AS f
        """,
        params,
    )
    return [
        fact
        for row in rows
        if (fact := _fact_from_row(row)) is not None
        and fact_matches_scope(fact, scope_metadata)
    ]


def _fact_from_row(row: JsonObject) -> JsonObject | None:
    fact = row.get("f")
    if not isinstance(fact, dict):
        return None
    required_fields = ("subject", "predicate", "object")
    if all(isinstance(fact.get(field_name), str) for field_name in required_fields):
        return fact
    return None


def fact_from_memory(memory: StoredMemory) -> JsonObject:
    return {
        "id": memory.memory_id,
        "subject": memory.subject,
        "predicate": memory.predicate,
        "object": memory.content,
        "metadata": dict(memory.metadata),
        "created_at": memory.created_at,
    }


def fact_freshness(fact: JsonObject) -> FactFreshness:
    metadata = _fact_raw_metadata(fact)
    created_at = fact.get("created_at")
    return FactFreshness(
        slot_key=slot_key(
            str(fact.get("subject", "")),
            str(fact.get("predicate", "")),
            _metadata_entities(metadata),
        ),
        event_date=_metadata_event_date(metadata),
        created_at=created_at if isinstance(created_at, str) and created_at else None,
    )


def memory_freshness(memory: StoredMemory) -> FactFreshness:
    return FactFreshness(
        slot_key=slot_key(
            memory.subject,
            memory.predicate,
            _metadata_entities(memory.metadata),
        ),
        event_date=_metadata_event_date(memory.metadata),
        created_at=memory.created_at,
    )


def _fact_raw_metadata(fact: JsonObject) -> JsonObject:
    metadata = fact.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            return JSON_OBJECT_ADAPTER.validate_json(metadata)
        except ValidationError:
            return {}
    return {}


def verbatim_expansion_targets(
    facts: list[JsonObject],
    *,
    cap: int,
) -> list[tuple[str, list[str]]]:
    """Highest-ranked extracted facts to expand, with novel source turn ids.

    Source ids already present as ranked facts are dropped so a verbatim turn
    independently in the result set is never double-rendered; at most ``cap``
    facts (rank order) are returned.
    """
    present_ids = {
        fact_id
        for fact in facts
        if isinstance(fact_id := fact.get("id"), str) and fact_id
    }
    targets: list[tuple[str, list[str]]] = []
    for fact in facts:
        if len(targets) >= cap:
            break
        fact_id = fact.get("id")
        if str(fact.get("predicate")) != EXTRACTED_FACT_PREDICATE or not isinstance(
            fact_id,
            str,
        ):
            continue
        source_ids = [
            source_id
            for source_id in _fact_source_memory_ids(fact)
            if source_id not in present_ids
        ]
        if source_ids:
            targets.append((fact_id, source_ids))
    return targets


def _fact_source_memory_ids(fact: JsonObject) -> list[str]:
    """Provenance ids of the verbatim turns an extracted fact was derived from."""
    source_ids = _fact_raw_metadata(fact).get("source_memory_ids")
    if not isinstance(source_ids, list):
        return []
    return [
        source_id
        for source_id in source_ids
        if isinstance(source_id, str) and source_id
    ]


def _metadata_entities(metadata: Mapping[str, JsonValue]) -> list[str]:
    entities = metadata.get("entities")
    if not isinstance(entities, list):
        return []
    return [entity for entity in entities if isinstance(entity, str)]


def _metadata_event_date(metadata: Mapping[str, JsonValue]) -> str | None:
    event_date = metadata.get("event_date")
    if isinstance(event_date, str) and event_date:
        return event_date
    return None


def log_supersession(dropped: int, candidates: int, *, surface: str) -> None:
    if dropped == 0:
        return
    _LOGGER.info(
        "read-time supersession dropped superseded facts",
        extra={"surface": surface, "dropped": dropped, "candidates": candidates},
    )


def fact_context_line(fact: JsonObject) -> str:
    """Render one fact as a single dated line of prompt-facing content.

    Provenance stays out of the prompt: it crowds the answer model's
    attention and remains available through the audit read paths. Subject
    and predicate only render when they carry signal beyond conversation
    plumbing (verbatim ``memory`` and turn ``said_*`` predicates repeat the
    speaker, not knowledge).
    """
    predicate = str(fact["predicate"])
    content = redacted_text(str(fact["object"]))
    if predicate != VERBATIM_MEMORY_PREDICATE and not predicate.startswith(
        TURN_MEMORY_PREDICATE_PREFIX,
    ):
        subject = redacted_text(str(fact["subject"]))
        content = f"{subject} {redacted_text(predicate)}: {content}"
    fact_date = _fact_date(fact, fact_metadata(fact))
    if fact_date:
        return f"- [{redacted_text(fact_date)}] {content}"
    return f"- {content}"


def stored_memory_line(memory: StoredMemory) -> str:
    """Render one search candidate for the recall filter prompt."""
    return fact_context_line(fact_from_memory(memory))


def memory_record_line(record: MemoryRecord) -> str:
    """Render one already-public memory record for the recall filter prompt."""
    metadata = string_metadata(record.metadata)
    record_date = next(
        (metadata[key] for key in _FACT_DATE_METADATA_KEYS if metadata.get(key)),
        (record.created_at or "")[:10],
    )
    if record_date:
        return f"- [{record_date}] {record.content}"
    return f"- {record.content}"


def fact_matches_scope(
    fact: JsonObject,
    scope_metadata: Mapping[str, JsonValue],
) -> bool:
    metadata = fact_metadata(fact)
    requested_scope = {
        field_name: scope_value
        for field_name, scope_value in scope_metadata.items()
        if field_name in _FACT_SCOPE_FIELDS
    }
    return all(
        metadata.get(field_name) == requested_value
        for field_name, requested_value in requested_scope.items()
    ) and all(
        requested_scope.get(field_name) == fact_value
        for field_name, fact_value in metadata.items()
        if field_name in _FACT_SCOPE_FIELDS
    )


def fact_metadata(fact: JsonObject) -> dict[str, str]:
    metadata = fact.get("metadata")
    if isinstance(metadata, str):
        return metadata_from_json(metadata)
    if isinstance(metadata, dict):
        return string_metadata(metadata)
    return {}


def _fact_date(fact: JsonObject, metadata: Mapping[str, str]) -> str:
    for key in _FACT_DATE_METADATA_KEYS:
        value = metadata.get(key)
        if value:
            return value
    created_at = fact.get("created_at")
    if isinstance(created_at, str):
        # The YYYY-MM-DD prefix of the ISO timestamp.
        return created_at[:10]
    return ""


def fact_markers(facts: list[JsonObject]) -> set[str]:
    markers: set[str] = set()
    for fact in facts:
        for field_name in ("id", "subject", "object"):
            value = fact.get(field_name)
            if isinstance(value, str) and value != "":
                markers.add(value)
    return markers


def graph_facts_to_candidates(facts: Sequence[JsonObject]) -> list[JsonObject]:
    """Adapt graph-QA nodes to long-term fact candidates for fused ranking.

    Each live graph node's summary becomes a verbatim-predicate candidate so
    ``fact_context_line`` renders it as a uniform dated line and read-time
    supersession never claims a slot for it (graph summaries are traversal
    evidence, not knowledge slots).
    """
    candidates: list[JsonObject] = []
    for fact in facts:
        summary = fact.get("summary")
        if not isinstance(summary, str) or not summary or fact.get("deleted") is True:
            continue
        identifier = fact.get("id")
        candidates.append(
            {
                "id": identifier if isinstance(identifier, str) else "",
                "subject": "",
                "predicate": VERBATIM_MEMORY_PREDICATE,
                "object": summary,
                "metadata": {},
                "graphqa": True,
            },
        )
    return candidates


def cut_with_graph_reserve(
    facts: list[JsonObject],
    max_items: int,
) -> list[JsonObject]:
    """Apply the item budget while reserving tail slots for graph candidates.

    ``fuse_graph_facts`` appends graph-derived candidates after the dense
    ranking, so a plain ``facts[:max_items]`` cut silently dropped every one
    of them whenever dense retrieval filled the candidate pool (always, on a
    populated store) - the fusion leg ran but never rendered. Up to a quarter
    of the budget now goes to the highest-ranked graph candidates; dense
    candidates keep the rest, in ranking order. A pure passthrough cut when
    no graph candidate is present.
    """
    graph = [fact for fact in facts if fact.get("graphqa") is True]
    if not graph or len(facts) <= max_items:
        return facts[:max_items]
    dense = [fact for fact in facts if fact.get("graphqa") is not True]
    reserve = min(len(graph), max(1, max_items // _GRAPH_RESERVE_DIVISOR))
    return [*dense[: max_items - reserve], *graph[:reserve]]


def fuse_graph_facts(
    dense: list[JsonObject],
    graph: list[JsonObject],
) -> list[JsonObject]:
    """Union graph-derived candidates into the dense set, dropping duplicates.

    A graph candidate is dropped when it shares a memory id with a dense fact
    or renders to the same dated line, so a node already surfaced by dense
    retrieval is never double-added. Dense ranking order is preserved; graph
    candidates append after it (Mnemis unions the unordered structured route).
    """
    if not graph:
        return dense
    existing_ids = {
        identifier
        for fact in dense
        if isinstance(identifier := fact.get("id"), str) and identifier
    }
    existing_lines = {fact_context_line(fact) for fact in dense}
    fused = list(dense)
    for candidate in graph:
        identifier = candidate.get("id")
        if isinstance(identifier, str) and identifier and identifier in existing_ids:
            continue
        line = fact_context_line(candidate)
        if line in existing_lines:
            continue
        existing_lines.add(line)
        fused.append(candidate)
    return fused


def dedupe_graph_context(
    graph: GraphContextResponse,
    long_term_markers: set[str],
) -> GraphContextResponse:
    if not long_term_markers or not graph.facts:
        return graph

    facts = [
        fact
        for fact in graph.facts
        if not _graph_fact_matches_markers(fact, long_term_markers)
    ]
    if len(facts) == len(graph.facts):
        return graph

    return GraphContextResponse(
        context="\n".join(_graph_fact_summary(fact) for fact in facts),
        facts=facts,
    )


def _graph_fact_matches_markers(fact: JsonObject, markers: set[str]) -> bool:
    return any(
        isinstance(value, str) and value in markers
        for field_name in ("id", "summary")
        if (value := fact.get(field_name)) is not None
    )


def _graph_fact_summary(fact: JsonObject) -> str:
    summary = fact.get("summary")
    if isinstance(summary, str):
        return summary
    return ""


def metadata_fragments(metadata: Mapping[str, JsonValue]) -> list[JsonValue]:
    fragments: list[JsonValue] = []
    for key, value in metadata.items():
        if key not in _FACT_READ_EXCLUDED_FIELDS and isinstance(value, str):
            fragments.append(f'"{key}": {json.dumps(value)}')
    return fragments


def long_term_enrichment_enabled(settings: Settings) -> bool:
    return (
        settings.gnosis_prompt_entities_enabled
        or settings.gnosis_prompt_preferences_enabled
    )


def append_context_section(
    sections: list[MemoryContextSection],
    source: str,
    content: str,
) -> None:
    if content:
        sections.append(MemoryContextSection(source=source, content=content))


def legacy_context_request(request: ContextRequest) -> MemoryContextRequest:
    """Map a legacy /v1/context request onto the combined memory-context path."""
    return MemoryContextRequest(
        scope=request.scope,
        query=request.query,
        include_short_term=True,
        include_long_term=False,
        include_reasoning=False,
        include_graph=False,
        max_items=request.limit,
    )


def legacy_context_response(response: MemoryContextResponse) -> ContextResponse:
    """Reduce a combined memory-context response to the legacy short-term shape."""
    context = next(
        (
            section.content
            for section in response.sections
            if section.source == "short_term"
        ),
        "",
    )
    return ContextResponse(context=context)
