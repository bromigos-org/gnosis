"""EMem-style memory-unit extraction at conversation ingest.

With ``GNOSIS_FACT_EXTRACTION_ENABLED`` on, one structured-output LLM call per
add decomposes the NEW turns into short, self-contained, entity-normalized,
absolutely-dated statements (enriched event units, arXiv 2511.17208) that the
backend writes as ordinary long-term facts alongside - never instead of - the
verbatim ``said_*`` turn facts. Extraction is strictly additive: any transport,
model, or schema failure logs a structured warning and yields no units, so the
add succeeds exactly as a verbatim-only ingest.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import ClassVar, Final, Protocol

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gnosis.graph_query_qa import proxy_model_name

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

EXTRACTION_VERSION: Final[str] = "edu-v1"

_EXTRACTION_GUIDE: Final[str] = """
You are a memory extraction system for a long-term conversational memory
store. Given recent conversation context and one or more NEW turns, your
task is to decompose the NEW turns into memory units - short statements
that are minimal yet complete in meaning. Each unit expresses a single
fact, event, preference, plan, or proposition and is atomic (not easily
divisible further while still making sense). Preserve all substantive
information from the NEW turns - no detail should be lost.

Requirements:
1. Each unit must be a self-contained statement that can be understood
   independently, without reading any other unit or the conversation.
2. Never use pronouns or ambiguous references ("he", "it", "the event",
   "that place"). Use specific names, and consistently use the most
   informative name for each entity in all units. Resolve references to
   things said in CONTEXT turns by incorporating the specific details into
   the unit so it stands alone.
3. Extract only from the NEW turns. CONTEXT turns are for reference
   resolution only - do not re-extract information that appears solely in
   CONTEXT turns.
4. The units must collectively capture everything substantive in the NEW
   turns: facts, events, decisions, preferences, plans, concerns, personal
   attributes, relationships, and states - regardless of how minor a
   detail may seem. Do not extract conversational pleasantries, greetings,
   acknowledgements, or filler.
5. Resolve all relative time references ("yesterday", "next month", "last
   Saturday", "this quarter") to absolute dates or periods using the
   conversation date given in the header. Include the temporal context in
   the unit text where it is needed for the unit to stand alone.
6. For each unit, set event_date to the ISO date (YYYY-MM-DD) when the
   described event happened or will happen, if it is stated or can be
   resolved from the conversation date. If the unit is an ongoing state or
   preference, or no date is stated or resolvable, set event_date to null.
   Never invent dates. If only a month or year is mentioned, use its first
   day.
7. For each unit, list source_turn_ids: the turn numbers of the NEW turns
   the unit was extracted from (a unit may span several turns).
8. For each unit, list entities: the specific named entities mentioned in
   the unit (people, places, organizations, products, projects, works),
   using the same canonical names as in the unit text. Do not list dates,
   generic nouns, or feelings.
9. Speaker attribution matters: state who did, said, prefers, or plans
   each thing, using the speaker's name as given in the turn labels.
10. Chat-platform content: treat user mentions (e.g. "@name" or "<@123>")
    as references to those people and resolve them to names where
    possible. Ignore bot commands, emoji reactions, and formatting markup
    unless they carry substantive meaning. Summarize the substance of
    links, code blocks, or attachments only as described by the speakers.
11. Write units in the same language as the conversation.
12. If the NEW turns contain nothing substantive, return an empty list.

Return JSON only, in this exact format:
{"facts": [{"text": "...", "source_turn_ids": [1],
            "entities": ["..."], "event_date": "YYYY-MM-DD" | null}]}
""".strip()

# Entity-graph addendum (arXiv 2405.14831 HippoRAG / Graphiti OpenIE): when
# GNOSIS_ENTITY_GRAPH_ENABLED is on the extractor also emits explicit
# subject-relation-object triples so the entities of each unit become the
# nodes and relations the directed edges of a traversable knowledge graph.
# Appended to - never replacing - the base guide, so the flag-off prompt and
# structured-output schema stay byte-identical.
_RELATIONS_GUIDE_ADDENDUM: Final[str] = """
Additionally, for each unit populate a "relations" field: a list of directed
(head, relation, tail) triples capturing how the named entities in the unit
relate. head and tail must each be one of the unit's own entities (use the
same canonical names); relation is a short verb phrase describing the link
(for example "works at", "lives in", "married to", "presented at"). Emit a
triple only when the unit states a concrete relationship between two distinct
named entities, and never relate an entity to itself. Return an empty list
when the unit states no such relationship. So each fact object is:
{"text": "...", "source_turn_ids": [1], "entities": ["..."],
 "event_date": "YYYY-MM-DD" | null,
 "relations": [{"head": "...", "relation": "...", "tail": "..."}]}
""".strip()

_RELATIONAL_EXTRACTION_GUIDE: Final[str] = (
    f"{_EXTRACTION_GUIDE}\n\n{_RELATIONS_GUIDE_ADDENDUM}"
)

# One-shot exemplar (system-adjacent, as EMem does): a six-turn dated exchange
# demonstrating multi-turn units, relative-date resolution to an absolute
# month, entity canonicalization, an undated ongoing preference, and
# pleasantries producing no units.
_EXEMPLAR_INPUT: Final[str] = """
Conversation date: 2024-03-16
Speakers: Alice, Bob

CONTEXT turns (reference only, do not extract):
(none)

NEW turns (extract from these):
Turn 1: Alice: Hey Bob! Long time no see.
Turn 2: Bob: Good to see you too, Alice! How have you been?
Turn 3: Alice: Great - I just got back from Tokyo, where I presented our \
robotics work at the International Robotics Symposium.
Turn 4: Bob: That's fantastic. How did the talk go?
Turn 5: Alice: Really well. The symposium organizers invited me to give a \
keynote at its next edition, which happens next month in Osaka.
Turn 6: Bob: Amazing. You know I hate long flights, but for a keynote like \
that I would fly anywhere.
""".strip()

_EXEMPLAR_OUTPUT: Final[str] = (
    '{"facts": ['
    '{"text": "Alice presented her robotics work at the International '
    'Robotics Symposium in Tokyo.", "source_turn_ids": [3], '
    '"entities": ["Alice", "International Robotics Symposium", "Tokyo"], '
    '"event_date": null}, '
    '{"text": "Alice was invited by the International Robotics Symposium '
    "organizers to give a keynote at the symposium's next edition in Osaka "
    'in April 2024.", "source_turn_ids": [3, 5], '
    '"entities": ["Alice", "International Robotics Symposium", "Osaka"], '
    '"event_date": "2024-04-01"}, '
    '{"text": "Bob hates long flights.", "source_turn_ids": [6], '
    '"entities": ["Bob"], "event_date": null}'
    "]}"
)

# The exemplar output for the entity-graph path: identical units, each also
# carrying its extracted (head, relation, tail) triples - the third unit
# demonstrates the empty list when a unit names only one entity.
_RELATIONAL_EXEMPLAR_OUTPUT: Final[str] = (
    '{"facts": ['
    '{"text": "Alice presented her robotics work at the International '
    'Robotics Symposium in Tokyo.", "source_turn_ids": [3], '
    '"entities": ["Alice", "International Robotics Symposium", "Tokyo"], '
    '"event_date": null, "relations": [{"head": "Alice", '
    '"relation": "presented at", '
    '"tail": "International Robotics Symposium"}]}, '
    '{"text": "Alice was invited by the International Robotics Symposium '
    "organizers to give a keynote at the symposium's next edition in Osaka "
    'in April 2024.", "source_turn_ids": [3, 5], '
    '"entities": ["Alice", "International Robotics Symposium", "Osaka"], '
    '"event_date": "2024-04-01", "relations": [{"head": "Alice", '
    '"relation": "invited by", '
    '"tail": "International Robotics Symposium"}]}, '
    '{"text": "Bob hates long flights.", "source_turn_ids": [6], '
    '"entities": ["Bob"], "event_date": null, "relations": []}'
    "]}"
)

# Extraction emits a handful of short units (~25 words each) plus JSON
# framing; this cap bounds runaway completions without truncating a normal
# turn-pair extraction.
_MAX_COMPLETION_TOKENS: Final[int] = 2000


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One rendered prompt turn: a speaker label and its verbatim content."""

    speaker: str
    content: str


class MemoryUnit(BaseModel):
    """One extracted unit: a self-contained dated statement with provenance."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    text: str = ""
    source_turn_ids: list[int] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    event_date: str | None = None


class FactRelation(BaseModel):
    """One directed knowledge-graph triple extracted from a memory unit."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    head: str = ""
    relation: str = ""
    tail: str = ""


class RelationalMemoryUnit(MemoryUnit):
    """A memory unit that also carries its extracted entity triples.

    A subclass of ``MemoryUnit`` (so every relational unit *is* a memory unit
    for validation and storage) with one additional ``relations`` field. Used
    only on the GNOSIS_ENTITY_GRAPH_ENABLED path; the base unit keeps the
    flag-off structured-output schema byte-identical.
    """

    relations: list[FactRelation] = Field(default_factory=list)


class MemoryUnitExtraction(BaseModel):
    """Structured extractor output: the unit list for one add request."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    facts: list[MemoryUnit] = Field(default_factory=list)


class RelationalMemoryUnitExtraction(BaseModel):
    """Entity-graph structured extractor output: units carrying triples."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    facts: list[RelationalMemoryUnit] = Field(default_factory=list)


def unit_relations(unit: MemoryUnit) -> tuple[FactRelation, ...]:
    """Recover a unit's extracted triples, empty when none were emitted."""
    if isinstance(unit, RelationalMemoryUnit):
        return tuple(unit.relations)
    return ()


class MemoryUnitExtractor(Protocol):
    async def extract_units(
        self,
        *,
        conversation_date: str,
        context_turns: Sequence[ConversationTurn],
        new_turns: Sequence[ConversationTurn],
    ) -> MemoryUnitExtraction | None: ...


@dataclass(frozen=True, slots=True)
class LiteLLMMemoryUnitExtractor:
    model: str
    base_url: str
    api_key: str
    # On the GNOSIS_ENTITY_GRAPH_ENABLED path the extractor also emits
    # (head, relation, tail) triples; off, the prompt and structured-output
    # schema are byte-identical to the verbatim edu-v1 extractor.
    emit_relations: bool = False

    async def extract_units(
        self,
        *,
        conversation_date: str,
        context_turns: Sequence[ConversationTurn],
        new_turns: Sequence[ConversationTurn],
    ) -> MemoryUnitExtraction | None:
        start = time.perf_counter()
        response_format: type[MemoryUnitExtraction | RelationalMemoryUnitExtraction] = (
            RelationalMemoryUnitExtraction
            if self.emit_relations
            else MemoryUnitExtraction
        )
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            response = await client.beta.chat.completions.parse(
                messages=extraction_messages(
                    conversation_date=conversation_date,
                    context_turns=context_turns,
                    new_turns=new_turns,
                    emit_relations=self.emit_relations,
                ),
                model=proxy_model_name(self.model),
                # gpt-5.x endpoints reject `temperature` and `max_tokens`, so
                # this call sends neither and caps via max_completion_tokens.
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                response_format=response_format,
            )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            _LOGGER.info(
                "fact extraction returned no content",
                extra={"model": self.model},
            )
            return None
        # Relational units are MemoryUnit subclasses, so narrowing to the base
        # extraction keeps the ``relations`` payload (recovered downstream via
        # ``unit_relations``) while the extractor honors its Protocol return.
        extraction = (
            MemoryUnitExtraction(facts=list(parsed.facts))
            if isinstance(parsed, RelationalMemoryUnitExtraction)
            else parsed
        )
        _LOGGER.info(
            "fact extraction produced units",
            extra={
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "model": self.model,
                "units_extracted": len(extraction.facts),
            },
        )
        return extraction


# How many times one add re-samples the extractor when the model emits
# malformed structured output. The chatgpt-routed gpt-5.5 sporadically appends
# trailing characters after the JSON document (~2-5% of LongMemEval-sized
# adds, observed 2026-07-04); a fresh sample almost always parses, so
# re-sampling preserves the extracted units instead of degrading the add to
# verbatim-only - and previously the ValidationError escaped the additive
# guarantee entirely and 500'd the add.
_EXTRACTION_PARSE_ATTEMPTS: Final[int] = 3


async def _extraction_with_reparse(
    extractor: MemoryUnitExtractor,
    *,
    conversation_date: str,
    context_turns: Sequence[ConversationTurn],
    new_turns: Sequence[ConversationTurn],
) -> MemoryUnitExtraction | None:
    """One extraction, re-sampled on malformed structured output."""
    for attempt in range(1, _EXTRACTION_PARSE_ATTEMPTS + 1):
        try:
            return await extractor.extract_units(
                conversation_date=conversation_date,
                context_turns=context_turns,
                new_turns=new_turns,
            )
        except ValidationError:
            if attempt == _EXTRACTION_PARSE_ATTEMPTS:
                raise
            _LOGGER.warning(
                "fact extraction emitted malformed JSON; re-sampling",
                extra={"attempt": attempt, "new_turns": len(new_turns)},
            )
    return None


async def extract_memory_units(
    extractor: MemoryUnitExtractor,
    *,
    conversation_date: str,
    context_turns: Sequence[ConversationTurn],
    new_turns: Sequence[ConversationTurn],
) -> list[MemoryUnit]:
    """Extract validated memory units from the NEW turns of one add request.

    Invalid units (empty text, ``source_turn_ids`` outside the NEW turn
    numbers, an unparseable ``event_date``) are dropped individually - never
    the batch. Every failure mode - transport or model errors and an empty
    parse - degrades to no units, so extraction can never fail the add.
    """
    if not new_turns:
        return []
    try:
        extraction = await _extraction_with_reparse(
            extractor,
            conversation_date=conversation_date,
            context_turns=context_turns,
            new_turns=new_turns,
        )
    except (RuntimeError, OSError, OpenAIError, ValidationError) as error:
        _LOGGER.warning(
            "fact extraction failed; keeping verbatim-only add",
            extra={
                "error_type": type(error).__name__,
                "new_turns": len(new_turns),
            },
        )
        return []
    if extraction is None:
        _LOGGER.warning(
            "fact extraction returned no units; keeping verbatim-only add",
            extra={"new_turns": len(new_turns)},
        )
        return []
    new_turn_ids = _new_turn_ids(len(context_turns), len(new_turns))
    units = [unit for unit in extraction.facts if _is_valid_unit(unit, new_turn_ids)]
    if len(units) < len(extraction.facts):
        _LOGGER.warning(
            "fact extraction dropped invalid units",
            extra={
                "dropped": len(extraction.facts) - len(units),
                "units_extracted": len(extraction.facts),
            },
        )
    return units


def extraction_messages(
    *,
    conversation_date: str,
    context_turns: Sequence[ConversationTurn],
    new_turns: Sequence[ConversationTurn],
    emit_relations: bool = False,
) -> tuple[ChatCompletionMessageParam, ...]:
    """Build the edu-v1 prompt: guide, one-shot exemplar, and the request.

    Turn numbering is continuous across CONTEXT and NEW turns so
    ``source_turn_ids`` are unambiguous; only NEW turn numbers are valid. With
    ``emit_relations`` the guide and exemplar additionally ask for entity
    triples; without it the prompt is byte-identical to the verbatim path.
    """
    guide = _RELATIONAL_EXTRACTION_GUIDE if emit_relations else _EXTRACTION_GUIDE
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": guide,
    }
    exemplar_input: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": _EXEMPLAR_INPUT,
    }
    exemplar_output: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": _RELATIONAL_EXEMPLAR_OUTPUT if emit_relations else _EXEMPLAR_OUTPUT,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": _request_text(
            conversation_date=conversation_date,
            context_turns=context_turns,
            new_turns=new_turns,
        ),
    }
    return (system_message, exemplar_input, exemplar_output, user_message)


def _request_text(
    *,
    conversation_date: str,
    context_turns: Sequence[ConversationTurn],
    new_turns: Sequence[ConversationTurn],
) -> str:
    speakers = ", ".join(
        dict.fromkeys(turn.speaker for turn in (*context_turns, *new_turns)),
    )
    context_block = (
        _rendered_turns(context_turns, start=1) if context_turns else "(none)"
    )
    new_block = _rendered_turns(new_turns, start=len(context_turns) + 1)
    return (
        f"Conversation date: {conversation_date}\n"
        f"Speakers: {speakers}\n"
        "\n"
        "CONTEXT turns (reference only, do not extract):\n"
        f"{context_block}\n"
        "\n"
        "NEW turns (extract from these):\n"
        f"{new_block}"
    )


def _rendered_turns(turns: Sequence[ConversationTurn], *, start: int) -> str:
    return "\n".join(
        f"Turn {number}: {turn.speaker}: {turn.content}"
        for number, turn in enumerate(turns, start=start)
    )


def _new_turn_ids(context_count: int, new_count: int) -> frozenset[int]:
    return frozenset(range(context_count + 1, context_count + new_count + 1))


def _is_valid_unit(unit: MemoryUnit, new_turn_ids: frozenset[int]) -> bool:
    if not unit.text.strip():
        return False
    if any(turn_id not in new_turn_ids for turn_id in unit.source_turn_ids):
        return False
    return _parses_as_event_date(unit.event_date)


def _parses_as_event_date(event_date: str | None) -> bool:
    if event_date is None:
        return True
    try:
        _ = date.fromisoformat(event_date)
    except ValueError:
        return False
    return True
