import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import httpx
import pytest
from openai import APIConnectionError

from gnosis.fact_extraction import (
    ConversationTurn,
    FactRelation,
    MemoryUnit,
    MemoryUnitExtraction,
    RelationalMemoryUnit,
    RelationalMemoryUnitExtraction,
    extract_memory_units,
    extraction_messages,
    unit_relations,
)

_CONTEXT_TURNS = (
    ConversationTurn(speaker="user", content="I went to Tokyo last week"),
    ConversationTurn(speaker="assistant", content="How was the trip?"),
)
_NEW_TURNS = (
    ConversationTurn(speaker="user", content="Great, I presented at the symposium"),
    ConversationTurn(speaker="assistant", content="Congratulations!"),
)


@pytest.mark.anyio
async def test_extract_memory_units_keeps_valid_units() -> None:
    # Given: an extractor producing well-formed units for two new turns.
    unit = MemoryUnit(
        text="Caroline presented at the Tokyo symposium",
        source_turn_ids=[3, 4],
        entities=["Caroline", "Tokyo"],
        event_date="2023-05-07",
    )
    extractor = RecordingExtractor(extraction=MemoryUnitExtraction(facts=[unit]))

    # When: units are extracted.
    units = await extract_memory_units(
        extractor,
        conversation_date="2023-05-07",
        context_turns=_CONTEXT_TURNS,
        new_turns=_NEW_TURNS,
    )

    # Then: the unit survives validation unchanged.
    assert units == [unit]


@pytest.mark.anyio
async def test_extract_memory_units_drops_invalid_units_never_the_batch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: one valid unit among units with empty text, out-of-range
    # source turn ids (context turns are not extractable), and a
    # non-ISO event date.
    valid = MemoryUnit(text="Caroline presented at the symposium", source_turn_ids=[3])
    extractor = RecordingExtractor(
        extraction=MemoryUnitExtraction(
            facts=[
                valid,
                MemoryUnit(text="   ", source_turn_ids=[3]),
                MemoryUnit(text="from a context turn", source_turn_ids=[1]),
                MemoryUnit(text="beyond the new turns", source_turn_ids=[5]),
                MemoryUnit(text="badly dated", source_turn_ids=[4], event_date="May 7"),
            ],
        ),
    )

    # When: units are extracted.
    with caplog.at_level(logging.WARNING, logger="gnosis.fact_extraction"):
        units = await extract_memory_units(
            extractor,
            conversation_date="2023-05-07",
            context_turns=_CONTEXT_TURNS,
            new_turns=_NEW_TURNS,
        )

    # Then: only the valid unit survives and the drops are logged.
    assert units == [valid]
    assert "dropped invalid units" in caplog.text


@pytest.mark.anyio
async def test_extract_memory_units_llm_failure_returns_no_units(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: an extractor failing at the transport.
    extractor = FailingExtractor()

    # When: units are extracted.
    with caplog.at_level(logging.WARNING, logger="gnosis.fact_extraction"):
        units = await extract_memory_units(
            extractor,
            conversation_date="2023-05-07",
            context_turns=(),
            new_turns=_NEW_TURNS,
        )

    # Then: the failure degrades to no units with a structured warning.
    assert units == []
    assert "fact extraction failed" in caplog.text


@pytest.mark.anyio
async def test_extract_memory_units_missing_parse_returns_no_units(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: an extractor whose structured output produced no content.
    extractor = RecordingExtractor(extraction=None)

    # When: units are extracted.
    with caplog.at_level(logging.WARNING, logger="gnosis.fact_extraction"):
        units = await extract_memory_units(
            extractor,
            conversation_date="2023-05-07",
            context_turns=(),
            new_turns=_NEW_TURNS,
        )

    # Then: the miss degrades to no units.
    assert units == []
    assert "returned no units" in caplog.text


@pytest.mark.anyio
async def test_extract_memory_units_without_new_turns_skips_the_call() -> None:
    # Given: nothing new to extract from.
    extractor = RecordingExtractor(extraction=MemoryUnitExtraction())

    # When: units are extracted with no new turns.
    units = await extract_memory_units(
        extractor,
        conversation_date="2023-05-07",
        context_turns=_CONTEXT_TURNS,
        new_turns=(),
    )

    # Then: the extractor is never consulted.
    assert units == []
    assert extractor.calls == 0


def test_extraction_messages_numbers_turns_continuously() -> None:
    # Given: two context turns preceding two new turns.
    messages = extraction_messages(
        conversation_date="2023-05-07",
        context_turns=_CONTEXT_TURNS,
        new_turns=_NEW_TURNS,
    )

    # Then: the request renders the spec's user template with continuous
    # numbering across CONTEXT and NEW, so source_turn_ids are unambiguous.
    request = _message(messages[-1])
    assert request["role"] == "user"
    assert request["content"] == (
        "Conversation date: 2023-05-07\n"
        "Speakers: user, assistant\n"
        "\n"
        "CONTEXT turns (reference only, do not extract):\n"
        "Turn 1: user: I went to Tokyo last week\n"
        "Turn 2: assistant: How was the trip?\n"
        "\n"
        "NEW turns (extract from these):\n"
        "Turn 3: user: Great, I presented at the symposium\n"
        "Turn 4: assistant: Congratulations!"
    )


def test_extraction_messages_render_empty_context_as_none() -> None:
    # Given: a first turn with no session history.
    messages = extraction_messages(
        conversation_date="2023-05-07",
        context_turns=(),
        new_turns=_NEW_TURNS[:1],
    )

    # Then: the context block renders the literal placeholder and new turn
    # numbering starts at one.
    content = _message(messages[-1])["content"]
    assert isinstance(content, str)
    assert "CONTEXT turns (reference only, do not extract):\n(none)\n" in content
    assert "Turn 1: user: Great, I presented at the symposium" in content


def test_extraction_messages_carry_guide_and_one_shot_exemplar() -> None:
    # Given: the assembled prompt.
    messages = extraction_messages(
        conversation_date="2023-05-07",
        context_turns=(),
        new_turns=_NEW_TURNS,
    )

    # Then: the system guide leads and a user/assistant exemplar pair sits
    # between the guide and the request.
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    guide = _message(messages[0])["content"]
    assert isinstance(guide, str)
    assert "decompose the NEW turns into memory units" in guide
    assert "Never invent dates" in guide

    # Then: the exemplar output is itself a valid extraction demonstrating a
    # multi-turn unit, a resolved relative month, and an undated preference.
    exemplar_output = _message(messages[2])["content"]
    assert isinstance(exemplar_output, str)
    exemplar = MemoryUnitExtraction.model_validate_json(exemplar_output)
    assert [unit.source_turn_ids for unit in exemplar.facts] == [[3], [3, 5], [6]]
    assert [unit.event_date for unit in exemplar.facts] == [None, "2024-04-01", None]


def test_extraction_messages_omit_relations_by_default() -> None:
    # Given: the default (entity-graph-off) prompt.
    messages = extraction_messages(
        conversation_date="2023-05-07",
        context_turns=(),
        new_turns=_NEW_TURNS,
    )

    # Then: neither the guide nor the exemplar mentions relations, keeping the
    # verbatim edu-v1 prompt byte-identical.
    guide = _message(messages[0])["content"]
    exemplar_output = _message(messages[2])["content"]
    assert isinstance(guide, str)
    assert isinstance(exemplar_output, str)
    assert '"relations"' not in guide
    assert '"relations"' not in exemplar_output


def test_extraction_messages_request_relations_when_enabled() -> None:
    # Given: the entity-graph prompt.
    messages = extraction_messages(
        conversation_date="2023-05-07",
        context_turns=(),
        new_turns=_NEW_TURNS,
        emit_relations=True,
    )

    # Then: the guide asks for (head, relation, tail) triples and the exemplar
    # output is a valid relational extraction carrying them.
    guide = _message(messages[0])["content"]
    assert isinstance(guide, str)
    assert "(head, relation, tail)" in guide
    exemplar_output = _message(messages[2])["content"]
    assert isinstance(exemplar_output, str)
    exemplar = RelationalMemoryUnitExtraction.model_validate_json(exemplar_output)
    assert exemplar.facts[0].relations == [
        FactRelation(
            head="Alice",
            relation="presented at",
            tail="International Robotics Symposium",
        ),
    ]
    # The undated single-entity unit states no relationship.
    assert exemplar.facts[2].relations == []


def test_unit_relations_recovers_triples_for_relational_units_only() -> None:
    # Then: relations surface for relational units and are empty for base ones.
    relational = RelationalMemoryUnit(
        text="Alice works at Acme",
        entities=["Alice", "Acme"],
        relations=[FactRelation(head="Alice", relation="works at", tail="Acme")],
    )
    assert unit_relations(relational) == (
        FactRelation(head="Alice", relation="works at", tail="Acme"),
    )
    assert unit_relations(MemoryUnit(text="plain", entities=["Alice"])) == ()


def _message(message: object) -> dict[str, str]:
    """View one prompt message as a plain dict for assertion purposes."""
    return cast("dict[str, str]", message)


@dataclass(slots=True)
class RecordingExtractor:
    extraction: MemoryUnitExtraction | None = None
    calls: int = field(default=0)

    async def extract_units(
        self,
        *,
        conversation_date: str,
        context_turns: Sequence[ConversationTurn],
        new_turns: Sequence[ConversationTurn],
    ) -> MemoryUnitExtraction | None:
        _ = (conversation_date, context_turns, new_turns)
        self.calls += 1
        return self.extraction


@dataclass(frozen=True, slots=True)
class FailingExtractor:
    async def extract_units(
        self,
        *,
        conversation_date: str,
        context_turns: Sequence[ConversationTurn],
        new_turns: Sequence[ConversationTurn],
    ) -> MemoryUnitExtraction | None:
        _ = (conversation_date, context_turns, new_turns)
        raise APIConnectionError(request=httpx.Request("POST", "http://litellm.local"))
