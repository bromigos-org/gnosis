"""Tests for GRAVITY-style entity-grouped context rendering."""

import json

from gnosis.context_assembly import (
    entity_group_key,
    entity_grouped_context_lines,
    fact_context_line,
)
from gnosis.models import JsonObject


def _fact(subject: str, obj: str, *, date: str = "2023-05-07") -> JsonObject:
    return {
        "id": f"id-{subject}-{obj[:8]}",
        "subject": subject,
        "predicate": "fact",
        "object": obj,
        "confidence": 1.0,
        "created_at": f"{date}T00:00:00Z",
        "metadata": json.dumps({"event_date": date}),
    }


def test_entity_group_key_skips_internal_subjects() -> None:
    assert entity_group_key(_fact("Caroline", "enjoys music")) == "Caroline"
    assert entity_group_key(_fact("tenant:bromigos:message:1", "x")) is None


def test_entity_grouped_context_lines_groups_by_subject() -> None:
    facts = [
        _fact("Melanie", "enjoys running", date="2023-05-01"),
        _fact("Caroline", "went to pride", date="2023-05-02"),
        _fact("Melanie", "does pottery", date="2023-05-03"),
    ]
    lines = entity_grouped_context_lines(
        facts,
        query="What activities does Melanie partake in?",
        line_for=fact_context_line,
    )
    text = "\n".join(lines)
    assert "#### Melanie" in text
    assert "#### Caroline" in text
    assert text.index("#### Melanie") < text.index("#### Caroline")
    assert "enjoys running" in text
    assert "does pottery" in text


def test_entity_grouped_context_lines_unchanged_when_flat() -> None:
    facts = [_fact("Caroline", "enjoys music")]
    flat = ["### Long-Term Facts", fact_context_line(facts[0])]
    grouped = entity_grouped_context_lines(
        facts,
        query="",
        line_for=fact_context_line,
    )
    assert grouped == ["### Long-Term Facts", "#### Caroline", flat[1]]
