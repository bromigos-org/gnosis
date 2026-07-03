from datetime import UTC, datetime

import pytest

from gnosis.memory_filters import (
    FilterCondition,
    FilterGroup,
    FilterNegation,
    FilterValidationError,
    MemoryFilterFields,
    build_cypher_filter,
    matches_filters,
    parse_filters,
)
from gnosis.models import JsonObject


def test_parse_filters_when_leaf_uses_implicit_eq() -> None:
    # Given: a mem0-style leaf filter without an explicit operator.
    payload: JsonObject = {"user_id": "789"}

    # When: the DSL is parsed.
    parsed = parse_filters(payload)

    # Then: the leaf becomes an eq condition.
    assert parsed == FilterCondition(field_name="user_id", operator="eq", value="789")


def test_parse_filters_when_leaf_has_multiple_operators() -> None:
    # Given: one field constrained by a range of operators.
    payload: JsonObject = {
        "created_at": {
            "gte": "2026-01-01T00:00:00+00:00",
            "lt": "2026-02-01T00:00:00+00:00",
        },
    }

    # When: the DSL is parsed.
    parsed = parse_filters(payload)

    # Then: the operators combine into an AND group.
    assert isinstance(parsed, FilterGroup)
    assert parsed.combinator == "AND"
    operators = {
        clause.operator
        for clause in parsed.clauses
        if isinstance(clause, FilterCondition)
    }
    assert operators == {"gte", "lt"}


def test_parse_filters_when_logical_operators_nest() -> None:
    # Given: nested AND/OR/NOT logical structure.
    payload: JsonObject = {
        "AND": [
            {"user_id": "789"},
            {
                "OR": [
                    {"metadata.topic": {"in": ["snacks", "games"]}},
                    {"NOT": {"agent_id": "operator"}},
                ],
            },
        ],
    }

    # When: the DSL is parsed.
    parsed = parse_filters(payload)

    # Then: the tree mirrors the logical structure.
    assert isinstance(parsed, FilterGroup)
    assert parsed.combinator == "AND"
    inner = parsed.clauses[1]
    assert isinstance(inner, FilterGroup)
    assert inner.combinator == "OR"
    assert isinstance(inner.clauses[1], FilterNegation)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"AND": []},
        {"AND": [{"user_id": "789"}], "user_id": "789"},
        {"NOT": [{"user_id": "789"}]},
        {"OR": "user_id"},
        {"session_id": "guild:1"},
        {"metadata.": "x"},
        {"user_id": {"like": "78%"}},
        {"user_id": {}},
        {"user_id": {"in": []}},
        {"user_id": {"in": [{"nested": "no"}]}},
        {"user_id": {"eq": ["789"]}},
        {"metadata.topic": {"contains": ""}},
        {"metadata.score": {"gte": "high"}},
        {"created_at": {"contains": "2026"}},
        {"created_at": {"gte": "not-a-date"}},
        {"created_at": {"gte": 1719878400}},
    ],
)
def test_parse_filters_when_payload_is_invalid(payload: JsonObject) -> None:
    # Given: an unknown field, unknown operator, or malformed operand.
    # When / Then: the parser rejects the DSL with a validation error.
    with pytest.raises(FilterValidationError):
        _ = parse_filters(payload)


def test_build_cypher_filter_when_conditions_are_positive() -> None:
    # Given: an AND filter over metadata tags and a created_at range.
    parsed = parse_filters(
        {
            "AND": [
                {"user_id": "789"},
                {"metadata.topic": {"in": ["snacks", "games"]}},
                {"created_at": {"gte": "2026-01-01T00:00:00+00:00"}},
            ],
        },
    )

    # When: the filter is translated for Cypher narrowing.
    narrowing = build_cypher_filter(parsed)

    # Then: every value is parameterized and fragments follow the JSON layout.
    assert narrowing.fragment == (
        "(f.metadata CONTAINS $filter_0"
        " AND any(fragment IN $filter_1 WHERE f.metadata CONTAINS fragment)"
        " AND f.created_at >= datetime($filter_2))"
    )
    assert narrowing.parameters == {
        "filter_0": '"user_id": "789"',
        "filter_1": ['"topic": "snacks"', '"topic": "games"'],
        "filter_2": "2026-01-01T00:00:00+00:00",
    }


def test_build_cypher_filter_when_negation_is_not_exactly_translatable() -> None:
    # Given: a negated metadata condition that only the gateway can evaluate.
    parsed = parse_filters({"NOT": {"metadata.topic": "snacks"}})

    # When: the filter is translated for Cypher narrowing.
    narrowing = build_cypher_filter(parsed)

    # Then: the narrowing stays neutral so no true match is excluded.
    assert narrowing.fragment == "true"
    assert narrowing.parameters == {}


def test_build_cypher_filter_when_negation_covers_created_at_only() -> None:
    # Given: a negated created_at condition, which translates exactly.
    parsed = parse_filters({"NOT": {"created_at": {"lt": "2026-01-01T00:00:00+00:00"}}})

    # When: the filter is translated for Cypher narrowing.
    narrowing = build_cypher_filter(parsed)

    # Then: the negation is preserved with a parameterized comparison.
    assert narrowing.fragment == "NOT (f.created_at < datetime($filter_0))"
    assert narrowing.parameters == {"filter_0": "2026-01-01T00:00:00+00:00"}


def test_build_cypher_filter_when_or_branch_cannot_narrow() -> None:
    # Given: an OR whose branch has no safe Cypher translation.
    parsed = parse_filters(
        {"OR": [{"user_id": "789"}, {"metadata.score": {"gte": 3}}]},
    )

    # When: the filter is translated for Cypher narrowing.
    narrowing = build_cypher_filter(parsed)

    # Then: the whole OR stays neutral instead of dropping the branch.
    assert narrowing.fragment == "true"
    assert narrowing.parameters == {}


def test_matches_filters_when_metadata_and_scope_fields_combine() -> None:
    # Given: a record with scope tags and custom metadata.
    fields = MemoryFilterFields(
        user_id="789",
        agent_id="pc-principal",
        created_at=datetime(2026, 6, 27, 1, 2, 3, tzinfo=UTC),
        metadata={"topic": "snacks", "score": 4},
    )
    parsed = parse_filters(
        {
            "AND": [
                {"user_id": "789"},
                {"metadata.topic": {"icontains": "SNACK"}},
                {"metadata.score": {"gt": 3}},
                {"created_at": {"gte": "2026-01-01T00:00:00+00:00"}},
                {"NOT": {"agent_id": "operator"}},
            ],
        },
    )

    # When / Then: exact evaluation accepts the record.
    assert matches_filters(parsed, fields) is True


def test_matches_filters_when_or_and_not_reject_the_record() -> None:
    # Given: a record that fails both OR branches.
    fields = MemoryFilterFields(
        user_id="789",
        agent_id="pc-principal",
        created_at=None,
        metadata={"topic": "games"},
    )
    parsed = parse_filters(
        {"OR": [{"metadata.topic": "snacks"}, {"NOT": {"user_id": "789"}}]},
    )

    # When / Then: exact evaluation rejects the record.
    assert matches_filters(parsed, fields) is False
