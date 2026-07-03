"""mem0-v2-style filter DSL for the memory provider surface.

The parser validates the JSON DSL into a typed tree, translates the tree into
safe parameterized Cypher narrowing fragments, and evaluates the exact filter
semantics in the gateway. Cypher narrowing follows the safety philosophy of
``graph_query_validation.py``: values are always bound as parameters and the
generated fragment may only shrink toward a superset of the true matches, so
the in-gateway evaluation stays authoritative.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Literal, TypeGuard, cast, final

from gnosis.models import JsonObject, JsonValue

type FilterOperator = Literal[
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "contains",
    "icontains",
]
type FilterCombinator = Literal["AND", "OR"]
type FilterScalar = str | int | float | bool

_LOGICAL_KEYS: Final[frozenset[str]] = frozenset({"AND", "OR", "NOT"})
_KNOWN_OPERATORS: Final[frozenset[str]] = frozenset(
    {"eq", "ne", "gt", "gte", "lt", "lte", "in", "contains", "icontains"},
)
_ORDER_OPERATORS: Final[frozenset[str]] = frozenset({"gt", "gte", "lt", "lte"})
_TEXT_OPERATORS: Final[frozenset[str]] = frozenset({"contains", "icontains"})
_KNOWN_FIELDS: Final[frozenset[str]] = frozenset({"user_id", "agent_id", "created_at"})
_METADATA_FIELD_PREFIX: Final[str] = "metadata."
_CREATED_AT_FIELD: Final[str] = "created_at"
_CREATED_AT_CYPHER_OPERATORS: Final[dict[str, str]] = {
    "eq": "=",
    "ne": "<>",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}


@final
class FilterValidationError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail: str = detail
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class FilterCondition:
    field_name: str
    operator: FilterOperator
    value: JsonValue


@dataclass(frozen=True, slots=True)
class FilterGroup:
    combinator: FilterCombinator
    clauses: tuple["MemoryFilter", ...]


@dataclass(frozen=True, slots=True)
class FilterNegation:
    clause: "MemoryFilter"


type MemoryFilter = FilterCondition | FilterGroup | FilterNegation


@dataclass(frozen=True, slots=True)
class CypherFilter:
    fragment: str
    parameters: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class MemoryFilterFields:
    user_id: str | None
    agent_id: str | None
    created_at: datetime | None
    metadata: JsonObject


def parse_filters(payload: JsonObject) -> MemoryFilter:
    if not payload:
        detail = "filters must not be empty"
        raise FilterValidationError(detail)
    logical_keys = payload.keys() & _LOGICAL_KEYS
    if logical_keys and payload.keys() - _LOGICAL_KEYS:
        detail = "filters must not mix logical operators with fields"
        raise FilterValidationError(detail)
    if logical_keys:
        return _parse_logical(payload)
    return _all_of([_parse_field(field, value) for field, value in payload.items()])


def build_cypher_filter(
    filters: MemoryFilter | None,
    *,
    alias: str = "f",
    parameter_prefix: str = "filter",
) -> CypherFilter:
    parameters: dict[str, JsonValue] = {}
    fragment = None
    if filters is not None:
        fragment = _narrowing_fragment(
            filters,
            alias=alias,
            parameter_prefix=parameter_prefix,
            parameters=parameters,
            exact=False,
        )
    if fragment is None:
        return CypherFilter(fragment="true", parameters={})
    return CypherFilter(fragment=fragment, parameters=parameters)


def matches_filters(filters: MemoryFilter | None, fields: MemoryFilterFields) -> bool:
    if filters is None:
        return True
    match filters:
        case FilterCondition():
            return _condition_matches(filters, fields)
        case FilterGroup(combinator="AND", clauses=clauses):
            return all(matches_filters(clause, fields) for clause in clauses)
        case FilterGroup(clauses=clauses):
            return any(matches_filters(clause, fields) for clause in clauses)
        case FilterNegation(clause=clause):
            return not matches_filters(clause, fields)


def _parse_logical(payload: JsonObject) -> MemoryFilter:
    if len(payload) != 1:
        detail = "filters must use exactly one logical operator per object"
        raise FilterValidationError(detail)
    key, value = next(iter(payload.items()))
    if key == "NOT":
        if not isinstance(value, dict):
            detail = "NOT expects a single filter object"
            raise FilterValidationError(detail)
        return FilterNegation(clause=parse_filters(value))
    if not isinstance(value, list) or not value:
        detail = f"{key} expects a non-empty list of filters"
        raise FilterValidationError(detail)
    clauses: list[MemoryFilter] = []
    for clause in value:
        if not isinstance(clause, dict):
            detail = f"{key} clauses must be filter objects"
            raise FilterValidationError(detail)
        clauses.append(parse_filters(clause))
    if key == "AND":
        return _all_of(clauses)
    return FilterGroup(combinator="OR", clauses=tuple(clauses))


def _all_of(clauses: list[MemoryFilter]) -> MemoryFilter:
    if len(clauses) == 1:
        return clauses[0]
    return FilterGroup(combinator="AND", clauses=tuple(clauses))


def _parse_field(field_name: str, value: JsonValue) -> MemoryFilter:
    _require_known_field(field_name)
    if isinstance(value, dict):
        if not value:
            detail = f"{field_name} has no operators"
            raise FilterValidationError(detail)
        return _all_of(
            [
                _condition(field_name, operator, operand)
                for operator, operand in value.items()
            ],
        )
    return _condition(field_name, "eq", value)


def _require_known_field(field_name: str) -> None:
    if field_name in _KNOWN_FIELDS:
        return
    if (
        field_name.startswith(_METADATA_FIELD_PREFIX)
        and field_name != _METADATA_FIELD_PREFIX
    ):
        return
    detail = f"unknown filter field: {field_name}"
    raise FilterValidationError(detail)


def _condition(field_name: str, operator: str, value: JsonValue) -> FilterCondition:
    if operator not in _KNOWN_OPERATORS:
        detail = f"unknown filter operator: {operator}"
        raise FilterValidationError(detail)
    validated = cast("FilterOperator", operator)
    if field_name == _CREATED_AT_FIELD:
        _require_created_at_operand(operator, value)
    else:
        _require_metadata_operand(field_name, operator, value)
    return FilterCondition(field_name=field_name, operator=validated, value=value)


def _require_created_at_operand(operator: str, value: JsonValue) -> None:
    if operator in _TEXT_OPERATORS:
        detail = f"created_at does not support the {operator} operator"
        raise FilterValidationError(detail)
    if operator == "in":
        for item in _require_scalar_list(_CREATED_AT_FIELD, value):
            _ = _require_timestamp(item)
        return
    _ = _require_timestamp(value)


def _require_metadata_operand(
    field_name: str,
    operator: str,
    value: JsonValue,
) -> None:
    if operator == "in":
        _ = _require_scalar_list(field_name, value)
        return
    if operator in _TEXT_OPERATORS:
        if not isinstance(value, str) or not value:
            detail = f"{field_name} {operator} expects a non-empty string"
            raise FilterValidationError(detail)
        return
    if operator in _ORDER_OPERATORS:
        if isinstance(value, bool) or not isinstance(value, int | float):
            detail = f"{field_name} {operator} expects a number"
            raise FilterValidationError(detail)
        return
    if not _is_scalar(value):
        detail = f"{field_name} {operator} expects a scalar value"
        raise FilterValidationError(detail)


def _require_scalar_list(field_name: str, value: JsonValue) -> list[FilterScalar]:
    if not isinstance(value, list) or not value:
        detail = f"{field_name} in expects a non-empty list"
        raise FilterValidationError(detail)
    scalars: list[FilterScalar] = []
    for item in value:
        if not _is_scalar(item):
            detail = f"{field_name} in expects scalar values"
            raise FilterValidationError(detail)
        scalars.append(item)
    return scalars


def _require_timestamp(value: JsonValue) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as error:
            detail = "created_at expects an ISO-8601 timestamp"
            raise FilterValidationError(detail) from error
    detail = "created_at expects an ISO-8601 timestamp"
    raise FilterValidationError(detail)


def _is_scalar(value: JsonValue) -> TypeGuard[FilterScalar]:
    return isinstance(value, str | int | float | bool)


def _narrowing_fragment(
    filters: MemoryFilter,
    *,
    alias: str,
    parameter_prefix: str,
    parameters: dict[str, JsonValue],
    exact: bool,
) -> str | None:
    match filters:
        case FilterCondition(field_name="created_at"):
            return _created_at_fragment(filters, alias, parameter_prefix, parameters)
        case FilterCondition():
            if exact:
                return None
            return _metadata_fragment(filters, alias, parameter_prefix, parameters)
        case FilterGroup(combinator=combinator, clauses=clauses):
            return _group_fragment(
                clauses,
                combinator=combinator,
                alias=alias,
                parameter_prefix=parameter_prefix,
                parameters=parameters,
                exact=exact,
            )
        case FilterNegation(clause=clause):
            inner = _narrowing_fragment(
                clause,
                alias=alias,
                parameter_prefix=parameter_prefix,
                parameters=parameters,
                exact=True,
            )
            if inner is None:
                return None
            return f"NOT ({inner})"


def _group_fragment(  # noqa: PLR0913 - mirrors the narrowing traversal inputs.
    clauses: tuple[MemoryFilter, ...],
    *,
    combinator: FilterCombinator,
    alias: str,
    parameter_prefix: str,
    parameters: dict[str, JsonValue],
    exact: bool,
) -> str | None:
    fragments: list[str] = []
    for clause in clauses:
        fragment = _narrowing_fragment(
            clause,
            alias=alias,
            parameter_prefix=parameter_prefix,
            parameters=parameters,
            exact=exact,
        )
        if fragment is None:
            if combinator == "OR" or exact:
                return None
            continue
        fragments.append(fragment)
    if not fragments:
        return None
    joiner = f" {combinator} "
    return f"({joiner.join(fragments)})"


def _created_at_fragment(
    condition: FilterCondition,
    alias: str,
    parameter_prefix: str,
    parameters: dict[str, JsonValue],
) -> str:
    if condition.operator == "in":
        name = _bind(parameters, parameter_prefix, condition.value)
        return f"{alias}.created_at IN [item IN ${name} | datetime(item)]"
    cypher_operator = _CREATED_AT_CYPHER_OPERATORS[condition.operator]
    name = _bind(parameters, parameter_prefix, condition.value)
    return f"{alias}.created_at {cypher_operator} datetime(${name})"


def _metadata_fragment(
    condition: FilterCondition,
    alias: str,
    parameter_prefix: str,
    parameters: dict[str, JsonValue],
) -> str | None:
    key = _metadata_key(condition.field_name)
    match condition.operator:
        case "eq":
            name = _bind(
                parameters,
                parameter_prefix,
                _metadata_json_fragment(key, condition.value),
            )
            return f"{alias}.metadata CONTAINS ${name}"
        case "in":
            fragments: list[JsonValue] = [
                _metadata_json_fragment(key, item)
                for item in _require_scalar_list(condition.field_name, condition.value)
            ]
            name = _bind(parameters, parameter_prefix, fragments)
            return f"any(fragment IN ${name} WHERE {alias}.metadata CONTAINS fragment)"
        case "contains" if _is_json_literal_substring(condition.value):
            name = _bind(parameters, parameter_prefix, condition.value)
            return f"{alias}.metadata CONTAINS ${name}"
        case "icontains" if _is_json_literal_substring(condition.value):
            name = _bind(parameters, parameter_prefix, condition.value)
            return f"toLower({alias}.metadata) CONTAINS toLower(${name})"
        case _:
            return None


def _metadata_key(field_name: str) -> str:
    return field_name.removeprefix(_METADATA_FIELD_PREFIX)


def _metadata_json_fragment(key: str, value: JsonValue) -> str:
    return f"{json.dumps(key)}: {json.dumps(value)}"


def _is_json_literal_substring(value: JsonValue) -> bool:
    return isinstance(value, str) and json.dumps(value)[1:-1] == value


def _bind(
    parameters: dict[str, JsonValue],
    parameter_prefix: str,
    value: JsonValue,
) -> str:
    name = f"{parameter_prefix}_{len(parameters)}"
    parameters[name] = value
    return name


def _condition_matches(
    condition: FilterCondition,
    fields: MemoryFilterFields,
) -> bool:
    actual = _resolve_field(condition.field_name, fields)
    if condition.field_name == _CREATED_AT_FIELD:
        return _created_at_matches(condition, fields.created_at)
    return _value_matches(condition, actual)


def _resolve_field(field_name: str, fields: MemoryFilterFields) -> JsonValue:
    match field_name:
        case "user_id":
            return fields.user_id
        case "agent_id":
            return fields.agent_id
        case "created_at":
            return None
        case _:
            return fields.metadata.get(_metadata_key(field_name))


def _created_at_matches(
    condition: FilterCondition,
    created_at: datetime | None,
) -> bool:
    if created_at is None:
        return condition.operator == "ne"
    actual = _as_utc(created_at)
    if condition.operator == "in":
        expected = [
            _as_utc(_require_timestamp(item))
            for item in _require_scalar_list(_CREATED_AT_FIELD, condition.value)
        ]
        return actual in expected
    return _compare_datetimes(
        condition.operator,
        actual,
        _as_utc(_require_timestamp(condition.value)),
    )


def _compare_datetimes(
    operator: FilterOperator,
    actual: datetime,
    expected: datetime,
) -> bool:
    match operator:
        case "eq":
            return actual == expected
        case "ne":
            return actual != expected
        case "gt":
            return actual > expected
        case "gte":
            return actual >= expected
        case "lt":
            return actual < expected
        case _:
            return actual <= expected


def _value_matches(condition: FilterCondition, actual: JsonValue) -> bool:
    match condition.operator:
        case "eq":
            return actual == condition.value
        case "ne":
            return actual != condition.value
        case "in":
            return isinstance(condition.value, list) and actual in condition.value
        case "contains":
            return _contains(actual, condition.value, fold_case=False)
        case "icontains":
            return _contains(actual, condition.value, fold_case=True)
        case _:
            return _compare_numbers(condition.operator, actual, condition.value)


def _contains(actual: JsonValue, expected: JsonValue, *, fold_case: bool) -> bool:
    if isinstance(actual, list):
        return expected in actual
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    if fold_case:
        return expected.casefold() in actual.casefold()
    return expected in actual


def _compare_numbers(
    operator: FilterOperator,
    actual: JsonValue,
    expected: JsonValue,
) -> bool:
    if isinstance(actual, bool) or not isinstance(actual, int | float):
        return False
    if isinstance(expected, bool) or not isinstance(expected, int | float):
        return False
    match operator:
        case "gt":
            return actual > expected
        case "gte":
            return actual >= expected
        case "lt":
            return actual < expected
        case _:
            return actual <= expected


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
