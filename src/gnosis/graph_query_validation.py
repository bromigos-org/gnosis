from dataclasses import dataclass
from typing import Final, final

from pydantic import TypeAdapter

from gnosis import graph_query_rules as rules
from gnosis.graph_query_qa import GraphQueryPlan, ValidatedGraphQuery
from gnosis.graph_types import CypherParameters
from gnosis.models import GraphContextRequest, JsonValue

_JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_WRITE_KEYWORD_REASON: Final[str] = "write keyword is not allowed"
_READ_PREFIX_REASON: Final[str] = "query must start with a read clause"
_PROCEDURE_REASON: Final[str] = "procedures are not allowed"
_RAW_SCOPE_REASON: Final[str] = "scope values must use parameters"
_LIMIT_REASON: Final[str] = "query must use LIMIT $limit"
_TENANT_SCOPE_REASON: Final[str] = "query must scope tenant_id with $tenant_id"
_USER_SCOPE_REASON: Final[str] = "query must scope user_id with $user_id"
_GUILD_SCOPE_REASON: Final[str] = "query must scope guild_id with $guild_id"
_CHANNEL_SCOPE_REASON: Final[str] = "channel queries must scope channel_id"
_AGENT_SCOPE_REASON: Final[str] = "query must scope agent_id with $agent_id"
_UNSUPPORTED_SCHEMA_REASON: Final[str] = "unsupported schema access syntax"
_RETURN_SHAPE_REASON: Final[str] = "query must return id, type, summary, and deleted"


@final
class GraphQueryValidationError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason: str = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class SafeGraphQueryValidator:
    def validate(
        self,
        plan: GraphQueryPlan,
        request: GraphContextRequest,
    ) -> ValidatedGraphQuery:
        cypher = plan.cypher.strip()
        _require_safe_cypher(cypher, request)
        return ValidatedGraphQuery(
            cypher=plan.cypher,
            parameters=_parameters(plan, request),
            answer_kind=plan.answer_kind,
        )


def _require_safe_cypher(cypher: str, request: GraphContextRequest) -> None:
    _require_safe_syntax(cypher)
    _require_alias_scope(cypher, request)
    if (
        request.scope.guild_id is not None
        and rules.GUILD_SCOPE_PATTERN.search(cypher) is None
    ):
        raise GraphQueryValidationError(_GUILD_SCOPE_REASON)
    if (
        request.scope.channel_id is not None
        and "Channel" in _labels(cypher)
        and rules.CHANNEL_SCOPE_PATTERN.search(cypher) is None
        and rules.GUILD_SCOPE_PATTERN.search(cypher) is None
    ):
        raise GraphQueryValidationError(_CHANNEL_SCOPE_REASON)
    _require_known_tokens(_labels(cypher), rules.SAFE_LABELS, "label")
    _require_known_tokens(
        _relationships(cypher),
        rules.SAFE_RELATIONSHIPS,
        "relationship",
    )
    _require_known_tokens(_properties(cypher), rules.SAFE_PROPERTIES, "property")


def _require_safe_syntax(cypher: str) -> None:
    upper_tokens = frozenset(rules.KEYWORD_PATTERN.findall(cypher.upper()))
    if upper_tokens & rules.UNSAFE_KEYWORDS:
        raise GraphQueryValidationError(_WRITE_KEYWORD_REASON)
    if not rules.READ_PREFIX_PATTERN.search(cypher):
        raise GraphQueryValidationError(_READ_PREFIX_REASON)
    if rules.UNSAFE_PROCEDURE_PATTERN.search(cypher):
        raise GraphQueryValidationError(_PROCEDURE_REASON)
    if rules.RAW_SCOPE_LITERAL_PATTERN.search(cypher):
        raise GraphQueryValidationError(_RAW_SCOPE_REASON)
    if rules.UNSUPPORTED_SCHEMA_SYNTAX_PATTERN.search(cypher):
        raise GraphQueryValidationError(_UNSUPPORTED_SCHEMA_REASON)
    if rules.LIMIT_PATTERN.search(cypher) is None:
        raise GraphQueryValidationError(_LIMIT_REASON)
    if rules.RETURN_SHAPE_PATTERN.search(cypher) is None:
        raise GraphQueryValidationError(_RETURN_SHAPE_REASON)
    if rules.SCOPE_PATTERN.search(cypher) is None:
        raise GraphQueryValidationError(_TENANT_SCOPE_REASON)


def _require_alias_scope(cypher: str, request: GraphContextRequest) -> None:
    labels_by_alias = _labels_by_alias(cypher)
    for alias, label in labels_by_alias.items():
        if label == "Tenant":
            continue
        _require_alias_predicate(cypher, alias, "tenant_id", _TENANT_SCOPE_REASON)
        if label == "GraphNode":
            _require_alias_predicate(cypher, alias, "agent_id", _AGENT_SCOPE_REASON)
        if label in rules.USER_SCOPED_LABELS:
            _require_alias_predicate(cypher, alias, "user_id", _USER_SCOPE_REASON)
        if request.scope.guild_id is not None and label in rules.GUILD_SCOPED_LABELS:
            _require_alias_predicate(cypher, alias, "guild_id", _GUILD_SCOPE_REASON)
        if (
            request.scope.channel_id is not None
            and label in rules.CHANNEL_SCOPED_LABELS
        ):
            _require_alias_predicate(cypher, alias, "channel_id", _CHANNEL_SCOPE_REASON)


def _labels_by_alias(cypher: str) -> dict[str, str]:
    return {
        match.group("alias"): match.group("label")
        for match in rules.ALIAS_LABEL_PATTERN.finditer(cypher)
    }


def _require_alias_predicate(
    cypher: str,
    alias: str,
    property_name: str,
    reason: str,
) -> None:
    pattern = rules.alias_predicate_pattern(alias, property_name)
    if pattern.search(cypher) is None and not _node_map_has_parameter(
        cypher,
        alias,
        property_name,
    ):
        raise GraphQueryValidationError(reason)


def _node_map_has_parameter(cypher: str, alias: str, property_name: str) -> bool:
    pattern = rules.node_map_parameter_pattern(alias, property_name)
    return pattern.search(cypher) is not None


def _parameters(plan: GraphQueryPlan, request: GraphContextRequest) -> CypherParameters:
    parameters: CypherParameters = {
        key: _JSON_VALUE_ADAPTER.validate_python(value)
        for key, value in plan.parameters.items()
    }
    parameters["tenant_id"] = request.scope.tenant_id
    parameters["agent_id"] = request.scope.agent_id
    parameters["user_id"] = request.scope.user_id
    parameters["guild_id"] = request.scope.guild_id
    parameters["channel_id"] = request.scope.channel_id
    parameters["limit"] = request.limit
    return parameters


def _labels(cypher: str) -> frozenset[str]:
    # Blank relationship brackets first so a relationship type inside ``[...]``
    # is not extracted as a node label (it is validated as a relationship).
    node_cypher = rules.RELATIONSHIP_BRACKET_PATTERN.sub(" ", cypher)
    return frozenset(rules.LABEL_PATTERN.findall(node_cypher))


def _relationships(cypher: str) -> frozenset[str]:
    return frozenset(rules.RELATIONSHIP_PATTERN.findall(cypher))


def _properties(cypher: str) -> frozenset[str]:
    return frozenset(rules.PROPERTY_PATTERN.findall(cypher))


def _require_known_tokens(
    tokens: frozenset[str],
    allowed: frozenset[str],
    token_type: str,
) -> None:
    unknown = tokens - allowed
    if unknown:
        reason = f"unknown {token_type}: {sorted(unknown)[0]}"
        raise GraphQueryValidationError(reason)
