import re
from dataclasses import dataclass
from re import Pattern
from typing import Final, final

from pydantic import TypeAdapter

from gnosis.graph_query_qa import GraphQueryPlan, ValidatedGraphQuery
from gnosis.graph_types import CypherParameters
from gnosis.models import GraphContextRequest, JsonValue

_LABEL_PATTERN: Final[Pattern[str]] = re.compile(r":([A-Za-z][A-Za-z0-9_]*)")
_RELATIONSHIP_PATTERN: Final[Pattern[str]] = re.compile(r"\[:([A-Z][A-Z0-9_]*)")
_PROPERTY_PATTERN: Final[Pattern[str]] = re.compile(r"\.([A-Za-z][A-Za-z0-9_]*)")
_LIMIT_PATTERN: Final[Pattern[str]] = re.compile(r"\bLIMIT\s+\$limit\b", re.IGNORECASE)
_SCOPE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\btenant_id\s*(?:[:=])\s*\$tenant_id\b",
    re.IGNORECASE,
)
_GUILD_SCOPE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bguild_id\s*=\s*\$guild_id\b|\$guild_id\s+IN\b",
    re.IGNORECASE,
)
_CHANNEL_SCOPE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bchannel_id\s*=\s*\$channel_id\b",
    re.IGNORECASE,
)
_RAW_SCOPE_LITERAL_PATTERN: Final[Pattern[str]] = re.compile(
    r"\b(?:tenant_id|guild_id|channel_id|user_id|agent_id)\s*(?:[:=])\s*['\"]",
    re.IGNORECASE,
)
_READ_PREFIX_PATTERN: Final[Pattern[str]] = re.compile(
    r"^\s*(?:MATCH|OPTIONAL\s+MATCH|WITH|CALL\s*\{)",
    re.IGNORECASE,
)
_UNSAFE_KEYWORDS: Final[frozenset[str]] = frozenset(
    {"CREATE", "DELETE", "DETACH", "DROP", "LOAD", "MERGE", "REMOVE", "SET"},
)
_UNSAFE_PROCEDURE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bCALL\s+(?!\{)",
    re.IGNORECASE,
)
_SAFE_LABELS: Final[frozenset[str]] = frozenset(
    {
        "Agent",
        "Attachment",
        "Bot",
        "Category",
        "Channel",
        "Client",
        "Event",
        "GraphNode",
        "Guild",
        "Link",
        "Message",
        "Role",
        "Tenant",
        "User",
    },
)
_SAFE_RELATIONSHIPS: Final[frozenset[str]] = frozenset(
    {
        "AFFECTS",
        "ATTACHED_TO",
        "AUTHORED",
        "HAS_ROLE",
        "IN_CATEGORY",
        "IN_CHANNEL",
        "IN_GUILD",
        "LINKED_FROM",
        "OWNS_AGENT",
        "OWNS_CLIENT",
        "OWNS_GUILD",
        "OWNS_ROLE",
        "USES_CLIENT",
    },
)
_SAFE_PROPERTIES: Final[frozenset[str]] = frozenset(
    {
        "agent_id",
        "category_id",
        "channel_id",
        "content",
        "deleted",
        "display_name",
        "event_id",
        "event_type",
        "filename",
        "guild_id",
        "id",
        "is_bot",
        "kind",
        "message_id",
        "name",
        "occurred_at",
        "payload",
        "role_id",
        "summary",
        "tenant_id",
        "type",
        "updated_at",
        "url",
        "user_id",
        "user_type",
        "visibility",
    },
)
_JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_WRITE_KEYWORD_REASON: Final[str] = "write keyword is not allowed"
_READ_PREFIX_REASON: Final[str] = "query must start with a read clause"
_PROCEDURE_REASON: Final[str] = "procedures are not allowed"
_RAW_SCOPE_REASON: Final[str] = "scope values must use parameters"
_LIMIT_REASON: Final[str] = "query must use LIMIT $limit"
_TENANT_SCOPE_REASON: Final[str] = "query must scope tenant_id with $tenant_id"
_GUILD_SCOPE_REASON: Final[str] = "query must scope guild_id with $guild_id"
_CHANNEL_SCOPE_REASON: Final[str] = "channel queries must scope channel_id"


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
    upper_tokens = frozenset(re.findall(r"\b[A-Z]+\b", cypher.upper()))
    if upper_tokens & _UNSAFE_KEYWORDS:
        raise GraphQueryValidationError(_WRITE_KEYWORD_REASON)
    if not _READ_PREFIX_PATTERN.search(cypher):
        raise GraphQueryValidationError(_READ_PREFIX_REASON)
    if _UNSAFE_PROCEDURE_PATTERN.search(cypher):
        raise GraphQueryValidationError(_PROCEDURE_REASON)
    if _RAW_SCOPE_LITERAL_PATTERN.search(cypher):
        raise GraphQueryValidationError(_RAW_SCOPE_REASON)
    if _LIMIT_PATTERN.search(cypher) is None:
        raise GraphQueryValidationError(_LIMIT_REASON)
    if _SCOPE_PATTERN.search(cypher) is None:
        raise GraphQueryValidationError(_TENANT_SCOPE_REASON)
    if (
        request.scope.guild_id is not None
        and _GUILD_SCOPE_PATTERN.search(cypher) is None
    ):
        raise GraphQueryValidationError(_GUILD_SCOPE_REASON)
    if (
        request.scope.channel_id is not None
        and "Channel" in _labels(cypher)
        and _CHANNEL_SCOPE_PATTERN.search(cypher) is None
        and _GUILD_SCOPE_PATTERN.search(cypher) is None
    ):
        raise GraphQueryValidationError(_CHANNEL_SCOPE_REASON)
    _require_known_tokens(_labels(cypher), _SAFE_LABELS, "label")
    _require_known_tokens(_relationships(cypher), _SAFE_RELATIONSHIPS, "relationship")
    _require_known_tokens(_properties(cypher), _SAFE_PROPERTIES, "property")


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
    return frozenset(_LABEL_PATTERN.findall(cypher))


def _relationships(cypher: str) -> frozenset[str]:
    return frozenset(_RELATIONSHIP_PATTERN.findall(cypher))


def _properties(cypher: str) -> frozenset[str]:
    return frozenset(_PROPERTY_PATTERN.findall(cypher))


def _require_known_tokens(
    tokens: frozenset[str],
    allowed: frozenset[str],
    token_type: str,
) -> None:
    unknown = tokens - allowed
    if unknown:
        reason = f"unknown {token_type}: {sorted(unknown)[0]}"
        raise GraphQueryValidationError(reason)
