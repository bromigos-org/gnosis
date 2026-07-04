import re
from re import Pattern
from typing import Final

LABEL_PATTERN: Final[Pattern[str]] = re.compile(r":([A-Za-z][A-Za-z0-9_]*)")
RELATIONSHIP_PATTERN: Final[Pattern[str]] = re.compile(
    r"\[[A-Za-z0-9_]*:([A-Z][A-Z0-9_]*)",
)
# Relationship bracket spans, blanked before node-label extraction so a
# relationship type (e.g. ``[:RELATES]``) is never mistaken for a node label -
# relationship types are checked separately against SAFE_RELATIONSHIPS.
RELATIONSHIP_BRACKET_PATTERN: Final[Pattern[str]] = re.compile(r"\[[^\]]*\]")
PROPERTY_PATTERN: Final[Pattern[str]] = re.compile(r"\.([A-Za-z][A-Za-z0-9_]*)")
KEYWORD_PATTERN: Final[Pattern[str]] = re.compile(r"\b[A-Z]+\b")
ALIAS_LABEL_PATTERN: Final[Pattern[str]] = re.compile(
    r"\((?P<alias>[A-Za-z][A-Za-z0-9_]*)\s*:(?P<label>[A-Za-z][A-Za-z0-9_]*)",
)
LIMIT_PATTERN: Final[Pattern[str]] = re.compile(r"\bLIMIT\s+\$limit\b", re.IGNORECASE)
SCOPE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\btenant_id\s*(?:[:=])\s*\$tenant_id\b",
    re.IGNORECASE,
)
GUILD_SCOPE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bguild_id\s*=\s*\$guild_id\b|\$guild_id\s+IN\b",
    re.IGNORECASE,
)
CHANNEL_SCOPE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bchannel_id\s*=\s*\$channel_id\b",
    re.IGNORECASE,
)
RAW_SCOPE_LITERAL_PATTERN: Final[Pattern[str]] = re.compile(
    r"\b(?:tenant_id|guild_id|channel_id|user_id|agent_id)\s*(?:[:=])\s*['\"]",
    re.IGNORECASE,
)
UNSUPPORTED_SCHEMA_SYNTAX_PATTERN: Final[Pattern[str]] = re.compile(
    r"`|\[[^\]]*(?:\$[A-Za-z][A-Za-z0-9_]*|['\"][^'\"]+['\"])\]|\bproperties\s*\(",
    re.IGNORECASE,
)
RETURN_SHAPE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bRETURN\b(?=.*\bAS\s+id\b)(?=.*\bAS\s+type\b)(?=.*\bAS\s+summary\b)(?=.*\bAS\s+deleted\b)",
    re.IGNORECASE | re.DOTALL,
)
READ_PREFIX_PATTERN: Final[Pattern[str]] = re.compile(
    r"^\s*(?:MATCH|OPTIONAL\s+MATCH|WITH|CALL\s*\{)",
    re.IGNORECASE,
)
UNSAFE_PROCEDURE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bCALL\s+(?!\{)",
    re.IGNORECASE,
)
UNSAFE_KEYWORDS: Final[frozenset[str]] = frozenset(
    {"CREATE", "DELETE", "DETACH", "DROP", "LOAD", "MERGE", "REMOVE", "SET"},
)
SAFE_LABELS: Final[frozenset[str]] = frozenset(
    {
        "Agent",
        "Attachment",
        "Bot",
        "Category",
        "Channel",
        "Client",
        "Entity",
        "Event",
        "Fact",
        "GraphNode",
        "Guild",
        "Link",
        "Message",
        "Role",
        "Tenant",
        "User",
    },
)
SAFE_RELATIONSHIPS: Final[frozenset[str]] = frozenset(
    {
        "AFFECTS",
        "ATTACHED_TO",
        "AUTHORED",
        "HAS_ROLE",
        "IN_CATEGORY",
        "IN_CHANNEL",
        "IN_GUILD",
        "LINKED_FROM",
        "MENTIONS",
        "OWNS_AGENT",
        "OWNS_CLIENT",
        "OWNS_GUILD",
        "OWNS_ROLE",
        "RELATES",
        "USES_CLIENT",
    },
)
SAFE_PROPERTIES: Final[frozenset[str]] = frozenset(
    {
        "agent_id",
        "category_id",
        "channel_id",
        "content",
        "deleted",
        "display_name",
        "event_date",
        "event_id",
        "event_type",
        "fact_id",
        "filename",
        "guild_id",
        "id",
        "is_bot",
        "kind",
        "message_id",
        "name",
        "object",
        "occurred_at",
        "payload",
        "predicate",
        "relation",
        "role_id",
        "subject",
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
GUILD_SCOPED_LABELS: Final[frozenset[str]] = frozenset(
    {"Category", "Channel", "GraphNode", "Message", "Role"},
)
CHANNEL_SCOPED_LABELS: Final[frozenset[str]] = frozenset(
    {"Channel", "GraphNode", "Message"},
)
# Knowledge-graph nodes are per-user: every Entity and Fact alias must be
# scoped by user_id = $user_id (in addition to tenant_id) so entity traversal
# can never read another user's remembered facts within the same tenant.
USER_SCOPED_LABELS: Final[frozenset[str]] = frozenset({"Entity", "Fact"})


def alias_predicate_pattern(alias: str, property_name: str) -> Pattern[str]:
    return re.compile(
        rf"\b{re.escape(alias)}\.{property_name}\s*=\s*\${property_name}\b",
        re.IGNORECASE,
    )


def node_map_parameter_pattern(alias: str, property_name: str) -> Pattern[str]:
    return re.compile(
        rf"\({re.escape(alias)}\s*:[^)]*\{{[^}}]*\b{property_name}\s*:\s*\${property_name}\b",
        re.IGNORECASE | re.DOTALL,
    )
