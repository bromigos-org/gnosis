# pyright: reportAny=false
import inspect
import operator
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import pytest
from neo4j_agent_memory import MemoryConfig, MemorySettings, Neo4jConfig
from neo4j_agent_memory.config.settings import ExtractionConfig, SchemaConfig
from neo4j_agent_memory.memory.short_term import ShortTermMemory
from pydantic import SecretStr

import agents_memory.backend as backend_module
from agents_memory.settings import Settings

NON_SECRET_TOKEN = "memory-token-sentinel"
NON_SECRET_READ_OPERATOR_TOKEN = "read-operator-token"
NON_SECRET_EXPORT_OPERATOR_TOKEN = "export-operator-token"
NON_SECRET_WRITE_OPERATOR_TOKEN = "write-operator-token"
NON_SECRET_ADMIN_OPERATOR_TOKEN = "admin-operator-token"
NON_SECRET_PASSWORD = "neo4j-password-sentinel"
NON_SECRET_API_KEY = "litellm-api-key-sentinel"


class CapabilityStatus(StrEnum):
    PRESENT = "present"
    MISSING = "missing"
    DIFFERENTLY_NAMED = "differently_named"
    NAMS_ONLY = "nams_only"
    DEFERRED = "deferred"


class FallbackPolicy(StrEnum):
    NONE = "none"
    DEPENDENCY_BUMP = "tested_dependency_bump_if_compatible"
    HTTP_501 = "typed_501_capability_unavailable"
    LOCAL_SHIM = "safe_local_shim"
    DOCUMENTED_DEFERRAL = "documented_deferral"


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    name: str
    module: str
    qualname: tuple[str, ...]
    status: CapabilityStatus
    fallback: FallbackPolicy = FallbackPolicy.NONE
    required_params: frozenset[str] = frozenset()
    bolt_compatible: bool = True
    async_expected: bool | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CapabilityFinding:
    spec: CapabilitySpec
    present: bool
    is_async: bool
    signature: str


def test_installed_sdk_version_and_project_pin_match_plan() -> None:
    pyproject = Path("pyproject.toml").read_text()

    installed_version = version("neo4j-agent-memory")

    assert installed_version == "0.5.0"
    assert '"neo4j-agent-memory[litellm]==0.5.0"' in pyproject


def test_current_backend_assumptions_match_installed_sdk() -> None:
    settings = Settings(
        agents_memory_token=NON_SECRET_TOKEN,
        agents_memory_read_operator_token=NON_SECRET_READ_OPERATOR_TOKEN,
        agents_memory_export_operator_token=NON_SECRET_EXPORT_OPERATOR_TOKEN,
        agents_memory_write_operator_token=NON_SECRET_WRITE_OPERATOR_TOKEN,
        agents_memory_admin_operator_token=NON_SECRET_ADMIN_OPERATOR_TOKEN,
        neo4j_uri="bolt://neo4j.local:7687",
        neo4j_password=NON_SECRET_PASSWORD,
        litellm_base_url="http://litellm.local/v1",
        litellm_api_key=NON_SECRET_API_KEY,
    )

    build_memory_settings = _callable_attr(backend_module, "_build_memory_settings")
    sdk_settings_object = build_memory_settings(settings)
    assert isinstance(sdk_settings_object, MemorySettings)
    sdk_settings = sdk_settings_object
    add_message = _signature_for(
        "neo4j_agent_memory.memory.short_term", "ShortTermMemory", "add_message"
    )

    assert sdk_settings.backend == "bolt"
    assert sdk_settings.memory == MemoryConfig(multi_tenant=True)
    assert isinstance(sdk_settings.neo4j, Neo4jConfig)
    assert isinstance(sdk_settings.neo4j.password, SecretStr)
    assert inspect.isclass(ShortTermMemory)
    assert {
        "metadata",
        "extract_entities",
        "extract_relations",
        "user_identifier",
    } <= set(add_message.parameters)


def test_capability_matrix_covers_every_planned_feature() -> None:
    matrix = [_inspect_capability(spec) for spec in CAPABILITY_MATRIX]
    names = {finding.spec.name for finding in matrix}

    assert names >= REQUIRED_CAPABILITIES
    assert {finding.spec.status for finding in matrix} >= {
        CapabilityStatus.PRESENT,
        CapabilityStatus.NAMS_ONLY,
        CapabilityStatus.DEFERRED,
    }
    for finding in matrix:
        _assert_finding_matches_policy(finding)


def test_matrix_fails_loudly_when_required_sdk_method_is_renamed() -> None:
    spec = CapabilitySpec(
        name="strictness fake required method",
        module=__name__,
        qualname=("FakeSdkSurface", "required_method"),
        status=CapabilityStatus.PRESENT,
        async_expected=True,
    )

    finding = _inspect_capability(spec)
    assert finding.present is False
    with pytest.raises(AssertionError, match="strictness fake required method"):
        _assert_finding_matches_policy(finding)


class FakeSdkSurface:
    async def renamed_method(self) -> None:
        return None


def _inspect_capability(spec: CapabilitySpec) -> CapabilityFinding:
    target = _resolve_target(spec.module, spec.qualname)
    if target is None:
        return CapabilityFinding(spec=spec, present=False, is_async=False, signature="")
    return CapabilityFinding(
        spec=spec,
        present=True,
        is_async=inspect.iscoroutinefunction(target),
        signature=str(inspect.signature(target)) if callable(target) else "<class>",
    )


def _resolve_target(module_name: str, qualname: tuple[str, ...]) -> object | None:
    try:
        target: object = import_module(module_name)
    except ModuleNotFoundError:
        return None
    for name in qualname:
        if not hasattr(target, name):
            return None
        target = operator.attrgetter(name)(target)
    return target


def _assert_finding_matches_policy(finding: CapabilityFinding) -> None:
    spec = finding.spec
    if spec.status in {CapabilityStatus.PRESENT, CapabilityStatus.DIFFERENTLY_NAMED}:
        assert finding.present, (
            f"{spec.name} missing at {spec.module}:{'.'.join(spec.qualname)}"
        )
        assert spec.fallback == FallbackPolicy.NONE, (
            f"{spec.name} must not carry fallback"
        )
    if spec.status in {
        CapabilityStatus.MISSING,
        CapabilityStatus.NAMS_ONLY,
        CapabilityStatus.DEFERRED,
    }:
        assert spec.fallback != FallbackPolicy.NONE, (
            f"{spec.name} needs exactly one fallback"
        )
    if spec.async_expected is not None and finding.present:
        assert finding.is_async is spec.async_expected, (
            f"{spec.name} async mismatch: {finding.signature}"
        )
    if finding.present and spec.required_params:
        assert spec.required_params <= set(
            inspect.signature(_resolved_callable(spec)).parameters
        ), spec.name


def _resolved(spec: CapabilitySpec) -> object:
    target = _resolve_target(spec.module, spec.qualname)
    assert target is not None, spec.name
    return target


def _resolved_callable(spec: CapabilitySpec) -> Callable[..., object]:
    target = _resolved(spec)
    assert callable(target), spec.name
    return target


def _required_attr(value: object, name: str) -> object:
    assert hasattr(value, name), name
    return operator.attrgetter(name)(value)


def _callable_attr(value: object, name: str) -> Callable[..., object]:
    target = _required_attr(value, name)
    assert callable(target), name
    return target


def _signature_for(module: str, class_name: str, method_name: str) -> inspect.Signature:
    target = _resolve_target(module, (class_name, method_name))
    assert target is not None, f"{module}.{class_name}.{method_name}"
    assert callable(target), f"{module}.{class_name}.{method_name}"
    return inspect.signature(target)


CAPABILITY_MATRIX = (
    CapabilitySpec(
        "MemoryClient root",
        "neo4j_agent_memory",
        ("MemoryClient",),
        CapabilityStatus.PRESENT,
    ),
    CapabilitySpec(
        "MemoryConfig root",
        "neo4j_agent_memory",
        ("MemoryConfig",),
        CapabilityStatus.PRESENT,
    ),
    CapabilitySpec(
        "short_term property",
        "neo4j_agent_memory",
        ("MemoryClient", "short_term"),
        CapabilityStatus.PRESENT,
    ),
    CapabilitySpec(
        "long_term property",
        "neo4j_agent_memory",
        ("MemoryClient", "long_term"),
        CapabilityStatus.PRESENT,
    ),
    CapabilitySpec(
        "reasoning property",
        "neo4j_agent_memory",
        ("MemoryClient", "reasoning"),
        CapabilityStatus.PRESENT,
    ),
    CapabilitySpec(
        "stats",
        "neo4j_agent_memory",
        ("MemoryClient", "get_stats"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "graph export/access",
        "neo4j_agent_memory",
        ("MemoryClient", "get_graph"),
        CapabilityStatus.PRESENT,
        required_params=frozenset({"memory_types", "limit"}),
        async_expected=True,
    ),
    CapabilitySpec(
        "write buffer flush",
        "neo4j_agent_memory",
        ("MemoryClient", "flush"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "write buffer pending wait",
        "neo4j_agent_memory",
        ("MemoryClient", "wait_for_pending"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "write buffer errors",
        "neo4j_agent_memory",
        ("MemoryClient", "write_errors"),
        CapabilityStatus.PRESENT,
    ),
    CapabilitySpec(
        "MemoryConfig write mode",
        "neo4j_agent_memory",
        ("MemoryConfig",),
        CapabilityStatus.PRESENT,
        required_params=frozenset({"write_mode", "max_pending"}),
    ),
    CapabilitySpec(
        "MemoryConfig audit read",
        "neo4j_agent_memory",
        ("MemoryConfig",),
        CapabilityStatus.PRESENT,
        required_params=frozenset({"audit_read"}),
    ),
    CapabilitySpec(
        "Extraction module",
        "neo4j_agent_memory.extraction",
        ("ExtractionResult",),
        CapabilityStatus.PRESENT,
    ),
    CapabilitySpec(
        "GLiNEREntityExtractor",
        "neo4j_agent_memory.extraction.gliner_extractor",
        ("GLiNEREntityExtractor",),
        CapabilityStatus.DIFFERENTLY_NAMED,
        notes="present in submodule, not top-level extraction export",
    ),
    CapabilitySpec(
        "LLMEntityExtractor",
        "neo4j_agent_memory.extraction.llm_extractor",
        ("LLMEntityExtractor",),
        CapabilityStatus.DIFFERENTLY_NAMED,
        notes="present in submodule, not top-level extraction export",
    ),
    CapabilitySpec(
        "ExtractionPipeline",
        "neo4j_agent_memory.extraction",
        ("ExtractionPipeline",),
        CapabilityStatus.PRESENT,
    ),
    CapabilitySpec(
        "batch extraction",
        "neo4j_agent_memory.extraction",
        ("ExtractionPipeline", "extract_batch"),
        CapabilityStatus.PRESENT,
        required_params=frozenset({"texts", "batch_size", "max_concurrency"}),
        async_expected=True,
    ),
    CapabilitySpec(
        "StreamingExtractor",
        "neo4j_agent_memory.extraction",
        ("StreamingExtractor",),
        CapabilityStatus.PRESENT,
    ),
    CapabilitySpec(
        "streaming extraction",
        "neo4j_agent_memory.extraction",
        ("StreamingExtractor", "extract"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "relation extraction",
        "neo4j_agent_memory.extraction",
        ("ExtractionPipeline", "extract"),
        CapabilityStatus.PRESENT,
        required_params=frozenset({"extract_relations"}),
        async_expected=True,
    ),
    CapabilitySpec(
        "POLE+O schema config",
        "neo4j_agent_memory.config.settings",
        ("SchemaConfig",),
        CapabilityStatus.PRESENT,
        required_params=frozenset({"model", "enable_subtypes", "strict_types"}),
    ),
    CapabilitySpec(
        "extraction config schema",
        "neo4j_agent_memory.config.settings",
        ("ExtractionConfig",),
        CapabilityStatus.PRESENT,
        required_params=frozenset(
            {"gliner_schema", "entity_types", "extract_relations"}
        ),
    ),
    CapabilitySpec(
        "entities",
        "neo4j_agent_memory.memory.long_term",
        ("LongTermMemory", "add_entity"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "facts",
        "neo4j_agent_memory.memory.long_term",
        ("LongTermMemory", "add_fact"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "preferences",
        "neo4j_agent_memory.memory.long_term",
        ("LongTermMemory", "add_preference"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "dedup stats",
        "neo4j_agent_memory.memory.long_term",
        ("LongTermMemory", "get_deduplication_stats"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "dedup review",
        "neo4j_agent_memory.memory.long_term",
        ("LongTermMemory", "review_duplicate"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "dedup merge",
        "neo4j_agent_memory.memory.long_term",
        ("LongTermMemory", "merge_duplicate_entities"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "provenance entity to message link",
        "neo4j_agent_memory.memory.long_term",
        ("LongTermMemory", "link_entity_to_message"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "provenance extractor link",
        "neo4j_agent_memory.memory.long_term",
        ("LongTermMemory", "link_entity_to_extractor"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "reasoning trace start",
        "neo4j_agent_memory.memory.reasoning",
        ("ReasoningMemory", "start_trace"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "reasoning trace stats",
        "neo4j_agent_memory.memory.reasoning",
        ("ReasoningMemory", "get_similar_traces"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "reasoning tool stats",
        "neo4j_agent_memory.memory.reasoning",
        ("ReasoningMemory", "get_tool_stats"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "reasoning trace message link",
        "neo4j_agent_memory.memory.reasoning",
        ("ReasoningMemory", "link_trace_to_message"),
        CapabilityStatus.PRESENT,
        async_expected=True,
    ),
    CapabilitySpec(
        "consolidation APIs",
        "neo4j_agent_memory.memory.consolidation",
        ("MemoryConsolidator",),
        CapabilityStatus.DEFERRED,
        FallbackPolicy.DOCUMENTED_DEFERRAL,
        notes=(
            "available module but service mutations are deferred to operator "
            "dry-run/apply tasks"
        ),
    ),
    CapabilitySpec(
        "hosted NAMS ontology endpoint",
        "neo4j_agent_memory.nams.ontology",
        ("NamsOntologyClient",),
        CapabilityStatus.NAMS_ONLY,
        FallbackPolicy.DOCUMENTED_DEFERRAL,
        bolt_compatible=False,
    ),
    CapabilitySpec(
        "MCP tools",
        "neo4j_agent_memory.mcp._tools",
        ("register_tools",),
        CapabilityStatus.DEFERRED,
        FallbackPolicy.DOCUMENTED_DEFERRAL,
        notes="red-team-gated later surface",
    ),
)

REQUIRED_CAPABILITIES = frozenset(spec.name for spec in CAPABILITY_MATRIX)


def test_config_models_expose_planned_schema_fields() -> None:
    extraction_fields = set(ExtractionConfig.model_fields)
    schema_fields = set(SchemaConfig.model_fields)

    assert {
        "enable_gliner",
        "enable_llm_fallback",
        "gliner_schema",
    } <= extraction_fields
    assert {
        "model",
        "entity_types",
        "enable_subtypes",
        "custom_schema_path",
    } <= schema_fields
