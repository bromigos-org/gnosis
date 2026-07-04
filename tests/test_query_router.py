"""Route-table semantics for adaptive query routing (query_router.py).

The integration behavior (routed context assembly, classifier failure
degradation, global-flag suppression) is covered in test_long_term_context.py;
this file pins the pure decision table: which read-path features each route
turns on, and that the unrouted decision mirrors the global settings.
"""

from os import environ

_ = environ.setdefault("GNOSIS_TOKEN", "test-token")
_ = environ.setdefault("GNOSIS_READ_OPERATOR_TOKEN", "read-operator-token")
_ = environ.setdefault("GNOSIS_EXPORT_OPERATOR_TOKEN", "export-operator-token")
_ = environ.setdefault("GNOSIS_WRITE_OPERATOR_TOKEN", "write-operator-token")
_ = environ.setdefault("GNOSIS_ADMIN_OPERATOR_TOKEN", "admin-operator-token")
_ = environ.setdefault("NEO4J_URI", "bolt://neo4j.local:7687")
_ = environ.setdefault("NEO4J_PASSWORD", "inert-password")
_ = environ.setdefault("LITELLM_BASE_URL", "http://litellm.local/v1")
_ = environ.setdefault("LITELLM_API_KEY", "inert-litellm-key")

import pytest  # noqa: E402

from gnosis.query_router import (  # noqa: E402
    QueryRoute,
    RouteDecision,
    parse_route,
)
from gnosis.settings import Settings  # noqa: E402


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "gnosis_token": "value",
        "neo4j_uri": "bolt://neo4j.local:7687",
        "neo4j_password": "value",
        "litellm_base_url": "http://litellm.local/v1",
        "litellm_api_key": "value",
        **overrides,
    }
    return Settings.model_validate(values)


def test_unrouted_decision_mirrors_global_flags() -> None:
    # Given: every routable read-path feature globally enabled.
    settings = _settings(
        gnosis_hybrid_retrieval_enabled=True,
        gnosis_graphqa_fusion_enabled=True,
        gnosis_fact_verbatim_expansion_enabled=True,
        gnosis_abstention_prompt_enabled=True,
        gnosis_graph_traversal_enabled=True,
    )

    # When: the unrouted decision is derived.
    decision = RouteDecision.from_settings(settings)

    # Then: it mirrors the global flags exactly and carries no route.
    assert decision == RouteDecision(
        route=None,
        hybrid_retrieval=True,
        graphqa_fusion=True,
        verbatim_expansion=True,
        abstention_prompt=True,
        graph_traversal=True,
    )


def test_unrouted_decision_defaults_all_off() -> None:
    decision = RouteDecision.from_settings(_settings())
    assert decision == RouteDecision(
        route=None,
        hybrid_retrieval=False,
        graphqa_fusion=False,
        verbatim_expansion=False,
        abstention_prompt=False,
        graph_traversal=False,
    )


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        # temporal won with hybrid BM25 (Run 6: 84.4 -> 92.2); nothing else.
        (
            "temporal",
            RouteDecision(
                route="temporal",
                hybrid_retrieval=True,
                graphqa_fusion=False,
                verbatim_expansion=False,
                abstention_prompt=False,
                graph_traversal=False,
            ),
        ),
        # multi-hop gets the graph traversal route plus verbatim expansion
        # (Run 8's only multi-hop gain) and explicitly NOT hybrid (Run 6
        # cost multi-hop -5.4).
        (
            "multi_hop",
            RouteDecision(
                route="multi_hop",
                hybrid_retrieval=False,
                graphqa_fusion=True,
                verbatim_expansion=True,
                abstention_prompt=False,
                graph_traversal=False,
            ),
        ),
        # the abstention prompt is quarantined to unanswerable-risk queries
        # so it cannot over-abstain on answerable ones (Run 7: -1.6).
        (
            "unanswerable_risk",
            RouteDecision(
                route="unanswerable_risk",
                hybrid_retrieval=False,
                graphqa_fusion=False,
                verbatim_expansion=False,
                abstention_prompt=True,
                graph_traversal=False,
            ),
        ),
        # single-hop peaked on the plain dense extraction store (Run 5).
        (
            "single_hop",
            RouteDecision(
                route="single_hop",
                hybrid_retrieval=False,
                graphqa_fusion=False,
                verbatim_expansion=False,
                abstention_prompt=False,
                graph_traversal=False,
            ),
        ),
        # no measured winner for aggregative/open-domain yet: plain dense.
        (
            "aggregative",
            RouteDecision(
                route="aggregative",
                hybrid_retrieval=False,
                graphqa_fusion=False,
                verbatim_expansion=False,
                abstention_prompt=False,
                graph_traversal=False,
            ),
        ),
    ],
)
def test_route_feature_table(route: QueryRoute, expected: RouteDecision) -> None:
    assert RouteDecision.for_route(route, _settings()) == expected


def test_routed_multi_hop_honors_the_traversal_flag() -> None:
    # Given: entity traversal enabled globally alongside adaptive routing.
    settings = _settings(gnosis_graph_traversal_enabled=True)

    # When/Then: only the multi-hop route runs traversal; every other route
    # leaves it off even though the flag is on.
    assert RouteDecision.for_route("multi_hop", settings).graph_traversal is True
    assert RouteDecision.for_route("temporal", settings).graph_traversal is False
    assert RouteDecision.for_route("single_hop", settings).graph_traversal is False


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        # the bare token the prompt asks for (what gpt-5.5 via LiteLLM sends)
        ("temporal", "temporal"),
        ("multi_hop", "multi_hop"),
        # JSON-wrapped structured-output shape
        ('{"route": "unanswerable_risk"}', "unanswerable_risk"),
        # quoted / case / hyphen noise
        ('"aggregative"', "aggregative"),
        ("Multi-Hop", "multi_hop"),
        ("  single_hop\n", "single_hop"),
        # ambiguous or unrecognizable replies fall back
        ("either temporal or multi_hop", None),
        ("no idea", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_route_lenient(content: str | None, expected: QueryRoute | None) -> None:
    assert parse_route(content) == expected
