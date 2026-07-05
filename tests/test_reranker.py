"""Listwise LLM reranker: reordering algebra, helpers, and backend gating."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from gnosis.backend import Neo4jAgentMemoryBackend
from gnosis.models import JsonObject, JsonValue
from gnosis.reranker import (
    RerankResult,
    apply_rerank,
    rerank_candidate_cap,
    rerank_model,
)
from gnosis.settings import Settings


def _settings(**overrides: JsonValue) -> Settings:
    values: dict[str, JsonValue] = {
        "gnosis_token": "value",
        "gnosis_read_operator_token": "value",
        "gnosis_export_operator_token": "value",
        "gnosis_write_operator_token": "value",
        "gnosis_admin_operator_token": "value",
        "gnosis_tenant_id": "bromigos",
        "neo4j_uri": "bolt://neo4j.local:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "value",
        "litellm_base_url": "http://litellm.local/v1",
        "litellm_api_key": "value",
    }
    values.update(overrides)
    return Settings(**values)  # pyright: ignore[reportArgumentType]


_RERANK_FAILURE = "reranker upstream failure"


def _facts(*ids: str) -> list[JsonObject]:
    facts: list[JsonObject] = []
    for identifier in ids:
        fact: JsonObject = {
            "id": identifier,
            "subject": "",
            "predicate": "memory",
            "object": identifier,
            "metadata": {},
        }
        facts.append(fact)
    return facts


def _ids(facts: list[JsonObject]) -> list[str]:
    return [str(fact["id"]) for fact in facts]


@dataclass
class _FakeReranker:
    result: RerankResult | None = None
    raises: bool = False
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    async def rerank(
        self,
        query: str,
        candidates: Sequence[str],
    ) -> RerankResult | None:
        self.calls.append((query, list(candidates)))
        if self.raises:
            raise RuntimeError(_RERANK_FAILURE)
        return self.result


# --- apply_rerank: pure reordering algebra --------------------------------


def test_apply_rerank_reorders_by_given_order() -> None:
    out = apply_rerank(_facts("a", "b", "c"), [2, 0, 1], cap=50)
    assert _ids(out) == ["c", "a", "b"]


def test_apply_rerank_ignores_out_of_range_and_duplicate_indices() -> None:
    # 2 then 0 are valid & unique; 2 repeat, 9 and -1 out of range -> ignored;
    # index 1 ("b") never named -> appended in original order.
    out = apply_rerank(_facts("a", "b", "c"), [2, 2, 9, -1, 0], cap=50)
    assert _ids(out) == ["c", "a", "b"]


def test_apply_rerank_appends_omitted_candidates_in_original_order() -> None:
    out = apply_rerank(_facts("a", "b", "c", "d"), [3], cap=50)
    assert _ids(out) == ["d", "a", "b", "c"]


def test_apply_rerank_empty_order_preserves_and_drops_nothing() -> None:
    facts = _facts("a", "b", "c", "d", "e")
    out = apply_rerank(facts, [], cap=50)
    assert _ids(out) == ["a", "b", "c", "d", "e"]
    assert len(out) == len(facts)


def test_apply_rerank_only_reorders_within_cap_tail_kept() -> None:
    # cap=3: only a,b,c are reranked; d,e stay after the reordered head.
    out = apply_rerank(_facts("a", "b", "c", "d", "e"), [2, 1, 0], cap=3)
    assert _ids(out) == ["c", "b", "a", "d", "e"]


def test_apply_rerank_index_into_tail_is_ignored() -> None:
    # cap=2: head=a,b. index 3 points into the tail -> out of head range -> ignored.
    out = apply_rerank(_facts("a", "b", "c", "d"), [1, 3, 0], cap=2)
    assert _ids(out) == ["b", "a", "c", "d"]


# --- settings helpers ------------------------------------------------------


def test_rerank_model_falls_back_to_main_llm() -> None:
    assert rerank_model(_settings(gnosis_llm="openai/gpt-5.5")) == "openai/gpt-5.5"


def test_rerank_model_prefers_explicit_override() -> None:
    settings = _settings(gnosis_llm="openai/gpt-5.5", gnosis_rerank_model="openai/rank")
    assert rerank_model(settings) == "openai/rank"


def test_rerank_candidate_cap_reads_setting() -> None:
    assert rerank_candidate_cap(_settings(gnosis_rerank_candidate_cap=25)) == 25


# --- backend gating: _reranked_facts --------------------------------------


def _backend(*, enabled: bool, reranker: _FakeReranker) -> Neo4jAgentMemoryBackend:
    return Neo4jAgentMemoryBackend(
        _settings(gnosis_rerank_enabled=enabled),
        reranker=reranker,
    )


async def _rerank(
    backend: Neo4jAgentMemoryBackend,
    query: str,
    facts: list[JsonObject],
) -> list[JsonObject]:
    return await backend._reranked_facts(query, facts)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_reranked_facts_noop_when_flag_off() -> None:
    fake = _FakeReranker(result=RerankResult(order=[2, 1, 0]))
    backend = _backend(enabled=False, reranker=fake)
    out = await _rerank(backend, "q", _facts("a", "b", "c"))
    assert _ids(out) == ["a", "b", "c"]
    assert fake.calls == []  # never invoked when disabled


@pytest.mark.anyio
async def test_reranked_facts_applies_order_when_enabled() -> None:
    fake = _FakeReranker(result=RerankResult(order=[2, 1, 0]))
    backend = _backend(enabled=True, reranker=fake)
    out = await _rerank(backend, "q", _facts("a", "b", "c"))
    assert _ids(out) == ["c", "b", "a"]
    assert len(fake.calls) == 1


@pytest.mark.anyio
async def test_reranked_facts_keeps_order_on_failure() -> None:
    fake = _FakeReranker(raises=True)
    backend = _backend(enabled=True, reranker=fake)
    out = await _rerank(backend, "q", _facts("a", "b", "c"))
    assert _ids(out) == ["a", "b", "c"]


@pytest.mark.anyio
async def test_reranked_facts_keeps_order_when_model_returns_none() -> None:
    fake = _FakeReranker(result=None)
    backend = _backend(enabled=True, reranker=fake)
    out = await _rerank(backend, "q", _facts("a", "b", "c"))
    assert _ids(out) == ["a", "b", "c"]


@pytest.mark.anyio
async def test_reranked_facts_noop_on_empty_query_or_single_candidate() -> None:
    fake = _FakeReranker(result=RerankResult(order=[1, 0]))
    backend = _backend(enabled=True, reranker=fake)
    assert _ids(await _rerank(backend, "", _facts("a", "b"))) == ["a", "b"]
    assert _ids(await _rerank(backend, "q", _facts("a"))) == ["a"]
    assert fake.calls == []
