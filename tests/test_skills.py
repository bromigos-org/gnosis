from os import environ

import pytest

environ["AGENTS_MEMORY_TOKEN"] = "test-token"
environ["NEO4J_URI"] = "bolt://neo4j.neo4j.svc.cluster.local:7687"
environ["NEO4J_PASSWORD"] = "test-password"
environ["LITELLM_BASE_URL"] = "http://litellm.litellm.svc.cluster.local:4000/v1"
environ["LITELLM_API_KEY"] = "test-litellm-key"

from agents_memory.backend import Neo4jAgentMemoryBackend
from agents_memory.models import (
    ClientEvent,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
    GraphContextResponse,
    MemoryVisibility,
    SkillListRequest,
    SkillProposal,
    SkillRecord,
    SkillStatus,
    SkillUsage,
)
from agents_memory.settings import Settings
from agents_memory.skill_registry import InMemorySkillRegistry


@pytest.mark.anyio
async def test_approved_skill_is_returned() -> None:
    # Given: reviewed and unreviewed skills exist for the same agent.
    backend = _backend(
        SkillRecord(
            skill_id="skill-approved",
            tenant_id="bromigos",
            agent_id="pc-principal",
            name="Summarize channel",
            description="Summarize visible Discord channel context for review.",
            status=SkillStatus.APPROVED,
            metadata={"reviewed": True},
        ),
        SkillRecord(
            skill_id="skill-proposed",
            tenant_id="bromigos",
            agent_id="pc-principal",
            name="Draft skill",
            description="A draft that must not enter prompt context.",
            status=SkillStatus.PROPOSED,
            metadata={"reviewed": False},
        ),
    )

    # When: PC Principal asks for runnable skill context.
    response = await backend.list_skills(
        SkillListRequest(tenant_id="bromigos", agent_id="pc-principal"),
    )

    # Then: only the approved reviewed skill is returned.
    assert [skill.skill_id for skill in response.skills] == ["skill-approved"]
    assert response.skills[0].metadata == {"reviewed": True}


@pytest.mark.anyio
async def test_unapproved_proposal_is_not_runnable() -> None:
    # Given: a proposed skill has been recorded but not reviewed or approved.
    backend = _backend()
    proposal = _proposal()
    _ = await backend.propose_skill(proposal)

    # When: the agent lists skills and tries to record usage for the proposal id.
    list_response = await backend.list_skills(
        SkillListRequest(tenant_id="bromigos", agent_id="pc-principal"),
    )
    usage_response = await backend.record_skill_usage(
        SkillUsage(
            skill_id=proposal.proposal_id,
            tenant_id="bromigos",
            agent_id="pc-principal",
            used_by="789",
            used_at="2026-06-27T01:02:05Z",
            metadata={"outcome": "blocked"},
        ),
    )

    # Then: proposed skills never become runnable instructions or accepted usage.
    assert list_response.skills == []
    assert usage_response.status == EventIngestStatus.REJECTED
    assert usage_response.reason == "skill is not approved"


@pytest.mark.anyio
async def test_skill_proposal_is_persisted_for_review() -> None:
    # Given: a skill registry with no proposals yet.
    registry = InMemorySkillRegistry()
    backend = _backend(registry=registry)
    proposal = _proposal()

    # When: PC Principal proposes a skill.
    response = await backend.propose_skill(proposal)

    # Then: the proposal is stored for review without creating an approved skill.
    assert response == proposal
    assert registry.proposals == (proposal,)
    skills = await backend.list_skills(
        SkillListRequest(tenant_id="bromigos", agent_id="pc-principal"),
    )
    assert skills.skills == []


@pytest.mark.anyio
async def test_approved_skill_usage_is_recorded() -> None:
    # Given: a reviewed approved skill exists.
    registry = InMemorySkillRegistry(
        records=(
            SkillRecord(
                skill_id="skill-approved",
                tenant_id="bromigos",
                agent_id="pc-principal",
                name="Summarize channel",
                description="Summarize visible Discord channel context for review.",
                status=SkillStatus.APPROVED,
                scope=MemoryVisibility.AGENT_SHARED,
                metadata={"reviewed": True},
            ),
        ),
    )
    backend = _backend(registry=registry)
    usage = SkillUsage(
        skill_id="skill-approved",
        tenant_id="bromigos",
        agent_id="pc-principal",
        used_by="789",
        used_at="2026-06-27T01:02:05Z",
        metadata={"outcome": "ok"},
    )

    # When: usage is recorded for the approved skill.
    response = await backend.record_skill_usage(usage)

    # Then: the usage event is accepted and persisted.
    assert response.status == EventIngestStatus.ACCEPTED
    assert registry.usages == (usage,)


def _backend(
    *records: SkillRecord,
    registry: InMemorySkillRegistry | None = None,
) -> Neo4jAgentMemoryBackend:
    return Neo4jAgentMemoryBackend(
        Settings(),
        graph_store=AvailableGraphStore(),
        skill_registry=registry or InMemorySkillRegistry(records=records),
    )


def _proposal() -> SkillProposal:
    return SkillProposal(
        proposal_id="proposal-1",
        tenant_id="bromigos",
        agent_id="pc-principal",
        proposed_by="789",
        name="Summarize channel",
        description="Summarize visible Discord channel context for review.",
        metadata={"source": "conversation"},
    )


class AvailableGraphStore:
    async def require_available(self) -> None:
        return None

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        return EventIngestResult(
            event_id=event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(
        self,
        request: GraphContextRequest,
    ) -> GraphContextResponse:
        _ = request
        return GraphContextResponse(context="")
