from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from gnosis.models import (
    EventIngestResult,
    EventIngestStatus,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillRecord,
    SkillStatus,
    SkillUsage,
)


class SkillRegistry(Protocol):
    async def list_skills(self, request: SkillListRequest) -> SkillListResponse: ...
    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal: ...
    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult: ...


@dataclass(slots=True)
class InMemorySkillRegistry:
    _records: list[SkillRecord] = field(default_factory=list)
    _proposals: list[SkillProposal] = field(default_factory=list)
    _usages: list[SkillUsage] = field(default_factory=list)

    def __init__(self, records: Iterable[SkillRecord] = ()) -> None:
        self._records = list(records)
        self._proposals = []
        self._usages = []

    @property
    def proposals(self) -> tuple[SkillProposal, ...]:
        return tuple(self._proposals)

    @property
    def usages(self) -> tuple[SkillUsage, ...]:
        return tuple(self._usages)

    async def list_skills(self, request: SkillListRequest) -> SkillListResponse:
        return SkillListResponse(
            skills=[
                record
                for record in self._records
                if _is_runnable_skill(record, request)
            ],
        )

    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal:
        self._proposals.append(proposal)
        return proposal

    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult:
        if not self._is_approved_skill(usage):
            return EventIngestResult(
                event_id=usage.skill_id,
                status=EventIngestStatus.REJECTED,
                reason="skill is not approved",
            )
        self._usages.append(usage)
        return EventIngestResult(
            event_id=usage.skill_id,
            status=EventIngestStatus.ACCEPTED,
        )

    def _is_approved_skill(self, usage: SkillUsage) -> bool:
        return any(
            _is_runnable_skill(
                record,
                SkillListRequest(tenant_id=usage.tenant_id, agent_id=usage.agent_id),
            )
            and record.skill_id == usage.skill_id
            for record in self._records
        )


def _is_runnable_skill(record: SkillRecord, request: SkillListRequest) -> bool:
    return (
        record.tenant_id == request.tenant_id
        and record.agent_id == request.agent_id
        and record.status is SkillStatus.APPROVED
        and record.metadata.get("reviewed") is True
    )
