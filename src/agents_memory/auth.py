from dataclasses import dataclass
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agents_memory.models import MemoryScope
from agents_memory.settings import Settings

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class Authenticator:
    token: str
    tenant_id: str

    def require_token(
        self,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_bearer),
        ],
    ) -> None:
        authorized = credentials is not None and compare_digest(
            credentials.credentials,
            self.token,
        )
        if not authorized:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
            )

    def require_scope(self, scope: MemoryScope) -> None:
        if not compare_digest(scope.tenant_id, self.tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="scope is not authorized for this token",
            )

    def require_tenant(self, tenant_id: str) -> None:
        if not compare_digest(tenant_id, self.tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="scope is not authorized for this token",
            )


def build_authenticator(settings: Settings) -> Authenticator:
    return Authenticator(
        token=settings.agents_memory_token,
        tenant_id=settings.agents_memory_tenant_id,
    )
