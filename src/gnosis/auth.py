from dataclasses import dataclass
from enum import StrEnum
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gnosis.models import MemoryScope
from gnosis.settings import Settings

_bearer = HTTPBearer(auto_error=False)


class MemoryCaller(StrEnum):
    SERVICE = "service"
    FEDERATED = "federated"


@dataclass(frozen=True, slots=True)
class Authenticator:
    token: str
    federation_token: str
    read_operator_token: str
    export_operator_token: str
    write_operator_token: str
    admin_operator_token: str
    tenant_id: str

    def require_token(
        self,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_bearer),
        ],
    ) -> None:
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
            )
        if compare_digest(credentials.credentials, self.token):
            return
        if self._matches_federation_token(credentials.credentials):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="federation token is not authorized for this route",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
        )

    def resolve_memory_caller(
        self,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_bearer),
        ],
    ) -> MemoryCaller:
        if credentials is not None:
            if compare_digest(credentials.credentials, self.token):
                return MemoryCaller.SERVICE
            if self._matches_federation_token(credentials.credentials):
                return MemoryCaller.FEDERATED
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
        )

    def _matches_federation_token(self, candidate: str) -> bool:
        return bool(self.federation_token) and compare_digest(
            candidate,
            self.federation_token,
        )

    def require_read_operator(
        self,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_bearer),
        ],
    ) -> None:
        self._require_operator_token(credentials, self.read_operator_token)

    def require_export_operator(
        self,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_bearer),
        ],
    ) -> None:
        self._require_operator_token(credentials, self.export_operator_token)

    def require_write_operator(
        self,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_bearer),
        ],
    ) -> None:
        self._require_operator_token(credentials, self.write_operator_token)

    def require_admin_operator(
        self,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_bearer),
        ],
    ) -> None:
        self._require_operator_token(credentials, self.admin_operator_token)

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

    def _require_operator_token(
        self,
        credentials: HTTPAuthorizationCredentials | None,
        token: str,
    ) -> None:
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
            )
        if not compare_digest(credentials.credentials, token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="operator token class is not authorized",
            )


def build_authenticator(settings: Settings) -> Authenticator:
    return Authenticator(
        token=settings.gnosis_token,
        federation_token=settings.gnosis_federation_token,
        read_operator_token=settings.gnosis_read_operator_token,
        export_operator_token=settings.gnosis_export_operator_token,
        write_operator_token=settings.gnosis_write_operator_token,
        admin_operator_token=settings.gnosis_admin_operator_token,
        tenant_id=settings.gnosis_tenant_id,
    )
