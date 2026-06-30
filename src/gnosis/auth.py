from dataclasses import dataclass
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gnosis.models import MemoryScope
from gnosis.settings import Settings

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class Authenticator:
    token: str
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
        authorized = credentials is not None and compare_digest(
            credentials.credentials,
            self.token,
        )
        if not authorized:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
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
        read_operator_token=settings.gnosis_read_operator_token,
        export_operator_token=settings.gnosis_export_operator_token,
        write_operator_token=settings.gnosis_write_operator_token,
        admin_operator_token=settings.gnosis_admin_operator_token,
        tenant_id=settings.gnosis_tenant_id,
    )
