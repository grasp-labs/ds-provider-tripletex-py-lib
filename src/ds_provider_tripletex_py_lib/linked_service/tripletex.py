"""
**File:** ``tripletex.py``
**Region:** ``ds_provider_tripletex_py_lib/linked_service/tripletex``

Linked service for the Tripletex accounting API.

Tripletex authenticates via a two-step token exchange: a long-lived
*consumer token* + per-employee *employee token* are exchanged for a
short-lived *session token*, combined with ``client_id`` into an HTTP Basic
``Authorization`` header (``Basic base64(client_id:session_token)``). This
doesn't fit any of ``ds_protocol_http_py_lib``'s built-in ``AuthType``
handlers, so ``connect()`` is fully overridden; transport (retries,
timeouts, rate limiting) is still inherited from ``HttpLinkedService``.

Example:
    >>> linked_service = TripletexLinkedService(
    ...     settings=TripletexLinkedServiceSettings(
    ...         consumer_token="my-consumer-token",
    ...         employee_token="my-employee-token",
    ...         client_id="my-client-id",
    ...     ),
    ... )
    >>> linked_service.connect()
    >>> linked_service.connection.get(f"{linked_service.settings.host}/company")
    >>> linked_service.close()
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar

from ds_protocol_http_py_lib import HttpLinkedService, HttpLinkedServiceSettings
from ds_protocol_http_py_lib.enums import AuthType
from ds_resource_plugin_py_lib.common.resource.errors import ResourceException
from ds_resource_plugin_py_lib.common.resource.linked_service.errors import (
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
)

from ..enums import ResourceType


@dataclass(kw_only=True)
class TripletexLinkedServiceSettings(HttpLinkedServiceSettings):
    """
    Settings required to connect to the Tripletex API.

    Attributes:
        consumer_token: Long-lived consumer token issued to the integration. Masked in logs.
        employee_token: Per-employee token issued by the Tripletex user. Masked in logs.
        client_id: Client identifier combined with the session token to form the Basic
            auth header. Tripletex accepts ``"0"`` for the default/anonymous client.
        host: API host. Defaults to the production host; pass ``host=`` explicitly
            to target a different environment.
        session_token_ttl_days: Requested validity window, in days, for the session
            token created during ``connect()``.
        auth_type: Unused by this provider -- ``connect()`` is fully overridden and
            never dispatches through the base class's auth handlers. Kept only for
            schema/dataclass alignment with ``HttpLinkedServiceSettings``.
    """

    consumer_token: str = field(metadata={"mask": True})
    """Long-lived consumer token issued to the integration."""

    employee_token: str = field(metadata={"mask": True})
    """Per-employee token issued by the Tripletex user."""

    client_id: str = "0"
    """Client identifier combined with the session token to form the Basic auth header."""

    host: str = "https://tripletex.no/v2"
    """API host. Defaults to the Tripletex production environment."""

    session_token_ttl_days: int = 1
    """Requested validity window, in days, for the session token."""

    auth_type: AuthType = AuthType.NO_AUTH
    """Unused -- ``connect()`` is fully overridden. Kept for dataclass alignment."""


TripletexLinkedServiceSettingsType = TypeVar(
    "TripletexLinkedServiceSettingsType",
    bound=TripletexLinkedServiceSettings,
)


@dataclass(kw_only=True)
class TripletexLinkedService(
    HttpLinkedService[TripletexLinkedServiceSettingsType],
    Generic[TripletexLinkedServiceSettingsType],
):
    """
    Linked service for connecting to the Tripletex API.

    Exposed (caller-configurable): ``id``, ``name``, ``description``,
    ``version``, ``settings``. Internal (inherited from ``HttpLinkedService``,
    ``init=False``, never user-settable): ``_session``, ``_http`` -- runtime
    connection state populated by ``connect()``.
    """

    settings: TripletexLinkedServiceSettingsType

    @property
    def type(self) -> ResourceType:  # type: ignore[override]
        """Return the resource type for the Tripletex linked service."""
        return ResourceType.LINKED_SERVICE

    def connect(self) -> None:
        """
        Exchange consumer/employee tokens for a session token and build the auth header.

        ``PUT``s ``{host}/token/session/:create`` with ``consumerToken``/
        ``employeeToken``/``expirationDate``, reads the session token from
        ``value.token``, then stores ``Basic base64(client_id:session_token)``
        as the ``Authorization`` header. Idempotent -- each call creates a
        fresh session token.

        Raises:
            AuthenticationError: If Tripletex responds but rejects the
                consumer/employee token (any non-2xx response other than
                403, e.g. a revoked or malformed employee token returns
                404/422, not 401 -- Tripletex's own message is included), or
                if the session token is missing from an otherwise-successful
                response.
            AuthorizationError: If Tripletex returns 403 (authenticated but
                insufficient permission).
            ConnectionError: If the request to Tripletex fails at the
                network level, or Tripletex never responds at all (e.g. a
                timeout) -- distinct from a response that arrives but
                rejects the request.
        """
        if self._http is None:
            self._http = self._init_http()

        expiration_date = (datetime.now(tz=timezone.utc) + timedelta(days=self.settings.session_token_ttl_days)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

        try:
            response = self._http.put(
                url=f"{self.settings.host}/token/session/:create",
                params={
                    "consumerToken": self.settings.consumer_token,
                    "employeeToken": self.settings.employee_token,
                    "expirationDate": expiration_date,
                },
                headers={"Content-Type": "application/json"},
            )
        except (AuthenticationError, AuthorizationError, ConnectionError):
            raise  # ResourceException subclasses too; must bypass the broader except below
        except ResourceException as exc:
            if "response_body" not in exc.details:
                # No HTTP response was ever received.
                raise ConnectionError(
                    message=f"Could not reach Tripletex to create a session: {exc.message}",
                    details={"type": self.type.value, **exc.details},
                ) from exc
            # Token exchange was rejected.
            # A revoked/malformed employee token returns 404/422, not 401.
            response_body = exc.details.get("response_body") or "no response body"
            raise AuthenticationError(
                message=f"Tripletex rejected the session token exchange: {response_body}",
                details={"type": self.type.value, **exc.details},
            ) from exc
        body = response.json()

        try:
            session_token = body["value"]["token"]
        except (KeyError, TypeError) as exc:
            raise AuthenticationError(
                message="Session token is missing in the response from Tripletex",
                details={"type": self.type.value, "response_body": body},
            ) from exc

        credentials = f"{self.settings.client_id}:{session_token}".encode()
        basic_token = base64.b64encode(credentials).decode("ascii")
        self._http.session.headers.update({"Authorization": f"Basic {basic_token}"})

        self._session = self._http

    def test_connection(self) -> tuple[bool, str]:
        """
        Verify the Tripletex session by calling the lightweight identity endpoint.

        Returns:
            tuple[bool, str]: ``(True, "")`` on success, otherwise ``(False, error_message)``.
        """
        try:
            if self._http is None:
                self.connect()
            self.connection.get(f"{self.settings.host}/token/session/>whoAmI")
            return True, ""
        except Exception as exc:
            return False, str(exc)
