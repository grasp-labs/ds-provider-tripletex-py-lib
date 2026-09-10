"""
**File:** ``test_tripletex_linked_service.py``
**Region:** ``tests/linked_service``

Unit tests for TripletexLinkedService and TripletexLinkedServiceSettings.
"""

import base64
import json
from uuid import uuid4

import pytest
from ds_resource_plugin_py_lib.common.resource.errors import ResourceException
from ds_resource_plugin_py_lib.common.resource.linked_service.errors import (
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
)

from ds_provider_tripletex_py_lib.enums import ResourceType
from ds_provider_tripletex_py_lib.linked_service.tripletex import (
    TripletexLinkedService,
    TripletexLinkedServiceSettings,
)


class FakeSession:
    """Mock ``requests.Session``-like object exposing only ``headers``."""

    def __init__(self):
        self.headers: dict[str, str] = {}


class FakeResponse:
    """Mock HTTP response."""

    def __init__(self, json_data):
        self._json = json_data

    def json(self):
        return self._json


class FakeHttp:
    """Mock ``Http`` provider recording calls and returning canned responses."""

    def __init__(self, put_response=None, get_response=None, put_exception=None):
        self.session = FakeSession()
        self._put_response = put_response
        self._get_response = get_response
        self._put_exception = put_exception
        self.put_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self.closed = False

    def put(self, url, **kwargs):
        self.put_calls.append({"url": url, **kwargs})
        if self._put_exception is not None:
            raise self._put_exception
        return self._put_response

    def get(self, url, **kwargs):
        self.get_calls.append({"url": url, **kwargs})
        return self._get_response

    def close(self):
        self.closed = True


def make_settings(**overrides):
    """Create TripletexLinkedServiceSettings with sane test defaults."""
    defaults = {"consumer_token": "consumer-token", "employee_token": "employee-token"}
    defaults.update(overrides)
    return TripletexLinkedServiceSettings(**defaults)


def make_service(**overrides) -> TripletexLinkedService:
    """Create a TripletexLinkedService instance for testing."""
    return TripletexLinkedService(
        settings=make_settings(**overrides),
        id=uuid4(),
        name="test",
        version="1.0.0",
    )


# -----------------------------------------------------------------------------
# Settings defaults
# -----------------------------------------------------------------------------


def test_settings_defaults():
    """It initializes settings with sane defaults."""
    settings = make_settings()
    assert settings.client_id == "0"
    assert settings.host == "https://tripletex.no/v2"
    assert settings.session_token_ttl_days == 1


def test_settings_host_override():
    """A custom host can be passed explicitly to override the production default."""
    settings = make_settings(host="https://custom.example.com")
    assert settings.host == "https://custom.example.com"


def test_settings_masks_tokens():
    """consumer_token and employee_token are marked for masking in logs."""
    fields = {f.name: f for f in TripletexLinkedServiceSettings.__dataclass_fields__.values()}
    assert fields["consumer_token"].metadata.get("mask") is True
    assert fields["employee_token"].metadata.get("mask") is True


# -----------------------------------------------------------------------------
# Contract: schema/dataclass alignment -- a representative payload must
# deserialize successfully (LINKED_SERVICE_CONTRACT.md, "Construction guarantee")
# -----------------------------------------------------------------------------


def test_deserialize_representative_payload():
    """A representative schema-valid payload constructs successfully via deserialize()."""
    payload = {
        "id": str(uuid4()),
        "name": "tripletex",
        "version": "1.0.0",
        "settings": {"consumer_token": "my-consumer-token", "employee_token": "my-employee-token"},
    }

    service = TripletexLinkedService.deserialize(payload)

    assert service.settings.consumer_token == "my-consumer-token"
    assert service.settings.employee_token == "my-employee-token"
    assert service.settings.host == "https://tripletex.no/v2"  # default applied


# -----------------------------------------------------------------------------
# type property
# -----------------------------------------------------------------------------


def test_type_property():
    """type property returns the Tripletex linked-service resource type."""
    service = make_service()
    assert service.type == ResourceType.LINKED_SERVICE


# -----------------------------------------------------------------------------
# connect() behavior
# -----------------------------------------------------------------------------


def test_connect_sets_basic_auth_header():
    """connect() exchanges tokens for a session token and sets a Basic auth header."""
    service = make_service(client_id="42")
    service._http = FakeHttp(put_response=FakeResponse({"value": {"token": "session-abc"}}))

    service.connect()

    expected = base64.b64encode(b"42:session-abc").decode("ascii")
    assert service._http.session.headers["Authorization"] == f"Basic {expected}"


def test_connect_calls_session_create_endpoint():
    """connect() PUTs the Tripletex session-create endpoint with the expected params."""
    service = make_service()
    fake_http = FakeHttp(put_response=FakeResponse({"value": {"token": "tok"}}))
    service._http = fake_http

    service.connect()

    call = fake_http.put_calls[0]
    assert call["url"] == "https://tripletex.no/v2/token/session/:create"
    assert call["params"]["consumerToken"] == "consumer-token"
    assert call["params"]["employeeToken"] == "employee-token"
    assert "expirationDate" in call["params"]


def test_connect_raises_authentication_error_when_token_missing():
    """connect() raises AuthenticationError when the response has no session token."""
    service = make_service()
    service._http = FakeHttp(put_response=FakeResponse({"value": {}}))

    with pytest.raises(AuthenticationError):
        service.connect()


def test_connect_raises_authentication_error_when_value_missing():
    """connect() raises AuthenticationError when the response has no 'value' key."""
    service = make_service()
    service._http = FakeHttp(put_response=FakeResponse({}))

    with pytest.raises(AuthenticationError):
        service.connect()


def test_connect_wraps_resource_exception_as_authentication_error():
    """A non-401/403 failure (e.g. a revoked employee token) becomes AuthenticationError."""
    service = make_service()
    response_body = json.dumps({"status": 404, "code": 6000, "message": "Object not found"})
    service._http = FakeHttp(
        put_exception=ResourceException(
            message="HTTP error: 404 Client Error: Not Found",
            status_code=404,
            details={"response_body": response_body},
        )
    )

    with pytest.raises(AuthenticationError, match="Object not found"):
        service.connect()


def test_connect_raises_connection_error_when_no_response_received():
    """A ResourceException with no response_body (e.g. a timeout) becomes ConnectionError, not AuthenticationError."""
    service = make_service()
    service._http = FakeHttp(
        put_exception=ResourceException(
            message="HTTP error: ReadTimeout",
            status_code=500,  # the ResourceException default, not a real Tripletex response
            details={"url": "https://tripletex.no/v2/token/session/:create", "error_type": "ReadTimeout"},
        )
    )

    with pytest.raises(ConnectionError, match="ReadTimeout"):
        service.connect()


def test_connect_reports_no_response_body_when_empty():
    """A real HTTP response with an empty body still produces a clear AuthenticationError message."""
    service = make_service()
    service._http = FakeHttp(
        put_exception=ResourceException(
            message="HTTP error: 404 Client Error: Not Found",
            status_code=404,
            details={"response_body": ""},
        )
    )

    with pytest.raises(AuthenticationError, match="no response body"):
        service.connect()


def test_connect_passes_through_authentication_error():
    """A 401 from the underlying HTTP client is re-raised unchanged, not double-wrapped."""
    service = make_service()
    original = AuthenticationError(message="Authentication error: 401", details={})
    service._http = FakeHttp(put_exception=original)

    with pytest.raises(AuthenticationError) as exc_info:
        service.connect()
    assert exc_info.value is original


def test_connect_passes_through_authorization_error():
    """A 403 from the underlying HTTP client is re-raised unchanged."""
    service = make_service()
    original = AuthorizationError(message="Authorization error: 403", details={})
    service._http = FakeHttp(put_exception=original)

    with pytest.raises(AuthorizationError) as exc_info:
        service.connect()
    assert exc_info.value is original


def test_connect_passes_through_connection_error():
    """A network-level failure from the underlying HTTP client is re-raised unchanged."""
    service = make_service()
    original = ConnectionError(message="Connection error", details={})
    service._http = FakeHttp(put_exception=original)

    with pytest.raises(ConnectionError) as exc_info:
        service.connect()
    assert exc_info.value is original


def test_connect_is_idempotent_and_reinitializes_http_when_needed():
    """connect() (re)initializes the HTTP client when it is missing."""
    service = make_service()
    service._http = None
    service._session = None

    # Real _init_http() will run; we still need connect() to reach the PUT call,
    # so swap in a fake immediately after initialization is triggered.
    original_init_http = service._init_http

    def fake_init_http():
        http = FakeHttp(put_response=FakeResponse({"value": {"token": "tok"}}))
        return http

    service._init_http = fake_init_http
    service.connect()
    service._init_http = original_init_http

    assert service.connection is not None


# -----------------------------------------------------------------------------
# test_connection() behavior
# -----------------------------------------------------------------------------


def test_connection_success():
    """test_connection() returns (True, '') when the identity probe succeeds."""
    service = make_service()
    service._http = FakeHttp(
        put_response=FakeResponse({"value": {"token": "tok"}}),
        get_response=FakeResponse({"id": 1}),
    )
    service.connect()

    success, message = service.test_connection()

    assert success is True
    assert message == ""


def test_connection_calls_who_am_i_endpoint():
    """test_connection() probes the whoAmI endpoint."""
    service = make_service()
    fake_http = FakeHttp(
        put_response=FakeResponse({"value": {"token": "tok"}}),
        get_response=FakeResponse({"id": 1}),
    )
    service._http = fake_http
    service.connect()

    service.test_connection()

    assert fake_http.get_calls[0]["url"] == "https://tripletex.no/v2/token/session/>whoAmI"


def test_connection_failure_returns_message():
    """test_connection() returns (False, message) when the probe raises."""
    service = make_service()

    class RaisingHttp(FakeHttp):
        def get(self, url, **kwargs):
            raise RuntimeError("boom")

    service._http = RaisingHttp(put_response=FakeResponse({"value": {"token": "tok"}}))
    service.connect()

    success, message = service.test_connection()

    assert success is False
    assert "boom" in message


def test_connection_connects_when_not_already_connected():
    """test_connection() calls connect() itself when _http is not initialized."""
    service = make_service()
    service._http = None
    service._session = None

    def fake_init_http():
        return FakeHttp(
            put_response=FakeResponse({"value": {"token": "tok"}}),
            get_response=FakeResponse({"id": 1}),
        )

    service._init_http = fake_init_http
    success, message = service.test_connection()

    assert success is True
    assert message == ""


# -----------------------------------------------------------------------------
# close() behavior
# -----------------------------------------------------------------------------


def test_close_is_idempotent():
    """close() does not raise, including when called before connect()."""
    service = make_service()
    service.close()
    service.close()


def test_close_closes_http_client():
    """close() closes the underlying HTTP client."""
    service = make_service()
    fake_http = FakeHttp(put_response=FakeResponse({"value": {"token": "tok"}}))
    service._http = fake_http
    service.connect()

    service.close()

    assert fake_http.closed is True
