# -*- coding: utf-8 -*-
"""Location: ./tests/unit/mcpgateway/routers/test_oauth_as_router.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0

Unit tests for OAuth 2.1 Authorization Server router.
Tests HTTP interface correctness, status codes, response shapes, and error handling.
The service layer is mocked to isolate router logic from real crypto/DB operations.
"""

# Standard
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Third-Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.orm import Session

# First-Party
from mcpgateway.routers.oauth_as import oauth_as_router
from mcpgateway.services.oauth_as_service import InvalidClientError, InvalidScopeError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic_auth_header(client_id: str, client_secret: str) -> str:
    """Build an HTTP Basic Authorization header value."""
    encoded = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    return f"Basic {encoded}"


def _admin_user_ctx(is_admin: bool = True):
    """Return a mock current_user_ctx dict for dependency override."""
    user = SimpleNamespace(is_admin=is_admin)
    return {"user": user}


def _mock_db_session():
    """Yield a mock DB session for dependency override."""
    yield MagicMock(spec=Session)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service():
    """Create a mock OAuthASService with sensible default return values."""
    svc = MagicMock()

    # Public endpoints
    svc.check_rate_limit.return_value = True
    svc.get_rate_limit_retry_after.return_value = 30
    svc.authenticate_client = AsyncMock(
        return_value=SimpleNamespace(
            client_id="test-client",
            client_name="Test Client",
            scopes=["read", "write"],
            teams=["team-a"],
            is_admin=False,
            is_active=True,
        )
    )
    svc.issue_token.return_value = {
        "access_token": "eyJhbGciOiJSUzI1NiJ9.test-token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": "read write",
    }
    svc.get_jwks.return_value = {
        "keys": [
            {
                "kty": "RSA",
                "kid": "test-kid-1",
                "use": "sig",
                "alg": "RS256",
                "n": "0vx7agoebGc...",
                "e": "AQAB",
            }
        ]
    }
    svc.get_as_metadata.return_value = {
        "issuer": "http://testserver",
        "token_endpoint": "http://testserver/oauth/token",
        "jwks_uri": "http://testserver/oauth/jwks",
        "grant_types_supported": ["client_credentials"],
        "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        "scopes_supported": ["read", "write"],
    }

    # Admin endpoints
    svc.register_client = AsyncMock(
        return_value={
            "client_id": "new-client",
            "client_name": "New Client",
            "client_secret": "generated-secret-abc123",
            "scopes": ["read"],
            "teams": [],
            "is_admin": False,
            "is_active": True,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        }
    )
    svc.list_clients = AsyncMock(
        return_value=[
            {
                "client_id": "client-1",
                "client_name": "Client One",
                "scopes": ["read"],
                "teams": [],
                "is_admin": False,
                "is_active": True,
            }
        ]
    )
    svc.get_client = AsyncMock(
        return_value={
            "client_id": "client-1",
            "client_name": "Client One",
            "scopes": ["read"],
            "teams": [],
            "is_admin": False,
            "is_active": True,
        }
    )
    svc.deactivate_client = AsyncMock(return_value=True)
    svc.rotate_client_secret = AsyncMock(
        return_value={
            "client_id": "client-1",
            "client_secret": "new-rotated-secret-xyz",
        }
    )
    svc.revoke_token = AsyncMock(return_value=True)
    svc.register_dcr_client = AsyncMock(
        return_value={
            "client_id": "dcr-abc123def456",
            "client_secret": "dcr-generated-secret",
            "client_id_issued_at": 1711468800,
            "client_secret_expires_at": 0,
            "client_name": "My MCP Client",
            "grant_types": ["client_credentials"],
            "token_endpoint_auth_method": "client_secret_basic",
            "scope": "mcp:access",
        }
    )

    return svc


@pytest.fixture
def client(mock_service):
    """Create a TestClient with a minimal app that includes the OAuth AS router.

    The router's dependencies are overridden:
    - settings.oauth_as_enabled = True
    - get_oauth_as_service returns mock_service
    - get_current_user_with_permissions returns admin context
    - _get_db returns a mock session
    """
    from mcpgateway.middleware.rbac import get_current_user_with_permissions
    from mcpgateway.routers.oauth_as import _get_db
    from mcpgateway.routers.well_known import router as well_known_router

    test_app = FastAPI()
    # well_known router must be included first for /.well-known/oauth-authorization-server
    test_app.include_router(well_known_router)
    test_app.include_router(oauth_as_router)

    # Override route-level dependencies
    test_app.dependency_overrides[get_current_user_with_permissions] = lambda: _admin_user_ctx(is_admin=True)
    test_app.dependency_overrides[_get_db] = _mock_db_session

    with (
        patch("mcpgateway.config.settings.oauth_as_enabled", True),
        patch("mcpgateway.config.settings.well_known_enabled", True),
        patch("mcpgateway.services.oauth_as_service.get_oauth_as_service", return_value=mock_service),
    ):
        yield TestClient(test_app)


@pytest.fixture
def client_disabled():
    """Create a TestClient where oauth_as_enabled=False."""
    from mcpgateway.middleware.rbac import get_current_user_with_permissions
    from mcpgateway.routers.oauth_as import _get_db
    from mcpgateway.routers.well_known import router as well_known_router

    test_app = FastAPI()
    test_app.include_router(well_known_router)
    test_app.include_router(oauth_as_router)

    test_app.dependency_overrides[get_current_user_with_permissions] = lambda: _admin_user_ctx(is_admin=True)
    test_app.dependency_overrides[_get_db] = _mock_db_session

    with (
        patch("mcpgateway.config.settings.oauth_as_enabled", False),
        patch("mcpgateway.config.settings.well_known_enabled", True),
    ):
        yield TestClient(test_app, raise_server_exceptions=False)


@pytest.fixture
def client_non_admin(mock_service):
    """Create a TestClient where the authenticated user is NOT admin."""
    from mcpgateway.middleware.rbac import get_current_user_with_permissions
    from mcpgateway.routers.oauth_as import _get_db

    test_app = FastAPI()
    test_app.include_router(oauth_as_router)

    test_app.dependency_overrides[get_current_user_with_permissions] = lambda: _admin_user_ctx(is_admin=False)
    test_app.dependency_overrides[_get_db] = _mock_db_session

    with (
        patch("mcpgateway.config.settings.oauth_as_enabled", True),
        patch("mcpgateway.services.oauth_as_service.get_oauth_as_service", return_value=mock_service),
    ):
        yield TestClient(test_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# POST /oauth/token
# ---------------------------------------------------------------------------


class TestOAuthToken:
    """Tests for the POST /oauth/token endpoint."""

    def test_token_success_form_body(self, client, mock_service):
        """Valid client_credentials grant via form body returns 200 with token."""
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "test-client",
                "client_secret": "test-secret",
                "scope": "read write",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "Bearer"
        assert body["expires_in"] == 3600
        assert body["scope"] == "read write"
        # Cache-Control: no-store per OAuth 2.0 spec
        assert resp.headers.get("cache-control") == "no-store"

    def test_token_success_basic_auth(self, client, mock_service):
        """Valid client_credentials grant via HTTP Basic auth returns 200."""
        resp = client.post(
            "/oauth/token",
            data={"grant_type": "client_credentials", "scope": "read"},
            headers={"Authorization": _basic_auth_header("test-client", "test-secret")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "Bearer"

    def test_token_unsupported_grant_type(self, client):
        """Non-client_credentials grant_type returns 400 unsupported_grant_type."""
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": "test-client",
                "client_secret": "test-secret",
            },
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"] == "unsupported_grant_type"

    def test_token_missing_grant_type(self, client):
        """Missing grant_type returns 400 unsupported_grant_type."""
        resp = client.post(
            "/oauth/token",
            data={
                "client_id": "test-client",
                "client_secret": "test-secret",
            },
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"] == "unsupported_grant_type"

    def test_token_missing_credentials(self, client):
        """Missing both form credentials and Basic auth returns 401."""
        resp = client.post(
            "/oauth/token",
            data={"grant_type": "client_credentials"},
        )
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"] == "invalid_client"

    def test_token_invalid_client(self, client, mock_service):
        """Invalid client credentials returns 401 invalid_client."""
        mock_service.authenticate_client = AsyncMock(side_effect=InvalidClientError("bad creds"))
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "bad-client",
                "client_secret": "wrong-secret",
            },
        )
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"] == "invalid_client"
        assert "error_description" in body

    def test_token_invalid_scope(self, client, mock_service):
        """Requesting a scope the client is not allowed returns 400 invalid_scope."""
        mock_service.issue_token.side_effect = InvalidScopeError("scope 'admin' not allowed")
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "test-client",
                "client_secret": "test-secret",
                "scope": "admin",
            },
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"] == "invalid_scope"
        assert "error_description" in body

    def test_token_rate_limited(self, client, mock_service):
        """Rate-limited request returns 429 with Retry-After header."""
        mock_service.check_rate_limit.return_value = False
        mock_service.get_rate_limit_retry_after.return_value = 60
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "test-client",
                "client_secret": "test-secret",
            },
        )
        assert resp.status_code == 429
        body = resp.json()
        assert body["error"] == "slow_down"
        assert resp.headers.get("retry-after") == "60"

    def test_token_feature_disabled(self, client_disabled):
        """When oauth_as_enabled=False, /oauth/token returns 404."""
        resp = client_disabled.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "test-client",
                "client_secret": "test-secret",
            },
        )
        assert resp.status_code == 404

    def test_token_basic_auth_malformed(self, client):
        """Malformed Basic auth header falls through to 401."""
        resp = client.post(
            "/oauth/token",
            data={"grant_type": "client_credentials"},
            headers={"Authorization": "Basic not-valid-base64!!!"},
        )
        assert resp.status_code == 401

    def test_token_basic_auth_no_colon(self, client):
        """Basic auth without colon separator falls through to 401."""
        encoded = base64.b64encode(b"no-colon-here").decode()
        resp = client.post(
            "/oauth/token",
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {encoded}"},
        )
        assert resp.status_code == 401

    def test_token_no_scope_omits_scope_field(self, client, mock_service):
        """When no scope is requested and none returned, scope may be absent."""
        mock_service.issue_token.return_value = {
            "access_token": "eyJhbGciOiJSUzI1NiJ9.no-scope",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "test-client",
                "client_secret": "test-secret",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "scope" not in body or body.get("scope") is None


# ---------------------------------------------------------------------------
# GET /oauth/jwks
# ---------------------------------------------------------------------------


class TestOAuthJWKS:
    """Tests for the GET /oauth/jwks endpoint."""

    def test_jwks_returns_keys(self, client):
        """JWKS endpoint returns a keys array."""
        resp = client.get("/oauth/jwks")
        assert resp.status_code == 200
        body = resp.json()
        assert "keys" in body
        assert isinstance(body["keys"], list)
        assert len(body["keys"]) >= 1
        assert body["keys"][0]["kty"] == "RSA"
        assert body["keys"][0]["kid"] == "test-kid-1"

    def test_jwks_cache_control(self, client):
        """JWKS response includes appropriate Cache-Control header."""
        resp = client.get("/oauth/jwks")
        assert resp.status_code == 200
        cache_control = resp.headers.get("cache-control", "")
        assert "max-age=300" in cache_control
        assert "no-transform" in cache_control

    def test_jwks_feature_disabled(self, client_disabled):
        """When oauth_as_enabled=False, /oauth/jwks returns 404."""
        resp = client_disabled.get("/oauth/jwks")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /.well-known/oauth-authorization-server
# ---------------------------------------------------------------------------


class TestOAuthASMetadata:
    """Tests for the GET /.well-known/oauth-authorization-server endpoint."""

    def test_metadata_returns_required_fields(self, client):
        """AS metadata contains the required RFC 8414 fields."""
        resp = client.get("/.well-known/oauth-authorization-server")
        assert resp.status_code == 200
        body = resp.json()
        assert "issuer" in body
        assert "token_endpoint" in body
        assert "jwks_uri" in body
        assert "grant_types_supported" in body
        assert "client_credentials" in body["grant_types_supported"]

    def test_metadata_token_endpoint_auth_methods(self, client):
        """AS metadata includes supported token endpoint auth methods."""
        resp = client.get("/.well-known/oauth-authorization-server")
        assert resp.status_code == 200
        body = resp.json()
        assert "token_endpoint_auth_methods_supported" in body
        methods = body["token_endpoint_auth_methods_supported"]
        assert "client_secret_basic" in methods
        assert "client_secret_post" in methods

    def test_metadata_scopes_supported(self, client):
        """AS metadata includes scopes_supported field."""
        resp = client.get("/.well-known/oauth-authorization-server")
        assert resp.status_code == 200
        body = resp.json()
        assert "scopes_supported" in body
        assert isinstance(body["scopes_supported"], list)

    def test_metadata_feature_disabled(self, client_disabled):
        """When oauth_as_enabled=False, metadata returns 404."""
        resp = client_disabled.get("/.well-known/oauth-authorization-server")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /admin/oauth-as/clients
# ---------------------------------------------------------------------------


class TestRegisterClient:
    """Tests for the POST /admin/oauth-as/clients endpoint."""

    def test_register_client_success(self, client, mock_service):
        """Registering a new client returns 201 with client_secret."""
        resp = client.post(
            "/admin/oauth-as/clients",
            json={
                "client_id": "new-client",
                "client_name": "New Client",
                "scopes": ["read"],
                "teams": [],
                "is_admin": False,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "client_secret" in body
        assert body["client_id"] == "new-client"
        assert body["client_name"] == "New Client"

    def test_register_client_duplicate(self, client, mock_service):
        """Registering a duplicate client_id returns 409."""
        mock_service.register_client = AsyncMock(return_value=None)
        resp = client.post(
            "/admin/oauth-as/clients",
            json={
                "client_id": "existing-client",
                "client_name": "Existing",
                "scopes": [],
                "teams": [],
                "is_admin": False,
            },
        )
        assert resp.status_code == 409
        body = resp.json()
        assert "already exists" in body["detail"]

    def test_register_client_non_admin(self, client_non_admin):
        """Non-admin user gets 403 when registering a client."""
        resp = client_non_admin.post(
            "/admin/oauth-as/clients",
            json={
                "client_id": "new-client",
                "client_name": "New Client",
                "scopes": [],
                "teams": [],
                "is_admin": False,
            },
        )
        assert resp.status_code == 403

    def test_register_client_with_admin_flag(self, client, mock_service):
        """Can register a client with is_admin=True."""
        mock_service.register_client = AsyncMock(
            return_value={
                "client_id": "admin-client",
                "client_name": "Admin Client",
                "client_secret": "admin-secret",
                "scopes": ["read", "write"],
                "teams": ["ops"],
                "is_admin": True,
                "is_active": True,
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
            }
        )
        resp = client.post(
            "/admin/oauth-as/clients",
            json={
                "client_id": "admin-client",
                "client_name": "Admin Client",
                "scopes": ["read", "write"],
                "teams": ["ops"],
                "is_admin": True,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["is_admin"] is True
        assert body["client_secret"] == "admin-secret"

    def test_register_client_missing_required_fields(self, client):
        """Missing required fields returns 422 validation error."""
        resp = client.post(
            "/admin/oauth-as/clients",
            json={"client_name": "Missing ID"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /admin/oauth-as/clients
# ---------------------------------------------------------------------------


class TestListClients:
    """Tests for the GET /admin/oauth-as/clients endpoint."""

    def test_list_clients_success(self, client, mock_service):
        """Admin can list all clients."""
        resp = client.get("/admin/oauth-as/clients")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        assert body[0]["client_id"] == "client-1"

    def test_list_clients_non_admin(self, client_non_admin):
        """Non-admin user gets 403 when listing clients."""
        resp = client_non_admin.get("/admin/oauth-as/clients")
        assert resp.status_code == 403

    def test_list_clients_empty(self, client, mock_service):
        """List clients returns empty array when none registered."""
        mock_service.list_clients = AsyncMock(return_value=[])
        resp = client.get("/admin/oauth-as/clients")
        assert resp.status_code == 200
        body = resp.json()
        assert body == []


# ---------------------------------------------------------------------------
# DELETE /admin/oauth-as/clients/{client_id}
# ---------------------------------------------------------------------------


class TestDeactivateClient:
    """Tests for the DELETE /admin/oauth-as/clients/{client_id} endpoint."""

    def test_deactivate_client_success(self, client, mock_service):
        """Deactivating an existing client returns 200."""
        resp = client.delete("/admin/oauth-as/clients/client-1")
        assert resp.status_code == 200
        body = resp.json()
        assert "deactivated" in body["detail"]
        assert "client-1" in body["detail"]

    def test_deactivate_client_not_found(self, client, mock_service):
        """Deactivating a non-existent client returns 404."""
        mock_service.deactivate_client = AsyncMock(return_value=False)
        resp = client.delete("/admin/oauth-as/clients/nonexistent")
        assert resp.status_code == 404
        body = resp.json()
        assert "not found" in body["detail"]

    def test_deactivate_client_non_admin(self, client_non_admin):
        """Non-admin user gets 403 when deactivating a client."""
        resp = client_non_admin.delete("/admin/oauth-as/clients/client-1")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /admin/oauth-as/clients/{client_id}/rotate-secret
# ---------------------------------------------------------------------------


class TestRotateClientSecret:
    """Tests for the POST /admin/oauth-as/clients/{client_id}/rotate-secret endpoint."""

    def test_rotate_secret_success(self, client, mock_service):
        """Rotating secret returns 200 with new secret."""
        resp = client.post("/admin/oauth-as/clients/client-1/rotate-secret")
        assert resp.status_code == 200
        body = resp.json()
        assert "client_secret" in body
        assert body["client_id"] == "client-1"
        assert body["client_secret"] == "new-rotated-secret-xyz"

    def test_rotate_secret_not_found(self, client, mock_service):
        """Rotating secret for non-existent client returns 404."""
        mock_service.rotate_client_secret = AsyncMock(return_value=None)
        resp = client.post("/admin/oauth-as/clients/nonexistent/rotate-secret")
        assert resp.status_code == 404
        body = resp.json()
        assert "not found" in body["detail"]

    def test_rotate_secret_non_admin(self, client_non_admin):
        """Non-admin user gets 403 when rotating secret."""
        resp = client_non_admin.post("/admin/oauth-as/clients/client-1/rotate-secret")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /admin/oauth-as/tokens/revoke
# ---------------------------------------------------------------------------


class TestRevokeToken:
    """Tests for the POST /admin/oauth-as/tokens/revoke endpoint."""

    def test_revoke_token_success(self, client, mock_service):
        """Revoking a token returns 200 with revoked=True."""
        resp = client.post(
            "/admin/oauth-as/tokens/revoke",
            json={"jti": "token-jti-12345"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["revoked"] is True
        assert body["jti"] == "token-jti-12345"

    def test_revoke_token_not_found(self, client, mock_service):
        """Revoking a non-existent token returns 200 with revoked=False."""
        mock_service.revoke_token = AsyncMock(return_value=False)
        resp = client.post(
            "/admin/oauth-as/tokens/revoke",
            json={"jti": "nonexistent-jti"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["revoked"] is False
        assert body["jti"] == "nonexistent-jti"

    def test_revoke_token_non_admin(self, client_non_admin):
        """Non-admin user gets 403 when revoking a token."""
        resp = client_non_admin.post(
            "/admin/oauth-as/tokens/revoke",
            json={"jti": "any-jti"},
        )
        assert resp.status_code == 403

    def test_revoke_token_missing_jti(self, client):
        """Missing jti field returns 422 validation error."""
        resp = client.post(
            "/admin/oauth-as/tokens/revoke",
            json={},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Feature-disabled deny tests (consolidated)
# ---------------------------------------------------------------------------


class TestFeatureDisabledDenyPaths:
    """Verify all OAuth AS endpoints return 404 when feature is disabled."""

    def test_token_disabled(self, client_disabled):
        """POST /oauth/token returns 404 when disabled."""
        resp = client_disabled.post(
            "/oauth/token",
            data={"grant_type": "client_credentials", "client_id": "x", "client_secret": "y"},
        )
        assert resp.status_code == 404

    def test_jwks_disabled(self, client_disabled):
        """GET /oauth/jwks returns 404 when disabled."""
        resp = client_disabled.get("/oauth/jwks")
        assert resp.status_code == 404

    def test_metadata_disabled(self, client_disabled):
        """GET /.well-known/oauth-authorization-server returns 404 when disabled."""
        resp = client_disabled.get("/.well-known/oauth-authorization-server")
        assert resp.status_code == 404

    def test_admin_register_client_disabled(self, client_disabled):
        """POST /admin/oauth-as/clients returns 404 when disabled."""
        resp = client_disabled.post(
            "/admin/oauth-as/clients",
            json={"client_id": "x", "client_name": "X", "scopes": [], "teams": [], "is_admin": False},
        )
        assert resp.status_code == 404

    def test_admin_list_clients_disabled(self, client_disabled):
        """GET /admin/oauth-as/clients returns 404 when disabled."""
        resp = client_disabled.get("/admin/oauth-as/clients")
        assert resp.status_code == 404

    def test_admin_deactivate_client_disabled(self, client_disabled):
        """DELETE /admin/oauth-as/clients/{id} returns 404 when disabled."""
        resp = client_disabled.delete("/admin/oauth-as/clients/any-id")
        assert resp.status_code == 404

    def test_admin_rotate_secret_disabled(self, client_disabled):
        """POST /admin/oauth-as/clients/{id}/rotate-secret returns 404 when disabled."""
        resp = client_disabled.post("/admin/oauth-as/clients/any-id/rotate-secret")
        assert resp.status_code == 404

    def test_admin_revoke_token_disabled(self, client_disabled):
        """POST /admin/oauth-as/tokens/revoke returns 404 when disabled."""
        resp = client_disabled.post(
            "/admin/oauth-as/tokens/revoke",
            json={"jti": "any"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DCR — Dynamic Client Registration (RFC 7591)
# ---------------------------------------------------------------------------


class TestDCRRegistration:
    """Tests for POST /oauth/register (RFC 7591 Dynamic Client Registration)."""

    # ------------------------------------------------------------------
    # Open mode fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def client_dcr_open(self, mock_service):
        """TestClient with DCR mode=open (no auth required)."""
        from mcpgateway.middleware.rbac import get_current_user_with_permissions
        from mcpgateway.routers.oauth_as import _get_db

        test_app = FastAPI()
        test_app.include_router(oauth_as_router)
        test_app.dependency_overrides[get_current_user_with_permissions] = lambda: _admin_user_ctx(is_admin=False)
        test_app.dependency_overrides[_get_db] = _mock_db_session

        with (
            patch("mcpgateway.config.settings.oauth_as_enabled", True),
            patch("mcpgateway.config.settings.oauth_dcr_mode", "open"),
            patch("mcpgateway.services.oauth_as_service.get_oauth_as_service", return_value=mock_service),
        ):
            yield TestClient(test_app)

    @pytest.fixture
    def client_dcr_auth(self, mock_service):
        """TestClient with DCR mode=authenticated (Bearer required)."""
        from mcpgateway.middleware.rbac import get_current_user_with_permissions
        from mcpgateway.routers.oauth_as import _get_db

        test_app = FastAPI()
        test_app.include_router(oauth_as_router)
        test_app.dependency_overrides[get_current_user_with_permissions] = lambda: _admin_user_ctx(is_admin=False)
        test_app.dependency_overrides[_get_db] = _mock_db_session

        with (
            patch("mcpgateway.config.settings.oauth_as_enabled", True),
            patch("mcpgateway.config.settings.oauth_dcr_mode", "authenticated"),
            patch("mcpgateway.services.oauth_as_service.get_oauth_as_service", return_value=mock_service),
        ):
            yield TestClient(test_app)

    @pytest.fixture
    def client_dcr_disabled(self, mock_service):
        """TestClient with DCR mode=disabled."""
        from mcpgateway.middleware.rbac import get_current_user_with_permissions
        from mcpgateway.routers.oauth_as import _get_db

        test_app = FastAPI()
        test_app.include_router(oauth_as_router)
        test_app.dependency_overrides[get_current_user_with_permissions] = lambda: _admin_user_ctx(is_admin=False)
        test_app.dependency_overrides[_get_db] = _mock_db_session

        with (
            patch("mcpgateway.config.settings.oauth_as_enabled", True),
            patch("mcpgateway.config.settings.oauth_dcr_mode", "disabled"),
            patch("mcpgateway.services.oauth_as_service.get_oauth_as_service", return_value=mock_service),
        ):
            yield TestClient(test_app, raise_server_exceptions=False)

    # ------------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------------

    def test_dcr_open_mode_no_auth_required(self, client_dcr_open):
        """Open mode: POST /oauth/register succeeds without Bearer token."""
        resp = client_dcr_open.post(
            "/oauth/register",
            json={"client_name": "My MCP Client", "grant_types": ["client_credentials"]},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "client_id" in data
        assert "client_secret" in data
        assert data["client_id_issued_at"] > 0
        assert data["client_secret_expires_at"] == 0  # non-expiring per RFC 7591

    def test_dcr_authenticated_mode_with_valid_token(self, client_dcr_auth, mock_service):
        """Authenticated mode: valid Bearer token allows registration."""
        with patch(
            "mcpgateway.utils.verify_credentials.verify_jwt_token",
            new_callable=AsyncMock,
            return_value={"sub": "some-user"},
        ):
            resp = client_dcr_auth.post(
                "/oauth/register",
                json={"client_name": "My MCP Client", "grant_types": ["client_credentials"]},
                headers={"Authorization": "Bearer valid-token-here"},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["grant_types"] == ["client_credentials"]
        assert "client_secret" in data

    def test_dcr_authenticated_mode_rejects_no_token(self, client_dcr_auth):
        """Authenticated mode: missing Bearer token returns 401."""
        resp = client_dcr_auth.post(
            "/oauth/register",
            json={"client_name": "My MCP Client"},
        )
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"] == "invalid_token"

    def test_dcr_authenticated_mode_rejects_bad_token(self, client_dcr_auth):
        """Authenticated mode: invalid Bearer token returns 401."""
        with patch(
            "mcpgateway.utils.verify_credentials.verify_jwt_token",
            new_callable=AsyncMock,
            side_effect=Exception("invalid signature"),
        ):
            resp = client_dcr_auth.post(
                "/oauth/register",
                json={"client_name": "My MCP Client"},
                headers={"Authorization": "Bearer bad-token"},
            )
        assert resp.status_code == 401
        assert resp.json()["error"] == "invalid_token"

    def test_dcr_disabled_mode_returns_404(self, client_dcr_disabled):
        """DCR mode=disabled: POST /oauth/register returns 404."""
        resp = client_dcr_disabled.post(
            "/oauth/register",
            json={"client_name": "My MCP Client"},
        )
        assert resp.status_code == 404

    def test_dcr_scope_restriction_enforced(self, client_dcr_open, mock_service):
        """DCR clients cannot request admin scope — service enforces restriction."""
        # Service mock returns only mcp:access regardless of request
        resp = client_dcr_open.post(
            "/oauth/register",
            json={
                "client_name": "Scope Pusher",
                "grant_types": ["client_credentials"],
                "scope": "mcp:access admin servers.manage",
            },
        )
        assert resp.status_code == 201
        # register_dcr_client was called (scope enforcement happens in service layer)
        mock_service.register_dcr_client.assert_called_once()
        call_kwargs = mock_service.register_dcr_client.call_args.kwargs
        assert call_kwargs["requested_scope"] == "mcp:access admin servers.manage"
        # Response scope is whatever the service returned (mcp:access only in mock)
        assert resp.json()["scope"] == "mcp:access"

    def test_dcr_registration_endpoint_in_metadata_when_enabled(self, client):
        """AS metadata includes registration_endpoint when DCR is not disabled."""
        with patch("mcpgateway.config.settings.oauth_dcr_mode", "authenticated"):
            # Patch service metadata to include registration_endpoint
            import unittest.mock as um

            client.app.dependency_overrides  # trigger fixture setup
            # Re-patch service to return metadata with registration_endpoint
            from mcpgateway.services.oauth_as_service import get_oauth_as_service as _svc_factory

            with patch(
                "mcpgateway.services.oauth_as_service.get_oauth_as_service",
                return_value=MagicMock(
                    get_as_metadata=MagicMock(
                        return_value={
                            "issuer": "http://testserver",
                            "token_endpoint": "http://testserver/oauth/token",
                            "jwks_uri": "http://testserver/oauth/jwks",
                            "grant_types_supported": ["client_credentials"],
                            "registration_endpoint": "http://testserver/oauth/register",
                        }
                    )
                ),
            ):
                resp = client.get("/.well-known/oauth-authorization-server")
        assert resp.status_code == 200
        assert "registration_endpoint" in resp.json()

    def test_dcr_invalid_client_name_returns_400(self, client_dcr_open, mock_service):
        """Empty client_name returns 400 invalid_client_metadata."""
        mock_service.register_dcr_client = AsyncMock(side_effect=ValueError("client_name is required"))
        resp = client_dcr_open.post(
            "/oauth/register",
            json={"client_name": "   "},
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_client_metadata"

    def test_dcr_none_auth_method_accepted(self, client_dcr_open, mock_service):
        """token_endpoint_auth_method=none is accepted (MCP public client flow).

        RFC 7591 §2: 'none' is valid for public clients. The service normalises
        it to client_secret_basic internally so a client_secret is still issued.
        """
        resp = client_dcr_open.post(
            "/oauth/register",
            json={
                "client_name": "claude-code",
                "grant_types": ["client_credentials"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["client_id"].startswith("dcr-")
        assert "client_secret" in data
