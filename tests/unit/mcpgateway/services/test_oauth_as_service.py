# -*- coding: utf-8 -*-
"""Location: ./tests/unit/mcpgateway/services/test_oauth_as_service.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0

Comprehensive tests for OAuth Authorization Server service.
"""

# Standard
import asyncio
import hashlib
import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

# Third-Party
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# First-Party
from mcpgateway.db import OAuthASClient, OAuthASRevokedToken
from mcpgateway.services.oauth_as_service import (
    InvalidClientError,
    InvalidScopeError,
    OAuthASError,
    OAuthASService,
    RateLimitExceededError,
    _RateLimitBucket,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rsa_key_pair():
    """Generate an RSA key pair and write to temporary PEM files."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as priv_f:
        priv_f.write(private_pem)
        priv_path = priv_f.name

    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as pub_f:
        pub_f.write(public_pem)
        pub_path = pub_f.name

    yield {
        "private_key": private_key,
        "public_key": public_key,
        "private_key_path": priv_path,
        "public_key_path": pub_path,
    }

    os.unlink(priv_path)
    os.unlink(pub_path)


@pytest.fixture
def mock_settings(rsa_key_pair):
    """Return a mock settings object configured for an enabled OAuth AS."""
    settings_mock = MagicMock()
    settings_mock.oauth_as_enabled = True
    settings_mock.oauth_rs256_private_key_path = rsa_key_pair["private_key_path"]
    settings_mock.oauth_rs256_public_key_path = rsa_key_pair["public_key_path"]
    settings_mock.oauth_rs256_kid = ""
    settings_mock.oauth_token_ttl = 900
    settings_mock.oauth_token_max_ttl = 3600
    settings_mock.oauth_issuer = "https://test-issuer.example.com"
    settings_mock.jwt_audience = "test-audience"
    settings_mock.oauth_rate_limit_per_client = 10
    settings_mock.oauth_rate_limit_per_ip = 20
    settings_mock.oauth_rate_limit_global = 100
    settings_mock.external_url = "https://gateway.example.com"
    return settings_mock


@pytest.fixture
def oauth_service(mock_settings):
    """Create an OAuthASService with mocked settings and Argon2 service."""
    with patch("mcpgateway.services.oauth_as_service.settings", mock_settings), \
         patch("mcpgateway.services.oauth_as_service.Argon2PasswordService") as MockArgon2:
        mock_argon2 = MockArgon2.return_value
        mock_argon2.hash_password.side_effect = lambda pw: f"$argon2id$hashed${pw}"
        mock_argon2.verify_password.side_effect = lambda pw, h: h == f"$argon2id$hashed${pw}"
        service = OAuthASService()
        service._mock_argon2 = mock_argon2  # expose for test assertions
        yield service


@pytest.fixture(autouse=True)
def _clean_oauth_tables(test_db):
    """Remove all OAuth AS rows between tests to prevent UNIQUE violations."""
    yield
    test_db.query(OAuthASRevokedToken).delete()
    test_db.query(OAuthASClient).delete()
    test_db.commit()


@pytest.fixture
def sample_client(test_db):
    """Insert a sample OAuthASClient into the test database."""
    client = OAuthASClient(
        client_id="test-client-001",
        client_name="Test Client",
        client_secret_hash="$argon2id$hashed$my-secret",
        scopes=["tools.read", "tools.execute", "resources.read"],
        teams=["team-alpha"],
        is_admin=False,
        is_active=True,
    )
    test_db.add(client)
    test_db.commit()
    test_db.refresh(client)
    return client


@pytest.fixture
def admin_client(test_db):
    """Insert an admin OAuthASClient into the test database."""
    client = OAuthASClient(
        client_id="admin-client-001",
        client_name="Admin Client",
        client_secret_hash="$argon2id$hashed$admin-secret",
        scopes=["tools.read", "tools.execute", "resources.read", "admin"],
        teams=["platform-ops"],
        is_admin=True,
        is_active=True,
    )
    test_db.add(client)
    test_db.commit()
    test_db.refresh(client)
    return client


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


class TestExceptions:
    """Test exception hierarchy."""

    def test_oauth_as_error_is_exception(self):
        """Test OAuthASError inherits from Exception."""
        assert issubclass(OAuthASError, Exception)

    def test_invalid_client_error_inherits_oauth_as_error(self):
        """Test InvalidClientError inherits from OAuthASError."""
        assert issubclass(InvalidClientError, OAuthASError)

    def test_invalid_scope_error_inherits_oauth_as_error(self):
        """Test InvalidScopeError inherits from OAuthASError."""
        assert issubclass(InvalidScopeError, OAuthASError)

    def test_rate_limit_exceeded_error_inherits_oauth_as_error(self):
        """Test RateLimitExceededError inherits from OAuthASError."""
        assert issubclass(RateLimitExceededError, OAuthASError)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestOAuthASServiceInit:
    """Test OAuthASService initialization."""

    def test_disabled_mode(self):
        """Test initialization with oauth_as_enabled=False sets keys to None."""
        with patch("mcpgateway.services.oauth_as_service.settings") as mock_s:
            mock_s.oauth_as_enabled = False
            service = OAuthASService()

            assert service._private_key is None
            assert service._public_key is None
            assert service.kid == ""

    def test_enabled_missing_key_paths_raises_runtime_error(self):
        """Test RuntimeError when AS is enabled but key paths are not configured."""
        with patch("mcpgateway.services.oauth_as_service.settings") as mock_s:
            mock_s.oauth_as_enabled = True
            mock_s.oauth_rs256_private_key_path = ""
            mock_s.oauth_rs256_public_key_path = ""

            with pytest.raises(RuntimeError, match="RSA key paths are not configured"):
                OAuthASService()

    def test_enabled_missing_private_key_file_raises_runtime_error(self):
        """Test RuntimeError when private key file does not exist."""
        with patch("mcpgateway.services.oauth_as_service.settings") as mock_s:
            mock_s.oauth_as_enabled = True
            mock_s.oauth_rs256_private_key_path = "/nonexistent/private.pem"
            mock_s.oauth_rs256_public_key_path = "/nonexistent/public.pem"

            with pytest.raises(RuntimeError, match="Failed to load RS256 private key"):
                OAuthASService()

    def test_enabled_with_valid_rsa_keys(self, rsa_key_pair):
        """Test successful initialization with valid RSA keys loads correctly."""
        with patch("mcpgateway.services.oauth_as_service.settings") as mock_s, \
             patch("mcpgateway.services.oauth_as_service.Argon2PasswordService"):
            mock_s.oauth_as_enabled = True
            mock_s.oauth_rs256_private_key_path = rsa_key_pair["private_key_path"]
            mock_s.oauth_rs256_public_key_path = rsa_key_pair["public_key_path"]
            mock_s.oauth_rs256_kid = ""

            service = OAuthASService()

            assert service._private_key is not None
            assert service._public_key is not None
            assert len(service.kid) == 16  # SHA-256 hex[:16]

    def test_enabled_with_custom_kid(self, rsa_key_pair):
        """Test that custom kid from settings overrides auto-generated kid."""
        with patch("mcpgateway.services.oauth_as_service.settings") as mock_s, \
             patch("mcpgateway.services.oauth_as_service.Argon2PasswordService"):
            mock_s.oauth_as_enabled = True
            mock_s.oauth_rs256_private_key_path = rsa_key_pair["private_key_path"]
            mock_s.oauth_rs256_public_key_path = rsa_key_pair["public_key_path"]
            mock_s.oauth_rs256_kid = "my-custom-kid-123"

            service = OAuthASService()

            assert service.kid == "my-custom-kid-123"

    def test_kid_derived_from_public_key_fingerprint(self, rsa_key_pair):
        """Test auto-generated kid matches SHA-256 fingerprint of public key."""
        with patch("mcpgateway.services.oauth_as_service.settings") as mock_s, \
             patch("mcpgateway.services.oauth_as_service.Argon2PasswordService"):
            mock_s.oauth_as_enabled = True
            mock_s.oauth_rs256_private_key_path = rsa_key_pair["private_key_path"]
            mock_s.oauth_rs256_public_key_path = rsa_key_pair["public_key_path"]
            mock_s.oauth_rs256_kid = ""

            service = OAuthASService()

            # Compute expected kid
            pub_der = rsa_key_pair["public_key"].public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            expected_kid = hashlib.sha256(pub_der).hexdigest()[:16]
            assert service.kid == expected_kid


# ---------------------------------------------------------------------------
# authenticate_client
# ---------------------------------------------------------------------------


class TestAuthenticateClient:
    """Test client authentication."""

    def test_successful_auth_primary_secret(self, oauth_service, sample_client, test_db):
        """Test successful authentication with primary secret."""
        result = asyncio.run(oauth_service.authenticate_client("test-client-001", "my-secret", test_db))
        assert result.client_id == "test-client-001"

    def test_successful_auth_previous_secret_during_grace(self, oauth_service, test_db):
        """Test successful authentication with previous secret during grace period."""
        client = OAuthASClient(
            client_id="grace-client",
            client_name="Grace Client",
            client_secret_hash="$argon2id$hashed$new-secret",
            previous_secret_hash="$argon2id$hashed$old-secret",
            previous_secret_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            scopes=["tools.read"],
            teams=["team-a"],
            is_admin=False,
            is_active=True,
        )
        test_db.add(client)
        test_db.commit()

        result = asyncio.run(oauth_service.authenticate_client("grace-client", "old-secret", test_db))
        assert result.client_id == "grace-client"

    def test_unknown_client_id_raises_invalid_client(self, oauth_service, test_db):
        """Test that unknown client_id raises InvalidClientError."""
        with pytest.raises(InvalidClientError, match="invalid_client"):
            asyncio.run(oauth_service.authenticate_client("nonexistent-client", "any-secret", test_db))

    def test_inactive_client_raises_invalid_client(self, oauth_service, test_db):
        """Test that inactive client raises InvalidClientError."""
        client = OAuthASClient(
            client_id="inactive-client",
            client_name="Inactive Client",
            client_secret_hash="$argon2id$hashed$secret",
            scopes=["tools.read"],
            teams=[],
            is_admin=False,
            is_active=False,
        )
        test_db.add(client)
        test_db.commit()

        with pytest.raises(InvalidClientError, match="invalid_client"):
            asyncio.run(oauth_service.authenticate_client("inactive-client", "secret", test_db))

    def test_wrong_secret_raises_invalid_client(self, oauth_service, sample_client, test_db):
        """Test that wrong secret raises InvalidClientError."""
        with pytest.raises(InvalidClientError, match="invalid_client"):
            asyncio.run(oauth_service.authenticate_client("test-client-001", "wrong-secret", test_db))

    def test_grace_period_expired_raises_invalid_client(self, oauth_service, test_db):
        """Test that expired grace period previous secret raises InvalidClientError."""
        client = OAuthASClient(
            client_id="expired-grace-client",
            client_name="Expired Grace Client",
            client_secret_hash="$argon2id$hashed$new-secret",
            previous_secret_hash="$argon2id$hashed$old-secret",
            previous_secret_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            scopes=["tools.read"],
            teams=[],
            is_admin=False,
            is_active=True,
        )
        test_db.add(client)
        test_db.commit()

        with pytest.raises(InvalidClientError, match="invalid_client"):
            asyncio.run(oauth_service.authenticate_client("expired-grace-client", "old-secret", test_db))


# ---------------------------------------------------------------------------
# issue_token
# ---------------------------------------------------------------------------


class TestIssueToken:
    """Test JWT token issuance."""

    def test_token_with_default_scopes(self, oauth_service, sample_client, mock_settings):
        """Test token issuance uses client's full granted scopes by default."""
        result = oauth_service.issue_token(sample_client)

        assert result["token_type"] == "bearer"
        assert result["expires_in"] == 900
        assert result["scope"] == "tools.read tools.execute resources.read"
        assert isinstance(result["access_token"], str)

    def test_token_with_requested_subset_scopes(self, oauth_service, sample_client, mock_settings):
        """Test token issuance with a subset of granted scopes."""
        result = oauth_service.issue_token(sample_client, requested_scope="tools.read resources.read")

        assert result["scope"] == "tools.read resources.read"

    def test_invalid_scope_raises_error(self, oauth_service, sample_client):
        """Test that requesting a non-granted scope raises InvalidScopeError."""
        with pytest.raises(InvalidScopeError, match="Scope\\(s\\) not granted"):
            oauth_service.issue_token(sample_client, requested_scope="admin")

    def test_token_contains_correct_claims(self, oauth_service, sample_client, rsa_key_pair, mock_settings):
        """Test token JWT payload contains all required claims."""
        result = oauth_service.issue_token(sample_client)
        decoded = jwt.decode(
            result["access_token"],
            rsa_key_pair["public_key"],
            algorithms=["RS256"],
            audience="test-audience",
        )

        assert decoded["iss"] == "https://test-issuer.example.com"
        assert decoded["sub"] == "test-client-001"
        assert decoded["aud"] == "test-audience"
        assert decoded["client_id"] == "test-client-001"
        assert decoded["teams"] == ["team-alpha"]
        assert decoded["is_admin"] is False
        assert decoded["auth_provider"] == "oauth_as"
        assert decoded["token_use"] == "m2m"
        assert "jti" in decoded
        assert "iat" in decoded
        assert "exp" in decoded
        assert "nbf" in decoded
        assert decoded["scope"] == "tools.read tools.execute resources.read"

    def test_token_header_contains_typ_and_kid(self, oauth_service, sample_client, rsa_key_pair):
        """Test token JWT header has correct typ and kid."""
        result = oauth_service.issue_token(sample_client)
        header = jwt.get_unverified_header(result["access_token"])

        assert header["alg"] == "RS256"
        assert header["typ"] == "at+jwt"
        assert header["kid"] == oauth_service.kid

    def test_token_is_valid_rs256_jwt(self, oauth_service, sample_client, rsa_key_pair, mock_settings):
        """Test token can be decoded and verified with the public key."""
        result = oauth_service.issue_token(sample_client)
        decoded = jwt.decode(
            result["access_token"],
            rsa_key_pair["public_key"],
            algorithms=["RS256"],
            audience="test-audience",
        )
        assert decoded["sub"] == "test-client-001"

    def test_token_for_admin_client(self, oauth_service, admin_client, rsa_key_pair, mock_settings):
        """Test token for admin client has is_admin=True."""
        result = oauth_service.issue_token(admin_client)
        decoded = jwt.decode(
            result["access_token"],
            rsa_key_pair["public_key"],
            algorithms=["RS256"],
            audience="test-audience",
        )
        assert decoded["is_admin"] is True
        assert decoded["teams"] == ["platform-ops"]

    def test_token_issuer_fallback_to_external_url(self, rsa_key_pair):
        """Test issuer falls back to external_url when oauth_issuer is empty."""
        with patch("mcpgateway.services.oauth_as_service.settings") as mock_s, \
             patch("mcpgateway.services.oauth_as_service.Argon2PasswordService"):
            mock_s.oauth_as_enabled = True
            mock_s.oauth_rs256_private_key_path = rsa_key_pair["private_key_path"]
            mock_s.oauth_rs256_public_key_path = rsa_key_pair["public_key_path"]
            mock_s.oauth_rs256_kid = "test-kid"
            mock_s.oauth_token_ttl = 900
            mock_s.oauth_issuer = ""
            mock_s.external_url = "https://my-gateway.example.com"
            mock_s.jwt_audience = "test-aud"

            service = OAuthASService()

            client = MagicMock()
            client.client_id = "c1"
            client.scopes = ["tools.read"]
            client.teams = []
            client.is_admin = False

            result = service.issue_token(client)
            decoded = jwt.decode(
                result["access_token"],
                rsa_key_pair["public_key"],
                algorithms=["RS256"],
                audience="test-aud",
            )
            assert decoded["iss"] == "https://my-gateway.example.com"


# ---------------------------------------------------------------------------
# check_rate_limit
# ---------------------------------------------------------------------------


class TestCheckRateLimit:
    """Test rate limiting."""

    def test_under_limit_returns_true(self, oauth_service):
        """Test request under all limits returns True."""
        assert oauth_service.check_rate_limit("client-1", "10.0.0.1") is True

    def test_per_client_limit_exceeded_returns_false(self, oauth_service, mock_settings):
        """Test per-client rate limit exceeded returns False."""
        mock_settings.oauth_rate_limit_per_client = 3

        for _ in range(3):
            oauth_service.check_rate_limit("flood-client", "10.0.0.1")

        assert oauth_service.check_rate_limit("flood-client", "10.0.0.2") is False

    def test_per_ip_limit_exceeded_returns_false(self, oauth_service, mock_settings):
        """Test per-IP rate limit exceeded returns False."""
        mock_settings.oauth_rate_limit_per_ip = 3
        # Use unique client IDs so client bucket stays under limit
        for i in range(3):
            oauth_service.check_rate_limit(f"client-{i}", "10.0.0.99")

        assert oauth_service.check_rate_limit("client-new", "10.0.0.99") is False

    def test_global_limit_exceeded_returns_false(self, oauth_service, mock_settings):
        """Test global rate limit exceeded returns False."""
        mock_settings.oauth_rate_limit_global = 5
        mock_settings.oauth_rate_limit_per_client = 1000
        mock_settings.oauth_rate_limit_per_ip = 1000

        for i in range(5):
            oauth_service.check_rate_limit(f"g-client-{i}", f"10.0.{i}.1")

        assert oauth_service.check_rate_limit("g-client-new", "10.0.255.1") is False

    def test_different_clients_independent_limits(self, oauth_service, mock_settings):
        """Test that different clients have independent rate limit buckets."""
        mock_settings.oauth_rate_limit_per_client = 2

        oauth_service.check_rate_limit("client-a", "10.0.0.1")
        oauth_service.check_rate_limit("client-a", "10.0.0.1")
        # client-a is at limit

        # client-b should still be allowed
        assert oauth_service.check_rate_limit("client-b", "10.0.0.2") is True


# ---------------------------------------------------------------------------
# revoke_token / is_token_revoked
# ---------------------------------------------------------------------------


class TestTokenRevocation:
    """Test token revocation via JTI deny-list."""

    def test_revoke_token_success(self, oauth_service, test_db):
        """Test revoking a JTI returns True and persists."""
        result = asyncio.run(oauth_service.revoke_token("jti-abc-123", test_db))
        assert result is True

        # Verify it was persisted
        record = test_db.query(OAuthASRevokedToken).filter(OAuthASRevokedToken.jti == "jti-abc-123").first()
        assert record is not None
        assert record.client_id == "admin-revocation"

    def test_is_token_revoked_returns_true_for_revoked(self, oauth_service, test_db):
        """Test is_token_revoked returns True for a revoked JTI."""
        asyncio.run(oauth_service.revoke_token("jti-revoked", test_db))
        result = asyncio.run(oauth_service.is_token_revoked("jti-revoked", test_db))
        assert result is True

    def test_is_token_revoked_returns_false_for_non_revoked(self, oauth_service, test_db):
        """Test is_token_revoked returns False for a non-revoked JTI."""
        result = asyncio.run(oauth_service.is_token_revoked("jti-never-revoked", test_db))
        assert result is False

    def test_revoke_already_revoked_is_idempotent(self, oauth_service, test_db):
        """Test revoking an already-revoked JTI returns True without error."""
        asyncio.run(oauth_service.revoke_token("jti-double", test_db))
        result = asyncio.run(oauth_service.revoke_token("jti-double", test_db))
        assert result is True

    def test_expired_revocation_not_considered_revoked(self, oauth_service, test_db):
        """Test that an expired revocation entry is not treated as revoked."""
        expired_entry = OAuthASRevokedToken(
            jti="jti-expired",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
            client_id="test",
        )
        test_db.add(expired_entry)
        test_db.commit()

        result = asyncio.run(oauth_service.is_token_revoked("jti-expired", test_db))
        assert result is False


# ---------------------------------------------------------------------------
# get_jwks
# ---------------------------------------------------------------------------


class TestGetJWKS:
    """Test JWKS endpoint response."""

    def test_jwks_structure(self, oauth_service):
        """Test JWKS returns valid structure with keys array."""
        jwks = oauth_service.get_jwks()

        assert "keys" in jwks
        assert isinstance(jwks["keys"], list)
        assert len(jwks["keys"]) == 1

    def test_jwks_key_fields(self, oauth_service):
        """Test JWKS key has correct kty, use, alg, kid, n, e."""
        jwks = oauth_service.get_jwks()
        key = jwks["keys"][0]

        assert key["kty"] == "RSA"
        assert key["use"] == "sig"
        assert key["alg"] == "RS256"
        assert key["kid"] == oauth_service.kid
        assert "n" in key
        assert "e" in key
        # Base64url encoded values should be non-empty strings
        assert len(key["n"]) > 0
        assert len(key["e"]) > 0

    def test_jwks_n_and_e_are_base64url(self, oauth_service):
        """Test JWKS n and e are valid base64url-encoded strings (no padding)."""
        jwks = oauth_service.get_jwks()
        key = jwks["keys"][0]

        # Base64url should not contain +, /, or = padding
        for field in ("n", "e"):
            assert "+" not in key[field]
            assert "/" not in key[field]
            assert "=" not in key[field]


# ---------------------------------------------------------------------------
# get_as_metadata
# ---------------------------------------------------------------------------


class TestGetASMetadata:
    """Test RFC 8414 Authorization Server Metadata."""

    def test_metadata_structure(self, oauth_service, mock_settings):
        """Test metadata returns correct structure."""
        meta = oauth_service.get_as_metadata("https://gateway.example.com")

        assert meta["token_endpoint"] == "https://gateway.example.com/oauth/token"
        assert meta["jwks_uri"] == "https://gateway.example.com/oauth/jwks"
        assert "client_credentials" in meta["grant_types_supported"]
        assert meta["service_documentation"] == "https://gateway.example.com/docs"

    def test_metadata_response_types_is_none(self, oauth_service, mock_settings):
        """Test response_types_supported is ["none"]."""
        meta = oauth_service.get_as_metadata("https://gateway.example.com")
        assert meta["response_types_supported"] == ["none"]

    def test_metadata_auth_methods(self, oauth_service, mock_settings):
        """Test supported auth methods include client_secret_basic and client_secret_post."""
        meta = oauth_service.get_as_metadata("https://gateway.example.com")
        assert "client_secret_basic" in meta["token_endpoint_auth_methods_supported"]
        assert "client_secret_post" in meta["token_endpoint_auth_methods_supported"]

    def test_metadata_issuer_from_settings(self, oauth_service, mock_settings):
        """Test issuer comes from oauth_issuer setting when set."""
        meta = oauth_service.get_as_metadata("https://gateway.example.com")
        assert meta["issuer"] == "https://test-issuer.example.com"

    def test_metadata_issuer_fallback_to_base_url(self, rsa_key_pair):
        """Test issuer falls back to base_url when oauth_issuer is empty."""
        with patch("mcpgateway.services.oauth_as_service.settings") as mock_s, \
             patch("mcpgateway.services.oauth_as_service.Argon2PasswordService"):
            mock_s.oauth_as_enabled = True
            mock_s.oauth_rs256_private_key_path = rsa_key_pair["private_key_path"]
            mock_s.oauth_rs256_public_key_path = rsa_key_pair["public_key_path"]
            mock_s.oauth_rs256_kid = "kid"
            mock_s.oauth_issuer = ""

            service = OAuthASService()
            meta = service.get_as_metadata("https://fallback.example.com")
            assert meta["issuer"] == "https://fallback.example.com"

    def test_metadata_scopes_supported(self, oauth_service, mock_settings):
        """Test scopes_supported includes expected scopes."""
        meta = oauth_service.get_as_metadata("https://gateway.example.com")
        expected = ["tools.read", "tools.execute", "resources.read", "prompts.read", "servers.read", "servers.manage", "admin", "mcp:access"]
        assert meta["scopes_supported"] == expected

    def test_metadata_authorization_endpoint(self, oauth_service, mock_settings):
        """Test authorization_endpoint is present (required by MCP SDK OAuthMetadataSchema)."""
        meta = oauth_service.get_as_metadata("https://gateway.example.com")
        assert meta["authorization_endpoint"] == "https://gateway.example.com/oauth/authorize"


# ---------------------------------------------------------------------------
# register_client
# ---------------------------------------------------------------------------


class TestRegisterClient:
    """Test client registration."""

    def test_successful_registration(self, oauth_service, test_db):
        """Test successful client registration returns dict with client_secret."""
        result = asyncio.run(
            oauth_service.register_client(
                client_id="new-client",
                client_name="New Client",
                scopes=["tools.read"],
                teams=["team-x"],
                is_admin=False,
                db=test_db,
            )
        )

        assert result is not None
        assert result["client_id"] == "new-client"
        assert result["client_name"] == "New Client"
        assert "client_secret" in result
        assert len(result["client_secret"]) > 0
        assert result["scopes"] == ["tools.read"]
        assert result["teams"] == ["team-x"]
        assert result["is_admin"] is False

    def test_duplicate_client_id_returns_none(self, oauth_service, sample_client, test_db):
        """Test registering a duplicate client_id returns None."""
        result = asyncio.run(
            oauth_service.register_client(
                client_id="test-client-001",
                client_name="Duplicate",
                scopes=["tools.read"],
                teams=[],
                is_admin=False,
                db=test_db,
            )
        )
        assert result is None

    def test_registration_persists_to_database(self, oauth_service, test_db):
        """Test that registered client is persisted in the database."""
        asyncio.run(
            oauth_service.register_client(
                client_id="persisted-client",
                client_name="Persisted",
                scopes=["tools.read"],
                teams=[],
                is_admin=False,
                db=test_db,
            )
        )

        db_client = test_db.query(OAuthASClient).filter(OAuthASClient.client_id == "persisted-client").first()
        assert db_client is not None
        assert db_client.client_name == "Persisted"


# ---------------------------------------------------------------------------
# rotate_client_secret
# ---------------------------------------------------------------------------


class TestRotateClientSecret:
    """Test client secret rotation."""

    def test_rotation_returns_new_secret(self, oauth_service, sample_client, test_db):
        """Test rotation returns new secret and old secret moves to previous."""
        result = asyncio.run(oauth_service.rotate_client_secret("test-client-001", test_db))

        assert result is not None
        assert result["client_id"] == "test-client-001"
        assert "client_secret" in result
        assert len(result["client_secret"]) > 0

    def test_rotation_sets_grace_period(self, oauth_service, sample_client, test_db):
        """Test rotation sets previous_secret_hash and 5-minute grace period."""
        old_hash = sample_client.client_secret_hash
        asyncio.run(oauth_service.rotate_client_secret("test-client-001", test_db))

        test_db.refresh(sample_client)
        assert sample_client.previous_secret_hash == old_hash
        assert sample_client.previous_secret_expires_at is not None
        # Grace period should be approximately 5 minutes from now
        # SQLite may store naive datetimes, so normalise both sides
        expires_at = sample_client.previous_secret_expires_at
        now = datetime.now(timezone.utc)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        delta = expires_at - now
        assert timedelta(minutes=4) < delta < timedelta(minutes=6)

    def test_rotation_nonexistent_client_returns_none(self, oauth_service, test_db):
        """Test rotation for non-existent client returns None."""
        result = asyncio.run(oauth_service.rotate_client_secret("nonexistent-client", test_db))
        assert result is None


# ---------------------------------------------------------------------------
# get_client / list_clients / deactivate_client / get_client_for_auth
# ---------------------------------------------------------------------------


class TestClientCRUD:
    """Test client CRUD operations."""

    def test_get_client_found(self, oauth_service, sample_client, test_db):
        """Test get_client returns dict for existing client."""
        result = asyncio.run(oauth_service.get_client("test-client-001", test_db))

        assert result is not None
        assert result["client_id"] == "test-client-001"
        assert result["client_name"] == "Test Client"
        assert result["scopes"] == ["tools.read", "tools.execute", "resources.read"]
        assert result["is_active"] is True

    def test_get_client_not_found(self, oauth_service, test_db):
        """Test get_client returns None for non-existent client."""
        result = asyncio.run(oauth_service.get_client("nonexistent", test_db))
        assert result is None

    def test_list_clients(self, oauth_service, sample_client, admin_client, test_db):
        """Test list_clients returns all registered clients."""
        result = asyncio.run(oauth_service.list_clients(test_db))

        assert isinstance(result, list)
        client_ids = {c["client_id"] for c in result}
        assert "test-client-001" in client_ids
        assert "admin-client-001" in client_ids

    def test_list_clients_empty(self, oauth_service, test_db):
        """Test list_clients returns empty list when no clients exist."""
        result = asyncio.run(oauth_service.list_clients(test_db))
        assert isinstance(result, list)

    def test_deactivate_client_success(self, oauth_service, sample_client, test_db):
        """Test deactivate_client returns True and marks client inactive."""
        result = asyncio.run(oauth_service.deactivate_client("test-client-001", test_db))
        assert result is True

        test_db.refresh(sample_client)
        assert sample_client.is_active is False

    def test_deactivate_client_not_found(self, oauth_service, test_db):
        """Test deactivate_client returns False for non-existent client."""
        result = asyncio.run(oauth_service.deactivate_client("nonexistent", test_db))
        assert result is False

    def test_get_client_for_auth_active(self, oauth_service, sample_client, test_db):
        """Test get_client_for_auth returns dict for active client."""
        result = asyncio.run(oauth_service.get_client_for_auth("test-client-001", test_db))

        assert result is not None
        assert result["client_id"] == "test-client-001"

    def test_get_client_for_auth_inactive(self, oauth_service, test_db):
        """Test get_client_for_auth returns None for inactive client."""
        client = OAuthASClient(
            client_id="inactive-for-auth",
            client_name="Inactive For Auth",
            client_secret_hash="hash",
            scopes=[],
            teams=[],
            is_admin=False,
            is_active=False,
        )
        test_db.add(client)
        test_db.commit()

        result = asyncio.run(oauth_service.get_client_for_auth("inactive-for-auth", test_db))
        assert result is None

    def test_get_client_for_auth_not_found(self, oauth_service, test_db):
        """Test get_client_for_auth returns None for non-existent client."""
        result = asyncio.run(oauth_service.get_client_for_auth("does-not-exist", test_db))
        assert result is None

    def test_client_to_dict_contains_timestamps(self, oauth_service, sample_client, test_db):
        """Test _client_to_dict includes created_at and updated_at."""
        result = asyncio.run(oauth_service.get_client("test-client-001", test_db))

        assert "created_at" in result
        assert "updated_at" in result


# ---------------------------------------------------------------------------
# _RateLimitBucket
# ---------------------------------------------------------------------------


class TestRateLimitBucket:
    """Test the internal _RateLimitBucket class."""

    def test_record_returns_count(self):
        """Test record returns count within window."""
        bucket = _RateLimitBucket()
        now = 1000.0
        assert bucket.record(now, 60.0) == 1
        assert bucket.record(now + 1, 60.0) == 2

    def test_count_without_recording(self):
        """Test count returns count without adding a new entry."""
        bucket = _RateLimitBucket()
        now = 1000.0
        bucket.record(now, 60.0)

        assert bucket.count(now + 1, 60.0) == 1
        # count should not have added an entry
        assert bucket.count(now + 1, 60.0) == 1

    def test_sliding_window_eviction(self):
        """Test old timestamps are evicted from sliding window."""
        bucket = _RateLimitBucket()
        bucket.record(1000.0, 60.0)
        bucket.record(1001.0, 60.0)

        # After window passes, old entries evicted
        assert bucket.record(1070.0, 60.0) == 1

    def test_oldest_in_window(self):
        """Test oldest_in_window returns oldest active timestamp."""
        bucket = _RateLimitBucket()
        bucket.record(1000.0, 60.0)
        bucket.record(1010.0, 60.0)

        oldest = bucket.oldest_in_window(1020.0, 60.0)
        assert oldest == 1000.0

    def test_oldest_in_window_empty(self):
        """Test oldest_in_window returns None when no entries in window."""
        bucket = _RateLimitBucket()
        assert bucket.oldest_in_window(1000.0, 60.0) is None


# ---------------------------------------------------------------------------
# cleanup_expired_revocations
# ---------------------------------------------------------------------------


class TestCleanupExpiredRevocations:
    """Test expired revocation cleanup."""

    def test_cleanup_deletes_expired(self, oauth_service, test_db):
        """Test cleanup removes expired revocation entries."""
        expired = OAuthASRevokedToken(
            jti="jti-cleanup-expired",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
            client_id="test",
        )
        active = OAuthASRevokedToken(
            jti="jti-cleanup-active",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            client_id="test",
        )
        test_db.add(expired)
        test_db.add(active)
        test_db.commit()

        count = asyncio.run(oauth_service.cleanup_expired_revocations(test_db))
        assert count >= 1

        # Active entry should remain
        remaining = test_db.query(OAuthASRevokedToken).filter(OAuthASRevokedToken.jti == "jti-cleanup-active").first()
        assert remaining is not None

        # Expired entry should be gone
        gone = test_db.query(OAuthASRevokedToken).filter(OAuthASRevokedToken.jti == "jti-cleanup-expired").first()
        assert gone is None

    def test_cleanup_no_expired_returns_zero(self, oauth_service, test_db):
        """Test cleanup returns zero when no expired entries exist."""
        count = asyncio.run(oauth_service.cleanup_expired_revocations(test_db))
        assert count == 0
