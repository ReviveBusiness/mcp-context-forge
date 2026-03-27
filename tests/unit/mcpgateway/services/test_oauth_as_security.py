# -*- coding: utf-8 -*-
"""Location: ./tests/unit/mcpgateway/services/test_oauth_as_security.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0

Security-focused tests for OAuth Authorization Server.

Covers PRD-1426 security invariants:
- RS256 required (HS256 = vulnerability)
- Algorithm pinning (never RS256+HS256 simultaneously)
- Issuer-bound claim validation
- Scope escalation prevention
- Token revocation enforcement
- Rate limiting
- Secret rotation grace period
"""

# Standard
import secrets
import tempfile
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

# Third-Party
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# First-Party
import mcpgateway.db as db_mod
from mcpgateway.db import OAuthASClient, OAuthASRevokedToken


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keypair():
    """Generate a real RSA key pair for testing."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture(scope="module")
def rsa_key_files(rsa_keypair):
    """Write RSA keys to temporary files and return paths."""
    private_key, public_key = rsa_keypair
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as priv_f:
        priv_f.write(priv_pem)
        priv_path = priv_f.name
    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as pub_f:
        pub_f.write(pub_pem)
        pub_path = pub_f.name
    return priv_path, pub_path


@pytest.fixture
def security_db():
    """Create a fresh in-memory SQLite session for security tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    db_mod.Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture
def mock_settings(rsa_key_files):
    """Patch settings for OAuth AS enabled with real RSA keys."""
    priv_path, pub_path = rsa_key_files
    with patch("mcpgateway.services.oauth_as_service.settings") as mock_s:
        mock_s.oauth_as_enabled = True
        mock_s.oauth_token_ttl = 900
        mock_s.oauth_token_max_ttl = 3600
        mock_s.oauth_issuer = "https://cf.test"
        mock_s.oauth_rs256_private_key_path = priv_path
        mock_s.oauth_rs256_public_key_path = pub_path
        mock_s.oauth_rs256_kid = ""
        mock_s.oauth_rate_limit_per_client = 10
        mock_s.oauth_rate_limit_per_ip = 20
        mock_s.oauth_rate_limit_global = 100
        mock_s.jwt_audience = "contextforge"
        mock_s.external_url = "https://cf.test"
        yield mock_s


@pytest.fixture
def oauth_service(mock_settings):
    """Create a fresh OAuthASService instance with real RSA keys."""
    # Reset singleton
    import mcpgateway.services.oauth_as_service as mod
    mod._oauth_as_service = None
    svc = mod.OAuthASService()
    return svc


@pytest.fixture
def registered_client(oauth_service, security_db):
    """Register a test client and return (client_id, raw_secret, OAuthASClient)."""
    import asyncio

    result = asyncio.run(
        oauth_service.register_client(
            client_id="test-client",
            client_name="Test Client",
            scopes=["tools.read", "tools.execute"],
            teams=["team-a"],
            is_admin=False,
            db=security_db,
        )
    )
    raw_secret = result["client_secret"]
    client = security_db.query(OAuthASClient).filter_by(client_id="test-client").first()
    return "test-client", raw_secret, client


# ---------------------------------------------------------------------------
# Security Tests
# ---------------------------------------------------------------------------


class TestRS256Enforcement:
    """PRD-1426 invariant: RS256 REQUIRED. HS256 = vulnerability, not tradeoff."""

    def test_token_is_signed_with_rs256(self, oauth_service, registered_client, rsa_keypair):
        """Tokens MUST be signed with RS256 algorithm."""
        client_id, raw_secret, client = registered_client
        token_data = oauth_service.issue_token(client)
        token = token_data["access_token"]

        # Decode header without verification to check algorithm
        header = jwt.get_unverified_header(token)
        assert header["alg"] == "RS256", "Token MUST use RS256"
        assert header.get("typ") == "at+jwt", "Token MUST have typ: at+jwt (RFC 9068)"
        assert "kid" in header, "Token MUST include kid for key identification"

    def test_token_verifiable_with_public_key(self, oauth_service, registered_client, rsa_keypair):
        """Tokens MUST be verifiable with the RS256 public key."""
        _, public_key = rsa_keypair
        client_id, raw_secret, client = registered_client
        token_data = oauth_service.issue_token(client)
        token = token_data["access_token"]

        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience="contextforge",
        )
        assert payload["sub"] == client_id
        assert payload["auth_provider"] == "oauth_as"
        assert payload["token_use"] == "m2m"

    def test_token_not_verifiable_with_hs256(self, oauth_service, registered_client):
        """Tokens MUST NOT be decodable with HS256. This prevents key confusion attacks."""
        client_id, raw_secret, client = registered_client
        token_data = oauth_service.issue_token(client)
        token = token_data["access_token"]

        with pytest.raises((jwt.exceptions.DecodeError, jwt.exceptions.InvalidAlgorithmError)):
            jwt.decode(
                token,
                "any-symmetric-key",
                algorithms=["HS256"],
                options={"verify_aud": False},
            )

    def test_service_refuses_to_start_without_rsa_keys(self):
        """Service MUST refuse to start when AS enabled but RSA keys missing."""
        with patch("mcpgateway.services.oauth_as_service.settings") as mock_s:
            mock_s.oauth_as_enabled = True
            mock_s.oauth_rs256_private_key_path = ""
            mock_s.oauth_rs256_public_key_path = ""

            import mcpgateway.services.oauth_as_service as mod
            mod._oauth_as_service = None

            with pytest.raises(RuntimeError, match="RSA key paths are not configured"):
                mod.OAuthASService()


class TestAlgorithmPinning:
    """PRD-1426 invariant: algorithms=["RS256"] ONLY. Never HS256+RS256 simultaneously."""

    def test_issued_token_header_only_rs256(self, oauth_service, registered_client):
        """Token header MUST specify alg=RS256 and nothing else."""
        _, _, client = registered_client
        token_data = oauth_service.issue_token(client)
        header = jwt.get_unverified_header(token_data["access_token"])
        assert header["alg"] == "RS256"

    def test_jwks_specifies_rs256(self, oauth_service):
        """JWKS MUST declare alg=RS256 for all keys."""
        jwks = oauth_service.get_jwks()
        assert len(jwks["keys"]) == 1
        key = jwks["keys"][0]
        assert key["alg"] == "RS256"
        assert key["kty"] == "RSA"
        assert key["use"] == "sig"


class TestIssuerBoundClaims:
    """PRD-1426 invariant: is_admin/teams/auth_provider ONLY trusted when iss matches AS."""

    def test_token_contains_issuer_bound_claims(self, oauth_service, registered_client):
        """M2M tokens MUST include issuer-bound claims."""
        _, _, client = registered_client
        token_data = oauth_service.issue_token(client)
        token = token_data["access_token"]
        payload = jwt.decode(token, options={"verify_signature": False})

        assert payload["iss"] == "https://cf.test"
        assert payload["auth_provider"] == "oauth_as"
        assert payload["token_use"] == "m2m"
        assert payload["teams"] == ["team-a"]
        assert payload["is_admin"] is False
        assert "jti" in payload

    def test_admin_client_gets_admin_claim(self, oauth_service, security_db):
        """Admin clients MUST receive is_admin=true in token."""
        import asyncio

        result = asyncio.run(
            oauth_service.register_client(
                client_id="admin-client",
                client_name="Admin Client",
                scopes=["admin"],
                teams=[],
                is_admin=True,
                db=security_db,
            )
        )
        admin_client = security_db.query(OAuthASClient).filter_by(client_id="admin-client").first()
        token_data = oauth_service.issue_token(admin_client)
        payload = jwt.decode(token_data["access_token"], options={"verify_signature": False})

        assert payload["is_admin"] is True

    def test_non_admin_cannot_escalate_via_token(self, oauth_service, registered_client):
        """Non-admin clients MUST NOT have is_admin=true. Token reflects DB state, not request."""
        _, _, client = registered_client
        # Client registered as non-admin
        token_data = oauth_service.issue_token(client)
        payload = jwt.decode(token_data["access_token"], options={"verify_signature": False})
        assert payload["is_admin"] is False


class TestScopeEscalation:
    """Prevent clients from requesting scopes beyond their grant."""

    def test_cannot_request_unganted_scope(self, oauth_service, registered_client):
        """Requesting a scope not in the client's grant MUST be rejected."""
        from mcpgateway.services.oauth_as_service import InvalidScopeError

        _, _, client = registered_client
        # Client has scopes: ["tools.read", "tools.execute"]
        with pytest.raises(InvalidScopeError):
            oauth_service.issue_token(client, requested_scope="admin")

    def test_can_request_subset_of_granted_scopes(self, oauth_service, registered_client):
        """Requesting a subset of granted scopes MUST succeed."""
        _, _, client = registered_client
        token_data = oauth_service.issue_token(client, requested_scope="tools.read")
        payload = jwt.decode(token_data["access_token"], options={"verify_signature": False})
        assert payload["scope"] == "tools.read"

    def test_default_scope_includes_all_granted(self, oauth_service, registered_client):
        """Omitting scope MUST include all granted scopes."""
        _, _, client = registered_client
        token_data = oauth_service.issue_token(client)
        payload = jwt.decode(token_data["access_token"], options={"verify_signature": False})
        assert "tools.read" in payload["scope"]
        assert "tools.execute" in payload["scope"]


class TestTokenRevocation:
    """PRD-1426: jti deny-list with clock skew buffer (+5 min)."""

    def test_revoked_token_is_detected(self, oauth_service, security_db):
        """Revoked JTI MUST be detected by is_token_revoked."""
        import asyncio

        jti = str(uuid4())
        asyncio.run(
            oauth_service.revoke_token(jti, security_db)
        )
        result = asyncio.run(
            oauth_service.is_token_revoked(jti, security_db)
        )
        assert result is True

    def test_non_revoked_token_passes(self, oauth_service, security_db):
        """Non-revoked JTI MUST pass revocation check."""
        import asyncio

        result = asyncio.run(
            oauth_service.is_token_revoked(str(uuid4()), security_db)
        )
        assert result is False

    def test_revocation_is_idempotent(self, oauth_service, security_db):
        """Revoking the same JTI twice MUST not error."""
        import asyncio

        jti = str(uuid4())
        r1 = asyncio.run(
            oauth_service.revoke_token(jti, security_db)
        )
        r2 = asyncio.run(
            oauth_service.revoke_token(jti, security_db)
        )
        assert r1 is True
        assert r2 is True

    def test_expired_revocation_cleaned_up(self, oauth_service, security_db):
        """Expired deny-list entries MUST be cleaned up."""
        import asyncio

        # Insert an already-expired entry
        expired = OAuthASRevokedToken(
            jti="expired-jti",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            client_id="test",
        )
        security_db.add(expired)
        security_db.commit()

        count = asyncio.run(
            oauth_service.cleanup_expired_revocations(security_db)
        )
        assert count == 1

        result = asyncio.run(
            oauth_service.is_token_revoked("expired-jti", security_db)
        )
        assert result is False


class TestRateLimiting:
    """PRD-1426: per-client (10/min) + per-IP (20/min) + global (100/min)."""

    def test_under_limit_allowed(self, oauth_service, mock_settings):
        """Requests under all limits MUST be allowed."""
        assert oauth_service.check_rate_limit("client-1", "1.2.3.4") is True

    def test_per_client_limit(self, oauth_service, mock_settings):
        """Exceeding per-client limit MUST be rejected."""
        mock_settings.oauth_rate_limit_per_client = 3
        for _ in range(3):
            oauth_service.check_rate_limit("flood-client", "1.2.3.4")
        assert oauth_service.check_rate_limit("flood-client", "1.2.3.4") is False

    def test_per_ip_limit(self, oauth_service, mock_settings):
        """Exceeding per-IP limit MUST be rejected."""
        mock_settings.oauth_rate_limit_per_ip = 3
        for i in range(3):
            oauth_service.check_rate_limit(f"client-{i}", "10.0.0.1")
        assert oauth_service.check_rate_limit("client-new", "10.0.0.1") is False

    def test_different_clients_different_buckets(self, oauth_service, mock_settings):
        """Different clients MUST have separate rate limit buckets."""
        mock_settings.oauth_rate_limit_per_client = 2
        for _ in range(2):
            oauth_service.check_rate_limit("client-a", "1.1.1.1")
        # client-a is at limit, but client-b should be fine
        assert oauth_service.check_rate_limit("client-b", "2.2.2.2") is True

    def test_retry_after_positive(self, oauth_service, mock_settings):
        """Retry-After MUST be a positive integer when rate limited."""
        mock_settings.oauth_rate_limit_per_client = 1
        oauth_service.check_rate_limit("retry-client", "3.3.3.3")
        retry = oauth_service.get_rate_limit_retry_after("retry-client", "3.3.3.3")
        assert retry >= 1


class TestSecretRotation:
    """PRD-1426: 5-minute grace period for old secret during rotation."""

    def test_old_secret_works_during_grace_period(self, oauth_service, security_db, registered_client):
        """Old secret MUST still authenticate during 5-min grace period."""
        import asyncio

        client_id, old_secret, _ = registered_client
        new_secret_result = asyncio.run(
            oauth_service.rotate_client_secret(client_id, security_db)
        )
        new_secret = new_secret_result["client_secret"]

        # Old secret should still work (grace period)
        old_client = asyncio.run(
            oauth_service.authenticate_client(client_id, old_secret, security_db)
        )
        assert old_client is not None
        assert old_client.client_id == client_id

        # New secret should also work
        new_client = asyncio.run(
            oauth_service.authenticate_client(client_id, new_secret, security_db)
        )
        assert new_client is not None

    def test_expired_grace_period_rejects_old_secret(self, oauth_service, security_db):
        """After grace period expires, old secret MUST be rejected."""
        import asyncio
        from mcpgateway.services.oauth_as_service import InvalidClientError

        result = asyncio.run(
            oauth_service.register_client(
                client_id="rotate-test",
                client_name="Rotate Test",
                scopes=["tools.read"],
                teams=[],
                is_admin=False,
                db=security_db,
            )
        )
        old_secret = result["client_secret"]

        asyncio.run(
            oauth_service.rotate_client_secret("rotate-test", security_db)
        )

        # Manually expire the grace period
        client = security_db.query(OAuthASClient).filter_by(client_id="rotate-test").first()
        client.previous_secret_expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        security_db.commit()

        with pytest.raises(InvalidClientError):
            asyncio.run(
                oauth_service.authenticate_client("rotate-test", old_secret, security_db)
            )


class TestClientAuthentication:
    """Client credential verification security."""

    def test_inactive_client_rejected(self, oauth_service, security_db, registered_client):
        """Inactive clients MUST be rejected regardless of valid secret."""
        import asyncio
        from mcpgateway.services.oauth_as_service import InvalidClientError

        client_id, raw_secret, client = registered_client
        client.is_active = False
        security_db.commit()

        with pytest.raises(InvalidClientError):
            asyncio.run(
                oauth_service.authenticate_client(client_id, raw_secret, security_db)
            )

        # Restore for other tests
        client.is_active = True
        security_db.commit()

    def test_wrong_secret_rejected(self, oauth_service, security_db, registered_client):
        """Wrong secret MUST be rejected."""
        import asyncio
        from mcpgateway.services.oauth_as_service import InvalidClientError

        client_id, _, _ = registered_client
        with pytest.raises(InvalidClientError):
            asyncio.run(
                oauth_service.authenticate_client(client_id, "wrong-secret", security_db)
            )

    def test_unknown_client_rejected(self, oauth_service, security_db):
        """Unknown client_id MUST be rejected."""
        import asyncio
        from mcpgateway.services.oauth_as_service import InvalidClientError

        with pytest.raises(InvalidClientError):
            asyncio.run(
                oauth_service.authenticate_client("nonexistent", "any-secret", security_db)
            )

    def test_secret_is_csprng(self, oauth_service, security_db):
        """Generated secrets MUST use CSPRNG (secrets.token_urlsafe)."""
        import asyncio

        result = asyncio.run(
            oauth_service.register_client(
                client_id="csprng-test",
                client_name="CSPRNG Test",
                scopes=[],
                teams=[],
                is_admin=False,
                db=security_db,
            )
        )
        secret = result["client_secret"]
        # token_urlsafe(32) produces 43 chars of base64url
        assert len(secret) >= 40, "Secret must be at least 40 chars (CSPRNG)"

    def test_secret_is_argon2_hashed(self, oauth_service, security_db):
        """Stored secrets MUST be Argon2-hashed, never plaintext."""
        import asyncio

        asyncio.run(
            oauth_service.register_client(
                client_id="hash-check",
                client_name="Hash Check",
                scopes=[],
                teams=[],
                is_admin=False,
                db=security_db,
            )
        )
        client = security_db.query(OAuthASClient).filter_by(client_id="hash-check").first()
        assert client.client_secret_hash.startswith("$argon2"), "Secret MUST be Argon2-hashed"


class TestTokenClaims:
    """RFC 9068 JWT Access Token Profile compliance."""

    def test_required_claims_present(self, oauth_service, registered_client):
        """All RFC 9068 required claims MUST be present."""
        _, _, client = registered_client
        token_data = oauth_service.issue_token(client)
        payload = jwt.decode(token_data["access_token"], options={"verify_signature": False})

        required = ["iss", "sub", "aud", "iat", "exp", "nbf", "jti", "client_id"]
        for claim in required:
            assert claim in payload, f"Missing required claim: {claim}"

    def test_nbf_equals_iat(self, oauth_service, registered_client):
        """nbf MUST equal iat (PRD-1426 requirement)."""
        _, _, client = registered_client
        token_data = oauth_service.issue_token(client)
        payload = jwt.decode(token_data["access_token"], options={"verify_signature": False})
        assert payload["nbf"] == payload["iat"]

    def test_exp_within_max_ttl(self, oauth_service, registered_client, mock_settings):
        """Token expiry MUST not exceed max TTL."""
        _, _, client = registered_client
        token_data = oauth_service.issue_token(client)
        payload = jwt.decode(token_data["access_token"], options={"verify_signature": False})

        ttl = payload["exp"] - payload["iat"]
        assert ttl == mock_settings.oauth_token_ttl
        assert ttl <= mock_settings.oauth_token_max_ttl

    def test_jti_is_unique(self, oauth_service, registered_client):
        """Each token MUST have a unique JTI (UUID4)."""
        _, _, client = registered_client
        jtis = set()
        for _ in range(10):
            token_data = oauth_service.issue_token(client)
            payload = jwt.decode(token_data["access_token"], options={"verify_signature": False})
            jtis.add(payload["jti"])
        assert len(jtis) == 10, "All JTIs must be unique"


class TestASMetadata:
    """RFC 8414 Authorization Server Metadata compliance."""

    def test_metadata_structure(self, oauth_service):
        """AS metadata MUST include required fields."""
        metadata = oauth_service.get_as_metadata("https://cf.test")
        assert metadata["issuer"] == "https://cf.test"
        assert metadata["token_endpoint"] == "https://cf.test/oauth/token"
        assert metadata["jwks_uri"] == "https://cf.test/oauth/jwks"
        assert metadata["grant_types_supported"] == ["client_credentials"]
        assert metadata["response_types_supported"] == ["none"]
        assert "client_secret_basic" in metadata["token_endpoint_auth_methods_supported"]
        assert "client_secret_post" in metadata["token_endpoint_auth_methods_supported"]
