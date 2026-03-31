# -*- coding: utf-8 -*-
"""Location: ./mcpgateway/services/oauth_as_service.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0

OAuth 2.1 Authorization Server Service for ContextForge.

This module implements the core OAuth Authorization Server logic including:
- Client authentication (client_credentials grant)
- RS256 JWT token issuance with at+jwt type
- Per-client, per-IP, and global rate limiting
- Token revocation via JTI deny-list
- JWKS endpoint support
- RFC 8414 Authorization Server Metadata
- Client registration and secret rotation
"""

# Standard
import base64
import hashlib
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

# Third-Party
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
from sqlalchemy.orm import Session

# First-Party
from mcpgateway.config import settings
from mcpgateway.db import OAuthASClient, OAuthASRevokedToken
from mcpgateway.services.argon2_service import Argon2PasswordService
from mcpgateway.services.logging_service import LoggingService

# Initialize logging
logging_service = LoggingService()
logger = logging_service.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OAuthASError(Exception):
    """Base exception for OAuth Authorization Server errors."""


class InvalidClientError(OAuthASError):
    """Raised when client authentication fails (unknown client or bad secret)."""


class InvalidScopeError(OAuthASError):
    """Raised when requested scope exceeds the client's granted scopes."""


class RateLimitExceededError(OAuthASError):
    """Raised when a rate limit has been exceeded."""


# ---------------------------------------------------------------------------
# Rate-limit bucket (in-memory, sliding window)
# ---------------------------------------------------------------------------


class _RateLimitBucket:
    """Thread-safe sliding-window counter for rate limiting."""

    __slots__ = ("_timestamps", "_lock")

    def __init__(self) -> None:
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def record(self, now: float, window: float) -> int:
        """Record a request and return the count within the window."""
        with self._lock:
            cutoff = now - window
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            self._timestamps.append(now)
            return len(self._timestamps)

    def count(self, now: float, window: float) -> int:
        """Return request count within window without recording."""
        with self._lock:
            cutoff = now - window
            return sum(1 for t in self._timestamps if t > cutoff)

    def oldest_in_window(self, now: float, window: float) -> Optional[float]:
        """Return the oldest timestamp still within the window."""
        with self._lock:
            cutoff = now - window
            active = [t for t in self._timestamps if t > cutoff]
            return min(active) if active else None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class OAuthASService:
    """OAuth 2.1 Authorization Server service.

    Handles token issuance, client authentication, rate limiting, token
    revocation, JWKS, and AS metadata for ContextForge's built-in OAuth AS.
    """

    # Rate-limit window in seconds
    _RATE_LIMIT_WINDOW: float = 60.0

    def __init__(self) -> None:
        """Initialize the OAuth AS service.

        Loads RSA key pair, derives kid, and sets up rate-limit tracking.

        Raises:
            RuntimeError: If ``oauth_as_enabled`` is True but RSA key paths
                are not configured or the key files cannot be loaded.
        """
        if not settings.oauth_as_enabled:
            logger.info("OAuth Authorization Server is disabled (oauth_as_enabled=False)")
            self._private_key = None
            self._public_key = None
            self.kid = ""
            return

        # ---- RSA key loading (mandatory when AS is enabled) ---- #
        private_key_path = settings.oauth_rs256_private_key_path
        public_key_path = settings.oauth_rs256_public_key_path

        if not private_key_path or not public_key_path:
            raise RuntimeError("OAuth AS is enabled but RSA key paths are not configured. " "Set OAUTH_RS256_PRIVATE_KEY_PATH and OAUTH_RS256_PUBLIC_KEY_PATH.")

        try:
            with open(private_key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
            logger.info("Loaded RS256 private key from %s", private_key_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load RS256 private key from {private_key_path}: {exc}") from exc

        try:
            with open(public_key_path, "rb") as f:
                self._public_key = serialization.load_pem_public_key(f.read())
            logger.info("Loaded RS256 public key from %s", public_key_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load RS256 public key from {public_key_path}: {exc}") from exc

        # ---- kid derivation ---- #
        if settings.oauth_rs256_kid:
            self.kid = settings.oauth_rs256_kid
        else:
            # Derive kid from SHA-256 fingerprint of the DER-encoded public key
            pub_der = self._public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            self.kid = hashlib.sha256(pub_der).hexdigest()[:16]
            logger.info("Auto-generated RS256 kid=%s from public key fingerprint", self.kid)

        # ---- Argon2 hasher ---- #
        self._argon2 = Argon2PasswordService()

        # ---- Rate-limit buckets (in-memory) ---- #
        self._client_buckets: Dict[str, _RateLimitBucket] = {}
        self._ip_buckets: Dict[str, _RateLimitBucket] = {}
        self._global_bucket = _RateLimitBucket()
        self._bucket_lock = threading.Lock()

        logger.info("OAuth Authorization Server service initialized (kid=%s)", self.kid)

    # ------------------------------------------------------------------
    # Client Authentication
    # ------------------------------------------------------------------

    async def authenticate_client(self, client_id: str, client_secret: str, db: Session) -> OAuthASClient:
        """Authenticate an OAuth client by client_id and client_secret.

        Args:
            client_id: The client identifier.
            client_secret: The plaintext client secret to verify.
            db: SQLAlchemy database session.

        Returns:
            The authenticated ``OAuthASClient`` record.

        Raises:
            InvalidClientError: If the client is not found, inactive, or the
                secret does not match (including grace-period check).
        """
        client: Optional[OAuthASClient] = db.query(OAuthASClient).filter(OAuthASClient.client_id == client_id, OAuthASClient.is_active.is_(True)).first()

        if client is None:
            logger.warning("OAuth AS client authentication failed: client_id=%s not found or inactive", client_id)
            raise InvalidClientError("invalid_client")

        # Primary secret check (argon2 is constant-time internally)
        if self._argon2.verify_password(client_secret, client.client_secret_hash):
            logger.info("OAuth AS client authenticated: client_id=%s (primary secret)", client_id)
            return client

        # Grace-period fallback to previous secret
        if client.previous_secret_hash and client.previous_secret_expires_at:
            # Handle both naive (SQLite) and aware (PostgreSQL) datetimes
            grace_expiry = client.previous_secret_expires_at
            if grace_expiry.tzinfo is None:
                grace_expiry = grace_expiry.replace(tzinfo=timezone.utc)
            if grace_expiry > datetime.now(timezone.utc):
                if self._argon2.verify_password(client_secret, client.previous_secret_hash):
                    logger.info(
                        "OAuth AS client authenticated: client_id=%s (previous secret, grace period until %s)",
                        client_id,
                        client.previous_secret_expires_at.isoformat(),
                    )
                    return client

        logger.warning("OAuth AS client authentication failed: client_id=%s invalid secret", client_id)
        raise InvalidClientError("invalid_client")

    # ------------------------------------------------------------------
    # Token Issuance
    # ------------------------------------------------------------------

    def issue_token(self, client: OAuthASClient, requested_scope: Optional[str] = None) -> dict:
        """Issue a signed RS256 JWT access token for an authenticated client.

        Args:
            client: The authenticated ``OAuthASClient`` record.
            requested_scope: Space-delimited scope string requested by the
                client. If ``None``, the client's full granted scope is used.

        Returns:
            Token response dict with ``access_token``, ``token_type``,
            ``expires_in``, and ``scope`` keys.

        Raises:
            InvalidScopeError: If any requested scope is not in the client's
                granted scopes.
        """
        granted_scopes: List[str] = client.scopes or []

        if requested_scope is not None:
            requested = requested_scope.split()
            invalid = [s for s in requested if s not in granted_scopes]
            if invalid:
                raise InvalidScopeError(f"Scope(s) not granted: {', '.join(invalid)}")
            effective_scopes = requested
        else:
            effective_scopes = granted_scopes

        scope_str = " ".join(effective_scopes)

        now = datetime.now(timezone.utc)
        token_ttl = settings.oauth_token_ttl
        jti = str(uuid4())

        # Determine issuer
        issuer = settings.oauth_issuer
        if not issuer:
            # Derive from a well-known external URL if available
            issuer = getattr(settings, "external_url", None) or "contextforge"

        # JWT header
        header = {
            "alg": "RS256",
            "typ": "at+jwt",
            "kid": self.kid,
        }

        # JWT payload (RFC 9068 JWT Access Token Profile)
        payload = {
            "iss": issuer,
            "sub": client.client_id,
            "aud": settings.jwt_audience,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=token_ttl)).timestamp()),
            "nbf": int(now.timestamp()),
            "jti": jti,
            "scope": scope_str,
            "client_id": client.client_id,
            "teams": client.teams or [],
            "is_admin": client.is_admin,
            "auth_provider": "oauth_as",
            "token_use": "m2m",
        }

        token = jwt.encode(payload, self._private_key, algorithm="RS256", headers=header)

        logger.info(
            "OAuth AS issued token: client_id=%s scope='%s' jti=%s ttl=%ds",
            client.client_id,
            scope_str,
            jti,
            token_ttl,
        )

        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": token_ttl,
            "scope": scope_str,
        }

    # ------------------------------------------------------------------
    # Rate Limiting
    # ------------------------------------------------------------------

    def _get_bucket(self, bucket_dict: Dict[str, _RateLimitBucket], key: str) -> _RateLimitBucket:
        """Get or create a rate-limit bucket for the given key."""
        with self._bucket_lock:
            if key not in bucket_dict:
                bucket_dict[key] = _RateLimitBucket()
            return bucket_dict[key]

    def check_rate_limit(self, client_id: str, ip_address: str) -> bool:
        """Check whether the request is within rate limits.

        Tracks per-client, per-IP, and global counters using a 60-second
        sliding window.

        Args:
            client_id: The OAuth client identifier.
            ip_address: The caller's IP address.

        Returns:
            ``True`` if the request is allowed, ``False`` if any limit is
            exceeded.
        """
        now = time.monotonic()
        window = self._RATE_LIMIT_WINDOW

        # Per-client
        client_bucket = self._get_bucket(self._client_buckets, client_id)
        client_count = client_bucket.record(now, window)
        if client_count > settings.oauth_rate_limit_per_client:
            logger.warning("OAuth AS rate limit exceeded: client_id=%s count=%d", client_id, client_count)
            return False

        # Per-IP
        ip_bucket = self._get_bucket(self._ip_buckets, ip_address)
        ip_count = ip_bucket.record(now, window)
        if ip_count > settings.oauth_rate_limit_per_ip:
            logger.warning("OAuth AS rate limit exceeded: ip=%s count=%d", ip_address, ip_count)
            return False

        # Global
        global_count = self._global_bucket.record(now, window)
        if global_count > settings.oauth_rate_limit_global:
            logger.warning("OAuth AS global rate limit exceeded: count=%d", global_count)
            return False

        return True

    def get_rate_limit_retry_after(self, client_id: str, ip_address: str) -> int:
        """Return seconds until the rate limit resets for the given identifiers.

        Args:
            client_id: The OAuth client identifier.
            ip_address: The caller's IP address.

        Returns:
            Number of seconds until the earliest bucket opens up, minimum 1.
        """
        now = time.monotonic()
        window = self._RATE_LIMIT_WINDOW
        retry_candidates: List[float] = []

        for bucket_dict, key in [
            (self._client_buckets, client_id),
            (self._ip_buckets, ip_address),
        ]:
            bucket = bucket_dict.get(key)
            if bucket:
                oldest = bucket.oldest_in_window(now, window)
                if oldest is not None:
                    retry_candidates.append(window - (now - oldest))

        oldest_global = self._global_bucket.oldest_in_window(now, window)
        if oldest_global is not None:
            retry_candidates.append(window - (now - oldest_global))

        if retry_candidates:
            return max(1, int(min(retry_candidates)) + 1)
        return 1

    # ------------------------------------------------------------------
    # Token Revocation
    # ------------------------------------------------------------------

    async def revoke_token(self, jti: str, db: Session) -> bool:
        """Add a token JTI to the revocation deny-list.

        Args:
            jti: The JWT ID of the token to revoke.
            db: SQLAlchemy database session.

        Returns:
            ``True`` if the token was revoked (or already revoked).
        """
        # Check if already revoked
        existing = db.query(OAuthASRevokedToken).filter(OAuthASRevokedToken.jti == jti).first()
        if existing is not None:
            return True

        # Default deny-list expiry: max TTL + 5 min buffer
        deny_until = datetime.now(timezone.utc) + timedelta(seconds=settings.oauth_token_max_ttl + 300)

        revoked = OAuthASRevokedToken(
            jti=jti,
            expires_at=deny_until,
            client_id="admin-revocation",
        )
        db.add(revoked)
        db.commit()

        logger.info("OAuth AS token revoked: jti=%s deny_until=%s", jti, deny_until.isoformat())
        return True

    async def is_token_revoked(self, jti: str, db: Optional[Session] = None) -> bool:
        """Check whether a token JTI appears in the revocation deny-list.

        Args:
            jti: The JWT ID to check.
            db: SQLAlchemy database session. If ``None``, creates a new session.

        Returns:
            ``True`` if the token has been revoked and the deny-list entry
            has not yet expired.
        """
        close_db = False
        if db is None:
            from mcpgateway.db import SessionLocal  # pylint: disable=import-outside-toplevel

            db = SessionLocal()
            close_db = True
        try:
            now = datetime.now(timezone.utc)
            revoked = (
                db.query(OAuthASRevokedToken)
                .filter(
                    OAuthASRevokedToken.jti == jti,
                    OAuthASRevokedToken.expires_at > now,
                )
                .first()
            )
            return revoked is not None
        finally:
            if close_db:
                db.close()

    # ------------------------------------------------------------------
    # JWKS
    # ------------------------------------------------------------------

    def get_jwks(self) -> dict:
        """Return the JSON Web Key Set containing the RS256 public key.

        The JWKS is formatted per RFC 7517 with the key parameters extracted
        from the loaded RSA public key.

        Returns:
            JWKS dict suitable for JSON serialization.
        """
        pub_numbers: RSAPublicNumbers = self._public_key.public_numbers()

        def _b64url_uint(value: int) -> str:
            """Encode an unsigned integer as base64url without padding."""
            byte_length = (value.bit_length() + 7) // 8
            value_bytes = value.to_bytes(byte_length, byteorder="big")
            return base64.urlsafe_b64encode(value_bytes).rstrip(b"=").decode("ascii")

        return {
            "keys": [
                {
                    "kty": "RSA",
                    "use": "sig",
                    "alg": "RS256",
                    "kid": self.kid,
                    "n": _b64url_uint(pub_numbers.n),
                    "e": _b64url_uint(pub_numbers.e),
                }
            ]
        }

    # ------------------------------------------------------------------
    # AS Metadata (RFC 8414)
    # ------------------------------------------------------------------

    def get_as_metadata(self, base_url: str) -> dict:
        """Return RFC 8414 Authorization Server Metadata.

        Args:
            base_url: The external base URL of the gateway (e.g.
                ``https://gateway.example.com``).

        Returns:
            Metadata dict suitable for JSON serialization.
        """
        issuer = settings.oauth_issuer or base_url

        metadata: dict = {
            "issuer": issuer,
            "authorization_endpoint": f"{base_url}/oauth/authorize",
            "token_endpoint": f"{base_url}/oauth/token",
            "jwks_uri": f"{base_url}/oauth/jwks",
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
            "grant_types_supported": ["client_credentials"],
            "scopes_supported": [
                "tools.read",
                "tools.execute",
                "resources.read",
                "prompts.read",
                "servers.read",
                "servers.manage",
                "admin",
                "mcp:access",
            ],
            "response_types_supported": ["none"],
            "service_documentation": f"{base_url}/docs",
        }

        # Include registration_endpoint only when DCR is enabled (RFC 7591 §3.1)
        if settings.oauth_dcr_mode != "disabled":
            metadata["registration_endpoint"] = f"{base_url}/oauth/register"

        return metadata

    # ------------------------------------------------------------------
    # Dynamic Client Registration (RFC 7591) — server-side
    # ------------------------------------------------------------------

    async def register_dcr_client(
        self,
        client_name: str,
        grant_types: List[str],
        token_endpoint_auth_method: str,
        requested_scope: Optional[str],
        db: Session,
    ) -> dict:
        """Register a new OAuth client via RFC 7591 Dynamic Client Registration.

        Auto-generates a ``client_id`` (the caller cannot supply their own).
        Scope is restricted to ``oauth_dcr_default_scopes`` — admin scope is
        never granted via DCR.

        Args:
            client_name: Human-readable client name (required per RFC 7591).
            grant_types: List of requested grant types; only
                ``client_credentials`` is supported — others are silently
                ignored.
            token_endpoint_auth_method: Requested auth method; must be
                ``client_secret_basic`` or ``client_secret_post``.
            requested_scope: Space-delimited scope string. Each scope is
                intersected with the DCR default scopes — unknown or elevated
                scopes are dropped (not rejected, per RFC 7591 §3.2).
            db: SQLAlchemy database session.

        Returns:
            RFC 7591 ClientInformation dict including ``client_id``,
            ``client_secret``, ``client_id_issued_at``, and
            ``client_secret_expires_at`` (0 = non-expiring).

        Raises:
            ValueError: If ``client_name`` is empty or
                ``token_endpoint_auth_method`` is not supported.
        """
        if not client_name or not client_name.strip():
            raise ValueError("client_name is required")

        supported_auth_methods = {"client_secret_basic", "client_secret_post", "none"}
        if token_endpoint_auth_method not in supported_auth_methods:
            raise ValueError(f"token_endpoint_auth_method must be one of: {', '.join(sorted(supported_auth_methods))}")
        # RFC 7591: "none" = public client (no client authentication). We still
        # issue a client_secret so the client can use client_credentials grant.
        # Normalise to client_secret_basic for token endpoint auth.
        if token_endpoint_auth_method == "none":
            token_endpoint_auth_method = "client_secret_basic"

        # Scope intersection: DCR clients get default scopes only, never admin
        default_scopes: List[str] = list(settings.oauth_dcr_default_scopes)
        if requested_scope:
            requested = requested_scope.split()
            # Intersection — unknown scopes are silently dropped per RFC 7591 §3.2
            effective_scopes = [s for s in requested if s in default_scopes]
            if not effective_scopes:
                effective_scopes = default_scopes
        else:
            effective_scopes = default_scopes

        # Auto-generate a stable, URL-safe client_id
        import uuid  # pylint: disable=import-outside-toplevel

        client_id = f"dcr-{uuid.uuid4().hex[:12]}"

        # Delegate to existing register_client (handles hashing, DB write, logging)
        result = await self.register_client(
            client_id=client_id,
            client_name=client_name.strip(),
            scopes=effective_scopes,
            teams=[],
            is_admin=False,
            db=db,
        )

        if result is None:
            # UUID collision — astronomically unlikely but handle gracefully
            client_id = f"dcr-{uuid.uuid4().hex[:12]}"
            result = await self.register_client(
                client_id=client_id,
                client_name=client_name.strip(),
                scopes=effective_scopes,
                teams=[],
                is_admin=False,
                db=db,
            )

        issued_at = int(datetime.now(timezone.utc).timestamp())

        return {
            "client_id": result["client_id"],
            "client_secret": result["client_secret"],
            "client_id_issued_at": issued_at,
            "client_secret_expires_at": 0,  # 0 = non-expiring per RFC 7591 §3.2.1
            "client_name": result["client_name"],
            "grant_types": ["client_credentials"],
            "token_endpoint_auth_method": token_endpoint_auth_method,
            "scope": " ".join(effective_scopes),
        }

    # ------------------------------------------------------------------
    # Client Registration
    # ------------------------------------------------------------------

    async def register_client(
        self,
        client_id: str,
        client_name: str,
        scopes: List[str],
        teams: list,
        is_admin: bool,
        db: Session,
    ) -> Optional[dict]:
        """Register a new OAuth AS client.

        Generates a CSPRNG secret, hashes it with Argon2, and persists the
        client record. The raw secret is returned once and cannot be recovered.

        Args:
            client_id: Unique client identifier.
            client_name: Human-readable client name.
            scopes: List of granted scope strings.
            teams: List of team identifiers the client belongs to.
            is_admin: Whether the client has admin privileges.
            db: SQLAlchemy database session.

        Returns:
            Dict with client info and raw secret, or None if client_id already exists.
        """
        # Check for existing client
        existing = db.query(OAuthASClient).filter(OAuthASClient.client_id == client_id).first()
        if existing is not None:
            return None

        raw_secret = secrets.token_urlsafe(32)
        secret_hash = self._argon2.hash_password(raw_secret)

        client = OAuthASClient(
            client_id=client_id,
            client_name=client_name,
            client_secret_hash=secret_hash,
            scopes=scopes,
            teams=teams,
            is_admin=is_admin,
        )
        db.add(client)
        db.commit()
        db.refresh(client)

        logger.info("OAuth AS client registered: client_id=%s client_name=%s", client_id, client_name)

        result = self._client_to_dict(client)
        result["client_secret"] = raw_secret
        return result

    # ------------------------------------------------------------------
    # Secret Rotation
    # ------------------------------------------------------------------

    async def rotate_client_secret(self, client_id: str, db: Session) -> Optional[dict]:
        """Rotate the client secret with a 5-minute grace period.

        The current secret hash is moved to ``previous_secret_hash`` with a
        5-minute expiry window, allowing in-flight requests to complete.

        Args:
            client_id: The client identifier to rotate.
            db: SQLAlchemy database session.

        Returns:
            Dict with client_id and new secret, or None if client not found.
        """
        client: Optional[OAuthASClient] = db.query(OAuthASClient).filter(OAuthASClient.client_id == client_id).first()
        if client is None:
            return None

        # Move current to previous with grace period
        client.previous_secret_hash = client.client_secret_hash
        client.previous_secret_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)

        # Generate and store new secret
        raw_secret = secrets.token_urlsafe(32)
        client.client_secret_hash = self._argon2.hash_password(raw_secret)

        db.commit()

        logger.info(
            "OAuth AS client secret rotated: client_id=%s grace_until=%s",
            client_id,
            client.previous_secret_expires_at.isoformat(),
        )
        return {"client_id": client_id, "client_secret": raw_secret}

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    async def cleanup_expired_revocations(self, db: Session) -> int:
        """Delete expired entries from the token revocation deny-list.

        Args:
            db: SQLAlchemy database session.

        Returns:
            Number of deleted records.
        """
        now = datetime.now(timezone.utc)
        count = db.query(OAuthASRevokedToken).filter(OAuthASRevokedToken.expires_at < now).delete()
        db.commit()

        if count:
            logger.info("OAuth AS cleaned up %d expired revocation entries", count)
        return count

    # ------------------------------------------------------------------
    # Client CRUD (admin operations)
    # ------------------------------------------------------------------

    async def get_client(self, client_id: str, db: Session) -> Optional[dict]:
        """Get a client by ID with secret masked.

        Args:
            client_id: The client identifier to look up.
            db: SQLAlchemy database session.

        Returns:
            Client dict with secret masked, or ``None`` if not found.
        """
        client: Optional[OAuthASClient] = db.query(OAuthASClient).filter(OAuthASClient.client_id == client_id).first()
        if client is None:
            return None
        return self._client_to_dict(client)

    async def list_clients(self, db: Session) -> List[dict]:
        """List all registered clients with secrets masked.

        Args:
            db: SQLAlchemy database session.

        Returns:
            List of client dicts with secrets masked.
        """
        clients = db.query(OAuthASClient).all()
        return [self._client_to_dict(c) for c in clients]

    async def deactivate_client(self, client_id: str, db: Session) -> bool:
        """Soft-delete a client by marking it inactive.

        Args:
            client_id: The client identifier to deactivate.
            db: SQLAlchemy database session.

        Returns:
            ``True`` if the client was deactivated, ``False`` if not found.
        """
        client: Optional[OAuthASClient] = db.query(OAuthASClient).filter(OAuthASClient.client_id == client_id).first()
        if client is None:
            return False
        client.is_active = False
        db.commit()
        logger.info("OAuth AS client deactivated: client_id=%s", client_id)
        return True

    async def get_client_for_auth(self, client_id: str, db: Optional[Session] = None) -> Optional[dict]:
        """Look up an active client by client_id for token validation.

        Used by the M2M code path in ``get_current_user()`` to verify that the
        client referenced in a JWT ``sub`` claim exists and is active.

        Args:
            client_id: The client identifier (from JWT ``sub``).
            db: SQLAlchemy session. If ``None``, creates a new session.

        Returns:
            Client dict or ``None`` if not found/inactive.
        """
        close_db = False
        if db is None:
            from mcpgateway.db import SessionLocal  # pylint: disable=import-outside-toplevel

            db = SessionLocal()
            close_db = True
        try:
            client: Optional[OAuthASClient] = db.query(OAuthASClient).filter(OAuthASClient.client_id == client_id, OAuthASClient.is_active.is_(True)).first()
            if client is None:
                return None
            return self._client_to_dict(client)
        finally:
            if close_db:
                db.close()

    @staticmethod
    def _client_to_dict(client: OAuthASClient) -> dict:
        """Convert an OAuthASClient ORM object to a response dict."""
        return {
            "client_id": client.client_id,
            "client_name": client.client_name,
            "scopes": client.scopes or [],
            "teams": client.teams or [],
            "is_admin": client.is_admin,
            "is_active": client.is_active,
            "created_at": client.created_at.isoformat() if client.created_at else None,
            "updated_at": client.updated_at.isoformat() if client.updated_at else None,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_oauth_as_service: Optional[OAuthASService] = None


def get_oauth_as_service() -> OAuthASService:
    """Return the module-level OAuthASService singleton.

    Creates the instance on first call. Thread-safe via GIL for the initial
    assignment; subsequent calls just return the cached reference.
    """
    global _oauth_as_service
    if _oauth_as_service is None:
        _oauth_as_service = OAuthASService()
    return _oauth_as_service
