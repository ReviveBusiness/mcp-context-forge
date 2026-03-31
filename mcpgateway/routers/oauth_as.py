# -*- coding: utf-8 -*-
"""Location: ./mcpgateway/routers/oauth_as.py
Copyright 2025
SPDX-License-Identifier: Apache-2.0

OAuth 2.1 Authorization Server Router for ContextForge.

This module implements the OAuth 2.1 Authorization Server endpoints:
- POST /oauth/token - Client credentials grant (unauthenticated)
- GET /oauth/jwks - JSON Web Key Set (unauthenticated)
- GET /.well-known/oauth-authorization-server - AS metadata (unauthenticated)
- Admin endpoints for client management, secret rotation, and token revocation
"""

# Standard
import base64
from typing import List, Optional

# Third-Party
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# First-Party
from mcpgateway.config import settings
from mcpgateway.db import SessionLocal
from mcpgateway.middleware.rbac import get_current_user_with_permissions
from mcpgateway.services.logging_service import LoggingService

# Get logger instance
logging_service = LoggingService()
logger = logging_service.get_logger(__name__)

oauth_as_router = APIRouter(tags=["oauth-as"])


# ---------------------------------------------------------------------------
# Guard: return 404 when the AS feature is disabled
# ---------------------------------------------------------------------------


def _require_oauth_as_enabled() -> None:
    """Dependency that raises 404 when oauth_as_enabled is False."""
    if not settings.oauth_as_enabled:
        raise HTTPException(status_code=404, detail="Not found")


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ClientRegistrationRequest(BaseModel):
    """Request body for registering a new OAuth client."""

    client_id: str = Field(..., description="Unique client identifier")
    client_name: str = Field(..., description="Human-readable client name")
    scopes: List[str] = Field(default_factory=list, description="Allowed scopes for this client")
    teams: List[str] = Field(default_factory=list, description="Team memberships for the client")
    is_admin: bool = Field(default=False, description="Whether the client has admin privileges")


class ClientResponse(BaseModel):
    """Response body for client information (secret masked)."""

    client_id: str
    client_name: str
    scopes: List[str]
    teams: List[str]
    is_admin: bool
    is_active: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ClientRegistrationResponse(ClientResponse):
    """Response body for client registration (includes raw secret, one-time only)."""

    client_secret: str = Field(..., description="Raw client secret (shown once)")


class SecretRotationResponse(BaseModel):
    """Response body for secret rotation (includes new raw secret, one-time only)."""

    client_id: str
    client_secret: str = Field(..., description="New raw client secret (shown once)")


class DCRRegistrationRequest(BaseModel):
    """RFC 7591 Dynamic Client Registration request body."""

    client_name: str = Field(..., description="Human-readable client name (required)")
    redirect_uris: List[str] = Field(default_factory=list, description="Redirect URIs (informational for client_credentials clients)")
    grant_types: List[str] = Field(default=["client_credentials"], description="Requested grant types")
    token_endpoint_auth_method: str = Field(default="client_secret_basic", description="Token endpoint auth method")
    scope: Optional[str] = Field(default=None, description="Requested scopes (space-delimited)")


class DCRRegistrationResponse(BaseModel):
    """RFC 7591 ClientInformation response."""

    client_id: str
    client_secret: str
    client_id_issued_at: int
    client_secret_expires_at: int
    client_name: str
    grant_types: List[str]
    token_endpoint_auth_method: str
    scope: Optional[str] = None


class TokenRevokeRequest(BaseModel):
    """Request body for revoking a token by JTI."""

    jti: str = Field(..., description="JWT ID of the token to revoke")


class TokenRevokeResponse(BaseModel):
    """Response body for token revocation."""

    revoked: bool
    jti: str


class TokenResponse(BaseModel):
    """OAuth 2.0 token response."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    scope: Optional[str] = None


class OAuthErrorResponse(BaseModel):
    """OAuth 2.0 error response per RFC 6749 Section 5.2."""

    error: str
    error_description: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: local DB session
# ---------------------------------------------------------------------------


def _get_db():
    """Get a database session for dependency injection."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helper: admin check
# ---------------------------------------------------------------------------


def _require_admin(current_user_ctx: dict) -> None:
    """Raise 403 if the current user is not an admin."""
    user = current_user_ctx.get("user")
    if not user or not getattr(user, "is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")


# ---------------------------------------------------------------------------
# Helper: parse client credentials from request
# ---------------------------------------------------------------------------


def _parse_client_credentials(request: Request, client_id_form: Optional[str], client_secret_form: Optional[str]) -> tuple:
    """Extract client_id and client_secret from form body or HTTP Basic auth.

    Supports two methods per OAuth 2.1:
    - client_secret_post: client_id + client_secret in form body
    - client_secret_basic: HTTP Basic auth header (base64 of client_id:client_secret)

    Args:
        request: FastAPI request object
        client_id_form: client_id from form body (may be None)
        client_secret_form: client_secret from form body (may be None)

    Returns:
        Tuple of (client_id, client_secret)

    Raises:
        HTTPException: If no valid credentials are found
    """
    # Try form body first (client_secret_post)
    if client_id_form and client_secret_form:
        return client_id_form, client_secret_form

    # Try HTTP Basic auth (client_secret_basic)
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            if ":" in decoded:
                basic_id, basic_secret = decoded.split(":", 1)
                return basic_id, basic_secret
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid_client",
        headers={"WWW-Authenticate": "Basic realm=\"oauth\""},
    )


# ---------------------------------------------------------------------------
# Public endpoints (unauthenticated)
# ---------------------------------------------------------------------------


@oauth_as_router.post(
    "/oauth/token",
    response_model=TokenResponse,
    responses={
        400: {"model": OAuthErrorResponse},
        401: {"model": OAuthErrorResponse},
        429: {"model": OAuthErrorResponse},
    },
    dependencies=[Depends(_require_oauth_as_enabled)],
)
async def oauth_token(
    request: Request,
    grant_type: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    scope: Optional[str] = None,
    db: Session = Depends(_get_db),
):
    """OAuth 2.1 Token Endpoint (client_credentials grant).

    Authenticates the client and issues an access token. Supports both
    client_secret_post (form body) and client_secret_basic (HTTP Basic) methods.

    Args:
        request: FastAPI request object
        grant_type: Must be "client_credentials"
        client_id: Client identifier (form body)
        client_secret: Client secret (form body)
        scope: Requested scopes (space-separated)
        db: Database session

    Returns:
        JSONResponse with access_token, token_type, expires_in, and scope

    Raises:
        HTTPException: On authentication failure, unsupported grant type, or rate limiting
    """
    # First-Party — lazy import to avoid circular dependencies
    from mcpgateway.services.oauth_as_service import (
        InvalidClientError,
        InvalidScopeError,
        get_oauth_as_service,
    )

    service = get_oauth_as_service()

    # Parse form data (FastAPI doesn't auto-parse form for optional fields in all cases)
    if grant_type is None:
        form = await request.form()
        grant_type = form.get("grant_type")
        client_id = form.get("client_id") or client_id
        client_secret = form.get("client_secret") or client_secret
        scope = form.get("scope") or scope

    # Validate grant_type
    if grant_type != "client_credentials":
        return JSONResponse(
            status_code=400,
            content={"error": "unsupported_grant_type", "error_description": "Only client_credentials grant is supported"},
        )

    # Parse credentials (form body or Basic auth)
    try:
        parsed_client_id, parsed_client_secret = _parse_client_credentials(request, client_id, client_secret)
    except HTTPException:
        return JSONResponse(status_code=401, content={"error": "invalid_client", "error_description": "Missing or invalid client credentials"})

    # Rate limiting (check before expensive argon2 verification)
    client_ip = request.client.host if request.client else "unknown"
    if not service.check_rate_limit(parsed_client_id, client_ip):
        retry_after = service.get_rate_limit_retry_after(parsed_client_id, client_ip)
        return JSONResponse(
            status_code=429,
            content={"error": "slow_down", "error_description": "Rate limit exceeded"},
            headers={"Retry-After": str(retry_after)},
        )

    # Authenticate client (raises InvalidClientError on failure)
    try:
        client_record = await service.authenticate_client(parsed_client_id, parsed_client_secret, db)
    except InvalidClientError:
        return JSONResponse(status_code=401, content={"error": "invalid_client", "error_description": "Invalid client credentials"})

    # Issue token (service handles scope validation internally)
    try:
        token_data = service.issue_token(client_record, scope)
    except InvalidScopeError as e:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_scope", "error_description": str(e)},
        )

    return JSONResponse(
        content=token_data,
        headers={"Cache-Control": "no-store"},
    )


@oauth_as_router.get(
    "/oauth/jwks",
    dependencies=[Depends(_require_oauth_as_enabled)],
)
async def oauth_jwks():
    """JSON Web Key Set (JWKS) endpoint.

    Returns the public keys used to verify tokens issued by this AS.
    Clients and resource servers use this to validate token signatures.

    Returns:
        JSONResponse with JWKS document, cached for 5 minutes
    """
    # First-Party
    from mcpgateway.services.oauth_as_service import get_oauth_as_service

    service = get_oauth_as_service()
    jwks = service.get_jwks()

    return JSONResponse(
        content=jwks,
        headers={"Cache-Control": "max-age=300, no-transform"},
    )


@oauth_as_router.post(
    "/oauth/register",
    response_model=DCRRegistrationResponse,
    status_code=201,
    responses={
        400: {"model": OAuthErrorResponse},
        401: {"model": OAuthErrorResponse},
    },
    dependencies=[Depends(_require_oauth_as_enabled)],
)
async def dcr_register(
    body: DCRRegistrationRequest,
    request: Request,
    db: Session = Depends(_get_db),
):
    """RFC 7591 Dynamic Client Registration endpoint.

    Allows MCP clients to self-register as OAuth clients without admin
    intervention. Behaviour is controlled by ``OAUTH_DCR_MODE``:

    - ``disabled`` — returns 404 (blocked by ``_require_dcr_enabled`` check).
    - ``open`` — no authentication required; anyone can register.
    - ``authenticated`` — a valid Bearer token is required (default).

    The server auto-generates ``client_id``. Clients may not supply their own.
    Scopes are restricted to the DCR default scopes; admin scope is never
    granted via this endpoint.

    Args:
        body: RFC 7591 client metadata request.
        request: FastAPI request (used to extract Bearer token in auth mode).
        db: Database session.

    Returns:
        RFC 7591 ClientInformation including ``client_id``, ``client_secret``,
        ``client_id_issued_at``, and ``client_secret_expires_at``.

    Raises:
        HTTPException 404: DCR mode is ``disabled``.
        HTTPException 401: Auth mode and no valid Bearer token supplied.
        HTTPException 400: Invalid ``client_name`` or unsupported auth method.
    """
    # First-Party
    from mcpgateway.services.oauth_as_service import get_oauth_as_service  # pylint: disable=import-outside-toplevel

    # DCR mode gate
    dcr_mode = settings.oauth_dcr_mode
    if dcr_mode == "disabled":
        raise HTTPException(status_code=404, detail="Not found")

    # Authenticated mode: require a valid Bearer token (any active client or user)
    if dcr_mode == "authenticated":
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "invalid_token", "error_description": "Bearer token required for client registration"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Validate token via existing CF JWT verification
        token = auth_header[7:]
        try:
            from mcpgateway.utils.verify_credentials import verify_jwt_token  # pylint: disable=import-outside-toplevel

            await verify_jwt_token(token)
        except Exception:  # pylint: disable=broad-except
            return JSONResponse(
                status_code=401,
                content={"error": "invalid_token", "error_description": "Invalid or expired Bearer token"},
                headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""},
            )

    service = get_oauth_as_service()
    try:
        result = await service.register_dcr_client(
            client_name=body.client_name,
            grant_types=body.grant_types,
            token_endpoint_auth_method=body.token_endpoint_auth_method,
            requested_scope=body.scope,
            db=db,
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_client_metadata", "error_description": str(exc)},
        )

    logger.info(
        "DCR registration successful: client_id=%s mode=%s",
        result["client_id"],
        dcr_mode,
    )

    return JSONResponse(content=result, status_code=201)


# NOTE: /.well-known/oauth-authorization-server (RFC 8414) is defined in
# well_known.py to ensure correct route ordering — the catch-all
# /.well-known/{filename:path} would shadow it if registered here.


# ---------------------------------------------------------------------------
# Admin endpoints (require authentication + admin)
# ---------------------------------------------------------------------------


@oauth_as_router.post(
    "/admin/oauth-as/clients",
    response_model=ClientRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_require_oauth_as_enabled)],
)
async def register_client(
    body: ClientRegistrationRequest,
    current_user_ctx: dict = Depends(get_current_user_with_permissions),
    db: Session = Depends(_get_db),
):
    """Register a new OAuth client.

    Admin-only endpoint. Returns the client info with the raw secret (shown once).

    Args:
        body: Client registration details
        current_user_ctx: Authenticated admin user context
        db: Database session

    Returns:
        Client info including the raw secret (one-time display)

    Raises:
        HTTPException: If not admin or client_id already exists
    """
    _require_admin(current_user_ctx)

    # First-Party
    from mcpgateway.services.oauth_as_service import get_oauth_as_service

    service = get_oauth_as_service()
    result = await service.register_client(
        client_id=body.client_id,
        client_name=body.client_name,
        scopes=body.scopes,
        teams=body.teams,
        is_admin=body.is_admin,
        db=db,
    )

    if result is None:
        raise HTTPException(status_code=409, detail=f"Client '{body.client_id}' already exists")

    return result


@oauth_as_router.get(
    "/admin/oauth-as/clients",
    response_model=List[ClientResponse],
    dependencies=[Depends(_require_oauth_as_enabled)],
)
async def list_clients(
    current_user_ctx: dict = Depends(get_current_user_with_permissions),
    db: Session = Depends(_get_db),
):
    """List all registered OAuth clients (secrets masked).

    Admin-only endpoint.

    Args:
        current_user_ctx: Authenticated admin user context
        db: Database session

    Returns:
        List of client records with secrets masked
    """
    _require_admin(current_user_ctx)

    # First-Party
    from mcpgateway.services.oauth_as_service import get_oauth_as_service

    service = get_oauth_as_service()
    return await service.list_clients(db)


@oauth_as_router.get(
    "/admin/oauth-as/clients/{client_id}",
    response_model=ClientResponse,
    dependencies=[Depends(_require_oauth_as_enabled)],
)
async def get_client(
    client_id: str,
    current_user_ctx: dict = Depends(get_current_user_with_permissions),
    db: Session = Depends(_get_db),
):
    """Get a single OAuth client by ID (secret masked).

    Admin-only endpoint.

    Args:
        client_id: The client identifier to look up
        current_user_ctx: Authenticated admin user context
        db: Database session

    Returns:
        Client record with secret masked

    Raises:
        HTTPException: If not admin or client not found
    """
    _require_admin(current_user_ctx)

    # First-Party
    from mcpgateway.services.oauth_as_service import get_oauth_as_service

    service = get_oauth_as_service()
    result = await service.get_client(client_id, db)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found")
    return result


@oauth_as_router.delete(
    "/admin/oauth-as/clients/{client_id}",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(_require_oauth_as_enabled)],
)
async def deactivate_client(
    client_id: str,
    current_user_ctx: dict = Depends(get_current_user_with_permissions),
    db: Session = Depends(_get_db),
):
    """Deactivate (soft delete) an OAuth client.

    Admin-only endpoint. The client record is retained but marked inactive.

    Args:
        client_id: The client identifier to deactivate
        current_user_ctx: Authenticated admin user context
        db: Database session

    Returns:
        Confirmation message

    Raises:
        HTTPException: If not admin or client not found
    """
    _require_admin(current_user_ctx)

    # First-Party
    from mcpgateway.services.oauth_as_service import get_oauth_as_service

    service = get_oauth_as_service()
    success = await service.deactivate_client(client_id, db)
    if not success:
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found")
    return {"detail": f"Client '{client_id}' deactivated"}


@oauth_as_router.post(
    "/admin/oauth-as/clients/{client_id}/rotate-secret",
    response_model=SecretRotationResponse,
    dependencies=[Depends(_require_oauth_as_enabled)],
)
async def rotate_client_secret(
    client_id: str,
    current_user_ctx: dict = Depends(get_current_user_with_permissions),
    db: Session = Depends(_get_db),
):
    """Rotate the secret for an OAuth client.

    Admin-only endpoint. Returns the new secret (shown once).

    Args:
        client_id: The client identifier whose secret to rotate
        current_user_ctx: Authenticated admin user context
        db: Database session

    Returns:
        New client secret (one-time display)

    Raises:
        HTTPException: If not admin or client not found
    """
    _require_admin(current_user_ctx)

    # First-Party
    from mcpgateway.services.oauth_as_service import get_oauth_as_service

    service = get_oauth_as_service()
    result = await service.rotate_client_secret(client_id, db)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found")
    return result


@oauth_as_router.post(
    "/admin/oauth-as/tokens/revoke",
    response_model=TokenRevokeResponse,
    dependencies=[Depends(_require_oauth_as_enabled)],
)
async def revoke_token(
    body: TokenRevokeRequest,
    current_user_ctx: dict = Depends(get_current_user_with_permissions),
    db: Session = Depends(_get_db),
):
    """Revoke a specific token by JTI.

    Admin-only endpoint.

    Args:
        body: Token revocation request containing the JTI
        current_user_ctx: Authenticated admin user context
        db: Database session

    Returns:
        Confirmation of revocation

    Raises:
        HTTPException: If not admin
    """
    _require_admin(current_user_ctx)

    # First-Party
    from mcpgateway.services.oauth_as_service import get_oauth_as_service

    service = get_oauth_as_service()
    success = await service.revoke_token(body.jti, db)
    return {"revoked": success, "jti": body.jti}
