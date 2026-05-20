"""
AuthN / AuthZ patterns.

Covers:
  1. JWT verification (HS256 for internal services, RS256 for public APIs)
  2. FastAPI dependency that extracts the current user from the Bearer token
  3. OAuth 2.1 PKCE flow — the code_verifier/code_challenge handshake
  4. mTLS notes — enforced at the infrastructure layer, not application code

JWT vs mTLS:
  JWT     = identity travels in the HTTP Authorization header (service-to-user)
  mTLS    = identity encoded in the client TLS certificate (service-to-service)
  Use JWT for user-facing APIs; mTLS for internal service mesh communication.
"""

import base64
import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import structlog
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .settings import settings

log = structlog.get_logger()
bearer = HTTPBearer()


# ── JWT ───────────────────────────────────────────────────────────────────────


def create_jwt(subject: str, extra_claims: dict[str, Any] | None = None) -> str:
    """Create a signed JWT. `subject` is typically a user ID."""
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
        **(extra_claims or {}),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def verify_jwt(token: str) -> dict[str, Any]:
    """
    Decode and verify a JWT. Raises HTTPException on any failure.

    jwt.decode() checks:
      - Signature (tamper detection)
      - `exp` claim (expiry)
      - `iat` claim (issued-at, if leeway is set)
    """
    try:
        return jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp", "iat"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as exc:
        log.warning("invalid_jwt", error=str(exc))
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> dict[str, Any]:
    """
    FastAPI dependency — inject into any route that requires authentication.

        @router.get("/me")
        async def me(user: dict = Depends(get_current_user)):
            return {"sub": user["sub"]}
    """
    return verify_jwt(credentials.credentials)


# ── OAuth 2.1 PKCE ────────────────────────────────────────────────────────────
# PKCE (Proof Key for Code Exchange) prevents authorization code interception.
# The client generates a random `code_verifier`, hashes it to `code_challenge`,
# sends the challenge to the auth server, then proves ownership by sending the
# original verifier when exchanging the code for tokens.
#
# OAuth 2.1 mandates PKCE for ALL clients (not just public clients as in 2.0).


def generate_pkce_pair() -> tuple[str, str]:
    """
    Returns (code_verifier, code_challenge).
    Call this on the client before starting the auth flow.
    """
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return code_verifier, code_challenge


def verify_pkce(code_verifier: str, code_challenge: str) -> bool:
    """
    Called on the auth server when the client exchanges the code for tokens.
    Verifies that the verifier matches the challenge stored at code issuance.
    """
    digest = hashlib.sha256(code_verifier.encode()).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    # compare_digest prevents timing attacks
    return secrets.compare_digest(expected, code_challenge)


# ── mTLS (informational) ──────────────────────────────────────────────────────
# mTLS is enforced at the TLS layer — your app code does NOT verify certificates.
# Configure it at:
#   - Nginx:   ssl_verify_client on; ssl_client_certificate /etc/ssl/ca.crt;
#   - Istio:   PeerAuthentication policy with mtls mode STRICT
#   - Granian: --tls-client-auth require --tls-ca /path/to/ca.pem
#
# The app can inspect the forwarded client cert via the X-Client-Cert header
# (set by the proxy) to extract the service identity (CN or SAN).
