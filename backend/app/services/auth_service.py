"""Single login gate (Section 7) — no role system, no RBAC, per Section 2's
explicit exclusion of admin/clinician/researcher roles. Just enough auth to
demo a real user flow: password hashing + a signed, expiring bearer token."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from backend.app.config import settings

_PBKDF2_ITERATIONS = 100_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()
    return f"{salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    salt, digest = password_hash.split("$")
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()
    return hmac.compare_digest(candidate, digest)


def create_access_token(user_id: str, expires_in_seconds: int = 60 * 60 * 24) -> str:
    expiry = int(time.time()) + expires_in_seconds
    payload = f"{user_id}:{expiry}"
    signature = hmac.new(settings.secret_key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"


def verify_access_token(token: str) -> str | None:
    try:
        user_id, expiry, signature = token.split(":")
    except ValueError:
        return None
    payload = f"{user_id}:{expiry}"
    expected = hmac.new(settings.secret_key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return None
    if int(expiry) < time.time():
        return None
    return user_id
