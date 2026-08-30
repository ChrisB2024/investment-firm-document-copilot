"""Who is asking, verified against Supabase Auth.

The backend never trusts a user id from the request body. The only credential is
the `Authorization: Bearer <token>` header, and the only thing that turns it into
a user is Supabase.

Verification calls Supabase's `/auth/v1/user` rather than validating the JWT
locally. Slower — one round trip per request — and correct by construction:
local validation means holding the signing keys, tracking key rotation, and
getting `aud`, `exp` and algorithm confusion right, which is a category of
mistake that fails open. Swap it for local verification when request volume
makes the round trip hurt, behind this same function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# `auto_error=False` so a missing header reaches our own 401 with a message,
# rather than FastAPI's bare "Not authenticated".
_bearer = HTTPBearer(auto_error=False)

# `Annotated[T, Depends(...)]` rather than a `Depends()` default throughout this
# service: ruff rejects a call in an argument default (B008), and the rule is
# worth keeping for the mutable cases. Every route below follows the same shape.
Bearer = Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]


@dataclass(frozen=True)
class CurrentUser:
    """The authenticated analyst. `id` is Supabase's `auth.users.id`."""

    id: UUID
    email: str


async def get_current_user(credentials: Bearer) -> CurrentUser:
    """Verify the bearer token, or 401.

    TODO: implement.
      - No credentials, or a scheme that is not Bearer -> 401. Do not fall back
        to a query parameter or a cookie; one credential channel is the point.
      - GET {settings.supabase_url}/auth/v1/user with `Authorization: Bearer
        <token>` and `apikey: <anon key>`. Supabase requires both.
      - 200 -> CurrentUser(id=UUID(body["id"]), email=body["email"]).
        401/403 from Supabase -> 401 here, with no detail from the upstream body:
        "token expired" and "token forged" are the same answer to the caller.
        Anything else -> 502, because that is our dependency failing, not their
        credential.
      - One shared httpx.AsyncClient at module scope, not one per request. A new
        client per request means a new TLS handshake per request.

    TODO: the app-side `users` row. `chat_threads.user_id` is a foreign key to
     `public.users`, which is a foreign key to `auth.users` — so a analyst who
     has signed up in Supabase but has no `public.users` row cannot create a
     thread, and the failure surfaces as an integrity error on their first
     message rather than at sign-in. Decide where the upsert goes: here (every
     request pays a write), in the thread-creation path (one place, but any
     other user-scoped write inherits the same problem), or a Supabase trigger
     on `auth.users` (no application code, but schema that Alembic does not own).
     Leaning to the thread-creation path — it is the only place that needs it
     today, and ../CLAUDE.md says extract at the third caller.
    """
    raise NotImplementedError


def _unauthorised(detail: str) -> HTTPException:
    """401 with the header a client needs to know how to retry."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )
