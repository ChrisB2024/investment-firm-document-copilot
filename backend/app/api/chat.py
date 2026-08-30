"""Chat routes. Thin: verify, delegate, translate errors.

Everything real happens in `app/chat/orchestrator.py` and `app/database/chats.py`.
What lives here is the HTTP contract and nothing else.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["chat"])

# TODO: implement.
#   GET  /chat/threads              list this user's threads
#   POST /chat/threads              create one
#   GET  /chat/threads/{id}         thread + messages, 403 if not theirs
#   POST /chat/stream               run a turn, stream events
#
# Every one takes `user: Annotated[CurrentUser, Depends(get_current_user)]` and
# passes `user.id` down. No route reads a user id from a body or a query string.
#
# TODO: 403 or 404 for another analyst's thread? architecture.md says 403, and
#  403 leaks existence — an attacker learns which thread ids are real. 404 leaks
#  nothing and is what a resource they cannot see should look like. The counter
#  is that this is a 40-analyst internal tool where thread ids are UUIDs nobody
#  can enumerate, and 403 is the more honest answer to a bug. Follow
#  architecture.md, and write the test asserting it so the choice is deliberate
#  rather than whatever the code happened to do.
#
# TODO: `/chat/stream` returns `StreamingResponse` with
#  `media_type="text/event-stream"`. Two headers matter behind Railway's proxy:
#  `Cache-Control: no-cache` and `X-Accel-Buffering: no`, or the proxy buffers
#  the whole stream and delivers it at once — which looks exactly like the
#  backend being slow.
