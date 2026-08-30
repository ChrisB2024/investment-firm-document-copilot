"""Wire format in, internal models out.

Blocked on the same decision as streaming.py: there is no AI SDK installed in
the frontend, so the shape this converts *from* is not fixed. See that module.

What is already decided and does not depend on the SDK: the backend validates
the wire payload at the boundary and never passes it inward untranslated. A
`thread_id` from the body is a claim, not a fact — ownership comes from the
verified token — and the message list from the browser is the client's view of
history, not the server's.

TODO: the history question, which is a product decision rather than a format
 one. `useChat` posts the whole message list every turn. The backend can trust
 it (cheap, and the client can rewrite history), or ignore it and load the
 thread from Postgres (one query, and the server's record is authoritative).
 Load from Postgres: the persisted thread is what the citations were validated
 against, and a client-supplied history could cite handles from a turn that
 never happened.
"""

from __future__ import annotations
