"""Server-sent events for one turn.

**This module has no target yet, and writing it now would be guesswork.**
architecture.md says the browser consumes the AI SDK's UI message format, and
`frontend/package.json` has no AI SDK dependency — so there is no installed
version to write against. The v4 and v5 data-stream protocols differ enough that
a wrong guess is a rewrite, not an adjustment.

Two orders, and the second is cheaper:

  1. Pin the AI SDK in the frontend now, read its protocol, implement this
     against it. Costs a frontend decision before any of Phase 5 works.
  2. Emit a small SSE contract of our own — the four event types below — get the
     turn working end to end with `curl`, and adapt to the AI SDK's shape in
     Phase 6 when `useChat` is actually wired up. The adapter is this file.

Taking (2). What the backend has to get right is *which* events exist and when
they fire; their JSON envelope is a translation, and translating once against a
real client beats translating twice against a guess.

The events, whatever they end up named on the wire:

  progress  — a tool call started or returned. What the analyst watches while
              retrieval runs. Never contains answer prose.
  answer    — the whole grounded answer, once, after the gate passed. See
              orchestrator.py for why this is not streamed token by token.
  citations — the resolved passages, so the UI can render a marker as a link
              without a second request.
  error     — a typed failure: unauthorised, thread not found, grounding
              failed, upstream unavailable. Distinct from a dropped connection,
              which the client sees as a closed stream and must not treat as an
              answer.
"""

from __future__ import annotations

# TODO: implement once the order above is settled. Whatever the envelope, two
#  things are not negotiable:
#
#  - `answer` never fires before grounding passed. A partial answer on the wire
#    is an unsourced claim in front of an analyst.
#  - `error` is a distinguishable event, not a closed stream. A stream that ends
#    silently after three progress events is indistinguishable from a network
#    drop, and the UI would show a spinner forever or, worse, an empty answer.
