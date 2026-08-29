"""Scratch CLI for reading what the agent actually answers.

    uv run python -m app.assistant.cli "how did Apple describe supplier risk"
    uv run python -m app.assistant.cli --brief
    uv run python -m app.assistant.cli --brief --quotes

`app.retrieval.cli` answers "did the right passages come back". This one answers
the two questions Phase 4's exit criterion asks, and neither is a unit test:

  - does question 10 refuse, rather than assert a causal claim the filings do
    not make?
  - do questions 1-9 produce answers whose citations all validate?

It also measures the number left open in grounding/validator.py. `_supports`
has no minimum quote length because every floor that closes that hole also
rejects a legitimate short quote — "$96.2 billion" is two words — and choosing
one needs the distribution the model actually emits rather than an argument.
`--quotes` prints it.

Costs real OpenAI calls. A full `--brief` run is ten agent turns, each with at
least one embedding and one retrieval.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import textwrap
from collections import Counter
from statistics import median
from uuid import uuid4

from pydantic_ai import UnexpectedModelBehavior, capture_run_messages
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import UsageLimits

from app.assistant.agent import agent
from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import GroundedAnswer, InsufficientEvidence
from app.config import settings
from app.database.session import dispose, session
from app.grounding.validator import GroundingError, validate

# The brief loader lives in the retrieval CLI and is imported rather than
# copied: two regexes over the same document would let the questions this tool
# runs drift from the ones Phase 3 was judged on, silently. Extract it to a
# shared module at the third caller, not this one.
from app.retrieval.cli import BRIEF_QUESTION_COUNT, load_brief_questions

WIDTH = 78
BODY_INDENT = " " * 3

# How much of a quote to echo per citation. Long enough to recognise which
# sentence it came from, short enough that six citations stay scannable.
QUOTE_PREVIEW = 64

# pydantic-ai delivers structured output through a synthetic tool call named
# `final_result...`. It is not a tool in the sense this trace is about, and
# printing it dumps the whole answer and every citation into the trace — the
# answer is rendered properly below. Skipped by prefix, which is pydantic-ai's
# naming convention rather than a guess: with a union output_type the name
# carries the chosen member, e.g. `final_result_GroundedAnswer`.
_OUTPUT_TOOL_PREFIX = "final_result"

# Tool arguments are a diagnostic, not a transcript. A long one means a
# malformed call, and 90 characters is enough to see that.
_ARG_CHARS = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.assistant.cli",
        description="Run the grounded agent and read what it answers.",
        epilog=(
            "examples:\n"
            '  %(prog)s "how did Apple describe supplier risk"\n'
            "  %(prog)s --brief --quotes\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("question", nargs="?", help="a single question to answer")
    target.add_argument(
        "--brief",
        action="store_true",
        help=f"run all {BRIEF_QUESTION_COUNT} questions from the client brief",
    )
    parser.add_argument(
        "--quotes",
        action="store_true",
        help="report the distribution of quote lengths, for the _supports floor",
    )
    parser.add_argument(
        "--passages",
        action="store_true",
        help="print each cited passage in full, not just its quote",
    )
    return parser.parse_args()


def _wrap(text: str, indent: str = BODY_INDENT) -> str:
    return textwrap.fill(
        text, width=WIDTH, initial_indent=indent, subsequent_indent=indent
    )


def tool_trace(messages: list[ModelMessage]) -> list[str]:
    """One line per tool call, in order, with what came back.

    The trace is the diagnosis. An answer that refuses is right or wrong
    depending entirely on what it searched first — a refusal after one narrow
    search is a retrieval failure wearing the costume of an honest one, and the
    only way to tell is to see the calls. Retries show here too, which is how a
    rejected ticker or a failed grounding check becomes visible rather than
    just slow.
    """
    outcomes: dict[str, str] = {}
    for message in messages:
        for part in getattr(message, "parts", []):
            kind = getattr(part, "part_kind", "")
            if kind == "tool-return":
                content = part.model_response_str()
                outcomes[part.tool_call_id] = f"{len(content):,} chars"
            elif kind == "retry-prompt":
                # Flagged rather than counted: a retry is the interesting event
                # in a run, and it is invisible in the answer.
                outcomes[part.tool_call_id] = f"RETRY {part.model_response()[:70]}"

    lines: list[str] = []
    for message in messages:
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") != "tool-call":
                continue
            if part.tool_name.startswith(_OUTPUT_TOOL_PREFIX):
                continue
            args = part.args_as_dict()
            shown = "  ".join(
                f"{key}={str(value)[:_ARG_CHARS]}"
                for key, value in args.items()
                # The question is echoed in full above; repeating it per call
                # would push the argument that varies off the line.
                if key != "question" and value
            )
            outcome = outcomes.get(part.tool_call_id, "(no return recorded)")
            lines.append(f"{part.tool_name}({shown}) -> {outcome}")
    return lines


def report(
    output: GroundedAnswer | InsufficientEvidence,
    deps: DocumentAgentDeps,
    *,
    passages: bool,
) -> list[int]:
    """Print one answer. Returns the length of every quote it cited.

    The lengths are the point of `--quotes`: `_supports` needs a floor under
    the model's shortest legitimate quote, and this is the only place that
    number can come from.
    """
    if isinstance(output, InsufficientEvidence):
        print("\nREFUSED (InsufficientEvidence)")
        print(_wrap(output.reason))
        for query in output.searched:
            print(f"{BODY_INDENT}searched: {query}")
        return []

    print("\nANSWERED (GroundedAnswer)")
    print(_wrap(output.answer))
    if output.limitations:
        print(f"\n{BODY_INDENT}limitations:")
        print(_wrap(output.limitations, BODY_INDENT * 2))

    # Re-run the gate the agent already passed. It cannot fail here — a failure
    # would mean `enforce_grounding` admitted something `validate` rejects — and
    # running it anyway is what makes the claim in that function's docstring
    # checkable rather than asserted: the passages shown below are resolved by
    # the same function that admitted the answer.
    validated = validate(output, deps.ledger)

    print(f"\nCITATIONS {len(output.citations)}, all resolved and verbatim")
    by_handle = {p.handle: p for p in validated.cited_passages}
    lengths: list[int] = []
    for citation in output.citations:
        source = by_handle[citation.handle]
        quote = " ".join(citation.quote.split())
        lengths.append(len(citation.quote))
        head = (
            f"{BODY_INDENT}[{citation.handle}] {source.ticker} FY{source.fiscal_year} "
            f"{source.source_type:<5} {len(citation.quote):>4} chars"
        )
        print(f'{head}  "{quote[:QUOTE_PREVIEW]}"')
        if passages:
            print(_wrap(source.title or "", BODY_INDENT * 2))
            print(_wrap(source.text, BODY_INDENT * 2))

    # Offered but never cited. Not a violation — the model is allowed to read a
    # passage and judge it irrelevant — but a turn that cited 2 of 25 either
    # answered narrowly or retrieved badly, and the ratio says which to check.
    print(
        f"{BODY_INDENT}({len(by_handle)} of {len(deps.ledger)} offered passages cited)"
    )
    return lengths


async def run_one(
    connection, number: int | None, question: str, args: argparse.Namespace
) -> tuple[bool, list[int], list[str]]:
    """One agent turn. Returns (succeeded, quote lengths, tool trace).

    A fresh `DocumentAgentDeps` per question, which is the contract: handles are
    minted per turn, so reusing one across questions would let question 5 cite a
    passage retrieved for question 2 and the ledger would happily resolve it.

    `user_id` and `thread_id` are invented. Nothing in Phase 4 writes a row, and
    the agent only carries them; Phase 5's orchestrator supplies real ones.
    """
    label = f"Q{number}" if number is not None else "Q"
    heading = f"── {label} "
    print(f"\n{heading}{'─' * max(WIDTH - len(heading), 0)}")
    print(textwrap.fill(question, width=WIDTH))

    deps = DocumentAgentDeps(
        session=connection, user_id=uuid4(), thread_id=uuid4()
    )

    # `capture_run_messages` rather than `result.all_messages()`, because the
    # run that matters most is the one that raises: an exhausted grounding
    # budget leaves no result to read the trace off, and the trace is exactly
    # what says whether the model was failing to copy a quote or citing a
    # passage it never retrieved.
    with capture_run_messages() as messages:
        try:
            result = await agent.run(
                question,
                deps=deps,
                usage_limits=UsageLimits(
                    request_limit=settings.openai_agent_request_limit
                ),
            )
        except UnexpectedModelBehavior as failure:
            trace = tool_trace(messages)
            print("\ntools")
            for line in trace:
                print(f"{BODY_INDENT}{line}")
            # The controlled failure, printed as one. This is the outcome the
            # product promises instead of a polished unsourced answer, so it is
            # a result to read rather than a crash to debug.
            print(f"\nFAILED (controlled): {failure}")
            return False, [], trace

    trace = tool_trace(result.all_messages())
    print("\ntools")
    for line in trace:
        print(f"{BODY_INDENT}{line}")

    try:
        lengths = report(result.output, deps, passages=args.passages)
    except GroundingError as leak:
        # Unreachable unless enforce_grounding and validate disagree, which
        # would be a real bug in the gate rather than in the answer. Loud on
        # purpose: silently reporting it as a bad answer would hide it.
        print(f"\nGATE LEAK — validate() rejects an answer the agent admitted:\n{leak}")
        return False, [], trace

    usage = result.usage
    print(
        f"\nusage {usage.requests} requests  "
        f"{usage.input_tokens:,} in / {usage.output_tokens:,} out"
    )
    return True, lengths, trace


async def _run(args: argparse.Namespace) -> int:
    """Run every requested question. Exit code is the number that failed."""
    questions = load_brief_questions() if args.brief else [(None, args.question)]

    failed: list[str] = []
    quotes: list[int] = []
    tools: Counter[str] = Counter()
    try:
        async with session() as connection:
            for number, question in questions:
                ok, lengths, trace = await run_one(
                    connection, number, question, args
                )
                if not ok:
                    failed.append(f"Q{number}" if number is not None else "question")
                quotes.extend(lengths)
                tools.update(line.split("(", 1)[0] for line in trace)
    finally:
        await dispose()

    if args.brief or args.quotes:
        print(f"\n{'═' * WIDTH}")
        print(f"{len(questions) - len(failed)} of {len(questions)} completed")
        for name, count in sorted(tools.items()):
            print(f"  {name}: {count}")
    if args.quotes and quotes:
        # The distribution `_supports` needs. The minimum is the number that
        # matters: any floor above it rejects a quote this model actually wrote.
        quotes.sort()
        print(
            f"  quote chars: min {quotes[0]}  p50 {int(median(quotes))}  "
            f"max {quotes[-1]}  (n={len(quotes)})"
        )
        print(f"  shortest ten: {quotes[:10]}")

    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}", file=sys.stderr)
    return len(failed)


def main() -> None:
    raise SystemExit(asyncio.run(_run(parse_args())))


if __name__ == "__main__":
    main()
