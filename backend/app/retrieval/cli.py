"""Scratch CLI for eyeballing retrieval quality.

    uv run python -m app.retrieval.cli "how did Apple describe supplier risk"
    uv run python -m app.retrieval.cli --brief
    uv run python -m app.retrieval.cli "..." --ticker AAPL --ticker NVDA --per-ticker 3

This exists to answer one question before any LLM is involved: for each of the
ten questions in the client brief, do the passages needed to answer it come
back in the top ~10? That is Phase 3's exit criterion, and it is a judgement
call a human makes by reading — so this tool's job is to make the reading fast
and to make a failure *diagnosable*, not merely visible.

`docs/todos.md` asks which questions fail and why, and names three causes worth
telling apart. Each has a different fix and a different signature in the output:

  the passage is not in the corpus  — no chunk anywhere contains the answer.
                                      An ingestion problem. Nothing retrieval
                                      does will help.
  the passage is chunked badly      — the answer exists but is split across a
                                      boundary, so no single chunk carries
                                      enough of it to rank. A chunking problem.
  one arm found it, the other did   — visible in `contributions`. A passage
  not, and fusion buried it           with {'text': 2} and no vector entry was
                                      never seen by half the system. A ranking
                                      or weighting problem, and the evidence
                                      that would justify weighting the arms.

Lives in `app/retrieval/` rather than a scripts directory because it is a lens
on this package specifically, and it should break loudly when the package's
signatures change.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

from app.database.session import dispose, session
from app.retrieval.queries import Filters
from app.retrieval.retriever import RetrievedPassage, retrieve, retrieve_per_ticker

BRIEF = Path(__file__).resolve().parents[3] / "docs" / "client-brief.md"

# The brief's ten questions are numbered list items and nothing else in the file
# is, so this pattern finds exactly them. Read from the document rather than
# copied into this file: a divergence between the questions the product is
# judged on and the questions actually run would be invisible.
_NUMBERED = re.compile(r"^(\d+)\.\s+(.+)$", re.MULTILINE)

# The brief is the exit criterion, so its length is part of the contract. Named
# rather than inlined so the check below reads as "the brief still has its ten
# questions" instead of a bare magic number.
BRIEF_QUESTION_COUNT = 10

# Enough of the passage to judge relevance without drowning the terminal. The
# section heading is the first line of every chunk, so a snippet that stops too
# early shows only the heading and tells you nothing.
SNIPPET_CHARS = 220

# Long enough for "Item 7. Management's Discussion and Analysis of Financial
# Condition", which is where the header line would otherwise wrap.
SECTION_CHARS = 62

# The arms `retrieve` fuses, in display order. Both are printed even when one
# missed the passage, because the gap is the whole diagnostic: a hit showing
# `text=2 vector=-` was never seen by half the system, and that is a different
# problem from a hit that neither arm found.
ARMS = ("vector", "text")

# Body lines are indented under their header so the rank column stays scannable
# down the left edge of a ten-question run.
BODY_INDENT = " " * 5

# Wide enough for a header line plus a long section name, narrow enough to read
# in a split terminal. The brief's questions run past 250 characters, so they
# are wrapped rather than left to the terminal.
WIDTH = 78


def load_brief_questions() -> list[tuple[int, str]]:
    """The ten analyst questions, as (number, text).

    Raises if the brief no longer yields exactly ten, numbered 1..10. If an edit
    silently drops this to eight, the exit criterion has moved and every run
    afterwards passes a smaller test while looking identical.

    Checking the numbers, not just the count, because the likely edit is a
    reformat: a markdown list written as ten `1.` items still counts ten while
    labelling every question "1", and the report would name the wrong question
    for nine of them.
    """
    found = _NUMBERED.findall(BRIEF.read_text(encoding="utf-8"))
    questions = [(int(number), text.strip()) for number, text in found]

    numbers = [number for number, _ in questions]
    if numbers != list(range(1, BRIEF_QUESTION_COUNT + 1)):
        raise ValueError(
            f"{BRIEF} yielded {numbers or 'no'} numbered questions, expected "
            f"1..{BRIEF_QUESTION_COUNT}. The brief is the exit criterion — fix "
            f"the document or move the criterion deliberately, not by accident."
        )
    return questions


def parse_args() -> argparse.Namespace:
    """Flags in, a namespace `run_one` can dispatch on directly.

    A question and `--brief` are mutually exclusive and one is required, so
    argparse refuses an invocation with neither instead of printing an empty
    report that reads like a corpus with no matches.

    Normalises `--ticker` to a tuple here rather than in `run_one`, so both
    consumers get the type they want without a conversion at the call site:
    `Filters.tickers` is a tuple, and `retrieve_per_ticker` takes a sequence.
    """
    parser = argparse.ArgumentParser(
        # Without this, usage says "cli.py" — which is not how this is run, and
        # the whole value of a usage line is that it can be pasted back.
        prog="python -m app.retrieval.cli",
        description="Eyeball retrieval quality for the brief's questions.",
        epilog=(
            "examples:\n"
            '  %(prog)s "how did Apple describe supplier risk"\n'
            "  %(prog)s --brief\n"
            '  %(prog)s "risk factor language" --ticker AAPL --ticker NVDA '
            "--per-ticker 3\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # A positional may sit in a mutually exclusive group as long as it can be
    # absent, which is what nargs="?" buys. `required=True` then makes "neither"
    # an argparse error with a usage line, rather than something run_one has to
    # notice and report for itself.
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "question", nargs="?", help="a single question to retrieve for"
    )
    target.add_argument(
        "--brief",
        action="store_true",
        help=f"run all {BRIEF_QUESTION_COUNT} questions from the client brief",
    )

    parser.add_argument(
        "--ticker",
        # Repeatable rather than comma-separated: an analyst question names
        # companies one at a time, and this is the shape the fan-out consumes.
        action="append",
        dest="tickers",
        metavar="T",
        # Not default=[]: argparse appends into the default object itself, so a
        # shared list would accumulate across calls in the same process.
        default=None,
        help="restrict to this ticker; repeat for several",
    )
    parser.add_argument(
        "--year-from", type=int, metavar="Y", help="earliest fiscal year"
    )
    parser.add_argument("--year-to", type=int, metavar="Y", help="latest fiscal year")
    parser.add_argument(
        "--per-ticker",
        type=int,
        metavar="N",
        help="fan out: N passages per --ticker, via retrieve_per_ticker",
    )
    # Mirrors retrieve()'s own default, so a bare run shows what the API does
    # when a caller says nothing.
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="top-k (default: %(default)s)",
    )
    parser.add_argument(
        "--context", action="store_true", help="widen each hit with its neighbours"
    )
    parser.add_argument(
        "--full", action="store_true", help="print whole passages, not a snippet"
    )

    args = parser.parse_args()

    # Which flags may appear together is the CLI's own business, and this pair
    # is worth catching here: retrieve_per_ticker refuses an empty ticker list,
    # and a usage line explains the mistake better than that traceback would.
    if args.per_ticker is not None and not args.tickers:
        parser.error("--per-ticker needs at least one --ticker to fan out over")

    args.tickers = tuple(args.tickers or ())
    return args


def _clip(text: str, limit: int, *, count: bool = True) -> str:
    """Truncate to `limit`, saying how much was dropped.

    The dropped count is not decoration either: "the passage is chunked badly"
    is one of the three diagnoses this tool exists to separate, and a hit whose
    snippet ends `(+2,940 more)` is a chunk that swallowed most of a section.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit].rstrip()
    return f"{head}…(+{len(text) - limit:,} more)" if count else f"{head}…"


def format_passage(result: RetrievedPassage, *, full: bool = False) -> str:
    """One result, formatted for reading.

    The header line carries rank, score, both arms' ranks, ticker, fiscal year,
    form, source type and section — everything needed to judge a hit without
    reading the body, so a ten-question run can be skimmed and only the
    suspicious hits read.

    `contributions` is the load-bearing field. A passage the corpus does not
    contain and a passage one arm found at rank 2 that fusion then buried look
    identical in a ranked list and need opposite fixes; `vector=- text=2` is the
    only place that difference is visible.

    Tables are marked TABLE in caps against a lowercase `chunk`, which is enough
    to see without colour. Their body is markdown and keeps its line breaks; a
    chunk's prose is collapsed to one paragraph, because a snippet that spends
    its 220 characters on newlines shows half as much text.
    """
    p = result.passage
    is_table = p.source_type == "table"

    if is_table:
        # A table's caption is a real column, not something to dig out of the
        # markdown, and the markdown itself never contains it.
        section, body = p.title or "(untitled table)", p.text
    else:
        # ingest.chunk prefixes every chunk with `heading + "\n\n"`, so the
        # section is the first paragraph and comes back out by splitting on it.
        # It is dropped from the body because the header line already shows it —
        # 220 characters is too few to spend repeating what is directly above.
        section, _, body = p.text.partition("\n\n")

    kind = "TABLE" if is_table else "chunk"
    if result.context is not None:
        # The widened window is what --context was asked for, so it replaces the
        # body rather than being appended to it — it already contains this
        # passage. Its heading is dropped for the same reason a chunk's is: the
        # header line above already carries it. Flagged in the header so a long
        # body is not mistaken for one oversized chunk, which is a thing this
        # tool is used to spot.
        _, _, body = result.context.partition("\n\n")
        kind = f"{kind}+ctx"

    arms = " ".join(f"{arm}={result.contributions.get(arm, '-')!s:>2}" for arm in ARMS)

    header = (
        f"{result.rank:>3}. {result.score:.5f}  {arms}  "
        f"{p.ticker:<5} FY{p.fiscal_year} {p.form:<5}  {kind:<9} "
        f"{_clip(section, SECTION_CHARS, count=False)}"
    )

    if full:
        shown = body.strip()
    elif is_table:
        shown = _clip(body, SNIPPET_CHARS)
    else:
        shown = _clip(" ".join(body.split()), SNIPPET_CHARS)

    # A section short enough to fit in one chunk leaves nothing after the
    # heading. Saying so beats a blank line that reads like a formatting bug.
    lines = shown.splitlines() or ["(heading only — this section is one chunk)"]
    return "\n".join([header, *(f"{BODY_INDENT}{line}" for line in lines)])


def coverage(results: list[RetrievedPassage]) -> str:
    """One line: which companies and years are represented.

    This is the check a passage-by-passage read cannot make. Question 6 names
    five companies and `retrieve` returns two; every individual passage is
    on-topic and well-ranked, and the question is still unanswerable. That
    failure has no per-passage symptom — only a shape.

    Companies are ordered by count so the shape is the first thing read:
    `NVDA×6 AAPL×4` against a question naming five is the finding, and it is
    obvious without counting anything.

    The year line reports distinct years inside the span, not just its ends.
    Nine of the ten brief questions ask how something changed over 2021-2025,
    and ten passages drawn from two of those five years look identical to good
    coverage when only the endpoints are printed.

    Tables are counted separately because 46% of a filing's figures appear only
    in one. A question about capital expenditure answered from prose alone is a
    likely miss however well each passage reads.
    """
    if not results:
        return "coverage: nothing retrieved"

    tickers = Counter(r.passage.ticker for r in results)
    # Ties broken on ticker so a re-run of the same question is diffable.
    by_company = " ".join(
        f"{ticker}×{n}"
        for ticker, n in sorted(tickers.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    years = sorted({r.passage.fiscal_year for r in results})
    span = f"FY{years[0]}" if len(years) == 1 else f"FY{years[0]}–{years[-1]}"
    reach = years[-1] - years[0] + 1
    if len(years) < reach:
        span += f" ({len(years)} of {reach} yrs)"

    # Noun first, so a count of 1 does not read as "1 chunks".
    tables = sum(1 for r in results if r.passage.source_type == "table")
    mix = f"chunks {len(results) - tables}  tables {tables}"

    plural = "" if len(results) == 1 else "s"
    return (
        f"coverage: {len(results)} passage{plural}  "
        f"{len(tickers)} cos  {by_company}  {span}  {mix}"
    )


def _call_line(args: argparse.Namespace) -> str:
    """What was actually run, as one line above the results.

    A report that does not say which code path produced it is unreadable a day
    later, and the two paths return deliberately different shapes — ten ranked
    by relevance, or N per company. Naming the function rather than describing
    it means this line stays honest when the dispatch below changes.
    """
    if args.per_ticker is not None:
        parts = [
            "retrieve_per_ticker",
            f"per_ticker={args.per_ticker}",
            f"tickers={','.join(args.tickers)}",
        ]
    else:
        parts = ["retrieve", f"limit={args.limit}"]
        if args.tickers:
            parts.append(f"tickers={','.join(args.tickers)}")

    if args.year_from is not None or args.year_to is not None:
        parts.append(f"FY{args.year_from or ''}–{args.year_to or ''}")
    if args.context:
        parts.append("+context")
    return "  ".join(parts)


async def run_one(
    session, number: int | None, question: str, args: argparse.Namespace
) -> int:
    """Retrieve for one question and print the report. Returns the hit count.

    Dispatches on `args.per_ticker`, and hands the two functions their tickers
    differently on purpose: `retrieve_per_ticker` refuses tickers passed through
    `filters`, because it overwrites that field per arm and would silently
    discard whatever the caller put there. Everything else on `Filters` — the
    year range — applies identically to both and is built once.

    Question, then passages, then coverage. The coverage line goes last because
    it is the finding: the per-passage read tells you whether each hit is good,
    and only the last line tells you whether the set can answer the question.
    """
    label = f"Q{number}" if number is not None else "Q"
    heading = f"── {label} "
    print(f"\n{heading}{'─' * max(WIDTH - len(heading), 0)}")
    print(textwrap.fill(question, width=WIDTH))
    print(f"\n{_call_line(args)}\n")

    years = {
        "fiscal_year_from": args.year_from,
        "fiscal_year_to": args.year_to,
    }
    if args.per_ticker is not None:
        results = await retrieve_per_ticker(
            session,
            question,
            args.tickers,
            per_ticker=args.per_ticker,
            filters=Filters(**years),
            with_context=args.context,
        )
    else:
        results = await retrieve(
            session,
            question,
            limit=args.limit,
            filters=Filters(tickers=args.tickers, **years),
            with_context=args.context,
        )

    for result in results:
        print(format_passage(result, full=args.full))
        # Blank line between passages: bodies are multi-line under --full and
        # --context, and without it the rank column stops being findable.
        print()

    print(coverage(results))
    return len(results)


async def _run(args: argparse.Namespace) -> int:
    """Run every requested question on one session. Exit code is the empty count.

    An empty result set is not the same as failing the exit criterion — that
    judgement needs a human reading the passages — but it is unambiguous, and
    it is the one failure a script can detect. Returning how many questions
    came back empty rather than a bare 1 means `echo $?` after a brief run says
    how bad it was, and the run is capped at ten so the count cannot collide
    with a shell's reserved codes.

    The brief is loaded before the session opens, so a malformed brief fails
    immediately instead of after connecting to Postgres.
    """
    questions = load_brief_questions() if args.brief else [(None, args.question)]

    empty: list[str] = []
    try:
        # One session for the whole run. Ten questions is ten round trips on a
        # connection that is already pooled upstream; building an engine per
        # question would spend more time connecting than retrieving.
        async with session() as active:
            for number, question in questions:
                if await run_one(active, number, question, args) == 0:
                    empty.append(f"Q{number}" if number is not None else "the question")
    finally:
        await dispose()

    if empty:
        # stderr, so redirecting the report to a file still surfaces this — and
        # nothing is lost from the file, where each question's own coverage line
        # already reads "nothing retrieved".
        print(f"\n{len(empty)} returned nothing: {', '.join(empty)}", file=sys.stderr)
    return len(empty)


def main() -> None:
    """Entry point. The exit code is how many questions returned nothing."""
    raise SystemExit(asyncio.run(_run(parse_args())))


if __name__ == "__main__":
    main()
