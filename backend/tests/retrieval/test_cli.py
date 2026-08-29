"""Tests for app.retrieval.cli.

All fast. The CLI's job is to make a retrieval failure diagnosable, and every
part of that — parsing the brief, formatting a hit, counting coverage — is a
pure function over data a test can construct.
"""

from __future__ import annotations

import itertools
import pathlib
import sys
from uuid import UUID

import pytest

from app.retrieval import cli
from app.retrieval.cli import (
    SNIPPET_CHARS,
    coverage,
    format_passage,
    load_brief_questions,
)
from app.retrieval.queries import Passage, SourceType
from app.retrieval.retriever import RetrievedPassage

# Distinct row ids without a test having to name them. `coverage` counts shapes
# across a result set and never looks at identity, but two passages sharing an
# id is a trap to leave lying around for whoever reuses this next.
_ids = itertools.count(1)


def _result(
    ticker: str, fiscal_year: int, source_type: SourceType = "chunk"
) -> RetrievedPassage:
    """A result carrying only the three fields `coverage` reads.

    Everything else is filler. Coverage is a claim about the shape of a whole
    result set, so a ten-passage set has to read as that shape rather than as a
    hundred lines of fields nothing looks at.
    """
    return RetrievedPassage(
        passage=Passage(
            source_type=source_type,
            row_id=UUID(int=next(_ids)),
            document_id=UUID(int=0),
            text="Item 1A. Risk Factors\n\nThe Company depends on a single supplier.",
            title="Segment revenue" if source_type == "table" else None,
            ticker=ticker,
            fiscal_year=fiscal_year,
            form="10-K",
        ),
        score=0.0286,
        rank=1,
        contributions={"vector": 1},
    )


def test_the_brief_still_holds_its_ten_questions():
    """The brief still asks ten questions, numbered 1..10.

    The brief is the exit criterion, so its shape is part of the contract. This
    is not a test of the regex — it is a test that nobody has quietly moved the
    bar. If an edit drops it to eight, every run afterwards passes a smaller
    test while looking identical.

    The numbers are asserted, not just the count: a reformat to ten `1.` items
    still counts ten while labelling every question "1", and the report would
    then name the wrong question for nine of them.
    """
    questions = load_brief_questions()

    # Ten written out rather than `BRIEF_QUESTION_COUNT`, which is the whole
    # point of the test. `load_brief_questions` already raises unless the
    # numbers are 1..BRIEF_QUESTION_COUNT, so importing the constant would move
    # the bar and the check of it together and pass at any value.
    assert [number for number, _ in questions] == list(range(1, 11))

    # `\s+` is greedy but gives back a character for `.+`, so a marker followed
    # by nothing but spaces parses as a question whose text strips to empty.
    assert all(text for _, text in questions)


def _write_brief(tmp_path, monkeypatch, questions: list[str]) -> pathlib.Path:
    """Point `cli.BRIEF` at a brief built from these numbered lines.

    Patching the module global rather than passing a path: `BRIEF` is resolved
    at call time out of the cli module's namespace, so this reaches the
    function imported directly above too.
    """
    brief = tmp_path / "client-brief.md"
    brief.write_text(
        "# Client brief\n\nThe analyst wants answers to the following.\n\n"
        + "\n".join(questions)
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "BRIEF", brief)
    return brief


def test_a_malformed_brief_raises_rather_than_silently_shrinking(tmp_path, monkeypatch):
    """A brief that no longer holds ten questions raises, naming what it found.

    The failure this guards is silent by nature, so the guard has to be loud.
    """
    brief = _write_brief(tmp_path, monkeypatch, [
        "1. How did Apple's revenue mix change?",
        "2. What did NVIDIA say about supply constraints?",
        "3. Which companies changed their AI risk language?",
    ])

    with pytest.raises(ValueError) as excinfo:
        load_brief_questions()

    message = str(excinfo.value)
    # What it found and what it wanted, both. "expected 1..10" alone leaves the
    # reader diffing the brief by hand to see how it went wrong.
    assert "[1, 2, 3]" in message
    assert "1..10" in message
    assert str(brief) in message


def test_a_rewritten_brief_is_caught_even_at_the_right_length(tmp_path, monkeypatch):
    """Ten questions all numbered `1.` are rejected, not accepted as ten.

    This is the reformat that makes the check on *numbers* rather than count
    worth having, and the one a shorter brief cannot stand in for: markdown
    renders a list of ten `1.` items as 1-10, so the document looks untouched
    while `_NUMBERED` reads ten ones. A count check passes, `--brief` then
    labels every question Q1, and the report names the wrong question for nine
    of them while looking entirely normal.
    """
    _write_brief(tmp_path, monkeypatch, [
        f"1. Question number {i} about the corpus." for i in range(1, 11)
    ])

    with pytest.raises(ValueError) as excinfo:
        load_brief_questions()

    # The repeated number is the finding. Asserting only that it raised would
    # pass on a check that rejected the brief for its length, which is exactly
    # the weaker check this exists to rule out.
    assert "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]" in str(excinfo.value)


def test_neither_a_question_nor_brief_is_a_usage_error(monkeypatch):
    """A question and `--brief` are exclusive, and one of them is required.

    argparse refusing beats `run_one` printing an empty report, which reads
    exactly like a corpus with no matches.
    """
    # `parse_args` reads sys.argv itself, so the invocation is set here rather
    # than passed in. argv[0] is the program name and argparse skips it.
    monkeypatch.setattr(sys, "argv", ["cli"])
    with pytest.raises(SystemExit) as neither:
        cli.parse_args()

    monkeypatch.setattr(
        sys, "argv", ["cli", "how did Apple describe supplier risk", "--brief"]
    )
    with pytest.raises(SystemExit) as both:
        cli.parse_args()

    # 2, not merely "it exited": `--help` also raises SystemExit, at 0. Two is
    # the usage-error code a caller can branch on.
    assert neither.value.code == 2
    assert both.value.code == 2


def test_per_ticker_requires_a_ticker(monkeypatch, capsys):
    """`--per-ticker` with nothing to fan out over is refused at the usage line.

    `retrieve_per_ticker` raises on an empty ticker list, and a usage line
    explains the mistake better than that traceback.
    """
    # A question is supplied deliberately. Without one this exits 2 for the
    # other usage error entirely — the exclusive group — and the test would
    # pass without `--per-ticker` ever being looked at.
    monkeypatch.setattr(
        sys, "argv", ["cli", "capital expenditures", "--per-ticker", "3"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_args()

    assert excinfo.value.code == 2
    # Both usage errors exit 2, so the code alone does not say which one fired.
    # The message is what tells the two apart.
    stderr = capsys.readouterr().err
    assert "--per-ticker" in stderr
    assert "--ticker" in stderr

    # The other half of the claim: the flag is rejected for want of a ticker,
    # not rejected outright. Without this the test passes on a parser that has
    # dropped --per-ticker altogether.
    monkeypatch.setattr(
        sys,
        "argv",
        ["cli", "capital expenditures", "--per-ticker", "3", "--ticker", "AAPL"],
    )
    args = cli.parse_args()
    assert args.per_ticker == 3
    assert args.tickers == ("AAPL",)


def test_repeated_ticker_flags_accumulate(monkeypatch):
    """Every `--ticker` is collected, in order; none at all yields `()`.

    Both consumers want a tuple — `Filters.tickers` is one, and
    `retrieve_per_ticker` takes a sequence — so the shape is the contract, not
    just the contents.
    """
    monkeypatch.setattr(
        sys, "argv", ["cli", "supplier risk", "--ticker", "AAPL", "--ticker", "NVDA"]
    )
    # An exact tuple rather than a membership check: repeating the flag has to
    # preserve the order it was written in, since that is the order the fan-out
    # reports companies in.
    assert cli.parse_args().tickers == ("AAPL", "NVDA")

    # A second call in the same process, deliberately after the first. Anything
    # held across calls — a parser hoisted to module scope, a default object
    # outliving the parser that owns it — is visible here and nowhere else.
    monkeypatch.setattr(sys, "argv", ["cli", "supplier risk"])
    tickers = cli.parse_args().tickers

    # `()`, not None. argparse leaves an `append` dest at its default when the
    # flag never fires, and `parse_args` normalises that away: `Filters` would
    # reject None, and `--per-ticker`'s own check reads this as a truth value.
    assert tickers == ()


def test_the_header_line_shows_both_arms():
    """An arm that missed the passage is named and marked, not left out.

    This is the CLI's whole purpose. "The corpus does not contain this" and
    "one arm found it at rank 2 and fusion buried it" are indistinguishable in
    a ranked list and need opposite fixes; the gap in the header is the only
    place that difference is visible.
    """
    passage = Passage(
        source_type="chunk",
        row_id=UUID(int=1),
        document_id=UUID(int=2),
        text="Item 1A. Risk Factors\n\nThe Company depends on a single supplier.",
        title=None,
        ticker="AAPL",
        fiscal_year=2024,
        form="10-K",
    )

    text_only = format_passage(
        RetrievedPassage(
            passage=passage, score=0.0164, rank=1, contributions={"text": 2}
        )
    ).splitlines()[0]

    # The exact rendering, padding included. The width is what keeps the arm
    # columns aligned down a ten-question run, which is what makes a missing
    # arm something you notice while skimming rather than something you find.
    assert "vector= -" in text_only
    assert "text= 2" in text_only

    # Both arms, to show the dash is conditional. Without this the test passes
    # on a formatter that prints `vector= -` unconditionally — which would hide
    # exactly the case it exists to reveal, an arm that did find the passage.
    both = format_passage(
        RetrievedPassage(
            passage=passage,
            score=0.0286,
            rank=1,
            contributions={"vector": 10, "text": 10},
        )
    ).splitlines()[0]

    assert "vector=10" in both
    assert "text=10" in both


def test_a_table_is_marked_and_keeps_its_markdown():
    """A table is labelled apart from a chunk and keeps one row per line.

    A chunk's prose is collapsed to one paragraph to spend the snippet on text;
    doing that to a table would render its rows unreadable.
    """
    rows = [
        "| Segment | 2024 | 2023 |",
        "| --- | --- | --- |",
        "| iPhone | 201,183 | 200,583 |",
        "| Services | 96,169 | 85,200 |",
    ]
    table = RetrievedPassage(
        passage=Passage(
            source_type="table",
            row_id=UUID(int=3),
            document_id=UUID(int=2),
            text="\n".join(rows),
            title="Segment revenue",
            ticker="AAPL",
            fiscal_year=2024,
            form="10-K",
        ),
        score=0.0286,
        rank=1,
        contributions={"vector": 1},
    )

    header, *body = format_passage(table).splitlines()

    # Caps against a lowercase `chunk`, which is what makes the two tellable
    # apart in a terminal with no colour.
    assert "TABLE" in header
    # The caption comes from the table's own column. The markdown never
    # contains it, so a formatter reading the body alone would print no label.
    assert "Segment revenue" in header
    # Every row on its own line, in order. Collapsed, a table is a single run
    # of pipes and digits that answers nothing at a glance.
    assert [line.strip() for line in body] == rows

    prose = RetrievedPassage(
        passage=Passage(
            source_type="chunk",
            row_id=UUID(int=4),
            document_id=UUID(int=2),
            text="Item 1A. Risk Factors\n\nThe Company depends\non a single supplier.",
            title=None,
            ticker="AAPL",
            fiscal_year=2024,
            form="10-K",
        ),
        score=0.0286,
        rank=1,
        contributions={"vector": 1},
    )

    header, *body = format_passage(prose).splitlines()

    assert "TABLE" not in header
    assert "chunk" in header
    # The other half of the claim, and the reason the table branch exists at
    # all: prose *is* collapsed, so a snippet spends its 220 characters on
    # words rather than on the line breaks the extractor happened to leave.
    assert body == ["     The Company depends on a single supplier."]


def test_the_snippet_says_how_much_it_dropped():
    """A clipped body ends with the number of characters it did not show.

    "The passage is chunked badly" is one of the three diagnoses this tool
    separates, and a hit whose snippet ends `(+2,940 more)` is a chunk that
    swallowed most of a section.
    """
    # 3,160 characters, so exactly 2,940 fall outside the snippet. Single
    # spaces throughout: the prose path collapses whitespace before clipping,
    # and a double space would move the count off the number written here.
    body = ("single-source supplier risks " * 109).strip()
    assert len(body) - SNIPPET_CHARS == 2940

    long_chunk = RetrievedPassage(
        passage=Passage(
            source_type="chunk",
            row_id=UUID(int=5),
            document_id=UUID(int=2),
            text=f"Item 1A. Risk Factors\n\n{body}",
            title=None,
            ticker="AAPL",
            fiscal_year=2024,
            form="10-K",
        ),
        score=0.0286,
        rank=1,
        contributions={"vector": 1},
    )

    _, snippet = format_passage(long_chunk).splitlines()

    # The grouped digits are the point of the marker. A section swallowed whole
    # shows four figures here, and `+2940 more` is read as a smaller number
    # than it is at exactly the moment that number is the finding.
    assert snippet.endswith("…(+2,940 more)")
    assert snippet.strip().startswith(body[:40])

    # --full is the escape hatch the flag advertises, so the marker must be an
    # artefact of clipping rather than something appended to every body.
    _, whole = format_passage(long_chunk, full=True).splitlines()
    assert whole.strip() == body

    short_chunk = RetrievedPassage(
        passage=Passage(
            source_type="chunk",
            row_id=UUID(int=6),
            document_id=UUID(int=2),
            text="Item 1A. Risk Factors\n\nThe Company depends on a single supplier.",
            title=None,
            ticker="AAPL",
            fiscal_year=2024,
            form="10-K",
        ),
        score=0.0286,
        rank=1,
        contributions={"vector": 1},
    )

    # Nothing dropped, nothing said. A marker on every hit would be noise on
    # the ones that are fine, and the whole value of it is that it stands out.
    _, untouched = format_passage(short_chunk).splitlines()
    assert "more)" not in untouched


def test_coverage_counts_companies_years_and_tables():
    """The line reports companies, the year span, the years in it, and the mix.

    This is the check a passage-by-passage read cannot make. Ten
    individually-good passages drawn from two of five companies answer nothing,
    and that failure has no per-passage symptom — only a shape.

    The distinct-year count is not decoration: ten passages drawn from two of
    five years look identical to good coverage when only the endpoints print.
    """
    # The failure this line exists to show: six passages that look fine one at
    # a time, covering two of five companies and two of the five years they
    # appear to span.
    gappy = [
        _result("NVDA", 2021),
        _result("NVDA", 2021),
        _result("NVDA", 2025),
        _result("NVDA", 2025, "table"),
        _result("AAPL", 2021),
        _result("AAPL", 2025, "table"),
    ]

    # The whole line, spacing included. Each field is a separate claim and the
    # arrangement is itself the contract — this is read at a glance under ten
    # questions of output, so the columns landing where the eye expects them is
    # not incidental. NVDA leads because companies sort by count: the shape is
    # meant to be the first thing read, not something to be counted up.
    assert coverage(gappy) == (
        "coverage: 6 passages  2 cos  NVDA×4 AAPL×2  "
        "FY2021–2025 (2 of 5 yrs)  chunks 4  tables 2"
    )

    # Every year present, so the count is dropped. Without this the test passes
    # on a line that always prints `(n of n yrs)`, which would turn the one
    # signal worth noticing into furniture that is read past.
    complete = [_result("AAPL", year) for year in range(2021, 2026)]
    assert coverage(complete) == (
        "coverage: 5 passages  1 cos  AAPL×5  FY2021–2025  chunks 5  tables 0"
    )

    # Singular, and a point rather than a span. A one-passage answer is a
    # normal result for a narrow question and should not read as a bug.
    assert coverage([_result("AAPL", 2024)]) == (
        "coverage: 1 passage  1 cos  AAPL×1  FY2024  chunks 1  tables 0"
    )


def test_coverage_of_nothing_says_so():
    """An empty result set still gets a line, and it says what happened.

    A blank line reads as a formatting bug. "Nothing retrieved" is a finding.
    """
    # Same `coverage:` prefix as every populated line, so the eye running down
    # a ten-question report lands on it in the column it expects rather than
    # skipping a question that produced nothing.
    assert coverage([]) == "coverage: nothing retrieved"
