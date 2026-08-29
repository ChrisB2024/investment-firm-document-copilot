"""Tests for app.retrieval.cli.

All fast. The CLI's job is to make a retrieval failure diagnosable, and every
part of that — parsing the brief, formatting a hit, counting coverage — is a
pure function over data a test can construct.
"""

from __future__ import annotations


def test_the_brief_still_holds_its_ten_questions():
    """TODO: assert `load_brief_questions` returns ten, numbered 1..10.

    The brief is the exit criterion, so its shape is part of the contract. This
    is not a test of the regex — it is a test that nobody has quietly moved the
    bar. If an edit drops it to eight, every run afterwards passes a smaller
    test while looking identical.

    Assert the numbers, not just the count: a reformat to ten `1.` items still
    counts ten while labelling every question "1", and the report would then
    name the wrong question for nine of them.
    """
    raise NotImplementedError


def test_a_malformed_brief_raises_rather_than_silently_shrinking(tmp_path, monkeypatch):
    """TODO: point `BRIEF` at a file with three questions and assert ValueError
    naming what it found.

    The failure this guards is silent by nature, so the guard has to be loud.
    """
    raise NotImplementedError


def test_neither_a_question_nor_brief_is_a_usage_error(monkeypatch):
    """TODO: assert `parse_args` exits 2 with neither, and with both.

    argparse refusing beats `run_one` printing an empty report, which reads
    exactly like a corpus with no matches.
    """
    raise NotImplementedError


def test_per_ticker_requires_a_ticker(monkeypatch):
    """TODO: assert `--per-ticker 3` with no `--ticker` is a usage error.

    `retrieve_per_ticker` raises on an empty ticker list, and a usage line
    explains the mistake better than that traceback.
    """
    raise NotImplementedError


def test_repeated_ticker_flags_accumulate(monkeypatch):
    """TODO: assert `--ticker AAPL --ticker NVDA` yields both, and that a run
    with none yields an empty tuple rather than None.

    argparse `action="append"` appends into its default object, so a shared
    list would carry results between calls in one process.
    """
    raise NotImplementedError


def test_the_header_line_shows_both_arms():
    """TODO: format a result found by one arm only; assert the header shows the
    missing arm as absent rather than omitting it.

    This is the CLI's whole purpose. "The corpus does not contain this" and
    "one arm found it at rank 2 and fusion buried it" are indistinguishable in
    a ranked list and need opposite fixes; the gap in the header is the only
    place that difference is visible.
    """
    raise NotImplementedError


def test_a_table_is_marked_and_keeps_its_markdown():
    """TODO: assert a table result is labelled distinctly from a chunk and its
    row structure survives formatting.

    A chunk's prose is collapsed to one paragraph to spend the snippet on text;
    doing that to a table would render its rows unreadable.
    """
    raise NotImplementedError


def test_the_snippet_says_how_much_it_dropped():
    """TODO: assert a clipped body reports the dropped character count.

    "The passage is chunked badly" is one of the three diagnoses this tool
    separates, and a hit whose snippet ends `(+2,940 more)` is a chunk that
    swallowed most of a section.
    """
    raise NotImplementedError


def test_coverage_counts_companies_years_and_tables():
    """TODO: assert the line reports company counts, the year span, how many of
    the years in that span are actually present, and the chunk/table split.

    This is the check a passage-by-passage read cannot make. Ten
    individually-good passages drawn from two of five companies answer nothing,
    and that failure has no per-passage symptom — only a shape.

    The distinct-year count is not decoration: ten passages drawn from two of
    five years look identical to good coverage when only the endpoints print.
    """
    raise NotImplementedError


def test_coverage_of_nothing_says_so():
    """TODO: assert an empty result set produces a coverage line rather than a
    blank one.

    A blank line reads as a formatting bug. "Nothing retrieved" is a finding.
    """
    raise NotImplementedError
