"""Tests for ingest.run — the CLI and the per-filing orchestration."""

from __future__ import annotations


def test_ticker_filter_is_case_insensitive():
    """TODO: assert --ticker aapl selects the AAPL filings."""
    raise NotImplementedError


def test_limit_rejects_zero_and_negatives():
    """TODO: assert argparse errors on --limit 0. Failing at parse time beats
    a run that silently does nothing."""
    raise NotImplementedError


def test_year_filters_on_fiscal_year_not_filing_date():
    """TODO: assert --year 2024 selects a filing whose report_date is in 2024
    even though its filing_date is 2025.

    Half the corpus files the following calendar year; filtering on
    filing_date would return the wrong filings for four of the five companies.
    """
    raise NotImplementedError


def test_dry_run_writes_nothing(monkeypatch):
    """TODO: assert --dry-run reaches no persistence function and makes no
    embedding call."""
    raise NotImplementedError


def test_content_hash_changes_with_extractor_version():
    """TODO: assert the same bytes hash differently when EXTRACTOR_VERSION
    changes.

    That is the mechanism for invalidating the corpus after a parser fix: the
    same HTML now produces different chunks, so every document must be re-read.
    """
    raise NotImplementedError


def test_unchanged_filing_skips_extraction(monkeypatch):
    """TODO: assert a filing whose stored content_hash matches is not
    re-extracted, and that embedding still resumes for chunks missing one."""
    raise NotImplementedError
