"""Tests for ingest.extract.

Every case below corresponds to a bug that reached the corpus. The heuristics in
extract.py look arbitrary until one of these regresses, so each test should say
in its name what breaks if the rule is dropped.
"""

from __future__ import annotations

import lxml.html
import pytest

from ingest.extract import MAX_HEADING_CHARS, _build_table, extract


def _sections(fixture_html, name: str) -> dict:
    filing = extract(str(fixture_html(name)))
    return {section.item: section for section in filing.sections}


def test_finds_sections_in_document_order(fixture_html):
    """apple_style has Items 1, 1A, 1B, 5, 7 in that order.

    Order matters beyond tidiness — chunk_index is assigned by walking
    sections, and the DB relies on it being contiguous.
    """
    filing = extract(str(fixture_html("apple_style")))

    assert [s.item for s in filing.sections] == ["1", "1A", "1B", "5", "7"]


def test_linked_toc_entry_is_not_a_section(fixture_html):
    """apple_style's TOC links "Item 1A." before the real heading.

    If the <a href> filter is dropped, the TOC entry becomes an empty
    duplicate section and the real Risk Factors body is attributed to it — so
    asserting on the body, not just the count, is what pins the failure.
    """
    filing = extract(str(fixture_html("apple_style")))
    items = [s.item for s in filing.sections]

    assert items.count("1A") == 1
    risk_factors = next(s for s in filing.sections if s.item == "1A")
    assert risk_factors.text == "The Company depends on a single supplier for certain components."


def test_dot_leader_toc_entry_is_not_a_section(fixture_html):
    """msft_style's TOC has no links at all, only dot leaders.

    This is what ITEM_TITLE_WORD is for: without it Microsoft yields 76
    sections against Apple's 23, every Item duplicated once from the TOC. The
    <a href> filter cannot help — there is nothing to filter on.
    """
    filing = extract(str(fixture_html("msft_style")))

    assert [s.item for s in filing.sections] == ["1", "1A"]
    assert all("..." not in s.title for s in filing.sections)


def test_long_heading_is_not_truncated_away(fixture_html):
    """apple_style's Item 5 heading is 105 characters.

    A 90-char limit silently swallowed Items 5, 7, 9 and 12 into the preceding
    section, which put 15,063 characters of Apple's MD&A under
    "Item 6. [Reserved]". The body assertions are the ones that catch it: a
    heading that fails to match does not disappear, it becomes text belonging
    to whichever section came before.
    """
    sections = _sections(fixture_html, "apple_style")
    item5 = sections["5"]

    assert item5.title == (
        "Market for Registrant’s Common Equity, Related Stockholder "
        "Matters and Issuer Purchases of Equity"
    )
    assert 90 < len(item5.heading) <= MAX_HEADING_CHARS
    assert item5.text.startswith("The Company’s common stock")
    assert sections["1B"].text == "None.", "Item 5 was absorbed by the section above it"


def test_heading_on_a_non_leaf_block_is_found(fixture_html):
    """amazon_style puts the full heading only on a non-leaf div.

    Matching leaf blocks only drops Items 10-15 from every Amazon filing: the
    leaves hold a bare "Item 10." with no title. Matching the outermost block
    and skipping its subtree is what handles both filers, and the body
    assertion is what proves the subtree was skipped rather than re-collected.
    """
    sections = _sections(fixture_html, "amazon_style")

    assert list(sections) == ["1", "10", "11"]
    assert sections["10"].title == "Directors, Executive Officers, and Corporate Governance"
    assert sections["11"].title == "Executive Compensation"
    assert sections["10"].text.startswith("Information required by Item 10")
    assert "Corporate Governance" not in sections["10"].text


def test_prose_cross_reference_is_not_a_heading(fixture_html):
    """apple_style's Item 1 body mentions "Part I, Item 1A".

    Filings cross-reference their own items constantly. A heading has to be a
    short standalone block, not merely a line containing "Item 1A".
    """
    filing = extract(str(fixture_html("apple_style")))
    sections = {s.item: s for s in filing.sections}

    assert "Part I, Item 1A" in sections["1"].text
    assert [s.item for s in filing.sections].count("1A") == 1


def test_hidden_xbrl_facts_are_dropped(fixture_html):
    """apple_style hides 999999 inside <ix:hidden>.

    WORKED EXAMPLE — shows the fixture_html pattern; the rest are yours.

    ix:hidden never renders. Leaving it in injects numbers with no context into
    chunks an analyst would then see cited.
    """
    filing = extract(str(fixture_html("apple_style")))

    assert "999999" not in filing.markdown
    for section in filing.sections:
        assert "999999" not in section.text, f"leaked into {section.heading}"
    for table in filing.tables:
        assert "999999" not in table.markdown


def test_empty_section_is_kept(fixture_html):
    """Item 1B's body is "None.".

    "Did they report unresolved staff comments?" has an answer, and dropping
    the section turns it into a gap the model has to guess at.
    """
    sections = _sections(fixture_html, "apple_style")

    assert sections["1B"].text == "None."
    assert sections["1B"].heading == "Item 1B. Unresolved Staff Comments"


def test_table_cells_are_coalesced(fixture_html):
    """apple_style's table splits "$", "201,183", "—", "%" across cells.

    These are the figures an analyst checks the answer against. A bare "$" or
    "%" cell means the row rendered as unreadable fragments.
    """
    filing = extract(str(fixture_html("apple_style")))
    table = filing.tables[0]
    rows = {row[0]: row for row in table.rows}

    assert rows["iPhone"] == ["iPhone", "$201,183", "—%", "$200,583"]
    assert rows["Total net sales"] == ["Total net sales", "$391,035", "2%", "$383,285"]
    assert all(cell not in ("$", "%", "") for row in table.rows for cell in row)
    assert "$201,183" in table.markdown
    assert table.units == "in millions"


def test_table_hash_ignores_presentation_changes():
    """Styling moves do not move source_html_hash, but figures do.

    The hash exists to tell a content change from a reformat; if styling moves
    it, every re-extraction looks like a change and every table is rewritten.
    The negative half matters just as much — a hash that ignores everything
    would pass the first assertion alone.
    """
    plain = "<table><tr><td>Revenue</td><td>$1,000</td></tr><tr><td>Cost</td><td>$400</td></tr></table>"
    restyled = (
        '<table class="fin"><tr style="height:12pt"><td id="a">Revenue</td>'
        '<td colspan="2">$1,000</td></tr><tr><td>Cost</td><td>$400</td></tr></table>'
    )
    edited = plain.replace("$1,000", "$9,000")

    def build(html: str):
        return _build_table(lxml.html.fromstring(html), 0)

    assert build(plain).source_html_hash == build(restyled).source_html_hash
    assert build(plain).source_html_hash != build(edited).source_html_hash


def test_duplicate_items_raise(tmp_path):
    """The same Item twice is a hard failure, not a silent duplicate.

    That guard is the last line of defence against a TOC variant nobody has
    seen yet — better a failed ingest than a silently duplicated corpus, where
    retrieval returns the same passage twice and one copy holds the wrong body.
    """
    path = tmp_path / "duplicate.htm"
    path.write_text(
        "<html><body>"
        '<div style="font-weight:700">Item 1A. Risk Factors</div>'
        "<div>Supply chain concentration.</div>"
        '<div style="font-weight:700">Item 1A. Risk Factors</div>'
        "<div>Currency exposure.</div>"
        "</body></html>"
    )

    with pytest.raises(ValueError, match="duplicate items") as excinfo:
        extract(str(path))

    assert "1A" in str(excinfo.value)


def test_comment_before_table_does_not_crash(fixture_html):
    """A comment sitting where _title_of walks must not raise.

    lxml gives comments a callable tag, so itertext() raises
    "Input object is not an XML element: HtmlComment" on them. SEC filings are
    full of filing-agent comments; this survived all 25 real filings only
    because none happened to sit where _title_of reaches.

    msft_style is the fixture that reproduces it — apple_style's commented TOC
    table is rejected on row count before _title_of ever runs. The title coming
    back None is the evidence the walk stepped over the comment and kept going.
    """
    path = fixture_html("msft_style")
    assert "-->" in path.read_text(), "fixture no longer holds the comment this pins"

    filing = extract(str(path))

    assert len(filing.tables) == 1
    assert filing.tables[0].title is None
