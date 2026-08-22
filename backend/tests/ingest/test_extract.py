"""Tests for ingest.extract.

Every case below corresponds to a bug that reached the corpus. The heuristics in
extract.py look arbitrary until one of these regresses, so each test should say
in its name what breaks if the rule is dropped.
"""

from __future__ import annotations

from ingest.extract import extract


def test_finds_sections_in_document_order(fixture_html):
    """apple_style has Items 1, 1A, 1B, 5, 7 in that order.

    TODO: assert the item codes come back in document order. Order matters
    beyond tidiness — chunk_index is assigned by walking sections, and the DB
    relies on it being contiguous.
    """
    raise NotImplementedError


def test_linked_toc_entry_is_not_a_section(fixture_html):
    """apple_style's TOC links "Item 1A." before the real heading.

    TODO: assert Item 1A appears exactly once. If the <a href> filter is
    dropped, the TOC entry becomes an empty duplicate section and the real
    Risk Factors body is attributed to it.
    """
    raise NotImplementedError


def test_dot_leader_toc_entry_is_not_a_section(fixture_html):
    """msft_style's TOC has no links at all, only dot leaders.

    TODO: assert exactly two sections. This is what ITEM_TITLE_WORD is for:
    without it Microsoft yields 76 sections against Apple's 23, every Item
    duplicated once from the TOC.
    """
    raise NotImplementedError


def test_long_heading_is_not_truncated_away(fixture_html):
    """apple_style's Item 5 heading is 116 characters.

    TODO: assert Item 5 is found and its title is the full text. A 90-char
    limit silently swallowed Items 5, 7, 9 and 12 into the preceding section,
    which put 15,063 characters of Apple's MD&A under "Item 6. [Reserved]".
    """
    raise NotImplementedError


def test_heading_on_a_non_leaf_block_is_found(fixture_html):
    """amazon_style puts the full heading only on a non-leaf div.

    TODO: assert Items 10 and 11 are found with their titles. Matching leaf
    blocks only drops Items 10-15 from every Amazon filing.
    """
    raise NotImplementedError


def test_prose_cross_reference_is_not_a_heading(fixture_html):
    """apple_style's Item 1 body mentions "Part I, Item 1A".

    TODO: assert that sentence stays in Item 1's text and creates no section.
    """
    raise NotImplementedError


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

    TODO: assert the section exists with that body. "Did they report
    unresolved staff comments?" has an answer, and dropping the section turns
    it into a gap the model has to guess at.
    """
    raise NotImplementedError


def test_table_cells_are_coalesced(fixture_html):
    """apple_style's table splits "$", "201,183", "—", "%" across cells.

    TODO: assert the rendered row keeps $201,183 and 200,583 together and
    intact. These are the figures an analyst checks the answer against.
    """
    raise NotImplementedError


def test_table_hash_ignores_presentation_changes():
    """TODO: build two tables differing only in style/class attributes and
    assert source_html_hash matches. The hash exists to tell a content change
    from a reformat; if styling moves it, every re-extraction looks like a
    change."""
    raise NotImplementedError


def test_duplicate_items_raise(tmp_path):
    """TODO: feed HTML with the same Item twice and assert ValueError.

    That guard is the last line of defence against a TOC variant nobody has
    seen yet — better a failed ingest than a silently duplicated corpus.
    """
    raise NotImplementedError


def test_comment_before_table_does_not_crash(fixture_html):
    """TODO: assert extract() succeeds on a fixture with an HTML comment
    preceding a table. lxml gives comments a callable tag, so itertext() raises
    on them; SEC filings are full of filing-agent comments."""
    raise NotImplementedError
