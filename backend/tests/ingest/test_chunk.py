"""Tests for ingest.chunk."""


from __future__ import annotations

import json
import re
from itertools import pairwise

import pytest
import tiktoken

from ingest.chunk import (
    _REQUIRED_METADATA,
    ENCODING,
    OVERLAP_TOKENS,
    TARGET_TOKENS,
    _get_encoder,
    chunk_filing,
    chunk_section,
    token_count,
)
from ingest.extract import Section


def _prose(marker: str, sentences: int = 200) -> str:
    """A body long enough to split, built from sentence-sized atoms."""
    return " ".join(
        f"{marker} sentence {i} describes a distinct risk to the business."
        for i in range(sentences)
    )


def _metadata(entry: dict) -> dict:
    """The filing-level dict ingest.run hands to chunk_filing.

    Rebuilt here rather than imported from ingest.run: that module pulls in
    app.config, and a chunking unit test has no business needing a .env. The
    key set is pinned against _REQUIRED_METADATA in
    test_missing_required_metadata_raises, so drift still fails loudly. Whether
    ingest.run itself produces these keys is test_run.py's job.
    """
    return {
        "ticker": entry["ticker"],
        "company": "Apple Inc.",
        "form": entry["form"],
        "filing_date": entry["filing_date"],
        "fiscal_year": int(entry["report_date"][:4]),
        "accession_number": entry["accession_number"],
    }


def test_heading_is_prepended_to_every_chunk():
    """Every chunk starts with its section heading.

    Retrieval matches on the chunk's own text. A passage reading "The Company
    relies on a single supplier..." names neither company nor section, so it is
    both unfindable and uncitable.
    """
    body = "\n\n".join(
        f"Paragraph {i}. The Company relies on a single supplier. " * 10
        for i in range(20)
    )
    section = Section(item="1A", title="Risk Factors", text=body)
    assert token_count(body) > TARGET_TOKENS * 2
    chunks = chunk_section(section, 0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith(section.heading)



def test_chunks_respect_the_token_budget():
    """Chunk token counts stay within TARGET_TOKENS for a body of prose.

    This pins current behaviour, not intended behaviour. The budget is enforced
    on summed atom counts while the chunk text is a span slice, so inter-atom
    separators go uncounted. Prose paragraphs are few enough and large enough
    that the gap stays under the limit here; test_never_splits_a_table_row feeds
    300 short rows, where the same gap puts every chunk over it.

    The lower bound matters as much as the upper one: without it a regression
    that emitted one atom per chunk would satisfy the upper bound trivially.
    """
    body = "\n\n".join(
        f"Paragraph {i}. The Company relies on a single supplier for this component. " * 10
        for i in range(20)
    )
    section = Section(item="1A", title="Risk Factors", text=body)
    chunks = chunk_section(section, 0)
    assert len(chunks) > 1
    assert max(c.token_count for c in chunks) <= TARGET_TOKENS
    assert min(c.token_count for c in chunks[:-1]) >= TARGET_TOKENS * 0.6


@pytest.mark.xfail(reason="j == i takes the first atom unconditionally; a paragraph with no sentence boundary is never split", strict=True)
def test_an_unsplittable_paragraph_still_blows_the_budget():
    blob = " ".join(f"word{i}" for i in range(3000))
    chunks = chunk_section(Section(item="1A", title="Risk Factors", text=blob), 0)
    assert chunks[0].token_count <= TARGET_TOKENS

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")

def test_never_splits_mid_sentence():
    """Every chunk body ends at a sentence boundary.

    A half-sentence citation costs trust faster than a missing answer.

    The body is a single paragraph with no blank line, which is what sends
    _atomize down _sentence_spans instead of _paragraph_spans. Reformatted into
    paragraphs this test still passes while no longer touching sentence
    splitting at all.

    The offset checks are not redundant with the regex: a cut after the "." in
    "$3.5 billion" satisfies the terminator but leaves a non-space next, which
    is what _SENTENCE_BOUNDARY's trailing-whitespace requirement exists to
    prevent. Neither check catches an abbreviation cut — that is
    test_abbreviations_do_not_end_a_sentence's job.
    """
    body = " ".join(
        f"Sentence {i} describes a distinct risk to the business of the Company."
        for i in range(200)
    )
    section = Section(item="1A", title="Risk Factors", text=body)
    chunks = chunk_section(section, 0)

    assert len(chunks) > 1
    for chunk in chunks:
        chunk_body = chunk.text.removeprefix(f"{section.heading}\n\n")
        assert _SENTENCE_END.search(chunk_body), f"cut mid-sentence: ... {chunk_body[-60:]!r}"

        start = body.find(chunk_body)
        assert start != -1, "chunk text is not a verbatim slice of the body"
        end = start + len(chunk_body)
        assert start == 0 or body[start - 1].isspace()
        assert end == len(body) or body[end].isspace()



def test_never_splits_a_table_row():
    """No chunk boundary falls inside a table row.

    Membership catches a cut inside a row. The slice check catches rows dropped
    or reordered, which membership alone would pass.
    """
    body = "\n".join(
        f"Product line {i:<8}{i * 1101:>12,}{i * 1303:>12,}{i * 1709:>12,}"
        for i in range(300)
    )
    section = Section(item="7", title="Management's Discussion and Analysis", text=body)
    chunks = chunk_section(section, 0)

    # Rows have no sentence terminator, so >1 chunk means _looks_like_table won.
    assert len(chunks) > 1

    rows = set(body.splitlines())
    for chunk in chunks:
        chunk_body = chunk.text.removeprefix(f"{section.heading}\n\n")
        for line in chunk_body.splitlines():
            assert line in rows, f"boundary split a row: {line!r}"

        start = body.find(chunk_body)
        assert start != -1, "chunk text is not a verbatim slice of the body"
        end = start + len(chunk_body)
        assert start == 0 or body[start - 1] == "\n"
        assert end == len(body) or body[end] == "\n"



_ABBREVIATION_END = re.compile(r"(?:\bInc|\bU\.S)\.$")


def test_abbreviations_do_not_end_a_sentence():
    """"Inc." and "U.S." do not create a split.

    _ABBREVIATION_TAIL exists for this; without it a chunk can end at
    "Apple Inc." mid-clause.

    Both abbreviations sit mid-sentence and the real terminator is a distinct
    tail, so ending anywhere but that tail means a boundary was taken early.
    Neutering _ABBREVIATION_TAIL takes this body from 200 atoms to 600 and ends
    9 of 10 chunks at "Apple Inc.", so the assertions are load-bearing.

    The count guard is not decoration: edit the template so it no longer holds
    both abbreviations and every remaining assertion still passes while testing
    nothing.
    """
    template = (
        "In fiscal year {i} the Company competed with Apple Inc. and other "
        "registrants across the U.S. market for components and services."
    )
    body = " ".join(template.format(i=2000 + i) for i in range(200))
    assert body.count("Apple Inc.") == 200
    assert body.count("U.S.") == 200

    section = Section(item="1A", title="Risk Factors", text=body)
    chunks = chunk_section(section, 0)

    assert len(chunks) > 1
    for chunk in chunks:
        chunk_body = chunk.text.removeprefix(f"{section.heading}\n\n")
        assert not _ABBREVIATION_END.search(chunk_body), (
            f"split after an abbreviation: ...{chunk_body[-40:]!r}"
        )
        assert chunk_body.endswith("market for components and services.")
        assert chunk_body.startswith("In fiscal year ")



def test_tiny_section_becomes_its_own_chunk(manifest_entry):
    """A 5-character section is its own chunk, not merged into a neighbour.

    Merging would attribute Item 1B's answer to another Item — a wrong answer
    carrying a real citation, which is the failure mode the whole product is
    built to avoid.
    """
    tiny = Section(item="1B", title="Unresolved Staff Comments", text="None.")
    chunks = chunk_section(tiny, 0)
    assert len(chunks) == 1
    assert chunks[0].text == "Item 1B. Unresolved Staff Comments\n\nNone."
    assert chunks[0].section == tiny.heading

    # An Item with no body at all still yields one headed, citable chunk rather
    # than nothing — the empty-atoms branch of chunk_section.
    empty = chunk_section(Section(item="1B", title="Unresolved Staff Comments", text=""), 0)
    assert len(empty) == 1
    assert empty[0].text == "Item 1B. Unresolved Staff Comments"
    assert empty[0].section == tiny.heading

    filing = chunk_filing(
        [
            Section(item="1A", title="Risk Factors", text=_prose("Risk")),
            tiny,
            Section(item="2", title="Properties", text=_prose("Property")),
        ],
        _metadata(manifest_entry),
    )
    owned = [c for c in filing if c.section == tiny.heading]
    assert len(owned) == 1
    assert owned[0].text.endswith("None.")
    assert "Risk sentence" not in owned[0].text
    assert "Property sentence" not in owned[0].text


def test_chunk_index_is_contiguous_across_the_filing(manifest_entry):
    """Indices are exactly range(len(chunks)).

    UNIQUE (document_id, chunk_index) depends on it, and so does neighbour
    lookup, which reads index-1 and index+1 for surrounding context.

    The mix is deliberate: a section that splits, a section that does not, and
    one either side of the tiny one, so the count restarting per section or the
    tiny section being skipped both show up as a gap.
    """
    sections = [
        Section(item="1", title="Business", text=_prose("Business")),
        Section(item="1B", title="Unresolved Staff Comments", text="None."),
        Section(item="7", title="Management's Discussion and Analysis", text=_prose("MD&A")),
    ]
    chunks = chunk_filing(sections, _metadata(manifest_entry))

    assert len(chunks) > len(sections), "nothing split; the assertion below would be trivial"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_metadata_is_carried_onto_every_chunk(manifest_entry):
    """Every chunk carries the filing metadata plus its own section.

    It lands in a JSONB column, so a non-serialisable value (a function, a date
    object) fails at write time, long after chunking looked fine. The
    round-trip through json is what catches that here instead.
    """
    metadata = _metadata(manifest_entry)
    sections = [
        Section(item="1", title="Business", text=_prose("Business")),
        Section(item="1A", title="Risk Factors", text=_prose("Risk")),
    ]
    chunks = chunk_filing(sections, metadata)

    assert len(chunks) > len(sections)
    for chunk in chunks:
        assert chunk.metadata == {**metadata, "section": chunk.section}
        assert json.loads(json.dumps(chunk.metadata)) == chunk.metadata
    assert {c.metadata["section"] for c in chunks} == {s.heading for s in sections}

    # AAPL files in the same calendar year it reports, so the fixture alone
    # cannot tell fiscal_year read off report_date from fiscal_year read off
    # filing_date. Amazon's dates disagree, which is the case that matters.
    amazon = {**manifest_entry, "ticker": "AMZN",
              "filing_date": "2025-02-07", "report_date": "2024-12-31"}
    assert _metadata(amazon)["fiscal_year"] == 2024


def test_missing_required_metadata_raises(manifest_entry):
    """Dropping any required key raises ValueError naming it.

    Failing at ingest beats producing chunks that retrieval cannot filter and
    the citation UI cannot render.
    """
    metadata = _metadata(manifest_entry)
    assert set(metadata) == set(_REQUIRED_METADATA), (
        "this file's _metadata has drifted from chunk.py's required keys"
    )
    sections = [Section(item="1", title="Business", text="Everything is fine.")]

    for key in _REQUIRED_METADATA:
        incomplete = {k: v for k, v in metadata.items() if k != key}
        with pytest.raises(ValueError, match=key):
            chunk_filing(sections, incomplete)


def test_overlap_carries_context_between_chunks():
    """Consecutive chunks share trailing/leading text, and the walk-back
    always makes progress.

    Strictly increasing start offsets are the termination guarantee: the
    walk-back stops at `k - 1 > i`, so a bug that let k reach i would loop
    forever on the same atom.

    Atoms here are sentence-sized on purpose. The walk-back only takes an atom
    back if it fits in OVERLAP_TOKENS, so a body of ~120-token paragraphs gets
    no overlap at all — 21.4% of consecutive pairs across the corpus are in
    that position. Rebuild this body from paragraphs and the overlap assertion
    fails for that reason, not because anything regressed.
    """
    body = _prose("Risk")
    section = Section(item="1A", title="Risk Factors", text=body)
    chunks = chunk_section(section, 0)
    prefix = f"{section.heading}\n\n"

    spans = []
    for chunk in chunks:
        chunk_body = chunk.text.removeprefix(prefix)
        start = body.find(chunk_body)
        assert start != -1, "chunk text is not a verbatim slice of the body"
        spans.append((start, start + len(chunk_body)))

    assert len(spans) > 2
    starts = [start for start, _ in spans]
    assert starts == sorted(set(starts)), "walk-back failed to make progress"

    for (_, previous_end), (next_start, _) in pairwise(spans):
        assert next_start < previous_end, "consecutive chunks share no text"
        assert token_count(body[next_start:previous_end]) <= OVERLAP_TOKENS


def test_token_count_matches_the_encoder():
    """token_count agrees with tiktoken, and the encoder is cached.

    The literal 18 pins the encoding itself. Comparing only against
    tiktoken.get_encoding(ENCODING) would keep agreeing if ENCODING were
    changed to a different model's vocabulary, which would silently move every
    chunk boundary.
    """
    text = "Apple Inc. reported net sales of $391,035 million in fiscal 2024."
    assert token_count(text) == len(tiktoken.get_encoding(ENCODING).encode_ordinary(text))
    assert token_count(text) == 18
    assert token_count("") == 0
    assert _get_encoder(ENCODING) is _get_encoder(ENCODING)
