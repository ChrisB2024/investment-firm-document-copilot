"""SEC filing HTML -> sections, tables, and normalized Markdown.

What the corpus actually looks like, measured across all 25 filings rather than
assumed:

- The files are Inline XBRL produced by filing agents (Workiva and friends).
  A 10-K runs 1.5-6.5 MB, most of it markup: one Apple filing carries 11,098
  inline `style` attributes and 963 `ix:nonfraction` tags.
- There is not a single <h1>-<h6> tag in any filing. Section headings are bold
  <span>s. Structure has to be recovered from text, not tags.
- `ix:hidden` holds XBRL facts that never render. Drop it before taking text or
  you get duplicated numbers with no context.
- Every filing opens with a table of contents whose entries read exactly like
  the real headings ("Item 1A."). The TOC entries are wrapped in <a href>; the
  real headings are not. That one distinction separates them.
- Prose also cross-references items ("...discussed in Part I, Item 1A of this
  Form 10-K..."), which is why a heading candidate must be a short standalone
  block, not merely a line containing "Item 1A".
- Filers nest headings differently. Apple puts the whole heading in one leaf
  <div>; Amazon splits it so the leaf elements hold a bare "Item 10." with no
  title and only a *non-leaf* ancestor carries the full text. Matching leaves
  only loses every Amazon Part III item. Match the outermost block that fits
  the pattern and skip its subtree, which handles both without duplicating.
- Not every TOC uses links. Microsoft's uses dot leaders, so its entries pass an
  <a href> filter and arrive as sections titled "Item 1. ." — every item then
  appears twice, once from the TOC and once real. Requiring the captured title
  to contain a real word drops MSFT from 76 detected sections to 23.
- Financial tables split values across cells: "$", "201,183", "(2)", "%" are
  four separate <td>s. Rendering them naively gives unreadable rows.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, field

import lxml.html

# Verified against all 25 filings: yields 17-20 headings each, matching a real
# 10-K's Item structure with no misses and no TOC bleed.
ITEM_HEADING = re.compile(r"^item\s+(\d{1,2}[A-C]?)\s*[.:—–-]*\s*(.+)$", re.IGNORECASE)

# A real title contains a word. This is what separates a genuine heading from a
# dot-leader table-of-contents entry ("Item 1. ....."), which no link filter
# catches because not every filer links its TOC.
ITEM_TITLE_WORD = re.compile(r"[A-Za-z]{3}")

# A heading is a short standalone block; longer than this and it is prose that
# happens to mention an Item. Measured across all 25 filings: the longest real
# heading is 119 chars (Amazon's Item 5, "Market for the Registrant's Common
# Stock, Related Shareholder Matters, and Issuer Purchases of Equity
# Securities"), and no candidate matching the strict pattern falls between 120
# and 400. 90 silently swallowed Items 5, 7, 9 and 12 into the preceding
# section.
MAX_HEADING_CHARS = 150

# Elements whose subtree never renders as body text.
DROP_TAGS = ("hidden", "header")


@dataclass
class Table:
    """A financial table, kept whole rather than chunked.

    Tables are a second retrievable source type alongside chunks, not a
    decoration on them. 46% of the comma-formatted figures in a filing appear
    only inside a table and nowhere in prose, so a chunk-only corpus cannot
    answer a question about product-level revenue at all.

    Kept whole because a 10-K revenue row spans three fiscal years, and chunking
    would split a figure away from the header naming its year.
    """

    table_index: int
    title: str | None
    units: str | None
    markdown: str
    rows: list[list[str]] = field(default_factory=list)
    # Ignores volatile presentation attributes, so a re-extraction can tell
    # whether the table's content actually changed. Required by document_tables.
    source_html_hash: str = ""
    section: str | None = None
    token_count: int = 0
    # Populated by ingest.embed, mirroring Chunk. Nullable so a run that dies
    # mid-embedding leaves rows a re-run can finish.
    embedding: list[float] | None = None

    @property
    def embed_text(self) -> str:
        """What gets embedded.

        The caption carries the semantics — "net sales by category" — while the
        rows carry the figures. A query like "Apple revenue by product" matches
        the caption; "201,183" matches the rows. Embedding the markdown alone
        loses the first, which is the one users actually type.

        Mirrors the search_vector expression on document_tables, so lexical and
        semantic retrieval see the same text.
        """
        parts = [p for p in (self.section, self.title, self.units) if p]
        return "\n\n".join([*parts, self.markdown])


@dataclass
class Section:
    """One Item of the filing, e.g. "Item 1A. Risk Factors"."""

    item: str
    title: str
    text: str
    tables: list[Table] = field(default_factory=list)

    @property
    def heading(self) -> str:
        """Display form, e.g. "Item 1A. Risk Factors".

        Chunks are prefixed with this, so it is also what retrieval matches on.
        """
        item = self.item if self.item.lower().startswith("item") else f"Item {self.item}"
        return f"{item}. {self.title}".strip()


@dataclass
class ExtractedFiling:
    sections: list[Section]
    tables: list[Table]
    markdown: str

BLOCK_TAGS = frozenset({"div", "p"})
_RAW_SCAN_CAP = 5000
_WS = re.compile(r"[\s\u00a0\u2007\u202f]+")

def _local(el) -> str:
    """Tag name without prefix. lxml.html keeps 'ix:hidden' literal, not namespaced."""
    tag = el.tag
    if not isinstance(tag, str):
        return ""
    return tag.rsplit(":", 1)[-1].lower()

def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()

def _drop_tags(root) -> None:
    for el in list(root.iter()):
        if not isinstance(el.tag, str) or _local(el) not in DROP_TAGS:
            continue
        parent = el.getparent()
        if parent is None:
            continue
        if el.tail:
            prev = el.getprevious()
            if prev is not None:
                prev.tail = (prev.tail or "") + el.tail
            else:
                parent.text = (parent.text or "") + el.tail
        parent.remove(el)

def _short_text(el) -> str | None:
    """Normalized text, or None once the subtree is obviously too big to be a heading."""
    parts, total = [], 0
    for piece in el.itertext():
        parts.append(piece)
        total += len(piece)
        if total > _RAW_SCAN_CAP:
            return None
    text = _norm("".join(parts))
    return text if len(text) <= MAX_HEADING_CHARS else None

def _heading_of(el) -> tuple[str, str] | None:
    text = _short_text(el)
    if not text:
        return None
    match = ITEM_HEADING.match(text)
    if match is None:
        return None
    title = match.group(2).strip(" .:\u2014\u2013-")
    if not ITEM_TITLE_WORD.search(title):
        return None
    if el.find(".//a[@href]") is not None:
        return None
    return match.group(1).upper(), title


def _scan(el, events: list) -> bool:
    """Depth-first in document order. Returns True if this subtree held a block.

    Tables are emitted as events rather than matched to sections afterwards.
    A single ordered walk gives "the last heading before this table" for free,
    where matching afterwards would have to rebuild it from document positions
    across two independently parsed trees. Returning early also stops the walk
    descending into the cells, which would otherwise dump uncoalesced figures
    into the section's prose.
    """
    tag = _local(el)

    if tag == "table":
        events.append(("table", el))
        return True

    is_block = tag in BLOCK_TAGS

    if is_block:
        heading = _heading_of(el)
        if heading is not None:
            events.append(("heading", *heading))
            return True

    nested = False
    child_events: list = []
    for child in el:
        if isinstance(child.tag, str) and _scan(child, child_events):
            nested = True

    if is_block and not nested:
        text = _norm("".join(el.itertext()))
        if text:
            events.append(("text", text))
        return True

    events.extend(child_events)
    return nested or is_block

def extract(html_path: str) -> ExtractedFiling:
    """Parse one filing.

    The shape verified across all 25 filings (21-23 items each, every core item
    present, no TOC bleed):

      1. lxml.html.parse, then remove DROP_TAGS by local-name() — the document
         is namespaced XHTML, so plain tag selectors miss.
      2. Walk div/p in document order. A block is a heading when:
         len(text) <= MAX_HEADING_CHARS, it matches ITEM_HEADING, its title
         matches ITEM_TITLE_WORD, and it has no <a href> descendant. All four
         conditions earn their place — dropping any one lets a TOC through or
         loses real sections.
      3. On a match, take that block and skip its whole subtree. Do *not*
         restrict to leaf blocks — Amazon's Part III headings live only on a
         non-leaf div, and leaf-only matching drops Items 10-15 from all five
         Amazon filings. Skipping the subtree is what stops a heading being
         counted once per ancestor.
      4. For body text, take leaf blocks only, or each paragraph is collected
         once per ancestor.
      5. Everything between one heading and the next belongs to that section.

    Sections whose body is "None." (Item 1B is routinely empty) are real and
    should be kept — an analyst asking "did they report unresolved staff
    comments?" deserves the answer, not a gap.
    """
    root = lxml.html.parse(html_path).getroot()
    _drop_tags(root)

    events: list = []
    _scan(root, events)

    sections: list[Section] = []
    buffers: list[list[str]] = []
    tables: list[Table] = []
    for kind, *payload in events:
        if kind == "heading":
            sections.append(Section(item=payload[0], title=payload[1], text=""))
            buffers.append([])
        elif kind == "table":
            # table_index runs filing-wide, not per section: the unique
            # constraint is (document_id, table_index), so restarting per
            # section would collide on the next section's first table.
            table = _build_table(payload[0], len(tables))
            if table is None:
                continue
            table.section = sections[-1].heading if sections else None
            tables.append(table)
            if sections:
                sections[-1].tables.append(table)
        elif sections:
            buffers[-1].append(payload[0])

    for section, parts in zip(sections, buffers, strict=True):
        section.text = "\n\n".join(parts)

    seen = [s.item for s in sections]
    duplicates = {i for i in seen if seen.count(i) > 1}
    if duplicates:
        raise ValueError(
            f"duplicate items {sorted(duplicates)} in {html_path} — table of "
            "contents bled into the section list"
        )
    return ExtractedFiling(
        sections=sections,
        tables=tables,
        markdown=to_markdown(sections),
    )

EXTRACTOR_VERSION = "tables-1"
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2
_VOLATILE_ATTRS = ("style", "class", "id", "colspan", "rowspan")
_PREFIX_GLUE = frozenset({"$", "€", "£", "¥", "("})
_SUFFIX_GLUE = frozenset({"%", ")", "*", "†", "‡", "bps"})
_UNITS = re.compile(
    r"\(?\s*(?:\$\s*)?in\s+(thousands|millions|billions)[^)]*\)?", re.IGNORECASE)

def _coalesce_row(cells: list[str]) -> list[str]:
    """['$','201,183','','(2)','%'] -> ['$201,183', '(2)%']"""
    out: list[str] = []
    glue_next = False
    for cell in (c for c in (_norm(x) for x in cells) if c):
        if out and (glue_next or cell in _SUFFIX_GLUE):
            out[-1] += cell
        else:
            out.append(cell)
        glue_next = cell in _PREFIX_GLUE
    return out

def _rows_of(table) -> list[list[str]]:
    rows = []
    for tr in table.iter():
        if _local(tr) != "tr":
            continue
        cells = [_norm("".join(td.itertext()))
                 for td in tr if _local(td) in ("td", "th")]
        row = _coalesce_row(cells)
        if row:
            rows.append(row)
    return rows

def _title_of(table) -> str | None:
    node = table
    while node is not None:
        prev = node.getprevious()
        while prev is not None:
            text = _norm("".join(prev.itertext()))
            if text and len(text) <= 200:
                return text
            prev = prev.getprevious()
        node = node.getparent()
    return None

def _units_of(title: str | None, table_text: str) -> str | None:
    for haystack in (title or "", table_text):
        match = _UNITS.search(haystack)
        if match:
            return _norm(match.group(0)).strip("()")
    return None

def _table_hash(table) -> str:
    clone = copy.deepcopy(table)
    for node in clone.iter():
        if isinstance(node.tag, str):
            for attr in _VOLATILE_ATTRS:
                node.attrib.pop(attr, None)
    raw = _norm(lxml.html.tostring(clone, encoding="unicode", method="html"))
    return hashlib.sha256(f"{EXTRACTOR_VERSION}\x00{raw}".encode()).hexdigest()


def _to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    def esc(cell: str) -> str:
        return cell.replace("|", "\\|")

    lines = ["| " + " | ".join(esc(c) for c in padded[0]) + " |",
             "| " + " | ".join("---" for _ in range(width)) + " |"]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in padded[1:]]
    return "\n".join(lines)
       

def _build_table(el, table_index: int) -> Table | None:
    """Build one Table from a <table> element, or None if it is not tabular.

    Coalescing is the real work: a currency row arrives as
    ["$", "201,183", "", "(2)", "%"] and has to render as "$201,183 | (2)%".
    These are the figures an analyst checks the answer against.

    `source_html_hash` ignores volatile presentation attributes so a
    re-extraction can tell whether the content actually changed.
    """
    # A table wrapping another is layout, not data; the inner one is the table.
    if any(_local(d) == "table" for d in el.iterdescendants()):
        return None
    rows = _rows_of(el)
    if len(rows) < MIN_TABLE_ROWS or max((len(r) for r in rows), default=0) < MIN_TABLE_COLS:
        return None
    title = _title_of(el)
    return Table(
        table_index=table_index,
        title=title,
        units=_units_of(title, _norm("".join(el.itertext()))),
        markdown=_to_markdown(rows),
        rows=rows,
        source_html_hash=_table_hash(el),
    )


def extract_tables(html_path: str) -> list[Table]:
    """Every table in a filing, in document order.

    Thin wrapper: tables are built during the single walk in `extract`, so this
    parses the file once rather than a second time alongside it.
    """
    return extract(html_path).tables
     

_MD_HEADING = re.compile(r"^##\s+(.+)$")

def to_markdown(sections: list[Section]) -> str:
    """Render sections to the Markdown stored on `source_documents`.

    This is what gets re-chunked later, so it must stand alone
    — the downloaded HTML is gitignored and may not exist on the machine that
    re-chunks.
    """
    blocks: list[str] = []
    for section in sections:
        blocks.append(f"## {section.heading}")
        if section.text:
            blocks.append(section.text)
        for table in section.tables:
            label = f"**Table {table.table_index}"
            if table.title:
                label += f": {table.title}"
            label += "**"
            if table.units:
                label += f" *({table.units})*"
            blocks.append(label)
            blocks.append(table.markdown)
    return "\n\n".join(blocks) + "\n"
