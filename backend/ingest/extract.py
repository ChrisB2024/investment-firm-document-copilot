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

import re
from dataclasses import dataclass, field

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
    """A financial table, kept whole rather than chunked."""

    table_index: int
    title: str | None
    units: str | None
    markdown: str
    rows: list[list[str]] = field(default_factory=list)


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
    raise NotImplementedError


def extract_tables(html_path: str) -> list[Table]:
    """Pull tables out whole, for `document_tables`.

    Coalescing is the real work: a currency row arrives as
    ["$", "201,183", "", "(2)", "%"] and has to render as "$201,183 (2)%".
    Decide the rule once and apply it consistently, because these numbers are
    what analysts will check the answer against.

    Carry `source_html_hash` (the model requires it) so a re-extraction can tell
    whether a table actually changed.
    """
    raise NotImplementedError


def to_markdown(sections: list[Section]) -> str:
    """Render sections to the Markdown stored on `source_documents`.

    This is what gets re-chunked later, so it must stand alone
    — the downloaded HTML is gitignored and may not exist on the machine that
    re-chunks.
    """
    raise NotImplementedError
