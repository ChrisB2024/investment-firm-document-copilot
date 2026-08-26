# Test fixtures

Miniature filings, each reproducing one quirk found in the real corpus. The
real filings live in `data/downloads/` and are gitignored, so the fast suite
cannot depend on them — these stand in.

Every quirk here cost a real bug. Do not "tidy" the markup: the awkwardness is
the point.

| file | reproduces |
| ---- | ---------- |
| `apple_style.htm` | leaf-`<div>` headings, linked TOC, `ix:hidden` XBRL facts, split table cells, a 105-char heading, prose cross-referencing an Item |
| `amazon_style.htm` | heading text only on a *non-leaf* `<div>`; leaves hold a bare "Item 10." with no title |
| `msft_style.htm` | dot-leader TOC with no `<a href>`, in both a `<table>` and `<div>`s; a comment sitting where `_title_of` walks |

## Placement is part of the fixture

Two of these were originally written so that the rule they pin could not fail.
Both are marked with a comment in the file; do not "simplify" them back.

- The `ix:hidden` fact in `apple_style` has to sit **inside a body `<div>`**.
  `_scan` only emits text from `div`/`p`, so a hidden fact in the header cannot
  leak into a section no matter what `DROP_TAGS` says — a test on it passes
  whether the rule is there or not.
- `msft_style` needs its dot-leader TOC in **`<div>`s, not only a `<table>`**.
  `_scan` returns early at `<table>` without descending, so entries inside one
  never reach `_heading_of` and `ITEM_TITLE_WORD` is never consulted.

Both were caught by mutation testing: deleting the rule from `extract.py` and
checking that a test goes red. A fixture that survives that deletion is
decoration.
