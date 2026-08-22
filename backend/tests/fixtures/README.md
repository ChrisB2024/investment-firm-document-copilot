# Test fixtures

Miniature filings, each reproducing one quirk found in the real corpus. The
real filings live in `data/downloads/` and are gitignored, so the fast suite
cannot depend on them — these stand in.

Every quirk here cost a real bug. Do not "tidy" the markup: the awkwardness is
the point.

| file | reproduces |
| ---- | ---------- |
| `apple_style.htm` | leaf-`<div>` headings, linked TOC, `ix:hidden` XBRL facts, split table cells, a heading over 90 chars, prose cross-referencing an Item |
| `amazon_style.htm` | heading text only on a *non-leaf* `<div>`; leaves hold a bare "Item 10." with no title |
| `msft_style.htm` | dot-leader TOC with no `<a href>`, which no link filter catches |
