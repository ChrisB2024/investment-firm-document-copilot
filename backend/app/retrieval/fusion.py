"""Reciprocal Rank Fusion over the two ranked lists.

RRF combines rankings without needing their scores to be comparable. A cosine
distance and a `ts_rank_cd` score share no scale, no range and no units, and
adding or normalising them invents a relationship that is not there. RRF throws
the scores away and keeps only position:

    score(d) = sum over lists of 1 / (k + rank(d))

Measured on the ingested corpus with top-20 from each search, not assumed:

- **The two arms genuinely disagree.** Mean overlap across six brief-style
  questions is 4 of 20, and "geographic revenue exposure by country" overlaps
  in *nothing*. That is the case for fusing at all — if the lists agreed there
  would be no second arm worth running.
- **k is not worth tuning.** At k = 10, 60 and 200 the top-10 membership is
  identical for every question tried; only the ordering shifts, and only at
  k = 10. Take the 60 from the original paper and spend the time on retrieval
  quality instead.
- **Ties are the normal case, not an edge case.** Where the lists do not
  overlap, every document scores 1/(k + rank) exactly once, so the two lists'
  rank-1 entries tie, their rank-2 entries tie, and so on. For the zero-overlap
  question only 5 of the top 10 scores were distinct. A tie-break rule is
  therefore load-bearing: without one, ordering falls out of dict insertion
  order, which means it depends on the order the caller passed the lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from app.retrieval.queries import Hit, SourceType

_Key = tuple[SourceType, UUID]

# From Cormack et al. 2009, where 60 was found to work across TREC collections
# without per-collection tuning. Measured above: this corpus is insensitive to
# it, so the constant is a default to leave alone rather than a knob.
RRF_K = 60


@dataclass
class FusedHit:
    """One result after fusion, with enough provenance to explain its rank.

    `contributions` is not debug clutter. Phase 3 asks which of the brief's ten
    questions fail *and why* — "the passage was never retrieved" and "it was
    retrieved by the vector arm at rank 18 and fusion buried it" are different
    problems with different fixes, and the ranks are the only way to tell them
    apart.
    """

    source_type: SourceType
    row_id: UUID
    document_id: UUID
    score: float
    rank: int
    # arm name -> rank in that arm, e.g. {"vector": 3, "text": 11}
    contributions: dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> _Key:
        """Mirrors Hit.key, so a fused result identifies a row the same way."""
        return (self.source_type, self.row_id)


def fuse(
    arms: dict[str, list[Hit]],
    *,
    limit: int,
    k: int = RRF_K,
) -> list[FusedHit]:
    """Fuse named ranked lists into one, best first.

    Keyed on `Hit.key` — (source_type, row_id) — because a chunk id and a table
    id are both UUIDs and could in principle collide. Fusing on row_id alone
    would silently merge a table into a chunk.

    Take `arms` as a dict rather than *args so the arm name lands in
    `contributions` without the caller having to remember which position was
    which.

    Two properties worth asserting in a test rather than trusting, because an
    implementation can get them backwards while still looking plausible:

      - A document at rank 10 in *both* arms must beat one at rank 1 in a single
        arm: 2/(60+10) = 0.0286 against 1/(60+1) = 0.0164. That is the entire
        point of RRF, and an implementation that sums scores instead of ranks,
        or that overwrites rather than accumulates, still looks plausible while
        getting this backwards.
      - Passing the same arms in a different order must produce identical
        output. That is what the tie-break rule below buys, and it is the
        cheapest possible regression test for it.
    """
    if limit < 1:
        raise ValueError(f"limit={limit} is not a retrieval")

    scores: dict[_Key, float] = {}
    contributions: dict[_Key, dict[str, int]] = {}
    best_rank: dict[_Key, int] = {}
    identity: dict[_Key, Hit] = {}

    for arm in sorted(arms):
        seen: set[_Key] = set()
        for hit in arms[arm]:
            key = hit.key
            if key in seen:
                raise ValueError(
                    f"arm {arm!r} ranks {key} at two positions; a ranked list "
                    f"cannot place one row twice"
                )
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + hit.rank)
            contributions.setdefault(key, {})[arm] = hit.rank
            best_rank[key] = min(best_rank.get(key, hit.rank), hit.rank)
            identity.setdefault(key, hit)

    ordered = sorted(
        scores,
        key=lambda key: (-scores[key], best_rank[key], str(key[1]), key[0]),
    )

    return [
        FusedHit(
            source_type=identity[key].source_type,
            row_id=identity[key].row_id,
            document_id=identity[key].document_id,
            score=scores[key],
            rank=i,
            contributions=contributions[key],
        )
        for i, key in enumerate(ordered[:limit], start=1)
    ]


# Ties break on best single rank, then on (row_id, source_type) as a total
# order. Best-single-rank stays inside RRF's own logic — position is the only
# thing it knows — where breaking toward a favoured arm would make fusion take
# the side RRF exists to avoid. The trailing key comparison is arbitrary but
# total, which is the property that matters: with zero overlap only 5 of the top
# 10 scores are distinct, so without it the ordering would fall out of dict
# insertion order and change with the order the caller built `arms` in.
#
# The arms are unweighted, and there is deliberately no parameter for it.
# Weighting is a real lever, not a no-op: at 2:1 toward either arm, between 5
# and 10 of the top 10 survive, so a weight can move half the results. That is
# precisely why it is not set here — a lever that strong, aimed by intuition,
# degrades retrieval exactly as easily as it improves it, and nothing measured
# so far says which way to point it. The evidence that could is Phase 3's own
# exit criterion: run the ten brief questions and see which passages fail to
# surface. Add the parameter then, with the numbers that justify the value.
