"""Goal-suit inference shared by the agents.

All posteriors are over the 12 deck compositions in engine.DECKS (uniform
prior), with multivariate-hypergeometric likelihood prod_s C(d_s, c_s).
"""
from __future__ import annotations

from functools import lru_cache
from math import comb

from .engine import DECKS, DECK_GOAL, POT, CARD_VALUE


@lru_cache(maxsize=None)
def deck_posterior(counts: tuple) -> tuple:
    """P(deck=i | observed suit counts) for the 12 decks."""
    w = [0.0] * 12
    for i, d in enumerate(DECKS):
        x = 1
        for s in range(4):
            if counts[s] > d[s]:
                x = 0
                break
            x *= comb(d[s], counts[s])
        w[i] = x
    tot = sum(w)
    if tot == 0:
        return tuple([1 / 12] * 12)
    return tuple(x / tot for x in w)


@lru_cache(maxsize=None)
def goal_posterior(counts: tuple) -> tuple:
    m = deck_posterior(counts)
    p = [0.0] * 4
    for i in range(12):
        p[DECK_GOAL[i]] += m[i]
    return tuple(p)


# ---- DiSilvio et al. (2021) fundamentalist valuation, as written in CS224R §5 ----

def majority_threshold(goal_size: int) -> int:
    return goal_size // 2 + 1          # 5 for an 8-card goal suit, 6 for 10


@lru_cache(maxsize=None)
def eb(counts: tuple, j: int, jn: int, r: float = 2.0) -> float:
    """Expected value of acquiring one more card of suit j when holding jn,
    given card-count vector `counts` (CS224R eq. for e_b)."""
    m = deck_posterior(counts)
    v = 0.0
    for i, d in enumerate(DECKS):
        if DECK_GOAL[i] != j or m[i] == 0:
            continue
        x = majority_threshold(d[j])
        B = POT - CARD_VALUE * d[j]
        vm = B * (1 - r) / (1 - r ** x) * r ** jn if jn < x else 0.0
        v += m[i] * (CARD_VALUE + vm)
    return v
