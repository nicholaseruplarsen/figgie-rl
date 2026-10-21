"""The CS224R 257-way discrete action space.

0            no_op
1..120       place_bid(s, p)  index 1 + 30*s + (p-1)
121..240     place_ask(s, p)  index 121 + 30*s + (p-1)
241..244     cancel_bid(s)
245..248     cancel_ask(s)
249..252     lift(s)
253..256     hit(s)
"""
from __future__ import annotations

import numpy as np

from .engine import ASK, BID, CANCEL_ASK, CANCEL_BID, HIT, LIFT, NOOP

N_ACTIONS = 257
MAXP = 30


def decode(i: int):
    if i == 0:
        return (NOOP,)
    if i <= 120:
        s, p = divmod(i - 1, MAXP)
        return (BID, s, p + 1)
    if i <= 240:
        s, p = divmod(i - 121, MAXP)
        return (ASK, s, p + 1)
    k, s = divmod(i - 241, 4)
    return ((CANCEL_BID, CANCEL_ASK, LIFT, HIT)[k], s)


def encode(act) -> int:
    kind = act[0]
    if kind == NOOP:
        return 0
    s = act[1]
    if kind == BID:
        return 1 + MAXP * s + act[2] - 1
    if kind == ASK:
        return 121 + MAXP * s + act[2] - 1
    return 241 + 4 * (CANCEL_BID, CANCEL_ASK, LIFT, HIT).index(kind) + s


def legal_mask(obs) -> np.ndarray:
    """Mask of actions the engine would accept (no self-cross, affordability,
    card in hand, an opposing quote exists for lift/hit, something to cancel)."""
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[0] = True
    me, cash, hand = obs.me, obs.cash, obs.hand
    pr = np.arange(1, MAXP + 1)
    for s in range(4):
        own_ask, own_bid = obs.asks[me][s], obs.bids[me][s]
        ok = pr <= cash
        if own_ask:
            ok &= pr < own_ask
        m[1 + MAXP * s: 1 + MAXP * (s + 1)] = ok
        if hand[s] > 0:
            ok = np.ones(MAXP, dtype=bool)
            if own_bid:
                ok &= pr > own_bid
            m[121 + MAXP * s: 121 + MAXP * (s + 1)] = ok
        m[241 + s] = own_bid > 0
        m[245 + s] = own_ask > 0
        a = obs.best_ask(s, True)
        m[249 + s] = a is not None and a[0] <= cash
        b = obs.best_bid(s, True)
        m[253 + s] = b is not None and hand[s] > 0
    return m


def legal_indices(obs):
    return np.flatnonzero(legal_mask(obs)).tolist()
