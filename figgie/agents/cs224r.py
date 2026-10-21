"""The five CS224R baselines (Yang, "Reinforcement Learning for Figgie", §5).

Reimplemented from the report's prose; no code was released. Every guessed
detail is tagged GUESS and collected in DESIGN.md.
All of them send exactly one primitive per tick, matching the report's
257-way action space.
"""
from __future__ import annotations

import math
import random

from ..belief import eb, goal_posterior
from ..engine import ASK, BID, HIT, LIFT, NOOP
from .. import actions as A


class RandomAgent:
    """Uniform over the legal action mask."""
    name = "Random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def reset(self, obs):
        pass

    def act(self, obs):
        legal = A.legal_indices(obs)
        return A.decode(self.rng.choice(legal))


class FairValue:
    """v(s) = 10 * P(goal=s | initial hand). Lift asks <= v-edge, hit bids
    >= v+edge, else post a bid on the best suit or an ask on the worst."""
    name = "FairValue"

    def __init__(self, edge=2.0, gate=0.0, seed=None):
        self.edge, self.gate = edge, gate
        self.rng = random.Random(seed)

    def reset(self, obs):
        self.p = goal_posterior(tuple(obs.initial_hand))
        self.v = [10 * x for x in self.p]

    def act(self, obs):
        if max(self.p) < self.gate:
            return (NOOP,)
        v, e, hand = self.v, self.edge, obs.hand
        for s in range(4):
            a = obs.best_ask(s, True)
            if a and a[0] <= v[s] - e and obs.cash >= a[0]:
                return (LIFT, s)
            b = obs.best_bid(s, True)
            if b and b[0] >= v[s] + e and hand[s] > 0:
                return (HIT, s)
        # GUESS: coin-flip between the bid and the ask; prices round(v -/+ edge)
        # clipped to [1, 30]; ask on the lowest-value suit we actually hold.
        if self.rng.random() < 0.5:
            s = max(range(4), key=lambda k: v[k])
            px = max(1, min(30, round(v[s] - e)))
            if obs.bids[obs.me][s] != px and obs.cash >= px:
                return (BID, s, px)
        held = [k for k in range(4) if hand[k] > 0]
        if held:
            s = min(held, key=lambda k: v[k])
            px = max(1, min(30, round(v[s] + e)))
            if obs.asks[obs.me][s] != px:
                return (ASK, s, px)
        return (NOOP,)


class Conservative(FairValue):
    """FairValue gated on max P(goal) >= 0.4, edge 4."""
    name = "Conservative"

    def __init__(self, seed=None):
        super().__init__(edge=4.0, gate=0.4, seed=seed)


class MarketMaker:
    """Two-sided quotes at FairValue +/- 3 on whichever own slot is empty.
    Never lifts or hits."""
    name = "MarketMaker"

    def __init__(self, half_spread=3, seed=None):
        self.h = half_spread
        self.rng = random.Random(seed)

    def reset(self, obs):
        self.v = [10 * x for x in goal_posterior(tuple(obs.initial_hand))]

    def act(self, obs):
        me = obs.me
        slots = [(s, side) for s in range(4) for side in (BID, ASK)]
        self.rng.shuffle(slots)   # GUESS: random empty slot each tick
        for s, side in slots:
            if side == BID and not obs.bids[me][s]:
                px = max(1, min(30, math.floor(self.v[s] - self.h)))
                a = obs.best_ask(s, True)
                if obs.cash >= px and not (a and a[0] <= px):
                    return (BID, s, px)
            if side == ASK and not obs.asks[me][s] and obs.hand[s] > 0:
                px = max(1, min(30, math.ceil(self.v[s] + self.h)))
                b = obs.best_bid(s, True)
                if not (b and b[0] >= px):
                    return (ASK, s, px)
        return (NOOP,)


class CardCounter:
    """Lower-bound card-counting matrix L[p][s] from DiSilvio et al. / CS224R."""

    def __init__(self, obs):
        self.me, self.n = obs.me, obs.n
        self.L = [[0] * 4 for _ in range(obs.n)]
        self.L[self.me] = list(obs.initial_hand)
        self.seen = 0

    def update(self, obs):
        tr = obs.trades
        for t in tr[self.seen:]:
            if t.seller != self.me:
                self.L[t.seller][t.suit] = max(self.L[t.seller][t.suit], 1) - 1
            if t.buyer != self.me:
                self.L[t.buyer][t.suit] += 1
        self.seen = len(tr)
        self.L[self.me] = list(obs.hand)
        return tuple(sum(self.L[p][s] for p in range(self.n)) for s in range(4))


class Bayesian:
    """DiSilvio et al. (2021) fundamentalist as ported in CS224R: card-counting
    posterior over 12 decks, geometric majority-bonus valuation (r=2), and a
    randomised quoting rule."""
    name = "Bayesian"

    def __init__(self, r=2.0, seed=None):
        self.r = r
        self.rng = random.Random(seed)

    def reset(self, obs):
        self.cc = CardCounter(obs)

    def values(self, obs):
        c = self.cc.update(obs)
        return c, [eb(c, s, obs.hand[s], self.r) for s in range(4)]

    def act(self, obs):
        c, _ = self.values(obs)
        s = self.rng.randrange(4)
        hand = obs.hand
        if self.rng.random() < 0.5:
            v = eb(c, s, hand[s], self.r)
            # GUESS: buy price ~ U{1..floor(v)} (DiSilvio: uniform below value).
            hi = min(30, math.floor(v))
            if hi < 1:
                return (NOOP,)
            px = self.rng.randint(1, hi)
            a = obs.best_ask(s, True)
            if a and a[0] <= px:
                return (LIFT, s) if obs.cash >= a[0] else (NOOP,)
            if obs.asks[obs.me][s] and obs.asks[obs.me][s] <= px:
                return (NOOP,)
            return (BID, s, px) if obs.cash >= px else (NOOP,)
        if hand[s] == 0:
            return (NOOP,)
        v = eb(c, s, hand[s] - 1, self.r)   # value of the card we would give up
        # GUESS: sell price ~ U{ceil(v)..ceil(v)+10}, clipped to [1, 30].
        lo = max(1, math.ceil(v))
        if lo > 30:
            return (NOOP,)
        px = self.rng.randint(lo, min(30, lo + 10))
        b = obs.best_bid(s, True)
        if b and b[0] >= px:
            return (HIT, s)
        if obs.bids[obs.me][s] and obs.bids[obs.me][s] >= px:
            return (NOOP,)
        return (ASK, s, px)
