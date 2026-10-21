"""Ports of revilo-25/Figgie-bot (no license file; reimplemented, not vendored).

Logic follows figgie/{inference,valuation,bots,ablations}.py at the commit in
third_party/Figgie-bot. Differences from the original:
- Their engine deals the goal suit as 8 or 10 cards with probability 1/2 each
  (16 compositions); standard Figgie and our engine use 1/3 vs 2/3 (12 decks).
  The ports keep their 16-hypothesis prior (that is part of the agent), and
  `prior="standard"` swaps in the correct one.
- Their sim gives each player one multi-order turn per tick for 200 ticks and
  never wipes the book on a trade; ours runs 240 ticks with canonical wipes.
- FlowMM's fitted weights are read from third_party/.../flow_model.json.
"""
from __future__ import annotations

import json
import math
import os
import random
from functools import lru_cache
from math import comb, prod

import numpy as np

from ..engine import ASK, BID, CANCEL_ASK, CANCEL_BID, PARTNER, POT, CARD_VALUE

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLOW_MODEL = os.path.join(_ROOT, "third_party/Figgie-bot/figgie-bot-x/figgie/flow_model.json")


def _hyps(prior):
    hyps = []
    for common in range(4):
        goal = PARTNER[common]
        others = [s for s in range(4) if s not in (common, goal)]
        for g in (8, 10):
            rest = (10, 10) if g == 8 else (8, 10)
            for a, b in ((0, 1), (1, 0)):
                sizes = [0] * 4
                sizes[common], sizes[goal] = 12, g
                sizes[others[a]], sizes[others[b]] = rest
                hyps.append((goal, tuple(sizes)))
    if prior == "standard":   # drop the duplicated (10,10) layouts -> 12 decks
        hyps = list(dict.fromkeys(hyps))
    return tuple(hyps)


HYPS = {"source": _hyps("source"), "standard": _hyps("standard")}


@lru_cache(maxsize=None)
def goal_posterior(init: tuple, prior="source"):
    w = [0.0] * 4
    for goal, sizes in HYPS[prior]:
        like = 1
        for s in range(4):
            like *= comb(sizes[s], init[s])
        w[goal] += like
    tot = sum(w)
    return tuple(x / tot for x in w)


def initial_hand(obs):
    """Their reconstruction: current hand minus buys plus sells."""
    h, me = list(obs.hand), obs.me
    for tr in obs.trades:
        if tr.buyer == me:
            h[tr.suit] -= 1
        if tr.seller == me:
            h[tr.suit] += 1
    return tuple(h)


# ---- valuation.py ----------------------------------------------------------
def _comps(m, k, cap):
    if k == 1:
        if m <= cap:
            yield (m,)
        return
    for x in range(min(m, cap) + 1):
        for rest in _comps(m - x, k - 1, cap):
            yield (x,) + rest


@lru_cache(maxsize=None)
def win_share(c, m, n, h_opp):
    total = comb(h_opp * (n - 1), m)
    acc = 0
    for counts in _comps(m, n - 1, h_opp):
        top = max(counts)
        if c > top:
            share = 1.0
        elif c == top:
            share = 1.0 / (1 + counts.count(top))
        else:
            continue
        acc += prod(comb(h_opp, x) for x in counts) * share
    return acc / total


def payoff(c, n_goal, n, h_opp):
    return CARD_VALUE * c + (POT - CARD_VALUE * n_goal) * win_share(c, n_goal - c, n, h_opp)


@lru_cache(maxsize=None)
def card_values(init: tuple, cur: tuple, n=4, prior="source"):
    h_opp = 40 // n
    hyps = []
    for goal, sizes in HYPS[prior]:
        if any(sizes[s] < cur[s] for s in range(4)):
            continue
        w = prod(comb(sizes[s], init[s]) for s in range(4))
        if w:
            hyps.append((goal, sizes, w))
    tot = sum(w for *_, w in hyps)
    res = []
    for s in range(4):
        c, buy, sell = cur[s], 0.0, 0.0
        for goal, sizes, w in hyps:
            if goal != s:
                continue
            w /= tot
            ng = sizes[s]
            base = payoff(c, ng, n, h_opp)
            up = payoff(c + 1, ng, n, h_opp) if c + 1 <= ng else base
            dn = payoff(c - 1, ng, n, h_opp) if c >= 1 else base
            buy += w * (up - base)
            sell += w * (base - dn)
        res.append((buy, sell))
    return tuple(res)


# ---- adapter ---------------------------------------------------------------
def _prim(a):
    if a[0] == "cancel":
        return (CANCEL_BID if a[2] == "bid" else CANCEL_ASK, a[1])
    return (BID if a[0] == "bid" else ASK, a[1], int(min(30, a[2])))


class _FB:
    def __init__(self, seed=None, prior="source"):
        self.rng = random.Random(seed)
        self.prior = prior

    def reset(self, obs):
        pass

    def act(self, obs):
        return [_prim(a) for a in self.decide(obs)]


class FBRandom(_FB):
    name = "fb.Random"

    def decide(self, obs):
        s = self.rng.randrange(4)
        ref = obs.last_price[s] if obs.last_price[s] is not None else 8
        if self.rng.random() < 0.5:
            return [("bid", s, max(1, ref + self.rng.randint(-4, 2)))]
        return [("offer", s, max(1, ref + self.rng.randint(-2, 4)))]


class ValueMM(_FB):
    """The shared quoting engine of PotMM and all ablations."""
    edge, take_edge = 1.0, 1.0

    def values(self, obs):
        raise NotImplementedError

    def decide(self, obs):
        me, vals, acts = obs.me, self.values(obs), []
        for s in range(4):
            buy_v, sell_v = vals[s]
            ask, bid = obs.best_ask(s), obs.best_bid(s)
            if ask and ask[1] != me and ask[0] <= buy_v - self.take_edge and obs.cash >= ask[0]:
                acts.append(("bid", s, ask[0]))
                continue
            if bid and bid[1] != me and bid[0] >= sell_v + self.take_edge and obs.hand[s] > 0:
                acts.append(("offer", s, bid[0]))
                continue
            b, a = math.floor(buy_v - self.edge), math.ceil(sell_v + self.edge)
            if b >= a:
                mid = (buy_v + sell_v) / 2
                b, a = math.floor(mid - self.edge), math.ceil(mid + self.edge)
            b, a = max(1, b), max(2, a)
            acts += [("cancel", s, "bid"), ("cancel", s, "offer")]
            if b <= obs.cash:
                acts.append(("bid", s, b))
            if obs.hand[s] > 0:
                acts.append(("offer", s, a))
        return acts


class PotMM(ValueMM):
    name = "fb.PotMM"

    def values(self, obs):
        return card_values(initial_hand(obs), tuple(obs.hand), obs.n, self.prior)


class HandMM(ValueMM):
    """Original HandMM: fair = 10 P(goal) - 0.25 * (hand - init); same quoting
    except no b>=a re-centring (cannot trigger with buy == sell)."""
    name = "fb.HandMM"
    skew = 0.25

    def values(self, obs):
        init = initial_hand(obs)
        post = goal_posterior(init, self.prior)
        return [(f, f) for s in range(4) for f in [CARD_VALUE * post[s] - self.skew * (obs.hand[s] - init[s])]]


class FlowMM(PotMM):
    """PotMM rescaled by the conditional-logit order-flow posterior."""
    name = "fb.FlowMM"
    BUCKETS = (0, 30, 80, 140)
    _model = None

    @classmethod
    def model(cls):
        if cls._model is None:
            with open(FLOW_MODEL) as f:
                cls._model = {int(k): np.array(v) for k, v in json.load(f).items()}
        return cls._model

    def flow_posterior(self, obs):
        init = initial_hand(obs)
        post = goal_posterior(init, self.prior)
        cnt = [0] * 4
        for tr in obs.trades:
            cnt[tr.suit] += 1
        last = [obs.last_price[s] or 0 for s in range(4)]
        bb = [obs.best_bid(s, True) for s in range(4)]
        ba = [obs.best_ask(s, True) for s in range(4)]
        X = np.array([[math.log(max(post[s], 1e-9)) for s in range(4)], last,
                      [b[0] if b else 0 for b in bb], [a[0] if a else 10 for a in ba], cnt], dtype=float).T
        X[:, 1:] -= X[:, 1:].mean(axis=0)
        k = max(i for i, b in enumerate(self.BUCKETS) if obs.tick >= b)
        z = X @ self.model()[k]
        e = np.exp(z - z.max())
        return e / e.sum()

    def values(self, obs):
        init = initial_hand(obs)
        ph, pf = goal_posterior(init, self.prior), self.flow_posterior(obs)
        v = super().values(obs)
        return [(v[s][0] * pf[s] / max(ph[s], 1e-9), v[s][1] * pf[s] / max(ph[s], 1e-9)) for s in range(4)]


class ConstVal(ValueMM):
    """No-information control: every card valued at 8.8, no skew."""
    name = "fb.ConstVal"

    def __init__(self, c=8.8, **kw):
        super().__init__(**kw)
        self.c = c

    def values(self, obs):
        return [(self.c, self.c)] * 4


class PotValShuffled(ValueMM):
    """PotMM price levels with suit labels permuted per game (information-scrambled control)."""
    name = "fb.PotValShuffled"

    def reset(self, obs):
        self.perm = list(range(4))
        random.Random(hash(tuple(obs.initial_hand)) ^ obs.me).shuffle(self.perm)

    def values(self, obs):
        v = card_values(initial_hand(obs), tuple(obs.hand), obs.n, self.prior)
        return [v[p] for p in self.perm]


class FixedSpread(_FB):
    name = "fb.FixedSpread"

    def decide(self, obs):
        s = self.rng.randrange(4)
        acts = [("cancel", s, "bid"), ("cancel", s, "offer"), ("bid", s, 2)]
        if obs.hand[s] > 0:
            acts.append(("offer", s, 8))
        return acts


class MomentumBot(_FB):
    name = "fb.Momentum"

    def decide(self, obs):
        cnt = [0] * 4
        for tr in obs.trades[-12:]:
            cnt[tr.suit] += 1
        hot = max(range(4), key=lambda s: (cnt[s], self.rng.random()))
        ask = obs.best_ask(hot, True)
        acts = []
        if ask and ask[0] <= 12 and obs.cash >= ask[0]:
            acts.append(("bid", hot, ask[0]))
        cold = min(range(4), key=lambda s: (cnt[s], self.rng.random()))
        bid = obs.best_bid(cold, True)
        if bid and obs.hand[cold] > 0:
            acts.append(("offer", cold, bid[0]))
        return acts
