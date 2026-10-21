"""Ports of the mneha05/figment agents (MIT, Copyright (c) 2026 mneha05).

Decision logic is transcribed from figment/agents/*.py and figment/belief.py at
the commit cloned into third_party/figment; the only change is the adapter
that builds figment's Observation from our engine state and turns figment's
single limit Order into one engine primitive. Figment's own engine lets one
random player act per step; on the shared engine every player acts every tick.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import permutations
from math import comb

import numpy as np

from ..engine import ASK, BID, NOOP, PARTNER

# ---- figment/belief.py ---------------------------------------------------
_ASSIGNMENTS = tuple(dict.fromkeys(permutations((12, 10, 10, 8))))
EXPECTED_GOAL_CARDS = 8 / 3 + 20 / 3
_SUIT_MAP = (0, 1, 2, 3)   # figment: SPADES, CLUBS, HEARTS, DIAMONDS == ours


def posterior_goal(hand) -> np.ndarray:
    w = np.zeros(len(_ASSIGNMENTS))
    for i, n in enumerate(_ASSIGNMENTS):
        x = 1.0
        for s in range(4):
            if hand[s] > n[s]:
                x = 0.0
                break
            x *= comb(n[s], int(hand[s]))
        w[i] = x
    if w.sum() == 0:
        w[:] = 1.0
    w /= w.sum()
    pc = np.zeros(4)
    for i, n in enumerate(_ASSIGNMENTS):
        pc[int(np.argmax(n))] += w[i]
    return np.array([pc[PARTNER[s]] for s in range(4)])


def belief_entropy(p) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


def card_values(p_goal, pot=200):
    return p_goal * (pot / EXPECTED_GOAL_CARDS)


# ---- adapter ---------------------------------------------------------------
@dataclass
class Quote:
    price: int
    player: int


@dataclass
class FObs:
    me: int
    hand: np.ndarray
    cash: int
    best_bid: list
    best_ask: list
    net_bought: np.ndarray
    net_sold: np.ndarray
    last_price: list
    step: int
    total_steps: int


def figment_obs(obs) -> FObs:
    nb, ns = np.zeros(4, dtype=int), np.zeros(4, dtype=int)
    for t in obs.trades:
        if t.buyer_aggressed:
            nb[t.suit] += 1
        else:
            ns[t.suit] += 1
    bb = [obs.best_bid(s) for s in range(4)]
    ba = [obs.best_ask(s) for s in range(4)]
    return FObs(obs.me, np.array(obs.hand), int(obs.cash),
                [Quote(*q) if q else None for q in bb], [Quote(*q) if q else None for q in ba],
                nb, ns, list(obs.last_price), obs.tick, obs.ticks)


def to_primitive(order):
    """order = (suit, price, is_buy) or None."""
    if order is None:
        return (NOOP,)
    s, px, is_buy = order
    return (BID if is_buy else ASK, s, int(min(30, max(1, px))))


class _Figment:
    def act(self, obs):
        return to_primitive(self.decide(figment_obs(obs)))


# ---- figment/agents/fundamental.py ---------------------------------------
class Fundamental(_Figment):
    name = "fig.Fundamental"

    def __init__(self, edge=3.0):
        self.edge = edge

    def reset(self, obs):
        self.values = card_values(posterior_goal(np.array(obs.initial_hand)))

    def decide(self, o: FObs):
        for s in range(4):
            v = self.values[s]
            ask = o.best_ask[s]
            if ask is not None and ask.player != o.me and ask.price <= v - self.edge and o.cash >= ask.price:
                return (s, ask.price, True)
            bid = o.best_bid[s]
            if bid is not None and bid.player != o.me and bid.price >= v + self.edge and o.hand[s] >= 1:
                return (s, bid.price, False)
        s = int(np.argmax(self.values))
        v = self.values[s]
        if o.hand[s] >= 1 and (o.step + s) % 2 == 0:
            return (s, max(1, round(v + self.edge)), False)
        bid_price = max(1, round(v - self.edge))
        if o.cash >= bid_price:
            return (s, bid_price, True)
        return None


# ---- figment/agents/market_maker.py --------------------------------------
@dataclass
class MMParams:
    base_spread: float = 2.5
    risk_aversion: float = 1.1
    uncertainty_weight: float = 1.6
    signal_weight: float = 0.35
    take_edge: float = 2.0
    time_pressure: float = 0.6


class BayesianMM(_Figment):
    name = "fig.BayesianMM"

    def __init__(self, params: MMParams | None = None):
        self.params = params or MMParams()

    def reset(self, obs):
        hand = np.array(obs.initial_hand)
        self.target = hand.sum() / 4.0
        self.prior_goal = posterior_goal(hand)

    def _belief(self, o):
        flow = o.net_bought.astype(float) - o.net_sold.astype(float)
        if np.any(flow):
            flow = flow / (np.abs(flow).max() + 1e-9)
        p = self.prior_goal * np.exp(self.params.signal_weight * flow)
        return p / p.sum()

    def decide(self, o: FObs):
        P = self.params
        p_goal = self._belief(o)
        values = p_goal * (200 / EXPECTED_GOAL_CARDS)
        progress = o.step / max(1, o.total_steps)
        hs = (P.base_spread + P.uncertainty_weight * belief_entropy(p_goal)) * (1.0 - P.time_pressure * progress)
        hs = max(1.0, hs)
        best = None
        for s in range(4):
            v = values[s]
            ask = o.best_ask[s]
            if ask is not None and ask.player != o.me and o.cash >= ask.price:
                e = v - ask.price
                if e >= P.take_edge and (best is None or e > best[0]):
                    best = (e, (s, ask.price, True))
            bid = o.best_bid[s]
            if bid is not None and bid.player != o.me and o.hand[s] >= 1:
                e = bid.price - v
                if e >= P.take_edge and (best is None or e > best[0]):
                    best = (e, (s, bid.price, False))
        if best is not None:
            return best[1]
        s = o.step % 4
        v = values[s]
        excess = o.hand[s] - self.target
        res = v - P.risk_aversion * excess
        if excess > 0 and o.hand[s] >= 1:
            return (s, max(1, int(np.ceil(res + hs))), False)
        bid_price = max(1, int(np.floor(res - hs)))
        if v > 1.0 and o.cash >= bid_price:
            return (s, bid_price, True)
        if o.hand[s] >= 1:
            return (s, max(1, int(np.ceil(res + hs))), False)
        return None


class AdaptiveMM(BayesianMM):
    """BayesianMM with the MMParams found by figment's own evolutionary tuner
    (scripts/evolve_figment.py -> results/figment_evolved_params.json)."""
    name = "fig.AdaptiveMM"
    PARAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "results/figment_evolved_params.json")

    def __init__(self):
        with open(self.PARAMS) as f:
            super().__init__(MMParams(**json.load(f)["params"]))


# ---- figment/agents/momentum.py ------------------------------------------
class Momentum(_Figment):
    name = "fig.Momentum"

    def __init__(self, aggression=1.5):
        self.aggression = aggression

    def reset(self, obs):
        pass

    def decide(self, o: FObs):
        flow = o.net_bought.astype(float) - o.net_sold.astype(float)
        hot = int(np.argmax(flow))
        ask = o.best_ask[hot]
        if flow[hot] > 0 and ask is not None and ask.player != o.me and o.cash >= ask.price:
            return (hot, ask.price, True)
        last = o.last_price[hot]
        if flow[hot] > 0 and last is not None:
            price = max(1, int(round(last - self.aggression)))
            if o.cash >= price:
                return (hot, price, True)
        cold = int(np.argmin(flow))
        if flow[cold] < 0 and o.hand[cold] >= 1:
            bid = o.best_bid[cold]
            if bid is not None and bid.player != o.me:
                return (cold, bid.price, False)
        return None


# ---- figment/agents/random_agent.py --------------------------------------
class FigRandom(_Figment):
    name = "fig.Random"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, obs):
        pass

    def decide(self, o: FObs):
        if self.rng.random() < 0.25:
            return None
        s = int(self.rng.integers(4))
        price = int(self.rng.integers(1, 15))
        is_buy = bool(self.rng.random() < 0.5)
        if not is_buy and o.hand[s] < 1:
            is_buy = True
        if is_buy and o.cash < price:
            return None
        return (s, price, is_buy)
