"""69-dim observation for the CS224R PPO agent (§6.1).

The report gives the size (69) and the ingredients (normalised hand counts,
cyclically permuted cash and quote arrays with own slot first, tick fraction,
4-dim lift-opportunity feature, Bayesian belief features) but not the exact
layout; this layout is a GUESS that uses exactly those ingredients:

  hand/10 (4) | initial hand/10 (4) | cash/350, own first (4)
  bids/30 [player x suit], own row first (16) | asks/30 (16) | tick/T (1)
  P(goal) from card counting (4) | e_b buy value/30 (4) | e_b sell value/30 (4)
  last price/30 (4) | trades per suit/10 (4) | lift opportunity (4)  = 69
"""
from __future__ import annotations

import numpy as np

from ..agents.cs224r import CardCounter
from ..belief import eb, goal_posterior

OBS_DIM = 69


class Featurizer:
    def __init__(self, obs, r=2.0):
        self.cc = CardCounter(obs)
        self.r = r
        self.counts = None

    def potential(self, obs):
        """phi(s) = cash + sum_j sum_{k < hand_j} e_b(j, k) - 350 (CS224R §6.2)."""
        c = self.counts
        return obs.cash + sum(eb(c, j, k, self.r) for j in range(4) for k in range(obs.hand[j])) - 350

    def __call__(self, obs) -> np.ndarray:
        c = self.counts = self.cc.update(obs)
        me, n = obs.me, obs.n
        rot = [(me + i) % n for i in range(n)]
        hand = obs.hand
        f = np.zeros(OBS_DIM, dtype=np.float32)
        f[0:4] = np.array(hand) / 10
        f[4:8] = np.array(obs.initial_hand) / 10
        f[8:12] = [obs.cash_all[p] / 350 for p in rot][:4]   # 4-player only; 5p truncates
        f[12:28] = (np.array([obs.bids[p] for p in rot][:4]) / 30).ravel()
        f[28:44] = (np.array([obs.asks[p] for p in rot][:4]) / 30).ravel()
        f[44] = obs.tick / obs.ticks
        f[45:49] = goal_posterior(c)
        buy = [eb(c, s, hand[s], self.r) for s in range(4)]
        f[49:53] = np.array(buy) / 30
        f[53:57] = [eb(c, s, hand[s] - 1, self.r) / 30 if hand[s] else 0 for s in range(4)]
        f[57:61] = [(obs.last_price[s] or 0) / 30 for s in range(4)]
        cnt = [0] * 4
        for t in obs.trades:
            cnt[t.suit] += 1
        f[61:65] = np.array(cnt) / 10
        for s in range(4):   # lift opportunity: (e_b - best external ask)+ / 10
            a = obs.best_ask(s, True)
            if a is not None:
                f[65 + s] = max(0.0, buy[s] - a[0]) / 10
        return f
