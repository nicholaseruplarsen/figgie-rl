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

import functools
import random

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


FLOW_DIM = OBS_DIM + 16


class FlowFeaturizer(Featurizer):
    """The 69 CS224R features plus who traded what since this player's last turn:

      net buys per opponent (rotated, own seat excluded) x suit, /3 (12)
      mean price of those trades per suit, /30 (4)                       = 85

    The CS224R features only keep cumulative counts per suit, so a policy can
    learn who is accumulating which suit only by integrating these over time.
    """

    def __init__(self, obs, r=2.0):
        super().__init__(obs, r)
        self.seen = 0

    def __call__(self, obs) -> np.ndarray:
        base = super().__call__(obs)
        me, n = obs.me, obs.n
        net = np.zeros((n, 4), np.float32)
        px, cnt = np.zeros(4, np.float32), np.zeros(4, np.float32)
        tr = obs.trades
        for t in tr[self.seen:]:
            net[t.buyer, t.suit] += 1
            net[t.seller, t.suit] -= 1
            px[t.suit] += t.price
            cnt[t.suit] += 1
        self.seen = len(tr)
        opp = net[[(me + i) % n for i in range(1, n)]][:3]
        f = np.zeros(FLOW_DIM, np.float32)
        f[:OBS_DIM] = base
        f[OBS_DIM:OBS_DIM + opp.size] = (opp / 3).ravel()
        f[OBS_DIM + 12:] = np.where(cnt > 0, px / np.maximum(cnt, 1), 0) / 30
        return f


RECALL_DIM = FLOW_DIM + 4


class RecallFeaturizer(FlowFeaturizer):
    """Known-answer recall check (docs/research_plan.tex, E0): the flow features plus a 4-dim cue,
    the one-hot goal suit, shown only at this player's first decision of the game and zero after.
    Not a real Figgie observation: the cue is read from the hidden game state on purpose, so a
    probe at the last tick can only find the goal suit beyond the input if the core kept it."""

    def __init__(self, obs, r=2.0):
        super().__init__(obs, r)
        self.cued = False

    def __call__(self, obs) -> np.ndarray:
        f = np.zeros(RECALL_DIM, np.float32)
        f[:FLOW_DIM] = super().__call__(obs)
        if not self.cued:
            f[FLOW_DIM + obs.game.goal] = 1.0
            self.cued = True
        return f


ENC_H = 128                     # width of the frozen supervised encoder (causal test, arm B)
_ENCODERS = {}


def load_encoder(path):
    """A frozen goal-suit GRU encoder from scripts/train_encoder.py: (gru, mu, sd), cached per process."""
    if path not in _ENCODERS:
        import torch
        torch.set_num_threads(1)        # env workers: one thread each
        d = torch.load(path, map_location="cpu", weights_only=True)
        gru = torch.nn.GRU(d["in_dim"], d["hidden"], batch_first=True)
        gru.load_state_dict(d["gru"])
        _ENCODERS[path] = (gru.eval(), d["mu"].numpy(), d["sd"].numpy())
    return _ENCODERS[path]


class EncoderFeaturizer(FlowFeaturizer):
    """Causal test, arm B: the flow features plus the hidden state of a frozen supervised GRU that was trained
    to predict the goal suit from this player's flow-feature history (FLOW_DIM + ENC_H dims)."""

    def __init__(self, obs, r=2.0, path=None):
        super().__init__(obs, r)
        import torch
        self.gru, self.mu, self.sd = load_encoder(path)
        self.h = torch.zeros(1, 1, self.gru.hidden_size)

    def __call__(self, obs) -> np.ndarray:
        import torch
        x = super().__call__(obs)
        with torch.no_grad():
            _, self.h = self.gru(torch.from_numpy((x - self.mu) / self.sd).float().view(1, 1, -1), self.h)
        return np.concatenate([x, self.h.view(-1).numpy()]).astype(np.float32)


class OracleFeaturizer(FlowFeaturizer):
    """Causal test, oracle arm: the flow features plus the true goal suit, one-hot, at every decision.
    The placebo variant shows a random suit drawn once per game instead."""
    placebo = False

    def __init__(self, obs, r=2.0):
        super().__init__(obs, r)
        self.suit = random.randrange(4) if self.placebo else obs.game.goal

    def __call__(self, obs) -> np.ndarray:
        f = np.zeros(FLOW_DIM + 4, np.float32)
        f[:FLOW_DIM] = super().__call__(obs)
        f[FLOW_DIM + self.suit] = 1.0
        return f


class OraclePlaceboFeaturizer(OracleFeaturizer):
    placebo = True


class AuxFeaturizer(FlowFeaturizer):
    """Causal test, arm A: the flow features plus one label channel (the goal suit, or a random suit per game
    for the placebo). The trainer strips the channel before the network sees it and uses it as the
    auxiliary target, so the policy's input is the plain 85-dim flow observation."""
    placebo = False

    def __init__(self, obs, r=2.0):
        super().__init__(obs, r)
        self.label = random.randrange(4) if self.placebo else obs.game.goal

    def __call__(self, obs) -> np.ndarray:
        return np.append(super().__call__(obs), np.float32(self.label)).astype(np.float32)


class AuxPlaceboFeaturizer(AuxFeaturizer):
    placebo = True


KINDS = {"oracle": OracleFeaturizer, "oracle_placebo": OraclePlaceboFeaturizer,
         "aux": AuxFeaturizer, "aux_placebo": AuxPlaceboFeaturizer}


def make_featurizer(flow):
    """flow: False (69 dims), True (85, trade flow), "recall" (89, flow + early goal cue), or a causal-test kind:
    "oracle"/"oracle_placebo" (89), "aux"/"aux_placebo" (85 + label channel), "enc:<path>" (85 + ENC_H)."""
    if flow == "recall":
        return RecallFeaturizer
    if isinstance(flow, str) and flow.startswith("enc:"):
        return functools.partial(EncoderFeaturizer, path=flow[4:])
    if flow in KINDS:
        return KINDS[flow]
    return FlowFeaturizer if flow else Featurizer


def obs_dim(flow):
    """Input width of the network for a feature kind (the aux label channel is not an input)."""
    if flow in ("recall", "oracle", "oracle_placebo"):
        return FLOW_DIM + 4
    if isinstance(flow, str) and flow.startswith("enc:"):
        return FLOW_DIM + ENC_H
    return FLOW_DIM if flow else OBS_DIM
