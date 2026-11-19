"""Belief-value check (follow-up to the causal test, sealed before any run): is the frozen encoder's goal-suit belief
worth money to a fixed scripted trader? Eval only, no training.

The trader is the CS224R Bayesian (figgie/agents/cs224r.py, DiSilvio et al. 2021), unchanged in every rule and
threshold. Each turn it picks a suit s uniformly, then with probability 0.5 buys: value v = e_b(s, hand[s]), price
px ~ U{1..min(30, floor v)}, lift the best ask if it is <= px (and cash allows), else bid px. Otherwise it sells
(if it holds s): v = e_b(s, hand[s] - 1), px ~ U{ceil v .. min(30, ceil v + 10)}, hit the best bid if it is >= px,
else ask px. e_b uses the geometric majority-bonus valuation, r = 2. The ONLY difference between arms is the deck
posterior inside e_b:
    cc       the agent's own card-counting posterior over the 12 decks (plain Bayesian)
    enc      p(deck) = p_cc(deck | goal(deck)) * q_enc(goal(deck)), q_enc = softmax of the frozen encoder's head on
             this seat's flow-feature history (the causal test's arm B encoder, out/enc/encoder.pt)
    placebo  q = one-hot of a random suit drawn once per game (random.Random(g)), the oracle placebo of the causal test
    oracle   q = one-hot true goal suit (exploratory upper bound, not scored)
Games g in [100000, 103000): the trader plus FairValue, Bayesian, MarketMaker, seats rotated by g, play seed
1_000_000 + g, agent seeds g*10 + i, identical across arms, so differences are paired per game. 90% bootstrap CI over
games (4000 resamples, seed 0). Decision line (sealed): the reading "PPO fails to act on a belief it could form" is
refuted if enc - cc has CI90 upper bound <= +1 $/game.
"""
import argparse
import json
import math
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from figgie.agents import cs224r as C  # noqa: E402
from figgie.belief import deck_posterior, majority_threshold  # noqa: E402
from figgie.engine import ASK, BID, CARD_VALUE, DECK_GOAL, DECKS, HIT, LIFT, NOOP, POT, play  # noqa: E402

EVALUATOR = "belief_value_eval"
OPP = ["FairValue", "Bayesian", "MarketMaker"]


def eb_m(m, j, jn, r=2.0):
    """C.eb with an explicit deck posterior m (same formula as figgie.belief.eb)."""
    v = 0.0
    for i, d in enumerate(DECKS):
        if DECK_GOAL[i] != j or m[i] == 0:
            continue
        x = majority_threshold(d[j])
        B = POT - CARD_VALUE * d[j]
        vm = B * (1 - r) / (1 - r ** x) * r ** jn if jn < x else 0.0
        v += m[i] * (CARD_VALUE + vm)
    return v


def reweight(m, q):
    g = [0.0] * 4
    for i in range(12):
        g[DECK_GOAL[i]] += m[i]
    w = [m[i] / g[DECK_GOAL[i]] * q[DECK_GOAL[i]] if g[DECK_GOAL[i]] > 0 else 0.0 for i in range(12)]
    t = sum(w)
    return tuple(x / t for x in w) if t > 0 else m


class BeliefBayes(C.Bayesian):
    name = "BeliefBayes"

    def __init__(self, source, enc=None, seed=None):
        super().__init__(seed=seed)
        self.source, self.enc = source, enc

    def reset(self, obs):
        super().reset(obs)
        if self.source == "enc":
            import torch
            from figgie.rl.features import FlowFeaturizer
            self.ff = FlowFeaturizer(obs)
            self.h = torch.zeros(1, 1, self.enc[0].hidden_size)
        self.goal = obs.game.goal if self.source == "oracle" else None
        if self.source == "placebo":
            self.goal = self.placebo_suit

    def post(self, obs, c):
        m = deck_posterior(c)
        if self.source == "cc":
            return m
        if self.source in ("oracle", "placebo"):
            return reweight(m, [float(s == self.goal) for s in range(4)])
        import torch
        gru, head, mu, sd = self.enc
        x = self.ff(obs)
        with torch.no_grad():
            _, self.h = gru(torch.from_numpy((x - mu) / sd).float().view(1, 1, -1), self.h)
            q = torch.softmax(head(self.h.view(1, -1)), -1).view(-1).tolist()
        return reweight(m, q)

    def act(self, obs):          # C.Bayesian.act with e_b on the arm's posterior; every rule and threshold as there
        c = self.cc.update(obs)
        m = self.post(obs, c)
        s = self.rng.randrange(4)
        hand = obs.hand
        if self.rng.random() < 0.5:
            v = eb_m(m, s, hand[s], self.r)
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
        v = eb_m(m, s, hand[s] - 1, self.r)
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


_ENC = {}


def load(path):
    if path and path not in _ENC:
        import torch
        torch.set_num_threads(1)
        d = torch.load(path, map_location="cpu", weights_only=True)
        gru = torch.nn.GRU(d["in_dim"], d["hidden"], batch_first=True)
        gru.load_state_dict(d["gru"])
        head = torch.nn.Linear(d["hidden"], 4)
        head.load_state_dict(d["head"])
        _ENC[path] = (gru.eval(), head.eval(), d["mu"].numpy(), d["sd"].numpy())
    return _ENC.get(path)


def one(job):
    source, path, g = job
    names = ["T"] + OPP
    rot = g % 4
    seated = names[rot:] + names[:rot]
    ags = [BeliefBayes(source, load(path), seed=g * 10 + i) if n == "T" else getattr(C, n)(seed=g * 10 + i)
           for i, n in enumerate(seated)]
    for x in ags:
        x.placebo_suit = random.Random(g).randrange(4)
    r = play(ags, seed=1_000_000 + g)
    return r.profits[seated.index("T")]


def ci90(d, boot=4000):
    d = np.asarray(d)
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (boot, len(d)))].mean(1)
    return [round(float(np.quantile(bs, .05)), 3), round(float(np.quantile(bs, .95)), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--offset", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gs = list(range(a.offset, a.offset + a.games))
    src = {"cc": None, "enc": a.enc, "placebo": None, "oracle": None}
    with Pool(a.workers) as p:
        prof = {k: p.map(one, [(k, v, g) for g in gs], chunksize=20) for k, v in src.items()}
    ev = f"{EVALUATOR}@{os.environ.get('CODE_COMMIT', 'unknown')[:12]}"
    out = {"evaluator": ev, "games": a.games, "offset": a.offset, "mean": {k: round(float(np.mean(v)), 3) for k, v in prof.items()}}
    for k in ("enc", "placebo", "oracle"):
        d = np.array(prof[k]) - np.array(prof["cc"])
        out[f"{k}_minus_cc"] = {"value": round(float(d.mean()), 3), "ci90": ci90(d), "evaluator": ev}
    out["refuted"] = out["enc_minus_cc"]["ci90"][1] <= 1.0
    out["oracle_minus_cc"]["exploratory"] = True
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
