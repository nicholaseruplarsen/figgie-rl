"""Check each port against its original implementation.

1. Decision equivalence: run the ORIGINAL simulator with the ORIGINAL agents
   and, at every decision point, feed the same state to our port and compare
   the action it would take. Deterministic agents must agree on 100% of calls.
2. Inference equivalence: posteriors / card values on every possible hand.
3. P&L: original sim vs our engine in a compat config (same tick count, no
   book wipe, same deck distribution) for the same table.

Usage: python scripts/verify_ports.py [games]   (needs third_party/ clones)
"""
import importlib
import itertools
import json
import os
import random
import statistics as st
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
G = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = {}

from figgie import engine as E
from figgie.agents import figment as PF, figgiebot as PB


def ci(xs):
    return st.mean(xs), 1.96 * st.stdev(xs) / len(xs) ** 0.5


class FakeObs:
    """Our Obs interface built from another simulator's state."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def best_bid(self, s, exclude_me=False):
        return self._bb(s, exclude_me)

    def best_ask(self, s, exclude_me=False):
        return self._ba(s, exclude_me)


# ---------------------------------------------------------------- figment
def verify_figment():
    sys.path.insert(0, os.path.join(ROOT, "third_party/figment"))
    fe = importlib.import_module("figment.engine")
    fa = importlib.import_module("figment.agents")
    fb = importlib.import_module("figment.belief")
    import numpy as np

    # inference: every 10-card hand
    worst = 0.0
    for h in itertools.product(range(11), repeat=4):
        if sum(h) != 10:
            continue
        worst = max(worst, float(np.abs(fb.posterior_goal(np.array(h)) - PF.posterior_goal(h)).max()))
    OUT["figment posterior max abs diff"] = worst

    pairs = [("Fundamental", fa.FundamentalAgent, PF.Fundamental),
             ("BayesianMM", fa.BayesianMarketMaker, PF.BayesianMM),
             ("Momentum", fa.MomentumAgent, PF.Momentum)]
    agree = {k: [0, 0] for k, *_ in pairs}

    class Spy:
        def __init__(self, key, orig, port):
            self.key, self.o, self.p, self.name = key, orig, port, orig.name

        def reset(self, player, hand, config):
            self.o.reset(player, hand, config)
            self.p.reset(types.SimpleNamespace(initial_hand=tuple(int(x) for x in hand)))

        def act(self, obs):
            a = self.o.act(obs)
            q = lambda x: PF.Quote(x.price, x.player) if x else None
            fo = PF.FObs(obs.me, obs.hand, obs.cash, [q(x) for x in obs.best_bid], [q(x) for x in obs.best_ask],
                         obs.net_bought, obs.net_sold, obs.last_price, obs.step, obs.total_steps)
            b = self.p.decide(fo)
            ka = None if a is None else (int(a.suit), int(a.price), bool(a.is_buy))
            kb = None if b is None else (int(b[0]), int(b[1]), bool(b[2]))
            agree[self.key][0] += ka == kb
            agree[self.key][1] += 1
            return a

    profits = {k: [] for k, *_ in pairs}
    for g in range(G):
        spies = [Spy(k, o(), p()) for k, o, p in pairs] + [fa.RandomAgent(seed=g)]
        order = random.Random(g).sample(range(4), 4)
        seated = [spies[k] for k in order]
        r = fe.play_game(seated, fe.GameConfig(seed=10_000 + g))
        for pos, k in enumerate(order):
            if k < 3:
                profits[pairs[k][0]].append(float(r.profits[pos]))
    OUT["figment decision agreement"] = {k: f"{a}/{n}" for k, (a, n) in agree.items()}
    OUT["figment original P&L (Fund, BMM, Mom, Random table)"] = {k: [round(x, 1) for x in ci(v)] for k, v in profits.items()}

    # our engine in figment-compat mode: no wipe, filled quote removed, 4x fewer
    # turns is not reproducible (we let every player act every tick), so compare
    # at 60 ticks = figment's 240 single-player steps.
    cfg = E.Config(ticks=60, cancel_on_trade=False, fill_removes_quote=True)
    ours = {k: [] for k, *_ in pairs}
    for g in range(G):
        ag = [PF.Fundamental(), PF.BayesianMM(), PF.Momentum(), PF.FigRandom(seed=g)]
        order = random.Random(g).sample(range(4), 4)
        r = E.play([ag[k] for k in order], seed=10_000 + g, cfg=cfg)
        for pos, k in enumerate(order):
            if k < 3:
                ours[pairs[k][0]].append(r.profits[pos])
    OUT["figment port P&L on shared engine (compat, 60 ticks)"] = {k: [round(x, 1) for x in ci(v)] for k, v in ours.items()}


# ---------------------------------------------------------------- Figgie-bot
def verify_figgiebot():
    sys.path.insert(0, os.path.join(ROOT, "third_party/Figgie-bot/figgie-bot-x"))
    oe = importlib.import_module("figgie.engine")
    oi = importlib.import_module("figgie.inference")
    ov = importlib.import_module("figgie.valuation")
    ob = importlib.import_module("figgie.bots")
    oa = importlib.import_module("figgie.ablations")
    osim = importlib.import_module("figgie.sim")
    S = oe.SUITS          # ('S','C','H','D') == our suit order

    worst_p = worst_v = 0.0
    for h in itertools.product(range(11), repeat=4):
        if sum(h) != 10:
            continue
        d = dict(zip(S, h))
        po, pp = oi.goal_posterior(d), PB.goal_posterior(h)
        worst_p = max(worst_p, max(abs(po[S[s]] - pp[s]) for s in range(4)))
        for cur in (h, tuple(max(0, x - 1) for x in h), tuple(x + (i == 0) for i, x in enumerate(h))):
            vo = ov.card_values(d, dict(zip(S, cur)), 4)
            vp = PB.card_values(h, cur, 4)
            worst_v = max(worst_v, max(abs(vo[S[s]][k] - vp[s][k]) for s in range(4) for k in range(2)))
    OUT["Figgie-bot posterior max abs diff"] = worst_p
    OUT["Figgie-bot card_values max abs diff"] = worst_v

    def adapt(o):
        top, tex = o["top"], o["top_ex"]
        tr = [E.Trade(t.t, S.index(t.suit), t.price, t.buyer, t.seller, False) for t in o["trades"]]
        lp = [o["last_price"].get(s) for s in S]
        return FakeObs(me=o["me"], n=o["n"], tick=o["t"], hand=[o["hand"][s] for s in S], cash=o["cash"],
                       trades=tr, last_price=lp,
                       _bb=lambda s, ex: (tex if ex else top)[S[s]]["bid"],
                       _ba=lambda s, ex: (tex if ex else top)[S[s]]["offer"])

    def norm_orig(acts):
        return [(a[0], S.index(a[1]), a[2]) for a in acts]

    pairs = [("PotMM", ob.PotMM, PB.PotMM), ("HandMM", ob.HandMM, PB.HandMM),
             ("FlowMM", ob.FlowMM, PB.FlowMM), ("ConstVal", oa.ConstVal, PB.ConstVal)]
    agree = {k: [0, 0] for k, *_ in pairs}

    class Spy:
        def __init__(self, key, o, p):
            self.key, self.o, self.p = key, o, p

        def act(self, obs, rng):
            a = self.o.act(obs, rng)
            b = self.p.decide(adapt(obs))
            agree[self.key][0] += norm_orig(a) == [(x[0], x[1], x[2]) for x in b]
            agree[self.key][1] += 1
            return a

    for g in range(max(20, G // 10)):
        bots = [Spy(k, o(), p()) for k, o, p in pairs]
        osim.play(bots, 200, 900_000 + g)
    OUT["Figgie-bot decision agreement"] = {k: f"{a}/{n}" for k, (a, n) in agree.items()}

    # P&L: their E1 table (README: PotMM +5.8, HandMM +1.0, FlowMM +3.2, FixedSpread -9.9 at 800 games)
    names = ["PotMM", "HandMM", "FlowMM", "FixedSpread"]
    orig = osim.tournament([ob.PotMM(), ob.HandMM(), ob.FlowMM(), ob.FixedSpreadBot()], games=G, seed=900_000)
    OUT["Figgie-bot original E1 P&L"] = {n: [round(m, 1), round(c, 1)] for n, (m, c) in zip(names, orig)}

    # Same table on our engine in compat mode, with their 16-composition deck distribution.
    hyp_to_deck = [E.DECKS.index(sz) for _, sz in PB.HYPS["source"]]
    cfg = E.Config(ticks=200, cancel_on_trade=False, fill_removes_quote=False)
    pnl = [[] for _ in names]
    for k in range(G):
        shift = k % 4
        bots = [PB.PotMM(), PB.HandMM(), PB.FlowMM(), PB.FixedSpread(seed=k)]
        seated = bots[shift:] + bots[:shift]
        deck = hyp_to_deck[random.Random(k).randrange(16)]
        r = E.play(seated, seed=900_000 + k, cfg=cfg, deck=deck)
        for seat, x in enumerate(r.profits):
            pnl[(seat + shift) % 4].append(x)
    OUT["Figgie-bot port E1 P&L on shared engine (compat)"] = {n: [round(x, 1) for x in ci(v)] for n, v in zip(names, pnl)}


if __name__ == "__main__":
    verify_figment()
    # Figgie-bot's package is also called `figgie`; load it under a clean module table.
    for m in [m for m in sys.modules if m == "figgie" or m.startswith("figgie.")]:
        del sys.modules[m]
    sys.path.remove(ROOT)
    verify_figgiebot()
    print(json.dumps(OUT, indent=1))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(OUT, open(os.path.join(ROOT, "results/verify_ports.json"), "w"), indent=1)
