"""Single-learner Figgie environment on the shared engine.

The learner occupies one random seat; the other seats are filled by an
opponent sampler. One env step = one learner turn (one tick). Reward:
  r_t = beta * (gamma * phi(s_{t+1}) - phi(s_t)) + lift bonus      (non-terminal)
  r_T = profit + beta * (0 - phi(s_T)) + lift bonus                  (terminal)
all multiplied by reward_scale (GUESS: 0.1; the report does not give one).
"""
from __future__ import annotations

import random

from ..actions import decode, legal_mask
from ..belief import eb
from ..engine import BID, LIFT, Config, Game, Obs
from .features import Featurizer


class FiggieEnv:
    def __init__(self, opp_sampler, seed=0, beta=0.5, gamma=0.995, lift_alpha=0.0, reward_scale=0.1, cfg=None):
        self.sample_opps = opp_sampler
        self.rng = random.Random(seed)
        self.beta, self.gamma, self.lift_alpha, self.scale = beta, gamma, lift_alpha, reward_scale
        self.cfg = cfg or Config()

    def reset(self):
        self.g = Game(self.cfg, self.rng.getrandbits(48))
        self.seat = self.rng.randrange(self.g.n)
        self.opp_ids, opps = self.sample_opps(self.rng)
        self.agents = {}
        k = 0
        for p in range(self.g.n):
            if p != self.seat:
                self.agents[p] = opps[k]
                k += 1
        self.obs = {p: Obs(self.g, p) for p in range(self.g.n)}
        for p, a in self.agents.items():
            a.reset(self.obs[p])
        self.me = self.obs[self.seat]
        self.feat = Featurizer(self.me)
        self.lifts = 0
        self.gen = self._run()
        next(self.gen)
        return self._observe()

    def _observe(self):
        f = self.feat(self.me)
        self.phi = self.feat.potential(self.me)
        return f, legal_mask(self.me)

    def _run(self):
        g = self.g
        order = list(range(g.n))
        for t in range(g.cfg.ticks):
            g.tick = t
            self.rng.shuffle(order)
            for p in order:
                if p == self.seat:
                    act = yield
                    self._apply_learner(act)
                else:
                    acts = self.agents[p].act(self.obs[p])
                    if acts:
                        for a in ((acts,) if isinstance(acts, tuple) else acts):
                            g.apply(p, a[0], a[1] if len(a) > 1 else 0, a[2] if len(a) > 2 else 0)

    def _apply_learner(self, act):
        g, s = self.g, act[1] if len(act) > 1 else 0
        jn = g.hands[self.seat][s]
        n0 = len(g.trades)
        g.apply(self.seat, act[0], s, act[2] if len(act) > 2 else 0)
        self.bonus = 0.0
        if len(g.trades) > n0 and g.trades[-1].buyer == self.seat and act[0] in (LIFT, BID):
            self.lifts += 1
            if self.lift_alpha:
                self.bonus = self.lift_alpha * max(0.0, eb(self.feat.counts, s, jn) - g.trades[-1].price)

    def step(self, a: int):
        phi0 = self.phi
        try:
            self.gen.send(decode(a))
            f, m = self._observe()
            r = self.beta * (self.gamma * self.phi - phi0) + self.bonus
            return f, m, r * self.scale, False, None
        except StopIteration:
            profits = self.g.settle()
            me = self.seat
            r = profits[me] - self.beta * phi0 + self.bonus
            info = dict(profit=profits[me], win=profits[me] == max(profits), lifts=self.lifts,
                        opp_ids=self.opp_ids, opp_profits=[profits[p] for p in self.agents])
            return None, None, r * self.scale, True, info
