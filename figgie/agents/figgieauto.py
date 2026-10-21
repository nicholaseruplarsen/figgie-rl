"""Ports of 0xDub/figgie-auto (Rust, tokio; no license file; reimplemented).

The original is real-time: each bot wakes after a random sleep and the match
maker processes orders as they arrive. Here 1 tick = 1 second, and a bot acts
on tick t if its wake-up timer has expired, with the same sleep ranges as
main.rs (Tilt 2-4 s, Spread 1-2 s, Seller 2-4 s, Noisy 4-8 s; PickOff is
event-driven, so it acts every tick). TheHoarder and PrayingMantis are not
seated in main.rs; they get GenericPlayer's 2-4 s.

Their book keeps only one best bid/ask per suit and `send_order` drops any
order that does not improve it (or when we already own it); that guard is
kept, using the best quote over all players. Their engine wipes all books on
every trade, like ours.
"""
from __future__ import annotations

import random

from ..engine import ASK, BID, PARTNER


class _FA:
    lo, hi = 2000, 4000

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def reset(self, obs):
        self.next = 0.0
        self.setup(obs)

    def setup(self, obs):
        pass

    def act(self, obs):
        if obs.tick < self.next:
            return None
        self.next = obs.tick + self.rng.randrange(self.lo, self.hi) / 1000
        self.out = []
        self.decide(obs, obs.ticks - obs.tick)
        return self.out

    def book(self, obs, s):
        b, a = obs.best_bid(s), obs.best_ask(s)
        return (b if b else (0, -1)), (a if a else (99, -1))

    def send(self, obs, s, price, buy):
        if price <= 0 or price > 30:
            return
        bid, ask = self.book(obs, s)
        if buy and bid[0] < price and bid[1] != obs.me:
            self.out.append((BID, s, price))
        elif not buy and ask[0] > price and ask[1] != obs.me:
            self.out.append((ASK, s, price))


class TiltInventory(_FA):
    name = "fa.TiltInventory"

    def setup(self, obs):
        h, best = obs.initial_hand, (0, 0)
        for s in (0, 1, 3, 2):            # their iteration order S, C, D, H
            if h[s] > best[1]:
                best = (s, h[s])
        self.goal = PARTNER[best[0]]

    def decide(self, obs, left):
        for s in range(4):
            if s != self.goal and obs.hand[s] > 0:
                _, ask = self.book(obs, s)
                if left > 30:
                    if ask[0] > 4:
                        self.send(obs, s, ask[0] - 1, False)
                else:
                    self.send(obs, s, 3, False)
        _, ask = self.book(obs, self.goal)
        bid, _ = self.book(obs, self.goal)
        if ask[0] <= 5:
            self.send(obs, self.goal, ask[0], True)
        if bid[0] < 8:
            self.send(obs, self.goal, bid[0] + 1, True)


class Spread(_FA):
    name = "fa.Spread"
    lo, hi = 1000, 2000

    def decide(self, obs, left):
        avg = sum(obs.hand) // 4
        for s in range(4):
            inv = obs.hand[s]
            bid, ask = self.book(obs, s)
            last = obs.last_price[s]
            if inv > 0:
                if last is not None:
                    self.send(obs, s, last + (1 if inv > avg else 2), False)
                elif ask[0] > 7:
                    self.send(obs, s, ask[0] - (2 if inv > avg else 1), False)
            if left > 20:
                if last is not None:
                    if last > 2:
                        self.send(obs, s, last - (2 if inv > avg else 1), True)
                    else:
                        self.send(obs, s, 1, True)
                elif bid[0] < 7:
                    self.send(obs, s, bid[0] + (1 if inv > avg else 2), True)


class Seller(_FA):
    name = "fa.Seller"

    def decide(self, obs, left):
        for s in range(4):
            if obs.hand[s] == 0:
                continue
            bid, _ = self.book(obs, s)
            if left >= 180:
                mkt, lim = 6, 8
            elif left > 120:
                mkt, lim = 5, 6
            elif left > 60:
                mkt, lim = 4, 6
            else:
                mkt, lim = 3, 4
            if bid[0] >= mkt:
                self.send(obs, s, bid[0], False)
            self.send(obs, s, lim, False)


class Noisy(_FA):
    name = "fa.Noisy"
    lo, hi = 4000, 8000

    def decide(self, obs, left):
        s = self.rng.randrange(4)
        price = self.rng.randrange(1, 15)
        if self.rng.random() < 0.5:
            if obs.hand[s] < 4:
                self.send(obs, s, price, True)
        elif obs.hand[s] > 0:
            self.send(obs, s, price, False)


class PickOff(_FA):
    name = "fa.PickOff"

    def act(self, obs):
        self.out = []
        self.decide(obs, obs.ticks - obs.tick)
        return self.out

    def decide(self, obs, left):
        if left < 20:
            op, cl = 0, 0
        elif left < 40:
            op, cl = 2, 3
        elif left < 60:
            op, cl = 3, 4
        elif left < 120:
            op, cl = 4, 6
        else:
            op, cl = 5, 8
        for s in range(4):
            inv = obs.hand[s]
            bid, ask = self.book(obs, s)
            if inv <= 2 and ask[0] < op:
                self.send(obs, s, ask[0], True)
            if inv > 0:
                if bid[0] >= cl:
                    self.send(obs, s, bid[0], False)
                if ask[0] > 5:
                    self.send(obs, s, ask[0] - 1, False)


class TheHoarder(_FA):
    name = "fa.TheHoarder"

    def decide(self, obs, left):
        for s in range(4):
            if obs.hand[s] >= 6:
                continue
            bid, ask = self.book(obs, s)
            cap = 7 if left >= 120 else (6 if left > 60 else 4)
            if ask[0] <= cap:
                self.send(obs, s, ask[0], True)
            elif bid[0] < cap:
                self.send(obs, s, bid[0] + 1, True)


class PrayingMantis(_FA):
    name = "fa.PrayingMantis"

    def decide(self, obs, left):
        for s in range(4):
            _, ask = self.book(obs, s)
            if left >= 60 and obs.hand[s] > 0 and ask[0] >= 7:
                self.send(obs, s, ask[0] - 1, False)
        last = [(obs.last_price[s] or 0) for s in range(4)]
        s = max((0, 1, 3, 2), key=lambda k: last[k])   # Rust stable sort over S, C, D, H: first max wins
        _, ask = self.book(obs, s)
        if left <= 60 and ask[0] <= 9:
            self.send(obs, s, ask[0], True)
