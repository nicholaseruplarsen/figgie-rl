"""Shared Figgie engine.

Rules (standard Figgie, cross-checked against figgie.com and all four sources):
- 40 cards; suit sizes are a uniformly random assignment of {12, 10, 10, 8}
  (12 equally likely decks). The goal suit is the same-colour partner of the
  12-card suit, so it has 8 cards w.p. 1/3 and 10 w.p. 2/3.
- 4 players (10 cards each) or 5 players (8 each). Everyone starts with $350
  and antes 200/n into a $200 pot.
- Settlement: $10 per goal card held; the rest of the pot (100 or 120) is
  split evenly among the players holding the most goal cards.
- profit_i = final wealth - 350; profits sum to zero.

Trading protocol (the CS224R discretisation, see DESIGN.md):
- T ticks (default 240, "one per second"). Every tick each player acts once;
  the acting order is a fresh random permutation each tick, and each player
  observes the state as left by the players before it in that tick.
- A player's turn is a list of primitive orders, applied in order (RL agents
  and the CS224R baselines always send one; ports of multi-order bots may send
  several, as they did in their original sims).
- Primitives: place_bid(s,p), place_ask(s,p) with integer p in [1, max_price],
  cancel_bid(s), cancel_ask(s), lift(s), hit(s).
- At most one bid and one ask per player per suit; a new quote replaces the old.
- A place_* that crosses the best opposing quote of another player executes
  immediately at the resting price (so it behaves as lift/hit); one crossing
  your own opposite quote is rejected.
- Bids need cash >= price, asks need a card in hand. No shorting.
- cancel_on_trade (canonical Figgie, default on): every trade wipes every
  standing quote in every suit.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

SUITS = "SCHD"            # spades, clubs, hearts, diamonds
PARTNER = (1, 0, 3, 2)    # same-colour suit
POT = 200
START_CASH = 350
CARD_VALUE = 10

# The 12 deck compositions, indexed by suit.
DECKS: tuple[tuple[int, int, int, int], ...] = tuple(
    tuple(12 if s == c else (8 if s == e else 10) for s in range(4))
    for c in range(4) for e in range(4) if e != c
)
DECK_GOAL = tuple(PARTNER[d.index(12)] for d in DECKS)

NOOP, BID, ASK, CANCEL_BID, CANCEL_ASK, LIFT, HIT = range(7)


@dataclass
class Trade:
    tick: int
    suit: int
    price: int
    buyer: int
    seller: int
    buyer_aggressed: bool


@dataclass
class Config:
    n_players: int = 4
    ticks: int = 240
    max_price: int = 30
    cancel_on_trade: bool = True
    # Only used with cancel_on_trade=False: whether a filled resting quote is
    # removed (figment) or stays until the owner runs out of cards/cash (Figgie-bot).
    fill_removes_quote: bool = True


class Obs:
    """What player `me` sees. Hidden: deck, goal, other hands."""
    __slots__ = ("game", "me")

    def __init__(self, game: "Game", me: int):
        self.game, self.me = game, me

    @property
    def n(self): return self.game.n
    @property
    def tick(self): return self.game.tick
    @property
    def ticks(self): return self.game.cfg.ticks
    @property
    def hand(self): return self.game.hands[self.me]
    @property
    def initial_hand(self): return self.game.hands0[self.me]
    @property
    def cash(self): return self.game.cash[self.me]
    @property
    def trades(self): return self.game.trades
    @property
    def bids(self): return self.game.bids          # [player][suit] price or 0
    @property
    def asks(self): return self.game.asks
    @property
    def last_price(self): return self.game.last_price  # [suit] or None
    @property
    def cash_all(self): return self.game.cash

    def best_bid(self, s, exclude_me=False):
        """(price, player) of the best bid in suit s, or None."""
        return self.game.best_bid(s, self.me if exclude_me else -1)

    def best_ask(self, s, exclude_me=False):
        return self.game.best_ask(s, self.me if exclude_me else -1)


class Game:
    def __init__(self, cfg: Config | None = None, seed: int | None = None, deck: int | None = None):
        self.cfg = cfg or Config()
        self.n = self.cfg.n_players
        assert 40 % self.n == 0
        self.rng = random.Random(seed)
        self.deck_id = self.rng.randrange(12) if deck is None else deck
        self.sizes = DECKS[self.deck_id]
        self.goal = DECK_GOAL[self.deck_id]
        pile = [s for s in range(4) for _ in range(self.sizes[s])]
        self.rng.shuffle(pile)
        k = 40 // self.n
        self.hands = [[0] * 4 for _ in range(self.n)]
        for i, s in enumerate(pile):
            self.hands[i // k][s] += 1
        self.hands0 = [h[:] for h in self.hands]
        self.cash = [START_CASH - POT // self.n] * self.n
        self.bids = [[0] * 4 for _ in range(self.n)]
        self.asks = [[0] * 4 for _ in range(self.n)]
        self.bid_seq = [[0] * 4 for _ in range(self.n)]
        self.ask_seq = [[0] * 4 for _ in range(self.n)]
        self._seq = 0
        self.trades: list[Trade] = []
        self.last_price: list[int | None] = [None] * 4
        self.tick = 0
        self.rejected = 0

    # ---- book ----------------------------------------------------------
    def best_bid(self, s, exclude=-1):
        best = None
        for p in range(self.n):
            px = self.bids[p][s]
            if px and p != exclude:
                if best is None or px > best[0] or (px == best[0] and self.bid_seq[p][s] < self.bid_seq[best[1]][s]):
                    best = (px, p)
        return best

    def best_ask(self, s, exclude=-1):
        best = None
        for p in range(self.n):
            px = self.asks[p][s]
            if px and p != exclude:
                if best is None or px < best[0] or (px == best[0] and self.ask_seq[p][s] < self.ask_seq[best[1]][s]):
                    best = (px, p)
        return best

    def _trade(self, s, buyer, seller, price, buyer_aggressed):
        self.cash[buyer] -= price
        self.cash[seller] += price
        self.hands[buyer][s] += 1
        self.hands[seller][s] -= 1
        self.last_price[s] = price
        self.trades.append(Trade(self.tick, s, price, buyer, seller, buyer_aggressed))
        if self.cfg.cancel_on_trade:
            for p in range(self.n):
                self.bids[p] = [0] * 4
                self.asks[p] = [0] * 4
        else:
            # Only drop the filled quote and quotes that can no longer be honoured.
            if self.cfg.fill_removes_quote:
                if buyer_aggressed:
                    self.asks[seller][s] = 0
                else:
                    self.bids[buyer][s] = 0
            for t in range(4):
                if self.bids[buyer][t] > self.cash[buyer]:
                    self.bids[buyer][t] = 0
                if self.asks[seller][t] and self.hands[seller][t] < 1:
                    self.asks[seller][t] = 0
        return True

    def apply(self, p: int, kind: int, s: int = 0, price: int = 0) -> bool:
        """Apply one primitive for player p. Returns False if rejected."""
        if kind == NOOP:
            return True
        if kind == CANCEL_BID:
            self.bids[p][s] = 0
            return True
        if kind == CANCEL_ASK:
            self.asks[p][s] = 0
            return True
        if kind == LIFT:
            a = self.best_ask(s, p)
            if a is None or self.cash[p] < a[0]:
                self.rejected += 1
                return False
            return self._trade(s, p, a[1], a[0], True)
        if kind == HIT:
            b = self.best_bid(s, p)
            if b is None or self.hands[p][s] < 1:
                self.rejected += 1
                return False
            return self._trade(s, b[1], p, b[0], False)
        price = int(price)
        if price < 1 or price > self.cfg.max_price:
            self.rejected += 1
            return False
        if kind == BID:
            own = self.asks[p][s]
            if self.cash[p] < price or (own and own <= price):
                self.rejected += 1
                return False
            a = self.best_ask(s, p)
            if a is not None and a[0] <= price:
                return self._trade(s, p, a[1], a[0], True)
            self._seq += 1
            self.bids[p][s], self.bid_seq[p][s] = price, self._seq
            return True
        if kind == ASK:
            own = self.bids[p][s]
            if self.hands[p][s] < 1 or (own and own >= price):
                self.rejected += 1
                return False
            b = self.best_bid(s, p)
            if b is not None and b[0] >= price:
                return self._trade(s, b[1], p, b[0], False)
            self._seq += 1
            self.asks[p][s], self.ask_seq[p][s] = price, self._seq
            return True
        raise ValueError(kind)

    # ---- settlement ----------------------------------------------------
    def settle(self) -> list[float]:
        g = self.goal
        counts = [h[g] for h in self.hands]
        top = max(counts)
        winners = [i for i, c in enumerate(counts) if c == top]
        bonus = POT - CARD_VALUE * self.sizes[g]
        return [self.cash[i] + CARD_VALUE * counts[i] + (bonus / len(winners) if i in winners else 0) - START_CASH
                for i in range(self.n)]

    def winners(self) -> list[int]:
        counts = [h[self.goal] for h in self.hands]
        top = max(counts)
        return [i for i, c in enumerate(counts) if c == top]


@dataclass
class Result:
    profits: list[float]
    game: Game
    stats: list[dict] = field(default_factory=list)


def play(agents, seed: int | None = None, cfg: Config | None = None, deck: int | None = None) -> Result:
    """Play one round. Each agent implements reset(obs) and act(obs) -> list of
    (kind, suit, price) primitives (or a single tuple, or None)."""
    g = Game(cfg, seed, deck)
    order_rng = random.Random(None if seed is None else seed * 7919 + 1)
    obs = [Obs(g, i) for i in range(g.n)]
    stats = [dict(lifts=0, hits=0, quotes=0, filled=0) for _ in range(g.n)]
    for i, a in enumerate(agents):
        a.reset(obs[i])
    order = list(range(g.n))
    for t in range(g.cfg.ticks):
        g.tick = t
        order_rng.shuffle(order)
        for p in order:
            acts = agents[p].act(obs[p])
            if not acts:
                continue
            if isinstance(acts, tuple):
                acts = (acts,)
            for act in acts:
                kind = act[0]
                n0 = len(g.trades)
                ok = g.apply(p, kind, act[1] if len(act) > 1 else 0, act[2] if len(act) > 2 else 0)
                if ok and len(g.trades) > n0:
                    tr = g.trades[-1]
                    if tr.buyer == p:
                        stats[p]["lifts"] += 1
                        stats[tr.seller]["filled"] += 1
                    else:
                        stats[p]["hits"] += 1
                        stats[tr.buyer]["filled"] += 1
                elif ok and kind in (BID, ASK):
                    stats[p]["quotes"] += 1
    return Result(g.settle(), g, stats)
