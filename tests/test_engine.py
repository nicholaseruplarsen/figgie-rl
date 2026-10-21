import copy
import random
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figgie import actions as A
from figgie.belief import eb, goal_posterior
from figgie.engine import ASK, BID, DECKS, DECK_GOAL, HIT, LIFT, Game, Obs, play
from figgie.agents.registry import AGENTS, make


def test_decks():
    assert len(set(DECKS)) == 12 and all(sorted(d) == [8, 10, 10, 12] for d in DECKS)
    assert sum(DECKS[i][DECK_GOAL[i]] == 8 for i in range(12)) == 4   # goal is 8 w.p. 1/3


def test_trade_wipes_book_and_conserves():
    g = Game(seed=1)
    s = next(s for s in range(4) if g.hands[1][s] > 0)
    assert g.apply(0, BID, s, 5) and g.apply(2, BID, (s + 1) % 4, 3)
    assert g.apply(1, ASK, s, 5)                      # crosses -> trade at 5
    assert g.trades[-1].price == 5 and g.trades[-1].buyer == 0
    assert all(x == 0 for row in g.bids + g.asks for x in row)
    assert abs(sum(g.settle())) < 1e-9


def test_self_cross_rejected():
    g = Game(seed=2)
    s = next(s for s in range(4) if g.hands[0][s] > 0)
    assert g.apply(0, ASK, s, 6)
    assert not g.apply(0, BID, s, 7)


def test_action_roundtrip_and_mask_matches_engine():
    for i in range(A.N_ACTIONS):
        assert A.encode(A.decode(i)) == i
    rng = random.Random(0)
    for seed in range(30):
        g = Game(seed=seed)
        for _ in range(rng.randrange(60)):
            p = rng.randrange(4)
            a = A.decode(rng.randrange(A.N_ACTIONS))
            g.apply(p, a[0], a[1] if len(a) > 1 else 0, a[2] if len(a) > 2 else 0)
        o = Obs(g, 0)
        m = A.legal_mask(o)
        for i in rng.sample(range(A.N_ACTIONS), 80):
            h = copy.deepcopy(g)
            a = A.decode(i)
            n0 = h.rejected
            h.apply(0, a[0], a[1] if len(a) > 1 else 0, a[2] if len(a) > 2 else 0)
            if 241 <= i < 249:   # cancelling nothing is a harmless no-op; the mask hides it
                assert h.rejected == n0
            else:
                assert (h.rejected == n0) == bool(m[i]), (i, a)


def test_beliefs():
    p = goal_posterior((0, 0, 0, 0))
    assert all(abs(x - .25) < 1e-12 for x in p)
    # e_b over a full goal pile telescopes to the card value + bonus
    c = (0, 0, 0, 0)
    tot = sum(eb(c, 0, k) for k in range(12))
    assert abs(tot - (10 * 12 + 4 / 12 * 120 + 8 / 12 * 100) / 4) < 1.0


def test_all_agents_play_zero_sum():
    names = list(AGENTS)
    for g in range(40):
        pick = random.Random(g).sample(names, 4)
        r = play([make(n, g) for n in pick], seed=g)
        assert abs(sum(r.profits)) < 1e-6
