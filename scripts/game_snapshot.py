"""Render one mid-game Figgie position as a table diagram (like a chess diagram).

Replays the game used in proposal_figure.py and draws the state after a chosen
tick: four seats with cash and hands (other players' hands greyed: hidden from
P0), the order book per suit, the last trades, and P0's hand-only belief.

    python scripts/game_snapshot.py [out.png]
"""
import copy
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figgie.agents.registry import make  # noqa: E402
from figgie.belief import goal_posterior  # noqa: E402
from figgie.engine import DECKS, Game, Obs, play  # noqa: E402

TABLE = ["fig.BayesianMM", "Bayesian", "fb.PotMM", "fig.Fundamental"]
SYM = ["♠", "♣", "♥", "♦"]
COL = ["#111111", "#111111", "#c62828", "#c62828"]  # black spades/clubs, red hearts/diamonds


def agents_for(i):
    return [make(name, seed=1000 + i * 7 + k) for k, name in enumerate(TABLE)]


def replay(i):
    """play() with a snapshot of the game after every tick."""
    seed = 50_000 + i
    g = Game(None, seed, None)
    order_rng = random.Random(seed * 7919 + 1)
    agents = agents_for(i)
    obs = [Obs(g, p) for p in range(g.n)]
    for p, a in enumerate(agents):
        a.reset(obs[p])
    order, snaps = list(range(g.n)), []
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
                g.apply(p, act[0], act[1] if len(act) > 1 else 0, act[2] if len(act) > 2 else 0)
        snaps.append(dict(hands=copy.deepcopy(g.hands), cash=g.cash[:], bids=copy.deepcopy(g.bids),
                          asks=copy.deepcopy(g.asks), trades=list(g.trades), last=g.last_price[:]))
    return g, snaps


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results/game_snapshot.png"
    # Same game as proposal_figure.py: most goal-suit trades among the first 300.
    best = max(range(300), key=lambda i: (lambda g: sum(tr.suit == g.goal for tr in g.trades))(
        play(agents_for(i), seed=50_000 + i).game))
    g, snaps = replay(best)
    # A tick with several trades done and the most quotes standing.
    def score(t):
        s = snaps[t]
        return sum(x > 0 for row in s["bids"] + s["asks"] for x in row) if len(s["trades"]) >= 8 else -1
    t = max(range(20, 120), key=score)
    s = snaps[t]

    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.add_patch(FancyBboxPatch((2.6, 1.9), 4.8, 3.3, boxstyle="round,pad=0.05,rounding_size=0.6",
                                fc="#e8f1e4", ec="#7a9a6a", lw=1.5))
    ax.text(5, 7.0, f"Figgie, tick {t + 1} of 240    pot $200    goal suit: hidden (actually {SYM[g.goal]})",
            ha="center", va="top", fontsize=12, weight="bold")

    # Order book in the middle.
    ax.text(5, 4.95, "order book: best quote per suit", ha="center", fontsize=10, color="#335", weight="bold")
    hdr_y = 4.6
    for x, h in [(3.3, "bidder"), (4.15, "bid"), (5.0, "suit"), (5.85, "ask"), (6.7, "asker")]:
        ax.text(x, hdr_y, h, ha="center", fontsize=8.5, color="grey")
    for su in range(4):
        y = 4.2 - su * 0.42
        bid = max(((s["bids"][p][su], p) for p in range(4)), default=(0, -1))
        asks = [(s["asks"][p][su], p) for p in range(4) if s["asks"][p][su] > 0]
        ask = min(asks) if asks else (0, -1)
        ax.text(5.0, y, SYM[su], ha="center", fontsize=15, color=COL[su])
        ax.text(4.15, y, f"${bid[0]}" if bid[0] else "–", ha="center", fontsize=10.5)
        ax.text(3.3, y, f"P{bid[1]}" if bid[0] else "", ha="center", fontsize=9, color="#555")
        ax.text(5.85, y, f"${ask[0]}" if ask[0] else "–", ha="center", fontsize=10.5)
        ax.text(6.7, y, f"P{ask[1]}" if ask[0] else "", ha="center", fontsize=9, color="#555")
    ax.text(5, 2.25, "every trade is public: tick, suit, price, buyer, seller", ha="center",
            fontsize=8.5, style="italic", color="#335")

    # Seats: P0 bottom (the viewpoint), P1 left, P2 top, P3 right.
    pos = {0: (5, 1.05), 1: (1.25, 3.55), 2: (5, 6.05), 3: (8.75, 3.55)}
    post = goal_posterior(tuple(g.hands0[0]))
    for p, (x, y) in pos.items():
        mine = p == 0
        ax.add_patch(FancyBboxPatch((x - 1.15, y - 0.62), 2.3, 1.24, boxstyle="round,pad=0.04",
                                    fc="white" if mine else "#f2f2f2", ec="#1f77b4" if mine else "#bbbbbb",
                                    lw=2 if mine else 1))
        name = TABLE[p].split(".")[-1]
        ax.text(x, y + 0.42, f"P{p} {name}" + ("  (viewpoint)" if mine else ""), ha="center", fontsize=9,
                weight="bold" if mine else None)
        ax.text(x, y + 0.14, f"cash ${s['cash'][p]}", ha="center", fontsize=8.5, color="#333")
        for su in range(4):
            ax.text(x - 0.78 + su * 0.52, y - 0.28, f"{SYM[su]}{s['hands'][p][su]}", ha="center", fontsize=11,
                    color=COL[su], alpha=1 if mine else 0.35)
        if not mine:
            ax.text(x, y - 0.55, "hand hidden from P0", ha="center", fontsize=7, style="italic", color="#888")
    # P0's belief.
    ax.add_patch(FancyBboxPatch((0.05, 0.05), 3.35, 1.65, boxstyle="round,pad=0.04", fc="white", ec="#bbbbbb", lw=1))
    ax.text(1.725, 1.5, "P0's goal-suit belief (dealt hand)", ha="center", fontsize=9, color="#335", weight="bold")
    for su in range(4):
        yy = 1.18 - su * 0.3
        ax.text(0.25, yy, SYM[su], fontsize=11, color=COL[su], va="center")
        ax.add_patch(FancyBboxPatch((0.6, yy - 0.08), 2.0 * post[su], 0.16, boxstyle="square,pad=0",
                                    fc="#1f5fa0", ec="none"))
        ax.text(0.7 + 2.0 * post[su], yy, f"{post[su]:.0%}", fontsize=8.5, va="center", color="#1f5fa0")

    # Recent trades.
    ax.add_patch(FancyBboxPatch((6.6, 0.05), 3.35, 1.65, boxstyle="round,pad=0.04", fc="white", ec="#bbbbbb", lw=1))
    ax.text(8.275, 1.5, "last trades (public)", ha="center", fontsize=9, color="#335", weight="bold")
    for k, tr in enumerate(s["trades"][-5:][::-1]):
        ax.text(6.75, 1.22 - k * 0.26, f"t{tr.tick + 1:<3} P{tr.buyer} buys {SYM[tr.suit]} from P{tr.seller}  ${tr.price}",
                ha="left", fontsize=8, color=COL[tr.suit], family="DejaVu Sans Mono")

    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out, "game", best, "tick", t + 1, "deck", DECKS[g.deck_id])


if __name__ == "__main__":
    main()
