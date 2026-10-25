"""Overview figure for the Figgie proposal.

(a) one real deal, (b) that game's order flow, (c) tournament standings from
results/, (d) a preliminary revelation measurement over many simulated games,
(e) a hypothesis sketch of the leakage-exploitability relation (not data).

    python scripts/proposal_figure.py [out.png] [--games 3000]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figgie.agents.registry import make  # noqa: E402
from figgie.belief import goal_posterior  # noqa: E402
from figgie.engine import DECKS, play  # noqa: E402

TABLE = ["fig.BayesianMM", "Bayesian", "fb.PotMM", "fig.Fundamental"]
SUIT_NAME = ["♠ spades", "♣ clubs", "♥ hearts", "♦ diamonds"]
SUIT_COL = ["#222222", "#2e7d32", "#c62828", "#ef6c00"]
CHECK = list(range(0, 241, 20))
ME = 0


def hands_at(g, t):
    h = [r[:] for r in g.hands0]
    for tr in g.trades:
        if tr.tick >= t:
            break
        h[tr.buyer][tr.suit] += 1
        h[tr.seller][tr.suit] -= 1
    return h


def features(g, t):
    """Hand-only log posterior for seat ME, plus public order flow up to tick t."""
    # The dealt hand stays known all game; the current hand is distorted by trading.
    hand = tuple(g.hands0[ME])
    post = np.array(goal_posterior(hand))
    net = np.zeros((4, 4))          # player x suit net buys
    cnt, px = np.zeros(4), np.zeros(4)
    for tr in g.trades:
        if tr.tick >= t:
            break
        net[tr.buyer, tr.suit] += 1
        net[tr.seller, tr.suit] -= 1
        cnt[tr.suit] += 1
        px[tr.suit] += tr.price
    mean_px = np.where(cnt > 0, px / np.maximum(cnt, 1), 0)
    others = np.delete(net, ME, axis=0)
    flow = np.concatenate([others.ravel(), np.abs(others).max(0), cnt, mean_px / 10])
    return np.log(post + 1e-9), flow


def simulate(n_games):
    games = []
    for i in range(n_games):
        agents = [make(name, seed=1000 + i * 7 + k) for k, name in enumerate(TABLE)]
        games.append(play(agents, seed=50_000 + i).game)
    return games


def revelation(games):
    y = np.array([g.goal for g in games])
    hand_ll, flow_ll, hand_acc, flow_acc = [], [], [], []
    for t in CHECK:
        F = [features(g, t) for g in games]
        lp = np.stack([f[0] for f in F])
        X = np.hstack([lp, np.stack([f[1] for f in F])])
        p_hand = np.exp(lp) / np.exp(lp).sum(1, keepdims=True)
        p_flow = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=5,
                                   method="predict_proba")
        idx = np.arange(len(y))
        hand_ll.append(-np.log2(p_hand[idx, y] + 1e-9).mean())
        flow_ll.append(-np.log2(p_flow[idx, y] + 1e-9).mean())
        hand_acc.append((p_hand.argmax(1) == y).mean())
        flow_acc.append((p_flow.argmax(1) == y).mean())
    return dict(t=CHECK, hand_ll=hand_ll, flow_ll=flow_ll, hand_acc=hand_acc, flow_acc=flow_acc,
                n=len(games))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="results/proposal_figure.png")
    ap.add_argument("--games", type=int, default=3000)
    args = ap.parse_args()

    games = simulate(args.games)
    rev = revelation(games)
    Path("results").mkdir(exist_ok=True)
    Path("results/proposal_revelation.json").write_text(json.dumps(rev, indent=1))
    print(json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in rev.items()}))

    # Pick a game with a clear goal-suit fight for panels (a, b).
    g = max(games[:300], key=lambda g: sum(tr.suit == g.goal for tr in g.trades))

    fig = plt.figure(figsize=(14, 7.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 0.95], hspace=0.42)
    top = outer[0].subgridspec(1, 3, width_ratios=[1, 1.35, 1.1], wspace=0.45)
    bot = outer[1].subgridspec(1, 2, wspace=0.25)
    a, b, c = (fig.add_subplot(top[i]) for i in range(3))
    d, e = (fig.add_subplot(bot[i]) for i in range(2))

    # (a) the deal
    bottom = np.zeros(4)
    for s in range(4):
        vals = [g.hands0[p][s] for p in range(4)]
        a.bar(range(4), vals, bottom=bottom, color=SUIT_COL[s], edgecolor="white",
              hatch="///" if s == g.goal else None, label=SUIT_NAME[s] + ("  ← goal" if s == g.goal else ""))
        bottom += vals
    a.set_xticks(range(4), [f"P{p}" for p in range(4)], fontsize=9)
    a.set_ylabel("cards in hand")
    sizes = DECKS[g.deck_id]
    a.set_title(f"(a) One deal: deck {'/'.join(map(str, sizes))}\nhidden goal suit (hatched)", fontsize=11)
    a.legend(fontsize=7, loc="upper center", ncol=2, frameon=False, handlelength=1.2, columnspacing=0.8)
    a.set_ylim(0, 13.5)

    # (b) order flow of that game
    for tr in g.trades:
        b.scatter(tr.tick, tr.price, s=26 if tr.suit == g.goal else 12, color=SUIT_COL[tr.suit],
                  marker="^" if tr.buyer_aggressed else "v", alpha=0.85, edgecolor="none")
    gt = [tr for tr in g.trades if tr.suit == g.goal]
    buys = np.zeros(4)
    for tr in gt:
        buys[tr.buyer] += 1
    top_buyer = int(buys.argmax())
    b.text(0.98, 0.97, f"{len(g.trades)} trades; goal-suit trades enlarged.\n"
           f"P{top_buyer} bought {int(buys[top_buyer])} goal cards: its demand is public.",
           transform=b.transAxes, va="top", ha="right", fontsize=8.5,
           bbox=dict(fc="white", ec="none", alpha=0.85))
    b.set_xlabel("tick (of 240)")
    b.set_ylabel("trade price ($)")
    b.set_title("(b) The public order flow of that game\n(▲ buyer-initiated, ▼ seller-initiated)", fontsize=11)
    b.set_xlim(0, 240)

    # (c) tournament standings (real, from results/)
    rows = [("fig.AdaptiveMM", 22.2, 2.4), ("fig.BayesianMM", 18.4, 2.3), ("Bayesian", 16.1, 2.6),
            ("PPO-v2 (ours)", 15.5, 2.2), ("fig.Fundamental", 13.5, 1.9), ("fb.PotMM", 11.4, 2.3),
            ("no-info control", 2.6, 3.0), ("PPO without BC", -3.5, 2.1)]
    names, mu, ci = zip(*rows)
    cols = ["#1f77b4" if "PPO" in n else ("#999999" if "control" in n else "#555555") for n in names]
    c.barh(range(len(rows))[::-1], mu, xerr=ci, color=cols, height=0.65)
    c.set_yticks(range(len(rows))[::-1], names, fontsize=8)
    c.axvline(0, color="black", lw=0.6)
    c.set_xlabel("mean profit per game ($), 95% CI")
    c.set_title("(c) Replicated agents, mixed-table\ntournament (26 agents, top shown)", fontsize=11)

    # (d) revelation: hand-only vs + public trades (preliminary, real)
    t = np.array(rev["t"])
    d.plot(t, rev["hand_ll"], color="#555555", lw=2, marker="o", ms=3, label="exact posterior from own dealt hand")
    d.plot(t, rev["flow_ll"], color="#1f77b4", lw=2, marker="o", ms=3,
           label="+ public trade history (logistic regression)")
    d.fill_between(t, rev["flow_ll"], rev["hand_ll"], where=np.array(rev["flow_ll"]) < np.array(rev["hand_ll"]),
                   color="#1f77b4", alpha=0.15, label="revelation gap Δ_rev(t)")
    d.axhline(2.0, color="grey", ls=":", lw=1)
    d.text(238, 2.0, "uniform guess (2 bits)", ha="right", va="bottom", fontsize=8, color="grey")
    d.set_xlabel("tick (of 240)")
    d.set_ylabel("goal-suit log-loss (bits), lower is better")
    d.set_title(f"(d) Preliminary result: public trades reveal the goal suit\n({rev['n']} simulated games, 5-fold cross-validation)", fontsize=11)
    d.legend(fontsize=8, loc="lower left", frameon=False)

    # (e) hypothesis sketch
    rng = np.random.default_rng(3)
    agents = ["fig.BayesianMM", "fig.AdaptiveMM", "Bayesian", "PPO-v2", "fb.PotMM", "fig.Fundamental",
              "FairValue", "MarketMaker", "history-aware PPO"]
    L = np.array([0.55, 0.45, 0.35, 0.3, 0.6, 0.2, 0.5, 0.7, 0.1])
    E = 8 + 40 * L + rng.normal(0, 4, len(L))
    for i, n in enumerate(agents):
        col = "#ff7f0e" if "history" in n else "#555555"
        e.scatter(L[i], E[i], color=col, s=40, zorder=3)
        e.annotate(n, (L[i], E[i]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=col)
    xs = np.linspace(0.05, 0.75, 10)
    e.plot(xs, 8 + 40 * xs, color="grey", ls="--", lw=1)
    e.set_xlabel("leakage $L_a$ (bits about own hand revealed by own actions)")
    e.set_ylabel("exploitability $\\hat\\varepsilon_a$ (\\$/game)")
    e.set_title("(e) Expected result (HYPOTHESIS SKETCH, not data):\nleakier agents are more exploitable",
                fontsize=11)
    for sp in e.spines.values():
        sp.set_linestyle((0, (4, 2)))
    e.set_xticks([]); e.set_yticks([])

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
