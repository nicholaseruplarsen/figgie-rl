"""Every figure in figgie_combined_nicholas.tex except the teaser (make_teaser.py).

All figures share paper.mplstyle and are drawn from primary data only:
  NEON = ~/worktrees/figgie-rl/neon/runs   (neon_e1, neon_e2a, neon_e2b summary.json, neon_e1/voi_grid_2d.csv)
  DATA = ../data_claude                     (fig6_activations.npz/.json, causal_real.json, belief_6607.json, bv2_pilot2.json)
  REPO/runs/memory-diag/transformer_s1/common_probe.json  (untrained vs trained cores)
Coverage counts in fig_prereg are the sealed-ledger counts of the paper (docs/PREREG_FINAL.md, Trackinizer).

usage: python make_figs.py [--neon PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_claude"
REPO = HERE.parents[2]
plt.style.use(HERE / "paper.mplstyle")
BLUE, VERM, GREEN, ORANGE, PINK, SKY, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#666666"
W1, W2 = 3.4, 5.5  # one-column-ish and full text width, inches


def j(p):
    return json.loads(Path(p).read_text())


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def err(v, lo, hi):
    return [[v - lo], [hi - v]]


# ---------------------------------------------------------------- value of information
def law(p, mi, t):
    return p["v0"] * (np.asarray(mi) / 2) ** p["gamma"] * (1 - np.asarray(t) / 240) ** p["delta"]


def fig_voi(e1):
    vs = e1["voi_surface"]
    fig, ax = plt.subplots(figsize=(W1 + 0.4, 2.6))
    x = np.linspace(0, 2, 200)
    for s, c, m, lab in [(vs["T_eval"]["PPOv2"], BLUE, "o", "PPO agent"),
                         (vs["T_heldout"]["PPOv2"], SKY, "o", "PPO agent,\nunseen opponents"),
                         (vs["T_eval"]["AdaptiveMM"], ORANGE, "s", "scripted trader")]:
        ax.plot(x, law(s["law_power"], x, 0), color=c, lw=1.2, alpha=0.8)
        pts = [p for p in s["points"] if p["t_rev"] == 0]
        ax.errorbar([p["mi"] for p in pts], [p["gain"] for p in pts],
                    yerr=np.array([[p["gain"] - p["gain_ci90"][0], p["gain_ci90"][1] - p["gain"]] for p in pts]).T,
                    fmt=m, color=c, ms=4, lw=1)
        ax.text(2.06, law(s["law_power"], 2, 0), lab, color=c, fontsize=7, va="center")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-3, 115)
    ax.set_xlabel("Information about the goal suit given at the start (bits)")
    ax.set_ylabel("Profit gain ($/game)")
    save(fig, "voi")


def fig_voi_tick(grid):
    rows = [r for r in grid if r["table"] == "T_eval" and r["actor"] == "PPOv2"]
    mis = sorted({float(r["mi_bits"]) for r in rows if float(r["mi_bits"]) > 0})
    cmap = plt.get_cmap("Blues")
    fig, ax = plt.subplots(figsize=(W1 + 0.4, 2.4))
    fitm = sorted({float(r["mi_bits"]) for r in rows if r["split"] == "fit" and int(r["t_rev"]) > 0})
    for k, mi in enumerate(fitm):
        rr = sorted((int(r["t_rev"]), float(r["gain"])) for r in rows if float(r["mi_bits"]) == mi and r["split"] != "holdout")
        c = cmap(0.35 + 0.65 * k / max(1, len(fitm) - 1))
        ax.plot(*zip(*rr), "o-", color=c, ms=3.5, lw=1.2)
        ax.text(rr[0][0] - 5, rr[0][1], f"{mi:g} bits", color=c, fontsize=6.5, ha="right", va="center")
    ho = [(int(r["t_rev"]), float(r["gain"])) for r in rows if r["split"] == "holdout"]
    ax.plot(*zip(*ho), "o", mfc="white", mec=GREY, ms=3.5)
    ax.set_xlim(-75, 245)
    ax.set_xticks([0, 60, 120, 180, 240])
    ax.set_xlabel("Tick at which the information is revealed (of 240)")
    ax.set_ylabel("Profit gain ($/game)")
    save(fig, "voi_tick")


# ---------------------------------------------------------------- probes
def boot_band(c, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    G = c.shape[1]
    idx = rng.integers(0, G, size=(n, G))
    W = np.zeros((n, G))
    np.add.at(W, (np.repeat(np.arange(n), G), idx.ravel()), 1.0)
    m = c @ W.T / G
    return np.quantile(m, 0.05, axis=1), np.quantile(m, 0.95, axis=1)


def fig_probe(d):
    y = d["y"]
    t = np.arange(d["p_ppo"].shape[0])
    fig, ax = plt.subplots(figsize=(W2 * 0.72, 2.5))
    for key, lab, c, ls in [("sup", "network trained to predict the suit", BLUE, "-"),
                            ("input", "current input only", GREY, "--"),
                            ("random", "untrained network", GREY, ":"),
                            ("ppo", "network trained by PPO", VERM, "-")]:
        acc = (d[f"p_{key}"].argmax(-1) == y[None]).astype(float)
        lo, hi = boot_band(acc)
        ax.fill_between(t, 100 * lo, 100 * hi, color=c, alpha=0.12, lw=0)
        ax.plot(t, 100 * acc.mean(1), color=c, ls=ls, lw=1.6 if ls == "-" else 1.2)
        ax.text(242, 100 * acc.mean(1)[-12:].mean(), lab, color=c, fontsize=7, va="center")
    ax.axhline(25, color=GREY, lw=0.6)
    ax.text(2, 26, "chance", color=GREY, fontsize=6.5, va="bottom")
    ax.set_xlim(0, 239)
    ax.set_ylim(20, 88)
    ax.set_xlabel("Tick in the game")
    ax.set_ylabel("Probe accuracy for the goal suit (%)")
    save(fig, "probe")


def fig_example_game(d, info):
    deal, goal = int(d["deal"]), int(d["y"][int(d["deal"])])
    T = d["p_ppo"].shape[0]
    tr = d["trades"]
    buys = np.unique(tr[(tr[:, 1] == goal) & (tr[:, 5] == 1), 0])
    fig, axs = plt.subplots(3, 1, figsize=(W2, 3.2), sharex=True)
    for ax, (name, p, c) in zip(axs, [("exact card counter", d["cc"][deal], GREEN),
                                      ("network trained to predict the suit", d["sup_head"][deal], BLUE),
                                      ("probe on the PPO network", d["p_ppo"][:, deal], VERM)]):
        for tk in buys:
            ax.axvline(tk, color="#dddddd", lw=0.6, zorder=0)
        for s in range(4):
            if s != goal:
                ax.plot(range(T), p[:, s], color="#bbbbbb", lw=0.7)
        ax.plot(range(T), p[:, goal], color=c, lw=1.5)
        ax.set_ylim(-0.04, 1.06)
        ax.set_yticks([0, 0.5, 1])
        ax.grid(axis="x", visible=False)
        ax.text(0.01, 0.92, name, transform=ax.transAxes, color=c, fontsize=7, va="top")
    axs[1].set_ylabel("Probability given to each suit")
    axs[-1].set_xlabel("Tick in the game")
    axs[-1].set_xlim(0, T)
    save(fig, "example_game")


def fig_activations(d):
    goal = int(d["y"][int(d["deal"])])
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0), sharey=True)
    for ax, (k, name) in zip(axs, [("random", "untrained"), ("ppo", "trained by PPO"), ("sup", "trained to predict the suit")]):
        order = np.argsort(-d[f"w_{k}"][goal])
        im = ax.imshow(d[f"z_{k}"][:, order].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(name, fontsize=8, fontweight="normal")
        ax.grid(False)
        ax.set_xlabel("Tick")
    axs[0].set_ylabel("Hidden unit (by probe weight)")
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, label="activation")
    save(fig, "activations_heatmaps")


def fig_mirage(common):
    inp = 100 * common["input"][0]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.axhline(inp, color=GREY, ls="--", lw=1)
    ax.text(0.5, inp + 0.5, "current input only", color=GREY, fontsize=7, ha="center", va="bottom")
    for a, name, c in [("mingru", "minGRU", GREEN), ("none", "no memory", BLUE),
                       ("transformer", "Transformer", PINK), ("gru", "GRU", VERM)]:
        rs = [100 * common["nets"][f"{a}_random{i}"]["acc"] for i in (0, 1)]
        ts = [100 * common["nets"][f"{a}_s{i}"]["acc"] for i in (0, 1)]
        for r, tv in zip(rs, ts):
            ax.plot([0, 1], [r, tv], color=c, lw=0.7, alpha=0.3)
        ax.plot([0, 1], [np.mean(rs), np.mean(ts)], "o-", color=c)
        ax.text(1.05, np.mean(ts) + {"mingru": 0.5, "none": -0.5, "transformer": 0.4, "gru": -0.4}[a], name, color=c, fontsize=7, va="center")
    ax.set_xticks([0, 1], ["untrained", "trained by PPO"])
    ax.set_xlim(-0.2, 1.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Probe accuracy at the last tick (%)")
    save(fig, "mirage")


# ---------------------------------------------------------------- payoff ladder and calibration
def fig_ladder(e1, e2a, e2b, causal, belief, p2):
    vgrid = {(r["table"], r["actor"], float(r["mi_bits"]), int(r["t_rev"])): r for r in VOI_GRID}
    o = vgrid[("T_eval", "PPOv2", 2.0, 0)]
    a1 = e1["plugin_eval"]["T_eval"]["PPOv2"]["arms"]["every_eval_opponly"]
    tt = e2a["plugin"]["T_eval"]["PPOv2"]["arms"]["teval_trades_only"]
    ct = e2b["eval"]["T_eval"]["contrasts"]
    c = lambda x: (x["value"], *x["ci90"])  # noqa: E731
    rows = [  # label, (value, lo, hi), colour
        ("Truth revealed: scripted trader", c(belief["oracle_minus_cc"]), GREEN),
        ("Truth revealed: PPO, frozen", (float(o["gain"]), float(o["ci90_lo"]), float(o["ci90_hi"])), GREEN),
        ("Truth revealed: PPO, trained with it", c(causal["oracle_minus_none"]), GREEN),
        ("Belief as per-suit values: PPO, frozen", (a1["gain_vs_base"], *a1["gain_vs_base_ci90"]), BLUE),
        ("Belief as per-suit values: PPO, trained", c(ct["slot_shaped_minus_base"]), BLUE),
        ("Trades-only belief: PPO, frozen", (tt["gain_vs_hand"], *tt["gain_vs_hand_ci90"]), BLUE),
        ("Trades-only belief: PPO, trained", c(ct["trades_shaped_minus_base"]), BLUE),
        ("Belief as raw features: PPO, trained", c(causal["b_minus_none"]), GREY),
        ("Calibrated belief: scripted trader*", c(p2["enc_T_minus_cc"]), GREY),
        ("Learned card counter: scripted trader*", c(p2["ccs_T_minus_cc"]), GREY),
        ("Overconfident belief: scripted trader", c(belief["enc_minus_cc"]), VERM),
        ("Random suit: scripted trader (control)", c(belief["placebo_minus_cc"]), VERM),
    ]
    fig, ax = plt.subplots(figsize=(W2 * 0.85, 3.0))
    yy = np.arange(len(rows))[::-1]
    ax.axvline(0, color="#444444", lw=0.8)
    for y, (lab, (v, lo, hi), col) in zip(yy, rows):
        ax.errorbar(v, y, xerr=err(v, lo, hi), fmt="o", color=col, ms=4.5, lw=1)
    ax.set_yticks(yy, [r[0] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-45, 100)
    ax.set_xlabel("Profit gain over the matched baseline ($/game, 90% CI)")
    save(fig, "payoff_ladder")


def fig_calibration(p2):
    cal = p2["calib_E"]
    pts = [("exact card counter", "cc", None, GREEN), ("learned card counter", "ccs_T", "ccs_T_minus_cc", ORANGE),
           ("original encoder", "enc", "enc_minus_cc", VERM), ("original, calibrated", "enc_T", "enc_T_minus_cc", BLUE),
           ("retrained encoder", "mix", "mix_minus_cc", PINK), ("retrained, calibrated", "mix_T", "mix_T_minus_cc", PINK)]
    off = {"cc": (-5, 2), "ccs_T": (-5, -2), "enc": (-6, -8), "enc_T": (7, -4), "mix": (-6, -12), "mix_T": (6, 4)}
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.axhline(0, color="#444444", lw=0.8)
    xy = {}
    for name, arm, key, col in pts:
        x = cal[arm]["239"]["ece"]
        v, lo, hi = (0.0, 0.0, 0.0) if key is None else (p2[key]["value"], *p2[key]["ci90"])
        xy[arm] = (x, v)
        ax.errorbar(x, v, yerr=err(v, lo, hi), fmt="o", color=col, mfc="white" if arm in ("enc", "mix") else col, ms=5, lw=1,
                    label=name)
    for a, b, col in [("enc", "enc_T", BLUE), ("mix", "mix_T", PINK)]:
        ax.annotate("", xy=xy[b], xytext=xy[a], arrowprops=dict(arrowstyle="->", color=col, lw=1))
    ax.set_xlim(0.05, 0.47)
    ax.legend(loc="lower left", fontsize=6.5, handletextpad=0.2)
    ax.set_xlabel("Calibration error of the belief (lower is better)")
    ax.set_ylabel("Profit vs exact card counter ($/game)")
    save(fig, "calibration")


# ---------------------------------------------------------------- interface
def fig_interface(e1, e2a, causal):
    arms = e1["plugin_eval"]["T_eval"]["PPOv2"]["arms"]
    cb = causal["b_minus_none"]
    bars = [("raw features\n(PPO trained)", cb["value"], *cb["ci90"], GREY)]
    for k, lab in [("armB_probe", "+ per-suit\nvalues"), ("every_probe", "+ every\ntick"),
                   ("every_matched", "+ same\nopponents"), ("every_eval_opponly", "+ opponents'\nactions only")]:
        bars.append((lab, arms[k]["gain_vs_base"], *arms[k]["gain_vs_base_ci90"], BLUE))
    tt = e2a["plugin"]["T_eval"]["PPOv2"]["arms"]["teval_trades_only"]
    bars.append(("trades only,\nno quotes", tt["gain_vs_hand"], *tt["gain_vs_hand_ci90"], SKY))
    xs = [0, 1, 2, 3, 4, 5.4]
    fig, ax = plt.subplots(figsize=(W2 * 0.9, 2.5))
    for x, (lab, v, lo, hi, col) in zip(xs, bars):
        ax.bar(x, v, 0.65, color=col, yerr=err(v, lo, hi), ecolor="#333333", error_kw=dict(lw=0.8))
    pv = e1["priced_in"]["T_eval"]["PPOv2"]["private_voi_integral_usd"]
    ax.axhline(pv, color=GREY, ls="--", lw=1)
    ax.text(-0.35, pv + 1, "same information handed over directly", color=GREY, fontsize=6.5)
    ax.set_xticks(xs, [b[0] for b in bars], fontsize=6.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 50)
    ax.set_ylabel("Profit gain ($/game, 90% CI)")
    save(fig, "interface")


def fig_trap(e1, e2a):
    pe, pa = e1["plugin_eval"]["T_eval"], e2a["plugin"]["T_eval"]
    names = {"PPOv2": "PPO", "Bayesian": "Bayesian trader", "AdaptiveMM": "market maker"}
    rows = [(f"PPO games \u2192 {names[a]}", pe[a]["arms"]["every_matched"]["gain_vs_base"],
             pe[a]["arms"]["every_eval_opponly"]["gain_vs_base"]) for a in ("PPOv2", "Bayesian", "AdaptiveMM")]
    rows += [(f"Bayesian games \u2192 {names[a]}", pa[a]["arms"]["bayesrec_full85"]["gain_vs_hand"],
              pa[a]["arms"]["bayesrec_opponly"]["gain_vs_hand"]) for a in ("AdaptiveMM", "Bayesian", "PPOv2")]
    fig, ax = plt.subplots(figsize=(W2 * 0.7, 2.4))
    ax.axvline(0, color="#444444", lw=0.8)
    for y, (lab, b, a) in zip(range(len(rows))[::-1], rows):
        ax.annotate("", xy=(a, y), xytext=(b, y), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
        ax.plot(b, y, "o", mfc="white", mec=VERM, mew=1.2)
        ax.plot(a, y, "o", color=BLUE)
    ax.plot([], [], "o", mfc="white", mec=VERM, label="encoder sees everything")
    ax.plot([], [], "o", color=BLUE, label="encoder sees opponents' actions only")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_yticks(range(len(rows))[::-1], [r[0] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Profit gain over baseline ($/game)")
    ax.set_ylabel("encoder trained on \u2192 used by")
    save(fig, "trap")


# ---------------------------------------------------------------- live RL limits and pre-registration
def fig_selfplay(e2b):
    ev, sp = e2b["eval"]["T_eval"], e2b["eval"]["T_ppo_selfplay"]
    pts = [("base_cc", "card counts only", GREY, (6, 4)), ("slot_placebo", "hand-only control", GREY, (6, -2)),
           ("raw_exog", "belief as raw features", VERM, (6, 4)), ("slot_exog", "belief as per-suit values", BLUE, (6, 4)),
           ("slot_exog_shaped", "... plus reward\nfor using it", BLUE, (0, -16)),
           ("slot_trades_shaped", "trades-only belief,\nplus reward", SKY, (0, 14))]
    fig, ax = plt.subplots(figsize=(W1 + 0.8, 3.0))
    ax.set_ylim(-17, 38)
    ax.axhline(0, color="#444444", lw=0.8)
    xy = {}
    for arm, lab, c, off in pts:
        a, b = ev["arms"][arm], sp["arms"][arm]
        x, y = a["profit"], b["profit"]
        xy[arm] = (x, y)
        ax.errorbar(x, y, xerr=err(x, *a["profit_ci90"]), yerr=err(y, *b["profit_ci90"]), fmt="o", color=c, lw=0.9, ms=5)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=6.5, color=c,
                    ha="right" if off[0] < 0 else "center" if off[0] == 0 else "left", va="center")
    for k, lab in [("zeroshot_cc", "frozen PPO,\ncard counts"), ("zeroshot_trades_only", "frozen PPO, trades-only belief"),
                   ("zeroshot_quotes_trades", "frozen PPO, full belief")]:
        x, y = ev["zeroshot_ref"][k]["profit"], sp["zeroshot_ref"][k]["profit"]
        ax.plot(x, y, "D", mfc="white", mec=GREY, ms=4.5)
        ax.annotate(lab, (x, y), xytext=(-6, 9 if k == "zeroshot_cc" else -2), textcoords="offset points", fontsize=6.5, color=GREY, ha="right", va="center")
    ax.annotate("", xy=xy["slot_exog_shaped"], xytext=xy["slot_exog"], arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    ax.set_xlabel("Profit vs scripted opponents, as in training ($/game)")
    ax.set_ylabel("Profit vs three PPO copies ($/game)")
    save(fig, "selfplay")


def fig_prereg(e2b):
    ok = bool(e2b["gates"]["valid"])
    rows = [("Phase I\n(unpiloted)", 8, 15), ("Phase II\nvalue of info", 7, 9), ("Phase II\ntrades vs quotes", 6, 8),
            ("Phase II\nlive RL", 8 if ok else 7, 8 if ok else 7)]
    fig, ax = plt.subplots(figsize=(W1 + 0.8, 2.2))
    for i, (lab, h, n) in enumerate(rows):
        ax.bar(i, 100 * h / n, 0.6, color=GREY if i == 0 else BLUE)
        ax.text(i, 100 * h / n + 2, f"{h}/{n}", ha="center", fontsize=7)
    ax.axhline(90, color=GREY, ls="--", lw=1)
    ax.text(-0.4, 92, "target 90%", color=GREY, fontsize=6.5, ha="left", va="bottom")
    ax.set_xticks(range(4), [r[0] for r in rows], fontsize=6.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Predictions inside their 90% interval (%)")
    save(fig, "prereg")


def main():
    global VOI_GRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--neon", type=Path, default=Path.home() / "worktrees/figgie-rl/neon/runs")
    a = ap.parse_args()
    e1, e2a, e2b = (j(a.neon / f"{n}/summary.json") for n in ("neon_e1", "neon_e2a", "neon_e2b"))
    VOI_GRID = list(csv.DictReader(open(a.neon / "neon_e1/voi_grid_2d.csv")))
    act = dict(np.load(DATA / "fig6_activations.npz"))
    causal, belief, p2 = j(DATA / "causal_real.json"), j(DATA / "belief_6607.json"), j(DATA / "bv2_pilot2.json")
    fig_voi(e1)
    fig_voi_tick(VOI_GRID)
    fig_probe(act)
    fig_example_game(act, j(DATA / "fig6_activations.json"))
    fig_activations(act)
    fig_mirage(j(REPO / "runs/memory-diag/transformer_s1/common_probe.json"))
    fig_ladder(e1, e2a, e2b, causal, belief, p2)
    fig_calibration(p2)
    fig_interface(e1, e2a, causal)
    fig_trap(e1, e2a)
    fig_selfplay(e2b)
    fig_prereg(e2b)


if __name__ == "__main__":
    main()
