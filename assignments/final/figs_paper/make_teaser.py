"""Teaser: what the agent knows vs what it earns.

Every point comes from one run (Neon-E2b, MLflow 397ec6e3) and one probe setup:
x = inner-CV ridge MI probe on the PPO GRU at tick 239 (summary.json probes.arms.<arm>.239.probe_mi_bits),
y = T_eval gain over base_cc with 90% CI (eval.T_eval.arms.<arm>.gain_vs_base[_ci90]).
Points without this probe (untrained net, Phase I Arm B, oracle) are left out on purpose.

usage: python make_teaser.py [path/to/neon_e2b/summary.json]
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use(Path(__file__).with_name("paper.mplstyle"))

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "worktrees/figgie-rl/neon/runs/neon_e2b/summary.json"
d = json.loads(src.read_text())
probes, arms = d["probes"]["arms"], d["eval"]["T_eval"]["arms"]

BLUE, VERM, SKY, GREY = "#0072B2", "#D55E00", "#56B4E9", "#666666"  # Okabe-Ito, as in make_figs.py
pts = [  # arm, label, colour (top to bottom)
    ("slot_exog_shaped", "Per-suit values + reward", BLUE),
    ("slot_exog", "Per-suit values", BLUE),
    ("raw_exog", "Raw features", VERM),
    ("slot_trades_shaped", "Trades-only belief, per-suit", SKY),
    ("slot_placebo", "Hand-only control", GREY),
    ("base_cc", "Card counts (baseline)", GREY),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.4), sharey=True, gridspec_kw=dict(wspace=0.08))
ys = range(len(pts))[::-1]
for y, (arm, label, c) in zip(ys, pts):
    bits = probes[arm]["239"]["probe_mi_bits"]
    g = arms[arm]["gain_vs_base"]; lo, hi = arms[arm]["gain_vs_base_ci90"]
    a1.barh(y, bits, color=c, height=0.6)
    a2.barh(y, g, color=c, height=0.6, xerr=[[g - lo], [hi - g]], error_kw=dict(lw=0.8, capsize=2))
for ax in (a1, a2):
    ax.axhspan(ys[2] - 0.45, ys[1] + 0.45, color="#eeeeee", zorder=-1)
    ax.grid(axis="y", visible=False)
a1.set_yticks(list(ys)); a1.set_yticklabels([p[1] for p in pts], fontsize=7)
a1.set_xlim(0, 1.05); a1.set_title("What the agent knows", fontsize=8, loc="left", fontweight="normal")
a1.set_xlabel("Goal-suit information (bits)")
a2.axvline(0, color="#444444", lw=0.8)
from matplotlib.patches import Patch
a2.legend(handles=[Patch(color="#eeeeee", label="same belief, two inputs")], loc="lower right", fontsize=6.5, frameon=False)
a2.set_title("What the agent earns", fontsize=8, loc="left", fontweight="normal")
a2.set_xlabel("Profit gain over baseline (\\$/game)")
out = Path(__file__).with_name("teaser.pdf")
fig.savefig(out)
print("wrote", out)
for arm, *_ in pts:
    print(arm, probes[arm]["239"]["probe_mi_bits"], arms[arm]["gain_vs_base"], arms[arm]["gain_vs_base_ci90"])
