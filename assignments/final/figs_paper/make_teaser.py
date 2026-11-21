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
pts = [  # arm, label, colour, label offset (points)
    ("base_cc", "card counts only\n(baseline)", GREY, (8, -16)),
    ("slot_placebo", "hand-only control", GREY, (8, 6)),
    ("slot_trades_shaped", "belief from trades only,\nfed in as per-suit values", SKY, (8, -4)),
    ("raw_exog", "belief fed in as\nraw features", VERM, (8, -10)),
    ("slot_exog", "same belief fed in\nas per-suit values", BLUE, (-95, 4)),
    ("slot_exog_shaped", "…plus reward\nfor using it", BLUE, (6, 2)),
]

fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.axhline(0, color="#444444", lw=0.8, zorder=0)
for arm, label, c, off in pts:
    x = probes[arm]["239"]["probe_mi_bits"]
    y = arms[arm]["gain_vs_base"]
    lo, hi = arms[arm]["gain_vs_base_ci90"]
    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color=c, ms=6, lw=1)
    ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=7, color=c, va="center")
xr, yr = probes["raw_exog"]["239"]["probe_mi_bits"], arms["raw_exog"]["gain_vs_base"]
xs, ys = probes["slot_exog"]["239"]["probe_mi_bits"], arms["slot_exog"]["gain_vs_base"]
ax.axvspan(xr - 0.025, xs + 0.025, color="#eeeeee", zorder=-1)
ax.annotate("", xy=(xs, ys - 1.5), xytext=(xr, yr + 1.5), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
ax.text((xr + xs) / 2 + 0.03, (yr + ys) / 2, f"same belief,\n{xr:.2f} \u2192 {xs:.2f} bits,\n+25 \\$/game", fontsize=7, color=BLUE, va="center")
ax.set_xlim(0, 1.2)
ax.set_ylim(-8, 30)
ax.set_xlabel("Goal-suit information a probe reads from the agent's network (bits, max 2)")
ax.set_ylabel("Profit gain over baseline\n(\\$/game, 90% CI)")
out = Path(__file__).with_name("teaser.pdf")
fig.savefig(out)
print("wrote", out)
for arm, *_ in pts:
    print(arm, probes[arm]["239"]["probe_mi_bits"], arms[arm]["gain_vs_base"], arms[arm]["gain_vs_base_ci90"])
