"""Midterm figure: (a) revelation from public trades, (b) linear probes of PPO-v2.

Reads results/proposal_revelation.json and results/probe_ppo_v2.json.

    python scripts/midterm_figure.py [proposals/midterm_results.pdf]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "proposals/midterm_results.pdf"
rev = json.loads(Path("results/proposal_revelation.json").read_text())
pr = json.loads(Path("results/probe_ppo_v2.json").read_text())

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True)
GREY, BLUE, ORANGE, GREEN = "#555555", "#1f77b4", "#ff7f0e", "#2ca02c"

a.plot(rev["t"], [100 * v for v in rev["hand_acc"]], color=GREY, marker="o", ms=2.5, label="exact posterior, own dealt hand")
a.plot(rev["t"], [100 * v for v in rev["flow_acc"]], color=BLUE, marker="o", ms=2.5, label="+ public trades (logistic regression)")
a.set_title(f"(a) Revelation, heuristic table (n = {rev['n']})", fontsize=8.5)

b.plot(pr["t"], [100 * v for v in pr["hand_acc"]], color=GREY, marker="o", ms=2.5, label="exact posterior, own dealt hand")
b.plot(pr["t"], [100 * v for v in pr["count_acc"]], color=GREEN, ls="--", marker="o", ms=2.5, label="card-counting posterior (an input)")
b.plot(pr["t"], [100 * v for v in pr["input_acc"]], color=ORANGE, marker="o", ms=2.5, label="probe on current 69-d input")
b.plot(pr["t"], [100 * v for v in pr["gru_acc"]], color=BLUE, marker="o", ms=2.5, label="probe on 128-d GRU state")
b.set_title(f"(b) Linear probes of PPO-v2 (n = {pr['n']})", fontsize=8.5)

for ax in (a, b):
    ax.axhline(25, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("tick (of 240)")
    ax.set_xlim(0, 240)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
a.set_ylabel("goal-suit accuracy (%)")
a.text(2, 25.5, "chance", fontsize=6.5, color="grey")
fig.tight_layout()
fig.savefig(out, bbox_inches="tight")
print(out)
