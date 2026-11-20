"""Midterm results for the memory agents: a LaTeX table and a probe figure.

Reads every finished run (eval.json + probe.json) under the given roots, groups
runs by memory core, and reports the mean over seeds with the per-seed values.
The probe on the agent's current input is the baseline each representation is
measured against: it shows how much goal-suit signal the observation already
carries without any memory.

    python scripts/memory_report.py --roots runs/memory runs/memory-gpu \
        --tex assignments/midterm/memory_table.tex --fig assignments/midterm/memory_probe.pdf
"""
import argparse
import json
from pathlib import Path

import numpy as np

CORES = [("none", "none"), ("mingru", "minGRU"), ("gru", "GRU"), ("transformer", "transformer")]
STRONG = ["vs FV + Cons + MM", "vs FV + Bayes + MM"]


def load(roots):
    runs = {}
    for root in map(Path, roots):
        for d in sorted(root.glob("*_s*")):
            if (d / "eval.json").exists() and (d / "probe.json").exists():
                arch, seed = d.name.rsplit("_s", 1)
                ev = json.loads((d / "eval.json").read_text())
                runs.setdefault(arch, []).append(dict(
                    seed=int(seed), probe=json.loads((d / "probe.json").read_text()),
                    eval={k.split(" ", 1)[1]: v["PPO@0"] for k, v in ev.items() if not k.startswith("_")}))
    return runs


def cell(vals, cis, pct=False):
    """mean [min-max over seeds] (+-mean per-run 95% CI); values in % if pct."""
    k = 100 if pct else 1
    v = [k * x for x in vals]
    ci = np.mean([k * c for c in cis])
    f = "{:.1f}" if pct else "{:+.1f}"
    g = lambda x: f.format(x).replace("-", "$-$")  # noqa: E731
    s = g(np.mean(v))
    if len(v) > 1:
        s += " [" + g(min(v)) + "--" + g(max(v)) + "]"
    return s + f" ($\\pm${ci:.1f})"


def acc_ci(p, n):
    return 1.96 * (p * (1 - p) / n) ** 0.5


def rows(runs):
    """Per core: seeds, core probe, input probe, profit on the two strong tables (cells as text)."""
    out = []
    for arch, label in CORES:
        rs = runs.get(arch, [])
        if not rs:
            out.append((label, 0, None))
            continue
        n = rs[0]["probe"]["n"]
        core = [r["probe"]["gru_acc"][-1] for r in rs]
        inp = [r["probe"]["input_acc"][-1] for r in rs]
        cells = [cell(core, [acc_ci(x, n) for x in core], pct=True),
                 cell(inp, [acc_ci(x, n) for x in inp], pct=True)]
        for m in STRONG:
            cells.append(cell([r["eval"][m]["profit"] for r in rs], [r["eval"][m]["ci95"] for r in rs]))
        out.append((label, len(rs), cells))
    return out


def table(runs):
    body = []
    for label, n, cells in rows(runs):
        body.append(f"    {label} & {n} & " + (" & ".join(cells) if cells else "\\multicolumn{4}{l}{pending}") + " \\\\")
    return "\n".join([
        "  \\begin{tabular}{@{}lrllll@{}}",
        "    \\toprule",
        "    & & \\multicolumn{2}{c}{Goal-suit probe accuracy (\\%), tick 239} & \\multicolumn{2}{c}{Profit (\\$/game)} \\\\",
        "    \\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
        "    Memory core & Seeds & Last hidden state & Current input & FV+Cons+MM & FV+Bayes+MM \\\\",
        "    \\midrule",
        *body,
        "    \\bottomrule",
        "  \\end{tabular}"])


def anomalies(runs, roots, updates):
    notes = []
    for arch, label in CORES:
        rs = runs.get(arch, [])
        if len(rs) > 1:
            n = rs[0]["probe"]["n"]
            core = [r["probe"]["gru_acc"][-1] for r in rs]
            if max(core) - min(core) > 2 * max(acc_ci(x, n) for x in core):
                notes.append(f"{label}: seeds disagree on the hidden-state probe ({', '.join(f'{100*x:.1f}%' for x in core)})")
            for m in STRONG:
                pr = [r["eval"][m]["profit"] for r in rs]
                ci = max(r["eval"][m]["ci95"] for r in rs)
                if max(pr) - min(pr) > 2 * ci:
                    notes.append(f"{label}: seeds disagree on profit {m} ({', '.join(f'{x:+.1f}' for x in pr)})")
    for root in map(Path, roots):
        for d in sorted(root.glob("*_s*")):
            log = d / "train_log.jsonl"
            if not log.exists():
                continue
            ups = [json.loads(l)["update"] for l in log.read_text().splitlines() if l.strip()]
            resumes = sum(1 for a, b in zip(ups, ups[1:]) if b <= a)
            if resumes:
                notes.append(f"{d.name}: resumed {resumes}x from latest.pt (full state restored when the row says full_state, else weights only)")
            if (d / "DONE").exists() and ups and max(ups) < updates:
                notes.append(f"{d.name}: cut short at {max(ups)} of {updates} updates")
    return notes


def figure(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    colors = {"none": "#555555", "mingru": "#2ca02c", "gru": "#1f77b4", "transformer": "#d62728"}
    any_run = None
    for arch, label in CORES:
        for i, r in enumerate(runs.get(arch, [])):
            p = r["probe"]
            any_run = p
            ax.plot(p["t"], [100 * v for v in p["gru_acc"]], color=colors[arch], lw=1.2, alpha=0.9,
                    label=f"{label}, last hidden state" if i == 0 else None)
            ax.plot(p["t"], [100 * v for v in p["input_acc"]], color=colors[arch], lw=0.8, ls=":", alpha=0.9)
    if any_run:
        ax.plot(any_run["t"], [100 * v for v in any_run["hand_acc"]], color="black", lw=0.8, ls="--",
                label="exact posterior, own dealt hand")
    ax.plot([], [], color="grey", ls=":", lw=0.8, label="same agent, probe on current input")
    ax.set_xlabel("tick (of 240)")
    ax.set_ylabel("goal-suit accuracy (%)")
    ax.set_xlim(0, 240)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--tex", required=True)
    ap.add_argument("--fig", required=True)
    ap.add_argument("--updates", type=int, default=2000)
    a = ap.parse_args()
    runs = load(a.roots)
    Path(a.tex).write_text(table(runs) + "\n")
    figure(runs, a.fig)
    for label, n, cells in rows(runs):
        print(f"{label:12s} n={n}  " + ("  |  ".join(c.replace("$\\pm$", "±").replace("$-$", "-").replace("--", "–") for c in cells) if cells else "pending"))
    for note in anomalies(runs, a.roots, a.updates):
        print("NOTE:", note)


if __name__ == "__main__":
    main()
