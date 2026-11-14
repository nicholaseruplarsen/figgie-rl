"""One table per finished memory run: CS224R matchup profit/win and goal-suit probe accuracy.

    python scripts/summarize_memory.py RUN_DIR [RUN_DIR ...]
"""
import json
import sys
from pathlib import Path

for d in map(Path, sys.argv[1:]):
    if not (d / "eval.json").exists() or not (d / "probe.json").exists():
        print(d.name, "unfinished")
        continue
    ev, pr = json.loads((d / "eval.json").read_text()), json.loads((d / "probe.json").read_text())
    rows = {k.split(" ", 1)[1]: v["PPO@0"] for k, v in ev.items()}
    i100, iend = pr["t"].index(100), len(pr["t"]) - 1
    print(f"== {d.name}")
    for m, s in rows.items():
        print(f"  {m:22s} {s['profit']:+7.1f} ±{s['ci95']:4.1f}  win {100 * s['win']:4.0f}%  lifts {s['lifts']:.1f}")
    for k in ("hand", "count", "input", "gru"):
        name = "core" if k == "gru" else k
        print(f"  probe {name:6s} acc t100 {100 * pr[k + '_acc'][i100]:5.1f}%  t239 {100 * pr[k + '_acc'][iend]:5.1f}%"
              f"  bits t239 {pr[k + '_ll'][iend]:.2f}")
