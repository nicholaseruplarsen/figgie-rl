"""Shuffled-label placebo for the last-hidden-state probe of scripts/probe_ppo.py.

Replays the probe games of each checkpoint (same table, same game seeds, so the same
hidden states), then fits the same 5-fold CV probe at tick 239 twice: on the true goal
suits, which must reproduce probe.json, and on goal suits permuted across games with a
fixed seed. The permuted accuracy should sit at chance (about 25% for four suits).

    python scripts/placebo_probe.py --ckpt runs/memory/mingru_s0/final.pt runs/memory/mingru_s1/final.pt \
        --out placebo_mingru.json
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_ppo import CHECK, probe, run_chunk, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--perm-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    k = CHECK.index(239)
    res = dict(games=a.games, tick=239, perm_seed=a.perm_seed, runs=[])
    for ckpt in a.ckpt:
        chunks = [(ckpt, list(range(w, a.games, a.workers))) for w in range(a.workers)]
        with Pool(a.workers) as pool:
            games = [g for c in pool.map(run_chunk, chunks) for g in c]
        y = np.array([g["goal"] for g in games])
        H = np.stack([g["h"][k] for g in games])
        y_perm = np.random.default_rng(a.perm_seed).permutation(y)
        true_acc = score(probe(H, y), y)[0]
        perm_acc = score(probe(H, y_perm), y_perm)[0]
        res["runs"].append(dict(ckpt=ckpt, true_acc=true_acc, shuffled_acc=perm_acc))
        print(ckpt, round(true_acc, 4), round(perm_acc, 4), flush=True)
    res["shuffled_mean_pct"] = round(100 * float(np.mean([r["shuffled_acc"] for r in res["runs"]])), 2)
    res["true_mean_pct"] = round(100 * float(np.mean([r["true_acc"] for r in res["runs"]])), 2)
    from provenance import provenance, flag_exploratory
    res["_provenance"] = provenance(script="scripts/placebo_probe.py")
    res = flag_exploratory(res)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(a.out, res["true_mean_pct"], res["shuffled_mean_pct"])


if __name__ == "__main__":
    main()
