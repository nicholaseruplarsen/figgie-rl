"""Placebo for sealed H1 (#5119): the H-suff sequence-model gain with goal labels shuffled across games.

Same recorder, games, folds and models as controlled_probe_v2.py --seq-model (imported from it,
so the pipeline is identical), but y is permuted once (seed 0) before anything is fitted. With no
real link between inputs and labels, both the sequence GRU and the last-input probe sit at chance
and their gain must be about 0; a gain would mean the pipeline manufactures one (leakage, fold
mixing). Writes placebo.gain with a 90% bootstrap CI.

    python scripts/hsuff_placebo.py --record nets/none_s0.pt --games 10000 --out placebo.json
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_probe import record_chunk  # noqa: E402
from controlled_probe_v2 import boot_ci, correct, probe, seq_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.workers)
    t0 = time.time()
    chunks = [(a.record, list(range(w, a.games, a.workers))) for w in range(a.workers)]
    with Pool(a.workers) as pool:
        games = sorted((g for c in pool.map(record_chunk, chunks) for g in c), key=lambda g: g[0])
    X = np.stack([g[2] for g in games])
    y_true = np.array([g[1] for g in games])
    y = np.random.default_rng(0).permutation(y_true)
    c_in = correct(probe(X[:, -1], y), y)
    c_seq = correct(seq_model(X, y), y)
    d = c_seq - c_in
    pts = lambda v: round(100 * float(v), 3)          # noqa: E731
    from provenance import flag_exploratory, provenance
    prov = provenance(script="scripts/hsuff_placebo.py")
    ev = f"hsuff_placebo@{prov['code_commit'][:12]}"
    res = dict(evaluator=ev, n_games=len(games), labels="permuted across games, seed 0",
               chance=pts(np.bincount(y).max() / len(y)),
               placebo=dict(acc_input=pts(c_in.mean()), acc_seq=pts(c_seq.mean()), gain=pts(d.mean()),
                            gain_ci90=[pts(v) for v in boot_ci(d, a.boot)], evaluator=ev),
               seconds=round(time.time() - t0, 1), _provenance=prov)
    res = flag_exploratory(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["placebo"]), flush=True)


if __name__ == "__main__":
    main()
