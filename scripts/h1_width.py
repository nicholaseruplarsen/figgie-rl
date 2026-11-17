"""H1 at matched width (exploratory check beside #5119): controlled_probe_v2's supervised sequence GRU with
hidden 128 (the PPO cores' width) instead of 64, on the same games (indices [0, --games), recorded by E0 none_s0).
Reports its accuracy minus the last-input probe, as H1 does. Never scored.

    python scripts/h1_width.py --record nets/none_s0.pt --games 10000 --out h1_width/summary.json
"""
import argparse
import json
import sys
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
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.workers)
    chunks = [(a.record, list(range(w, a.games, a.workers))) for w in range(a.workers)]
    with Pool(a.workers) as pool:
        games = sorted((g for c in pool.map(record_chunk, chunks) for g in c), key=lambda g: g[0])
    X = np.stack([g[2] for g in games])
    y = np.array([g[1] for g in games])
    c_in = correct(probe(X[:, -1], y), y)
    c_seq = correct(seq_model(X, y, hidden=a.hidden), y)
    d = c_seq - c_in
    res = dict(record=a.record, n_games=len(y), hidden=a.hidden, acc_input=round(100 * c_in.mean(), 3),
               acc_seq=round(100 * c_seq.mean(), 3), gain=round(100 * d.mean(), 3),
               gain_ci90=[round(100 * v, 3) for v in boot_ci(d, a.boot)])
    from provenance import flag_exploratory, provenance
    res["_provenance"] = provenance(script="scripts/h1_width.py")
    res = flag_exploratory(res, always=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
