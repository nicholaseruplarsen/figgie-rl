"""Audit of the controlled probe (E0's GRU delta_cond = -1.4): is it the probe or the state?

On the same 3000 recorded games as scripts/controlled_probe.py, per net:
  input            probe on the input alone (the reference)
  cat              probe on [input, state]                          (what delta_cond uses)
  cat_zero         [input, zeros]: must reproduce `input` exactly (the input block is unchanged)
  cat_shuffled     [input, state with rows permuted across games]: same dimensions and scales,
                   no information, so any drop is the cost of the extra dimensions
  cat_randinit     [input, same-arch random-init state]
each with (a) the evaluator's probe: per-dimension standardisation, C = 1 fixed, and
(b) a nested probe: C chosen by an inner 5-fold CV on the outer train folds only, from 1e-3..1e2.
Exploratory, no prediction rides on it.
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
from common_probe import last_hidden, record_chunk  # noqa: E402

CS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def probe(X, y, nested):
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr = (LogisticRegressionCV(Cs=CS, cv=StratifiedKFold(5, shuffle=True, random_state=1), max_iter=3000)
          if nested else LogisticRegression(max_iter=3000, C=1.0))
    clf = make_pipeline(StandardScaler(), lr)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
    return round(100 * float((p.argmax(1) == y).mean()), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from figgie.rl.features import FLOW_DIM
    from figgie.rl.policy import ActorCritic, load_net
    t0 = time.time()
    chunks = [(a.record, list(range(w, a.games, a.workers))) for w in range(a.workers)]
    with Pool(a.workers) as pool:
        games = sorted((g for c in pool.map(record_chunk, chunks) for g in c), key=lambda g: g[0])
    X = np.stack([g[2] for g in games]); y = np.array([g[1] for g in games]); xin = X[:, -1]
    rng = np.random.default_rng(0)
    res = dict(n_games=len(games), input_dim=int(xin.shape[1]), cs=CS)
    res["input"] = {m: probe(xin, y, m == "nested") for m in ("fixed", "nested")}
    nets = {}
    for c in a.ckpt:
        name, path = c.split("=", 1)
        net = load_net(path)
        Z = last_hidden(net, X)
        torch.manual_seed(0)
        Zr = last_hidden(ActorCritic(net.arch, obs_dim=FLOW_DIM, window=128).eval(), X)
        perm = rng.permutation(len(Z))
        row = dict(arch=net.arch, state_dim=int(Z.shape[1]),
                   state_sd=dict(min=float(Z.std(0).min()), median=float(np.median(Z.std(0))), max=float(Z.std(0).max())))
        for m in ("fixed", "nested"):
            nest = m == "nested"
            row[m] = dict(state=probe(Z, y, nest), cat=probe(np.hstack([xin, Z]), y, nest),
                          cat_zero=probe(np.hstack([xin, np.zeros_like(Z)]), y, nest),
                          cat_shuffled=probe(np.hstack([xin, Z[perm]]), y, nest),
                          cat_randinit=probe(np.hstack([xin, Zr]), y, nest))
            row[m]["delta_cond"] = round(row[m]["cat"] - res["input"][m], 3)
            row[m]["delta_cond_vs_shuffled"] = round(row[m]["cat"] - row[m]["cat_shuffled"], 3)
        nets[name] = row
        print(name, json.dumps({m: row[m] for m in ("fixed", "nested")}), flush=True)
    res["nets"] = nets
    res["seconds"] = round(time.time() - t0, 1)
    from provenance import flag_exploratory, provenance
    res["_provenance"] = provenance(script="scripts/probe_audit.py")
    res = flag_exploratory(res, always=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
