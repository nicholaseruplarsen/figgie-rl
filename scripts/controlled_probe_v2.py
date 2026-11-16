"""Controlled goal-suit probe, version 2 (docs/research_plan.tex: recall check and H-suff).

Changes from v1 (scripts/controlled_probe.py, frozen: it is the evaluator of sealed P1-P3):
  - the logistic probe's C is chosen by an inner 5-fold CV on each outer train fold only
    (Cs 1e-3..1e2), after per-dimension standardisation, instead of C = 1 fixed;
  - random-init nets take the trained nets' input size (89 for the recall check);
  - optional --seq-model: a supervised GRU (64 units) trained on the whole input sequence to
    predict the goal suit, 5-fold CV on the same folds. seq_gain = its accuracy minus the
    last-tick input probe: the information in the public history beyond the last input,
    available to an ideal learner (the H-suff test).
v1's docstring follows.

Controlled goal-suit probe (docs/research_plan.tex, E0 and P1-P3).

One recorder plays the probe table on fixed game seeds and its inputs are saved
(scripts/common_probe.py). Games are sorted by id before the 5-fold split, so the folds
do not depend on --workers. Every network is run over exactly those inputs.

  acc_h      probe on the core output at the last tick
  acc_in     probe on the input alone at the last tick
  acc_cond   probe on [input, core output]
  delta_cond = acc_cond - acc_in      (what the state adds beyond the input)
  delta_rand = acc_h(trained) - acc_h(same-arch random init, mean over --random-seeds)

The none core's state is a function of the current input, so delta_cond(none) is the
placebo: if it lies outside +-1 point, the probe is declared invalid. Memory helps if
delta_cond(mingru) - delta_cond(none) >= 2 points and its 90% bootstrap CI (games
resampled, seeds averaged per game) excludes 0.

Trained nets are named group/seed=path, e.g. --ckpt none/s0=runs/memory/none_s0/final.pt.
--placebo and --treatment name the two groups compared.

    python scripts/controlled_probe.py --ckpt none/s0=... mingru/s0=... --out summary.json
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


def probe(X, y):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=StratifiedKFold(5, shuffle=True, random_state=1),
                                                               max_iter=3000))
    return cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")


def seq_model(X, y, epochs=30, hidden=64, seed=0):
    """Out-of-fold goal-suit probabilities from a GRU trained on the full input sequences."""
    import torch
    from sklearn.model_selection import StratifiedKFold
    torch.manual_seed(seed)
    mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    Y = torch.from_numpy(y.astype(np.int64))
    P = np.zeros((len(y), 4))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        gru = torch.nn.GRU(X.shape[-1], hidden, batch_first=True)
        head = torch.nn.Linear(hidden, 4)
        opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=3e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            for b in torch.randperm(len(tr), generator=g).split(128):
                i = torch.from_numpy(tr)[b]
                out, _ = gru(Xn[i])
                loss = torch.nn.functional.cross_entropy(head(out[:, -1]), Y[i])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            out, _ = gru(Xn[torch.from_numpy(te)])
            P[te] = torch.softmax(head(out[:, -1]), -1).numpy()
    return P



def correct(p, y):
    return (p.argmax(1) == y).astype(np.float64)


def boot_ci(d, n=4000, q=(0.05, 0.95), seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return [float(np.quantile(m, q[0])), float(np.quantile(m, q[1]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="runs/memory/none_s0/final.pt")
    ap.add_argument("--ckpt", nargs="+", required=True, help="group/seed=path")
    ap.add_argument("--random-seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--placebo", default="none")
    ap.add_argument("--treatment", default="mingru")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seq-model", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.workers)
    from figgie.rl.policy import ActorCritic, load_net

    t0 = time.time()
    chunks = [(a.record, list(range(w, a.games, a.workers))) for w in range(a.workers)]
    with Pool(a.workers) as pool:
        games = sorted((g for c in pool.map(record_chunk, chunks) for g in c), key=lambda g: g[0])
    assert [g[0] for g in games] == list(range(a.games))
    lens = {len(g[2]) for g in games}
    assert len(lens) == 1, f"games differ in decision count: {sorted(lens)[:5]}"
    X = np.stack([g[2] for g in games])
    y = np.array([g[1] for g in games])
    xin = X[:, -1]
    c_in = correct(probe(xin, y), y)

    trained = {}
    for c in a.ckpt:
        name, path = c.split("=", 1)
        group = name.split("/")[0]
        net = load_net(path)
        Z = last_hidden(net, X)
        c_h = correct(probe(Z, y), y)
        c_cond = correct(probe(np.concatenate([xin, Z], 1), y), y)
        trained[name] = dict(group=group, arch=net.arch, path=path, c_h=c_h, c_cond=c_cond)
        print(name, net.arch, round(c_h.mean(), 4), round(c_cond.mean() - c_in.mean(), 4), flush=True)

    rand = {}
    for arch in sorted({t["arch"] for t in trained.values()}):
        accs = []
        for s in a.random_seeds:
            torch.manual_seed(s)
            net = ActorCritic(arch, obs_dim=X.shape[-1], window=128).eval()
            accs.append(float(correct(probe(last_hidden(net, X), y), y).mean()))
        rand[arch] = dict(acc_seeds=accs, acc=float(np.mean(accs)))

    pts = lambda v: round(100 * float(v), 3)                       # noqa: E731
    groups = {}
    for g in sorted({t["group"] for t in trained.values()}):
        ts = [t for t in trained.values() if t["group"] == g]
        arch = ts[0]["arch"]
        assert all(t["arch"] == arch for t in ts), f"group {g} mixes archs"
        dc = np.mean([t["c_cond"] for t in ts], 0) - c_in          # per game, seeds averaged
        acc_h = float(np.mean([t["c_h"].mean() for t in ts]))
        groups[g] = dict(arch=arch, n_seeds=len(ts), evaluator=None, acc_h=pts(acc_h),
                         acc_cond=pts(np.mean([t["c_cond"].mean() for t in ts])),
                         delta_cond=pts(dc.mean()), delta_cond_ci90=[pts(v) for v in boot_ci(dc, a.boot)],
                         delta_rand=pts(acc_h - rand[arch]["acc"]),
                         delta_cond_seeds=[pts(t["c_cond"].mean() - c_in.mean()) for t in ts])
        groups[g]["_dc"] = dc

    from provenance import flag_exploratory, provenance
    prov = provenance(script="scripts/controlled_probe_v2.py")
    EVALUATOR = f"controlled_probe_v2@{prov['code_commit'][:12]}"   # the evaluator commit the predictions name
    res = dict(evaluator=EVALUATOR, record=a.record, n_games=len(games), decisions=lens.pop(),
               acc_input=pts(c_in.mean()), random_init={k: dict(acc_seeds=[pts(x) for x in v["acc_seeds"]],
                                                                 acc=pts(v["acc"])) for k, v in rand.items()})
    if a.seq_model:
        c_seq = correct(seq_model(X, y), y)
        d = c_seq - c_in
        res["seq"] = dict(acc=pts(c_seq.mean()), gain=pts(d.mean()), gain_ci90=[pts(v) for v in boot_ci(d, a.boot)],
                          evaluator=EVALUATOR)
    if a.placebo in groups:
        pd = groups[a.placebo]["delta_cond"]
        res["placebo"] = dict(group=a.placebo, delta_cond=pd, valid=bool(abs(pd) <= 1.0), evaluator=EVALUATOR)
    if a.placebo in groups and a.treatment in groups:
        d = groups[a.treatment]["_dc"] - groups[a.placebo]["_dc"]
        ci = [pts(v) for v in boot_ci(d, a.boot)]
        diff = pts(d.mean())
        valid = res["placebo"]["valid"]
        res["memory"] = dict(treatment=a.treatment, placebo=a.placebo, diff=diff, ci90=ci,
                             helps=bool(valid and diff >= 2.0 and ci[0] > 0),
                             verdict=("invalid probe" if not valid else
                                      "memory helps" if diff >= 2.0 and ci[0] > 0 else "not shown"),
                             evaluator=EVALUATOR)
    for g in groups.values():
        g.pop("_dc")
        g["evaluator"] = EVALUATOR
    res["groups"] = groups
    res["seconds"] = round(time.time() - t0, 1)
    res["_provenance"] = prov
    res = flag_exploratory(res)             # COMPUTE_EXPLORATORY=1 (E0, smoke): never scored
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("acc_input", "placebo", "memory") if k in res}), flush=True)


if __name__ == "__main__":
    main()
