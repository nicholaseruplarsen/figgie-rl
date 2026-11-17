"""Causal test of the learning-failure hypothesis (docs/PREREG_FINAL.md, after H1 #5119).

Arms (group names), 3 seeds each: none, oracle (true goal suit in the input), b (frozen supervised encoder in the
input), gru, a (GRU plus the aux goal-suit loss). Primary measure: $/game over fresh games (scripts/pilot_eval.py
--game-offset 100000, per_game), seeds averaged per game, 90% bootstrap CI over games.

    oracle_minus_none, b_minus_none, a_minus_gru            $/game differences
    closure_b = b_minus_none / oracle_minus_none,  closure_a = a_minus_gru / oracle_minus_none
    out_b, out_a      output probe: what [input, logits, value] at the last tick add beyond the input alone,
                      arm minus its baseline, points (games recorded by --record, probe indices [0, --games))
    qb, qa            state probe (controlled_probe_v2's delta_cond), arm minus baseline: sanity check only

Pass rule, per arm X in {b, a}: closure_X >= 0.5, the $/game difference's CI90 above 0, and the output-probe
difference's CI90 above 0. If oracle_minus_none's CI90 includes 0, the verdict is "no incentive" for both arms.
The probe is controlled_probe_v2's (nested C); the network's input is rebuilt from the recorded flow features for
each feature kind (oracle one-hot, or the frozen encoder run over the sequence).

    python scripts/causal_eval.py --record r/none_s1/final.pt --ckpt none/s1=... --evals none/s1=e/none_s1.json ...
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
from controlled_probe_v2 import boot_ci, correct, probe  # noqa: E402

PAIRS = {"b": "none", "a": "gru"}


def net_input(net, X, y):
    """The sequence this network saw at play time, rebuilt from the recorded flow features X [G, T, 85]."""
    import torch
    from figgie.rl.features import load_encoder
    feat = net.config.get("feat")
    if feat is None:
        return X
    G, T, _ = X.shape
    if feat in ("oracle", "oracle_placebo"):
        suit = y if feat == "oracle" else np.random.default_rng(0).integers(0, 4, G)
        oh = np.zeros((G, T, 4), np.float32)
        oh[np.arange(G), :, suit] = 1.0
        return np.concatenate([X, oh], 2)
    assert feat.startswith("enc:"), feat
    gru, mu, sd = load_encoder(feat[4:])
    with torch.no_grad():
        hs = np.concatenate([gru(torch.from_numpy(((X[b:b + 200] - mu) / sd).astype(np.float32)))[0].numpy()
                             for b in range(0, G, 200)])
    return np.concatenate([X, hs], 2).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="group/seed=path")
    ap.add_argument("--evals", nargs="+", required=True, help="group/seed=pilot_eval json")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.workers)
    from figgie.rl.policy import load_net

    t0 = time.time()
    chunks = [(a.record, list(range(w, a.games, a.workers))) for w in range(a.workers)]
    with Pool(a.workers) as pool:
        games = sorted((g for c in pool.map(record_chunk, chunks) for g in c), key=lambda g: g[0])
    assert [g[0] for g in games] == list(range(a.games))
    X = np.stack([g[2] for g in games])
    y = np.array([g[1] for g in games])
    xin = X[:, -1]
    c_in = correct(probe(xin, y), y)

    per = {}
    for c in a.ckpt:
        name, path = c.split("=", 1)
        net = load_net(path)
        Z = last_hidden(net, net_input(net, X, y))
        with torch.no_grad():
            zt = torch.from_numpy(Z)
            outs = np.concatenate([net.actor(zt).numpy(), net.critic(zt).numpy()], 1)
        per[name] = dict(group=name.split("/")[0], c_state=correct(probe(np.concatenate([xin, Z], 1), y), y),
                         c_out=correct(probe(np.concatenate([xin, outs], 1), y), y))
        print(name, net.config.get("feat"), round(per[name]["c_state"].mean() - c_in.mean(), 4),
              round(per[name]["c_out"].mean() - c_in.mean(), 4), flush=True)
    money = {}
    for e in a.evals:
        name, path = e.split("=", 1)
        d = json.loads(Path(path).read_text())
        money.setdefault(name.split("/")[0], []).append(np.array(d["per_game"], np.float64))

    groups = sorted({p["group"] for p in per.values()})
    pts = lambda v: round(100 * float(v), 3)                       # noqa: E731
    probe_d = {g: {k: np.mean([p[k] for p in per.values() if p["group"] == g], 0) - c_in for k in ("c_state", "c_out")}
               for g in groups}
    usd = {g: np.mean(v, 0) for g, v in money.items()}               # per eval game, seeds averaged
    lens = {len(v) for v in usd.values()}
    assert len(lens) == 1, "arms were evaluated on different game counts"

    from provenance import flag_exploratory, provenance
    prov = provenance(script="scripts/causal_eval.py")
    EVALUATOR = f"causal_eval@{prov['code_commit'][:12]}"
    rng = np.random.default_rng(0)
    G = lens.pop()
    idx = rng.integers(0, G, size=(a.boot, G))

    def dollars(x, base):
        d = usd[x] - usd[base]
        m = d[idx].mean(1)
        return d, dict(value=round(float(d.mean()), 3), ci90=[round(float(np.quantile(m, q)), 3) for q in (0.05, 0.95)],
                       evaluator=EVALUATOR)

    res = dict(evaluator=EVALUATOR, record=a.record, probe_games=a.games, eval_games=G,
               acc_input=pts(c_in.mean()), n_seeds={g: len(v) for g, v in money.items()})
    do, res["oracle_minus_none"] = dollars("oracle", "none")
    incentive = res["oracle_minus_none"]["ci90"][0] > 0
    om = do[idx].mean(1)
    for x, base in PAIRS.items():
        dx, res[f"{x}_minus_{base}"] = dollars(x, base)
        r = dx[idx].mean(1) / np.where(np.abs(om) > 1e-9, om, np.nan)
        clo = float(dx.mean() / do.mean()) if abs(do.mean()) > 1e-9 else None
        res[f"closure_{x}"] = dict(value=None if clo is None else round(clo, 3),
                                   ci90=[round(float(np.nanquantile(r, q)), 3) for q in (0.05, 0.95)], evaluator=EVALUATOR)
        for k, key in (("c_out", "out"), ("c_state", "q")):
            d = probe_d[x][k] - probe_d[base][k]
            res[f"{key}_{x}"] = dict(value=pts(d.mean()), ci90=[pts(v) for v in boot_ci(d, a.boot)], evaluator=EVALUATOR)
        ok = (clo is not None and clo >= 0.5 and res[f"{x}_minus_{base}"]["ci90"][0] > 0 and res[f"out_{x}"]["ci90"][0] > 0)
        res[f"verdict_{x}"] = "no incentive" if not incentive else "pass" if ok else "fail"
    res["learning_failure_supported"] = bool(incentive and "pass" in (res["verdict_b"], res["verdict_a"]))
    res["groups"] = {g: dict(usd=round(float(usd[g].mean()), 3), state=pts(probe_d[g]["c_state"].mean()),
                             out=pts(probe_d[g]["c_out"].mean())) for g in groups}
    res["seconds"] = round(time.time() - t0, 1)
    res["_provenance"] = prov
    res = flag_exploratory(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("groups", "_provenance")}), flush=True)


if __name__ == "__main__":
    main()
