"""Reproduce the CS224R evaluation tables on the shared engine.

1. Baseline tournament (§5, Fig. 1): Bayesian, FairValue, MarketMaker,
   Conservative at fixed seats, 200 games (we also run 2000 for tighter CIs).
2. Table 1: PPO (deterministic argmax) vs the six standard matchups,
   1000 games each, seats rotated; profit, win %, aggressive buys per game,
   plus every opponent's profit/win %.

  python scripts/eval_cs224r.py --ppo v1=runs/ppo_v1/final.pt v2=runs/ppo_v2/final.pt
"""
import argparse
import json
import multiprocessing as mp
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from figgie.agents import cs224r as C  # noqa: E402
from figgie.engine import play  # noqa: E402

MATCHUPS = {
    "vs 3xRandom": ["Random", "Random", "Random"],
    "vs FV + 2xR": ["FairValue", "Random", "Random"],
    "vs Bayes + 2xR": ["Bayesian", "Random", "Random"],
    "vs FV + Bayes + R": ["FairValue", "Bayesian", "Random"],
    "vs FV + Cons + MM": ["FairValue", "Conservative", "MarketMaker"],
    "vs FV + Bayes + MM": ["FairValue", "Bayesian", "MarketMaker"],
}
REPORTED_V2 = {  # Table 1 of the report: profit, win, lifts
    "vs 3xRandom": (164, .98, 4.5), "vs FV + 2xR": (121, .33, 1.6), "vs Bayes + 2xR": (112, .10, 1.0),
    "vs FV + Bayes + R": (27, .15, 3.2), "vs FV + Cons + MM": (8, .54, 3.5), "vs FV + Bayes + MM": (2, .36, 6.5)}
REPORTED_V1 = {
    "vs 3xRandom": (118, .89, 0.7), "vs FV + 2xR": (109, .19, 0.4), "vs Bayes + 2xR": (77, .04, 0.3),
    "vs FV + Bayes + R": (43, .13, 0.0), "vs FV + Cons + MM": (6, .33, 0.0), "vs FV + Bayes + MM": (2, .22, 0.0)}
CLS = {"Random": C.RandomAgent, "FairValue": C.FairValue, "Conservative": C.Conservative,
       "MarketMaker": C.MarketMaker, "Bayesian": C.Bayesian}
_NET = {}


def agent(name, seed, ckpt=None):
    if name == "PPO":
        from figgie.rl.policy import PolicyAgent, load
        if ckpt not in _NET:
            import torch
            torch.set_num_threads(1)
            _NET[ckpt] = load(ckpt).net
        return PolicyAgent(_NET[ckpt], deterministic=True)
    return CLS[name](seed=seed)


def run(job):
    names, g, ckpt = job
    rot = g % len(names)
    seated = names[rot:] + names[:rot]
    r = play([agent(n, g * 10 + i, ckpt) for i, n in enumerate(seated)], seed=1_000_000 + g)
    top = max(r.profits)
    nw = sum(p == top for p in r.profits)
    res = {}
    for i, n in enumerate(seated):
        key = f"{n}@{(i - rot) % len(names)}"
        res[key] = (r.profits[i], (r.profits[i] == top) / nw, r.stats[i]["lifts"],
                    r.stats[i]["filled"] / max(r.stats[i]["quotes"], 1))
    return res


def summarize(results):
    out = {}
    for k in results[0]:
        p = [r[k][0] for r in results]
        out[k] = dict(profit=round(st.mean(p), 1), ci95=round(1.96 * st.stdev(p) / len(p) ** .5, 1),
                      win=round(st.mean(r[k][1] for r in results), 3),
                      lifts=round(st.mean(r[k][2] for r in results), 2),
                      quote_accept=round(st.mean(r[k][3] for r in results), 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", nargs="*", default=[])
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/cs224r_eval.json"))
    a = ap.parse_args()
    OUT = {}
    with mp.Pool(a.procs) as pool:
        base = ["Bayesian", "FairValue", "MarketMaker", "Conservative"]
        for n in (200, 2000):
            # fixed seats: no rotation (names list of length 4 but rot forced to 0 via g multiple of 4)
            res = pool.map(run, [(base, 4 * g, None) for g in range(n)])
            OUT[f"baseline tournament fixed seats, {n} games"] = summarize(res)
            print(json.dumps(OUT[f"baseline tournament fixed seats, {n} games"]), flush=True)
        for spec in a.ppo:
            tag, ckpt = spec.split("=")
            for m, opp in MATCHUPS.items():
                res = pool.map(run, [(["PPO"] + opp, g, ckpt) for g in range(a.games)], chunksize=16)
                OUT[f"{tag} {m}"] = summarize(res)
                s = OUT[f"{tag} {m}"]["PPO@0"]
                print(tag, m, json.dumps(s), flush=True)
    json.dump(OUT, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
