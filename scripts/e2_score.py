"""Scorer for P4-P7 of docs/research_plan.tex (the evaluator the sealed predictions name).

Reads the E1 cell summaries (scripts/e1_tune.py) and the E2 runs (scripts/e2_confirm.py):
  P4 PROTEIN minus random search: mean over the 4 (core, optimizer) pairs of the difference of
     the two winners' E2 means ($/game), with a 90% bootstrap CI (seeds resampled per arm).
  P5 Muon minus AdamW: the same over the 4 (core, tuner) pairs.
  P6 the best tuned none arm (highest E2 mean) minus the none recipe arm, E2 means.
  P7 mean over the 4 PROTEIN cells of the share of B_F after which E1's best-so-far stays
     within 2 $/game of the cell's final best.
Only seeds present for every arm are used (n=10 if seeds 6-10 finished, else n=5).

    python scripts/e2_score.py --e1 e1_out --e2 e2_out --out summary.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

CORES, TUNERS, OPTS = ("none", "mingru"), ("random", "protein"), ("adamw", "muon")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e2", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    e1 = {s["cell"]: s for s in (json.load(open(f)) for f in glob.glob(os.path.join(a.e1, "*", "summary.json")))}
    runs = {}
    for f in glob.glob(os.path.join(a.e2, "*", "s*", "eval.json")):
        arm, seed = f.split(os.sep)[-3], int(f.split(os.sep)[-2][1:])
        runs.setdefault(arm, {})[seed] = json.load(open(f))["profit"]
    names = [f"{c}-{t}-{o}" for c in CORES for t in TUNERS for o in OPTS] + [f"{c}-recipe" for c in CORES]
    missing = [n for n in names if n not in runs]
    assert not missing, f"arms without E2 runs: {missing}"
    seeds = sorted(set.intersection(*(set(runs[n]) for n in names)))
    assert len(seeds) >= 5, f"only {len(seeds)} common seeds"
    S = {n: np.array([runs[n][s] for s in seeds]) for n in names}
    rng = np.random.default_rng(0)
    # Bootstrap means per arm, resampled independently (each arm's seeds are its own training runs).
    B = {n: S[n][rng.integers(0, len(seeds), size=(a.boot, len(seeds)))].mean(1) for n in names}

    def eff(pairs):
        v = float(np.mean([S[x].mean() - S[y].mean() for x, y in pairs]))
        b = np.mean([B[x] - B[y] for x, y in pairs], 0)
        return dict(value=round(v, 3), ci90=[round(float(np.quantile(b, .05)), 3), round(float(np.quantile(b, .95)), 3)],
                    pairs=[[x, y, round(float(S[x].mean() - S[y].mean()), 3)] for x, y in pairs])

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from provenance import flag_exploratory, provenance
    prov = provenance(script="scripts/e2_score.py")
    ev = f"e2_score@{prov['code_commit'][:12]}"
    p4 = eff([(f"{c}-protein-{o}", f"{c}-random-{o}") for c in CORES for o in OPTS])
    p5 = eff([(f"{c}-{t}-muon", f"{c}-{t}-adamw") for c in CORES for t in TUNERS])
    tuned_none = [f"none-{t}-{o}" for t in TUNERS for o in OPTS]
    best_none = max(tuned_none, key=lambda n: S[n].mean())
    p6 = eff([(best_none, "none-recipe")]) | dict(best_arm=best_none)
    shares = [e1[f"{c}-protein-{o}"]["stabilise_share"] for c in CORES for o in OPTS]
    p7 = dict(value=round(float(np.mean([s if s is not None else 1.0 for s in shares])), 4), cells=shares)
    best_mingru = max([f"mingru-{t}-{o}" for t in TUNERS for o in OPTS], key=lambda n: S[n].mean())
    res = dict(evaluator=ev, seeds=seeds, n_seeds=len(seeds),
               arms={n: dict(mean=round(float(S[n].mean()), 3), sd=round(float(S[n].std(ddof=1)), 3),
                             scores=[round(float(x), 3) for x in S[n]]) for n in names},
               best_tuned=dict(none=best_none, mingru=best_mingru),
               p4=p4 | dict(evaluator=ev), p5=p5 | dict(evaluator=ev), p6=p6 | dict(evaluator=ev),
               p7=p7 | dict(evaluator=ev), _provenance=prov)
    res = flag_exploratory(res)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k]["value"] for k in ("p4", "p5", "p6", "p7")}))


if __name__ == "__main__":
    main()
