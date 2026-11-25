"""E1 read at a common checkpoint (the 1e8 deviation, docs/PREREG_FINAL.md).

Reads each cell's trials.jsonl and applies e1_tune.anytime (the sealed definition: best score among trials,
in id order, whose cumulative cost_steps is at most the checkpoint, defined only if the cell reached it).
Writes per-cell best at every checkpoint, P7's stabilise share at B, the E1 pair differences that P4 and P5
are built from (E1 best-so-far, not the E2 means the sealed P4/P5 use), and a ranking check: whether the
order of the 8 cells at B holds at every later checkpoint all of them reached, and in each pair that has both.

    python scripts/e1_at_budget.py --cells state/e1 random/e1 --budget 1e8 --out e1_at_1e8.json
"""
import argparse
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_tune import anytime  # noqa: E402

CORES, TUNERS, OPTS = ("none", "mingru"), ("random", "protein"), ("adamw", "muon")
CHECKPOINTS = (1e8, 2e8, 4e8, 8e8)


# ---- pure core ----
def tells(lines):
    return [r for r in lines if r.get("kind") == "tell"]


def stabilise_share(curve, budget, tol=2.0):
    pts = [(s, b) for s, b, _ in curve if s <= budget]
    if not pts or pts[-1][1] is None:
        return None
    final = pts[-1][1]
    for s, b in pts:
        if b is not None and all(bb is not None and final - bb <= tol for ss, bb in pts if ss >= s):
            return round(s / budget, 4)
    return None


def cell_view(done, budget):
    curve, at = anytime(done, CHECKPOINTS)
    return dict(steps=sum(t["cost_steps"] for t in done), n_trials=len(done),
                at={k: v for k, v in at.items()}, best_at_budget=at[f"{budget:.0e}"],
                stabilise_share=stabilise_share(curve, budget))


def pair_diffs(cells, key, a, b, budget):
    """Mean over the other two axes of cell(a) - cell(b) at budget, with the 4 differences."""
    out = []
    for c in CORES:
        for t in TUNERS:
            for o in OPTS:
                name = dict(core=c, tuner=t, opt=o)
                if name[key] != a:
                    continue
                o_ = dict(name, **{key: b})
                other = f"{o_['core']}-{o_['tuner']}-{o_['opt']}"
                x, y = cells[f"{c}-{t}-{o}"]["best_at_budget"], cells[other]["best_at_budget"]
                out.append(dict(pair=f"{c}-{t}-{o} vs {other}", diff=round(x - y, 3)))
    return dict(mean=round(sum(d["diff"] for d in out) / len(out), 3), pairs=out)


def rank_check(cells, budget):
    """Does each pair's order at budget hold at every later checkpoint both cells reached?"""
    flips = []
    for x, y in combinations(sorted(cells), 2):
        s0 = cells[x]["at"][f"{budget:.0e}"] - cells[y]["at"][f"{budget:.0e}"]
        for c in CHECKPOINTS:
            if c <= budget:
                continue
            vx, vy = cells[x]["at"][f"{c:.0e}"], cells[y]["at"][f"{c:.0e}"]
            if vx is None or vy is None:
                continue
            if (vx - vy) * s0 < 0:
                flips.append(dict(pair=[x, y], at=f"{c:.0e}", at_budget=round(s0, 3), later=round(vx - vy, 3)))
    comparable = sum(1 for x, y in combinations(sorted(cells), 2) for c in CHECKPOINTS
                     if c > budget and cells[x]["at"][f"{c:.0e}"] is not None and cells[y]["at"][f"{c:.0e}"] is not None)
    return dict(comparable=comparable, flips=flips)


def score(ledgers, budget):
    cells = {n: cell_view(tells(lines), budget) for n, lines in ledgers.items()}
    missing = [n for n, c in cells.items() if c["best_at_budget"] is None]
    p7 = [cells[f"{c}-protein-{o}"]["stabilise_share"] for c in CORES for o in OPTS]
    return dict(budget=budget, cells=cells, missing=missing,
                e1_protein_minus_random=pair_diffs(cells, "tuner", "protein", "random", budget),
                e1_muon_minus_adamw=pair_diffs(cells, "opt", "muon", "adamw", budget),
                p7=dict(value=round(sum(p7) / len(p7), 4) if None not in p7 else None, cells=p7),
                ranking=rank_check(cells, budget))


# ---- shell ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True, help="dirs holding <cell>/trials.jsonl")
    ap.add_argument("--budget", type=float, default=1e8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ledgers = {}
    for d in a.cells:
        for n in sorted(os.listdir(d)):
            f = os.path.join(d, n, "trials.jsonl")
            if os.path.exists(f):
                ledgers[n] = [json.loads(l) for l in open(f)]
    want = [f"{c}-{t}-{o}" for c in CORES for t in TUNERS for o in OPTS]
    assert sorted(ledgers) == sorted(want), f"cells: {sorted(ledgers)}"
    res = score(ledgers, a.budget)
    res["exploratory_note"] = "E1 differences are best-so-far scores, not the E2 means the sealed P4/P5 use"
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("missing", "e1_protein_minus_random", "e1_muon_minus_adamw", "p7")}
                     | {"flips": len(res["ranking"]["flips"]), "comparable": res["ranking"]["comparable"]}, indent=1))


if __name__ == "__main__":
    main()
