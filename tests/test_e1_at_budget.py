import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from e1_at_budget import cell_view, pair_diffs, rank_check, stabilise_share  # noqa: E402


def t(i, steps, score):
    return dict(kind="tell", id=i, cost_steps=steps, score=score)


def test_checkpoint_excludes_trial_crossing_it():
    v = cell_view([t(0, 6e7, 5.0), t(1, 6e7, 9.0)], 1e8)
    assert v["best_at_budget"] == 5.0 and v["at"]["2e+08"] is None


def test_stabilise_share():
    curve = [(2e7, 1.0, 0), (5e7, 9.0, 1), (1e8, 10.0, 2)]
    assert stabilise_share(curve, 1e8) == 0.5


def test_pair_diffs_and_no_flip():
    cells = {f"{c}-{u}-{o}": dict(best_at_budget=10.0 + (u == "protein") * 2, at={"1e+08": 1.0, "2e+08": None,
             "4e+08": None, "8e+08": None}) for c in ("none", "mingru") for u in ("random", "protein") for o in ("adamw", "muon")}
    assert pair_diffs(cells, "tuner", "protein", "random", 1e8)["mean"] == 2.0
    assert rank_check(cells, 1e8)["flips"] == []
