"""One-matchup eval for the cost pilot: a PPO checkpoint against FV + Bayes + MM, seats rotated,
deterministic actions, single process (the pilot runs many of these side by side)."""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cs224r import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--game-offset", type=int, default=0, help="first game seed (E2 uses fresh games: 100000)")
    a = ap.parse_args()
    res = [run((["PPO", "FairValue", "Bayesian", "MarketMaker"], g, a.ckpt)) for g in range(a.game_offset, a.game_offset + a.games)]
    p = [r["PPO@0"][0] for r in res]
    out = dict(profit=round(st.mean(p), 2), ci95=round(1.96 * st.stdev(p) / len(p) ** .5, 2), games=len(p), game_offset=a.game_offset,
               matchup="vs FV + Bayes + MM")
    json.dump(out, open(a.out, "w"))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from figgie import runlog                  # FIGGIE_RSRUN=1: eval summary as an rsrun run beside a.out
    rs = runlog.start(os.path.splitext(a.out)[0], vars(a), prefix_run="eval-")
    rs.log({"eval/profit": out["profit"], "eval/profit_ci95": out["ci95"]}, step=0)
    rs.finish(summary={k: v for k, v in out.items() if k != "per_game"})


if __name__ == "__main__":
    main()
