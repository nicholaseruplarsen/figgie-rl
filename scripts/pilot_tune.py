"""Cost pilot for the tuner x optimizer factorial (docs/research_plan.tex, pilot milestone).

Exploratory: it measures core-hours per trial, not which configuration wins. It is outside the
2x2 comparison, since only runs through tsinghua-rl's iig/tune count there.

Phase A, worker scaling: one PPO run (none core, flow features, BC init) for a few updates at
2, 4 and 8 env workers, one at a time on an otherwise idle box.
Phase B, tuning: rounds of concurrent trials proposed by a PROTEIN-style stub (cost-aware: each
trial draws its own length from FIDELITIES, and after round 1 a Gaussian process over
(log hyperparameters, log length) picks candidates by upper confidence bound per unit cost,
sampled around the best trials). Each trial is train + a seat-rotated eval against
FV + Bayes + MM. CPU time is read from wait4's rusage, which includes the env workers.

  python scripts/pilot_tune.py --init bc.pt --out out --ncpu 32
  SMOKE=1 python scripts/pilot_tune.py --init bc.pt --out out --ncpu 2
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS_PER_UPDATE = 32 * 512
# Shared core space (lr, entropy, clip, GAE lambda, value coef): default x 2^U(-2, 2), clip and lambda bounded.
SPACE = {"lr": 2.5e-4, "ent": 0.01, "clip": 0.2, "lam": 0.95, "vf": 0.5}
BOUNDS = {"clip": (0.05, 0.5), "lam": (0.8, 0.99)}


def sample(rng, around=None, scale=2.0):
    p = {}
    for k, d in SPACE.items():
        base = math.log2(around[k]) if around else math.log2(d)
        p[k] = 2 ** (base + rng.uniform(-scale, scale))
        if k in BOUNDS:
            lo, hi = BOUNDS[k]
            p[k] = min(max(p[k], lo), hi)
    return p


def feats(p, upd):
    return [math.log(p[k]) for k in SPACE] + [math.log(upd)]


def propose(rng, done, n, fids):
    """PROTEIN-style stub: random in round 1, then GP-UCB per unit cost around the top trials."""
    scored = [t for t in done if t.get("score") is not None]
    if len(scored) < 4:
        return [(sample(rng), fids[j % len(fids)]) for j in range(n)]
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    X = np.array([feats(t["params"], t["updates"]) for t in scored])
    y = np.array([t["score"] for t in scored])
    mu, sd = y.mean(), y.std() + 1e-9
    gp = GaussianProcessRegressor(Matern(nu=2.5) + WhiteKernel(), normalize_y=False).fit(X, (y - mu) / sd)
    top = sorted(scored, key=lambda t: -t["score"])[:3]
    cand = [(sample(rng, rng.choice(top)["params"], 1.0), rng.choice(fids)) for _ in range(400)]
    m, s = gp.predict(np.array([feats(p, u) for p, u in cand]), return_std=True)
    ucb = (m + s) / np.log1p([u for _, u in cand])          # cost-aware: per log-cost
    # A pilot must measure every length, so slots cycle through the fidelities and each takes its best candidate.
    out, used = [], set()
    for j in range(n):
        u = fids[j % len(fids)]
        best = max((i for i, (_, cu) in enumerate(cand) if cu == u and i not in used), key=lambda i: ucb[i])
        used.add(best)
        out.append(cand[best])
    return out


def run_timed(cmd, log):
    """Run cmd, return (wall s, cpu s incl. waited-for children)."""
    t0 = time.time()
    with open(log, "w") as f:
        pr = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    return pr, t0


def finish(pr, t0):
    _, status, ru = os.wait4(pr.pid, 0)
    pr.returncode = os.waitstatus_to_exitcode(status)
    return time.time() - t0, ru.ru_utime + ru.ru_stime, pr.returncode


def train_cmd(a, out, upd, procs, p, extra):
    return [sys.executable, "-m", "figgie.rl.train", "ppo", "--arch", "none", "--flow", "--seed", "0",
            "--device", "cpu", "--init", a.init, "--lift-alpha", "0.1", "--updates", str(upd),
            "--procs", str(procs), "--threads", "1", "--out", out,
            "--lr", f"{p['lr']:.3g}", "--ent", f"{p['ent']:.3g}", "--clip", f"{p['clip']:.3g}",
            "--lam", f"{p['lam']:.4g}", "--vf", f"{p['vf']:.3g}"] + extra


def eval_cmd(ckpt, games, out):
    return [sys.executable, os.path.join(ROOT, "scripts", "pilot_eval.py"), "--ckpt", ckpt,
            "--games", str(games), "--out", out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ncpu", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=4.0, help="start no round that would end after this")
    a = ap.parse_args()
    t_start = time.time()
    smoke = os.environ.get("SMOKE") == "1"
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    if smoke:
        fids, extra, games, scal, rounds, batch, scal_upd = [2, 3], ["--envs", "4", "--steps", "64",
                                                                    "--warmup", "1", "--snap-every", "1"], 8, [2], 1, 2, 2
    else:
        fids, extra, games, scal, rounds, batch, scal_upd = [125, 250, 500], [], 500, [2, 4, 8], a.rounds, \
            max(1, (a.ncpu - 2) // 3), 30
    steps_per_update = (4 * 64) if smoke else STEPS_PER_UPDATE
    log = open(os.path.join(a.out, "trials.jsonl"), "a")

    # Phase A: worker scaling, one run at a time.
    scaling = []
    for procs in scal:
        d = os.path.join(a.out, f"scale_p{procs}")
        pr, t0 = run_timed(train_cmd(a, d, scal_upd, procs, SPACE, extra), os.path.join(a.out, f"scale_p{procs}.log"))
        wall, cpu, rc = finish(pr, t0)
        secs = [json.loads(l)["sec"] for l in open(os.path.join(d, "train_log.jsonl")) if '"sec"' in l] if rc == 0 else []
        row = dict(phase="scaling", procs=procs, updates=scal_upd, rc=rc, wall_s=round(wall, 1), cpu_s=round(cpu, 1),
                   sec_per_update_median=float(np.median(secs[len(secs) // 3:])) if secs else None)
        scaling.append(row)
        log.write(json.dumps(row) + "\n"); log.flush()

    # Phase B: tuning rounds, `batch` trials at once, 2 env workers + learner each.
    done, tid = [], 0
    t_phase = time.time()
    last_round = 0.0
    for r in range(rounds):
        if time.time() - t_start + last_round > a.max_hours * 3600:
            break
        t_round = time.time()
        props = propose(rng, done, batch, fids)
        running = []
        for p, upd in props:
            d = os.path.join(a.out, f"trial_{tid:03d}")
            pr, t0 = run_timed(train_cmd(a, d, upd, 2, p, extra), d + ".train.log")
            running.append(dict(id=tid, round=r, params=p, updates=upd, dir=d, pr=pr, t0=t0))
            tid += 1
        for t in running:
            t["train_wall_s"], t["train_cpu_s"], t["train_rc"] = finish(t.pop("pr"), t.pop("t0"))
        evals = []
        for t in running:
            if t["train_rc"] == 0:
                pr, t0 = run_timed(eval_cmd(os.path.join(t["dir"], "final.pt"), games, t["dir"] + "/eval.json"),
                                   t["dir"] + ".eval.log")
                evals.append((t, pr, t0))
        for t, pr, t0 in evals:
            t["eval_wall_s"], t["eval_cpu_s"], t["eval_rc"] = finish(pr, t0)
            if t["eval_rc"] == 0:
                e = json.load(open(t["dir"] + "/eval.json"))
                t["score"], t["score_ci95"] = e["profit"], e["ci95"]
        for t in running:
            t["env_steps"] = t["updates"] * steps_per_update
            t["core_h"] = round((t["train_cpu_s"] + t.get("eval_cpu_s", 0)) / 3600, 4)
            t.pop("dir")
            row = dict(phase="tune", **t)
            done.append(row)
            log.write(json.dumps(row) + "\n"); log.flush()
        last_round = time.time() - t_round
    wall_tune = time.time() - t_phase

    ok = [t for t in done if t.get("score") is not None]
    msteps = sum(t["env_steps"] for t in ok) / 1e6
    coreh = sum(t["core_h"] for t in ok)
    by_fid = {}
    for u in fids:
        ts = [t for t in ok if t["updates"] == u]
        if ts:
            by_fid[str(u)] = dict(n=len(ts), core_h_mean=round(float(np.mean([t["core_h"] for t in ts])), 4),
                                  train_wall_s_mean=round(float(np.mean([t["train_wall_s"] for t in ts])), 1))
    summary = dict(
        pilot=dict(
            core_h_per_Msteps=round(coreh / msteps, 4) if msteps else None,
            plan_core_h_per_Msteps=round(5 / 32.768, 4),       # plan: 5 core-h per 2000-update none run
            ratio_to_plan=round(coreh / msteps / (5 / 32.768), 3) if msteps else None,
            core_h_per_cell_at_B=round(coreh / msteps * 4500, 1) if msteps else None,   # B = 4.5e9 steps
            n_trials=len(done), n_ok=len(ok), tune_wall_h=round(wall_tune / 3600, 3),
            box_core_utilisation=round(coreh / (wall_tune / 3600 * a.ncpu), 3) if wall_tune else None,
            by_fidelity=by_fid, scaling=scaling,
            best=max(ok, key=lambda t: t["score"])["params"] if ok else None,
            evaluator="core-h = rusage utime+stime of train (incl. env workers) + eval, per trial",
            ncpu=a.ncpu, smoke=smoke, job=os.environ.get("COMPUTE_JOB", "local")))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from provenance import provenance, flag_exploratory
    summary["_provenance"] = provenance(script="scripts/pilot_tune.py")
    summary = flag_exploratory(summary, always=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary["pilot"]), flush=True)


if __name__ == "__main__":
    main()
