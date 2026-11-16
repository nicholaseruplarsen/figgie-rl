"""E1 of docs/research_plan.tex: one tuning cell, PPO x core x tuner x optimizer, on Figgie.

The tuners and the optimizer factory are tsinghua-rl's iig/tune (pinned copy under --iig), the
same code as the IIG half. Each trial trains PPO (flow features, BC init, PFSP self-play, 2 env
workers) for its length and is scored on 4000 duplicate games against FV + Bayes + MM
(scripts/pilot_eval.py), seat 0, the same game seeds for every trial. Score = mean $/game.

Random search: every trial 1000 updates (16.4 M steps). PROTEIN: lengths 500/1000/2000 cycled
over the slots of a batch (iig/tune `costs`); early stopping is off (config). A cell stops
proposing once finished + running steps reach --budget (B_F = 8e8). Up to --batch trials run at
once; finished trials are told to the tuner as soon as a batch completes.

Resumable: trials.jsonl is replayed through a fresh tuner with the same seed (asks are
deterministic given the history), so a lease that ends mid-cell loses only the running trials.

    python scripts/e1_tune.py --core none --tuner protein --opt muon --init bc.pt --iig iig_pin \
        --out out/none-protein-muon --batch 3
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS_PER_UPDATE = 32 * 512
# Figgie's benchmark values (the CS224R recipe as reproduced, figgie/rl/train.py defaults).
DEFAULTS = dict(learning_rate=2.5e-4, ent_coef=0.01, clip_coef=0.2, gae_lambda=0.95, vf_coef=0.5, max_grad_norm=0.5)
FLAGS = dict(learning_rate="--lr", ent_coef="--ent", clip_coef="--clip", gae_lambda="--lam", vf_coef="--vf",
             max_grad_norm="--max-grad-norm")


def train_cmd(a, t, d):
    upd = t["cost_steps"] // a.steps_per_update
    cmd = [sys.executable, "-m", "figgie.rl.train", "ppo", "--arch", a.core, "--flow", "--seed", str(a.train_seed),
           "--device", "cpu", "--init", a.init, "--lift-alpha", "0.1", "--updates", str(upd),
           "--procs", "2", "--threads", "1", "--out", d, "--opt", a.opt, "--iig-path", a.iig]
    for k, v in t["params"].items():
        cmd += [FLAGS[k], f"{v:.6g}"]
    return cmd + a.extra


def eval_cmd(a, d):
    return [sys.executable, os.path.join(ROOT, "scripts", "pilot_eval.py"), "--ckpt", os.path.join(d, "final.pt"),
            "--games", str(a.games), "--out", os.path.join(d, "eval.json")]


def anytime(done, checkpoints):
    """Best score so far against cumulative env steps, trials in id order (finished ones only)."""
    curve, best, steps = [], -math.inf, 0
    for t in sorted(done, key=lambda t: t["id"]):
        steps += t["cost_steps"]
        if t.get("score") is not None:
            best = max(best, t["score"])
        curve.append((steps, best if best > -math.inf else None, t["id"]))
    at = {}
    for c in checkpoints:
        pts = [b for s, b, _ in curve if s <= c]
        at[f"{c:.0e}"] = pts[-1] if pts and steps >= c else None
    return curve, at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", choices=["none", "mingru"], required=True)
    ap.add_argument("--tuner", choices=["random", "protein"], required=True)
    ap.add_argument("--opt", choices=["adamw", "muon"], required=True)
    ap.add_argument("--init", required=True, help="BC checkpoint of this core")
    ap.add_argument("--iig", required=True, help="dir holding the pinned tsinghua-rl iig/ package")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=8e8)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0, help="tuner seed")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=1e9, help="start no new trial after this (lease end)")
    a = ap.parse_args()
    smoke = os.environ.get("SMOKE") == "1"
    a.steps_per_update = 4 * 64 if smoke else STEPS_PER_UPDATE
    a.extra = ["--envs", "4", "--steps", "64", "--warmup", "1", "--snap-every", "1"] if smoke else []
    lengths = [2, 3] if smoke else ([1000] if a.tuner == "random" else [500, 1000, 2000])
    if smoke:
        a.games, a.budget = 8, 4 * 3 * a.steps_per_update
    sys.path.insert(0, a.iig)
    from iig.tune import load
    from iig.tune.tuners import make_tuner
    cfg = load()
    assert cfg["protein"]["early_stopping"] is False
    checkpoints = [c for c in cfg["anytime"]["checkpoints"] if c <= a.budget] if not smoke else [a.budget]
    tuner = make_tuner(a.tuner, DEFAULTS, seed=a.seed, costs=[u * a.steps_per_update for u in lengths])
    os.makedirs(a.out, exist_ok=True)
    tpath = os.path.join(a.out, "trials.jsonl")
    t_start = time.time()

    # Replay finished trials through the tuner, in the order they were asked and told.
    logged = [json.loads(l) for l in open(tpath)] if os.path.exists(tpath) else []
    done, queue = [], []
    for row in logged:
        if row["kind"] == "ask":
            got = tuner.ask(len(row["ids"]))
            assert [g["id"] for g in got] == row["ids"], "replay diverged: tuner asks differ from the log"
            for g, p in zip(got, row["params"]):
                assert all(abs(g["params"][k] - p[k]) <= 1e-12 * max(1, abs(p[k])) for k in p), "replay params differ"
            queue += got
        elif row["kind"] == "tell":
            t = next(q for q in queue if q["id"] == row["id"])
            tuner.tell(t, row["score"], row["cost_steps"])
            t.update(row)
            done.append(t)
    queue = [q for q in queue if q["id"] not in {d["id"] for d in done}]   # asked but never finished: rerun
    log = open(tpath, "a")

    def write(row):
        log.write(json.dumps(row) + "\n"); log.flush()

    def used():
        return sum(t["cost_steps"] for t in done) + sum(t["cost_steps"] for t in running.values())

    running = {}                                     # pid -> trial
    while True:
        while len(running) < a.batch and time.time() - t_start < a.max_hours * 3600:
            if not queue:
                if used() >= a.budget:
                    break
                got = tuner.ask(a.batch)
                write(dict(kind="ask", ids=[g["id"] for g in got], params=[g["params"] for g in got],
                           exponents=[g["exponents"] for g in got], cost_steps=[g["cost_steps"] for g in got]))
                queue += got
            if used() >= a.budget:                   # budget spent: asked-but-unstarted trials are dropped
                queue.clear()
                break
            t = queue.pop(0)
            d = os.path.join(a.out, f"trial_{t['id']:03d}")
            os.makedirs(d, exist_ok=True)
            f = open(d + ".train.log", "w")
            pr = subprocess.Popen(train_cmd(a, t, d), stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
            running[pr.pid] = dict(trial=t, dir=d, t0=time.time(), stage="train", pr=pr)
        if not running:
            break
        pid, status, ru = os.wait4(-1, 0)            # whichever finishes first
        r = running.pop(pid, None)
        if r is None:
            continue
        rc = os.waitstatus_to_exitcode(status)
        t = r["trial"]
        cpu = ru.ru_utime + ru.ru_stime
        if r["stage"] == "train":
            t["train_rc"], t["train_wall_s"], t["train_cpu_s"] = rc, round(time.time() - r["t0"], 1), round(cpu, 1)
            if rc == 0:
                pr = subprocess.Popen(eval_cmd(a, r["dir"]), stdout=open(r["dir"] + ".eval.log", "w"),
                                      stderr=subprocess.STDOUT, cwd=ROOT)
                running[pr.pid] = dict(r, t0=time.time(), stage="eval", pr=pr)
                continue
            score = None
        else:
            t["eval_rc"], t["eval_wall_s"], t["eval_cpu_s"] = rc, round(time.time() - r["t0"], 1), round(cpu, 1)
            e = json.load(open(os.path.join(r["dir"], "eval.json"))) if rc == 0 else {}
            score = e.get("profit")
            t["score_ci95"] = e.get("ci95")
        for fn in os.listdir(r["dir"]):              # keep final.pt, drop intermediate snapshots
            if fn.startswith("update_") and fn.endswith(".pt"):
                os.remove(os.path.join(r["dir"], fn))
        tuner.tell(t, score, t["cost_steps"])
        t["score"] = score
        row = dict(kind="tell", id=t["id"], score=score, cost_steps=t["cost_steps"], params=t["params"],
                   exponents=t["exponents"], **{k: t.get(k) for k in ("train_rc", "train_wall_s", "train_cpu_s",
                                                                       "eval_rc", "eval_wall_s", "eval_cpu_s", "score_ci95")})
        write(row)
        done.append(t)

    # Summary (written every time the runner stops; complete=True once the budget is spent).
    steps = sum(t["cost_steps"] for t in done)
    curve, at = anytime(done, checkpoints)
    ok = [t for t in done if t.get("score") is not None]
    best = max(ok, key=lambda t: t["score"]) if ok else None
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from provenance import flag_exploratory, provenance
    prov = provenance(script="scripts/e1_tune.py", iig=open(os.path.join(a.iig, "IIG_PIN")).read().strip()
                      if os.path.exists(os.path.join(a.iig, "IIG_PIN")) else None)
    ev = f"e1_tune@{prov['code_commit'][:12]}"
    final = best["score"] if best else None
    # P7: first share of the budget after which best-so-far stays within 2 $/game of the final best.
    stab = None
    if final is not None:
        for s, b, _ in curve:
            if b is not None and all(bb is not None and final - bb <= 2.0 for ss, bb, _ in curve if ss >= s):
                stab = round(s / a.budget, 4); break
    cell = f"{a.core}-{a.tuner}-{a.opt}"
    res = dict(cell=cell, core=a.core, tuner=a.tuner, opt=a.opt, budget=a.budget, steps=steps,
               complete=bool(steps >= a.budget),
               n_trials=len(done), n_ok=len(ok), lengths_updates=lengths,
               best=dict(id=best["id"], score=best["score"], ci95=best.get("score_ci95"), params=best["params"],
                         exponents=best["exponents"], updates=best["cost_steps"] // a.steps_per_update) if best else None,
               anytime=at, curve=[dict(steps=s, best=b, id=i) for s, b, i in curve], stabilise_share=stab,
               core_h=round(sum((t.get("train_cpu_s") or 0) + (t.get("eval_cpu_s") or 0) for t in done) / 3600, 3),
               smoke=smoke, evaluator=ev, _provenance=prov)
    res = flag_exploratory(res)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("cell", "steps", "complete", "n_trials", "anytime", "stabilise_share")}), flush=True)


if __name__ == "__main__":
    main()
