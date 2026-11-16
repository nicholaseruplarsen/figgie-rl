"""E2 of docs/research_plan.tex: confirmation runs of fixed arms on fresh seeds.

Arms: each E1 cell's best configuration (from its summary.json) and the current recipe (Adam,
Figgie defaults) for none and minGRU, so 10 arms. Each (arm, seed) trains 2000 updates from the
core's BC init and is scored like E1 (scripts/pilot_eval.py, 4000 games vs FV + Bayes + MM) but on fresh game
seeds (100000 on), so the selection games of E1 are not reused.
Resumable: a run whose eval.json exists is skipped. Results: <out>/<arm>/s<seed>/eval.json.

    python scripts/e2_confirm.py --e1 e1_out --init-none bc_none.pt --init-mingru bc_mingru.pt \
        --iig iig_pin --seeds 1 2 3 4 5 --out e2_out --batch 3
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from e1_tune import DEFAULTS, FLAGS, STEPS_PER_UPDATE  # noqa: E402


def arms(e1_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(e1_dir, "*", "summary.json"))):
        s = json.load(open(f))
        assert s["complete"], f"{f}: E1 cell not complete"
        out[s["cell"]] = dict(core=s["core"], opt=s["opt"], params=s["best"]["params"])
    for core in ("none", "mingru"):
        out[f"{core}-recipe"] = dict(core=core, opt="adam", params=dict(DEFAULTS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--init-none", required=True)
    ap.add_argument("--init-mingru", required=True)
    ap.add_argument("--iig", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arm names")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=1e9)
    a = ap.parse_args()
    smoke = os.environ.get("SMOKE") == "1"
    extra = ["--envs", "4", "--steps", "64", "--warmup", "1", "--snap-every", "1"] if smoke else []
    if smoke:
        a.updates, a.games = 2, 8
    A = arms(a.e1)
    names = a.arms or list(A)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(arms=A, seeds=a.seeds, updates=a.updates, games=a.games), open(os.path.join(a.out, "arms.json"), "w"), indent=1)
    todo = [(n, s) for s in a.seeds for n in names if not os.path.exists(os.path.join(a.out, n, f"s{s}", "eval.json"))]
    t_start, running = time.time(), {}
    while todo or running:
        while todo and len(running) < a.batch and time.time() - t_start < a.max_hours * 3600:
            n, s = todo.pop(0)
            arm, d = A[n], os.path.join(a.out, n, f"s{s}")
            os.makedirs(d, exist_ok=True)
            cmd = [sys.executable, "-m", "figgie.rl.train", "ppo", "--arch", arm["core"], "--flow", "--seed", str(s),
                   "--device", "cpu", "--init", getattr(a, f"init_{arm['core']}"), "--lift-alpha", "0.1",
                   "--updates", str(a.updates), "--procs", "2", "--threads", "1", "--out", d,
                   "--opt", arm["opt"], "--iig-path", a.iig]
            for k, v in arm["params"].items():
                cmd += [FLAGS[k], f"{v:.6g}"]
            pr = subprocess.Popen(cmd + extra, stdout=open(d + ".train.log", "w"), stderr=subprocess.STDOUT, cwd=ROOT)
            running[pr.pid] = dict(d=d, stage="train", t0=time.time())
        if not running:
            break
        pid, status, ru = os.wait4(-1, 0)
        r = running.pop(pid, None)
        if r is None:
            continue
        rc = os.waitstatus_to_exitcode(status)
        rec = dict(stage=r["stage"], rc=rc, wall_s=round(time.time() - r["t0"], 1), cpu_s=round(ru.ru_utime + ru.ru_stime, 1))
        with open(os.path.join(r["d"], "cost.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        for fn in os.listdir(r["d"]):
            if fn.startswith("update_") and fn.endswith(".pt"):
                os.remove(os.path.join(r["d"], fn))
        if r["stage"] == "train" and rc == 0:
            pr = subprocess.Popen([sys.executable, os.path.join(ROOT, "scripts", "pilot_eval.py"), "--ckpt",
                                   os.path.join(r["d"], "final.pt"), "--games", str(a.games), "--game-offset", "100000", "--out",
                                   os.path.join(r["d"], "eval.json")],
                                  stdout=open(r["d"] + ".eval.log", "w"), stderr=subprocess.STDOUT, cwd=ROOT)
            running[pr.pid] = dict(d=r["d"], stage="eval", t0=time.time())
    print(json.dumps(dict(done=len(glob.glob(os.path.join(a.out, "*", "s*", "eval.json"))))), flush=True)


if __name__ == "__main__":
    main()
