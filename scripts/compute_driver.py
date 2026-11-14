"""Drive the GPU memory-PPO runs through `compute` (the only way to get a GPU).

One job at a time on the free tier (compute-01). Each job is BC -> PPO -> probe ->
CS224R eval for one (arch, seed), writing to a persistent run dir on compute-01 so
that a job whose lease expires is relaunched and resumes from latest.pt. On
`oom`, `failed` or `stopped` the driver halts and launches nothing more. A finished
run is copied to wsl2, verified with a checksum pass, then deleted on compute-01.

    python scripts/compute_driver.py --only transformer:0 transformer:1 gru:2 none:2 mingru:2 transformer:2
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_queue import job_script  # noqa: E402

HOST = "compute-01"
REMOTE_PY = "/home/nicholas/figgie-mem/.venv/bin/python"
REMOTE_CODE = "/home/nicholas/figgie-mem/code"
REMOTE_RUNS = "/home/nicholas/figgie-mem/runs"


def log(m):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), m, flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def state(job_id, tenant):
    r = sh(["compute", "status", "--all"], env=dict(os.environ, COMPUTE_TENANT=tenant), timeout=300)
    m = re.search(rf"^\s*{job_id}\s+\S+\s+(\S+)", r.stdout, re.M)
    return m.group(1) if m else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", required=True, help="runs in launch order, like transformer:0 gru:2")
    ap.add_argument("--tenant", default="geoff")
    ap.add_argument("--vram", type=float, default=3)
    ap.add_argument("--ram", type=float, default=4)
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--local-root", required=True, help="wsl2 dir for finished runs")
    ap.add_argument("--peer-root", required=True, help="wsl2 CPU queue's root: skip runs it started, claim ours")
    a = ap.parse_args()
    env = dict(os.environ, COMPUTE_TENANT=a.tenant)
    jargs = argparse.Namespace(code=REMOTE_CODE, procs=a.procs, updates=a.updates, window=a.window,
                               device="cuda", learner_threads=1, max_vram_gb=a.vram)
    for spec in a.only:
        arch, seed = spec.split(":")
        name = f"{arch}_s{seed}"
        peer = Path(a.peer_root) / name
        if (peer / "job.sh").exists() or (peer / "DONE").exists():
            log(f"skip {name}: started by the wsl2 CPU queue")
            continue
        peer.mkdir(parents=True, exist_ok=True)
        (peer / "SKIP").write_text("claimed by compute_driver (GPU)\n")
        run = f"{REMOTE_RUNS}/{name}"
        sh(["ssh", HOST, f"mkdir -p {run}"], check=True)
        while True:                                   # one lease per iteration; resume on expiry
            work = Path(tempfile.mkdtemp(prefix=f"vint-{name}-"))
            (work / "job.sh").write_text(job_script(arch, int(seed), run, jargs, py=REMOTE_PY))
            cmd = ["compute", "run", "--tenant", a.tenant, "--tier", "free", "--vram", str(a.vram),
                   "--ram", str(a.ram), "--hours", str(a.hours), "--json", "--workdir", str(work),
                   "--", "bash", "job.sh"]
            r = sh(cmd, env=env, timeout=600)
            m = re.search(r'"id":\s*"(\w+)"', r.stdout)
            if r.returncode != 0 or not m:
                log(f"{name}: not launched ({(r.stdout + r.stderr).strip()[-200:]}), retry in 10 min")
                time.sleep(600)
                continue
            jid = m.group(1)
            log(f"{name}: launched job {jid}")
            st = "running"
            while st in ("running", "queued", "starting", "unknown"):
                time.sleep(300)
                st = state(jid, a.tenant)
            log(f"{name}: job {jid} ended {st}")
            if st == "done":
                break
            if st == "expired":
                continue                              # lease ran out: relaunch, job.sh resumes
            log(f"HALT: {name} ended {st}; launching nothing more until a person restarts the driver")
            return
        dst = Path(a.local_root) / name
        dst.mkdir(parents=True, exist_ok=True)
        src = f"{HOST}:{run}/"
        sh(["rsync", "-a", src, str(dst) + "/"], check=True)
        diff = sh(["rsync", "-a", "-c", "-n", "-i", src, str(dst) + "/"], check=True).stdout.strip()
        if diff or not (dst / "DONE").exists():
            log(f"HALT: copy of {name} not verified ({diff[:200]}); remote kept")
            return
        sh(["ssh", HOST, f"rm -r {shlex.quote(run)}"], check=True)
        log(f"{name}: copied to {dst}, verified, removed from {HOST}")
    log("all runs finished")


if __name__ == "__main__":
    main()
