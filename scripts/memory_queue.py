"""Run the memory-architecture PPO jobs (BC -> PPO -> probe per arch and seed)
as a queue on a shared GPU box.

Before each launch it waits until
  - no process outside this queue is using the GPU (the owner comes first),
  - available RAM stays above --min-free-gb plus the per-job estimate,
and it never runs more than --slots jobs at once. Every job is pinned to the
CPUs in --cpus and to GPU --gpu. A finished job writes DONE in its run dir.

    python scripts/memory_queue.py --root ~/figgie-mem/runs --gpu 1 --cpus 0-5 --slots 3
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ORDER = [("gru", 0), ("none", 0), ("mingru", 0), ("transformer", 0),
         ("gru", 1), ("none", 1), ("mingru", 1), ("transformer", 1),
         ("gru", 2), ("none", 2), ("mingru", 2), ("transformer", 2)]
MINE = ("figgie.rl.train", "probe_ppo.py", "memory_queue.py")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def gpu_foreign(gpu):
    """PIDs on this GPU that do not belong to the queue."""
    uuid = sh(f"nvidia-smi -i {gpu} --query-gpu=uuid --format=csv,noheader")
    pids = sh(f"nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep {uuid} | cut -d, -f2")
    out = []
    for pid in pids.split():
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
        except OSError:
            cmd = ""
        if not any(m in cmd for m in MINE):
            out.append(pid)
    return out


def gpu_free_gb(gpu):
    return float(sh(f"nvidia-smi -i {gpu} --query-gpu=memory.free --format=csv,noheader,nounits") or 0) / 1024


def jobs_running(root):
    """Run dirs with a live job.sh, including jobs started by an earlier queue process."""
    out = set()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        for part in cmd:
            t = part.decode(errors="ignore")
            if t.startswith(str(root)) and t.endswith("/job.sh"):
                out.add(Path(t).parent.name)
    return out


def free_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return 0.0


def job_script(arch, seed, run, args):
    q = shlex.quote
    py, code, R = q(sys.executable), q(str(args.code)), str(run)
    run = q(R)
    p = lambda name: q(f"{R}/{name}")  # noqa: E731
    tr = f"{py} -m figgie.rl.train"
    common = f"--arch {arch} --flow --seed {seed} --window {args.window}"
    dev, th = args.device, args.learner_threads
    return f"""set -e
cd {code}
[ -f {p("bc.pt")} ] || {tr} bc {common} --device {dev} --out {p("bc.pt")} > {p("bc.log")} 2>&1
[ -f {p("final.pt")} ] || {tr} ppo {common} --device {dev} --init {p("bc.pt")} --lift-alpha 0.1 --resume \\
    --updates {args.updates} --procs {args.procs} --threads {th} --max-vram-gb {args.max_vram_gb} --out {run} > {p("ppo.log")} 2>&1
[ -f {p("probe.json")} ] || {py} scripts/probe_ppo.py --ckpt {p("final.pt")} --games 3000 --workers {args.procs} \\
    --out {p("probe.json")} > {p("probe.log")} 2>&1
touch {p("DONE")}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/figgie-mem/runs"))
    ap.add_argument("--code", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--gpu", default="1", help="'' with --device cpu hides every GPU")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--share-gpu", action="store_true",
                    help="launch next to other users when the GPU has --min-free-vram-gb free")
    ap.add_argument("--min-free-vram-gb", type=float, default=4.0)
    ap.add_argument("--max-vram-gb", type=float, default=3.0)
    ap.add_argument("--learner-threads", type=int, default=1)
    ap.add_argument("--cpus", default="0-5")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--job-gb", type=float, default=3.0)
    ap.add_argument("--peer", help="host:root of another queue; skip runs started there and claim ours")
    ap.add_argument("--only", nargs="*", help="subset like gru:0 transformer:1")
    args = ap.parse_args()

    jobs = [(a, s) for a, s in ORDER if not args.only or f"{a}:{s}" in args.only]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running = {}
    log = lambda m: print(time.strftime("%Y-%m-%d %H:%M:%S"), m, flush=True)  # noqa: E731
    while jobs or running or jobs_running(args.root):
        for k, p in list(running.items()):
            if p.poll() is not None:
                log(f"{k} exited {p.returncode}")
                del running[k]
        live = jobs_running(args.root)
        if jobs and jobs[0] and f"{jobs[0][0]}_s{jobs[0][1]}" in live:
            jobs.pop(0)                     # already running under an earlier queue
            continue
        if jobs and len(live | set(running)) < args.slots:
            foreign = gpu_foreign(args.gpu) if args.device == "cuda" and not args.share_gpu else []
            if args.device == "cuda" and args.share_gpu and gpu_free_gb(args.gpu) < args.min_free_vram_gb:
                foreign = [f"only {gpu_free_gb(args.gpu):.1f} GB free"]
            ram = free_gb()
            if foreign:
                log(f"waiting: foreign processes on GPU {args.gpu}: {foreign}")
            elif ram < args.min_free_gb + args.job_gb:
                log(f"waiting: {ram:.1f} GB RAM available")
            else:
                arch, seed = jobs.pop(0)
                run = Path(args.root) / f"{arch}_s{seed}"
                if (run / "DONE").exists() or (run / "SKIP").exists():
                    log(f"skip {run.name}: done or claimed by another machine")
                    continue
                if args.peer:
                    host, root = args.peer.split(":", 1)
                    there = f"{root}/{run.name}"
                    r = subprocess.run(["ssh", host, f"test -e {there}/job.sh || test -e {there}/DONE"])
                    if r.returncode == 0:
                        log(f"skip {run.name}: already started on {host}")
                        continue
                    subprocess.run(["ssh", host, f"mkdir -p {there} && touch {there}/SKIP"], check=True)
                run.mkdir(parents=True, exist_ok=True)
                (run / "job.sh").write_text(job_script(arch, seed, run, args))
                running[run.name] = subprocess.Popen(["nice", "-n", "10", "taskset", "-c", args.cpus,
                                                      "bash", str(run / "job.sh")], env=env)
                log(f"launched {run.name} ({ram:.1f} GB free)")
                time.sleep(120)        # let it allocate before judging RAM again
                continue
        time.sleep(60)
    log("queue empty")


if __name__ == "__main__":
    main()
