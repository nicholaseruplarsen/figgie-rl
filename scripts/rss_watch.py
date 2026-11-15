"""Run a command and log the memory of it and its child processes over time.

Every --every seconds it writes one JSON line to --out with the latest update
number from --train-log, the RSS and PSS of the main process (the learner) and
the summed RSS and PSS of its descendants (the env workers), and the memory use
of the enclosing cgroup. PSS splits shared copy-on-write pages between forked
processes, so learner PSS + worker PSS is close to what the cgroup is charged.

    python scripts/rss_watch.py --out run/rss.jsonl --train-log run/train_log.jsonl -- python -m figgie.rl.train ppo ...
"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def kb(pid, field, path="status"):
    try:
        for line in Path(f"/proc/{pid}/{path}").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def children(pid):
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        try:
            kids = Path(f"/proc/{p}/task/{p}/children").read_text().split()
        except OSError:
            kids = []
        out += [int(k) for k in kids]
        todo += [int(k) for k in kids]
    return out


def cgroup_bytes():
    rel = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    base = Path("/sys/fs/cgroup") / rel.lstrip("/")
    def read(name):
        try:
            return int((base / name).read_text().split()[0])
        except (OSError, ValueError):
            return None
    return read("memory.current"), read("memory.peak")


def last_update(log):
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            lines = f.read().decode(errors="ignore").strip().splitlines()
        return json.loads(lines[-1])["update"]
    except (OSError, ValueError, IndexError, KeyError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("--kill-mb", type=int, default=0, help="stop the run once the cgroup peak passes this")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    p = subprocess.Popen(cmd)
    t0 = time.time()
    with open(a.out, "a") as f:
        while p.poll() is None:
            kids = children(p.pid)
            cur, peak = cgroup_bytes()
            row = dict(t=round(time.time() - t0, 1), update=last_update(a.train_log),
                       learner_rss_mb=kb(p.pid, "VmRSS") // 1024,
                       learner_pss_mb=kb(p.pid, "Pss", "smaps_rollup") // 1024,
                       workers=len(kids),
                       workers_rss_mb=sum(kb(k, "VmRSS") for k in kids) // 1024,
                       workers_pss_mb=sum(kb(k, "Pss", "smaps_rollup") for k in kids) // 1024,
                       cgroup_mb=cur // 2**20 if cur else None, cgroup_peak_mb=peak // 2**20 if peak else None)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if a.kill_mb and (row["cgroup_peak_mb"] or row["cgroup_mb"] or 0) > a.kill_mb:
                for k in [p.pid] + kids:
                    try:
                        os.kill(k, signal.SIGTERM)
                    except OSError:
                        pass
                f.write(json.dumps(dict(t=row["t"], killed=f"cgroup peak {row['cgroup_peak_mb']} MB > {a.kill_mb} MB")) + "\n")
                f.flush()
                p.wait()
                break
            time.sleep(a.every)
        f.write(json.dumps(dict(t=round(time.time() - t0, 1), exit=p.returncode)) + "\n")
    raise SystemExit(p.returncode)


if __name__ == "__main__":
    main()
