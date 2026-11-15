"""Sealed-prediction scores for docs/PREREGISTRATION.md: one JSON per memory core.

The predictions are for the mean over seeds 0 and 1 of the last-hidden-state probe
accuracy at tick 239. For each core this writes runs/memory/prereg/<arch>.json only once
probe.json exists for both seeds, atomically (temp file, then rename), and never includes
seed 2. A file that is already written is left alone.

    python scripts/prereg_aggregate.py
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("/Users/nicho/Desktop/Machine Learning (80245013)/figgie-rl/runs")
OUT = RUNS / "memory" / "prereg"
SOURCES = {
    "none": [RUNS / "memory/none_s0", RUNS / "memory/none_s1"],
    "mingru": [RUNS / "memory/mingru_s0", RUNS / "memory/mingru_s1"],
    "gru": [RUNS / "memory/gru_s0", RUNS / "memory/gru_s1"],
    "transformer": [RUNS / "memory-gpu/transformer_s0", RUNS / "memory-diag/transformer_s1"],
}


def probe_code_commit(code_commit):
    """Last commit at or before code_commit that changed the probe script."""
    return subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", code_commit, "--",
                           "scripts/probe_ppo.py"], capture_output=True, text=True, check=True).stdout.strip()


def aggregate(arch):
    dst = OUT / f"{arch}.json"
    paths = [d / "probe.json" for d in SOURCES[arch]]
    if dst.exists() or not all(p.exists() for p in paths):
        return None
    probes = [json.loads(p.read_text()) for p in paths]
    for p in probes:
        assert p["t"][-1] == 239, "probe must end at tick 239"
    commits = [probe_code_commit(p["_provenance"]["code_commit"]) for p in probes]
    newest, note = commits[-1], None
    if len(set(commits)) > 1:
        # Allowed only if the probe script differs by the provenance stamp alone.
        newest = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", *set(commits), "--",
                                 "scripts/probe_ppo.py"], capture_output=True, text=True, check=True).stdout.strip()
        diff = subprocess.run(["git", "-C", str(ROOT), "diff", "-U0", *sorted(set(commits) - {newest}), newest, "--",
                               "scripts/probe_ppo.py"], capture_output=True, text=True, check=True).stdout
        changed = [l for l in diff.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        assert all("provenance" in l for l in changed), f"{arch}: seeds were probed with different probe code"
        note = (f"seeds were probed at probe-script commits {commits}, which differ only by the "
                f"provenance stamp, so the probe computation is identical")
    res = dict(arch=arch,
               probe_hidden_mean_s0s1=round(100 * sum(p["gru_acc"][-1] for p in probes) / 2, 2),
               per_seed=[round(100 * p["gru_acc"][-1], 2) for p in probes],
               evaluator_commit=newest, seeds=[0, 1], sources=[str(p) for p in paths])
    if note:
        res["evaluator_note"] = note
    OUT.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=OUT, prefix=f".{arch}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(res, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, dst)
    return dst


def main():
    for arch in SOURCES:
        dst = aggregate(arch)
        print(arch, dst or "not written (exists already, or a seed is missing)")


if __name__ == "__main__":
    main()
