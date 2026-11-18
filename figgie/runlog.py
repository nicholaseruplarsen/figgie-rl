"""rsrun logging for Figgie runs (research-mlflow CONVENTIONS v2). Off unless FIGGIE_RSRUN=1.

On: rsrun writes <out>/rsrun/{run.json, metrics.jsonl, artifacts/} with git provenance. Boxes never talk to MLflow
and hold no credentials: the dir comes back with `compute fetch`, then wsl2 runs `rs-ingest <dir>`.
Never raises: if rsrun is missing or refuses (dirty tree, unpushed commit, no git checkout), it warns and logs nothing.
Env: FIGGIE_RSRUN=1, FIGGIE_TRACK (experiment tsinghua/figgie/<track>, default adhoc), FIGGIE_RUN_NAME
(registration id or <ticket>-<short>-s<seed>), FIGGIE_KB_POST (KB post id), FIGGIE_PREDICTIONS (comma ids),
FIGGIE_REPO (git checkout to record, default this repo), and rsrun's own RS_* variables.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class _Off:
    def log(self, *a, **k): pass
    def finish(self, *a, **k): pass
    def artifact(self, *a, **k): pass


def start(out, config, prefix_run=""):
    if os.environ.get("FIGGIE_RSRUN") != "1":
        return _Off()
    try:
        import rsrun
        cfg = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in config.items()}
        seed = cfg.get("seed") if isinstance(cfg.get("seed"), int) else None
        name = os.environ.get("FIGGIE_RUN_NAME") or (prefix_run + Path(out).name)
        preds = [p for p in os.environ.get("FIGGIE_PREDICTIONS", "").split(",") if p]
        run = rsrun.init("tsinghua/figgie/" + os.environ.get("FIGGIE_TRACK", "adhoc"), name, config=cfg,
                         predictions=preds, seed=seed, out=Path(out) / "rsrun",
                         repo=os.environ.get("FIGGIE_REPO", str(REPO)), ticket=os.environ.get("COMPUTE_JOB"))
        c = run.meta["code"]
        url = (c.get("remote_url") or "").removesuffix(".git").replace("git@github.com:", "https://github.com/")
        run.meta["links"] = {"commit_url": f"{url}/commit/{c['commit']}" if url else None,
                             "post": os.environ.get("FIGGIE_KB_POST")}
        run._write()
        return run
    except Exception as e:  # logging must never break a run
        print(f"[runlog] rsrun off: {e}", file=sys.stderr, flush=True)
        return _Off()


def metrics(row, prefix, step_key):
    """Row of a JSONL log -> ({prefix/key: value}, step)."""
    return {f"{prefix}/{k}": v for k, v in row.items() if k != step_key}, row.get(step_key)
