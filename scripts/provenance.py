"""Which code produced a result: the git commit of this checkout, or the CODE_VERSION
file written when the code is synced to a machine without .git (compute-01)."""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def provenance(**extra):
    try:
        h = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                           check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
        src = "git"
    except (subprocess.CalledProcessError, FileNotFoundError):
        f = ROOT / "CODE_VERSION"
        h, dirty, src = (f.read_text().split()[0] if f.exists() else "unknown"), None, "CODE_VERSION"
    if os.environ.get("PROVENANCE_NOTE"):          # e.g. a device change mid-run
        extra.setdefault("note", os.environ["PROVENANCE_NOTE"])
    return dict(code_commit=h, dirty=dirty, source=src, stamped=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                stamp="at evaluation", **extra)
