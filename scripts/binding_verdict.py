"""Binding gates (scorer v2, from 2026-11-21): any failed gate makes the verdict invalid.

The sealed evaluators (causal_eval.py, belief_value_eval.py, controlled_probe_v2.py) are not changed. This scorer reads
their summary.json, and optionally a placebo run's summary.json, and writes a new file in which every verdict is
"invalid" if any gate failed. It never turns a failed or invalid verdict into a pass.

Gates found in the result (any depth):
    "valid": false                       e.g. controlled_probe_v2 placebo.valid, cue_recall placebo.valid
    "gates": {...: false}                any false member
    "placebo_gate": false | "fail"       belief_v2
    "verdict": "invalid probe"           controlled_probe_v2 memory.verdict
Placebo run (--placebo, a causal_eval summary from PLACEBO=1): for each arm X, the arm's dollar difference
(X_minus_base) must have a CI90 that contains 0. If it doesn't, the gate for X fails. A failed placebo gate for
the oracle arm fails every arm, because the incentive check rests on it.

Verdict fields rewritten: top-level verdict, verdict_<arm>, and any "verdict" inside a nested dict. Booleans
derived from verdicts (learning_failure_supported, helps) are set to false when the result is invalid.

    python scripts/binding_verdict.py --result out/causal/summary.json --placebo out/causal_placebo/summary.json \
        --out out/causal/verdict.json
"""
import argparse
import json
from pathlib import Path

SCORER = "binding_verdict"
DERIVED = ("learning_failure_supported", "helps")


def find_gates(obj, path=""):
    """Yield (path, passed) for every gate in a result."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k == "valid" and isinstance(v, bool):
                yield p, v
            elif k == "gates" and isinstance(v, dict):
                for g, ok in v.items():
                    if isinstance(ok, bool):
                        yield f"{p}.{g}", ok
            elif k == "placebo_gate" and isinstance(v, (bool, str)):
                yield p, (v is True) or (isinstance(v, str) and v.lower() == "pass")
            elif k == "verdict" and v == "invalid probe":
                yield p, False
            if isinstance(v, (dict, list)):
                yield from find_gates(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from find_gates(v, f"{path}[{i}]")


def placebo_gates(pl):
    """Per-arm gate from a PLACEBO=1 causal_eval summary: the dollar difference's CI90 must contain 0."""
    out = {}
    for k, v in pl.items():
        if k.endswith("_minus_none") or k.endswith("_minus_gru"):
            if isinstance(v, dict) and "ci90" in v:
                lo, hi = v["ci90"]
                out[k.split("_minus_")[0]] = bool(lo <= 0.0 <= hi)
    return out


def invalidate(obj):
    if isinstance(obj, dict):
        for k in list(obj):
            v = obj[k]
            if (k == "verdict" or k.startswith("verdict_")) and isinstance(v, str):
                obj[k] = "invalid"
            elif k in DERIVED and isinstance(v, bool):
                obj[k] = False
            elif isinstance(v, (dict, list)):
                invalidate(v)
    elif isinstance(obj, list):
        for v in obj:
            invalidate(v)


def bind(res, placebo=None):
    result_gates = dict(find_gates(res))
    arm_gate = placebo_gates(placebo) if placebo is not None else {}
    gates = result_gates | {f"placebo.{arm}": ok for arm, ok in arm_gate.items()}
    failed = sorted(p for p, ok in gates.items() if not ok)
    out = json.loads(json.dumps(res))
    if not all(result_gates.values()) or not arm_gate.get("oracle", True):
        invalidate(out)                       # a result-level gate or the oracle placebo failed: everything is invalid
    else:
        for arm, ok in arm_gate.items():      # otherwise only the arms whose placebo failed
            if not ok and f"verdict_{arm}" in out:
                out[f"verdict_{arm}"] = "invalid"
        if "learning_failure_supported" in out:
            out["learning_failure_supported"] = bool(out["learning_failure_supported"] and any(
                out.get(k) == "pass" for k in out if k.startswith("verdict_")))
    out["binding"] = dict(scorer=SCORER, gates=gates, failed=failed, invalid=bool(failed))
    if failed and "verdict" not in out:
        out["verdict"] = "invalid"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--placebo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = json.loads(Path(a.result).read_text())
    pl = json.loads(Path(a.placebo).read_text()) if a.placebo else None
    out = bind(res, pl)
    try:
        from provenance import provenance
        out["binding"]["_provenance"] = provenance(script="scripts/binding_verdict.py")
        out["binding"]["scorer"] = f"{SCORER}@{out['binding']['_provenance']['code_commit'][:12]}"
    except Exception:
        pass
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["binding"] | {"_provenance": None}), flush=True)


if __name__ == "__main__":
    main()
