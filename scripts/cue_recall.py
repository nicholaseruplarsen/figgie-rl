"""Known-answer recall test: can a memory core read back a cue shown only at decision 0?

One recorder (none_s0) plays the probe table on fixed seeds and its input sequences are saved
(scripts/common_probe.py). Each game gets a random cue c in {0..3}. A fixed random direction u_c
with the scale of the inputs is added to the decision-0 input only. Every net is run from a fresh
state over the cued inputs, and a 5-fold CV linear probe (controlled_probe_v2.probe) reads c from
the core output at decisions 0, 1, 60, 120 and 239 (clipped to the game length).

The cue is independent of the game, and the inputs are recorded before the cue is added, so after
decision 0 nothing in the inputs carries it. The only way to read it later is memory.

  none core          placebo gate: its state is a function of the current input only, so its
                     accuracy at decisions >= 1 must be at chance (25%). If the 90% CI excludes 25,
                     the test is invalid.
  decision 0         positive control: every net should read the cue where it is shown.
  random init        same arch and width, untrained: the floor a memory core gets from
                     architecture alone.

Confound: the frozen policies were never trained with a cue, so a positive result shows the core
carries the cue forward, not that the trained policy would use it. A null can mean training never
asked the core to keep inputs of this kind. Exploratory, never scored.

    python scripts/cue_recall.py --ckpt none=runs/memory/none_s0/final.pt mingru=... --out out/summary.json
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_probe import last_hidden, record_chunk  # noqa: E402
from controlled_probe_v2 import boot_ci, correct, probe  # noqa: E402

PROBE_AT = (0, 1, 60, 120, 239)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="runs/memory/none_s0/final.pt")
    ap.add_argument("--ckpt", nargs="+", required=True, help="arch=path of trained nets")
    ap.add_argument("--random-archs", nargs="*", default=["none", "mingru", "gru", "transformer"])
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cue-scale", type=float, default=1.0, help="cue norm in units of the mean input norm")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.workers)
    from figgie.rl.policy import ActorCritic, load_net

    t0 = time.time()
    chunks = [(a.record, list(range(w, a.games, a.workers))) for w in range(a.workers)]
    with Pool(a.workers) as pool:
        games = sorted((g for c in pool.map(record_chunk, chunks) for g in c), key=lambda g: g[0])
    lens = {len(g[2]) for g in games}
    assert len(lens) == 1, f"games differ in decision count: {sorted(lens)[:5]}"
    T = lens.pop()
    X = np.stack([g[2] for g in games])
    rng = np.random.default_rng(a.seed)
    cue = rng.integers(0, 4, size=len(games))
    U = rng.standard_normal((4, X.shape[-1])).astype(np.float32)
    U *= a.cue_scale * float(np.linalg.norm(X[:, 0], axis=1).mean()) / np.linalg.norm(U, axis=1, keepdims=True)
    Xc = X.copy()
    Xc[:, 0] += U[cue]
    at = sorted({min(t, T - 1) for t in PROBE_AT})

    def curve(net):
        out = {}
        for t in at:
            c = correct(probe(last_hidden(net, Xc[:, :t + 1]), cue), cue)
            out[t] = dict(acc=round(100 * c.mean(), 3), ci90=[round(100 * v, 3) for v in boot_ci(c, a.boot)])
        return out

    nets = {}
    for spec in a.ckpt:
        arch, path = spec.split("=", 1)
        net = load_net(path)
        assert net.arch == arch, f"{spec}: checkpoint arch is {net.arch}"
        nets[f"{arch}/trained"] = dict(arch=arch, trained=True, path=path, by_decision=curve(net))
        print(arch, "trained", {t: v["acc"] for t, v in nets[f"{arch}/trained"]["by_decision"].items()}, flush=True)
    for arch in a.random_archs:
        torch.manual_seed(a.random_seed)
        net = ActorCritic(arch, obs_dim=X.shape[-1], window=128).eval()
        nets[f"{arch}/random"] = dict(arch=arch, trained=False, by_decision=curve(net))
        print(arch, "random", {t: v["acc"] for t, v in nets[f"{arch}/random"]["by_decision"].items()}, flush=True)

    gate_key = "none/trained" if "none/trained" in nets else "none/random"
    late = [t for t in at if t >= 1]
    gate = {t: nets[gate_key]["by_decision"][t] for t in late}
    valid = all(v["ci90"][0] <= 25.0 <= v["ci90"][1] for v in gate.values())
    shown = all(n["by_decision"][at[0]]["acc"] > 40.0 for n in nets.values() if n["trained"])

    from provenance import flag_exploratory, provenance
    prov = provenance(script="scripts/cue_recall.py")
    res = dict(test="cue_recall", n_games=len(games), decisions=T, probe_at=at, cue_scale=a.cue_scale,
               chance=25.0, placebo=dict(net=gate_key, at=gate, valid=bool(valid)),
               positive_control=dict(decision=at[0], all_trained_above_40=bool(shown)),
               nets=nets, confound="frozen policies never trained with the cue: recall shows carried "
               "information, not use", seconds=round(time.time() - t0, 1), _provenance=prov)
    res = flag_exploratory(res, always=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(placebo_valid=valid, positive_control=shown, seconds=res["seconds"])), flush=True)


if __name__ == "__main__":
    main()
