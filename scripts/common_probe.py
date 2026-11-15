"""Probe many networks on one common set of trajectories.

One recorder agent (default: none_s0) plays the probe table of scripts/probe_ppo.py
(same opponents, same game seeds) and its full input sequence is saved. Every network,
trained checkpoints and randomly initialised ones of each memory core, is then run over
exactly those inputs from a fresh state, and a 5-fold cross-validated linear probe reads
the goal suit from its last hidden state (the core output) at the last tick. Because the
games are identical, differences between networks are not caused by their policies.

    python scripts/common_probe.py --record runs/memory/none_s0/final.pt \
        --ckpt none_s0=runs/memory/none_s0/final.pt ... --random-seeds 0 1 --out common_probe.json
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
from probe_ppo import OPPONENTS, probe, score  # noqa: E402


def record_chunk(job):
    import torch
    torch.set_num_threads(1)
    from figgie.agents.registry import make
    from figgie.engine import play
    from figgie.rl.policy import PolicyAgent, load_net

    class Recorder(PolicyAgent):
        def reset(self, obs):
            super().reset(obs)
            self.xs, self.ticks, f = [], [], self.feat

            def featurize(o):
                x = f(o)
                self.xs.append(x)
                self.ticks.append(o.tick)
                return x
            self.feat = featurize

    ckpt, idx = job
    net = load_net(ckpt)
    out = []
    for i in idx:
        agent = Recorder(net, deterministic=True)
        opps = [make(n, seed=1000 + i * 7 + k + 1) for k, n in enumerate(OPPONENTS)]
        res = play([agent] + opps, seed=50_000 + i)
        out.append((i, res.game.goal, np.stack(agent.xs).astype(np.float32), agent.ticks[-1]))
    return out


def last_hidden(net, X, batch=100):
    """X [G, T, D] -> core output after the last tick, [G, H]."""
    import torch
    zs = []
    with torch.no_grad():
        for b in range(0, len(X), batch):
            x = torch.from_numpy(X[b:b + batch]).transpose(0, 1)          # [T, B, D]
            T, B = x.shape[:2]
            e = net.enc(x)
            done = torch.zeros(T, B)
            s0 = net.init_state(B)
            if net.arch == "none":
                z = e
            elif net.arch == "gru":
                z = net._gru_unroll(e, s0, done)
            elif net.arch == "mingru":
                h, hs = s0, []
                for t in range(T):
                    g = torch.sigmoid(net.mz(e[t]))
                    h = (1 - g) * h + g * net.mh(e[t])
                hs = h
                z = hs.unsqueeze(0)
            else:
                z = net._tf_unroll(e, s0, done)
            zs.append(z[-1].numpy())
    return np.concatenate(zs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="checkpoint of the agent that plays the games")
    ap.add_argument("--ckpt", nargs="*", default=[], help="name=path of trained networks to probe")
    ap.add_argument("--random-seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.workers)
    from figgie.rl.features import FLOW_DIM
    from figgie.rl.policy import ARCHS, ActorCritic, load_net

    t0 = time.time()
    chunks = [(a.record, list(range(w, a.games, a.workers))) for w in range(a.workers)]
    with Pool(a.workers) as pool:
        games = sorted(g for c in pool.map(record_chunk, chunks) for g in c)
    lens = {len(g[2]) for g in games}
    assert len(lens) == 1, f"games differ in decision count: {sorted(lens)[:5]}"
    X = np.stack([g[2] for g in games])
    y = np.array([g[1] for g in games])
    res = dict(record=a.record, opponents=OPPONENTS, n=len(games), decisions=lens.pop(),
               last_tick=int(games[0][3]), sim_seconds=round(time.time() - t0, 1))
    res["input"] = score(probe(X[:, -1], y), y)
    nets = [(f"{arch}_random{s}", None, arch, s) for arch in ARCHS for s in a.random_seeds]
    nets += [(n, p, None, None) for n, p in (c.split("=", 1) for c in a.ckpt)]
    res["nets"] = {}
    for name, path, arch, seed in nets:
        if path:
            net = load_net(path)
        else:
            torch.manual_seed(seed)
            net = ActorCritic(arch, obs_dim=FLOW_DIM, window=128).eval()
        Z = last_hidden(net, X)
        acc, ll = score(probe(Z, y), y)
        res["nets"][name] = dict(arch=net.arch, acc=acc, ll=ll, trained=bool(path))
        print(name, round(acc, 3), round(ll, 3), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    from provenance import provenance
    res["_provenance"] = provenance(script="scripts/common_probe.py")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(a.out, res["seconds"], "s")


if __name__ == "__main__":
    main()
