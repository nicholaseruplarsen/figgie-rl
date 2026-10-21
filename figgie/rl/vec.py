"""Run FiggieEnvs in worker processes. The policy forward pass stays in the
learner; workers step their envs and sample opponents from a local copy of
the PFSP pool, which the learner re-syncs once per update."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _worker(conn, env_kwargs, seeds, pool_kwargs):
    import torch
    torch.set_num_threads(1)
    from .env import FiggieEnv
    from .train import Pool
    pool = Pool(**pool_kwargs)
    states = {}
    envs = [FiggieEnv(pool.sampler(), seed=s, **env_kwargs) for s in seeds]
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            obs = [e.reset() for e in envs]
            conn.send((np.stack([o[0] for o in obs]), np.stack([o[1] for o in obs])))
        elif cmd == "step":
            X, M, R, D, infos = [], [], [], [], []
            for e, a in zip(envs, data):
                f, m, r, d, info = e.step(int(a))
                if d:
                    infos.append(info)
                    f, m = e.reset()
                X.append(f); M.append(m); R.append(r); D.append(d)
            conn.send((np.stack(X), np.stack(M), np.array(R, np.float32), np.array(D), infos))
        elif cmd == "sync":
            meta, new = data            # meta: [(id, winrate)], new: {id: state_dict}
            states.update(new)
            states = {i: states[i] for i, _ in meta}
            pool.members = [[i, states[i], w] for i, w in meta]
        elif cmd == "close":
            return


class VecEnv:
    def __init__(self, n_envs, n_procs, seed, env_kwargs, pool_kwargs):
        ctx = mp.get_context("fork")
        per = np.array_split(np.arange(n_envs), n_procs)
        self.split = [len(p) for p in per]
        self.conns, self.procs = [], []
        for p in per:
            a, b = ctx.Pipe()
            pr = ctx.Process(target=_worker, args=(b, env_kwargs, [seed * 1000 + int(i) for i in p], pool_kwargs),
                             daemon=True)
            pr.start()
            self.conns.append(a)
            self.procs.append(pr)
        self.sent = set()

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        out = [c.recv() for c in self.conns]
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    def step(self, actions):
        i = 0
        for c, k in zip(self.conns, self.split):
            c.send(("step", actions[i:i + k]))
            i += k
        out = [c.recv() for c in self.conns]
        infos = [x for o in out for x in o[4]]
        return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]),
                np.concatenate([o[2] for o in out]), np.concatenate([o[3] for o in out]), infos)

    def sync(self, pool):
        meta = [(m[0], m[2]) for m in pool.members]
        new = {m[0]: m[1] for m in pool.members if m[0] not in self.sent}
        self.sent |= set(new)
        for c in self.conns:
            c.send(("sync", (meta, new)))

    def close(self):
        for c in self.conns:
            c.send(("close", None))
