"""CS224R pipeline: behavioural cloning from Bayesian, then recurrent PPO with
potential-based shaping and PFSP (§6-7).

  python -m figgie.rl.train bc  --out checkpoints/bc_v2.pt --lift-weight 15
  python -m figgie.rl.train ppo --init checkpoints/bc_v2.pt --out checkpoints/ppo_v2 --lift-alpha 0.1

Reported hyperparameters used as-is: beta_shape 0.5, beta_lift 0.1, p_sp 0.3,
PFSP q=2, pool 16, snapshot every 20 updates after 50 warm-up updates,
32 envs x 512 steps, 300 updates, GRU 128, TBPTT chunk 32 with gradients
accumulated over 4 chunks, BC 500 episodes x 10 epochs, lift weight 15.
GUESSES (not in the report): lr 2.5e-4, gamma 0.995, GAE lambda 0.95,
clip 0.2, entropy 0.01, value coef 0.5, 4 PPO epochs, reward scale 0.1,
BC lr 1e-3, PFSP EMA rate 0.1, heuristic pool = the five baselines uniformly.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..actions import encode, legal_mask, LIFT
from ..agents import cs224r as C
from ..engine import play
from .env import FiggieEnv
from .features import Featurizer
from .policy import ActorCriticRecurrent, PolicyAgent, H

BASELINES = [C.RandomAgent, C.FairValue, C.Conservative, C.MarketMaker, C.Bayesian]
LIFT_IDS = set(range(249, 253))


# ---------------------------------------------------------------- BC
class Recorder:
    def __init__(self, teacher):
        self.t, self.name = teacher, teacher.name

    def reset(self, obs):
        self.t.reset(obs)
        self.f = Featurizer(obs)
        self.X, self.M, self.A = [], [], []

    def act(self, obs):
        x, m = self.f(obs), legal_mask(obs)
        a = self.t.act(obs)
        self.X.append(x)
        self.M.append(m)
        self.A.append(encode(a))
        return a


def bc(args):
    rng = random.Random(args.seed)
    eps = []
    for e in range(args.episodes):
        rec = Recorder(C.Bayesian(seed=rng.getrandbits(32)))
        opps = [rng.choice(BASELINES)(seed=rng.getrandbits(32)) for _ in range(3)]
        play([rec] + opps, seed=rng.getrandbits(48))
        eps.append((np.stack(rec.X), np.stack(rec.M), np.array(rec.A)))
    net = ActorCriticRecurrent()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    log = []
    for ep in range(args.epochs):
        rng.shuffle(eps)
        tot = n = lift_hit = lift_n = top1 = 0
        for i in range(0, len(eps), args.batch):
            chunk = eps[i:i + args.batch]
            X = torch.tensor(np.stack([c[0] for c in chunk]).swapaxes(0, 1))
            M = torch.tensor(np.stack([c[1] for c in chunk]).swapaxes(0, 1))
            A = torch.tensor(np.stack([c[2] for c in chunk]).swapaxes(0, 1))
            done = torch.zeros(X.shape[:2])
            logits, _, _ = net.unroll(X, torch.zeros(X.shape[1], H), M, done)
            legal = M.gather(-1, A.unsqueeze(-1)).squeeze(-1).float()
            w = torch.where((A >= 249) & (A < 253), args.lift_weight, 1.0) * legal
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), A.reshape(-1), reduction="none")
            loss = (nll * w.reshape(-1)).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            pred = logits.argmax(-1)
            isl = (A >= 249) & (A < 253)
            tot += float(loss.detach()) * len(chunk)
            n += len(chunk)
            top1 += float((pred == A).float().mean()) * len(chunk)
            lift_hit += int((pred == A)[isl].sum())
            lift_n += int(isl.sum())
        row = dict(epoch=ep + 1, loss=tot / n, top1=top1 / n, lift_top1=lift_hit / max(lift_n, 1))
        log.append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(net.state_dict(), args.out)
    json.dump(log, open(args.out + ".log.json", "w"), indent=1)


# ---------------------------------------------------------------- PFSP
class Pool:
    def __init__(self, size=16, q=2.0, p_sp=0.3, ema=0.1):
        self.size, self.q, self.p_sp, self.ema = size, q, p_sp, ema
        self.members = []   # [id, state_dict, winrate]
        self.next_id = 0

    def add(self, net):
        self.members.append([self.next_id, copy.deepcopy(net.state_dict()), 0.5])
        self.next_id += 1
        if len(self.members) > self.size:
            self.members.pop(0)

    def sampler(self):
        def sample(rng):
            ids, opps = [], []
            for _ in range(3):
                if self.members and rng.random() < self.p_sp:
                    w = [(1 - m[2]) ** self.q + 1e-6 for m in self.members]
                    m = rng.choices(self.members, weights=w)[0]
                    net = ActorCriticRecurrent()
                    net.load_state_dict(m[1])
                    net.eval()
                    opps.append(PolicyAgent(net, deterministic=False, seed=rng.getrandbits(31)))
                    ids.append(("pool", m[0]))
                else:
                    cls = rng.choice(BASELINES)
                    opps.append(cls(seed=rng.getrandbits(32)))
                    ids.append(("heur", cls.name))
            return ids, opps
        return sample

    def record(self, info):
        """Learner 'beats' a pool member if it out-earned it this episode."""
        for (kind, mid), op in zip(info["opp_ids"], info["opp_profits"]):
            if kind != "pool":
                continue
            for m in self.members:
                if m[0] == mid:
                    m[2] += self.ema * (float(info["profit"] > op) - m[2])


# ---------------------------------------------------------------- PPO
def ppo(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    net = ActorCriticRecurrent()
    if args.init:
        net.load_state_dict(torch.load(args.init, weights_only=True))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    pool = Pool(p_sp=args.p_sp)
    envs = [FiggieEnv(pool.sampler(), seed=args.seed * 1000 + i, beta=args.beta, gamma=args.gamma,
                      lift_alpha=args.lift_alpha) for i in range(args.envs)]
    B, T = args.envs, args.steps
    cur = [e.reset() for e in envs]
    h = torch.zeros(B, H)
    start = torch.ones(B)
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "train_log.jsonl"), "a")
    for upd in range(1, args.updates + 1):
        t0 = time.time()
        Xs = torch.zeros(T, B, 69)
        Ms = torch.zeros(T, B, 257, dtype=torch.bool)
        As = torch.zeros(T, B, dtype=torch.long)
        LPs, Vs, Rs = torch.zeros(T, B), torch.zeros(T, B), torch.zeros(T, B)
        Ds = torch.zeros(T, B)
        Hs = torch.zeros(T, B, H)
        infos = []
        net.eval()
        with torch.no_grad():
            for t in range(T):
                X = torch.from_numpy(np.stack([c[0] for c in cur]))
                M = torch.from_numpy(np.stack([c[1] for c in cur]))
                h = h * (1 - start).unsqueeze(-1)
                Hs[t], Ds[t] = h, start
                logits, v, h = net.step(X, h, M)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                Xs[t], Ms[t], As[t], LPs[t], Vs[t] = X, M, a, dist.log_prob(a), v
                start = torch.zeros(B)
                for i, e in enumerate(envs):
                    f, m, r, d, info = e.step(int(a[i]))
                    Rs[t, i] = r
                    if d:
                        infos.append(info)
                        pool.record(info)
                        f, m = e.reset()
                        start[i] = 1
                    cur[i] = (f, m)
            Xl = torch.from_numpy(np.stack([c[0] for c in cur]))
            Ml = torch.from_numpy(np.stack([c[1] for c in cur]))
            _, v_last, _ = net.step(Xl, h * (1 - start).unsqueeze(-1), Ml)
        # GAE; Ds[t+1]==1 means step t ended its episode.
        adv = torch.zeros(T, B)
        last = torch.zeros(B)
        for t in reversed(range(T)):
            nonterm = 1 - (Ds[t + 1] if t + 1 < T else start)
            nv = Vs[t + 1] if t + 1 < T else v_last
            delta = Rs[t] + args.gamma * nv * nonterm - Vs[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        ret = adv + Vs
        net.train()
        chunks = list(range(0, T, args.chunk))
        stats = dict(pg=0.0, vf=0.0, ent=0.0, n=0)
        for _ in range(args.epochs):
            random.shuffle(chunks)
            for k, c0 in enumerate(chunks):
                sl = slice(c0, c0 + args.chunk)
                logits, v, _ = net.unroll(Xs[sl], Hs[c0], Ms[sl], Ds[sl])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(As[sl])
                ratio = (lp - LPs[sl]).exp()
                ad = adv[sl]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pg = -torch.min(ratio * ad, ratio.clamp(1 - args.clip, 1 + args.clip) * ad).mean()
                vf = 0.5 * (v - ret[sl]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = (pg + args.vf * vf - args.ent * ent) / args.accum
                loss.backward()
                stats["pg"] += pg.item(); stats["vf"] += vf.item(); stats["ent"] += ent.item(); stats["n"] += 1
                if (k + 1) % args.accum == 0 or k == len(chunks) - 1:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
                    opt.zero_grad()
        if upd > args.warmup and upd % args.snap_every == 0:
            pool.add(net)
        n = max(stats.pop("n"), 1)
        row = dict(update=upd, sec=round(time.time() - t0, 1), episodes=len(infos),
                   profit=float(np.mean([i["profit"] for i in infos])) if infos else None,
                   win=float(np.mean([i["win"] for i in infos])) if infos else None,
                   lifts=float(np.mean([i["lifts"] for i in infos])) if infos else None,
                   pool=len(pool.members), **{k: round(v / n, 4) for k, v in stats.items()})
        print(json.dumps(row), flush=True)
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        if upd % 25 == 0 or upd == args.updates:
            torch.save(net.state_dict(), os.path.join(args.out, f"update_{upd}.pt"))
    torch.save(net.state_dict(), os.path.join(args.out, "final.pt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["bc", "ppo"])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    # BC
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--lift-weight", type=float, default=15.0)
    p.add_argument("--batch", type=int, default=1)
    # PPO
    p.add_argument("--init")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--lift-alpha", type=float, default=0.0)
    p.add_argument("--p-sp", type=float, default=0.3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--snap-every", type=int, default=20)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.mode == "bc":
        a.epochs, a.lr = a.epochs or 10, a.lr or 1e-3
        bc(a)
    else:
        a.epochs, a.lr = a.epochs or 4, a.lr or 2.5e-4
        ppo(a)


if __name__ == "__main__":
    main()
