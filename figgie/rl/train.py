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
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..actions import encode, legal_mask, LIFT
from ..agents import cs224r as C
from ..engine import play
from .vec import VecEnv
from .features import FLOW_DIM, OBS_DIM, make_featurizer
from .policy import ActorCritic, PolicyAgent, load_net, save

BASELINES = [C.RandomAgent, C.FairValue, C.Conservative, C.MarketMaker, C.Bayesian]
LIFT_IDS = set(range(249, 253))


# ---------------------------------------------------------------- BC
class Recorder:
    def __init__(self, teacher, flow=False):
        self.t, self.name = teacher, teacher.name
        self.Feat = make_featurizer(flow)

    def reset(self, obs):
        self.t.reset(obs)
        self.f = self.Feat(obs)
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
        rec = Recorder(C.Bayesian(seed=rng.getrandbits(32)), args.flow)
        opps = [rng.choice(BASELINES)(seed=rng.getrandbits(32)) for _ in range(3)]
        play([rec] + opps, seed=rng.getrandbits(48))
        eps.append((np.stack(rec.X), np.stack(rec.M), np.array(rec.A)))
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    net = new_net(args).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    log = []
    for ep in range(args.epochs):
        rng.shuffle(eps)
        tot = n = lift_hit = lift_n = top1 = 0
        for i in range(0, len(eps), args.batch):
            chunk = eps[i:i + args.batch]
            X = torch.tensor(np.stack([c[0] for c in chunk]).swapaxes(0, 1), device=dev)
            M = torch.tensor(np.stack([c[1] for c in chunk]).swapaxes(0, 1), device=dev)
            A = torch.tensor(np.stack([c[2] for c in chunk]).swapaxes(0, 1), device=dev)
            done = torch.zeros(X.shape[:2], device=dev)
            logits, _, _ = net.unroll(X, net.init_state(X.shape[1], dev), M, done)
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
    save(net.cpu(), args.out)
    json.dump(log, open(args.out + ".log.json", "w"), indent=1)


def new_net(args):
    return ActorCritic(args.arch, obs_dim=FLOW_DIM if args.flow else OBS_DIM, window=args.window)


# ---------------------------------------------------------------- PFSP
class Pool:
    def __init__(self, size=16, q=2.0, p_sp=0.3, ema=0.1, net_config=None):
        self.size, self.q, self.p_sp, self.ema = size, q, p_sp, ema
        self.net_config = net_config or dict(arch="gru")
        self.members = []   # [id, state_dict, winrate]
        self.next_id = 0

    def add(self, net):
        self.members.append([self.next_id, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0.5])
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
                    net = ActorCritic(**self.net_config)
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
    dev = torch.device(args.device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if dev.type == "cuda" and args.max_vram_gb:
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_gb * 2**30 / total), dev)
    first = 0
    latest = os.path.join(args.out, "latest.pt")
    if args.resume and os.path.exists(latest):
        # Resume weights only: the Adam state and the PFSP pool start afresh.
        first = torch.load(latest, map_location="cpu", weights_only=True).get("update") or args.start_update
        assert first, "latest.pt has no update number, pass --start-update"
        net = load_net(latest)
        print(json.dumps(dict(resumed_from=first)), flush=True)
    else:
        net = load_net(args.init) if args.init else new_net(args)
    assert net.arch == args.arch and net.config["obs_dim"] == (FLOW_DIM if args.flow else OBS_DIM), net.config
    net = net.to(dev)
    if args.opt == "adam":                      # the recipe so far (PPO-v2 and the memory runs)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    else:                                       # shared tsinghua-rl iig/tune factory (E1/E2 arms)
        if args.iig_path:
            sys.path.insert(0, args.iig_path)
        from iig.tune.optim import make_optimizer
        opt, split = make_optimizer(net, args.opt, args.lr, eps=1e-5)
        print(json.dumps(dict(optimizer=args.opt, split=[(r["name"], r["group"]) for r in split])), flush=True)
    pool = Pool(p_sp=args.p_sp, net_config=net.config)
    venv = VecEnv(args.envs, args.procs, args.seed,
                  dict(beta=args.beta, gamma=args.gamma, lift_alpha=args.lift_alpha, flow=args.flow),
                  dict(p_sp=args.p_sp, net_config=net.config), start=args.mp_start)
    B, T, D = args.envs, args.steps, net.config["obs_dim"]
    Xc, Mc = venv.reset()
    h = net.init_state(B, dev)
    start = torch.ones(B, device=dev)
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "train_log.jsonl"), "a")
    for upd in range(first + 1, args.updates + 1):
        t0 = time.time()
        Xs = torch.zeros(T, B, D, device=dev)
        Ms = torch.zeros(T, B, 257, dtype=torch.bool, device=dev)
        As = torch.zeros(T, B, dtype=torch.long, device=dev)
        LPs, Vs, Rs = (torch.zeros(T, B, device=dev) for _ in range(3))
        Ds = torch.zeros(T, B, device=dev)
        Hs = {}                         # recurrent state at each chunk start
        infos = []
        venv.sync(pool)
        net.eval()
        with torch.no_grad():
            for t in range(T):
                X, M = torch.from_numpy(Xc).to(dev), torch.from_numpy(Mc).to(dev)
                h = h * (1 - start).unsqueeze(-1)
                Ds[t] = start
                if t % args.chunk == 0:
                    Hs[t] = h
                logits, v, h = net.step(X, h, M)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                Xs[t], Ms[t], As[t], LPs[t], Vs[t] = X, M, a, dist.log_prob(a), v
                Xc, Mc, R, Dn, new = venv.step(a.cpu().numpy())
                Rs[t] = torch.from_numpy(R).to(dev)
                start = torch.from_numpy(Dn.astype(np.float32)).to(dev)
                for info in new:
                    infos.append(info)
                    pool.record(info)
            Xl, Ml = torch.from_numpy(Xc).to(dev), torch.from_numpy(Mc).to(dev)
            _, v_last, _ = net.step(Xl, h * (1 - start).unsqueeze(-1), Ml)
        # GAE; Ds[t+1]==1 means step t ended its episode.
        adv = torch.zeros(T, B, device=dev)
        last = torch.zeros(B, device=dev)
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
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
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
            save(net, latest, update=upd)     # one rolling checkpoint keeps disk use small
    save(net, os.path.join(args.out, "final.pt"))
    venv.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["bc", "ppo"])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", choices=["none", "mingru", "gru", "transformer"], default="gru")
    p.add_argument("--flow", action="store_true", help="85-dim observation with per-opponent trade flow")
    p.add_argument("--window", type=int, default=128, help="transformer context in ticks")
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true", help="continue from OUT/latest.pt if present")
    p.add_argument("--start-update", type=int, default=0, help="update number of a latest.pt saved without one")
    p.add_argument("--max-vram-gb", type=float, default=0, help="cap on this process's CUDA allocations")
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
    p.add_argument("--opt", choices=["adam", "adamw", "muon"], default="adam",
                   help="adam: the old recipe; adamw/muon: tsinghua-rl iig/tune make_optimizer")
    p.add_argument("--iig-path", default=os.environ.get("IIG_PATH"), help="dir holding tsinghua-rl's iig/ package")
    p.add_argument("--max-grad-norm", type=float, default=0.5, help="PPO gradient clip (shared iig/tune core space)")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--lift-alpha", type=float, default=0.0)
    p.add_argument("--p-sp", type=float, default=0.3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--snap-every", type=int, default=20)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--procs", type=int, default=4, help="env worker processes")
    p.add_argument("--mp-start", default="fork", choices=["fork", "spawn", "forkserver"],
                   help="how env workers are started, spawn avoids inheriting the learner's memory")
    a = p.parse_args()
    if a.mode == "bc":
        a.epochs, a.lr = a.epochs or 10, a.lr or 1e-3
        bc(a)
    else:
        a.epochs, a.lr = a.epochs or 4, a.lr or 2.5e-4
        ppo(a)


if __name__ == "__main__":
    main()
