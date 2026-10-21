"""Mixed-table tournament over every agent on the shared engine.

Each "table" draws 4 distinct agents uniformly at random and one deal seed;
the deal is then replayed under all 4 cyclic seat rotations (duplicate
format), so every agent at the table sees every seat on the same cards.

Reports per agent: mean P&L per game with 95% CI (clustered by table),
win rate (top profit, ties split), Elo (figment's pairwise-duel Elo,
averaged over 20 random game orders) and TrueSkill (mu - 3 sigma, games
treated as 4-way free-for-all ranked by profit).

  python scripts/tournament.py --tables 3000 --procs 8 --out results/tournament
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import statistics as st
import sys
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from figgie.agents.registry import AGENTS, make  # noqa: E402
from figgie.engine import play  # noqa: E402

PPO = {}   # name -> checkpoint path, filled from --ppo
_NETS = {}


def init_worker(ppo):
    import torch
    torch.set_num_threads(1)
    PPO.update(ppo)


def build(name, seed):
    if name in PPO:
        from figgie.rl.policy import PolicyAgent, load
        if name not in _NETS:
            _NETS[name] = load(PPO[name]).net
        return PolicyAgent(_NETS[name], deterministic=True, name=name)
    return make(name, seed)


def run_table(job):
    tid, names, seed = job
    out = []
    for rot in range(4):
        seated = names[rot:] + names[:rot]
        r = play([build(n, seed * 4 + rot) for n in seated], seed=seed)
        out.append((tid, seated, r.profits, [s["lifts"] for s in r.stats]))
    return out


def elo(games, names, k=24.0, shuffles=20, seed=0):
    rng = random.Random(seed)
    acc = {n: 0.0 for n in names}
    for _ in range(shuffles):
        R = {n: 1500.0 for n in names}
        order = list(range(len(games)))
        rng.shuffle(order)
        for gi in order:
            seated, prof = games[gi][1], games[gi][2]
            d = {n: 0.0 for n in seated}
            for i, j in combinations(range(len(seated)), 2):
                a, b = seated[i], seated[j]
                sa = 1.0 if prof[i] > prof[j] else 0.0 if prof[i] < prof[j] else 0.5
                ea = 1 / (1 + 10 ** ((R[b] - R[a]) / 400))
                d[a] += k * (sa - ea)
                d[b] -= k * (sa - ea)
            for n in seated:
                R[n] += d[n]
        for n in names:
            acc[n] += R[n] / shuffles
    return acc


def trueskill_ratings(games, names):
    import trueskill
    env = trueskill.TrueSkill(draw_probability=0.01)
    R = {n: env.create_rating() for n in names}
    for _, seated, prof, _ in games:
        ranks = [sorted(prof, reverse=True).index(p) for p in prof]
        new = env.rate([(R[n],) for n in seated], ranks=ranks)
        for n, (r,) in zip(seated, new):
            R[n] = r
    return {n: (R[n].mu, R[n].sigma) for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ppo", nargs="*", default=[], help="name=path.pt")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--out", default=os.path.join(ROOT, "results/tournament"))
    a = ap.parse_args()
    for spec in a.ppo:
        n, p = spec.split("=")
        PPO[n] = p
    names = [n for n in list(AGENTS) + list(PPO) if n not in a.exclude]
    rng = random.Random(a.seed)
    jobs = [(t, rng.sample(names, 4), rng.getrandbits(40)) for t in range(a.tables)]
    with mp.Pool(a.procs, initializer=init_worker, initargs=(PPO,)) as pool:
        games = [g for tab in pool.imap_unordered(run_table, jobs, chunksize=8) for g in tab]
    games.sort(key=lambda g: g[0])

    # per-table mean profit per agent, for clustered CIs
    per_table = {n: {} for n in names}
    wins = {n: 0.0 for n in names}
    cnt = {n: 0 for n in names}
    lifts = {n: 0 for n in names}
    for tid, seated, prof, lf in games:
        top = max(prof)
        nw = sum(p == top for p in prof)
        for n, p, l in zip(seated, prof, lf):
            per_table[n].setdefault(tid, []).append(p)
            cnt[n] += 1
            wins[n] += (p == top) / nw
            lifts[n] += l
    E = elo(games, names)
    TS = trueskill_ratings(games, names)
    rows = []
    for n in names:
        xs = [st.mean(v) for v in per_table[n].values()]
        m = st.mean(xs)
        ci = 1.96 * st.stdev(xs) / len(xs) ** 0.5
        mu, sig = TS[n]
        rows.append(dict(agent=n, games=cnt[n], mean_pnl=round(m, 2), ci95=round(ci, 2),
                         win_rate=round(wins[n] / cnt[n], 3), elo=round(E[n]), trueskill=round(mu - 3 * sig, 2),
                         lifts_per_game=round(lifts[n] / cnt[n], 2)))
    rows.sort(key=lambda r: -r["mean_pnl"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(args=vars(a), rows=rows), open(a.out + ".json", "w"), indent=1)
    hdr = "| # | agent | games | mean P&L ± 95% CI | win rate | Elo | TrueSkill (μ−3σ) | aggressive buys/game |\n|---|---|---|---|---|---|---|---|\n"
    md = hdr + "\n".join(
        f"| {i+1} | {r['agent']} | {r['games']} | {r['mean_pnl']:+.1f} ± {r['ci95']:.1f} | {r['win_rate']:.1%} | {r['elo']} | {r['trueskill']:.1f} | {r['lifts_per_game']:.1f} |"
        for i, r in enumerate(rows))
    open(a.out + ".md", "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
