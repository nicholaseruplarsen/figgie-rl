"""Linear probe of the PPO agent's GRU state for the hidden goal suit.

PPO sits at seat 0 against the other three agents of the revelation table in
scripts/proposal_figure.py. At each probe tick we record the agent's 69-dim
input and the memory core's output after acting (the vector the actor reads: the GRU state for
GRU agents), then fit 5-fold cross-validated
logistic regressions predicting the true goal suit from (a) the input alone
and (b) the GRU state. References: the exact posterior from the dealt hand and
the card-counting posterior that is part of the input (features 45:49).

    python scripts/probe_ppo.py [--ckpt checkpoints/ppo_v2.pt] [--games 3000]
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OPPONENTS = ["Bayesian", "fb.PotMM", "fig.Fundamental"]
CHECK = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 239]


def run_chunk(job):
    import torch
    torch.set_num_threads(1)
    from figgie.agents.registry import make
    from figgie.belief import goal_posterior
    from figgie.engine import play
    from figgie.rl.policy import PolicyAgent, load_net

    class Recorder(PolicyAgent):
        def reset(self, obs):
            super().reset(obs)
            self.rec, f = {}, self.feat

            def featurize(o):
                self.last_x = f(o)
                return self.last_x
            self.feat = featurize

        def act(self, obs):
            a = super().act(obs)
            if obs.tick in CHECK:
                self.rec[obs.tick] = (self.last_x.copy(), self.net.last_z[0].numpy().copy())
            return a

    ckpt, idx = job
    net = load_net(ckpt)
    out = []
    for i in idx:
        agent = Recorder(net, deterministic=True)
        opps = [make(n, seed=1000 + i * 7 + k + 1) for k, n in enumerate(OPPONENTS)]
        res = play([agent] + opps, seed=50_000 + i)
        g = res.game
        out.append(dict(goal=g.goal, hand=np.array(goal_posterior(tuple(g.hands0[0]))),
                        x=np.stack([agent.rec[t][0] for t in CHECK]),
                        h=np.stack([agent.rec[t][1] for t in CHECK])))
    return out


def probe(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba")


def score(p, y):
    idx = np.arange(len(y))
    return float((p.argmax(1) == y).mean()), float(-np.log2(p[idx, y] + 1e-9).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ppo_v2.pt")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/probe_ppo_v2.json")
    args = ap.parse_args()

    t0 = time.time()
    chunks = [(args.ckpt, list(range(w, args.games, args.workers))) for w in range(args.workers)]
    with Pool(args.workers) as pool:
        games = [g for c in pool.map(run_chunk, chunks) for g in c]
    sim_s = time.time() - t0

    y = np.array([g["goal"] for g in games])
    hand = np.stack([g["hand"] for g in games])
    res = dict(ckpt=args.ckpt, opponents=OPPONENTS, n=len(games), t=CHECK, sim_seconds=round(sim_s, 1),
               hand_acc=[], hand_ll=[], count_acc=[], count_ll=[], input_acc=[], input_ll=[],
               gru_acc=[], gru_ll=[], both_acc=[], both_ll=[])
    a, l = score(hand, y)
    for k, t in enumerate(CHECK):
        X = np.stack([g["x"][k] for g in games])
        H = np.stack([g["h"][k] for g in games])
        cnt = X[:, 45:49] / X[:, 45:49].sum(1, keepdims=True)
        rows = dict(hand=(a, l), count=score(cnt, y), input=score(probe(X, y), y),
                    gru=score(probe(H, y), y), both=score(probe(np.hstack([X, H]), y), y))
        for name, (acc, ll) in rows.items():
            res[f"{name}_acc"].append(acc)
            res[f"{name}_ll"].append(ll)
        print(t, {k2: (round(v[0], 3), round(v[1], 3)) for k2, v in rows.items()}, flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    from provenance import provenance, flag_exploratory
    res["_provenance"] = provenance(script="scripts/probe_ppo.py")
    res = flag_exploratory(res)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(args.out, res["seconds"], "s")


if __name__ == "__main__":
    main()
