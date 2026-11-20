"""Card-counter check (Demis gate before the belief-value seal). Eval only, no training.

A. Synthetic known decks: draw a deck uniformly, deal a uniformly random n-card subset, and compare the empirical
   deck/goal frequencies given the counts with figgie.belief.deck_posterior / goal_posterior. If the posterior is the
   exact Bayes posterior for a random subset, the goal ECE is ~0 and the binned frequencies match the predictions.
B. Real games, the #6607 cc arm (BeliefBayes source=cc vs FairValue, Bayesian, MarketMaker), exploratory games
   100000+g (the diagnosis range, disjoint from any sealed range). Per tick:
   - lower-bound validity: every L[p][s] <= true hands[p][s] (a violation is a bookkeeping bug),
   - the goal ECE / accuracy of goal_posterior(counts),
   - selection: the share of known non-own cards in the goal suit, against its share of the deck.
"""
import argparse, json, os, random, sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figgie.belief import deck_posterior, goal_posterior  # noqa: E402
from figgie.engine import DECKS, DECK_GOAL, play  # noqa: E402

TB = [0, 10, 30, 60, 120, 180, 239]


def ece(conf, hit, nb=10):
    conf, hit = np.asarray(conf), np.asarray(hit, float)
    b = np.minimum((conf * nb).astype(int), nb - 1)
    return float(sum(abs(hit[b == k].mean() - conf[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any()))


def synth(n_draws, sizes, seed):
    rng = random.Random(seed)
    out = {}
    for n in sizes:
        conf, hit, ll, tv = [], [], [], []
        groups = defaultdict(lambda: np.zeros(12))
        for _ in range(n_draws):
            d = rng.randrange(12)
            pile = [s for s in range(4) for _ in range(DECKS[d][s])]
            c = [0] * 4
            for s in rng.sample(pile, n):
                c[s] += 1
            c = tuple(c)
            gp = goal_posterior(c)
            k = int(np.argmax(gp))
            conf.append(gp[k]); hit.append(k == DECK_GOAL[d]); ll.append(-np.log(max(gp[DECK_GOAL[d]], 1e-12)))
            groups[c][d] += 1
        # binned check per count vector with enough mass: empirical deck frequencies vs deck_posterior (total variation)
        for c, f in groups.items():
            if f.sum() >= 200:
                tv.append(0.5 * float(np.abs(f / f.sum() - np.array(deck_posterior(c))).sum()))
        out[n] = dict(n=n_draws, acc=round(100 * float(np.mean(hit)), 2), mean_conf=round(float(np.mean(conf)), 4),
                      ece=round(ece(conf, hit), 4), logloss=round(float(np.mean(ll)), 4),
                      tv_max=round(max(tv), 4) if tv else None, tv_groups=len(tv))
    return out


def one_game(g):
    import belief_value_eval as E
    names = ["T"] + E.OPP
    rot = g % 4
    seated = names[rot:] + names[:rot]
    ags = []
    for i, nm in enumerate(seated):
        ags.append(E.BeliefBayes("cc", None, seed=g * 10 + i) if nm == "T" else getattr(E.C, nm)(seed=g * 10 + i))
    me = seated.index("T")
    T = ags[me]
    log = []

    real_reset = T.reset

    def reset(obs):
        real_reset(obs)
        up = T.cc.update

        def update(o):
            c = up(o)
            game = o.game
            viol = sum(1 for p in range(game.n) for s in range(4) if T.cc.L[p][s] > game.hands[p][s])
            known_other = [sum(T.cc.L[p][s] for p in range(game.n) if p != o.me) for s in range(4)]
            log.append((o.tick, c, viol, known_other))
            return c
        T.cc.update = update
    T.reset = reset
    r = play(ags, seed=1_000_000 + g)
    return r.game.goal, r.game.deck_id, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=int, default=200000)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--offset", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="out/cc_check.json")
    a = ap.parse_args()
    res = dict(script="scripts/cc_check.py", synth=synth(a.synth, [10, 20, 30], seed=7))
    with Pool(a.workers) as p:
        ev = p.map(one_game, range(a.offset, a.offset + a.games), chunksize=5)
    per = {}
    for t0 in TB:
        conf, hit, viol, sel_goal, sel_share, nk = [], [], 0, [], [], []
        for goal, deck, log in ev:
            rows = [x for x in log if x[0] == t0]
            if not rows:
                continue
            tk, c, v, ko = rows[-1]
            gp = goal_posterior(tuple(c))
            k = int(np.argmax(gp)); conf.append(gp[k]); hit.append(k == goal); viol += v
            tot = sum(ko); nk.append(tot)
            if tot:
                sel_goal.append(ko[goal] / tot); sel_share.append(DECKS[deck][goal] / 40)
        per[t0] = dict(n=len(hit), acc=round(100 * float(np.mean(hit)), 1), mean_conf=round(float(np.mean(conf)), 3),
                       ece=round(ece(conf, hit), 3), lower_bound_violations=int(viol),
                       known_other_cards=round(float(np.mean(nk)), 2),
                       goal_share_known=round(float(np.mean(sel_goal)), 3) if sel_goal else None,
                       goal_share_deck=round(float(np.mean(sel_share)), 3) if sel_share else None)
    res["games"] = per
    res["games_range"] = [a.offset, a.offset + a.games]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
