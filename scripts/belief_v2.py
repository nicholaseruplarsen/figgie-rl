"""Belief value v2 (follow-up to #6607): does a calibrated, opponent-robust goal-suit belief pay the fixed trader?
Eval and supervised fitting only (CPU, no PPO). The trader, its rules and the combination are belief_value_eval's,
imported unchanged. Only the belief source q differs between arms.

The #6607 encoder lost 23 $/game. It was overconfident (ECE 0.16-0.30 late on its own training table) and it read
an unseen table worse (49-57% vs 78-80% at ticks 120-239). The arms separate those two causes:

  cc        the trader's own card-count posterior (verified by scripts/cc_check.py, job 26c738b8)
  enc       the #6607 encoder, unchanged
  enc_T     the #6607 encoder with one temperature fitted on its own training table   -> overconfidence only
  mix       a per-tick encoder trained on three mixed opponent tables, no temperature  -> table coverage only
  mix_T     mix with a temperature fitted on mixed-table validation games              -> both
  ccs_T     placebo: the mix_T pipeline trained on the trader's own per-tick cc goal posterior as soft targets, with a
            temperature fitted the same way. It carries no information beyond cc, so it should act like cc (~0 $/game).
            Design change after pilot 1 (f7f04b27): the earlier placebo trained on one hard label drawn per game from the
            final-tick cc posterior. It was calibrated but less accurate than cc (35-38% vs 41-50%) and lost 12.8 $/game.
  perm_T    secondary placebo: labels permuted across games (a near-uniform belief, expected <= 0)
  oracle    the true goal suit (upper bound, exploratory)

Opponent tables. E = FairValue, Bayesian, MarketMaker (cs224r) is the #6607 eval table and is held out of all
mixed training. Training tables (figgie.agents.registry):
  A = fb.PotMM, fig.Fundamental, fa.Noisy
  B = fig.AdaptiveMM, fb.FlowMM, fa.Spread
  C = fig.BayesianMM, fb.Momentum, Conservative
Every game has the trader plus one table, seats rotated by g % 4, play seed 1_000_000 + g, agent seeds g*10 + i.

Game ranges, all disjoint from each other and from every earlier exploratory or sealed range
(#6607 sealed 100000-102999, diagnosis 100000-100299, recorder 20000-20299, encoder training 10000-19999):
  mixed training       table k in (A, B, C): g = 200000 + 10000 k + i, i < --train-games
  mixed temperature    g = 230000 + 1000 k + i, i < --val-games
  enc_T temperature    the #6607 recorder (E0 none_s0, probe table), common_probe indices 20300-20799
  pilot 1              E, g in [100500, 100800)  (job f7f04b27)
  pilot 2              E, g in [105000, 105300)
  sealed               E, g in [110000, 113000), and table A, g in [120000, 123000) (held-in opponents)

Recording: the trader seat plays as cc and records its FlowFeaturizer input at each of its decisions. Training is the
#6607 recipe (GRU 128, Adam 3e-3, wd 1e-4) with cross-entropy at every tick instead of only the last. The
temperature is one scalar per encoder, fitted by NLL over every tick of the validation games.

    python scripts/belief_v2.py fit  --enc0 enc/encoder.pt --rec0 nets/none_s0.pt --out out/fit
    python scripts/belief_v2.py eval --fit out/fit --offset 110000 --games 3000 --out out/sealed.json
"""
import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import belief_value_eval as E  # noqa: E402
from figgie.belief import goal_posterior  # noqa: E402
from figgie.engine import play  # noqa: E402

EVALUATOR = "belief_v2"
TABLES = {"A": ["fb.PotMM", "fig.Fundamental", "fa.Noisy"],
          "B": ["fig.AdaptiveMM", "fb.FlowMM", "fa.Spread"],
          "C": ["fig.BayesianMM", "fb.Momentum", "Conservative"]}


def opponents(table, g):
    from figgie.agents.registry import make
    names = E.OPP if table == "E" else TABLES[table]
    mk = (lambda n, s: getattr(E.C, n)(seed=s)) if table == "E" else make
    return names, mk


def seat(table, g, trader):
    """trader(seed) -> agent. Seeds g*10 + seat index for every seat, as in belief_value_eval.one."""
    names, mk = opponents(table, g)
    order = ["T"] + names
    rot = g % 4
    seated = order[rot:] + order[:rot]
    ags = [trader(g * 10 + i) if n == "T" else mk(n, g * 10 + i) for i, n in enumerate(seated)]
    me = seated.index("T")
    return ags, me, ags[me]


# ---------- recording ----------

class Rec(E.BeliefBayes):
    def reset(self, obs):
        super().reset(obs)
        from figgie.rl.features import FlowFeaturizer
        self.ff, self.xs, self.ccq = FlowFeaturizer(obs), [], []

    def post(self, obs, c):
        self.xs.append(self.ff(obs))
        self.ccq.append(goal_posterior(tuple(c)))
        return super().post(obs, c)


def rec_one(job):
    table, g = job
    ags, me, t = seat(table, g, lambda s: Rec("cc", None, seed=s))
    r = play(ags, seed=1_000_000 + g)
    return g, r.game.goal, np.stack(t.xs).astype(np.float32), np.array(t.ccq, np.float64)


def record(jobs, workers):
    with Pool(workers) as p:
        got = sorted(p.map(rec_one, jobs, chunksize=10), key=lambda z: z[0])
    lens = {len(z[2]) for z in got}
    T = min(lens)
    X = np.stack([z[2][:T] for z in got])
    return X, np.array([z[1] for z in got]), np.stack([z[3][:T] for z in got]), sorted(lens)


# ---------- fitting ----------

def train(X, y, hidden=128, epochs=30, seed=0):
    import torch
    torch.manual_seed(seed)
    mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    soft = y.ndim == 3                                       # [G, T, 4] soft targets (placebo) or [G] labels
    Y = torch.from_numpy(y.astype(np.float32 if soft else np.int64))
    gru = torch.nn.GRU(X.shape[-1], hidden, batch_first=True)
    head = torch.nn.Linear(hidden, 4)
    opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for b in torch.randperm(len(y), generator=g).split(128):
            out, _ = gru(Xn[b])
            lg = head(out)                                   # [B, T, 4]: every tick
            if soft:
                loss = -(Y[b][:, :lg.shape[1]] * torch.log_softmax(lg, -1)).sum(-1).mean()
            else:
                loss = torch.nn.functional.cross_entropy(lg.reshape(-1, 4), Y[b].repeat_interleave(lg.shape[1]))
            opt.zero_grad(); loss.backward(); opt.step()
    return dict(gru=gru.state_dict(), head=head.state_dict(), mu=torch.from_numpy(mu.astype(np.float32)),
                sd=torch.from_numpy(sd.astype(np.float32)), in_dim=X.shape[-1], hidden=hidden)


def logits(enc, X):
    import torch
    gru, head, mu, sd = enc
    with torch.no_grad():
        out, _ = gru(torch.from_numpy(((X - mu) / sd).astype(np.float32)))
        return head(out).numpy()                             # [G, T, 4]


def fit_temp(lg, y):
    """One scalar T > 0 minimising NLL over every tick (golden-section search on log T)."""
    L = lg.reshape(-1, 4)
    if y.ndim == 3:                                           # soft targets: cross-entropy against the distribution
        Q = y[:, :lg.shape[1]].reshape(-1, 4)

        def nll(lt):
            z = L / np.exp(lt)
            z = z - z.max(1, keepdims=True)
            return float(-(Q * (z - np.log(np.exp(z).sum(1, keepdims=True)))).sum(1).mean())
        return _golden(nll)
    Y = np.repeat(y[:, None], lg.shape[1], 1).reshape(-1)

    def nll(lt):
        z = L / np.exp(lt)
        z = z - z.max(1, keepdims=True)
        return float(-(z[np.arange(len(Y)), Y] - np.log(np.exp(z).sum(1))).mean())
    return _golden(nll)


def _golden(nll):
    a, b = np.log(0.05), np.log(50.0)
    r = (5 ** 0.5 - 1) / 2
    c, d = b - r * (b - a), a + r * (b - a)
    for _ in range(60):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - r * (b - a), a + r * (b - a)
    lt = (a + b) / 2
    return float(np.exp(lt)), nll(0.0), nll(lt)


def calib(lg, y, T=1.0, ticks=(0, 10, 30, 60, 120, 180, 239)):
    out = {}
    for t in ticks:
        t = min(t, lg.shape[1] - 1)
        z = lg[:, t] / T
        P = np.exp(z - z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        conf, hit = P.max(1), P.argmax(1) == y
        b = np.minimum((conf * 10).astype(int), 9)
        ece = sum(abs(hit[b == k].mean() - conf[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any())
        out[t] = dict(acc=round(100 * float(hit.mean()), 1), conf=round(float(conf.mean()), 3), ece=round(float(ece), 3))
    return out


def save(path, d, T):
    import torch
    d = dict(d, temperature=float(T))
    torch.save(d, path)


def cmd_fit(a):
    import torch
    torch.set_num_threads(a.workers)
    os.makedirs(a.out, exist_ok=True)
    info = {}
    jobs_tr = [(k, 200000 + 10000 * j + i) for j, k in enumerate("ABC") for i in range(a.train_games)]
    jobs_va = [(k, 230000 + 1000 * j + i) for j, k in enumerate("ABC") for i in range(a.val_games)]
    X, y, ccq, lens = record(jobs_tr, a.workers)
    Xv, yv, ccv, lv = record(jobs_va, a.workers)
    T0 = min(X.shape[1], Xv.shape[1]); X, Xv, ccq, ccv = X[:, :T0], Xv[:, :T0], ccq[:, :T0], ccv[:, :T0]
    info["decisions"] = dict(train=lens[:3] + lens[-3:], val=lv[:3] + lv[-3:], used=T0)
    rng = np.random.default_rng(0)
    labels = {"mix": y, "perm": rng.permutation(y),
              "ccs": ccq}   # placebo (design change after pilot 1): cc's own per-tick goal posterior, soft targets
    for name, lab in labels.items():
        d = train(X, lab, epochs=a.epochs)
        p = os.path.join(a.out, f"{name}.pt")
        save(p, d, 1.0)
        enc = E.load(p)
        lgv = logits(enc, Xv)
        yv_lab = {"mix": yv, "perm": rng.permutation(yv), "ccs": ccv}[name]
        T, n0, n1 = fit_temp(lgv, yv_lab)
        save(os.path.join(a.out, f"{name}_T.pt"), d, T)
        info[name] = dict(temperature=round(T, 4), val_nll_T1=round(n0, 4), val_nll_T=round(n1, 4),
                          val_true_goal_T1=calib(lgv, yv), val_true_goal_T=calib(lgv, yv, T))
    # enc_T: the #6607 encoder, temperature fitted on its own training table (recorder none_s0, probe table)
    from common_probe import record_chunk
    idx = list(range(20300, 20300 + a.val_games))
    with Pool(a.workers) as p:
        got = [z for c in p.map(record_chunk, [(a.rec0, idx[k::a.workers]) for k in range(a.workers)]) for z in c]
    Xr, yr = np.stack([z[2] for z in got]), np.array([z[1] for z in got])
    enc0 = E.load(a.enc0)
    lgr = logits(enc0, Xr)
    T, n0, n1 = fit_temp(lgr, yr)
    d0 = torch.load(a.enc0, map_location="cpu", weights_only=True)
    save(os.path.join(a.out, "enc_T.pt"), d0, T)
    info["enc_T"] = dict(temperature=round(T, 4), val_nll_T1=round(n0, 4), val_nll_T=round(n1, 4),
                         train_table_T1=calib(lgr, yr), train_table_T=calib(lgr, yr, T))
    json.dump(info, open(os.path.join(a.out, "fit.json"), "w"), indent=1)
    print(json.dumps(info))


# ---------- evaluation ----------

class TBayes(E.BeliefBayes):
    """BeliefBayes with q = softmax(logits / T). With T = 1 it is the sealed #6607 enc arm exactly."""

    def __init__(self, source, enc=None, seed=None, T=1.0):
        super().__init__(source, enc, seed)
        self.T = T

    def post(self, obs, c):
        if self.source == "ccid":   # identity check: cc's exact goal marginal through the swap interface
            return E.reweight(E.deck_posterior(c), goal_posterior(tuple(c)))
        if self.source != "enc" or self.T == 1.0:
            return super().post(obs, c)
        import torch
        m = E.deck_posterior(c)
        gru, head, mu, sd = self.enc
        x = self.ff(obs)
        with torch.no_grad():
            _, self.h = gru(torch.from_numpy((x - mu) / sd).float().view(1, 1, -1), self.h)
            q = torch.softmax(head(self.h.view(1, -1)) / self.T, -1).view(-1).tolist()
        return E.reweight(m, q)


_T = {}


def temp_of(path):
    if path not in _T:
        import torch
        _T[path] = float(torch.load(path, map_location="cpu", weights_only=True).get("temperature", 1.0))
    return _T[path]


def ev_one(job):
    arm, path, table, g = job
    src = {"cc": "cc", "oracle": "oracle", "ccid": "ccid"}.get(arm, "enc")
    T = temp_of(path) if path else 1.0
    ags, me, t = seat(table, g, lambda s: TBayes(src, E.load(path), seed=s, T=T))
    r = play(ags, seed=1_000_000 + g)
    return r.profits[me]


def cmd_eval(a):
    f = a.fit
    arms = {"cc": None, "enc": a.enc0, "enc_T": f"{f}/enc_T.pt", "mix": f"{f}/mix.pt", "mix_T": f"{f}/mix_T.pt",
            "ccs_T": f"{f}/ccs_T.pt", "perm_T": f"{f}/perm_T.pt", "oracle": None, "ccid": None}
    if a.only:
        arms = {k: v for k, v in arms.items() if k in a.only.split(",")}
    gs = list(range(a.offset, a.offset + a.games))
    prof = {}
    with Pool(a.workers) as p:
        for k, v in arms.items():
            prof[k] = np.array(p.map(ev_one, [(k, v, "E", g) for g in gs], chunksize=20))
        held = list(range(a.held_offset, a.held_offset + a.games)) if a.held_offset is not None else []
        if held:
            for k in ("cc", "mix_T"):
                prof[f"A_{k}"] = np.array(p.map(ev_one, [(k, arms[k], "A", g) for g in held], chunksize=20))
    ev = f"{EVALUATOR}@{os.environ.get('CODE_COMMIT', 'unknown')[:12]}"
    if a.only:
        out = {"evaluator": ev, "games": a.games, "offset": a.offset,
               "mean": {k: round(float(v.mean()), 3) for k, v in prof.items()}}
        for k in prof:
            if k != "cc":
                d = prof[k] - prof["cc"]
                out[f"{k}_minus_cc"] = {"value": round(float(d.mean()), 4), "ci90": E.ci90(d),
                                        "games_differing": int((np.abs(d) > 1e-9).sum())}
        json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out)); return

    def diff(x, y):
        d = prof[x] - prof[y]
        return {"value": round(float(d.mean()), 3), "ci90": E.ci90(d)}
    out = {"evaluator": ev, "games": a.games, "offset": a.offset, "held_offset": a.held_offset,
           "temperatures": {k: temp_of(v) for k, v in arms.items() if v},
           "mean": {k: round(float(v.mean()), 3) for k, v in prof.items()}}
    for k in ("enc", "enc_T", "mix", "mix_T", "ccs_T", "perm_T", "oracle"):
        out[f"{k}_minus_cc"] = diff(k, "cc")
    out["overconfidence_effect"] = diff("enc_T", "enc")       # temperature only, training-table fit
    out["coverage_effect"] = diff("mix", "enc")               # mixed per-tick training, no temperature
    out["temperature_on_mix"] = diff("mix_T", "mix")
    if held:
        d = (prof["A_mix_T"] - prof["A_cc"])
        out["A_mix_T_minus_cc"] = {"value": round(float(d.mean()), 3), "ci90": E.ci90(d)}
        out["note_shift"] = "opponent shift = A_mix_T_minus_cc - mix_T_minus_cc (different games; compare intervals)"
    # belief quality on table E, same games: the trader seat recorded as cc, every encoder read on that flow
    Xe, ye, cce, _ = record([("E", g) for g in gs], a.workers)
    out["calib_E"] = {"cc": calib(np.log(np.clip(cce, 1e-9, 1)), ye)}
    for k, v in arms.items():
        if v:
            out["calib_E"][k] = calib(logits(E.load(v), Xe), ye, temp_of(v))
    pl = prof["ccs_T"] - prof["cc"]
    out["placebo_gate"] = dict(rule="ccs_T - cc within +-1 $/game", value=round(float(pl.mean()), 3),
                               ci90=E.ci90(pl), passed=bool(abs(pl.mean()) <= 1.0))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--enc0", required=True); f.add_argument("--rec0", required=True)
    f.add_argument("--train-games", type=int, default=3000); f.add_argument("--val-games", type=int, default=500)
    f.add_argument("--epochs", type=int, default=30); f.add_argument("--workers", type=int, default=2)
    f.add_argument("--out", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--fit", required=True); e.add_argument("--enc0", required=True)
    e.add_argument("--offset", type=int, required=True); e.add_argument("--games", type=int, required=True)
    e.add_argument("--held-offset", type=int, default=None)
    e.add_argument("--only", default=None, help="comma list of arms; skips the full report (e.g. cc,ccid)")
    e.add_argument("--workers", type=int, default=2); e.add_argument("--out", required=True)
    a = ap.parse_args()
    {"fit": cmd_fit, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
