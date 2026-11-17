"""Causal test, arm B (docs/PREREG_FINAL.md): train the frozen supervised goal-suit encoder.

A recorder agent (E0 none_s0) plays probe-table games with indices [--first, --first + --games), which are game
seeds 50_000 + i (common_probe.record_chunk). Index ranges in the causal test:
    encoder  [10000, 20000)   seeds 60000-69999
    probe    [0, 3000)        seeds 50000-52999 (scripts/causal_eval.py)
    $/game   g in [100000, 103000), play seed 1_000_000 + g (scripts/pilot_eval.py --game-offset 100000)
PPO training games take 48-bit seeds from random.Random(train seed), so they never fall in these ranges in practice.

A one-layer GRU (hidden 128) reads the standardised flow-feature sequence and predicts the goal suit from its last
state, 30 epochs, Adam 3e-3, weight decay 1e-4 (the H1 recipe of controlled_probe_v2.seq_model, wider). 10% of the
games are held out to report accuracy. --placebo permutes the labels across games (seed 0) first.

    python scripts/train_encoder.py --record nets/none_s0.pt --out enc/encoder.pt
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_probe import record_chunk  # noqa: E402


def record(ckpt, first, games, workers):
    idx = list(range(first, first + games))
    chunks = [(ckpt, idx[w::workers]) for w in range(workers)]
    with Pool(workers) as pool:
        got = sorted(g for c in pool.map(record_chunk, chunks) for g in c)
    lens = {len(g[2]) for g in got}
    assert len(lens) == 1, f"games differ in decision count: {sorted(lens)[:5]}"
    return np.stack([g[2] for g in got]), np.array([g[1] for g in got])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--first", type=int, default=10000)
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.workers)
    X, y = record(a.record, a.first, a.games, a.workers)
    if a.placebo:
        y = np.random.default_rng(0).permutation(y)
    torch.manual_seed(0)
    mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    Y = torch.from_numpy(y.astype(np.int64))
    perm = np.random.default_rng(1).permutation(len(y))
    te, tr = perm[: len(y) // 10], perm[len(y) // 10:]
    gru = torch.nn.GRU(X.shape[-1], a.hidden, batch_first=True)
    head = torch.nn.Linear(a.hidden, 4)
    opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    for _ in range(a.epochs):
        for b in torch.randperm(len(tr), generator=g).split(128):
            i = torch.from_numpy(tr)[b]
            out, _ = gru(Xn[i])
            loss = torch.nn.functional.cross_entropy(head(out[:, -1]), Y[i])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out, _ = gru(Xn[torch.from_numpy(te)])
        acc = float((head(out[:, -1]).argmax(-1) == Y[torch.from_numpy(te)]).float().mean()) * 100
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(gru=gru.state_dict(), head=head.state_dict(), mu=torch.from_numpy(mu.astype(np.float32)),
                    sd=torch.from_numpy(sd.astype(np.float32)), in_dim=X.shape[-1], hidden=a.hidden), a.out)
    info = dict(record=a.record, games=[a.first, a.first + a.games], hidden=a.hidden, epochs=a.epochs,
                placebo=a.placebo, holdout_acc=round(acc, 2), n_holdout=len(te))
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
