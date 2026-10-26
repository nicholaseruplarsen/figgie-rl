# figgie-rl

One shared Figgie engine (standard rules, CS224R 240-tick / 257-action discretisation) with
agents re-implemented from four public sources, a reproduction of the CS224R PPO+PFSP agent,
and a mixed-table tournament (P&L with CIs, Elo, TrueSkill).

| Source | Agents | Status |
|---|---|---|
| [CS224R "RL for Figgie"](https://cs224r.stanford.edu/projects/pdfs/Daniel%20Li%20Yang%20submission_416100137/CS224R_Project_Report-5.pdf) | Random, FairValue, Conservative, MarketMaker, Bayesian, PPO | reimplemented from the report |
| [mneha05/figment](https://github.com/mneha05/figment) (MIT) | Fundamental, BayesianMM, AdaptiveMM, Momentum, Random | decisions identical to original |
| [revilo-25/Figgie-bot](https://github.com/revilo-25/Figgie-bot) | PotMM, HandMM, FlowMM, FixedSpread, Momentum, controls | decisions identical to original |
| [0xDub/figgie-auto](https://github.com/0xDub/figgie-auto) | Tilt, Spread, Seller, Noisy, PickOff, Hoarder, Mantis | reimplemented, code review only |

## Play

Play a game in the browser against the three most profitable agents from the tournament (`fig.AdaptiveMM`, `fig.BayesianMM`, `Bayesian`):

```bash
python scripts/play.py
```

Open the printed URL (http://127.0.0.1:8766). You sit at seat 0. Pick a suit with `1`–`4` or the arrow keys, set the price with `↑`/`↓`, then `b` bid, `a` ask, `l` lift, `h` hit, `c` cancel the suit's quotes, `x` cancel everything. Space pauses and `.` steps one tick. Orders go through `Game.apply` on your turn in the next tick, and the agents see the same `Obs` as in the tournament.

Swap opponents with `--opponents` (any registry name or `PPO-v1`, `PPO-v2`, `PPO-v2-long`, `PPO-noBC`), change the pace with `--tick 0.3`, or start paused with `--paused`. `--reload` restarts the server when a Python file changes, and the page is re-read from `scripts/play_web/index.html` on every load.

```bash
python scripts/play.py --opponents fig.AdaptiveMM PPO-v2 Bayesian --reload
```

Results: see [RESULTS.md](RESULTS.md).
See `DESIGN.md` for rules, every guessed detail, and provenance.

```bash
python -m pytest -q tests
python scripts/verify_ports.py       # needs the source repos cloned into third_party/
python scripts/tournament.py --tables 3000
./scripts/train_all.sh
```
