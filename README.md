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

Work in progress: PPO training and the full tournament are running; results will go in `RESULTS.md`.
See `DESIGN.md` for rules, every guessed detail, and provenance.

```bash
python -m pytest -q tests
python scripts/verify_ports.py       # needs the source repos cloned into third_party/
python scripts/tournament.py --tables 3000
./scripts/train_all.sh
```
