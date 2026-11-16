# Cost pilot for the tuner x optimizer factorial: what we expect, written before launch

Written by Judea on 2026-11-16, committed before the `compute run` (tenant geoff). Exploratory: launched with
`--exploratory`, never scored on Bret's platform, and outside the 2x2 comparison (only runs through
tsinghua-rl's `iig/tune` count there). This file is the pre-registration.

## What runs
- Code: memory-agents 31f5289, `scripts/pilot_tune.py` and `scripts/pilot_eval.py`. Job: `runs/pilot/exp/job.sh`, sha256 `0951614de1cc6e807e5e70fb3e8c842a74deba2e6217ca5ed8fc1a4d057fe789`.
- Box: one rented CPU box, >= 32 vCPUs, 32 GB RAM, region eu (quoted $0.243/h, 3090 host in Spain), lease 5 h, --max-usd 2.
- Phase A: worker scaling, PPO none core, 30 updates at 2, 4 and 8 env workers, one at a time.
- Phase B: up to 3 rounds of (ncpu-2)//3 concurrent trials (10 on 32 vCPUs), none core, PPO, AdamW, BC init = none_s0's bc.pt,
  lengths cycling 125/250/500 updates (2.0/4.1/8.2 M steps), hyperparameters from a PROTEIN-style stub (GP-UCB per log-cost).
  Each trial: train (2 env workers + learner) + 500 seat-rotated games against FV + Bayes + MM.
- Smoke: `SMOKE=1 bash job.sh` on wsl2 (2 cores), log sha256 `4663e3e7322f6965683dc1a71c4b6616e48fc4438a7b59e3bb5c75d2a360fe03`.
- Binding dry-run: ok on the smoke summary (`runs/pilot/binding-dry-run.json`), field `pilot.core_h_per_Msteps`.

## Predictions (median, 90% interval)
The plan's figure is 0.153 vCPU-core-h per million env steps (5 core-h per 2000-update none run on wsl2).
Rented Xeon/EPYC vCPUs are often slower per thread than wsl2's Ryzen 5700X and are hyperthreads, so the intervals lean high.
1. core_h_per_Msteps (train + eval, rusage): 0.20, [0.10, 0.45].
2. ratio_to_plan: 1.3, [0.65, 3.0].
3. sec/update at 8 workers divided by sec/update at 2 workers: 0.55, [0.30, 0.95].
4. box_core_utilisation (summed trial CPU time / (tuning wall time x ncpu)): 0.75, [0.40, 0.95].
5. Billed cost of the ticket: $0.80, [$0.40, $1.90].

## What follows from it
If ratio_to_plan > 1.5, every tier of docs/research_plan.tex is re-costed before 23 Nov.
