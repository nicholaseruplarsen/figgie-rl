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

## Addendum, 2026-11-16, before the second launch
The first launch (compute f79c7429, $0.014) produced no result: the box's system Python had no torch 2.15.0 wheel,
so the install step failed before any training. Changes for the relaunch, made before any pilot number was seen:
- The box installs Python 3.12 and torch 2.15.0+cpu with uv (tested from an empty HOME on wsl2).
- Code memory-agents d8f4853: trial lengths 250/500/1000 updates (4.1/8.2/16.4 M steps), 2 rounds of 10, so the learning curve covers half a default run.
  Job sha256 `b7631ebef27f7d6c8fbdc1e354834dafb96d2c561463791ad33176f7f0dd6a34`, smoke log sha256 `5608eeb527a60d7498657c5c074aacc2b4d75b3aa9b8aca6de8004a47c1fe340`, binding dry-run ok.
- Pilot total stays within $3 (f79c7429 $0.014 + this ticket's --max-usd 2).
- Predictions 1–4 above are unchanged. Prediction 5 (billed cost of this ticket): $0.85, [$0.45, $1.90].
- Added, exploratory, for the per-trial budget of the reduced plan: among trials of 1000 updates, the Spearman
  correlation between training profit (mean of the last 25 logged updates) at update 250 and at update 500, and the
  final eval profit. Prediction: rho(500 vs final) 0.6, [0.0, 0.95], and rho(250 vs final) 0.4, [-0.3, 0.9].
  If rho(500) >= 0.7, per-trial lengths up to 500 updates are judged enough for ranking trials in the reduced plan.

## Addendum 2, 2026-11-16: first on compute-01's free slot
The paid relaunch was refused by compute's global $30/day cap. The pilot runs first on compute-01 (free tier, GPU1 slot,
4 GB RAM share, 3 trials at a time, 3 rounds), code memory-agents b1d7e58, job `runs/pilot/free/job.sh` sha256
`0262b638dc4dc4c7a98d17bc2e0e4e87171f4917bce5552c8006fb48407fb678` (smoke from an empty HOME: log sha256 `2fba68ce…`).
It measures compute-01's own throughput. A rented retry (≤$0.5, cheapest value box) follows when the day cap clears.
Predictions for compute-01 (median, 90%): core_h_per_Msteps 0.17, [0.10, 0.35]; train wall per 1000-update trial 1.3 h, [0.8, 2.5].
Predictions 1–4 above apply to the rented retry. If the slot OOMs, the job is relaunched with fewer trials at a time, never bigger.

## Addendum 3, 2026-11-16: commit hashes after the history rewrite
Bram rewrote the unpushed history (author and trailer cleanup) with byte-identical trees, so code is unchanged.
Hashes cited above map as: 59c2d30→446e730, d792d73→7f01288, 75497d9→c61e933, 0312375→988014d, 31f5289→771d2e8,
d8f4853→70131e7, b1d7e58→f42257b. Pilot results stamped CODE_VERSION b1d7e58 ran the code of f42257b.
Full map: `../figgie_rewrite_map.txt`. Old commits: refs/backup/pre-unsign/*.

## Results, 2026-11-16 (exploratory, scored against the predictions above)
- compute-01, free (d3a1e630, 9 trials, 3 rounds): core_h_per_Msteps 0.131 (predicted 0.17, [0.10, 0.35], inside). Wall per 1000-update trial 1.32 h with 3 at once (predicted 1.3 h, [0.8, 2.5], inside). 37 M steps per wall-clock hour, peak 2.47 GB of 4, no OOM. No thermal pause seen after the 5 s test at 06:58Z.
- Rented (78abf6c2): a mis-provisioned box (an RTX 3060 Ti host in Ontario, 24 vCPUs of a Xeon E5-2690 v3, $0.174/h, about $0.56). Only round 1 finished, 7 trials. Predictions: 1. core_h_per_Msteps 0.290 (inside [0.10, 0.45]). 2. ratio_to_plan 1.90 (inside [0.65, 3.0]). 3. 8-vs-2-worker time per update 0.72 (inside [0.30, 0.95]), 0.77 on compute-01. 4. box_core_utilisation 0.25 (miss, below [0.40, 0.95]). 5. Billed cost about $0.56 (inside [$0.45, $1.90]).
- Rank stability (compute-01, 1000-update trials, n = 3): rho(training profit at 500, final eval) 0.5 (predicted 0.6, [0.0, 0.95], inside). rho(at 250, final eval) -0.5 (predicted 0.4, [-0.3, 0.9], miss). Across all 6 trials of 500 updates or more, rho(at 250, eval) was -0.89. The rule (rho(500) >= 0.7) is not met, so trials of up to 500 updates are not judged enough. The reduced plan uses 500-2000 updates per trial and 4000-game evals.
- Bug, no effect on these numbers: pilot_tune.py waits for trials in launch order, so a trial that finished early has its train_wall_s recorded late (78abf6c2 trials 3-6). Round wall times and CPU times are unaffected.
- First launch f79c7429: $0.014, no result (install failed). Pilot total about $0.58, inside $3.
