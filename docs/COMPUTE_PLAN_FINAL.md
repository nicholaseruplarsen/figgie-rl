# Compute plan for the final report

Written 16 Nov 2026. Deadlines: poster in the week-15 session (21–27 Dec), final report by 8 Jan.
Sized by what each question needs, not by a budget. Every rate below is measured in our own logs unless it says "estimate".

## What our logs say about speed

| Quantity | Value | Source |
|---|---|---|
| One PPO run, 2000 updates = 32.8 M agent steps (32 envs × 512 steps × 2000) | | `figgie/rl/train.py` defaults |
| none core, 2 env workers + learner, wsl2 CPU | 2.4–2.5 h of updates (4.1 h when two runs shared 4 cores under load) | `runs/memory/none_s{0,1,2}/train_log.jsonl`, sum of `sec` |
| minGRU core, same setup | 3.1–3.4 h (3.8 h under load) | `runs/memory/mingru_s*/train_log.jsonl` |
| GRU core, same setup | 3.3–3.6 h (5.2 h under load) | `runs/memory/gru_s*/train_log.jsonl` |
| transformer core, learner on a compute-01 3090, 2 env workers | 8.5 h, with update time rising from 4.5 s to 15.6 s as transformer snapshots fill the self-play pool | `runs/memory-gpu/transformer_s0/train_log.jsonl` |
| transformer core s1, updates 151–2000, learner on a rented Vast 3090, 16 env workers (31 CPUs) | 3.4 h, 6.57 s/update on average (5.5 min per 50 updates), $0.87 for the whole job | `runs/memory-diag/transformer_s1/train_log.jsonl`, compute job 47e2677b |
| Probe (3000 games) + eval (6 matchups × 1000 games) per run | 0.5–1.5 h on 2 cores | queue log timestamps minus training time |
| Recording games for a probe, one core | 40 games in 3.3 s, so 3000 games in about 4 min | `scripts/common_probe.py` smoke test, 15 Nov |
| Engine alone, heuristic agents | about 11 ms per game, 85,000 decisions/s per core | `RESULTS.md` §1 |

The bottleneck is CPU simulation, not the GPU. Every step runs four agents in Python on CPU, and the self-play opponents run their own network forward pass per tick inside the env workers. The learner's backward pass is small (128-wide nets). The GPU run of the transformer was no faster per update than CPU runs of the other cores, because its time went into the transformer opponents in the CPU workers.

So the useful unit is the core-hour. One 2000-update run costs about 5 core-h for none, 7 for minGRU and GRU, and about 25 for the transformer (3 busy cores for 8.5 h). The probe and eval add about 2 core-h per run.

Not measured cleanly: how one run scales past 2 env workers. The one data point is transformer s1 above, at 16 workers, but its self-play pool started empty at update 150, so it is not comparable to s0. The learner steps all 32 envs in one process and waits for the slowest worker, so going from 2 to 8 workers will give less than 4×. The plan therefore gets its speed from running many runs side by side, 2–3 cores each, and measures 2/4/8 workers in the first hour of the first rented box.

## Machines and prices

| Option | What we get | Price | Notes |
|---|---|---|---|
| wsl2 CPU | 4 of 8 cores | $0 | Not available: Nicholas's rule since 15 Nov is no research or training on wsl2. |
| compute-01 (free tier) | one job at a time, 4 GB RAM cap, 3090 when the owner isn't on it | $0 | GPU 1 was held by the owner again on 15 Nov. Quote on 16 Nov: our tenant can only borrow idle room. Good for small jobs, too slow for the long runs (see item 1). |
| Vast, many-core box | 32 cores with any GPU | quoted $0.20/h (3090 box), $0.36/h (4090), $0.52/h (5090) | `compute quote --data-class public --vram 4 --ram 32 --cpus 32`, 16 Nov. The auto ranking picked these on GPU benchmark, but for us the cores are what count. |
| Vast, single GPU | 1× 3090 / 4090 / 5090, 8 cores | quoted $0.33/h / $0.80/h / $1.17/h | `compute quote --data-class public --gpu <type> --cpus 8`, 16 Nov |

Andrew's warning: real Vast bills have run about 3× the quote (open bug T-5004, 3090 quoted $0.095/h and billed $0.29–0.31/h). Below, "quoted" uses the quote and "billed" uses 3× it.

A faster GPU does not help us. A 5090 box costs 2.6× a 3090 box for the same 32 cores. Its GPU benchmark advantage applies only to the learner, which is a small part of our time. More cores win: one 32-core box runs about 12 PPO runs side by side, each at the speed we measured on wsl2.

Two limits apply. Each `compute` request is capped at 12 h and $10, so long work is split into resumable leases (latest.pt every 25 updates, written into `out/`). And tenant geoff's paid cap is $10 in total, of which Ashish's transformer s1 run may use up to $6. The plan below needs about $40–75 billed, so it needs a new cap from Nicholas before item 1 starts.

## The items

### 1. Three seeds of all four memory cores, trained long enough for recall to pass

What length: we do not know yet, and guessing would be wrong. At 2000 updates no core's last hidden state carries more goal-suit information than its current input, and training profit was still rising at update 2000 for every core except the transformer (minGRU s1 from 72 to 85 $/game, GRU s1 from 53 to 70, `train_log.jsonl`). The known-answer recall task (item 3) measures directly how many updates each core needs before it can carry a fact across a game. The length for item 1 is the larger of that number and the point where the training profit curve flattens.

Working estimate: 4× the current length, 8000 updates (131 M steps) per run. That is long enough for profit to flatten if it follows the current trend, and it is the size we budget for. If item 3 says recall needs more, we extend.

Runs × steps: 4 cores × 3 seeds × 131 M = 12 runs, 1.6 G steps. The 9 none/minGRU/GRU runs continue from their 2000-update `final.pt` (6000 more updates each). The 3 transformer runs start fresh, since s1 is still finishing at 2000.

| | none | minGRU | GRU | transformer | total |
|---|---|---|---|---|---|
| core-h per run at 8000 updates (4× measured) | 20 | 28 | 28 | 100 | |
| × 3 seeds, plus probe and eval | 66 | 90 | 90 | 306 | about 550 core-h |

| Where | Wall time | $ per finished run (billed) | Verdict |
|---|---|---|---|
| wsl2 CPU | not allowed | | |
| compute-01 free, one job at a time | about 12 runs × 14–34 h ≈ 230 h ≈ 10 days, if the GPU and RAM are free | $0 | Too slow: finishes after the poster if anything slips. |
| One rented 4090 or 5090 box with 8 cores | ≈ 550 / 7 usable cores ≈ 80 h | 5090: about $94 quoted, $280 billed for all 12, ≈ $23 per run | Worst option: paying for GPU speed we can't use. |
| Rented 32-core box (3090-class GPU) | ≈ 550 / 30 ≈ 18–20 h in two 10 h leases | quoted ≈ $4, billed ≈ $12 for all 12, ≈ $1 per run | Recommended. |

Recommended: one 32-core Vast box, ≥ 64 GB RAM (12 runs × ≤ 5 GB, after the transformer s1 OOM fix), two 10 h leases, 23–25 Nov. Measure 2/4/8 workers in its first hour and pack runs accordingly.

### 2. Common-trajectory and random-init probes

Runs × steps: 1 recording of 3000 games by none_s0 plus a forward pass of every network (12 trained, 8 random-init) over them. No training.
Bottleneck: CPU, about 4 min to record, then minutes of probe fitting.
Hours: about 0.5 core-h anywhere. compute-01 free tier fits it (well under 4 GB).
$ per result: $0 on compute-01, or cents if it rides along on the item-1 box.
Recommended: compute-01 free tier now (after transformer s1 lands, about 20 Nov), and again on the item-1 box after the long runs, so every final agent is probed on the same games.

### 3. Known-answer recall task

Design: a cue that only appears in the first ticks of a game (for example a random suit shown in an otherwise unused input channel at tick 0) and pays a bonus at the end if the agent acts on it. A linear probe at tick 239 must recover it. A core passes when the probe and the bonus show it carries the cue.
Runs × steps: 4 cores × 2 seeds, trained until they pass or to a ceiling of 4000 updates (65 M steps). It uses the same engine, so the rates above apply. Plus a minutes-long synthetic version (no Figgie) as a unit test of each core.
Bottleneck: CPU simulation.
Hours (estimate): at most 8 runs × 4000 updates ≈ 8 × (10–50 core-h) ≈ 150 core-h worst case, much less if they pass early. Wall time about 6 h on a 32-core box, or 3–5 days on compute-01.
$ per result: billed about $0.5 per core type on a 32-core box.
Recommended: rented 32-core box, 19–22 Nov, before item 1, because its answer sets item 1's length. The synthetic version runs first on compute-01.

### 4. Q1 sequence model

Runs × steps: data from 5 opponent populations × 200,000 games, then one small sequence model (GRU or transformer, about 1 M parameters) trained on all four seats' trade histories, 240 ticks each.
Bottleneck: data generation is CPU (heuristic tables at 11 ms per game ≈ 3 core-h in total). Training is GPU.
Hours (estimate, not measured): about 2 h on a 3090, about 1.4 h on a 4090, about 1 h on a 5090. The data is 800,000 sequences × 240 steps per epoch, for about 20 epochs.
$ per result: compute-01 3090 $0. Rented, billed: 3090 about $2, 5090 about $3.5. The GPU types are close per finished model, so pick by availability.
Recommended: generate data on the item-1 box (it is idle during the leases' last hours) or compute-01, and train on compute-01's 3090 when the owner frees it. Otherwise use the cheapest rented GPU. 23–29 Nov.

### 5. Leakage classifiers

Runs × steps: 29 agents × 10,000 games each to record actions and hands, then one classifier per agent (logistic, then a small MLP).
Bottleneck: CPU simulation, with PPO agents slower (about 80 ms per game) than heuristic ones (11 ms).
Hours (estimate): about 1 core-h for the heuristic agents plus about 0.25 core-h per PPO agent, about 6 core-h in all. Classifiers take minutes each on CPU.
$ per result: $0 on compute-01, cents on a rented box.
Recommended: compute-01 free tier, 1–7 Dec.

### 6. Best-response PPO agents

Runs × steps: about 10 target agents × 20 M steps (1220 updates) each, with one best-response learner against three copies of the target. One seed each, plus a second seed for the three targets the poster leans on: 13 runs.
Bottleneck: CPU simulation. Cost per run depends on the target's own per-tick cost.
Hours: heuristic targets ≈ 0.6 × the GRU run ≈ 4–5 core-h each. PPO targets ≈ 6–8 core-h. A transformer target ≈ 15 core-h. About 90 core-h in all.
Wall time: about 4 h on a 32-core box, or about 9 days on compute-01 one job at a time.
$ per result: billed ≈ $0.25 per best response on a 32-core box (6 h × $0.20 × 3 / 13).
Recommended: rented 32-core box, one 6 h lease, 8–12 Dec, leaving 13–20 Dec for analysis and the poster.

## Summary

| Item | Runs × steps | Bottleneck | Core-h (or GPU-h) | Best option | Billed $ (≈3× quote) | Dates |
|---|---|---|---|---|---|---|
| 1 Memory cores, 3 seeds, 8000 updates | 12 × 131 M | CPU | ≈ 550 | 32-core Vast box, 2 × 10 h | ≈ 12 (single 5090 box: ≈ 280) | 23–25 Nov |
| 2 Common-trajectory + random-init probes | 20 nets × 3000 games | CPU | ≈ 0.5 | compute-01 free | 0 | ≈ 20 Nov, again after item 1 |
| 3 Known-answer recall | 8 × ≤ 65 M | CPU | ≤ 150 | 32-core Vast box, 6 h | ≤ 4 | 19–22 Nov |
| 4 Q1 sequence model | 1M sequences, ≈ 20 epochs | GPU (data on CPU) | ≈ 2 GPU-h + 3 core-h | compute-01 3090, else cheapest rented GPU | 0–3.5 | 23–29 Nov |
| 5 Leakage classifiers | 29 agents × 10k games | CPU | ≈ 6 | compute-01 free | 0 | 1–7 Dec |
| 6 Best responses | 13 × 20 M | CPU | ≈ 90 | 32-core Vast box, 6 h | ≈ 3.5 | 8–12 Dec |
| Contingency (reruns, longer item 1 if recall needs it) | | | | | ≈ 20–50 | |

Total billed, about $40–75 including contingency, against a tenant cap of $10. Nicholas needs to set a cap before item 3.

The many-core box beats a faster GPU for everything except item 4. With 32 cores at a 3090-box price, the twelve long runs cost about $1 each billed, against about $23 each on a lone 5090.

## Probe protocol: sort the games

`scripts/probe_ppo.py` and `scripts/placebo_probe.py` concatenate the recorded games in worker-chunk order (`range(w, games, workers)`), and the 5-fold `StratifiedKFold(shuffle=True, random_state=0)` assigns folds by position. So `--workers` changes the CV split, though the games and hidden states are identical. Every sealed probe used `--workers 2`. A rerun at 31 workers moved transformer s1 from 44.03% to 45.03% and minGRU s0's placebo true-label accuracy from 58.30% to 56.9%. At 2 workers, main's script reproduces none_s0, minGRU_s0 and GRU_s0 exactly (compute-01 job 2e38f311). Every scored probe should keep `--workers 2`. New probes should sort games by index before the CV, which changes the split once, so those numbers are a new protocol and must not be mixed with the sealed ones.

