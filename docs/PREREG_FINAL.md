# Pre-registration of the final runs (docs/research_plan.tex)

Judea keeps this file. Every prediction listed here is sealed on Bret's platform (project tsinghua)
before the run it predicts. A sealed prediction names its evaluator stamp and a binding: a JSON
path and field. A stamp is `<script>@<first 12 hex of the figgie-rl commit>`, and the sealed
evaluators run from frozen copies of that commit, so later code changes cannot alter what they
compute. Results from runs flagged exploratory never resolve a prediction.

## Sealed

| Seq | Sealed (UTC) | Quantity | Median [90%] | Evaluator | Binding (under figgie-rl/runs/) |
|---|---|---|---|---|---|
| 4366 | 16 Nov 10:46 | P1 Δ_cond(none), placebo, points | 0 [−1, +1] | controlled_probe@dec139d550e3 | e2/probe/out/summary.json `placebo.delta_cond` |
| 4367 | 16 Nov 10:46 | P2 Δ_cond(minGRU), best tuned arm | +0.5 [−1.5, +5] | same | `groups.mingru.delta_cond` |
| 4368 | 16 Nov 10:46 | P3 Δ_rand, tuned none | −5 [−12, +2] | same | `groups.none.delta_rand` |
| 4369 | 16 Nov 10:46 | P3 Δ_rand, tuned minGRU | −4 [−12, +4] | same | `groups.mingru.delta_rand` |
| 4382 | 16 Nov 11:00 | P4 PROTEIN − random at B_F, mean of 4 pairs, $/game | +1 [−5, +8] | e2_score@cf88eaa0e42d | e2/score/summary.json `p4.value` |
| 4383 | 16 Nov 11:00 | P5 Muon − AdamW, mean of 4 pairs | 0 [−6, +6] | same | `p5.value` |
| 4384 | 16 Nov 11:00 | P6 best tuned none − none recipe (E2) | +4 [−3, +15] | same | `p6.value` |
| 4385 | 16 Nov 11:00 | P7 share of B_F at which PROTEIN stabilises | 0.4 [0.1, 1.0] | same | `p7.value` |
| 5117 | 16 Nov 16:17 | R1 recall: Δ_cond(minGRU) − Δ_cond(none), cue at first decision | +15 [0, +38] | controlled_probe_v2@ffc5a04797f0 | recall/recall_final/recall_probe/summary.json `memory.diff` |
| 5118 | 16 Nov 16:17 | R2 recall: Δ_cond(GRU) | +10 [−2, +38] | same | same file, `groups.gru.delta_cond` |
| 5119 | 16 Nov 16:17 | H1 supervised sequence GRU − last-input probe, 10000 E0 games | +1 [−1, +6] | same | recall/recall_final/hsuff/summary.json `seq.gain` |
| 5441 | 17 Nov 02:01 | Causal: oracle − none, $/game | +3 [−1, +10] | causal_eval@bccdb9547894 | causal/final/causal/summary.json `oracle_minus_none.value` |
| 5442 | 17 Nov 02:01 | Causal: B (frozen encoder) − none, $/game | +1 [−3, +6] | same | `b_minus_none.value` |
| 5443 | 17 Nov 02:01 | Causal: A (aux 0.1) − GRU, $/game | +0.5 [−3, +5] | same | `a_minus_gru.value` |
| 5444 | 17 Nov 02:01 | Causal: closure_b | 0.3 [−0.5, 1.2] | same | `closure_b.value` |
| 5445 | 17 Nov 02:01 | Causal: closure_a | 0.15 [−0.5, 1.0] | same | `closure_a.value` |
| 5446 | 17 Nov 02:01 | Causal: output probe, B − none, points | +2 [−1, +7] | same | `out_b.value` |
| 5447 | 17 Nov 02:01 | Causal: output probe, A − GRU, points | +1 [−1, +5] | same | `out_a.value` |
| 5448 | 17 Nov 02:01 | Causal: state probe, B − none (sanity check) | +8 [+3, +10] | same | `q_b.value` |
| 5449 | 17 Nov 02:01 | Causal: state probe, A − GRU (sanity check) | +5 [0, +10] | same | `q_a.value` |

The 90% intervals and medians are the plan's. The quartiles were interpolated when sealing; they
are in `runs/prereg-final/*_sealed.json`.

## Deviations and addenda (each made before the run it affects)

1. **16 Nov 14:05 UTC+1: tsinghua-rl iig/tune re-pinned from 4db2c4f to bb616dd.**
   - P4–P7's note names 4db2c4f. bb616dd adds PROTEIN batch diversity: the Kriging believer plus a
     minimum pairwise distance of 1.0 in exponent space, inside a batch. Rich made this a blocker
     for every PROTEIN cell in both halves, so that near-duplicate trials don't make PROTEIN look
     worse than it is. bb616dd also adds `Space(section=...)`.
   - No PROTEIN cell had run when the pin changed. The four random-search cells run first. A
     regression test pins RandomSearch's draws as bit-identical across the two pins, so the random
     cell already running is unaffected.
   - Known open issue (Kent): the PROTEIN acquisition (mu + beta·sd)/log(cost) favours longer
     trials when mu < 0. Both halves fix the cost per batch slot, so it does not change any choice
     here.
2. **16 Nov: E1 cell order.** The cells were reordered so that all random-search cells run before
   any PROTEIN cell (point 1). The design is unchanged.
3. **16 Nov: evaluator v2 for the recall check and H-suff.**
   - `scripts/controlled_probe_v2.py` chooses C by nested CV on the outer train folds, sizes
     random-init nets to the trained nets' input, and adds a supervised sequence GRU.
   - v1 stays the evaluator of P1–P3 unchanged. Whether the v1 GRU result (Δ_cond −1.4 on E0) is a
     fixed-C artefact is checked by `scripts/probe_audit.py` (exploratory), and the answer goes here
     before E2 is scored.

## Probe audit (exploratory, job 489626d0, before R1-R2 and H1 were sealed)

The audit used `scripts/probe_audit.py` on the 11 E0 nets and 3000 games. Every net's state is 128 wide.

- **Standardisation and the input block.** v1 standardises each dimension. With zeros in place of the state, the
  probe reproduces the input-only accuracy (59.97 against 59.93, one game in 3000).
- **Fixed C is part of GRU's −1.4.** If C is chosen by nested CV on the train folds, GRU's Δ_cond goes from −0.83,
  −1.73, −1.57 to −0.47, −0.90, −1.23 (mean −1.38 → −0.87). The minGRU − none difference shrinks from +1.04 (v1) to
  about +0.35.
- **Width costs accuracy.** Appending a shuffled, information-free state costs 2–3 points under either probe. So
  Δ_cond is biased down by width, and the none placebo comes out near 0 in v1 only because that loss cancels the
  gain from none's nonlinear input features (+2.5 to +3.3 over the shuffled control).
- **Measured against the shuffled control** (same width, nested C): none +2.73, minGRU +2.39, GRU +1.48. Memory adds
  nothing over none here.
- **Effect on the filed predictions.**
  - P1–P3 stay sealed against v1 and are scored as filed. v1 is biased toward finding memory by roughly 0.7 points
    (E0), and the scoring will report the v2 and shuffled-control numbers beside them as a sensitivity analysis.
  - P3 (Δ_rand) uses the state-only probe. There nested C moves accuracy by 0.0–0.9 points, so its interval is
    unaffected.
  - R1–R2 and H1 use v2.

## To be sealed before their runs

Nothing is pending. R1, R2 and H1 were sealed at 16:17 UTC, before the recall job launched.

## Causal test of the learning-failure hypothesis (sealed as #5441–5449 before launch)

H1 showed that the public history holds about 10 points of goal-suit information that the PPO cores' states do not
use. One explanation is a learning failure: supervised training gets a dense label, PPO only a sparse reward. This
test checks whether giving PPO the information, or a dense signal toward it, changes what it does.

- **Arms**, 3 seeds each (1–3), 2000 updates (32.8M steps), BC init, the E0 recipe, all trained fresh in one job:
  - `none` (baseline).
  - `oracle`: none with the true goal suit, one-hot, in the input at every decision. This is the upper bound on what
    goal information is worth.
  - `b`: none with the hidden state of a frozen supervised GRU encoder (hidden 128) in the input. The encoder is
    trained on goal suit from flow-feature histories.
  - `gru` (baseline).
  - `a`: GRU plus an auxiliary goal-suit cross-entropy head on the core output at every decision, coefficient 0.1.
- **Games, disjoint by construction.**
  - Encoder: probe-table indices [10000, 20000), game seeds 60000–69999, recorded by E0 none_s0.
  - Probes: indices [0, 3000), seeds 50000–52999, recorded by this job's none s1.
  - $/game: g in [100000, 103000), play seeds 1,100,000–1,102,999.
  - PPO training: 48-bit seeds drawn from `random.Random(train seed)`, outside these ranges in practice.
- **Primary measure: $/game** against FairValue, Bayesian and MarketMaker, seats rotated. Seeds are averaged per
  game, with a 90% bootstrap CI over games.
  - closure_b = (b − none)/(oracle − none)
  - closure_a = (a − gru)/(oracle − none)
- **Secondary measure: the output probe.** A logistic probe on [input, logits, value] at the last tick, minus the
  input alone, with arm minus baseline. It tests whether the information reaches the decisions.
- **Sanity check only:** the state probe (controlled_probe_v2's Δ_cond, arm minus baseline). For b it passes almost
  by construction.
- **Pass rule.** For arm b or a, the learning-failure claim is supported if all three hold:
  - closure ≥ 0.5,
  - the $/game difference has its CI90 above 0,
  - the output-probe difference has its CI90 above 0.

  If oracle − none has a CI90 that includes 0, goal information is not worth money here, and the verdict is "no
  incentive", not a pass or a fail.
- **Single aux coefficient.** Arm a runs at one coefficient, 0.1, chosen before any result. A failed a does not refute
  aux-loss methods in general, only this setting, and no coefficient sweep follows the results.
- **Named placebo** (the same driver, `PLACEBO=1` in runs/causal/job.sh): goal labels permuted across games (seed 0)
  for the encoder, and a random suit per game for the oracle input and the aux target. Each difference must land
  inside its interval on the unfavourable side, or within ±1.
- **Evaluator:** `scripts/causal_eval.py`, stamp causal_eval@<commit>, frozen as a code copy in the job dir.
  Matched-width H1 (`scripts/h1_width.py`, hidden 128) runs in the same job as an exploratory check.
- **Compute and why it is rented.**
  - compute-01's six shared threads are full until E1 ends, about 23 Nov (E1 at 4, m2 at 2).
  - At 4 cpus the test would take about 5 days after that, landing about 28 Nov, after the ~23 Nov midterm.
  - So one rented box runs it: --gpu auto, about 48 vCPU, 15 runs side by side, about 10–12 h.
  - Estimate: $2.65 at the measured $4.7 per 1e9 steps, $3.3 with margin. Approved within the $22 stop.
  - A lease that ends mid-run resumed from latest.pt, weights only (Adam, the PFSP pool and the aux head
    restarted). Any such resume will be reported. This held for every run up to 2026-11-20.
  - From 2026-11-20 (judea/e1-resume), latest.pt also holds the optimizer state (Adam or Muon), the PFSP pool with
    its EMA win rates and the aux head, and a resume restores all of them. Each resume writes a row
    `resumed: true, resumed_at_update: N` to the run's train_log.jsonl. E1 copies it into the trial's row in
    trials.jsonl, so the ledger shows which trials resumed and where.
  - Resume proof (compute-01, 2026-11-20): a CPU run was killed with SIGTERM at update 60 and resumed from latest.pt
    at update 50 with full state (optimizer present, PFSP pool of 16 with its win rates, pool_next_id 49). Updates
    then ran on to 200. A smoke E1 cell was killed after 2 tells and rerun: tells 0-4 were each recorded once and none
    were null.

## E1 deviation: PROTEIN cells stop at the 4e8 anytime checkpoint (2026-11-21, logged before any results were read)

- Paid CPU throughput was about 15M steps/h, measured on none-protein-adamw in lease 1. The plan assumed about 34M/h.
  Taking all four PROTEIN cells to 8e8 needed about 2.60B more steps (about $13.7), against $5.37 left under the
  $6.50 cap.
- Decision (Demis, adopted by Geoff): every PROTEIN cell stops at the pre-registered 4e8 anytime checkpoint
  (research_plan.tex l.63), so the 2×2 stays balanced. If the cap arrives first, every cell stops at the highest
  equal checkpoint reached inside it.
- P4–P7 are scored at 4e8 (or at that lower checkpoint), and every report states this deviation. The random cells
  are read at the same checkpoint.

## E1 deviation: every cell is scored at 1e8, not 4e8 (2026-11-22, logged before any 1e8 score was read)

- Reason: the $6.50 PROTEIN budget cap. After lease 3 ($3.36 spent), the steps finished were none-protein-adamw 401M,
  none-protein-muon 369M, mingru-protein-adamw 82M and mingru-protein-muon 82M. minGRU gained about 28M per cell in a
  10.5 h lease, so reaching 2e8 would need about 3 more full leases, which is over the cap. 1e8 is the highest
  common checkpoint that fits. Lease 4 (about $0.30) finishes minGRU trial 5 to get there.
- All eight cells, random search included, are scored at 1e8 from their logged anytime curves (best-so-far
  over trials whose cumulative cost_steps is at most 1e8). No cell is scored at 4e8 or 8e8.
- If the curves show that 1e8 is shorter than the length needed to rank the variants (best-so-far still moving
  by more than its CI, or the rank order changing after 1e8 in cells that ran longer), the tuner comparison is
  marked exploratory.

## E2 retired; P1–P6 retired unscored (2026-11-25, CRO ruling by Demis)

- E2 (the multi-seed confirmation of the E1 winners) will not run, on paid or free compute. The reason is opportunity
  cost: the primary question is the belief-to-action gap, and E1 already showed Muon − AdamW ≈ 0 (−0.63 $/game) and
  all four PROTEIN cells peaking on the same trial-3 proposal.
- P1–P3 (read from the E2 probe) and P4–P6 (read from E2 means by e2_score.py) are retired and unscored. They are
  not pending or overdue. P7 is reported from E1 at 1e8 (0.655), with the deviation above.
- E1 is reported on its own at 1e8 by scripts/e1_at_budget.py (commit 82d93a6), as exploratory input, not as P4/P5.

## E1 is exploratory (2026-11-25, the 1e8 deviation's first trigger fired)

- The best-so-far was still moving by more than its CI after 1e8. none-random-adamw went 14.06 → 15.92 (2e8)
  → 16.90 (8e8), +2.84 against a single-trial 95% CI of about ±2.3, and the none-core PROTEIN − random gap fell
  from +4.4 at 1e8 to about +1.5 at 8e8. No rank flipped (second trigger), but one trigger is enough, so the tuner
  comparison is labelled exploratory.
- 95% intervals at 1e8: PROTEIN − random +6.20 ± 1.61, Muon − AdamW −0.63 ± 1.61. These cover evaluation noise
  only (each winner's score_ci95, combined in quadrature, pairs treated as independent). They ignore the
  winner's-curse selection and the shared draw below, so they are too narrow.
- All four PROTEIN wins come from the same trial-3 proposal (one set of parameters, chosen identically in every
  cell). They are one draw, not four replications.

## Standing rules

- **Binding gates (from 2026-11-21).** Every scored result is passed through scripts/binding_verdict.py, sealed at
  commit 6c40afc15f83. Any failed gate (a "valid": false, a false member of "gates", a failed placebo_gate, an
  "invalid probe" verdict, or a PLACEBO=1 arm whose dollar CI90 excludes 0) sets that verdict to invalid. A failed
  oracle placebo invalidates every arm. The sealed evaluators are unchanged, and reports quote the binding verdict.
- **No seal without a pilot (from 2026-11-20).** A quantity can be sealed only if a pilot has measured it first.
  The pilot value, its job id and its game or seed range go into this file before the seal. The pilot range must
  be disjoint from the sealed range. Prompted by the belief-v2 placebo, whose sealed form would have failed.

## Results

### H1, H-suff (#5119), job 1b61b9aa, evaluator controlled_probe_v2@ffc5a04797f0

On 10000 E0 games, the supervised sequence GRU reaches 73.54% and the last-input probe 63.72%, a gain of
**+9.82 points [8.97, 10.66]**. That is above the sealed 90% interval [−1, +6], so **H-suff is refuted**: the public
history holds about 10 points of goal-suit information beyond the last input. On the same games, trained minGRU
adds +1.18 [0.81, 1.58] over none (v2; placebo none +0.28, valid). The small memory gain is therefore a learning
failure, not sufficiency: a hypothesis until a causal test (pre-registered separately) supports it.

**Control.** Placebo job 3030a827 (driver `scripts/hsuff_placebo.py` @a8ea46b, scoring code
controlled_probe_v2.py blob 4ff4cd60, identical to the frozen ffc5a04; goal labels permuted across games, seed 0)
gives +0.71 [−0.30, +1.67], with chance at 25.3%. The control passed, and #5119 is resolved at +9.82.

**Capacity.** The supervised GRU is 1 layer, hidden 64, on the 85 input features: 29,952 parameters plus a
260-parameter head. The PPO cores have a 128-wide state on a 27.5k-parameter encoder. The GRUCell adds 99.1k, minGRU
adds 33.0k, and none adds nothing. The supervised model is narrower and smaller than both memory cores, so capacity
cannot explain its advantage. Its accuracy comes from its own head, not from a probe on a wide state, so the width
penalty does not enter H1. That penalty lowers the PPO cores' Δ_cond by up to 2–3 points, so the gap between the
supervised model and the PPO cores is roughly 5–8 points, not 9.8.

### R1, R2, the recall check (#5117, #5118), jobs 1b61b9aa and 8c82fa4b, evaluator controlled_probe_v2@ffc5a04797f0

**Deviation.** The first lease (6 h) ended with minGRU s1/s2 and GRU s1 at 1800, 1825 and 1725 updates (their last
checkpoints). Job 8c82fa4b resumed them to 2000 from those weights, with the Adam state and the PFSP pool reset.
The none runs had finished.

| Core (seeds) | State acc | Δ_cond, points [CI90] | Per seed |
|---|---|---|---|
| none (1, 2) | 53.5 | −1.55 [−2.38, −0.77] | −1.70, −1.40 |
| minGRU (1, 2) | 55.3 | +0.60 [−0.28, +1.53] | +0.87, +0.33 |
| GRU (1) | 45.7 | −0.97 [−1.97, +0.07] | −0.97 |

The input alone at tick 239 scores 63.0% (3000 games recorded by none_s1, recall).

- **R1** (minGRU − none) = **+2.15 [1.30, 3.05]**, inside the sealed interval [0, +38] near its low end.
- **R2** (GRU) = **−0.97**, inside [−2, +38].
- **No core passes.** The bar was ≥ 10 points over none.
- Under the sealed rule this probe run is **invalid**: the none placebo is −1.55, outside ±1. It is reported as
  invalid and not rerun.
- **Cause of the drift (checked after the run).**
  - The resume is not the cause: the none runs finished all 2000 updates on the first lease and were never resumed.
  - The cue is not the cause either: it is shown only at the first decision, and none has no memory, so its features
    at tick 239 do not depend on it.
  - Width is what is left. A random-init none net of the same width already scores 62.13 against the input's 63.0.
    The trained none scores 61.45. In E0, none's trained features made up the width cost; here they do not.
- **Exploratory, post hoc.** Measured against the same-width random-init net of each core: none −0.68, minGRU
  +0.60, GRU −0.60, minGRU − none +1.28. No core comes near the bar, but the run does not show that these cores
  cannot carry the cue.

## Belief value v2 (scripts/belief_v2.py, not yet sealed)

**Design change before the seal, made after pilot 1 and prompted by its failed placebo (2026-11-20).**
- Pilot 1 (job f7f04b27, 300 games on table E at offset 100500, belief_v2@91f75656b170) was exploratory.
  - Its placebo trained on one hard label per game, drawn from the trader's final-tick card-count posterior.
  - The placebo lost 12.8 $/game against cc [−19.4, −6.4], which fails the ±1 gate.
  - On validation games the placebo was calibrated (ECE ≤ 0.025 after its temperature, T = 6.31).
  - It was only 35–38% right, against cc's 41–50%. It carried less information than the cc belief it replaces, so it
    was not a fair null.
- The new placebo (ccs_T) trains on cc's own per-tick goal posterior as soft targets. Its temperature is fitted by
  soft cross-entropy on the same validation games as the other arms.
- Pilot 2 runs on fresh games 105000–105299. These are disjoint from pilot 1 (100500–100799), the temperature-fit
  range (104000–104499) and the sealed ranges (110000–112999 and 120000–122999).
- Every arm and cc are scored on the same games. The run also logs accuracy and ECE per tick on table E for every
  encoder.
- Gate: ccs_T − cc must lie within ±1 $/game. If it fails again, the test stops and is reported. There is no second
  redesign.
