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

## Causal test of the learning-failure hypothesis (to be sealed before job launch)

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
  - A lease that ends mid-run resumes from latest.pt, weights only (Adam, the PFSP pool and the aux head restart).
    Any such resume will be reported.

## Results

### H1, H-suff (#5119), job 1b61b9aa, evaluator controlled_probe_v2@ffc5a04797f0

On 10000 E0 games, the supervised sequence GRU reaches 73.54% and the last-input probe 63.72%, a gain of
**+9.82 points [8.97, 10.66]**. That is above the sealed 90% interval [−1, +6], so **H-suff is refuted**: the public
history holds about 10 points of goal-suit information beyond the last input. On the same games, trained minGRU
adds +1.18 [0.81, 1.58] over none (v2; placebo none +0.28, valid). The small memory gain is therefore a learning
failure, not sufficiency. The platform holds the result at `pending-control` until a placebo runs:
`scripts/hsuff_placebo.py` (same pipeline, labels permuted).

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
- Under the sealed rule this probe run is **invalid**: the none placebo is −1.55, outside ±1. The width bias the
  audit measured (−2 to −3 points for 128 extra dimensions) fits that.
- Neither reading changes the answer. No memory core kept a reward-relevant fact shown only at the first decision,
  so this PPO recipe (BC init, 2000 updates, γ 0.995, λ 0.95) does not learn memory over ~240 decisions.
