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
