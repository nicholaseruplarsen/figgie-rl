# Post-mortem: prediction #699, transformer goal-suit probe (MISS)

Prediction (docs/PREREGISTRATION.md, b64e174): the transformer core's mean probe accuracy at tick 239 over seeds 0 and 1 would be 48–58%.
Observed: 44.8% (45.6 and 44.0), auto-scored 2026-11-16 09:00Z from `runs/memory/prereg/transformer.json`. It is a miss on the low side.

The platform opens no post-mortem for it. #699 is a legacy prereg, a plain range with no stated coverage level, so it resolves as "miss" with level "unknown" (Bret, 16 Nov). This note is the explanation of record. The test below is pre-registered in docs/research_plan.tex.

## Scorer audit
- **Canaries.** A shuffled-label placebo of the same probe on minGRU s0 and s1 (3000 games, tick 239) gives 24.6%, against 25% chance, while the true labels give 58.0% (`runs/memory-diag/placebo_mingru.json`, #705). So the probe does not leak labels. No canary ran on the transformer itself.
- **Stale input share: 0.** Each seed was probed on 3000 games freshly recorded by its own final checkpoint. The bound file lists the two `probe.json` sources, and neither reuses another run's games.
- **Decision clock.** The hidden state is read at tick 239, the agent's last step, and is computed only from observations up to then. The goal suit becomes known only at settlement, after tick 239. The prediction was committed at 15:16 CET on 14 Nov, before either transformer seed's probe existed.
- **Synthetic signal: not run.** No known-answer test has shown that this probe recovers a fact a transformer is known to store. The recall check (a cue at tick 0, probed at tick 239) is planned (E0 in docs/research_plan.tex) but has not run for any core.

## The assumption that failed
We assumed that a linear probe of the last hidden state measures learned goal-suit belief, so that a memory core would hold at least what the no-memory core holds. The common-trajectory probes (`runs/memory-diag/transformer_s1/common_probe.json`, the none_s0 table, 3000 games) refute that. Untrained random-init nets of every core reach 58.9–60.4%, the same as the input itself (60.1%), while every trained net is lower: none 53.2/54.6, minGRU 53.1/55.3, GRU 44.3/44.3, transformer 45.2/45.2. The probe mostly reads input features that a random projection keeps, and training removes some of them.

## Rival explanations
1. **Information loss, not belief.** Training compresses the state toward what the policy uses, and discards input information the probe could read. The transformer and GRU compress most. GRU sits at 44% with a normal policy (5–6 lifts per game), which supports this rival over rival 2.
2. **Degenerate policy.** The transformer plays about 1 lift per game, against 5–6 for the other cores, so its state may carry little game state. This explains the transformer, but not the GRU.
3. **Window or training length.** A 128-tick window and 2000 updates may be too short for a transformer to learn to integrate the order flow. Rival 3 predicts that its probe gain rises with training length.

## Test
The controlled probe in docs/research_plan.tex, pre-registered there before any run. On one common set of trajectories, with games sorted by id before the 5-fold split:
- Δ_rand = accuracy(trained hidden state) − accuracy(same-architecture random-init hidden state).
- Δ_cond = accuracy(probe on [input, hidden state]) − accuracy(probe on input alone), which is what the state adds beyond the current input. The none core is the placebo: its Δ_cond must be 0 ± 1 point.

Rival 1 predicts Δ_rand < 0 for every trained core and Δ_cond ≈ 0 for every core. Rival 2 predicts that only the degenerate transformer stands apart from the other cores. Rival 3 predicts that Δ_cond grows across the checkpoints of longer runs.
