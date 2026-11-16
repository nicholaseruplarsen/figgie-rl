# Figgie memory agents: prediction written before the full table

Written by Geoff on 2026-11-14 before the full table. The "~11:15 UTC" time first written here was a guess and is wrong: the file was written some time between 11:12 and 14:52 CET, and the only verified time is its commit, b64e174 at 15:16 CET. By then minGRU s0, GRU s1 and none s1 had finished, but Geoff had not seen their numbers. GRU s0 and none s0 were already seen (47.1% and 53.3% probe), so this is not blind to them.

Caveat: the GRU and no-memory predictions were made after seeing their seed-0 results, so they are not blind. Only the transformer and minGRU predictions are clean. The midterm must say this where it scores the table.

## Main metric
Goal-suit linear-probe accuracy on the agent's last hidden state at tick 239, 3000 probe games, mean over the seeds available (2 for the midterm). Hand-only baseline is 38.9%. The probe on the current input (no hidden state) is reported alongside as the ceiling the observation alone gives.

## Predicted ranking and intervals (2-seed mean)
1. transformer (128-tick window): 48–58%
2. none (no memory): 50–56%
3. GRU: 44–52%
4. minGRU: 43–52%

Expected order: transformer ≥ none > GRU ≈ minGRU. Most likely outcome: no memory type clearly beats none.

## What counts as "memory helps"
A memory type helps if both hold:
- Its 2-seed mean probe accuracy beats none's by at least 3 percentage points.
- Its worse seed beats none's better seed.

Secondary (profit): its 2-seed mean profit against FV + Bayes + MM beats none's by more than the larger of the two runs' 95% CIs.

Anything else, the midterm reports as "no evidence memory helps at this training length".

## Rules
- Every comparison not listed here, chosen after seeing the numbers, is labelled exploratory.
- The midterm scores the table against this file.
