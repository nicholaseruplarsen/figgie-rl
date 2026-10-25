# Results

Everything here comes from the shared engine in `figgie/engine.py` (standard
rules; CS224R discretisation: 240 ticks, 257 actions, prices 1–30, every
trade wipes the book). Guessed details are listed in [DESIGN.md](DESIGN.md).
Raw numbers are in `results/`; trained checkpoints are in `checkpoints/`.

## 1. Replication fidelity per source

| Source | What was checked | Result |
|---|---|---|
| CS224R report (Yang 2026) | 5 baselines + PPO pipeline reimplemented from prose (no code released) | baseline ranking reproduced; PPO v2 matches or beats every reported number (§2) |
| mneha05/figment (MIT) | decisions of the port vs the original agent, inside the original simulator, 800 games | **144,000 / 144,000 identical** (Fundamental, BayesianMM, Momentum); goal posterior identical on every 10-card hand |
| revilo-25/Figgie-bot | same decision check, plus inference outputs | **64,000 / 64,000 identical** (PotMM, HandMM, FlowMM, ConstVal); posterior and pot-aware card values identical on every hand |
| 0xDub/figgie-auto | code review only | the original is real-time tokio with entropy-seeded RNG and 1–8 s sleeps, so it cannot be replayed; ported with 1 tick = 1 s and its sleep ranges |

**P&L under compat rules** (each original's own table, 800 games, 95% CI):

| Table | Agent | Original simulator | Port on shared engine (compat) |
|---|---|---|---|
| Figgie-bot E1 | PotMM | +5.8 ± 4.1 | +5.9 ± 4.0 |
| | HandMM | +1.0 ± 1.6 | −0.8 ± 1.7 |
| | FlowMM | +3.2 ± 4.3 | +5.2 ± 4.2 |
| | FixedSpread | −9.9 ± 2.0 | −10.2 ± 1.9 |
| figment (Fund, BMM, Mom, Random) | Fundamental | +38.6 ± 3.8 | +25.0 ± 4.1 |
| | BayesianMM | +47.3 ± 2.6 | +31.7 ± 2.9 |
| | Momentum | −30.0 ± 4.9 | −26.8 ± 4.6 |

Our run of the original Figgie-bot code gives exactly the numbers in its
README, and the port matches them within the CIs. The figment gap is a
protocol difference, not a logic difference: in figment one random player
acts per step, so a player gets about 60 turns, unevenly spread. On the
shared engine everyone acts every tick. The decision check above shows the
logic itself is identical.

## 2. CS224R reproduction

**Baseline tournament** (fixed seats; §5, Fig. 1 of the report):

| Agent | Reported (200 games) | Ours, 200 games | Ours, 2000 games |
|---|---|---|---|
| Bayesian | +18, 69% wins | +37.7 ± 8.9, 74% | +21.8 ± 3.1, 64% |
| FairValue | +6, 14% | −6.6 ± 5.5, 12% | +2.8 ± 2.1, 19% |
| MarketMaker | −8, 10% | −13.6 ± 5.0, 8% | −11.3 ± 1.6, 9% |
| Conservative | −16, 6% | −17.5 ± 4.4, 6% | −13.2 ± 1.6, 8.5% |

Same ranking; levels within a few dollars at 2000 games.

**Behaviour cloning from Bayesian** (500 episodes, 10 epochs): with lift
weight 15, lift top-1 accuracy rises from 0.2% to **79%** (reported: 1.7% →
54%); unweighted (v1) it reaches 17%. Loss 3.75 → 3.07 (reported 4.55 → 3.4).

**Table 1** (1000 games per matchup, seats rotated, deterministic argmax;
profit $/game, CI ≈ ±4–5; win = top profit; lifts = own aggressive buys/game):

| Matchup | Reported v1 | Ours v1 | Reported v2 | Ours v2 (300 upd) | Ours v2-long (2000 upd) | Ours no-BC PPO |
|---|---|---|---|---|---|---|
| vs 3×Random | +118, 89%, 0.7 | +135, 76%, 3.1 | +164, 98%, 4.5 | +278, 99%, 17.5 | +295, 99%, 11.5 | +178, 95%, 1.4 |
| vs FV + 2×R | +109, 19%, 0.4 | +112, 57%, 1.8 | +121, 33%, 1.6 | +244, 97%, 16.5 | +271, 97%, 13.5 | +140, 85%, 0.6 |
| vs Bayes + 2×R | +77, 4%, 0.3 | +107, 38%, 1.9 | +112, 10%, 1.0 | +233, 92%, 15.2 | +258, 94%, 12.4 | +144, 64%, 0.6 |
| vs FV + Bayes + R | +43, 13%, 0.0 | +66, 26%, 1.0 | +27, 15%, 3.2 | +136, 63%, 12.9 | +144, 68%, 11.5 | +63, 19%, 0.3 |
| **vs FV + Cons + MM** | +6, 33%, 0.0 | +2, 28%, 0.4 | **+8, 54%, 3.5** | **+21.5, 52%, 5.0** | +22.6, 52%, 5.3 | +0.2, 30%, 0.5 |
| vs FV + Bayes + MM | +2, 22%, 0.0 | −6, 19%, 0.6 | +2, 36%, 6.5 | +12.7, 53%, 7.1 | +14.6, 52%, 6.2 | −17, 20%, 0.9 |

What reproduces: v2 > v1 everywhere; lift-targeting turns a non-lifting
agent into one that lifts 5–7×/game against strong tables; v2 wins about
half its games in the report's headline heuristic-only matchup (52% vs 54%
reported); quote acceptance stays low (0.2–0.6% vs strong fields, report:
1–2%). Without BC, PPO learns to beat Random but loses to the all-strong table.

Where we differ: our agents earn 2× the reported profit against tables
with Random. Our Random samples uniformly from the legal mask, including bids
up to $30, so it loses about $200 per game and is much easier to farm. The report's Random is
probably milder; the report does not say. Training 2000 instead of 300 updates
adds little (+1–2 $/game).

## 3. Tournament: everyone vs everyone

`scripts/tournament.py`, 4000 tables. Each table draws 4 distinct agents at
random and plays one deal under all 4 seat rotations (duplicate format), so
16,000 games and about 2,200 games per agent. P&L CI is clustered by table.
Elo is figment's pairwise-duel Elo averaged over 20 game orders; TrueSkill
is μ−3σ with each game ranked as a 4-way free-for-all by profit.

### 3a. Without the uniform-random controls (`results/tournament_norandom.md`)

The three uniform-random agents (Random, fig.Random, fb.Random) are left out.
They lose so much that whoever sits next to them most often looks best.

| # | Agent | Source | Mean P&L ± 95% CI | Win rate | Elo | TrueSkill |
|---|---|---|---|---|---|---|
| 1 | fig.AdaptiveMM | figment | +22.2 ± 2.4 | 31.7% | 1565 | 25.6 |
| 2 | fig.BayesianMM | figment | +18.4 ± 2.3 | 28.5% | 1566 | 24.3 |
| 3 | Bayesian | CS224R | +16.1 ± 2.6 | 42.0% | 1551 | 25.6 |
| 4 | **PPO-v2** | CS224R (ours) | +15.5 ± 2.2 | **44.7%** | **1574** | 25.0 |
| 5 | PPO-v2-long | CS224R (ours) | +14.5 ± 1.9 | 43.8% | 1556 | 24.6 |
| 6 | fig.Fundamental | figment | +13.5 ± 1.9 | 28.4% | 1562 | 24.3 |
| 7 | fb.PotMM | Figgie-bot | +11.4 ± 2.3 | 43.4% | 1552 | 24.9 |
| 8 | fb.FlowMM | Figgie-bot | +7.7 ± 2.3 | 42.0% | 1550 | 24.2 |
| 9 | fa.TheHoarder | figgie-auto | +6.7 ± 2.1 | 38.4% | 1548 | 24.9 |
| 10 | fa.TiltInventory | figgie-auto | +5.1 ± 1.8 | 22.1% | 1495 | 24.5 |
| 11 | PPO-v1 | CS224R (ours) | +5.1 ± 1.8 | 24.2% | 1502 | 24.3 |
| 12 | fb.FixedSpread | Figgie-bot | +4.6 ± 1.8 | 17.1% | 1524 | 23.8 |
| 13 | FairValue | CS224R | +4.0 ± 1.5 | 13.7% | 1529 | 24.4 |
| 14 | **fb.ConstVal** (no-info control) | Figgie-bot | +2.6 ± 3.0 | 38.2% | 1510 | 23.9 |
| 15 | fb.HandMM | Figgie-bot | +1.9 ± 1.7 | 11.9% | 1539 | 24.6 |
| 16 | fa.Seller | figgie-auto | −0.3 ± 1.2 | 10.1% | 1525 | 24.1 |
| 17 | fa.PickOff | figgie-auto | −1.5 ± 1.8 | 10.1% | 1528 | 23.9 |
| 18 | fa.Spread | figgie-auto | −2.0 ± 1.7 | 18.0% | 1473 | 23.6 |
| 19 | PPO-noBC | CS224R (ours) | −3.5 ± 2.1 | 18.3% | 1469 | 23.0 |
| 20 | fa.PrayingMantis | figgie-auto | −4.6 ± 1.5 | 17.7% | 1474 | 22.5 |
| 21 | MarketMaker | CS224R | −5.1 ± 1.4 | 12.2% | 1476 | 23.7 |
| 22 | Conservative | CS224R | −6.0 ± 1.1 | 16.4% | 1470 | 23.5 |
| 23 | fa.Noisy | figgie-auto | −15.6 ± 1.9 | 16.9% | 1385 | 21.1 |
| 24 | fig.Momentum | figment | −23.3 ± 2.9 | 24.8% | 1427 | 22.8 |
| 25 | fb.Momentum | Figgie-bot | −39.7 ± 3.0 | 18.8% | 1304 | 20.5 |
| 26 | fb.PotValShuffled (scrambled-info control) | Figgie-bot | −48.4 ± 3.6 | 15.9% | 1345 | 19.6 |

### 3b. Full field including the random controls (`results/tournament.md`)

The top tier is the same (figment MMs, Bayesian, PPO-v2), and uniform Random
is last at −198 ± 6.5 $/game. The no-info ConstVal control jumps to +31.8,
level with Bayesian, because a constant $8.8 quote farms the random agents. This
is the same price-level effect the Figgie-bot README describes.

## 4. Takeaways

- No agent dominates. Four agents from three sources are statistically tied
  around +15 to +22 $/game: figment's Bayesian market makers, CS224R's
  Bayesian, and PPO-v2. Profit, win rate and rating each give a different #1.
  PPO-v2 has the highest win rate and Elo; figment's AdaptiveMM has the
  highest mean P&L.
- The no-information control is a useful test. Against a realistic field, the
  no-info ConstVal (+2.6) sits mid-table, close to FairValue and HandMM, so
  information-free price levels explain those agents. Against random fields
  it looks like a top agent. Scrambling the goal suit (PotValShuffled) is the
  worst agent at −48, so goal-suit information does matter for PotMM's
  asymmetric quoting.
- Lifting matters. The PPO variants without lift targeting (v1, no-BC) rank
  11th and 19th; the v2 variants rank 4th–5th.

## 5. Caveats

- This is not an equilibrium analysis. Exploitability is undefined for
  4-player general-sum games, and all numbers depend on the field. PPO was
  trained against the CS224R baselines and its own snapshots, not this field.
- Every gap in the CS224R report was filled with a guess (DESIGN.md), and one
  seed per configuration was trained. The PPO numbers need several seeds
  before they belong in a paper.
- Heuristic ports may send several orders per tick, as their originals did,
  while PPO and the CS224R baselines send one. That favours the multi-order
  market makers.
- Within a tick, players act in random sequential order (the report says
  "simultaneous" without giving a conflict rule).
- figgie-auto's ports are unverified against the original's behaviour.
- Figgie-bot's ports keep that repo's non-standard prior (goal suit 8 vs 10
  cards with probability ½ each).

## Reproduce

```
python scripts/verify_ports.py 800        # needs third_party/ clones (see README)
python scripts/evolve_figment.py
sh scripts/train_all.sh                   # BC + PPO v1/v2/long/no-BC, about 1 h on 12 cores
python scripts/eval_cs224r.py --ppo v1=checkpoints/ppo_v1.pt v2=checkpoints/ppo_v2.pt
python scripts/tournament.py --tables 4000 --ppo PPO-v1=checkpoints/ppo_v1.pt PPO-v2=checkpoints/ppo_v2.pt \
    PPO-v2-long=checkpoints/ppo_v2_long.pt PPO-noBC=checkpoints/ppo_scratch.pt
python scripts/tournament.py --tables 4000 --seed 777 --exclude Random fig.Random fb.Random --ppo ...
```
