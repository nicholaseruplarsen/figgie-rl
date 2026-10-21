# Design notes and every guess

## Engine (`figgie/engine.py`)

Standard rules: 40 cards in 12 equally likely decks ({12,10,10,8} over suits),
goal = same-colour partner of the 12-suit (so 8 cards w.p. 1/3, 10 w.p. 2/3),
$350 start, ante 200/n, $10 per goal card, remainder (100 or 120) split among
the players holding the most goal cards. 4 or 5 players.

Trading follows the CS224R discretisation (§4 of the report):

| Item | Report | Here |
|---|---|---|
| Clock | 240 "simultaneous-action ticks" | 240 ticks; **GUESS:** within a tick, players act in a fresh random order and each sees the earlier players' moves. True simultaneity needs a conflict rule the report does not give. |
| Actions | 257: no_op, place_bid/ask(s, p∈1..30), cancel_bid/ask(s), lift(s), hit(s) | same, `figgie/actions.py` |
| Quotes | one bid + one ask per player per suit | same; a new quote replaces the old |
| Book wipe | every trade cancels every standing quote | same (`cancel_on_trade=True`) |
| Crossing place_* | not stated | **GUESS:** executes at the resting price (same as an explicit lift/hit); crossing your own quote is rejected |
| Cash/shorting | not stated | bids need cash ≥ price, asks need a card; no shorting |
| Tie-break between equal quotes | not stated | time priority |

Compat switches (`Config.cancel_on_trade`, `fill_removes_quote`, `ticks`) let
the ports run under rules close to their original simulators for verification.
Heuristic ports may submit several primitives per turn when their original did
(Figgie-bot re-quotes every suit each tick; figgie-auto may send several
orders per wake-up). The PPO agent and CS224R baselines send one.

Speed: pure Python, ~3 ms per 240-tick game with heuristic bots, ~3 s per PPO
update of 16k learner steps on 2 Mac cores, so no C/PufferLib port was needed;
the bottleneck is Python opponents, not the network.

## CS224R baselines (`figgie/agents/cs224r.py`)

| Agent | Guesses |
|---|---|
| Random | none (uniform over the legal mask) |
| FairValue | when nothing to take: coin flip between a bid on the best suit and an ask on the lowest-value suit **we hold**; price round(v∓edge) clipped to [1,30] |
| Conservative | as FairValue (gate 0.4, edge 4) |
| MarketMaker | which empty slot to fill: uniformly random each tick; price floor(v−3)/ceil(v+3) clipped to ≥1 |
| Bayesian | buy price ~ U{1..floor(e_b(s, j_n))}; sell price ~ U{ceil(v)..ceil(v)+10} with v = e_b(s, j_n−1); card-counting lower bound: buyer +1, seller max(L,1)−1; majority threshold x = goal_size//2 + 1 (5 or 6), r = 2 |

## CS224R PPO agent (`figgie/rl/`)

Taken from the report: GRU-128 after a 2-layer encoder, masked categorical over
257 actions, 69-dim observation, TBPTT chunk 32 with 4-chunk gradient
accumulation, potential shaping β=0.5 with φ = cash + Σ_j Σ_{k<hand_j} e_b(j,k)
and φ(terminal)=0, PFSP (pool 16, snapshot every 20 updates after 50, weight
(1−w)^2, p_sp=0.3), BC from Bayesian (500 episodes, 10 epochs, lift weight 15
in v2), lift bonus 0.1·max(0, e_b−p) in v2, 32 envs × 512 steps, 300 updates,
deterministic argmax at evaluation.

GUESSES:
- 69-dim layout (see `features.py` docstring), including the
  lift-opportunity feature = (e_b − best external ask)+/10.
- lr 2.5e-4 (Adam), γ 0.995, GAE λ 0.95, clip 0.2, entropy 0.01, value coef
  0.5, 4 PPO epochs, grad-norm 0.5, reward scale 0.1, BC lr 1e-3.
- φ is shifted by −350 so it is near zero at the start (constant shifts keep
  the shaping policy-invariant up to the (γ−1)c term).
- "Lift" for the bonus and for the lifts/game statistic = any learner buy that
  executes on its own action (explicit lift or crossing bid).
- PFSP win = learner out-earned that pool member in the episode; EMA rate 0.1;
  non-pool seats draw uniformly from the five baselines (the report says
  "heuristic baseline pool").
- BC: "episode by episode" read as batch size 1 (an earlier run with batch 16
  only reached 4.5% lift top-1, versus the reported 54%).
- v1 = BC without lift weighting + PPO without lift bonus; v2 = both. The
  report's v2 also adds the lift-opportunity feature; we keep the 69-dim
  observation identical for both so the checkpoints share one network shape.

## Ports

| Source | License | How ported | Verified by |
|---|---|---|---|
| mneha05/figment | MIT | transcribed decision code + adapter (`agents/figment.py`) | identical decisions on every call inside the original simulator; identical posteriors on every 10-card hand |
| revilo-25/Figgie-bot | none | reimplemented (`agents/figgiebot.py`) | identical decisions inside the original sim; posterior and pot-aware card values identical on every hand; P&L in compat mode |
| 0xDub/figgie-auto | none | reimplemented (`agents/figgieauto.py`), 1 tick = 1 s, original sleep ranges | code review only: the original is a real-time tokio program with entropy-seeded RNG, so neither its decisions nor its P&L can be replayed |

Source quirks kept in the ports:
- Figgie-bot's deal draws the goal-suit size 8 vs 10 with probability ½ each
  (16 compositions), and its posterior uses that prior. The ports keep it;
  `prior="standard"` switches to the correct 12-deck prior.
- figgie-auto's deal picks the 8-card suit with probabilities ½, ¼, ¼ instead
  of ⅓ each. Its bots only use the "suit I hold most" heuristic, so nothing to keep.
- figment's settlement and deck are standard; its engine lets one random
  player act per step (so 60 turns each in 240 steps), whereas here everyone acts every tick.
- figment AdaptiveMM uses the parameters from running figment's own
  `evolve()` (seed 0, defaults) in its own simulator: `results/figment_evolved_params.json`.
