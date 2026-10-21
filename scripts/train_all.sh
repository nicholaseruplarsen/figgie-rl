#!/bin/sh
# Reproduces the CS224R training pipeline (v1 and v2), a longer v2 run, and a
# no-BC PPO control. ~1 h on a 12-core box.
set -e
cd "$(dirname "$0")/.."
mkdir -p checkpoints runs
[ -n "$SKIP_SCRATCH" ] || python3 -m figgie.rl.train ppo --out runs/ppo_scratch --threads 1 --procs 3 > runs/ppo_scratch.log 2>&1 &
[ -f checkpoints/bc_v1.pt ] || python3 -m figgie.rl.train bc --out checkpoints/bc_v1.pt --lift-weight 1  > runs/bc_v1.log 2>&1 &
[ -f checkpoints/bc_v2.pt ] || python3 -m figgie.rl.train bc --out checkpoints/bc_v2.pt --lift-weight 15 > runs/bc_v2.log 2>&1
while [ ! -f checkpoints/bc_v1.pt ]; do sleep 5; done
python3 -m figgie.rl.train ppo --init checkpoints/bc_v1.pt --out runs/ppo_v1 --threads 1 --procs 3 > runs/ppo_v1.log 2>&1 &
python3 -m figgie.rl.train ppo --init checkpoints/bc_v2.pt --out runs/ppo_v2 --lift-alpha 0.1 --threads 1 --procs 3 > runs/ppo_v2.log 2>&1 &
python3 -m figgie.rl.train ppo --init checkpoints/bc_v2.pt --out runs/ppo_v2_long --lift-alpha 0.1 --updates 2000 --seed 1 --threads 1 --procs 3 > runs/ppo_v2_long.log 2>&1 &
wait
