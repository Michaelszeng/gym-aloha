#!/bin/bash

# Create log directory
mkdir -p logs/ppo_insertion

# Start training with reasonable defaults
echo "Starting PPO training..."
python ppo/train_ppo.py \
    --total-timesteps 1000000 \
    --n-envs 4 \
    --learning-rate 3e-4 \
    --batch-size 64 \
    --log-dir logs/ppo_insertion \
    --checkpoint-freq 50000 \
    --eval-freq 25000 \
    --device auto

echo "Training complete! Check logs/ppo_insertion for results."
echo "To view tensorboard: tensorboard --logdir logs/ppo_insertion/tensorboard"

