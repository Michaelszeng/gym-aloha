#!/bin/bash

# Create log directory
LOG_DIR="logs/ppo_insertion"

mkdir -p $LOG_DIR

pkill -f tensorboard
tensorboard --logdir logs/ppo_insertion/tensorboard --port 6006 --bind_all &

# Start training with reasonable defaults
echo "Starting PPO training..."
python ppo/train_ppo.py \
    --total-timesteps 10000000 \
    --n-envs 8 \
    --learning-rate 5e-6 \
    --batch-size 512 \
    --log-dir $LOG_DIR \
    --checkpoint-freq 50000 \
    --eval-freq 25000 \
    --device auto

echo "Training complete! Check logs/ppo_insertion for results."
echo "To view tensorboard: tensorboard --logdir logs/ppo_insertion/tensorboard"

