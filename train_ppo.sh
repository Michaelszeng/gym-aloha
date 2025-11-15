#!/bin/bash

# Resume from checkpoint (leave empty to start fresh)
RESUME_FROM="logs/ppo_insertion/PPO_2/checkpoints/ppo_aloha_10000000_steps.zip"  # e.g., "logs/ppo_insertion/PPO_2/checkpoints/ppo_aloha_10000000_steps.zip.zip"

# Create log directory
LOG_DIR="logs/ppo_insertion"

mkdir -p $LOG_DIR

pkill -f tensorboard
tensorboard --logdir logs/ppo_insertion/tensorboard --port 6006 --bind_all &

# Wandb configuration
USE_WANDB=true
WANDB_PROJECT="gym-aloha-insertion-ppo"
WANDB_ENTITY=""  # Optional: set to your wandb username or team

if [ -n "$RESUME_FROM" ]; then
    echo "Starting PPO training (RESUME from: $RESUME_FROM)..."
else
    echo "Starting PPO training (fresh start)..."
fi

python ppo/train_ppo.py \
    ${RESUME_FROM:+--resume-from $RESUME_FROM} \
    --total-timesteps 30000000 \
    --n-envs 32 \
    --n-steps 2048 \
    --batch-size 8192 \
    --n-epochs 10 \
    --learning-rate 3e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-range 0.2 \
    --ent-coef 0.01 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --target-kl 0.01 \
    --log-dir $LOG_DIR \
    --checkpoint-freq 100000 \
    --eval-freq 50000 \
    --device auto \
    ${USE_WANDB:+--use-wandb} \
    --wandb-project $WANDB_PROJECT \
    ${WANDB_ENTITY:+--wandb-entity $WANDB_ENTITY}

echo "Training complete! Check logs/ppo_insertion for results."
echo "To view tensorboard: tensorboard --logdir logs/ppo_insertion/tensorboard"