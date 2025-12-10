#!/bin/bash
#SBATCH --job-name=gym-aloha-ppo
#SBATCH --account=locomotion
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --time=24:00:00                  # max runtime
#SBATCH --gpus=1                         # number of GPUs
#SBATCH --cpus-per-task=16               # tune for vectorized envs
#SBATCH --mem=256G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --requeue                        # optional: restart if preempted

# sbatch train_ppo_csail.sbatch

# For interactive run:
# srun --account=locomotion --partition=csail-shared-h200 --qos=shared-if-available --time=5:00 --cpus-per-task=16 --gpus=1 --mem=64G export MUJOCO_GL=egl && python ppo/train_ppo.py --total-timesteps 100000000 --n-envs 256 --n-steps 8192 --batch-size 262144 --n-epochs 5 --learning-rate 3e-4 --gamma 0.99 --gae-lambda 0.95 --clip-range 0.2 --ent-coef 0.02 --vf-coef 0.5 --max-grad-norm 0.5 --target-kl 0.01 --log-dir "logs/ppo_insertion" --checkpoint-freq 100000 --eval-freq 50000 --device auto --use-wandb --wandb-project "gym-aloha-insertion-ppo" --wandb-entity ""

# Load cluster-wide environment
source /etc/profile

source env/bin/activate

# Limit threading to avoid oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Suppress TF/PyTorch warnings, XLA, headless rendering
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore::FutureWarning"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export MUJOCO_GL=egl

# Resume from checkpoint (leave empty to start fresh)
RESUME_FROM=""  # e.g. "logs/ppo_insertion/PPO_2/checkpoints/ppo_aloha_10000000_steps.zip"

# Create log directory (on NFS)
LOG_DIR="logs/ppo_insertion"
mkdir -p "$LOG_DIR"

# Wandb configuration (you may want online mode on CSAIL; this is your choice)
USE_WANDB=true
WANDB_PROJECT="gym-aloha-insertion-ppo"
WANDB_ENTITY=""

if [ -n "$RESUME_FROM" ]; then
    echo "Starting PPO training (RESUME from: $RESUME_FROM)..."
else
    echo "Starting PPO training (fresh start)..."
fi

python ppo/train_ppo.py \
    ${RESUME_FROM:+--resume-from "$RESUME_FROM"} \
    --total-timesteps 100000000 \
    --n-envs 256 \
    --n-steps 8192 \
    --batch-size 262144 \
    --n-epochs 5 \
    --learning-rate 3e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-range 0.2 \
    --ent-coef 0.02 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --target-kl 0.01 \
    --log-dir "$LOG_DIR" \
    --checkpoint-freq 100000 \
    --eval-freq 50000 \
    --device auto \
    ${USE_WANDB:+--use-wandb} \
    --wandb-project "$WANDB_PROJECT" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"}

echo "Training complete! Check $LOG_DIR for results."
