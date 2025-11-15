import argparse
import os
import subprocess
import time


def get_training_args():
    parser = argparse.ArgumentParser(description="Train PPO on ALOHA Insertion Task")

    # Environment args
    parser.add_argument(
        "--env-id",
        type=str,
        default="gym_aloha/AlohaInsertion-v0",
        help="Gym environment ID",
    )

    # Training args
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps over all rollouts over the whole run",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=64,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="Number of steps during each rollout phase",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Minibatch size during optimization",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of optimization epochs per rollout phase",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="Factor for trade-off of bias vs variance for GAE",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.1,
        help="Clipping parameter for PPO; prevents too large of a policy update/change in one step (trust region)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for the loss calculation; encourages exploration",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=1.0,
        help="Value function coefficient/weight for the loss calculation",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="If the KL divergence between the new and old policy is greater than this, training is stopped",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="The maximum value for the gradient clipping",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/ppo_insertion",
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from "
        "(e.g., logs/ppo_insertion/checkpoints/PPO_2/ppo_aloha_9900000_steps.zip)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=250_000,
        help="Checkpoint save frequency (in timesteps)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Evaluation frequency (in timesteps)",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )

    # Wandb integration
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gym-aloha-insertion-ppo",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity (username or team name)",
    )

    args = parser.parse_args()
    return args
