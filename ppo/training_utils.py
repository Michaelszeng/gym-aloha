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
        help="Total training timesteps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
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
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps to run for each environment per update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of epochs when optimizing the surrogate loss",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
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
        default=0.2,
        help="Clipping parameter for PPO",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for the loss calculation",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value function coefficient for the loss calculation",
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
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Checkpoint save frequency (in timesteps)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Evaluation frequency (in timesteps)",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="Port for TensorBoard server",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )

    args = parser.parse_args()
    return args


def launch_tensorboard(log_dir, port=6006):
    """Launch tensorboard in the background and return the process."""
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

    try:
        # Try to launch tensorboard
        process = subprocess.Popen(
            ["tensorboard", "--logdir", tensorboard_dir, "--port", str(port), "--bind_all"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for tensorboard to start
        time.sleep(1)

        # Check if it's still running
        if process.poll() is None:
            print("=" * 80)
            print("🚀 TensorBoard started successfully!")
            print(f"📊 View at: http://localhost:{port}")
            print(f"📁 Logging to: {tensorboard_dir}")
            print("=" * 80)
            return process
        else:
            print(f"⚠️  TensorBoard failed to start (may already be running on port {port})")
            return None
    except Exception as e:
        print(f"⚠️  Could not start TensorBoard: {e}")
        return None
