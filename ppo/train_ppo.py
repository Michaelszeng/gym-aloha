"""
PPO Training Script for ALOHA Tasks with State Observations

This script trains a PPO policy using state observations (agent positions + object poses)
instead of images. This is more sample-efficient than vision-based learning.

The state observation includes:
- agent_pos: joint positions of both robot arms (14 dimensions)
- env_state: object poses (7 dims for cube, 14 dims for peg+socket)

Usage:
    python train_ppo.py --total-timesteps 1000000 --log-dir logs/ppo_insertion
"""

import os

os.environ["MUJOCO_GL"] = "egl"  # GPU rendering
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from model.feature_extractors import AlohaStateExtractor
from rewards import InsertionRewardShapingWrapper
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from training_utils import get_training_args, launch_tensorboard

import gym_aloha  # Register custom environments


def train(args):
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/checkpoints", exist_ok=True)

    tensorboard_process = launch_tensorboard(args.log_dir, port=args.tensorboard_port)

    print("=" * 80)
    print("PPO Training Configuration")
    print("=" * 80)
    print(f"Environment: {args.env_id}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Number of parallel environments: {args.n_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Log directory: {args.log_dir}")
    print("=" * 80)

    # Create vectorized training environments with reward shaping
    # Use a callable to ensure gym_aloha is imported in each subprocess
    def make_aloha_env():
        import gym_aloha  # noqa: F811 - Import needed in subprocess

        return gym.make(args.env_id, obs_type="state", render_mode="rgb_array")

    train_env = make_vec_env(
        env_id=make_aloha_env,  # Pass callable instead of string
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=InsertionRewardShapingWrapper,
    )
    train_env = VecMonitor(train_env)

    # Create separate evaluation environment (no reward shaping for fair benchmarking)
    eval_env = make_vec_env(
        env_id=make_aloha_env,  # Pass callable instead of string
        n_envs=1,
        seed=args.seed + 1000,
        vec_env_cls=SubprocVecEnv,
    )
    eval_env = VecMonitor(eval_env)

    # Policy kwargs with custom state feature extractor
    policy_kwargs = dict(
        features_extractor_class=AlohaStateExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Separate networks for policy and value
        activation_fn=nn.ReLU,
    )

    # Create PPO agent
    model = PPO(
        policy="MultiInputPolicy",  # For Dict observation spaces
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"{args.log_dir}/tensorboard",
        device=args.device,
        verbose=1,
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,  # Adjust for parallel envs
        save_path=f"{args.log_dir}/checkpoints",
        name_prefix="ppo_aloha",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.log_dir}/best_model",
        log_path=f"{args.log_dir}/eval",
        eval_freq=args.eval_freq // args.n_envs,  # Adjust for parallel envs
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # Train the agent
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback_list,
        progress_bar=True,
    )

    # Save final model
    final_model_path = f"{args.log_dir}/final_model"
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")

    # Cleanup
    train_env.close()
    eval_env.close()

    # Keep tensorboard running
    if tensorboard_process is not None and tensorboard_process.poll() is None:
        print(f"\n💡 TensorBoard is still running at http://localhost:{args.tensorboard_port}")
        print("   To stop it, find the process with: ps aux | grep tensorboard")
        print("   Or kill it with: pkill -f tensorboard")

    return model


def main():
    args = get_training_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
