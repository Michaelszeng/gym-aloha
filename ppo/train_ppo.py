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
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from action_wrappers import ClipActionWrapper, RateLimitActionWrapper
from gymnasium.wrappers import TimeLimit
from model.feature_extractors import AlohaStateExtractor
from rewards_wrappers import InsertionRewardShapingWrapper, SmoothnessPenaltyWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

# Local utilities
from training_utils import get_training_args

import gym_aloha  # Register custom environments
from ppo.logging_utils import InfoStatsCallback

# Local quiet video recorder
from ppo.quiet_video_recorder import QuietVecVideoRecorder as VecVideoRecorder


def train(args):
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/checkpoints", exist_ok=True)

    tb_root = f"{args.log_dir}/tensorboard"
    algo_name = "PPO"
    run_id = get_latest_run_id(tb_root, algo_name) + 1
    run_name = f"{algo_name}_{run_id}"

    # Create monitor folder
    monitor_folder = f"{args.log_dir}/monitor/{run_name}"
    os.makedirs(monitor_folder, exist_ok=True)

    # Video recording directory
    video_folder = f"{args.log_dir}/videos/{run_name}"
    os.makedirs(video_folder, exist_ok=True)

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

        env = gym.make(args.env_id, obs_type="state", render_mode="rgb_array")
        env = ClipActionWrapper(env)  # enforce joint limits
        env = RateLimitActionWrapper(env, max_delta=0.1)  # prevent huge accel/vels causing Mujoco crashes
        env = SmoothnessPenaltyWrapper(env, coeff=0.05)
        return env

    train_env = make_vec_env(
        env_id=make_aloha_env,  # Pass callable instead of string
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=partial(InsertionRewardShapingWrapper, gamma=1.0),
    )
    monitor_file = f"{monitor_folder}/train_monitor.csv"
    train_env = VecMonitor(train_env, filename=monitor_file, info_keywords=("sparse_r", "potential"))
    train_env = VecNormalize(  # Normalise obs, keep rewards untouched
        train_env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0
    )

    # Create separate evaluation environment (no reward shaping for fair benchmarking)
    eval_env = make_vec_env(
        env_id=make_aloha_env,  # Pass callable instead of string
        n_envs=1,
        seed=args.seed + 1000,
        vec_env_cls=SubprocVecEnv,
    )
    monitor_file = f"{monitor_folder}/eval_monitor.csv"
    eval_env = VecMonitor(eval_env, filename=monitor_file, info_keywords=("sparse_r"))
    # Add video recorder BEFORE normalization so the outermost env remains VecNormalize
    EPISODE_LEN = eval_env.get_attr("_max_episode_steps")[0]
    RECORD_EVERY_N_EVS = 50
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder,
        record_video_trigger=lambda step: step % (EPISODE_LEN * RECORD_EVERY_N_EVS) == 0,
        video_length=EPISODE_LEN,  # Record only first episode of eval
        name_prefix="eval",
    )
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.obs_rms = train_env.obs_rms  # Share running means/stds

    # Policy kwargs with custom state feature extractor
    policy_kwargs = dict(
        features_extractor_class=AlohaStateExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[512, 512, 256]),  # Separate networks for policy and value
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
        target_kl=args.target_kl,
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

    callback_list = CallbackList([checkpoint_callback, eval_callback, InfoStatsCallback()])

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

    return model


def main():
    args = get_training_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
