import os
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from action_wrappers import ClipActionWrapper, RateLimitActionWrapper
from model.feature_extractors import AlohaStateExtractor
from rewards_wrappers import InsertionRewardShapingWrapperV2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from training_utils import get_training_args

import gym_aloha
import wandb
from ppo.logging_utils import InfoStatsCallback, WandbCallback
from ppo.quiet_video_recorder import QuietVecVideoRecorder as VecVideoRecorder


def print_training_config(args):
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
    print(f"Wandb: {'Enabled' if args.use_wandb else 'Disabled'}")
    print("=" * 80)


def train(args):
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Determine run ID by checking existing run folders
    algo_name = "PPO"
    existing_runs = [
        d
        for d in os.listdir(args.log_dir)
        if d.startswith(f"{algo_name}_") and os.path.isdir(os.path.join(args.log_dir, d))
    ]
    if existing_runs:
        run_ids = [int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()]
        run_id = max(run_ids) + 1 if run_ids else 1
    else:
        run_id = 1
    run_name = f"{algo_name}_{run_id}"

    # Create run-specific directory structure
    run_dir = f"{args.log_dir}/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Create subdirectories for this run
    monitor_folder = f"{run_dir}/monitor"
    video_folder = f"{run_dir}/videos"
    os.makedirs(monitor_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,  # Sync tensorboard logs
            monitor_gym=True,
        )

    print_training_config(args)

    # First, create a temporary environment to read max_episode_steps from the registered environment
    temp_env = gym.make(args.env_id, obs_type="state", render_mode="rgb_array")
    MAX_EPISODE_STEPS = temp_env.spec.max_episode_steps
    temp_env.close()
    del temp_env

    # Create vectorized training environments with reward shaping
    # Use a callable to ensure gym_aloha is imported in each subprocess
    def make_aloha_env():
        import gym_aloha  # noqa: F811 - Import needed in subprocess

        env = gym.make(args.env_id, obs_type="state", render_mode="rgb_array")
        # NOTE: not sure if this should be used or not (usually not, but supposedly "prevents exploding")
        env = ClipActionWrapper(env)  # enforce joint limits
        env = RateLimitActionWrapper(env, max_delta=0.5)  # prevent huge accel/vels causing Mujoco crashes
        return env

    train_env = make_vec_env(
        env_id=make_aloha_env,  # Pass callable instead of string
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=partial(InsertionRewardShapingWrapperV2, gamma=0.99, max_episode_steps=MAX_EPISODE_STEPS),
        monitor_dir=None,  # Disable automatic Monitor wrapping to avoid double-wrapping
    )
    train_monitor_file = f"{monitor_folder}/train_monitor.csv"
    train_env = VecMonitor(train_env, filename=train_monitor_file, info_keywords=("dense_r", "is_success"))
    train_env = VecNormalize(  # Normalise obs, keep rewards untouched
        train_env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0
    )

    # Create separate evaluation environment (no reward shaping for fair benchmarking)
    eval_env = make_vec_env(
        env_id=make_aloha_env,  # Pass callable instead of string
        n_envs=1,
        seed=args.seed + 1000,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=None,  # Disable automatic Monitor wrapping to avoid double-wrapping
    )
    eval_monitor_file = f"{monitor_folder}/eval_monitor.csv"
    eval_env = VecMonitor(eval_env, filename=eval_monitor_file)
    # Add video recorder BEFORE normalization so the outermost env remains VecNormalize
    RECORD_EVERY_N_EVS = 50
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder,
        record_video_trigger=lambda step: step % (MAX_EPISODE_STEPS * RECORD_EVERY_N_EVS) == 0,
        video_length=MAX_EPISODE_STEPS,  # Record only first episode of eval
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

    # Create tensorboard directory with the exact structure we want
    tensorboard_folder = f"{run_dir}/tensorboard"
    os.makedirs(tensorboard_folder, exist_ok=True)

    # Create or load PPO agent
    if args.resume_from:
        print(f"\nResuming training from checkpoint: {args.resume_from}")

        # Load the model
        model = PPO.load(
            args.resume_from,
            env=train_env,
            device=args.device,
            custom_objects={
                "learning_rate": args.learning_rate,
                "clip_range": args.clip_range,
            },
        )

        # Load VecNormalize stats if available
        vecnormalize_path = args.resume_from.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vecnormalize_path):
            print(f"Loading VecNormalize stats from {vecnormalize_path}")
            train_env = VecNormalize.load(vecnormalize_path, train_env)
            eval_env.obs_rms = train_env.obs_rms  # Update eval env with loaded stats
            eval_env.ret_rms = train_env.ret_rms
        else:
            raise FileNotFoundError(f"VecNormalize stats file not found at {vecnormalize_path}")

        print("Successfully loaded checkpoint!")
    else:
        # Create new PPO agent
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
            tensorboard_log=None,  # Don't pass tensorboard_log to avoid SB3's automatic subdirectory creation
            device=args.device,
            verbose=1,
        )

        # Manually configure the logger to write to our exact directory structure
        # This bypasses SB3's automatic PPO_X subdirectory creation
        new_logger = configure(tensorboard_folder, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

    # Setup callbacks
    checkpoint_folder = f"{run_dir}/checkpoints"
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,  # Adjust for parallel envs
        save_path=checkpoint_folder,
        name_prefix="ppo_aloha",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{run_dir}/best_model",
        log_path=f"{run_dir}/eval",
        eval_freq=args.eval_freq // args.n_envs,  # Adjust for parallel envs
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks = [checkpoint_callback, eval_callback, InfoStatsCallback()]
    if args.use_wandb:
        callbacks.append(WandbCallback())
    callback_list = CallbackList(callbacks)

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback_list,
        progress_bar=True,
    )

    # Save final model
    final_model_folder = f"{run_dir}/final_model"
    os.makedirs(final_model_folder, exist_ok=True)
    final_model_path = f"{final_model_folder}/final_model"
    model.save(final_model_path)
    train_env.save(f"{final_model_path}_vecnormalize.pkl")
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    print(f"VecNormalize stats saved to {final_model_path}_vecnormalize.pkl")

    # Cleanup
    train_env.close()
    eval_env.close()

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

    return model


def main():
    args = get_training_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
