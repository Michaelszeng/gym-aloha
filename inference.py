#!/usr/bin/env python3
"""
Loads a checkpoint and displays real-time execution through the Mujoco viewer.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import gym_aloha
from ppo.action_wrappers import ClipActionWrapper, RateLimitActionWrapper
from ppo.model import feature_extractors
from ppo.model.feature_extractors import AlohaImageExtractor, AlohaStateExtractor
from ppo.rewards_wrappers import InsertionRewardShapingWrapperV2

# Create module alias for backward compatibility with checkpoints that reference 'model'
# instead of 'ppo.model'. This allows loading checkpoints regardless of the training script's
# import structure.
sys.modules["model"] = sys.modules["ppo.model"]
sys.modules["model.feature_extractors"] = feature_extractors


def create_inference_env(env_id: str, use_reward_shaping: bool = True):
    """
    Create a single environment for inference with the same wrappers used during training.

    Args:
        env_id: Gymnasium environment ID
        use_reward_shaping: Whether to use dense reward shaping (default: True for visualization,
                          set to False to see sparse rewards for fair benchmarking)

    Returns:
        Wrapped environment
    """
    # ALOHA env only supports render_mode="rgb_array"
    env = gym.make(env_id, obs_type="state", render_mode="rgb_array")
    # Read max_episode_steps from the environment spec (set during registration)
    max_episode_steps = env.spec.max_episode_steps
    env = ClipActionWrapper(env)
    env = DeltaJointPositionWrapper(env)
    # env = RateLimitActionWrapper(env, max_delta=0.1)

    # Apply reward shaping wrapper if requested (matches training env)
    if use_reward_shaping:
        env = InsertionRewardShapingWrapperV2(env, gamma=0.95, max_episode_steps=max_episode_steps)

    return env


def load_model_and_normalizer(checkpoint_path: str, env, device: str = "auto"):
    """
    Load PPO model and VecNormalize statistics from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint .zip file
        env: The environment to set for the model (allows different num_envs than training)
        device: Device to load model on ("auto", "cpu", "cuda")

    Returns:
        Tuple of (model, vecnormalize_path) or (model, None) if no normalization file found
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading model from: {checkpoint_path}")

    # Provide custom objects to handle feature extractors saved with different module paths
    custom_objects = {
        "features_extractor_class": AlohaStateExtractor,
        "AlohaStateExtractor": AlohaStateExtractor,
        "AlohaImageExtractor": AlohaImageExtractor,
    }

    # Pass env when loading to allow different number of environments than training
    model = PPO.load(checkpoint_path, env=env, device=device, custom_objects=custom_objects)
    vecnormalize_path = checkpoint_path.with_name(checkpoint_path.stem + "_vecnormalize.pkl")
    if vecnormalize_path.exists():
        print(f"Found VecNormalize stats: {vecnormalize_path}")
        return model, str(vecnormalize_path)
    raise FileNotFoundError(f"VecNormalize stats file not found at {vecnormalize_path}")


def run_inference(
    checkpoint_path: str,
    env_id: str = "gym_aloha/AlohaInsertion-v0",
    n_episodes: int = 10,
    deterministic: bool = True,
    use_viewer: bool = True,
    device: str = "auto",
    seed: int = None,
    use_reward_shaping: bool = True,
):
    """
    Run inference with a trained model and display in real-time using MuJoCo viewer.

    Args:
        checkpoint_path: Path to model checkpoint (.zip file)
        env_id: Gymnasium environment ID
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        use_viewer: Whether to use MuJoCo viewer for visualization
        device: Device to use for inference
        seed: Random seed (optional)
        use_reward_shaping: Whether to use dense reward shaping (shows shaped rewards during inference)
    """
    # Create environment first
    print(f"Creating environment: {env_id}")

    def make_env():
        return create_inference_env(env_id, use_reward_shaping=use_reward_shaping)

    env = DummyVecEnv([make_env])

    # Load model with environment (allows different num_envs than training)
    model, vecnormalize_path = load_model_and_normalizer(checkpoint_path, env, device)

    # Apply VecNormalize if stats are available
    if vecnormalize_path:
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False  # Disable updates to running statistics
        env.norm_reward = False  # Don't normalize rewards during inference
        print("VecNormalize stats loaded successfully")
        # Update the model's environment to use the normalized version
        model.set_env(env)

    print(f"\nRunning inference for {n_episodes} episodes...")
    print(f"Deterministic actions: {deterministic}")
    print(f"Reward shaping: {'Enabled (dense rewards)' if use_reward_shaping else 'Disabled (sparse rewards)'}")
    print(f"Visualization: {'MuJoCo Viewer' if use_viewer else 'None'}")
    print("=" * 80)

    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Setup MuJoCo viewer if requested
    viewer = None
    mj_model = None
    mj_data = None

    if use_viewer:
        # Get the underlying ALOHA environment
        aloha_env = env.envs[0].unwrapped

        # Determine the correct XML file based on env_id
        if "Insertion" in env_id:
            xml_path = "gym_aloha/assets/bimanual_viperx_insertion.xml"
        elif "TransferCube" in env_id:
            xml_path = "gym_aloha/assets/bimanual_viperx_transfer_cube.xml"
        else:
            print(f"Warning: Unknown environment {env_id}, viewer may not work correctly")
            xml_path = "gym_aloha/assets/bimanual_viperx_insertion.xml"

        # Create MuJoCo model and viewer
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        print("Launched MuJoCo viewer")

    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    try:
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            # Sync viewer with environment state immediately after reset
            if use_viewer and viewer is not None and viewer.is_running():
                aloha_env = env.envs[0].unwrapped
                mj_data.qpos[:] = aloha_env._env.physics.data.qpos
                mj_data.qvel[:] = aloha_env._env.physics.data.qvel
                mj_data.ctrl[:] = aloha_env._env.physics.data.ctrl
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

            print(f"\nEpisode {episode + 1}/{n_episodes}")

            while not done:
                # Get action from policy
                action, _states = model.predict(obs, deterministic=deterministic)

                # Step environment
                obs, reward, done, info = env.step(action)

                episode_reward += reward[0]
                episode_length += 1

                # Update viewer if active
                if use_viewer and viewer is not None and viewer.is_running():
                    # Sync state from dm_control to MuJoCo viewer
                    aloha_env = env.envs[0].unwrapped
                    mj_data.qpos[:] = aloha_env._env.physics.data.qpos
                    mj_data.qvel[:] = aloha_env._env.physics.data.qvel
                    mj_data.ctrl[:] = aloha_env._env.physics.data.ctrl
                    mujoco.mj_forward(mj_model, mj_data)
                    viewer.sync()
                    time.sleep(0.01)  # Small delay for smooth visualization

                # Check if episode is done
                if done[0]:
                    is_success = info[0].get("is_success", False)
                    episode_successes.append(is_success)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)

                    print(f"  Reward: {episode_reward:.2f}")
                    print(f"  Length: {episode_length} steps")
                    print(f"  Success: {'✓' if is_success else '✗'}")
                    break

                # Debugging
                # left_gripper_vel = (
                #     env.envs[0].unwrapped._env.physics.named.data.cvel["vx300s_left/gripper_link"][3:6].copy()
                # )
                # right_gripper_vel = (
                #     env.envs[0].unwrapped._env.physics.named.data.cvel["vx300s_right/gripper_link"][3:6].copy()
                # )
                # print(f"Left gripper velocity: {left_gripper_vel}")
                # print(f"Right gripper velocity: {right_gripper_vel}")

            # Brief pause between episodes
            if use_viewer and viewer is not None and viewer.is_running():
                time.sleep(1.0)

    finally:
        # Close viewer
        if viewer is not None:
            viewer.close()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Episodes completed: {len(episode_rewards)}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success rate: {np.mean(episode_successes):.2%} ({sum(episode_successes)}/{len(episode_successes)})")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print("=" * 80)

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained PPO model on ALOHA Insertion task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to model checkpoint (.zip file). "
            "Example: logs/ppo_insertion/PPO_2/checkpoints/ppo_aloha_1000000_steps.zip"
        ),
    )

    # Optional arguments
    parser.add_argument(
        "--env-id",
        type=str,
        default="gym_aloha/AlohaInsertion-v0",
        help="Gymnasium environment ID",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy actions (recommended for evaluation)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions (disables deterministic)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Disable MuJoCo viewer visualization (useful for headless mode)",
    )
    parser.add_argument(
        "--no-reward-shaping",
        action="store_true",
        help="Disable dense reward shaping (use sparse rewards for fair benchmarking)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Handle deterministic flag
    deterministic = args.deterministic and not args.stochastic

    # Run inference
    try:
        run_inference(
            checkpoint_path=args.checkpoint,
            env_id=args.env_id,
            n_episodes=args.n_episodes,
            deterministic=deterministic,
            use_viewer=not args.no_viewer,
            device=args.device,
            seed=args.seed,
            use_reward_shaping=not args.no_reward_shaping,
        )
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
