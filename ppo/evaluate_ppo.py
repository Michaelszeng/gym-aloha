"""
Evaluation Script for Trained PPO Policy

Usage:
    python evaluate_ppo.py --model-path logs/ppo_insertion/best_model/best_model.zip --n-episodes 10 --render
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO

import gym_aloha


def evaluate_policy(model_path: str, n_episodes: int = 10, render: bool = False, save_video: bool = False):
    """
    Evaluate a trained PPO policy.

    Args:
        model_path: Path to the saved model
        n_episodes: Number of episodes to evaluate
        render: Whether to render during evaluation
        save_video: Whether to save videos of episodes
    """

    # Load the trained model
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # Create environment
    env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels", render_mode="rgb_array")

    # Statistics
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    print(f"\nEvaluating for {n_episodes} episodes...")
    print("=" * 80)

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        frames = []

        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Render if requested
            if render or save_video:
                frame = env.render()
                frames.append(frame)

        # Track success
        is_success = info.get("is_success", False)
        if is_success:
            success_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(
            f"Episode {episode + 1:3d} | Reward: {episode_reward:6.2f} | Length: {episode_length:4d} | Success: {is_success}"
        )

        # Save video if requested
        if save_video and len(frames) > 0:
            video_path = f"eval_episode_{episode + 1}.mp4"
            imageio.mimsave(video_path, np.stack(frames), fps=25)
            print(f"  └─ Video saved to {video_path}")

    # Print summary statistics
    print("=" * 80)
    print("Evaluation Summary:")
    print(f"  Mean Reward:    {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean Length:    {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Success Rate:   {success_count}/{n_episodes} ({100 * success_count / n_episodes:.1f}%)")
    print(f"  Min Reward:     {np.min(episode_rewards):.2f}")
    print(f"  Max Reward:     {np.max(episode_rewards):.2f}")
    print("=" * 80)

    env.close()

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_rate": success_count / n_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model (.zip file)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes during evaluation",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save videos of evaluation episodes",
    )

    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        render=args.render,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
