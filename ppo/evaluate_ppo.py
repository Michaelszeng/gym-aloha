"""
Evaluation Script for Trained PPO Policy

Usage:
    python evaluate_ppo.py --model-path logs/ppo_insertion/best_model/best_model.zip --n-episodes 10 --render
"""

import argparse
import os
from typing import List

# Configure headless rendering *before* importing Mujoco / gym
os.environ["MUJOCO_GL"] = "egl"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

import gym_aloha
from ppo.action_wrappers import ClipActionWrapper, RateLimitActionWrapper
from ppo.rewards_wrappers import (
    InsertionRewardShapingWrapper,
)


def evaluate_policy(
    model_path: str,
    n_episodes: int = 10,
    render: bool = False,
    save_video: bool = False,
    plot_curves: bool = False,
):
    """
    Evaluate a trained PPO policy.

    Args:
        model_path: Path to the saved model
        n_episodes: Number of episodes to evaluate
        render: Whether to render during evaluation
        save_video: Whether to save videos of episodes
    """
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # Create evaluation environment:
    # Use same observation type and wrappers as during training so the
    #   model receives matching inputs.
    # If plot_curves is requested, we *keep* the reward-shaping wrapper because it provides the dense reward as well as
    #   the potential in info dict. Otherwise we omit it (original behaviour).
    def _make_env():
        env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="state", render_mode="rgb_array")
        env = ClipActionWrapper(env)
        env = RateLimitActionWrapper(env, max_delta=0.1)

        if plot_curves:
            # Add potential-based shaping *last* so info["potential"] is logged
            env = InsertionRewardShapingWrapper(env)

        return env

    env = _make_env()

    # Statistics
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    print(f"\nEvaluating for {n_episodes} episodes...")
    print("=" * 80)

    # Containers for trajectories (optional plotting)
    reward_trajectories: List[List[float]] = []
    sparse_trajectories: List[List[float]] = []
    potential_trajectories: List[List[float]] = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        frames = []

        # Temporary lists for this episode (for plotting)
        if plot_curves:
            dense_ts: List[float] = []
            sparse_ts: List[float] = []
            potential_ts: List[float] = []

        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if plot_curves:
                dense_ts.append(float(reward))
                sparse_ts.append(float(info.get("sparse_r", np.nan)))
                potential_ts.append(float(info.get("potential", np.nan)))

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
            f"Ep {episode + 1:3d} | Reward: {episode_reward:6.2f} | Length: {episode_length:4d} | Success: {is_success}"
        )

        # Store trajectories for plotting
        if plot_curves:
            sparse_trajectories.append(sparse_ts)
            reward_trajectories.append(dense_ts)
            potential_trajectories.append(potential_ts)

            # ------------------------------------------------------------------
            # Debug print: show first 30 steps of episode 1 (index 0)
            # ------------------------------------------------------------------
            if episode == 0 and plot_curves:
                n = 50
                print(f"\nFirst {n} environment steps (episode 1):")
                hdr = "step |   Φ(s)   |    ΔΦ    |  R_dense"
                print(hdr)
                print("-" * len(hdr))

                # ΔΦ: prepend nan for step 0
                dphi = [np.nan] + list(np.diff(potential_ts))
                for t in range(min(n, len(dense_ts))):
                    print(f"{t:4d} | {potential_ts[t]:6.5f} | {dphi[t]:6.5f} | {dense_ts[t]:7.5f}")

        # Save video if requested
        if save_video and len(frames) > 0:
            video_path = f"eval_episode_{episode + 1}.mp4"
            imageio.mimsave(video_path, np.stack(frames), fps=25)
            print(f"  └─ Video saved to {video_path}")

    # Print summary statistics
    print("=" * 80)
    print("Evaluation Summary across all episodes:")
    print(f"  Mean Reward:    {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean Length:    {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Success Rate:   {success_count}/{n_episodes} ({100 * success_count / n_episodes:.1f}%)")
    print(f"  Min Reward:     {np.min(episode_rewards):.2f}")
    print(f"  Max Reward:     {np.max(episode_rewards):.2f}")
    print("=" * 80)

    env.close()

    if plot_curves and len(reward_trajectories) > 0:
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        # Dense reward plot
        for idx, tr in enumerate(reward_trajectories):
            axes[0].plot(tr, label=f"Ep {idx + 1}")
        axes[0].set_ylabel("Dense Reward per step")
        axes[0].legend()
        axes[0].grid(True, linewidth=0.3)

        # Sparse reward plot
        for idx, tr in enumerate(sparse_trajectories):
            axes[1].plot(tr, label=f"Ep {idx + 1}")
        axes[1].set_ylabel("Sparse Reward per step")
        axes[1].legend()
        axes[1].grid(True, linewidth=0.3)

        # Potential plot
        for idx, tr in enumerate(potential_trajectories):
            axes[2].plot(tr, label=f"Ep {idx + 1}")
        axes[2].set_ylabel("Potential Φ(s)")
        axes[2].set_xlabel("Environment Step")
        axes[2].legend()
        axes[2].grid(True, linewidth=0.3)

        plt.tight_layout()
        plt.show()
        plot_path = "reward_potential_plot.png"
        plt.savefig(plot_path)
        print(f"\nPlot saved to {plot_path}")

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
        default=1,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Render episodes during evaluation",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Save videos of evaluation episodes",
    )

    parser.add_argument(
        "--plot-curves",
        action="store_true",
        default=True,
        help="Plot dense reward and shaping potential for each episode",
    )

    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        render=args.render,
        save_video=args.save_video,
        plot_curves=args.plot_curves,
    )


if __name__ == "__main__":
    main()
