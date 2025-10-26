"""
Quick test to verify training setup before starting full training.
This will test environment creation, model initialization, and a few training steps.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import gym_aloha
from train_ppo import AlohaImageExtractor, RewardShapingWrapper


def test_environment():
    """Test environment creation and basic functionality."""
    print("Testing environment creation...")
    env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels", render_mode="rgb_array")

    print("✓ Environment created successfully")

    # Check observation space
    obs, info = env.reset()
    print("✓ Environment reset successful")
    print(f"  - Observation shape: {obs['top'].shape}")
    print(f"  - Observation dtype: {obs['top'].dtype}")

    # Check action space
    print(f"  - Action space: {env.action_space}")

    # Take a random step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("✓ Step successful")
    print(f"  - Reward: {reward}")
    print(f"  - Terminated: {terminated}")

    # Test rendering
    frame = env.render()
    print("✓ Rendering successful")
    print(f"  - Frame shape: {frame.shape}")

    env.close()
    print()


def test_reward_shaping():
    """Test reward shaping wrapper."""
    print("Testing reward shaping wrapper...")
    env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels", render_mode="rgb_array")
    env = RewardShapingWrapper(env)

    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print("✓ Reward shaping wrapper working")
    if "original_reward" in info:
        print(f"  - Original reward: {info['original_reward']}")
        print(f"  - Shaped reward: {reward}")

    env.close()
    print()


def test_model_creation():
    """Test PPO model creation with custom CNN."""
    print("Testing PPO model creation...")

    env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels", render_mode="rgb_array")

    policy_kwargs = dict(
        features_extractor_class=AlohaImageExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  - Using device: {device}")

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=128,  # Small for testing
        batch_size=32,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
    )

    print("✓ Model created successfully")
    print(f"  - Policy: {type(model.policy).__name__}")
    print(f"  - Feature extractor: {type(model.policy.features_extractor).__name__}")

    # Test prediction
    obs, info = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    print("✓ Model prediction successful")
    print(f"  - Action shape: {action.shape}")

    env.close()
    print()


def test_training_step():
    """Test a few training steps."""
    print("Testing training steps...")

    env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels", render_mode="rgb_array")

    policy_kwargs = dict(
        features_extractor_class=AlohaImageExtractor,
        features_extractor_kwargs=dict(features_dim=256),  # Smaller for testing
        net_arch=dict(pi=[128], vf=[128]),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=64,  # Very small for testing
        batch_size=32,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
    )

    print("  - Running 128 training steps...")
    model.learn(total_timesteps=128, progress_bar=False)

    print("✓ Training steps successful")

    env.close()
    print()


def main():
    print("=" * 80)
    print("PPO Training Setup Test")
    print("=" * 80)
    print()

    try:
        test_environment()
        test_reward_shaping()
        test_model_creation()
        test_training_step()

        print("=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
        print()
        print("Your setup is ready for training. To start training, run:")
        print("  python train_ppo.py --total-timesteps 1000000 --log-dir logs/ppo_insertion")
        print()
        print("Or use the quickstart script:")
        print("  bash train_ppo_quickstart.sh")
        print()

    except Exception as e:
        print("=" * 80)
        print("✗ Test failed!")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        import traceback

        traceback.print_exc()
        print()
        print("Please fix the error before starting training.")


if __name__ == "__main__":
    main()
