import os
os.environ["MUJOCO_GL"] = "egl"  # or 'osmesa' for CPU rendering
os.environ["SDL_VIDEODRIVER"] = "dummy"
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha

env = gym.make("gym_aloha/AlohaInsertion-v0")
observation, info = env.reset()

# Check rendering backend
print(f"MuJoCo GL backend: {os.environ.get('MUJOCO_GL', 'default')}")
print("Rendering test frame to initialize OpenGL context...")
test_frame = env.render()
print(f"Rendered frame shape: {test_frame.shape}")


frames = []
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
