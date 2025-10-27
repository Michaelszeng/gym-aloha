import os

os.environ["MUJOCO_GL"] = "egl"  # or 'osmesa' for CPU rendering
os.environ["SDL_VIDEODRIVER"] = "dummy"
import gymnasium as gym
import imageio
import mujoco
import numpy as np
from dm_control.mujoco.wrapper.core import MjvOption
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib
from gym_aloha.constants import START_ARM_POSE, ACTIONS  # ACTIONS == 14 names

import gym_aloha

env = gym.make("gym_aloha/AlohaInsertion-v0")
observation, info = env.reset()

# Check rendering backend
print(f"MuJoCo GL backend: {os.environ.get('MUJOCO_GL', 'default')}")
print("Rendering test frame to initialize OpenGL context...")
test_frame = env.render()
print(f"Rendered frame shape: {test_frame.shape}")

# Draw coordinate frames for each body (Source: https://github.com/google-deepmind/mujoco/issues/160)
model = env.unwrapped._env.physics.model
model.vis.scale.framewidth  = 0.02
model.vis.scale.framelength = 0.2
scene_option = MjvOption()
scene_option.frame = enums.mjtFrame.mjFRAME_BODY
# Other options: mjFRAME_NONE, mjFRAME_GEOM, mjFRAME_SITE, mjFRAME_CAMERA, mjFRAME_LIGHT, mjFRAME_WORLD
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

frames = []
# action = env.action_space.sample()
# START_ARM_POSE consists of 2 values per gripper rather than the single normalized gripper position
start_actions = START_ARM_POSE[:6] + [0.0] + START_ARM_POSE[8:14] + [0.0]
action = np.array(start_actions, dtype=np.float32)

action[6] = 0.0
action[13] = 0.0
direction = 1
print(f"Action: {action}")
for _ in range(200):
    # Open and close grippers
    action[6] += 0.01 * direction
    action[13] += 0.01 * direction
    if action[6] >= 1.0 or action[6] < 0.0:
        direction *= -1

    observation, reward, terminated, truncated, info = env.step(action)
    img = env.unwrapped._env.physics.render(camera_id="top", scene_option=scene_option)
    frames.append(img)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
