"""
This script is quite janky.

We effectively run Mujoco's viewer separately, and copy its state data into the dm_control physics object, and use that
to calculate the potential function.
"""

import time

import gymnasium as gym
import mujoco
import mujoco.viewer

from gym_aloha.constants import START_ARM_POSE
from ppo.rewards import InsertionRewardShapingWrapper

# Remove direct model creation and instead use the environment's physics
# Create Gym environment wrapped with potential-based reward shaping
base_env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="state", render_mode="rgb_array")
wrapped_env = InsertionRewardShapingWrapper(base_env, gamma=0.95)

aloha_env = wrapped_env.unwrapped  # stripped of InsertionRewardShapingWrapper wrappers

# Reset to get initial state and establish physics handle
obs, _ = wrapped_env.reset()

# Access dm_control Physics object used internally by the environment
physics = aloha_env._env.physics
model = physics.model
data = physics.data

# Build a viewer-compatible MuJoCo model by loading the same XML file used by the env.
# This avoids relying on dm_control's private wrappers.
model = mujoco.MjModel.from_xml_path("gym_aloha/assets/bimanual_viperx_insertion.xml")
# Instantiate a fresh data object for the viewer
data = mujoco.MjData(model)


def sync_dm_to_viewer():
    """Copy dm_control physics state into the viewer data object, including controls."""
    data.qpos[:] = aloha_env._env.physics.data.qpos
    data.qvel[:] = aloha_env._env.physics.data.qvel
    data.ctrl[:] = aloha_env._env.physics.data.ctrl  # initialize slider positions
    mujoco.mj_forward(model, data)


def sync_viewer_to_dm():
    """Copy viewer data state back into dm_control physics so reward logic sees edits, including controls."""
    aloha_env._env.physics.data.qpos[:] = data.qpos
    aloha_env._env.physics.data.qvel[:] = data.qvel
    aloha_env._env.physics.data.ctrl[:] = data.ctrl
    aloha_env._env.physics.forward()


# Initial sync so viewer shows correct starting pose
sync_dm_to_viewer()

print("Launching MuJoCo viewer…  Drag bodies/joints with the mouse to explore the potential function.\n")
# Use non-blocking viewer so our loop can run in the main thread
viewer = mujoco.viewer.launch_passive(model, data)

step_count = 0
last_print_time = time.time()
try:
    while viewer.is_running():
        # Step forward the viewer's model
        mujoco.mj_step(model, data)

        # Push the new joint state into dm_control’s Physics
        sync_viewer_to_dm()

        # We don't advance dynamics; this is a static pose inspector. If you want gravity etc. uncomment:
        # aloha_env._env.physics.step()

        # Compute potential periodically
        if time.time() - last_print_time > 0.1:
            raw_obs = aloha_env._env.task.get_observation(aloha_env._env.physics)
            formatted_obs = aloha_env._format_raw_obs(raw_obs)
            sparse_reward = aloha_env._env.task.get_reward(aloha_env._env.physics)
            is_grasped = sparse_reward >= 2
            potential = wrapped_env._calculate_potential(formatted_obs, is_grasped)
            print(
                f"potential={potential:+.3f}  sparse_reward={sparse_reward}   step={step_count}",
                end="\r",
                flush=True,
            )
            last_print_time = time.time()

        viewer.sync()
        step_count += 1
        time.sleep(0.01)
finally:
    viewer.close()
