"""
This script is quite janky.

We effectively run Mujoco's viewer separately, and copy its state data into the dm_control physics object, and use that
to calculate the potential function.
"""

import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from gym_aloha.constants import START_ARM_POSE
from ppo.rewards_wrappers import InsertionRewardShapingWrapperV2

# Remove direct model creation and instead use the environment's physics
# Create Gym environment wrapped with dense reward shaping
base_env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="state", render_mode="rgb_array")
wrapped_env = InsertionRewardShapingWrapperV2(base_env, gamma=0.95)

aloha_env = wrapped_env.unwrapped  # stripped of InsertionRewardShapingWrapper wrappers

# Reset to get initial state and establish physics handle
obs, info = wrapped_env.reset()
print(f"initial info: {info}")

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
num_lines_printed = 0
try:
    while viewer.is_running():
        # Step forward the viewer's model
        mujoco.mj_step(model, data)

        # Push the new joint state into dm_control's Physics
        sync_viewer_to_dm()

        # Compute potential periodically
        if time.time() - last_print_time > 0.1:
            # Debug: print contact and force info
            physics = aloha_env._env.physics

            # Build output buffer
            output_lines = []
            output_lines.append("")

            # Show collision force
            collision_force = aloha_env.compute_robot_collision_force(exclude_object_contacts=True)
            output_lines.append(f"Robot collision force: {collision_force:.6f}")

            obs, original_reward, terminated, truncated, info = wrapped_env.step(aloha_env._env.physics.data.ctrl)

            output_lines.append("")
            output_lines.append("=== Step Info (Post-Step) ===")

            # Show grasp state and success status used for rewards
            output_lines.append(
                f"Grasped (used for rewards): L={info['is_grasped_left']}, R={info['is_grasped_right']}"
            )
            output_lines.append("")

            # Show all other info
            fields_to_skip = ["raw_collision_force"]
            for key, value in info.items():
                if key in fields_to_skip:
                    continue
                if isinstance(value, float):
                    output_lines.append(f"{key:30s}: {value:+.4f}")
                else:
                    output_lines.append(f"{key:30s}: {value}")
            output_lines.append("=" * 50)

            # Move cursor up if we've printed before
            if num_lines_printed > 0:
                print(f"\033[{num_lines_printed}A", end="")

            # Print all lines with line clearing
            for line in output_lines:
                print(f"\033[K{line}")

            num_lines_printed = len(output_lines)
            last_print_time = time.time()

        viewer.sync()
        step_count += 1
        time.sleep(0.01)
finally:
    viewer.close()
