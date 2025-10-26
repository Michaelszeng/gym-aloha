"""
3D Visualization of Gripper Frames in MuJoCo

This script provides a matplotlib-based 3D visualization of:
- Left and right gripper frames (as triads)
- Peg and socket positions
- Interactive rotation and viewing

Usage:
    python visualize_gripper_frames.py
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

import gym_aloha


def plot_frame(ax, position, rotation_matrix, scale=0.05, label="", alpha=1.0):
    """
    Plot a coordinate frame (triad) in 3D.

    Args:
        ax: Matplotlib 3D axis
        position: 3D position [x, y, z]
        rotation_matrix: 3x3 rotation matrix
        scale: Length of the axes
        label: Label for the frame
        alpha: Transparency
    """
    # Define unit axes
    axes = np.eye(3) * scale

    # Rotate axes
    axes_rotated = rotation_matrix @ axes

    # Colors for X, Y, Z
    colors = ["r", "g", "b"]
    labels = ["X", "Y", "Z"]

    for i in range(3):
        # Draw axis
        ax.quiver(
            position[0],
            position[1],
            position[2],
            axes_rotated[0, i],
            axes_rotated[1, i],
            axes_rotated[2, i],
            color=colors[i],
            arrow_length_ratio=0.3,
            linewidth=2,
            alpha=alpha,
        )

    # Add label
    if label:
        ax.text(position[0], position[1], position[2] + scale * 1.5, label, fontsize=10, weight="bold")


def visualize_current_state(env):
    """Visualize the current state of the environment."""
    # Get physics
    physics = env.unwrapped._env.physics

    # Get gripper poses
    left_pos = physics.named.data.xpos["vx300s_left/gripper_link"].copy()
    right_pos = physics.named.data.xpos["vx300s_right/gripper_link"].copy()

    left_xmat = physics.named.data.xmat["vx300s_left/gripper_link"].copy().reshape(3, 3)
    right_xmat = physics.named.data.xmat["vx300s_right/gripper_link"].copy().reshape(3, 3)

    # Get object positions
    obs, _ = env.reset()
    peg_pos = obs["env_state"][0:3]
    peg_quat = obs["env_state"][3:7]
    socket_pos = obs["env_state"][7:10]
    socket_quat = obs["env_state"][10:14]

    # Convert quaternions to rotation matrices
    peg_rot = Rotation.from_quat(peg_quat).as_matrix()
    socket_rot = Rotation.from_quat(socket_quat).as_matrix()

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot gripper frames
    plot_frame(ax, left_pos, left_xmat, scale=0.08, label="Left Gripper", alpha=0.8)
    plot_frame(ax, right_pos, right_xmat, scale=0.08, label="Right Gripper", alpha=0.8)

    # Plot object frames
    plot_frame(ax, peg_pos, peg_rot, scale=0.06, label="Peg", alpha=0.6)
    plot_frame(ax, socket_pos, socket_rot, scale=0.06, label="Socket", alpha=0.6)

    # Plot points for objects
    ax.scatter(*peg_pos, c="red", s=100, marker="o", label="Peg Position")
    ax.scatter(*socket_pos, c="blue", s=100, marker="s", label="Socket Position")

    # Plot lines from grippers to objects
    ax.plot(
        [right_pos[0], peg_pos[0]],
        [right_pos[1], peg_pos[1]],
        [right_pos[2], peg_pos[2]],
        "r--",
        alpha=0.3,
        linewidth=1,
        label="Right Gripper → Peg",
    )
    ax.plot(
        [left_pos[0], socket_pos[0]],
        [left_pos[1], socket_pos[1]],
        [left_pos[2], socket_pos[2]],
        "b--",
        alpha=0.3,
        linewidth=1,
        label="Left Gripper → Socket",
    )

    # Print information
    print("\n" + "=" * 80)
    print("GRIPPER FRAME INFORMATION")
    print("=" * 80)
    print("\nLeft Gripper:")
    print(f"  Position: {left_pos}")
    print(f"  Rotation Matrix:\n{left_xmat}")
    print("\nRight Gripper:")
    print(f"  Position: {right_pos}")
    print(f"  Rotation Matrix:\n{right_xmat}")
    print("\nPeg:")
    print(f"  Position: {peg_pos}")
    print(f"  Quaternion: {peg_quat}")
    print("\nSocket:")
    print(f"  Position: {socket_pos}")
    print(f"  Quaternion: {socket_quat}")
    print("\nDistances:")
    print(f"  Right Gripper → Peg:    {np.linalg.norm(right_pos - peg_pos):.4f} m")
    print(f"  Left Gripper → Socket:  {np.linalg.norm(left_pos - socket_pos):.4f} m")
    print(f"  Peg → Socket:           {np.linalg.norm(peg_pos - socket_pos):.4f} m")
    print("=" * 80 + "\n")

    # Set labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("ALOHA Gripper Frames and Objects\n(RGB triads = XYZ axes)")

    # Set equal aspect ratio
    max_range = 0.3
    mid_x = (left_pos[0] + right_pos[0]) / 2
    mid_y = (left_pos[1] + right_pos[1]) / 2
    mid_z = (left_pos[2] + right_pos[2]) / 2

    ax.set_xlim([mid_x - max_range, mid_x + max_range])
    ax.set_ylim([mid_y - max_range, mid_y + max_range])
    ax.set_zlim([mid_z - max_range, mid_z + max_range])

    # Add legend
    ax.legend(loc="upper right")

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def list_all_bodies_and_geoms(env):
    """List all bodies and geoms in the MuJoCo model."""
    physics = env.unwrapped._env.physics

    print("\n" + "=" * 80)
    print("MUJOCO MODEL STRUCTURE")
    print("=" * 80)

    print("\nBODIES:")
    print("-" * 80)
    for i in range(physics.model.nbody):
        body_name = physics.model.id2name(i, "body")
        if body_name:
            pos = physics.named.data.xpos[body_name]
            print(f"  [{i:3d}] {body_name:40s} pos: {pos}")

    print("\nGEOMS:")
    print("-" * 80)
    for i in range(physics.model.ngeom):
        geom_name = physics.model.id2name(i, "geom")
        if geom_name:
            print(f"  [{i:3d}] {geom_name}")

    print("\nSITES:")
    print("-" * 80)
    for i in range(physics.model.nsite):
        site_name = physics.model.id2name(i, "site")
        if site_name:
            pos = physics.named.data.site_xpos[site_name]
            print(f"  [{i:3d}] {site_name:40s} pos: {pos}")

    print("=" * 80 + "\n")


def main():
    print("Initializing ALOHA environment...")
    env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="state", render_mode="rgb_array")
    env.reset()

    # List all available bodies and geoms
    list_all_bodies_and_geoms(env)

    # Visualize current state
    print("Opening 3D visualization...")
    print("Tip: You can rotate the view by clicking and dragging")
    visualize_current_state(env)

    env.close()


if __name__ == "__main__":
    main()
