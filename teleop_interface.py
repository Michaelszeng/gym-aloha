"""
Interactive Teleop Interface for ALOHA Insertion Task

This script provides a real-time interface to:
- Control each joint and gripper with sliders
- Visualize gripper link poses with triads (RGB = XYZ axes)
- Display the current potential function value
- Test the reward shaping function interactively

Usage:
    python teleop_interface.py
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import cv2
import gymnasium as gym
import numpy as np

import gym_aloha
from ppo.rewards import InsertionRewardShapingWrapper


class TeleopInterface:
    """Interactive interface for controlling ALOHA robot and visualizing potential function."""

    def __init__(self):
        # Create environment
        self.env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="state", render_mode="rgb_array")
        self.env = InsertionRewardShapingWrapper(self.env, gamma=0.99)

        # Reset environment
        self.obs, _ = self.env.reset()

        # Joint positions (14 total: left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1))
        self.joint_positions = self.obs["agent_pos"].copy()

        # Joint names for display
        self.joint_names = [
            "L_waist",
            "L_shoulder",
            "L_elbow",
            "L_forearm",
            "L_wrist_ang",
            "L_wrist_rot",
            "L_gripper",
            "R_waist",
            "R_shoulder",
            "R_elbow",
            "R_forearm",
            "R_wrist_ang",
            "R_wrist_rot",
            "R_gripper",
        ]

        # Joint limits (approximate)
        self.joint_limits = [
            (-3.14, 3.14),  # Left waist
            (-1.88, 1.99),  # Left shoulder
            (-2.15, 1.60),  # Left elbow
            (-3.14, 3.14),  # Left forearm roll
            (-1.74, 2.15),  # Left wrist angle
            (-3.14, 3.14),  # Left wrist rotate
            (0.0, 1.0),  # Left gripper (normalized)
            (-3.14, 3.14),  # Right waist
            (-1.88, 1.99),  # Right shoulder
            (-2.15, 1.60),  # Right elbow
            (-3.14, 3.14),  # Right forearm roll
            (-1.74, 2.15),  # Right wrist angle
            (-3.14, 3.14),  # Right wrist rotate
            (0.0, 1.0),  # Right gripper (normalized)
        ]

        # Create window
        self.window_name = "ALOHA Teleop Interface"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 900)

        # Create trackbars for each joint
        for i, (name, limits) in enumerate(zip(self.joint_names, self.joint_limits)):
            # Scale to 0-1000 range for trackbar
            initial_value = int(self._normalize_to_trackbar(self.joint_positions[i], limits[0], limits[1]))
            cv2.createTrackbar(name, self.window_name, initial_value, 1000, lambda x: None)

        # Create trackbar for grasp status toggle
        cv2.createTrackbar("Grasped (0/1)", self.window_name, 0, 1, lambda x: None)

        print("=" * 80)
        print("ALOHA Teleop Interface Started")
        print("=" * 80)
        print("Controls:")
        print("  - Use sliders to control each joint")
        print("  - Toggle 'Grasped' to switch between pre-grasp and post-grasp phases")
        print("  - Press 'r' to reset to initial pose")
        print("  - Press 'q' or ESC to quit")
        print("=" * 80)

    def _normalize_to_trackbar(self, value, min_val, max_val):
        """Convert joint value to trackbar range [0, 1000]."""
        return (value - min_val) / (max_val - min_val) * 1000

    def _denormalize_from_trackbar(self, trackbar_value, min_val, max_val):
        """Convert trackbar value [0, 1000] to joint range."""
        return min_val + (trackbar_value / 1000.0) * (max_val - min_val)

    def get_joint_positions_from_trackbars(self):
        """Read current joint positions from trackbars."""
        positions = np.zeros(14)
        for i, (name, limits) in enumerate(zip(self.joint_names, self.joint_limits)):
            trackbar_val = cv2.getTrackbarPos(name, self.window_name)
            positions[i] = self._denormalize_from_trackbar(trackbar_val, limits[0], limits[1])
        return positions

    def draw_triad(self, img, pos_2d, orientation_matrix, scale=30):
        """
        Draw a 3D coordinate frame (triad) on the image.

        Args:
            img: Image to draw on
            pos_2d: 2D position (x, y) on the image
            orientation_matrix: 3x3 rotation matrix
            scale: Size of the triad axes in pixels
        """
        if pos_2d is None:
            return

        # Define axes in 3D (XYZ)
        axes_3d = (
            np.array(
                [
                    [1, 0, 0],  # X axis (red)
                    [0, 1, 0],  # Y axis (green)
                    [0, 0, 1],  # Z axis (blue)
                ]
            )
            * scale
        )

        # Rotate axes by orientation
        axes_rotated = axes_3d @ orientation_matrix.T

        # Project to 2D (simple orthographic projection)
        # We'll just use X and Z for the 2D projection (top-down view adjustment)
        axes_2d = axes_rotated[:, [0, 1]]  # Use X, Y components

        colors = [
            (0, 0, 255),  # Red for X
            (0, 255, 0),  # Green for Y
            (255, 0, 0),  # Blue for Z
        ]

        pos_2d = (int(pos_2d[0]), int(pos_2d[1]))

        for i, (axis_2d, color) in enumerate(zip(axes_2d, colors)):
            end_point = (int(pos_2d[0] + axis_2d[0]), int(pos_2d[1] + axis_2d[1]))
            cv2.arrowedLine(img, pos_2d, end_point, color, 2, tipLength=0.3)

    def project_3d_to_2d(self, pos_3d, camera_matrix=None):
        """
        Simple projection of 3D point to 2D image coordinates.
        This is a simplified version - in practice, you'd use the actual camera matrix.
        """
        # Simple orthographic projection (adjust as needed)
        # Scale and offset to fit in image
        scale = 800  # Adjust this based on your workspace
        offset_x = 320
        offset_y = 240

        x = int(pos_3d[0] * scale + offset_x)
        y = int(-pos_3d[1] * scale + offset_y)  # Flip Y for image coordinates

        return (x, y)

    def render_with_overlays(self):
        """Render the scene with gripper triads and potential function overlay."""
        # Get the rendered image from environment
        img = self.env.unwrapped._env.physics.render(height=480, width=640, camera_id="top")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Get gripper poses from physics
        physics = self.env.unwrapped._env.physics

        left_pos = physics.named.data.xpos["vx300s_left/gripper_link"].copy()
        right_pos = physics.named.data.xpos["vx300s_right/gripper_link"].copy()

        # Get gripper orientations (rotation matrices)
        left_xmat = physics.named.data.xmat["vx300s_left/gripper_link"].copy().reshape(3, 3)
        right_xmat = physics.named.data.xmat["vx300s_right/gripper_link"].copy().reshape(3, 3)

        # Project 3D positions to 2D
        left_pos_2d = self.project_3d_to_2d(left_pos)
        right_pos_2d = self.project_3d_to_2d(right_pos)

        # Draw triads for grippers
        self.draw_triad(img, left_pos_2d, left_xmat, scale=40)
        self.draw_triad(img, right_pos_2d, right_xmat, scale=40)

        # Add labels
        cv2.putText(
            img, "LEFT", (left_pos_2d[0] + 50, left_pos_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            img, "RIGHT", (right_pos_2d[0] + 50, right_pos_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Get potential function value
        is_grasped = cv2.getTrackbarPos("Grasped (0/1)", self.window_name) == 1
        potential = self.env._calculate_potential(self.obs, is_grasped)

        # Get object positions
        peg_pos = self.obs["env_state"][0:3]
        socket_pos = self.obs["env_state"][7:10]

        # Calculate distances
        dist_right_peg = np.linalg.norm(right_pos - peg_pos)
        dist_left_socket = np.linalg.norm(left_pos - socket_pos)
        dist_peg_socket = np.linalg.norm(peg_pos - socket_pos)

        # Create info panel
        info_height = 300
        info_panel = np.zeros((info_height, 640, 3), dtype=np.uint8)

        y_offset = 30
        line_height = 25

        # Title
        cv2.putText(
            info_panel, "POTENTIAL FUNCTION INFO", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        y_offset += line_height * 1.5

        # Phase
        phase_text = "PHASE: GRASPED (Post-Grasp)" if is_grasped else "PHASE: PRE-GRASP"
        phase_color = (0, 255, 0) if is_grasped else (255, 128, 0)
        cv2.putText(info_panel, phase_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2)
        y_offset += line_height * 1.5

        # Potential value (large and prominent)
        potential_text = f"POTENTIAL: {potential:.4f}"
        cv2.putText(info_panel, potential_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += line_height * 1.5

        # Distances
        cv2.putText(info_panel, "DISTANCES:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height

        cv2.putText(
            info_panel,
            f"  Right Gripper -> Peg:    {dist_right_peg:.4f} m",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        y_offset += line_height

        cv2.putText(
            info_panel,
            f"  Left Gripper -> Socket:  {dist_left_socket:.4f} m",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        y_offset += line_height

        cv2.putText(
            info_panel,
            f"  Peg -> Socket:           {dist_peg_socket:.4f} m",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        y_offset += line_height * 1.5

        # Gripper positions
        cv2.putText(
            info_panel, "GRIPPER POSITIONS (XYZ):", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        y_offset += line_height

        cv2.putText(
            info_panel,
            f"  Left:  [{left_pos[0]:6.3f}, {left_pos[1]:6.3f}, {left_pos[2]:6.3f}]",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        y_offset += line_height

        cv2.putText(
            info_panel,
            f"  Right: [{right_pos[0]:6.3f}, {right_pos[1]:6.3f}, {right_pos[2]:6.3f}]",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Legend for triad colors
        y_offset += line_height * 1.5
        cv2.putText(info_panel, "TRIAD LEGEND: ", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(info_panel, "  X-axis", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(info_panel, "  Y-axis", (150, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(info_panel, "  Z-axis", (290, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Combine image and info panel
        combined = np.vstack([img, info_panel])

        return combined

    def reset_to_initial(self):
        """Reset robot to initial pose."""
        self.obs, _ = self.env.reset()
        self.joint_positions = self.obs["agent_pos"].copy()

        # Update trackbars
        for i, (name, limits) in enumerate(zip(self.joint_names, self.joint_limits)):
            trackbar_val = int(self._normalize_to_trackbar(self.joint_positions[i], limits[0], limits[1]))
            cv2.setTrackbarPos(name, self.window_name, trackbar_val)

    def run(self):
        """Main loop for the teleop interface."""
        while True:
            # Get joint positions from trackbars
            target_positions = self.get_joint_positions_from_trackbars()

            # Apply action (target positions become the action)
            action = target_positions.astype(np.float32)

            # Step environment
            self.obs, reward, terminated, truncated, info = self.env.step(action)

            # Render with overlays
            display_img = self.render_with_overlays()

            # Show image
            cv2.imshow(self.window_name, display_img)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC
                print("Quitting...")
                break
            elif key == ord("r"):  # Reset
                print("Resetting to initial pose...")
                self.reset_to_initial()

        # Cleanup
        cv2.destroyAllWindows()
        self.env.close()


def main():
    interface = TeleopInterface()
    interface.run()


if __name__ == "__main__":
    main()
