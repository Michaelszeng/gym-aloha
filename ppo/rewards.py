import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation


class InsertionRewardShapingWrapper(gym.Wrapper):
    """
    Multi-phase potential-based reward shaping function for the peg insertion task.
    """

    def __init__(
        self,
        env,
        gamma: float,
        w_g_peg: float = 1.0,  # Weight for gripper -> peg
        w_g_socket: float = 1.0,  # Weight for other gripper -> socket
        w_peg_sock: float = 2.0,  # Weight for peg -> socket
        w_angle: float = 0.5,  # Weight for peg/socket alignment
        c_grasp: float = 2.0,
    ):  # Bonus for achieving grasp
        super().__init__(env)
        self.gamma = gamma
        self.w_g_peg = w_g_peg
        self.w_g_socket = w_g_socket
        self.w_peg_sock = w_peg_sock
        self.w_angle = w_angle
        self.c_grasp = c_grasp

        # Stores Φ(s) of the *previous* state
        self.potential = 0.0

    def reset(self, **kwargs):
        """
        Resets the environment and the potential.
        """
        obs, info = self.env.reset(**kwargs)

        # Calculate initial potential Φ(s_0)
        self.potential = self._calculate_potential(obs, is_grasped=False)

        return obs, info

    def step(self, action):
        """
        Takes a step, calculates the shaping reward, and adds it to the
        original environment reward.
        """
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Check if the peg is grasped in the new state 'obs'
        is_grasped = original_reward >= 2

        # Calculate potential of the new state: Φ(s')
        new_potential = self._calculate_potential(obs, is_grasped)

        # Calculate shaping reward: F = γ * Φ(s') - Φ(s)
        shaping_reward = (self.gamma * new_potential) - self.potential

        # Update the previous potential: self.potential = Φ(s')
        self.potential = new_potential

        # Add the shaping reward to the original sparse reward
        total_reward = original_reward + shaping_reward

        # Clip reward for stability (optional, but often good)
        total_reward = np.clip(total_reward, -5.0, 5.0)

        return obs, total_reward, terminated, truncated, info

    def _calculate_potential(self, obs: dict, is_grasped: bool) -> float:
        """
        Calculates the potential Φ(s) for a given observation.
        """
        peg_pos = obs["env_state"][0:3]
        peg_quat = obs["env_state"][3:7]
        socket_pos = obs["env_state"][7:10]
        socket_quat = obs["env_state"][10:14]

        if not is_grasped:  # PHASE 1: PRE-GRASP
            # Guide the grippers to the objects

            # Access physics object from the wrapped environment
            physics = self.env.unwrapped._env.physics

            # Get gripper positions from physics named data
            # Note: right gripper should grasp peg, left gripper should grasp socket
            left_gripper_pos = physics.named.data.xpos["vx300s_left/gripper_link"].copy()
            right_gripper_pos = physics.named.data.xpos["vx300s_right/gripper_link"].copy()

            # 1a. Distance from right gripper to peg
            dist_right_g_peg = np.linalg.norm(right_gripper_pos - peg_pos)

            # 1b. Distance from left gripper to socket
            dist_left_g_socket = np.linalg.norm(left_gripper_pos - socket_pos)

            potential = -self.w_g_peg * dist_right_g_peg - self.w_g_socket * dist_left_g_socket
            return potential

        else:  # PHASE 2: PEG IS GRASPED
            # Guide the peg (now held) to the socket

            # 2a. Distance from peg to socket
            dist_peg_socket = np.linalg.norm(peg_pos - socket_pos)

            # 2b. Alignment of peg and socket
            peg_rot = Rotation.from_quat(peg_quat)
            socket_rot = Rotation.from_quat(socket_quat)
            peg_z_axis = peg_rot.apply([0, 0, 1])
            socket_z_axis = socket_rot.apply([0, 0, 1])
            dot_product = np.clip(np.dot(peg_z_axis, socket_z_axis), -1.0, 1.0)
            err_angle = np.arccos(dot_product)

            potential = (
                self.c_grasp  # Start with the grasp bonus
                - self.w_peg_sock * dist_peg_socket
                - self.w_angle * err_angle
            )
            return potential
