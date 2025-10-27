from enum import Enum

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation


class InsertionPhase(Enum):
    PRE_GRASP = 0
    READY_GRASP_PEG = 1
    READY_GRASP_SOCKET = 2
    READY_GRASP_BOTH = 3
    # GRASPED_PEG = 4
    # GRASPED_SOCKET = 5
    GRASPED_BOTH = 6


class InsertionRewardShapingWrapper(gym.Wrapper):
    """
    Multi-phase potential-based reward shaping function for the peg insertion task.
    """

    def __init__(
        self,
        env,
        gamma: float = 1.0,
        w_g_peg: float = 1.0,  # Weight for gripper -> peg
        w_g_socket: float = 1.0,  # Weight for other gripper -> socket
        w_finger_left: float = 1.0,  # Weight for distance between left fingers
        w_finger_right: float = 1.0,  # Weight for distance between right fingers
        w_peg_sock: float = 1.0,  # Weight for peg -> socket
        w_table: float = 0.25,  # Weight for desired distance from table
        w_angle: float = 1.0,  # Weight for peg/socket alignment
        k_x: float = 0.5,  # Scale x-axis error in peg-socket distance
        # c_ready_grasp_peg: float = 0.2,  # Bonus for ready grasp peg
        # c_ready_grasp_socket: float = 0.2,  # Bonus for ready grasp socket
        # c_ready_grasp_both: float = 0.2,  # Bonus for ready grasp both
        # c_grasp: float = 0.25,  # Bonus for achieving grasp
        c_ready_grasp_peg: float = 0.5,  # Bonus for ready grasp peg
        c_ready_grasp_socket: float = 0.5,  # Bonus for ready grasp socket
        c_ready_grasp_both: float = 0.5,  # Bonus for ready grasp both
        c_grasp: float = 0.5,  # Bonus for achieving grasp
    ):
        super().__init__(env)
        self.gamma = gamma
        self.w_g_peg = w_g_peg
        self.w_g_socket = w_g_socket
        self.w_finger_left = w_finger_left
        self.w_finger_right = w_finger_right
        self.w_peg_sock = w_peg_sock
        self.w_table = w_table
        self.w_angle = w_angle
        self.c_ready_grasp_peg = c_ready_grasp_peg
        self.c_ready_grasp_socket = c_ready_grasp_socket
        self.c_ready_grasp_both = c_ready_grasp_both
        self.c_grasp = c_grasp

        # --- Hysteresis parameters for contact-based phase detection ---
        self._contact_hysteresis_steps = 2  # consecutive frames required
        self._left_contact_count = 0
        self._right_contact_count = 0
        self.k_x = k_x
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
        info["sparse_r"] = original_reward  # Log sparse reward
        info["potential"] = self.potential  # Log potential

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

        return obs, total_reward, terminated, truncated, info

    def _any_contact(self, physics, geom_names_a, geom_names_b):
        """Return True if *any* geom in list A is touching *any* geom in list B."""
        for i in range(physics.data.ncon):
            id1 = physics.data.contact[i].geom1
            id2 = physics.data.contact[i].geom2
            name1 = physics.model.id2name(id1, "geom")
            name2 = physics.model.id2name(id2, "geom")
            if (name1 in geom_names_a and name2 in geom_names_b) or (name2 in geom_names_a and name1 in geom_names_b):
                return True
        return False

    def _grasped_both(self, physics) -> bool:
        """Both left fingertips touch (or are very near) socket, right fingertips touch peg."""
        socket_geoms = ["socket-1", "socket-2", "socket-3", "socket-4"]
        peg_geom = "red_peg"
        left_fingers = ["vx300s_left/10_left_gripper_finger", "vx300s_left/10_right_gripper_finger"]
        right_fingers = ["vx300s_right/10_left_gripper_finger", "vx300s_right/10_right_gripper_finger"]

        # Instantaneous contact
        left_now = all(self._any_contact(physics, [lf], socket_geoms) for lf in left_fingers)
        right_now = all(self._any_contact(physics, [rf], [peg_geom]) for rf in right_fingers)

        # Hysteresis: update counters
        self._left_contact_count = self._left_contact_count + 1 if left_now else 0
        self._right_contact_count = self._right_contact_count + 1 if right_now else 0

        left_ok = self._left_contact_count >= self._contact_hysteresis_steps
        right_ok = self._right_contact_count >= self._contact_hysteresis_steps

        return left_ok and right_ok

    def _update_phase(self, obs):
        """Single-frame phase classification (no memory)."""
        physics = self.env.unwrapped._env.physics

        # Contact-based grasp test
        if self._grasped_both(physics):
            return InsertionPhase.GRASPED_BOTH

        # Distance tests
        peg_pos = obs["env_state"][:3]
        socket_pos = obs["env_state"][7:10]
        l_tip = (
            physics.named.data.xpos["vx300s_left/left_finger_tip"]
            + physics.named.data.xpos["vx300s_left/right_finger_tip"]
        ) / 2
        r_tip = (
            physics.named.data.xpos["vx300s_right/left_finger_tip"]
            + physics.named.data.xpos["vx300s_right/right_finger_tip"]
        ) / 2
        dl = np.linalg.norm(l_tip - socket_pos)
        dr = np.linalg.norm(r_tip - peg_pos)
        if dl < 0.015 and dr < 0.015:
            return InsertionPhase.READY_GRASP_BOTH
        if dl < 0.015:
            return InsertionPhase.READY_GRASP_SOCKET
        if dr < 0.015:
            return InsertionPhase.READY_GRASP_PEG
        return InsertionPhase.PRE_GRASP

    def _calculate_potential(self, obs: dict, is_grasped: bool) -> float:
        """
        Calculates the potential Φ(s) for a given observation.
        """
        peg_pos = obs["env_state"][0:3]
        peg_quat = obs["env_state"][3:7]
        socket_pos = obs["env_state"][7:10]
        socket_quat = obs["env_state"][10:14]

        # Access physics object from the wrapped environment
        physics = self.env.unwrapped._env.physics

        phase = self._update_phase(obs)

        if phase == InsertionPhase.PRE_GRASP:
            # Guide the grippers to the objects
            left_finger_avg_pos = (
                physics.named.data.xpos["vx300s_left/left_finger_tip"].copy()
                + physics.named.data.xpos["vx300s_left/right_finger_tip"].copy()
            ) / 2
            right_finger_avg_pos = (
                physics.named.data.xpos["vx300s_right/left_finger_tip"].copy()
                + physics.named.data.xpos["vx300s_right/right_finger_tip"].copy()
            ) / 2

            # Distance from left gripper to socket
            dist_left_g_socket = np.linalg.norm(left_finger_avg_pos - socket_pos)

            # Distance from right gripper to peg
            dist_right_g_peg = np.linalg.norm(right_finger_avg_pos - peg_pos)

            # Distance between fingers for each arm (we want the fingers to be open, i.e. distance to be larger)
            dist_left_fingers = np.linalg.norm(
                physics.named.data.xpos["vx300s_left/left_finger_tip"].copy()
                - physics.named.data.xpos["vx300s_left/right_finger_tip"].copy()
            )

            dist_right_fingers = np.linalg.norm(
                physics.named.data.xpos["vx300s_right/left_finger_tip"].copy()
                - physics.named.data.xpos["vx300s_right/right_finger_tip"].copy()
            )

            potential = (
                -self.w_g_peg * dist_right_g_peg
                - self.w_g_socket * dist_left_g_socket
                + self.w_finger_left * dist_left_fingers
                + self.w_finger_right * dist_right_fingers
            )
            return potential

        elif phase == InsertionPhase.READY_GRASP_PEG:
            # Guide left gripper to socket, close right gripper
            left_finger_avg_pos = (
                physics.named.data.xpos["vx300s_left/left_finger_tip"].copy()
                + physics.named.data.xpos["vx300s_left/right_finger_tip"].copy()
            ) / 2
            # Distance from left gripper to socket
            dist_left_g_socket = np.linalg.norm(left_finger_avg_pos - socket_pos)

            # Distance between fingers (we now want this to be small)
            dist_right_fingers = np.linalg.norm(
                physics.named.data.xpos["vx300s_right/left_finger_tip"].copy()
                - physics.named.data.xpos["vx300s_right/right_finger_tip"].copy()
            )

            potential = (
                -self.w_g_socket * dist_left_g_socket
                - self.w_finger_right * dist_right_fingers
                + self.c_ready_grasp_peg
            )
            return potential

        elif phase == InsertionPhase.READY_GRASP_SOCKET:
            # Guide right gripper to peg, close left gripper
            right_finger_avg_pos = (
                physics.named.data.xpos["vx300s_right/left_finger_tip"].copy()
                + physics.named.data.xpos["vx300s_right/right_finger_tip"].copy()
            ) / 2
            # Distance from right gripper to peg
            dist_right_g_peg = np.linalg.norm(right_finger_avg_pos - peg_pos)

            # Distance between fingers (we now want this to be small)
            dist_left_fingers = np.linalg.norm(
                physics.named.data.xpos["vx300s_left/left_finger_tip"].copy()
                - physics.named.data.xpos["vx300s_left/right_finger_tip"].copy()
            )

            potential = (
                -self.w_g_peg * dist_right_g_peg - self.w_finger_left * dist_left_fingers + self.c_ready_grasp_socket
            )
            return potential

        elif phase == InsertionPhase.READY_GRASP_BOTH:
            # Close both fingers
            dist_left_fingers = np.linalg.norm(
                physics.named.data.xpos["vx300s_left/left_finger_tip"].copy()
                - physics.named.data.xpos["vx300s_left/right_finger_tip"].copy()
            )
            dist_right_fingers = np.linalg.norm(
                physics.named.data.xpos["vx300s_right/left_finger_tip"].copy()
                - physics.named.data.xpos["vx300s_right/right_finger_tip"].copy()
            )
            potential = (
                -self.w_finger_left * dist_left_fingers
                - self.w_finger_right * dist_right_fingers
                + self.c_ready_grasp_both
            )
            return potential

        elif phase == InsertionPhase.GRASPED_BOTH:
            # Guide the peg (now held) to the socket

            # Connect peg and socket ~0.1 above table
            dist_peg_table = np.abs(peg_pos[2] - 0.1)
            dist_socket_table = np.abs(socket_pos[2] - 0.1)

            # Distance from peg to socket with lighter x-axis penalty
            diff = peg_pos - socket_pos
            diff[0] *= self.k_x  # scale x-component
            dist_peg_socket = np.linalg.norm(diff)

            # Alignment of peg and socket
            peg_rot = Rotation.from_quat(peg_quat)
            socket_rot = Rotation.from_quat(socket_quat)
            peg_z_axis = peg_rot.apply([0, 0, 1])
            socket_z_axis = socket_rot.apply([0, 0, 1])
            dot_product = np.dot(peg_z_axis, socket_z_axis)
            err_angle = np.arccos(dot_product)

            potential = (
                self.c_grasp  # grasp bonus
                - self.w_peg_sock * dist_peg_socket
                - self.w_table * dist_peg_table
                - self.w_table * dist_socket_table
                - self.w_angle * err_angle
            )
            return potential


# -----------------------------------------------------------------------------
# Generic smoothness penalty wrapper
# -----------------------------------------------------------------------------


class SmoothnessPenaltyWrapper(gym.Wrapper):
    """Penalise large per-step changes in the action vector.

    Reward_t = Reward_t_original - coeff * ||a_t - a_{t-1}||_2
    """

    def __init__(self, env, coeff: float = 0.1):
        super().__init__(env)
        self.coeff = float(coeff)
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._prev_action.fill(0.0)
        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        penalty = -self.coeff * np.linalg.norm(action - self._prev_action)
        reward += penalty
        self._prev_action = action.copy()
        return obs, reward, terminated, truncated, info
