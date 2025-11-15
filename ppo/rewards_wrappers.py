from enum import Enum

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from gym_aloha.constants import START_ARM_POSE


class InsertionPhase(Enum):
    PRE_GRASP = 0
    READY_GRASP_PEG = 1
    READY_GRASP_SOCKET = 2
    READY_GRASP_BOTH = 3
    # GRASPED_PEG = 4
    # GRASPED_SOCKET = 5
    GRASPED_BOTH = 6


class InsertionRewardShapingWrapperV2(gym.Wrapper):
    """
    Dense reward shaping wrapper for the dual-arm peg insertion task.

    Reward structure guides the robot through these phases:
    1. Reach toward both objects (peg with right gripper, socket with left gripper)
    2. Grasp both objects
    3. Align peg with socket (position and orientation)
    4. Insert peg into socket
    5. Success - maintain successful insertion

    Unlike potential-based shaping, this uses direct dense rewards at each timestep.
    """

    def __init__(
        self,
        env,
        gamma: float = 0.99,
        # Collision force parameters
        robot_force_mult: float = 0.01,  # Multiplier to scale raw forces to normalized units
        robot_force_penalty_min: float = 0.5,  # Minimum force threshold (below this = no penalty)
        robot_cumulative_force_limit: float = 1000.0,  # Max cumulative force before penalty
        # Phase thresholds
        insertion_threshold: float = 0.05,  # Distance to start insertion phase
        gripper_over_obj_threshold: float = 0.2,  # Max XY distance to receive gripper-over-object reward
        # Episode configuration
        max_episode_steps: int = 500,  # Maximum steps per episode for success reward calculation
        # Reward scaling
        normalize_rewards: bool = False,  # Whether to normalize to [0, 1]
    ):
        super().__init__(env)
        self.gamma = gamma

        # Collision force parameters
        self.robot_force_mult = robot_force_mult
        self.robot_force_penalty_min = robot_force_penalty_min
        self.robot_cumulative_force_limit = robot_cumulative_force_limit

        # Thresholds
        self.insertion_threshold = insertion_threshold
        self.gripper_over_obj_threshold = gripper_over_obj_threshold
        self.normalize_rewards = normalize_rewards

        # Episode tracking
        self.max_episode_steps = max_episode_steps
        self._episode_step = 0

        # Contact tracking for grasp detection (with asymmetric bidirectional hysteresis)
        self._contact_hysteresis_grasp = 2  # Steps to transition from no grasp -> grasp
        self._contact_hysteresis_release = 3  # Steps to transition from grasp -> no grasp
        self._left_contact_count = 0
        self._right_contact_count = 0
        self._left_grasped_stable = False  # Stable grasp state with hysteresis
        self._right_grasped_stable = False

        # Cumulative collision force tracking
        self._cumulative_collision_force = 0.0

    def reset(self, **kwargs):
        """Resets the environment and contact tracking."""
        obs, info = self.env.reset(**kwargs)
        self._left_contact_count = 0
        self._right_contact_count = 0
        self._left_grasped_stable = False
        self._right_grasped_stable = False
        self._cumulative_collision_force = 0.0
        self._episode_step = 0
        return obs, info

    def _detect_grasp_both_with_hysteresis(self) -> tuple[bool, bool, bool]:
        """
        Detect grasps with asymmetric bidirectional hysteresis to avoid flickering.

        Requires self._contact_hysteresis_grasp consecutive contacts to transition to grasped state,
        but self._contact_hysteresis_release consecutive non-contacts to transition back to not grasped.
        """
        left_now, right_now = self.env.unwrapped.detect_grasp()

        # Update counters: +1 if contact detected, -1 if not (clamped to [0, release_threshold])
        self._left_contact_count = max(
            0, min(self._contact_hysteresis_release, self._left_contact_count + (1 if left_now else -1))
        )
        self._right_contact_count = max(
            0, min(self._contact_hysteresis_release, self._right_contact_count + (1 if right_now else -1))
        )

        # Update stable state:
        # - Transition to grasped when counter >= grasp threshold
        # - Transition to not grasped when counter = 0
        self._left_grasped_stable = self._left_contact_count >= self._contact_hysteresis_grasp
        self._right_grasped_stable = self._right_contact_count >= self._contact_hysteresis_grasp

        return (
            self._left_grasped_stable,
            self._right_grasped_stable,
            self._left_grasped_stable and self._right_grasped_stable,
        )

    def step(self, action):
        """
        Takes a step, calculates dense reward, and adds it to the original environment reward.
        """
        # Detect grasps BEFORE stepping (more stable than post-step detection)
        is_grasped_left_pre, is_grasped_right_pre, is_grasped_both_pre = self._detect_grasp_both_with_hysteresis()

        obs, _, terminated, truncated, info = self.env.step(action)

        # We override terminated and truncated here so that all terminations (including truncation, i.e. timeouts)
        # are treated as true terminations and SB3 will not bootstrap the value.
        # In other words, we treat timeouts as true failures that cannot accumulate any more rewards after the timeout.
        if truncated and not terminated:
            terminated = True
            truncated = False

        # Increment episode step counter
        self._episode_step += 1

        # Calculate dense reward using pre-step grasp state
        dense_reward = self._calculate_dense_reward(
            obs, info, is_grasped_left_pre, is_grasped_right_pre, is_grasped_both_pre
        )
        info["dense_r"] = dense_reward  # Log dense reward

        total_reward = dense_reward * self.gamma

        return obs, total_reward, terminated, truncated, info

    def _calculate_dense_reward(
        self,
        obs: dict,
        info: dict,
        is_grasped_left: bool,
        is_grasped_right: bool,
        is_grasped_both: bool,
    ) -> float:
        """
        Calculates the dense reward for a given observation.

        The reward structure:
        - Universal: ~22 points (5 reach + 2 stillness + 1 arm resting + 1 gripper Y-align +
                                  5 collision + 2 grasp + 6 success bonus)
        - Phase 1 (Not Grasped): ~1 point (1 gripper-over-obj)
        - Phase 2 (Grasped Both): ~9 points (1 phase bonus + 5 position + 3 orientation)

        Total max reward per step: ~32 points (unnormalized)
        """
        reward = 0.0

        # Access physics for detailed state information
        physics = self.env.unwrapped._env.physics

        # Extract object positions and orientations from observation
        current_qpos = obs["agent_pos"]  # Shape: (14,) - all joint positions
        peg_pos = obs["env_state"][0:3]
        peg_quat = obs["env_state"][3:7]
        socket_pos = obs["env_state"][7:10]
        socket_quat = obs["env_state"][10:14]

        # Get gripper positions (average of finger tips)
        left_finger_avg_pos = (
            physics.named.data.xpos["vx300s_left/left_finger_tip"].copy()
            + physics.named.data.xpos["vx300s_left/right_finger_tip"].copy()
        ) / 2
        right_finger_avg_pos = (
            physics.named.data.xpos["vx300s_right/left_finger_tip"].copy()
            + physics.named.data.xpos["vx300s_right/right_finger_tip"].copy()
        ) / 2

        # Get gripper orientations
        # MuJoCo quaternions are in [w, x, y, z] format, but scipy expects [x, y, z, w]
        left_gripper_quat_mj = physics.named.data.xquat["vx300s_left/gripper_link"].copy()
        right_gripper_quat_mj = physics.named.data.xquat["vx300s_right/gripper_link"].copy()
        # Convert from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        left_gripper_quat = np.array(
            [left_gripper_quat_mj[1], left_gripper_quat_mj[2], left_gripper_quat_mj[3], left_gripper_quat_mj[0]]
        )
        right_gripper_quat = np.array(
            [right_gripper_quat_mj[1], right_gripper_quat_mj[2], right_gripper_quat_mj[3], right_gripper_quat_mj[0]]
        )

        # ---------------------------------------------------
        # PHASE DETECTION (passed from pre-step state)
        # ---------------------------------------------------
        is_success = info.get("is_success", False)

        # Log phase information (with hysteresis applied, from pre-step)
        info["is_grasped_left"] = is_grasped_left
        info["is_grasped_right"] = is_grasped_right
        info["is_grasped_both"] = is_grasped_both

        # ---------------------------------------------------
        # UNIVERSAL REWARDS (applied in all phases)
        # ---------------------------------------------------

        # REACHING REWARDS (max: 5.0)
        # Guide grippers toward their respective objects
        dist_right_to_peg = np.linalg.norm(right_finger_avg_pos - peg_pos)
        dist_left_to_socket = np.linalg.norm(left_finger_avg_pos - socket_pos)

        reach_peg_reward = 2.5 * (1 - np.tanh(3 * dist_right_to_peg))
        reach_socket_reward = 2.5 * (1 - np.tanh(3 * dist_left_to_socket))
        reward += reach_peg_reward + reach_socket_reward

        info["reach_peg_r"] = reach_peg_reward
        info["reach_socket_r"] = reach_socket_reward
        info["dist_right_to_peg"] = dist_right_to_peg
        info["dist_left_to_socket"] = dist_left_to_socket

        # END-EFFECTOR STILLNESS REWARD (max: 1.0)
        # Penalizes excessive end-effector velocity to encourage smooth, controlled motion.
        # Get gripper body velocities (linear velocity)
        # cvel returns [angular_vel (3), linear_vel (3)], so we take indices 3:6
        left_gripper_vel = physics.named.data.cvel["vx300s_left/gripper_link"][3:6].copy()
        right_gripper_vel = physics.named.data.cvel["vx300s_right/gripper_link"][3:6].copy()
        left_vel_norm = np.linalg.norm(left_gripper_vel)
        right_vel_norm = np.linalg.norm(right_gripper_vel)

        # Dividing by 1 before tanh means velocities < 1 m/s receive less penalty
        left_still_reward = 1.0 * (1 - np.tanh(left_vel_norm / 1.0))
        right_still_reward = 1.0 * (1 - np.tanh(right_vel_norm / 1.0))
        ee_still_reward = left_still_reward + right_still_reward

        reward += ee_still_reward
        info["ee_still_r"] = ee_still_reward
        info["left_gripper_vel"] = left_gripper_vel
        info["right_gripper_vel"] = right_gripper_vel
        info["left_gripper_vel_norm"] = left_vel_norm
        info["right_gripper_vel_norm"] = right_vel_norm

        # ARM RESTING ORIENTATION REWARD (max: 1.0)
        # Encourages arm joints (excluding grippers) to stay near resting configuration.
        # This helps the robot maintain a canonical pose for consistency across episodes.
        # Extract arm joints only (exclude grippers at indices 6-7 and 13-14)
        # Left arm: indices 0-5, Right arm: indices 7-12 (after removing left gripper)
        left_arm_qpos = current_qpos[0:6]
        right_arm_qpos = current_qpos[7:13]
        current_arm_qpos = np.concatenate([left_arm_qpos, right_arm_qpos])
        # Get resting arm positions (exclude grippers)
        resting_arm_qpos = np.array(START_ARM_POSE)[np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])]
        arm_to_resting_diff = np.linalg.norm(current_arm_qpos - resting_arm_qpos)
        # Dividing by 5 provides gentle shaping - allows some deviation while encouraging rest pose
        arm_resting_reward = 1.0 * (1 - np.tanh(arm_to_resting_diff / 5.0))
        reward += arm_resting_reward
        info["arm_resting_r"] = arm_resting_reward
        info["arm_to_resting_diff"] = arm_to_resting_diff

        # GRIPPER Y-AXIS ALIGNMENT REWARD (max: 1.0)
        # Encourages gripper Y-axes to align with world Y-axis for consistent orientation.
        # This helps maintain stable grasping orientation throughout the task.
        left_rot = Rotation.from_quat(left_gripper_quat)
        right_rot = Rotation.from_quat(right_gripper_quat)
        left_y_axis = left_rot.apply([0, 1, 0])
        right_y_axis = right_rot.apply([0, 1, 0])
        world_y_axis = np.array([0, 1, 0])

        # Compute alignment errors (0 when perfectly aligned)
        left_y_dot = np.clip(np.dot(left_y_axis, world_y_axis), -1.0, 1.0)
        right_y_dot = np.clip(np.dot(right_y_axis, world_y_axis), -1.0, 1.0)
        left_y_error = np.arccos(np.abs(left_y_dot))  # Use abs to allow flipped alignment
        right_y_error = np.arccos(np.abs(right_y_dot))

        # Reward for alignment (0.5 per gripper, max 1.0 total)
        left_y_align_reward = 0.5 * (1 - np.tanh(2 * left_y_error))
        right_y_align_reward = 0.5 * (1 - np.tanh(2 * right_y_error))
        gripper_y_align_reward = left_y_align_reward + right_y_align_reward

        reward += gripper_y_align_reward
        info["gripper_y_align_r"] = gripper_y_align_reward
        info["left_y_error"] = left_y_error
        info["right_y_error"] = right_y_error

        # GRASP REWARD
        if is_grasped_both:
            grasp_reward = 2.0
            reward += grasp_reward
            info["grasp_r"] = grasp_reward

        # SUCCESS REWARD remaining_steps * current_step_reward (at success)
        if is_success:
            reward += 6.0  # success bonus
            remaining_steps = max(0, self.max_episode_steps - self._episode_step)
            success_reward = remaining_steps * reward  # reward here is the dense reward so far
        else:
            success_reward = 0.0

        reward += success_reward
        info["success_r"] = success_reward
        info["episode_step"] = self._episode_step
        info["remaining_steps"] = max(0, self.max_episode_steps - self._episode_step)

        # ---------------------------------------------------
        # COLLISION PENALTIES
        # ---------------------------------------------------

        # Get raw collision force from environment (computed in step)
        raw_collision_force = info["env/collision_force"]
        info["raw_collision_force"] = raw_collision_force

        # STEP COLLISION PENALTY (max: ~3.0, min: ~0.0)
        # Penalizes contact forces at each timestep to discourage collisions.
        # - robot_force_mult scales raw forces to normalized units
        # - Forces below robot_force_penalty_min are not penalized (clamp)
        # - The outer multiplier of 3 and inner multiplier of 3 create strong
        #   penalty for collisions while allowing light contact
        # Returns ~3 when no collision, ~0 when high collision forces
        scaled_force = self.robot_force_mult * raw_collision_force
        clamped_force = np.maximum(scaled_force - self.robot_force_penalty_min, 0.0)
        step_no_collision_reward = 3.0 * (1 - np.tanh(3 * clamped_force))
        reward += step_no_collision_reward
        info["step_no_collision_r"] = step_no_collision_reward
        info["scaled_collision_force"] = scaled_force

        # Update cumulative collision force
        self._cumulative_collision_force += raw_collision_force
        info["cumulative_collision_force"] = self._cumulative_collision_force

        # CUMULATIVE COLLISION THRESHOLD REWARD (max: 2.0, min: 0.0)
        # Binary reward for keeping total accumulated collision force below threshold.
        # If cumulative force exceeds limit, this reward becomes 0 for rest of episode.
        # Encourages collision-free trajectories throughout the entire task.
        cumulative_under_threshold = float(self._cumulative_collision_force < self.robot_cumulative_force_limit)
        cumulative_collision_reward = 2.0 * cumulative_under_threshold
        reward += cumulative_collision_reward
        info["cumulative_collision_r"] = cumulative_collision_reward

        # ---------------------------------------------------
        # PHASE 1: NOT GRASPED BOTH (Reaching and Grasping)
        # Max total: ~12 points (6 reach + 2 gripper-over-obj + 4 grasp progress)
        # ---------------------------------------------------
        if not is_grasped_both:
            not_grasped_reward = 0.0

            # GRIPPER OVER OBJECT REWARDS (max: 1.0)
            # Encourage gripper to align with object in xy-plane (top-down view).
            # This helps ensure proper grasp orientation before descending.
            # Uses only xy coordinates to check horizontal alignment.
            # Only rewards if gripper is within threshold distance (otherwise reward = 0)
            # Distances < 0.02m receive maximum reward (0.5), then decay smoothly
            right_over_peg_dist = np.linalg.norm(right_finger_avg_pos[:2] - peg_pos[:2])
            left_over_socket_dist = np.linalg.norm(left_finger_avg_pos[:2] - socket_pos[:2])

            # Only give reward if within threshold distance
            # Clamp distance to 0 if < 0.03m for max reward
            if right_over_peg_dist < self.gripper_over_obj_threshold:
                adjusted_right_dist = max(0, right_over_peg_dist - 0.03)
                right_over_peg_reward = 0.5 * (1 - np.tanh(5 * adjusted_right_dist))
            else:
                right_over_peg_reward = 0.0

            if left_over_socket_dist < self.gripper_over_obj_threshold:
                adjusted_left_dist = max(0, left_over_socket_dist - 0.03)
                left_over_socket_reward = 0.5 * (1 - np.tanh(5 * adjusted_left_dist))
            else:
                left_over_socket_reward = 0.0

            not_grasped_reward += right_over_peg_reward + left_over_socket_reward

            info["right_over_peg_r"] = right_over_peg_reward
            info["left_over_socket_r"] = left_over_socket_reward
            info["right_over_peg_dist"] = right_over_peg_dist
            info["left_over_socket_dist"] = left_over_socket_dist

            reward += not_grasped_reward
            info["phase"] = "not_grasped"
            info["not_grasped_r"] = not_grasped_reward

        # ---------------------------------------------------
        # PHASE 2: GRASPED BOTH (Alignment and Insertion)
        # Additional: ~15 points
        # ---------------------------------------------------
        else:
            grasped_reward = 0.0

            # PHASE TRANSITION BONUS (+1.0)
            # Adds constant to grasped phase to match max reward from not_grasped phase.
            # This ensures reward monotonically increases as task progresses,
            # preventing reward drops when transitioning between phases.
            grasped_reward += 1.0

            # Calculate peg-socket alignment metrics
            peg_socket_diff = peg_pos - socket_pos

            # Weighted distance: penalize YZ misalignment more than X
            # This encourages vertical alignment before X-axis approach
            x_weight = 0.3  # Lower weight for X distance
            yz_weight = 1.0  # Higher weight for YZ distance
            weighted_diff = peg_socket_diff.copy()
            weighted_diff[0] *= x_weight  # Scale X component down
            weighted_diff[1] *= yz_weight  # Y component
            weighted_diff[2] *= yz_weight  # Z component
            peg_socket_weighted_dist = np.linalg.norm(weighted_diff)

            # Also compute unweighted distance for logging
            peg_socket_dist = np.linalg.norm(peg_socket_diff)
            peg_socket_xy_dist = np.linalg.norm(peg_socket_diff[:2])
            peg_socket_z_dist = np.abs(peg_socket_diff[2])

            # POSITION ALIGNMENT REWARD (max: 5.0)
            # Encourage bringing peg and socket close together (prioritizing YZ alignment)
            align_pos_reward = 5.0 * (1 - np.tanh(3 * peg_socket_weighted_dist))
            grasped_reward += align_pos_reward
            info["align_pos_r"] = align_pos_reward
            info["peg_socket_dist"] = peg_socket_dist
            info["peg_socket_weighted_dist"] = peg_socket_weighted_dist
            info["peg_socket_xy_dist"] = peg_socket_xy_dist
            info["peg_socket_z_dist"] = peg_socket_z_dist

            # ORIENTATION ALIGNMENT REWARD (max: 3.0)
            # Encourage aligning peg and socket x-axes for insertion
            # Note: x-axis is the length-wise axis for both peg and socket
            # Convert MuJoCo [w,x,y,z] quaternions to scipy [x,y,z,w] format
            peg_quat_scipy = np.array([peg_quat[1], peg_quat[2], peg_quat[3], peg_quat[0]])
            socket_quat_scipy = np.array([socket_quat[1], socket_quat[2], socket_quat[3], socket_quat[0]])
            peg_rot = Rotation.from_quat(peg_quat_scipy)
            socket_rot = Rotation.from_quat(socket_quat_scipy)
            peg_x_axis = peg_rot.apply([1, 0, 0])
            socket_x_axis = socket_rot.apply([1, 0, 0])

            # Compute alignment error (want parallel or anti-parallel)
            dot_product = np.clip(np.dot(peg_x_axis, socket_x_axis), -1.0, 1.0)
            alignment_error = np.arccos(np.abs(dot_product))  # 0 when aligned

            align_orient_reward = 3.0 * (1 - np.tanh(2 * alignment_error))
            grasped_reward += align_orient_reward
            info["align_orient_r"] = align_orient_reward
            info["alignment_error"] = alignment_error

            reward += grasped_reward
            info["phase"] = "grasped_both"
            info["grasped_r"] = grasped_reward

        # ---------------------------------------------------
        # NORMALIZATION
        # ---------------------------------------------------
        if self.normalize_rewards:
            max_reward = 32.0  # Approximate maximum
            reward = reward / max_reward
            info["max_reward_estimate"] = max_reward

        info["dense_r"] = reward

        return reward


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
        w_finger_left: float = 0.01,  # Weight for distance between left fingers
        w_finger_right: float = 0.01,  # Weight for distance between right fingers
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
            # Note: x-axis is the length-wise axis for both peg and socket
            # Convert MuJoCo [w,x,y,z] quaternions to scipy [x,y,z,w] format
            peg_quat_scipy = np.array([peg_quat[1], peg_quat[2], peg_quat[3], peg_quat[0]])
            socket_quat_scipy = np.array([socket_quat[1], socket_quat[2], socket_quat[3], socket_quat[0]])
            peg_rot = Rotation.from_quat(peg_quat_scipy)
            socket_rot = Rotation.from_quat(socket_quat_scipy)
            peg_x_axis = peg_rot.apply([1, 0, 0])
            socket_x_axis = socket_rot.apply([1, 0, 0])
            dot_product = np.clip(np.dot(peg_x_axis, socket_x_axis), -1.0, 1.0)
            err_angle = np.arccos(np.abs(dot_product))  # Use abs for parallel or anti-parallel

            potential = (
                self.c_grasp  # grasp bonus
                - self.w_peg_sock * dist_peg_socket
                - self.w_table * dist_peg_table
                - self.w_table * dist_socket_table
                - self.w_angle * err_angle
            )
            return potential
