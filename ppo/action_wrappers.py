import gymnasium as gym
import numpy as np

from gym_aloha.constants import ACTIONS, XML_JOINT_NAMES, normalize_puppet_gripper_position


class DeltaJointPositionWrapper(gym.ActionWrapper):
    """
    Converts actions from [-1, 1]^14 to delta joint positions.
    The action represents a desired change in joint position, scaled by max_delta.
    """
    def __init__(self, env, max_delta=0.04):
        super().__init__(env)
        self.max_delta = float(max_delta)
        
        # Build arrays for joint limits (14 dimensions: 12 arm joints + 2 grippers)
        physics = self.env.unwrapped._env.physics
        self.qpos_low = np.empty(len(ACTIONS), dtype=np.float32)
        self.qpos_high = np.empty_like(self.qpos_low)
        for i, jnt_name in enumerate(XML_JOINT_NAMES):
            low, high = physics.named.model.jnt_range[jnt_name]
            self.qpos_low[i if i < 6 else i + 1] = low  # +1 skips gripper slot
            self.qpos_high[i if i < 6 else i + 1] = high
        
        # Grippers (normalized 0–1)
        self.qpos_low[6] = 0.0
        self.qpos_high[6] = 1.0
        self.qpos_low[13] = 0.0
        self.qpos_high[13] = 1.0

    def action(self, act):
        # act is in [-1, 1]^14 from PPO
        act = np.clip(act, -1.0, 1.0)
        delta = self.max_delta * act

        # Get qpos for controlled joints (first 16 are the two 6-DOF arms + grippers)
        physics = self.env.unwrapped._env.physics
        qpos_raw = physics.data.qpos[:16].copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        # Normalize gripper positions to [0, 1]
        left_gripper_norm = normalize_puppet_gripper_position(left_qpos_raw[6])
        right_gripper_norm = normalize_puppet_gripper_position(right_qpos_raw[6])
        
        # Concatenate to match action format: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        current_qpos = np.concatenate([
            left_qpos_raw[:6],
            [left_gripper_norm],
            right_qpos_raw[:6],
            [right_gripper_norm]
        ])

        # Compute target position by adding delta
        target_qpos = np.clip(current_qpos + delta, self.qpos_low, self.qpos_high)
        return target_qpos.astype(np.float32)


class ClipActionWrapper(gym.ActionWrapper):
    """
    Clip actions to the true joint limits specified in the XML.

    The wrapper queries the underlying ``dm_control`` Physics instance to obtain
    the per-joint ranges (``model.jnt_range``).  For the two gripper scalar
    actions (normalized 0-1) we hard-code the [0, 1] interval.
    """

    def __init__(self, env):
        super().__init__(env)

        physics = env.unwrapped._env.physics

        # Build look-up tables for per-dimension clipping
        self.low = np.empty(len(ACTIONS), dtype=np.float32)
        self.high = np.empty_like(self.low)

        # Arm joints (absolute radians)
        for i, jnt_name in enumerate(XML_JOINT_NAMES):
            low, high = physics.named.model.jnt_range[jnt_name]
            self.low[i if i < 6 else i + 1] = low  # +1 skips gripper slot
            self.high[i if i < 6 else i + 1] = high

        # Grippers (normalized 0–1)
        self.low[6] = 0.0
        self.high[6] = 1.0
        self.low[13] = 0.0
        self.high[13] = 1.0

    # ------------------------------------------------------------------
    # gym.ActionWrapper API
    # ------------------------------------------------------------------
    def action(self, act):  # noqa: D401 – short signature required by Gym
        return np.clip(act, self.low, self.high)


class RateLimitActionWrapper(gym.ActionWrapper):
    """
    Limit the per-step change (delta) in the action vector. For safety/preventing commanding huge accelerations or
    velocities.
    """

    def __init__(self, env, max_delta=0.1):
        super().__init__(env)
        self.max_delta = float(max_delta)
        self._prev_action = None

    def reset(self, **kwargs):  # noqa: D401
        obs, info = self.env.reset(**kwargs)
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return obs, info

    def action(self, act):  # noqa: D401
        if self._prev_action is None:
            delta = np.zeros_like(act)
        else:
            delta = np.clip(act - self._prev_action, -self.max_delta, self.max_delta)
        smooth_action = (self._prev_action if self._prev_action is not None else 0) + delta
        self._prev_action = smooth_action.copy()
        return smooth_action
