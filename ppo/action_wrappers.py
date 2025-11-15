import gymnasium as gym
import numpy as np

from gym_aloha.constants import ACTIONS

# ---------------------------------------------------------------------------
# Time-limit masking wrapper (prevents value bootstrap bias on truncation)
# ---------------------------------------------------------------------------


class TimeLimitMask(gym.Wrapper):
    """
    Marks truncations caused by hitting the episode step limit.

    Stable-Baselines3 will NOT bootstrap the value if the info dict contains
    ``{"TimeLimit.truncated": True}``.  This avoids critic bias from fake
    terminations when the environment simply times out.
    """

    def __init__(self, env: gym.Env, max_episode_steps: int = 500):
        super().__init__(env)
        self.max_episode_steps = int(max_episode_steps)
        self.elapsed_steps = 0

    def reset(self, **kwargs):  # type: ignore[override]
        self.elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1

        if self.elapsed_steps >= self.max_episode_steps:
            # Convert hard time-limit into truncation flag that SB3 understands
            truncated = True
            info["TimeLimit.truncated"] = True

        return obs, reward, terminated, truncated, info


class ClipActionWrapper(gym.ActionWrapper):
    """
    Clip actions to the true joint limits specified in the XML.

    The wrapper queries the underlying ``dm_control`` Physics instance to obtain
    the per-joint ranges (``model.jnt_range``).  For the two gripper scalar
    actions (normalized 0-1) we hard-code the [0, 1] interval.
    """

    # Order of joints in ACTIONS that have a direct MuJoCo counterpart. The two
    # gripper scalars (indices 6 and 13) are treated separately.
    _XML_JOINT_NAMES = [
        # left arm 6 DOF
        "vx300s_left/waist",
        "vx300s_left/shoulder",
        "vx300s_left/elbow",
        "vx300s_left/forearm_roll",
        "vx300s_left/wrist_angle",
        "vx300s_left/wrist_rotate",
        # right arm 6 DOF
        "vx300s_right/waist",
        "vx300s_right/shoulder",
        "vx300s_right/elbow",
        "vx300s_right/forearm_roll",
        "vx300s_right/wrist_angle",
        "vx300s_right/wrist_rotate",
    ]

    def __init__(self, env):
        super().__init__(env)

        physics = env.unwrapped._env.physics

        # Build look-up tables for per-dimension clipping
        self.low = np.empty(len(ACTIONS), dtype=np.float32)
        self.high = np.empty_like(self.low)

        # Arm joints (absolute radians)
        for i, jnt_name in enumerate(self._XML_JOINT_NAMES):
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
        # return np.clip(act, self.low, self.high)
        return act


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
