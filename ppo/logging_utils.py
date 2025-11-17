import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import wandb


class WandbCallback(BaseCallback):
    """Callback for logging to Weights & Biases."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Log metrics at the end of each rollout."""
        if wandb.run is not None:
            # Log all metrics from the logger
            for key, value in self.logger.name_to_value.items():
                wandb.log({key: value}, step=self.num_timesteps)


class InfoStatsCallback(BaseCallback):
    """
    Callback to log important metrics from the info dictionary to TensorBoard and wandb.
    
    NOTE: logging is done per-step (averaged over all parallel environments).
    """

    def __init__(self):
        super().__init__(verbose=0)

    def _on_step(self):
        infos = self.locals["infos"]

        # Track which metrics we've seen this step (for averaging across parallel envs)
        metrics = {}

        for info in infos:
            # === Core Reward Components ===
            for key in [
                "dense_r",
                "reach_peg_r",
                "reach_socket_r",
                "ee_still_r",
                "arm_resting_r",
                "grasp_r",
                "success_r",
            ]:
                if key in info:
                    metrics.setdefault(key, []).append(info[key])

            # === Collision Metrics ===
            for key in [
                "step_no_collision_r",
                "cumulative_collision_r",
                "scaled_collision_force",
                "cumulative_collision_force",
            ]:
                if key in info:
                    metrics.setdefault(key, []).append(info[key])

            # === Phase-specific Rewards ===
            for key in [
                "not_grasped_r",
                "grasped_r",
                "right_over_peg_r",
                "left_over_socket_r",
                "align_pos_r",
                "align_orient_r",
            ]:
                if key in info:
                    metrics.setdefault(key, []).append(info[key])

            # === Distance Metrics ===
            for key in ["dist_right_to_peg", "dist_left_to_socket", "peg_socket_dist", "alignment_error"]:
                if key in info:
                    metrics.setdefault(key, []).append(info[key])

            # === Grasp State (boolean - use majority voting) ===
            for key in ["is_grasped_left", "is_grasped_right", "is_grasped_both"]:
                if key in info:
                    metrics.setdefault(key, []).append(float(info[key]))

            # === Episode Tracking ===
            for key in ["episode_step", "remaining_steps"]:
                if key in info:
                    metrics.setdefault(key, []).append(info[key])

        # Log averaged metrics to tensorboard/wandb
        for key, values in metrics.items():
            if values:
                avg_value = np.mean(values)
                self.logger.record(f"train/{key}", avg_value)

        return True
