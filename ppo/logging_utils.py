from stable_baselines3.common.callbacks import BaseCallback


class InfoStatsCallback(BaseCallback):
    """
    Callback to log the sparse reward and potential to TensorBoard.
    """

    def __init__(self):
        super().__init__(verbose=0)

    def _on_step(self):
        for info in self.locals["infos"]:
            if "sparse_r" in info:
                self.logger.record("train/sparse_r", info["sparse_r"])
            if "potential" in info:
                self.logger.record("train/potential", info["potential"])
        return True
