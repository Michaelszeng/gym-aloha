from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder


class QuietVecVideoRecorder(VecVideoRecorder):
    """A VecVideoRecorder that disables MoviePy's verbose progress bar output."""

    def _stop_recording(self) -> None:  # type: ignore[override]
        """Stop current recording and save the video without printing progress bars."""
        assert self.recording, "_stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            # No frames collected; nothing to save.
            return

        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
        clip.write_videofile(self.video_path, logger=None)

        # Reset state
        self.recorded_frames = []
        self.recording = False
