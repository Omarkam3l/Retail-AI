class FrameSampler:
    """Handles frame rate downsampling dynamically using time intervals or step metrics."""

    def __init__(self, source_fps: float, target_fps: float) -> None:
        self._source_fps = source_fps
        self._target_fps = min(target_fps, source_fps)
        self._step = source_fps / self._target_fps
        self._counter = self._step - 1.0

    def should_keep(self) -> bool:
        """Determines if the next frame should be processed or skipped."""
        self._counter += 1.0
        if self._counter >= self._step:
            self._counter -= self._step
            return True
        return False

    def get_target_fps(self) -> float:
        return self._target_fps
