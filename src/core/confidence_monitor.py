import collections
from typing import Deque


class ConfidenceMonitor:
    """Monitor token probabilities to detect a sequence lock.

    The monitor keeps a sliding window of recent token confidences and reports
    when a sustained high confidence streak is observed. This behaviour is
    inspired by the ConfGuard defence which flags abnormally confident
    generation of a target sequence.
    """

    def __init__(self, window_size: int = 5, threshold: float = 0.9, min_streak: int = 3) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.min_streak = min_streak
        self.window: Deque[float] = collections.deque(maxlen=window_size)

    def update(self, prob: float) -> None:
        """Update the monitor with a new token probability."""
        self.window.append(prob)

    def is_sequence_lock(self) -> bool:
        """Return ``True`` if a high-confidence streak is detected."""
        if len(self.window) < self.window_size:
            return False

        streak = 0
        for p in self.window:
            if p >= self.threshold:
                streak += 1
                if streak >= self.min_streak:
                    return True
            else:
                streak = 0
        return False

    def reset(self) -> None:
        """Clear the monitor state."""
        self.window.clear()
