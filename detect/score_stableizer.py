import time
from collections import deque
from typing import Dict, Deque

class ScoreStabilizer:
    """
    Stabilize raw loitering scores per object ID using:
      1. EMA smoothing
      2. Hysteresis thresholds
      3. Debounce minimum duration
      4. Sliding-window majority vote
    """
    def __init__(
        self,
        alpha: float = 0.3,
        on_thresh: float = 0.45,
        off_thresh: float = 0.35,
        debounce_sec: float = 2.0,
        window_size: int = 5,
        vote_thresh: int = 3
    ):
        # EMA smoothing factor
        self.alpha = alpha
        # Hysteresis thresholds
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        # Debounce time in seconds
        self.debounce_sec = debounce_sec
        # Sliding window parameters
        self.window_size = window_size
        self.vote_thresh = vote_thresh

        # Per-ID state
        self._hat_scores: Dict[int, float] = {}           # smoothed score
        self._prov_state: Dict[int, bool] = {}            # provisional boolean
        self._last_change: Dict[int, float] = {}          # timestamp of prov flip
        self._confirmed: Dict[int, bool] = {}             # last confirmed state
        self._history: Dict[int, Deque[bool]] = {}        # history deque for final states

    def update(
        self,
        object_id: int,
        raw_score: float,
        timestamp: float = None
    ) -> bool:
        """
        Update and return stabilized loitering state for the given ID.

        :param object_id: Unique identifier for tracked object
        :param raw_score: Newly computed raw loiter score (0.0–1.0)
        :param timestamp: Current time; defaults to time.time()
        :return: Final stabilized boolean state
        """
        now = timestamp if timestamp is not None else time.time()

        # 1) EMA smoothing
        prev_hat = self._hat_scores.get(object_id, raw_score)
        hat = self.alpha * raw_score + (1 - self.alpha) * prev_hat
        self._hat_scores[object_id] = hat

        # 2) Hysteresis provisional state
        prev_prov = self._prov_state.get(object_id, False)
        if hat >= self.on_thresh:
            prov = True
        elif hat <= self.off_thresh:
            prov = False
        else:
            prov = prev_prov

        # record time of provisional change
        if prov != prev_prov:
            self._last_change[object_id] = now
        change_time = self._last_change.get(object_id, now)
        self._prov_state[object_id] = prov

        # 3) Debounce: only confirm after provisional held long enough
        prev_conf = self._confirmed.get(object_id, False)
        if prov != prev_conf:
            if now - change_time >= self.debounce_sec:
                conf = prov
            else:
                conf = prev_conf
        else:
            conf = prev_conf
        self._confirmed[object_id] = conf

        # 4) Sliding-window majority vote
        hist = self._history.get(object_id)
        if hist is None:
            hist = deque(maxlen=self.window_size)
            # initialize with current confirmed state
            hist.extend([conf] * self.window_size)
            self._history[object_id] = hist
        hist.append(conf)

        # majority vote
        true_count = sum(hist)
        final = true_count >= self.vote_thresh

        return final
