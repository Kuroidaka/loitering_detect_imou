import cv2
import math
import numpy as np
from typing import List, Dict, Tuple
from common_model import TraceData
from datetime import datetime
class BehaviourDetector:
    """
    Analyze tracked object traces to detect behaviors such as loitering,
    compute dwell, pause, speed statistics, and spatial metrics within a defined zone.
    Supports day/night adaptive thresholding.
    """
    # Day/Night threshold presets
    THRESHOLDS = {
        'day': {
            'dwell_time': 45.0,
            'pause_duration': 15.0,
            'movement_radius': 5.0,
            'avg_speed': 0.7,
            'path_variance': 1.2,
            'score_weights': (0.35, 0.35, 0.15, 0.15),
            'score_thresh': 0.50
        },
        'night': {
            'dwell_time': 20.0,
            'pause_duration': 8.0,
            'movement_radius': 3.0,
            'avg_speed': 0.5,
            'path_variance': 0.8,
            'score_weights': (0.40, 0.40, 0.10, 0.10),
            'score_thresh': 0.45
        }
    }

    def __init__(
        self,
        loiter_max_disp: int = 50,
        loiter_min_frames: int = 30
    ):
        """
        :param loiter_max_disp:   Pixel threshold for loiter-box detection
        :param loiter_min_frames: Frames window for loiter detection
        """
        self.loiter_max_disp   = loiter_max_disp
        self.loiter_min_frames = loiter_min_frames

    def _get_time_mode(self) -> str:
        """
        Determine 'day' or 'night' based on current hour.
        Night is before 6am or after 8pm.
        """
        h = datetime.now().hour
        return 'night' if (h < 6 or h >= 20) else 'day'

    def is_loitering(
        self,
        entry: TraceData
    ) -> bool:
        """
        Rule-based loitering with day/night thresholds: must meet at least two criteria.
        """
        mode = self._get_time_mode()
        t = self.THRESHOLDS[mode]
        conds = [
            entry.dwell_time >= t['dwell_time'],
            entry.pause_duration >= t['pause_duration'],
            entry.movement_radius <= t['movement_radius'],
            entry.avg_speed <= t['avg_speed'],
            entry.path_variance <= t['path_variance']
        ]
        return sum(conds) >= 2

    def compute_loiter_score(
        self,
        entry: TraceData
    ) -> float:
        """
        Score-based loitering metric with day/night weights.
        """
        mode = self._get_time_mode()
        cfg = self.THRESHOLDS[mode]
        wT, wR, wV, wP = cfg['score_weights']
        Tmax = cfg['dwell_time']
        Rmax = cfg['movement_radius']
        Vmin = cfg['avg_speed']
        Pmax = cfg['pause_duration']
        # normalize
        t_norm = min(entry.dwell_time / Tmax, 1.0)
        r_norm = max(1.0 - entry.movement_radius / Rmax, 0.0)
        v_norm = max(1.0 - entry.avg_speed / Vmin, 0.0)
        p_norm = min(entry.pause_duration / Pmax, 1.0)
        return wT*t_norm + wR*r_norm + wV*v_norm + wP*p_norm

    def is_loitering_score(
        self,
        entry: TraceData
    ) -> bool:
        """
        Flag loitering if composite score exceeds threshold for current day/night.
        """
        mode = self._get_time_mode()
        thresh = self.THRESHOLDS[mode]['score_thresh']
        return self.compute_loiter_score(entry) >= thresh
    
    def compute_speed_list(
        self,
        history: List[Tuple[int, int]],
        fps: float,
        m_per_px: float
    ) -> List[float]:
        """
        Compute instantaneous speeds (m/s) for each consecutive pair in history.
        """
        speeds: List[float] = []
        if len(history) < 2 or fps <= 0:
            return speeds
        dt = 1.0 / fps
        for (x0, y0), (x1, y1) in zip(history, history[1:]):
            dist_px = math.hypot(x1 - x0, y1 - y0)
            dist_m  = dist_px * m_per_px
            speeds.append(dist_m / dt)
        return speeds

    def compute_speeds(
        self,
        history: List[Tuple[int, int]],
        fps: float,
        m_per_px: float
    ) -> Tuple[float, float]:
        """
        Given a centroid history, return (avg_speed, min_speed) in m/s using compute_speed_list.
        Ensures min_speed is the smallest positive speed if available.
        """
        speeds = self.compute_speed_list(history, fps, m_per_px)
        if not speeds:
            return 0.0, 0.0
        avg_speed = sum(speeds) / len(speeds)
        # Filter out zero speeds for min_speed calculation
        positive_speeds = [s for s in speeds if s > 0]
        min_speed = min(positive_speeds) if positive_speeds else 0.0
        return avg_speed, min_speed

    def compute_dwell_time(
        self,
        history: List[Tuple[int, int]],
        fps: float,
        zone_pts: List[Tuple[int, int]]
    ) -> float:
        """
        Compute total time (seconds) the object spent within the polygon defined by zone_pts.
        """
        if not history or fps <= 0:
            return 0.0
        contour = np.array(zone_pts, dtype=np.int32)
        inside = [i for i, pt in enumerate(history)
                  if cv2.pointPolygonTest(contour, pt, False) >= 0]
        if not inside:
            return 0.0
        return (inside[-1] - inside[0]) / fps

    def compute_pause_duration(
        self,
        history: List[Tuple[int, int]],
        fps: float,
        v_min: float,
        m_per_px: float
    ) -> float:
        """
        Compute total paused time (seconds) where instantaneous speed < v_min.
        """
        speeds = self.compute_speed_list(history, fps, m_per_px)
        if not speeds:
            return 0.0
        paused_frames = sum(1 for s in speeds if s < v_min)
        return paused_frames / fps

    def compute_movement_radius(
        self,
        history: List[Tuple[int, int]],
        m_per_px: float
    ) -> float:
        """
        Compute the maximum distance (radius in meters) between any two points in history.
        r = max_{i,j} ||x_i - x_j||
        """
        if len(history) < 2:
            return 0.0
        max_dist_px = 0.0
        for i in range(len(history)):
            x0, y0 = history[i]
            for j in range(i+1, len(history)):
                x1, y1 = history[j]
                d = math.hypot(x1-x0, y1-y0)
                if d > max_dist_px:
                    max_dist_px = d
        return max_dist_px * m_per_px

    def compute_path_variance(
        self,
        history: List[Tuple[int, int]],
        m_per_px: float
    ) -> float:
        """
        Compute the variance of the person's path positions in m^2:
        sigma^2 = (1/N) sum ||x_i - mean||^2
        """
        if not history:
            return 0.0
        # convert to meters and compute mean
        pts_m = [(x*m_per_px, y*m_per_px) for x,y in history]
        mean_x = sum(p[0] for p in pts_m) / len(pts_m)
        mean_y = sum(p[1] for p in pts_m) / len(pts_m)
        var_sum = sum((p[0]-mean_x)**2 + (p[1]-mean_y)**2 for p in pts_m)
        return var_sum / len(pts_m)

    
    def analyze_behaviour(
        self,
        trace_data: List[TraceData],
        fps: float,
        zone_pts: List[Tuple[int, int]],
        v_min: float,
        m_per_px: float
    ) -> List[dict]:
        """
        Annotate each TraceData with dwell_time, pause_duration, loitering,
        and speed statistics.
        :param trace_data:  list of TraceData models
        :param fps:         video frame rate
        :param zone_pts:    polygon vertices defining the zone
        :param v_min:       speed threshold (m/s) below which counts as pause
        :param m_per_px:    meter-per-pixel calibration
        """
        annotated: List[dict] = []
        for entry in trace_data:
            history = entry.trace
            dwell  = self.compute_dwell_time(history, fps, zone_pts)
            pause  = self.compute_pause_duration(history, fps, v_min, m_per_px)
            # loiter = self.is_loitering(history)
            avg_spd, min_spd = self.compute_speeds(history, fps, m_per_px)
            
            radius     = self.compute_movement_radius(history, m_per_px)
            variance   = self.compute_path_variance(history, m_per_px)

            updated = entry.model_copy(update={
                'dwell_time': dwell,
                'pause_duration': pause,
                'avg_speed': avg_spd,
                'min_speed': min_spd,
                'movement_radius': radius,
                'path_variance': variance
            })

            # **NEW**: apply the two decision functions
            rule_flag  = self.is_loitering(updated)
            score_flag = self.is_loitering_score(updated)

            updated = updated.model_copy(update={
                'rule_loiter': rule_flag,
                'score_loiter': score_flag
            })
            annotated.append(updated)
        return annotated
