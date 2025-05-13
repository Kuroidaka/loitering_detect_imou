import cv2
import math
import numpy as np
from typing import List, Dict, Tuple, Any
from common_model import TraceData
from datetime import datetime
import time
from dataclasses import replace

from utils import save_dicts_to_json
from .score_stableizer import ScoreStabilizer

class BehaviourDetector:
    """
    Analyze tracked object traces to detect behaviors such as loitering,
    compute dwell, pause, speed statistics, and spatial metrics within a defined zone.
    Supports day/night adaptive thresholding.
    """

    # Day/Night threshold presets
    THRESHOLDS = {
        'day': {
            'dwell_time': 180.0, # 
            'pause_duration': 120.0, # s
            'movement_radius': 2.0, # m
            'avg_speed': 0.3,
            # 'path_variance': 1.2,
            'score_weights': (0.35, 0.35, 0.15, 0.15), # dwell_time, movement_radius, avg_speed, pause_duration
            'score_thresh': 0.50
        },
        'night': {
            'dwell_time': 20.0,
            'pause_duration': 8.0,
            'movement_radius': 3.0,
            'avg_speed': 0.5,
            # 'path_variance': 0.8,
            'score_weights': (0.40, 0.40, 0.10, 0.10),
            'score_thresh': 0.45
        }
    }

    def __init__(
        self
    ):
        self.stabilizer = ScoreStabilizer()

    def _get_time_mode(self) -> str:
        """
        Determine 'day' or 'night' based on current hour.
        Night is before 6am or after 8pm.
        """
        h = datetime.now().hour
        return 'night' if (h < 6 or h >= 20) else 'day'

    def is_loitering_by_rule(
        self,
        dwell_time: float,
        movement_radius: float,
        avg_speed: float,
        pause_duration: float,
        # path_variance: float
    ) -> bool:
        """
        Rule-based loitering with day/night thresholds: must meet at least two criteria.
        """
        mode = self._get_time_mode()
        t = self.THRESHOLDS[mode]
        conds = [
            dwell_time >= t['dwell_time'],
            pause_duration >= t['pause_duration'],
            movement_radius <= t['movement_radius'],
            avg_speed <= t['avg_speed'],
            # path_variance <= t['path_variance']
        ]
        return sum(conds) >= 2

    def compute_raw_loiter_score(
        self,
        dwell_time: float,
        movement_radius: float,
        avg_speed: float,
        pause_duration: float
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
        t_norm = min(dwell_time / Tmax, 1.0)
        r_norm = max(1.0 - movement_radius / Rmax, 0.0)
        v_norm = max(1.0 - avg_speed / Vmin, 0.0)
        p_norm = min(pause_duration / Pmax, 1.0)
        return wT*t_norm + wR*r_norm + wV*v_norm + wP*p_norm

    def check_loitering(
        self,
        id: int,
        raw_loiter_score: float
    ) -> bool:
        now = time.time()

        final_loiter = self.stabilizer.update(
            id, raw_loiter_score, timestamp=now
        )
        return final_loiter

    # def check_loitering(self, raw_loiter_score: float) -> bool:
    #     """
    #     Flag loitering if composite score exceeds threshold for current day/night.
    #     """
    #     mode = self._get_time_mode()
    #     thresh = self.THRESHOLDS[mode]['score_thresh']
    #     return raw_loiter_score >= thresh
    
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
 
    def compute_dwell_time(self, history, fps, zone_pts, max_gap=2):
        if not history or fps <= 0:
            return 0.0

        contour = np.array(zone_pts, dtype=np.int32)
        # Boolean list of “inside zone” per frame
        flags = np.array([cv2.pointPolygonTest(contour, pt, False) >= 0
                        for pt in history], dtype=int)

        total = 0
        run = 0
        gap = 0

        for f in flags:
            if f:
                if gap > 0:
                    run += gap  # bridge short gaps
                    gap = 0
                run += 1
            else:
                if run > 0:
                    gap += 1
                    if gap > max_gap:
                        total += run
                        run = 0
                        gap = 0

        total += run
        return total / float(fps)

    def compute_pause_duration(
        self,
        history: List[Tuple[int, int]],
        fps: float,
        v_min: float,
        m_per_px: float,
        zone_pts: List[Tuple[int, int]]
    ) -> float:
        """
        Compute total paused time (seconds) where instantaneous speed < v_min
        AND the object is inside the polygon defined by zone_pts.
        """
        # 1) Guard clauses
        if not history or fps <= 0:
            return 0.0

        # 2) Build the polygon contour once
        contour = np.array(zone_pts, dtype=np.int32)

        # 3) Precompute inside/outside for each history point
        inside_flags = [
            cv2.pointPolygonTest(contour, pt, False) >= 0
            for pt in history
        ]
        # If it never enters the zone, no paused time to count
        if not any(inside_flags):
            return 0.0

        # 4) Compute speeds (len = len(history)-1)
        speeds = self.compute_speed_list(history, fps, m_per_px)
        if not speeds:
            return 0.0

        # 5) Count only those low-speed segments fully inside the zone
        paused_frames = 0
        for i, s in enumerate(speeds):
            # segment from history[i] -> history[i+1]
            if (s < v_min) and inside_flags[i] and inside_flags[i+1]:
                paused_frames += 1

        # 6) Convert frame count to seconds
        return paused_frames / float(fps)


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

    # def compute_path_variance(
    #     self,
    #     history: List[Tuple[int, int]],
    #     m_per_px: float
    # ) -> float:
    #     """
    #     Compute the variance of the person's path positions in m^2:
    #     sigma^2 = (1/N) sum ||x_i - mean||^2
    #     """
    #     if not history:
    #         return 0.0
    #     # convert to meters and compute mean
    #     pts_m = [(x*m_per_px, y*m_per_px) for x,y in history]
    #     mean_x = sum(p[0] for p in pts_m) / len(pts_m)
    #     mean_y = sum(p[1] for p in pts_m) / len(pts_m)
    #     var_sum = sum((p[0]-mean_x)**2 + (p[1]-mean_y)**2 for p in pts_m)
    #     return var_sum / len(pts_m)

    
    def analyze_behaviour(
        self,
        entry: TraceData,
        fps: float,
        zone_pts: List[Tuple[int, int]],
        v_min: float,
        m_per_px: float
    ) -> TraceData:
        """
        Compute all metrics for a single TraceData entry, return updated model.
        """
        history = entry.trace
        dwell     = self.compute_dwell_time(history, fps, zone_pts)
        pause     = self.compute_pause_duration(history, fps, v_min, m_per_px, zone_pts)
        avg_spd, min_spd = self.compute_speeds(history, fps, m_per_px)
        radius    = self.compute_movement_radius(history, m_per_px)
        # variance  = self.compute_path_variance(history, m_per_px)
        raw_loiter_score =self.compute_raw_loiter_score(
            dwell_time=dwell,
            movement_radius=radius,
            avg_speed=avg_spd,
            pause_duration=pause
        )
        
        is_loitering = self.check_loitering(
            id=entry.id,
            raw_loiter_score=raw_loiter_score
        )
        
        rule_flag  = self.is_loitering_by_rule(
            dwell_time=dwell,
            movement_radius=radius,
            avg_speed=avg_spd,
            pause_duration=pause,
            # path_variance=variance                                       
        )
        # first update numeric metrics
        return replace(
            entry,
            dwell_time=dwell,
            pause_duration=pause,
            avg_speed=avg_spd,
            min_speed=min_spd,
            movement_radius=radius,
            # path_variance=variance,
            raw_loiter_score=raw_loiter_score,
            rule_loiter=rule_flag,
            is_loitering=is_loitering
        )
