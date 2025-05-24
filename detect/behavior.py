import cv2
import math
import numpy as np
from typing import List, Dict, Tuple, Any
from common_model import TraceData
from datetime import datetime
import time
from dataclasses import replace
from math import acos

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
            'dwell_time':    180.0,   # seconds
            'pause_duration':120.0,   # seconds
            'curl_count':       3,    # minimum “curls” to count
            'avg_speed':      0.5,    # m/s
            'score_weights': (0.35, 0.35, 0.15, 0.15),  # (dwell, curl, speed, pause)
            'score_thresh':   0.50
        },
        'night': {
            'dwell_time':    20.0, 
            'pause_duration': 8.0, 
            'curl_count':       3, 
            'avg_speed':      0.2,
            'score_weights': (0.40, 0.40, 0.10, 0.10), # (dwell, curl, speed, pause)
            'score_thresh':   0.45
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
        return "night"
        # h = datetime.now().hour
        # return 'night' if (h < 6 or h >= 20) else 'day'

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
        curl_count: int,
        avg_speed: float,
        pause_duration: float
    ) -> float:
        """
        Score-based: normalize each metric, weight by importance, sum to S.
        """
        mode = self._get_time_mode()
        cfg = self.THRESHOLDS[mode]
        wT, wC, wV, wP = cfg['score_weights']
        Tmax, Cmax = cfg['dwell_time'], cfg['curl_count']
        Vmin, Pmax = cfg['avg_speed'], cfg['pause_duration']

        t_norm = min(dwell_time / Tmax, 1.0)
        c_norm = min(curl_count  / Cmax, 1.0)
        v_norm = max(1.0 - avg_speed / Vmin, 0.0)
        p_norm = min(pause_duration / Pmax, 1.0)

        return wT*t_norm + wC*c_norm + wV*v_norm + wP*p_norm
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

    def compute_vertical_curl_count(
        self,
        history: List[Tuple[int, int]],
        min_peak_distance: int = 10,
        min_peak_separation: int = 5
    ) -> int:
        """
        Count vertical curls based on peak detection in the y-axis movement.

        - min_peak_distance: minimum change in Y to consider a peak (to reduce noise)
        - min_peak_separation: minimum number of frames between peaks (to avoid false positives)
        """
        if len(history) < 3:
            return 0

        ys = [y for _, y in history]
        peaks = []
        state = None  # None, "up", or "down"
        last_peak_idx = -min_peak_separation

        for i in range(1, len(ys) - 1):
            prev, curr, next = ys[i - 1], ys[i], ys[i + 1]
            if curr < prev and curr < next:  # local minimum (arm down)
                if state != "down" and (i - last_peak_idx) >= min_peak_separation:
                    peaks.append(("min", i))
                    last_peak_idx = i
                    state = "down"
            elif curr > prev and curr > next:  # local maximum (arm up)
                if state != "up" and (i - last_peak_idx) >= min_peak_separation:
                    peaks.append(("max", i))
                    last_peak_idx = i
                    state = "up"

        # Count each (min → max → min) or (max → min → max) as 1 curl
        curl_count = 0
        for i in range(1, len(peaks) - 1, 2):
            a, b, c = peaks[i - 1], peaks[i], peaks[i + 1]
            if a[0] != b[0] and b[0] != c[0]:
                dy1 = abs(ys[a[1]] - ys[b[1]])
                dy2 = abs(ys[b[1]] - ys[c[1]])
                if dy1 >= min_peak_distance and dy2 >= min_peak_distance:
                    curl_count += 1

        return curl_count



    def count_non_forward_events(
        self,
        trace: List[Tuple[int,int]],
        angle_threshold: float = 30.0,
        recovery_length: int = 15
    ) -> int:
        """
        Count how many times the tracked point 'turns' by more than angle_threshold
        degrees, requiring at least `recovery_length` straight segments
        before counting a new event.
        """
        # 1. Build segment vectors
        vecs = [
            (x2-x1, y2-y1)
            for (x1,y1),(x2,y2) in zip(trace, trace[1:])
            if (x2-x1, y2-y1) != (0,0)
        ]
        if len(vecs) < 2:
            return 0

        # 2. Compute angles between consecutive vectors
        angles = []
        for (vx1,vy1), (vx2,vy2) in zip(vecs, vecs[1:]):
            dot = vx1*vx2 + vy1*vy2
            mag1 = math.hypot(vx1, vy1)
            mag2 = math.hypot(vx2, vy2)
            # clamp for numerical stability
            cosθ = max(-1, min(1, dot/(mag1*mag2)))
            θ = math.degrees(math.acos(cosθ))
            angles.append(θ)

        # 3. Flag turn vs straight segments
        is_turn = [θ > angle_threshold for θ in angles]

        # 4. Walk through and count with recovery
        events = 0
        i = 0
        n = len(is_turn)
        while i < n:
            # look for a rise edge: straight→turn
            if is_turn[i] and (i == 0 or not is_turn[i-1]):
                events += 1
                # skip this turn region
                while i < n and is_turn[i]:
                    i += 1
                # now require `recovery_length` straights before next possible event
                straight_count = 0
                while i < n and straight_count < recovery_length:
                    if not is_turn[i]:
                        straight_count += 1
                    else:
                        # if another turn pops up too soon, we consider it the same event
                        straight_count = 0
                    i += 1
                # now ready to look for a new event
            else:
                i += 1

        return events
    
    def compute_curl_count(
        self,
        history: List[Tuple[int, int]],
        min_turn_frames: int = 3
    ) -> int:
        """
        Count how many times the tracked path 'curls' (turns one way then back).
        - history: list of (x,y) points
        - min_turn_frames: minimum consecutive frames to treat as a valid turn segment
        """
        
        n = len(history)
        if n < 3:
            return 0

        # 1) Compute signed 'turn' for each triplet
        signs = []
        for i in range(n - 2):
            (x0, y0), (x1, y1), (x2, y2) = history[i], history[i+1], history[i+2]
            cross = (x1-x0)*(y2-y1) - (y1-y0)*(x2-x1)
            if cross > 0:
                signs.append( 1)   # left turn
            elif cross < 0:
                signs.append(-1)   # right turn
            else:
                signs.append( 0)   # straight / noise

        # 2) Compress into segments of the same sign (ignore zeros)
        segments = []
        curr_sign = signs[0]
        length = 1
        for s in signs[1:]:
            if s == curr_sign:
                length += 1
            else:
                if curr_sign != 0 and length >= min_turn_frames:
                    segments.append(curr_sign)
                curr_sign = s
                length = 1
        # last segment
        if curr_sign != 0 and length >= min_turn_frames:
            segments.append(curr_sign)

        # 3) Count sign-flips → each flip is half a curl
        flips = sum(1 for a, b in zip(segments, segments[1:]) if a != b)
        # full curls ≈ flips//2
        return flips // 2
        
    def _filter_stationary(self, history, min_move=2):
        filtered = []
        last = None
        for pt in history:
            if last is None or (abs(pt[0]-last[0]) > min_move or abs(pt[1]-last[1]) > min_move):
                filtered.append(pt)
                last = pt
        return filtered
    
    def _smooth_history(self, history, window_size=3):
        smoothed = []
        for i in range(len(history)):
            x_vals = [pt[0] for pt in history[max(0, i-window_size):min(len(history), i+window_size+1)]]
            y_vals = [pt[1] for pt in history[max(0, i-window_size):min(len(history), i+window_size+1)]]
            smoothed.append((sum(x_vals)//len(x_vals), sum(y_vals)//len(y_vals)))
        return smoothed


    def analyze_behaviour(
        self,
        entry: TraceData,
        fps: float,
        zone_pts: List[Tuple[int, int]],
        v_min: float,
        m_per_px: float
    ) -> TraceData:
        # 1) only track people
        if entry.class_name != 'person':
            return entry

        # 2) compute zone-based dwell and bail if never visited
        history = entry.trace

        
        dwell = self.compute_dwell_time(history, fps, zone_pts)
        if dwell == 0.0:
            return replace(entry,
                dwell_time=0.0,
                pause_duration=0.0,
                avg_speed=0.0,
                min_speed=0.0,
                curl_count=0,
                raw_loiter_score=0.0,
                rule_loiter=False,
                is_loitering=False
            )

        # 3) compute remaining metrics
        # history = self._filter_stationary(history, min_move=1)
        # history = self._smooth_history(history, window_size=2)
        
        pause      = self.compute_pause_duration(history, fps, v_min, m_per_px, zone_pts)
        avg_spd, min_spd = self.compute_speeds(history, fps, m_per_px)
        curl_count = self.count_non_forward_events(history)

        # 4) compute score & flags
        raw_score  = self.compute_raw_loiter_score(dwell, curl_count, avg_spd, pause)
        # rule_flag  = self.is_loitering_by_rule(dwell, curl_count, avg_spd, pause)
        final_flag = self.check_loitering(entry.id, raw_score)

        # 5) single replace with all updated fields
        return replace(entry,
            dwell_time       = dwell,
            pause_duration   = pause,
            avg_speed        = avg_spd,
            min_speed        = min_spd,
            curl_count       = curl_count,
            raw_loiter_score = raw_score,
            # rule_loiter      = rule_flag,
            is_loitering     = final_flag,
            trace=entry.trace
        )