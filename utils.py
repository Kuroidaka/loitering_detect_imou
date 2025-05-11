import threading
from typing import List, Tuple
import math
import numpy as np 
import cv2

def run_in_thread(target_func, *args, **kwargs):
    """
    Runs any target_func with given args/kwargs in a daemon thread.

    :param target_func: The function to run in a background thread.
    :param args: Positional args for the function.
    :param kwargs: Keyword args for the function.
    :return: Thread object (optional: you can join or monitor it)
    """
    thread = threading.Thread(target=target_func, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    return thread



def compute_speeds(
    trace: List[Tuple[int,int]],
    fps: float,
    m_per_px: float
) -> Tuple[float, float]:
    """
    Given trace=[(x0,y0),(x1,y1),…], return
      - avg_speed: mean instantaneous speed in m/s
      - min_speed: smallest instantaneous speed in m/s

    :param fps:         Frames per second of the video feed.
    :param m_per_px:    Physical meters represented by one pixel.
    """
    if len(trace) < 2:
        return 0.0, 0.0

    dt = 1.0 / fps
    speeds = []
    for (x0, y0), (x1, y1) in zip(trace, trace[1:]):
        # 1) pixel distance
        dist_px = math.hypot(x1 - x0, y1 - y0)
        # 2) convert to meters
        dist_m  = dist_px * m_per_px
        # 3) speed in m/s
        speeds.append(dist_m / dt)

    avg_speed = sum(speeds) / len(speeds)
    min_speed = min(speeds)
    return avg_speed, min_speed



def draw_dashed_polygon(img, pts, color, thickness=2, dash_length=20):
    """
    Draws a closed, dashed polygon on img.
    - pts: list of (x,y) tuples
    - dash_length: approximate length of each drawn segment (in px).
    """
    pts = [np.array(p, dtype=np.int32) for p in pts]
    n = len(pts)
    for i in range(n):
        p1, p2 = pts[i], pts[(i+1)%n]
        # vector from p1 to p2
        v = p2 - p1
        dist = int(np.hypot(*(v)))
        # how many full dashes fit
        num_dashes = max(dist // dash_length, 1)
        # draw alternating segments
        for j in range(num_dashes):
            start = p1 + (v * (j / num_dashes))
            end   = p1 + (v * ((j + 0.5) / num_dashes))
            cv2.line(
                img,
                tuple(start.astype(int)),
                tuple(end.astype(int)),
                color,
                thickness,
            )