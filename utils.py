from typing import List, Tuple, Dict, Any
import math
import threading
import numpy as np 
import cv2
import logging
import json

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
            
def save_dicts_to_json(
    data: List[Dict[str, Any]],
    path: str = 'data.json'
) -> None:
    """
    Save a list of dictionaries to a JSON file.

    :param data: List of JSON-serializable dictionaries
    :param path: Output file path
    """
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Data successfully written to {path}")
    except Exception as e:
        logging.error(f"Failed to write JSON file {path}: {e}")
        # Re-raise to let caller handle if necessary
        raise
