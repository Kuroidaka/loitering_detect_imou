from typing import List, Tuple, Dict, Any, Callable, Optional
import math
import threading
import numpy as np 
import cv2
import logging
import json
import time
import os
from datetime import datetime


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



def save_video(
    source: str,
    output_path: str,
    process_frame: Optional[Callable[[cv2.Mat], cv2.Mat]] = None,
    loop: bool = False
) -> None:
    """
    Read from a video source (file or stream), optionally process each frame,
    and save the result to a new video file.

    :param source:         Input path or URL for cv2.VideoCapture
    :param output_path:    Path where the output video will be written
    :param process_frame:  Optional function(frame)->frame to apply per-frame
    :param loop:           If True and source is a file, loop on EOF
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    # Fetch playback parameters
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    logging.info(f"Saving video to {output_path} @ {fps:.1f} FPS, {width}x{height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            # If looping on a file, rewind and continue
            if loop and not source.lower().startswith(('rtsp://','http://','https://')):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # Apply any per-frame processing
        out_frame = process_frame(frame) if process_frame else frame
        writer.write(out_frame)

    cap.release()
    writer.release()
    logging.info(f"Finished saving video to {output_path}")


def wait_for_frame(grabber, timeout: float = 5.0):
    """Block until grabber has delivered a non-None frame (or timeout)."""
    start = time.time()
    while True:
        frame = grabber.get_frame()
        if frame is not None:
            return frame
        if time.time() - start > timeout:
            raise RuntimeError("Timed out waiting for first frame")
        time.sleep(0.05)


def save_frame(
    frame: np.ndarray,
    save_path: Optional[str] = None,
    base_dir: str = "assets/detected",
    ext: str = "png"
) -> str:
    """
    Save a numpy image frame to disk.

    Args:
        frame: Image as an ndarray (BGR format).
        save_path: Full path (including filename) where the image will be saved. If provided,
                   the directory will be created if it doesn't exist.
        base_dir: Root directory for saving when save_path is not provided.
        ext: File extension/format (e.g., 'png', 'jpg', 'bmp'). Used only if save_path is None.

    Returns:
        The path to the saved image file.
    """
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        file_path = save_path
    else:
        date_str = datetime.now().strftime("%d-%m-%Y")
        time_str = datetime.now().strftime("%H-%M-%S")
        save_dir = os.path.join(base_dir, date_str)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{time_str}.{ext}")

    # Write the image (supports any format based on file extension)
    success = cv2.imwrite(file_path, frame)
    if not success:
        raise IOError(f"Failed to save frame to {file_path}")
    return file_path

def zone_contains(pts, x: float, y: float) -> bool:
    """
    Returns True if point (x,y) lies inside the user-defined polygon.
    """
    return cv2.pointPolygonTest(
        np.array(pts, dtype=np.int32),
        (x, y),
        False
    ) >= 0
