import cv2
import numpy as np
from utils import draw_dashed_polygon
from common_model import TraceData
from typing import List


class OverlayRenderer:
    """
    Draws zones, FPS, status, and trespass alerts onto frames.
    """
    def __init__(self, zone_pts: list[tuple[int,int]]):
        self.zone_pts = zone_pts

    def render(
        self,
        frame: np.ndarray,
        fps: float,
        detection_on: bool,
        detections: List[TraceData],
        zone_selector
    ) -> np.ndarray:
        # 1) translucent fill
        overlay = frame.copy()
        cv2.fillPoly(
            overlay,
            [np.array(self.zone_pts, dtype=np.int32)],
            color=(255,255,255)
        )
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # 2) dashed outline
        draw_dashed_polygon(
            frame,
            self.zone_pts,
            color=(0,128,0),
            thickness=2,
            dash_length=15
        )

        # 3) trespass alerts
        for det in detections:
            bbox = det.bbox
            if bbox:
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if zone_selector.contains(cx, cy):
                    cv2.putText(
                        frame,
                        "TRESPASS!",
                        (int(cx), int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,255),
                        2
                    )

        # 4) FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        # 5) detection status
        status = "Detection: ON" if detection_on else "Detection: OFF"
        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        return frame
