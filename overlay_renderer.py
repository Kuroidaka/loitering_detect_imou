import cv2
import numpy as np
from typing import List
from utils import draw_dashed_polygon
from common_model import TraceData

class OverlayRenderer:
    """
    Draws zones, FPS, status, and trespass/loiter alerts onto frames.
    """
    def __init__(self, zone_pts: List[tuple[int,int]]):
        self.zone_pts = zone_pts

    def render(
        self,
        frame: np.ndarray,
        fps: float,
        detection_on: bool,
        detections: List[TraceData],
        zone_selector
    ) -> np.ndarray:
        # determine if any loiter alert exists
        zone_alert = any(det.is_loitering for det in detections)

        # 1) zone overlay (color adapts to alert)
        self._draw_zone_overlay(frame, alert=zone_alert)

        # 2) detection boxes
        self._draw_detection_boxes(frame, detections, alert=zone_alert)

        # 3) trespass alerts
        self._draw_trespass_alerts(frame, detections, zone_selector)

        # 4) FPS
        self._draw_fps(frame, fps)

        # 5) detection status
        self._draw_status(frame, detection_on)

        return frame

    def _draw_zone_overlay(
        self,
        frame: np.ndarray,
        alert: bool = False
    ) -> None:
        """
        Draw translucent fill and dashed outline for the defined zone.
        If alert is True, outline in red; otherwise in green.
        """
        # translucent fill
        overlay = frame.copy()
        cv2.fillPoly(
            overlay,
            [np.array(self.zone_pts, dtype=np.int32)],
            color=(255, 255, 255)
        )
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # outline color based on alert flag
        color = (0, 0, 255) if alert else (0, 128, 0)
        draw_dashed_polygon(
            frame,
            self.zone_pts,
            color=color,
            thickness=2,
            dash_length=15
        )
    def _draw_detection_boxes(
        self,
        frame: np.ndarray,
        detections: List[TraceData],
        alert = False
    ) -> None:
        """
        Draw detection bounding boxes: green for normal, dashed red for loiter.
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            color = (0, 0, 255) if alert else (0, 128, 0)
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color=color,
                thickness=2
            )

    def _draw_trespass_alerts(
        self,
        frame: np.ndarray,
        detections: List[TraceData],
        zone_selector
    ) -> None:
        """
        Overlay "TRESPASS!" text at centroid for detections inside the zone.
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
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

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """
        Draw the current FPS in the top-left corner.
        """
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    def _draw_status(self, frame: np.ndarray, detection_on: bool) -> None:
        """
        Draw the detection ON/OFF status in the top-left corner.
        """
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
