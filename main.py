# main.py
import asyncio
import time
import cv2
from typing import List, Optional, Tuple, Dict

from typing import List
from config.env import settings
from frame_grabber import FrameGrabber
from calibration_tool import CalibrationTool
from zone_selector import ZoneSelector
from service.detector_service import DetectorService
from overlay_renderer import OverlayRenderer
from common_model import TraceData
from common import FPSCounter
from utils import wait_for_frame, save_frame
from zlapi import ZaloAPI
from zlapi.models import ThreadType, Message


class LoiteringDetectorApp:
    """
    Encapsulates the loitering detection application with alert throttling.
    """
    # Time-to-live for a loitering alert (seconds)
    LOITER_ALERT_TTL: float = 300.0  # e.g., 5 minutes

    def __init__(self, rtsp_url: str):
        
        self.zalo_client = ZaloAPI(phone=settings.ZALO_NUMBER, password=settings.ZALO_PASSWROD, imei=settings.ZALO_IMEI, cookies=settings.ZALO_COOKIES)
        
        # Frame grabbing
        self.grabber = FrameGrabber(rtsp_url)

        # Calibration
        self.m_per_px: Optional[float] = None

        # Zone selection
        self.zone_pts = []

        # Detector and renderer
        self.detector = None
        self.renderer = None

        # FPS counter
        self.fps_counter = FPSCounter()
        self.use_detection: bool = True

        # Track alert timestamps per object ID
        self.alerted_times: Dict[int, float] = {}
        self.target_zalo_group_id: str = settings.TARGET_ZALO_GROUP_ID

    def send_loitering_alert(self, detection: TraceData, detected_file_path: str) -> None:
        """
        Trigger an alert for a loitering detection.
        Override this method to integrate email, push notifications, etc.
        """
        self.zalo_client.sendLocalImage(            
            imagePath=detected_file_path,
            thread_id=self.target_zalo_group_id,
            thread_type=ThreadType.GROUP,
            width=2560,
            height=2560,
            message=Message(text="Tên này đang lãng vãng trước camera nhà bạn!!!"),
        )
        
    def initialize(self) -> None:
        self.grabber.start()
        calib_frame = wait_for_frame(self.grabber)
        calib = CalibrationTool(calib_frame)
        self.m_per_px = calib.m_per_px

        zone_frame = wait_for_frame(self.grabber)
        zone = ZoneSelector(zone_frame)
        self.zone_pts = zone.pts

        self.detector = DetectorService(
            {
                "human": {
                    "model_path": "./model/yolov8n.pt",
                    "pose_model_path": "./model/yolov8m-pose.pt"
                },
                "custom": {"model_path": "./model/custom.pt"}
            },
            zone_pts=self.zone_pts
        )
        self.renderer = OverlayRenderer(self.zone_pts)

    def _cleanup_alerted(self, now: float) -> None:
        expired = [obj_id for obj_id, ts in self.alerted_times.items()
                   if now - ts > self.LOITER_ALERT_TTL]
        for obj_id in expired:
            del self.alerted_times[obj_id]

    def run(self) -> None:
        if self.m_per_px is None:
            self.initialize()

        while True:
            frame = self.grabber.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            fps = self.fps_counter.tick()
            now = time.time()
            self._cleanup_alerted(now)

            detections: List[TraceData] = []
            if self.use_detection and self.detector:
                frame, detections = self.detector.detect(frame, fps, self.m_per_px)

                for det in detections:
                    if det.is_loitering and det.id not in self.alerted_times:
                        file_path = save_frame(frame, ext="png")
                        self.send_loitering_alert(det, file_path)
                        self.alerted_times[det.id] = now
                        # save snapshot frame to disk, choose format by ext
                    elif not det.is_loitering:
                        self.alerted_times.pop(det.id, None)

            output = self.renderer.render(
                frame,
                fps=fps,
                detection_on=self.use_detection,
                detections=detections,
            )
            cv2.imshow("Live Feed", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.use_detection = not self.use_detection

        self.grabber.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LoiteringDetectorApp(settings.RTSP_URL)
    app.run()
