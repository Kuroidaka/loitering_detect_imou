# main.py
import time
import cv2

from config.env import settings
from frame_grabber import FrameGrabber
from calibration_tool import CalibrationTool
from zone_selector import ZoneSelector
from service.detector_service import DetectorService
from overlay_renderer import OverlayRenderer
from common_model import TraceData
from typing import List

class FPSCounter:
    """Simple FPS tracker."""
    def __init__(self):
        self.prev = time.time()
        self.fps = 0.0

    def tick(self):
        now = time.time()
        dt = now - self.prev
        self.prev = now
        if dt > 0:
            self.fps = 1.0 / dt
        return self.fps

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

def pause_for_frame_draw(grabber, timeout: float = 5.0):
    """
    Pause the grabber thread and grab a single frame from its open VideoCapture.
    """
    # 1) Pause background grabbing (but keep cap open)
    grabber.pause()

    cap = grabber.cap
    if cap is None:
        raise RuntimeError("VideoCapture has been closed!")

    # 2) Rewind to start if it's a file
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 3) Read one frame (with timeout)
    start = time.time()
    while True:
        ret, frame = cap.read()
        if ret:
            break
        if time.time() - start > timeout:
            raise RuntimeError("Timed out waiting for first frame from source")
        time.sleep(0.05)

    # 4) Resume background grabbing
    grabber.start()

    return frame



def main():
    # 1) Start grabbing frames
    grabber = FrameGrabber(settings.RTSP_URL)
    grabber.start()

    # 2) Calibration (m/px)
    calib_frame = wait_for_frame(grabber)
    calib = CalibrationTool(calib_frame)
    m_per_px = calib.m_per_px

    # 3) Zone selection
    zone_frame = wait_for_frame(grabber)
    zone   = ZoneSelector(zone_frame)

    # 4) Detector service (you can pass model paths via settings or config)
    detector = DetectorService({
        "human": {
            "model_path": "./model/yolov8s.pt",
            "pose_model_path": "./model/yolov8m-pose.pt"
        },
        "custom": {
            "model_path": "./model/custom.pt"
        }
    }, zone_pts=zone.pts)

    # 5) Renderer for overlays
    renderer = OverlayRenderer(zone.pts)

    fps_counter   = FPSCounter()
    use_detection = True

    # 6) Main loop
    while True:
        frame = grabber.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        fps = fps_counter.tick()

        # run detection if enabled
        detections:List[TraceData] = []
        if use_detection:
            frame, detections = detector.detect(frame, fps, m_per_px)

        # render all overlays (zone, FPS, trespass warnings, etc.)
        out = renderer.render(
            frame,
            fps=fps,
            detection_on=use_detection,
            detections=detections,
            zone_selector=zone
        )

        cv2.imshow("Live Feed", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            use_detection = not use_detection

    # cleanup
    grabber.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
