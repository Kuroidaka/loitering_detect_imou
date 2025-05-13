import cv2
import time
import argparse
from typing import List

from local_frame_grabber import LocalFrameGrabber
from calibration_tool import CalibrationTool
from zone_selector import ZoneSelector
from service.detector_service import DetectorService
from overlay_renderer import OverlayRenderer
from common_model import TraceData
from main import pause_for_frame_draw, FPSCounter

def index(source: str):
    # 1) Start grabbing frames (RTSP or file)
    grabber = LocalFrameGrabber(source)

    # Step 1: Read the first frame
    first_frame = grabber.get_frame()

    # Step 2: Calibrate using the first frame (UI loop happens here)
    calibrator = CalibrationTool(first_frame)
    print("Meters per pixel:", calibrator.m_per_px)

    # 3) Zone selection
    zone   = ZoneSelector(first_frame)
    grabber.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # 4) Detector service (you can pass model paths via settings or config)
    detector = DetectorService({
        "human": {
            "model_path": "./model/yolov8n.pt",
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
    prev_time = time.time()

    # read nominal FPS *once*, and compute display delay from that
    nominal_fps = grabber.cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    
    try:
        # 6) Main loop
        while True:
            frame = grabber.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            
               # Compute instantaneous FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = curr_time
            
            # tick for diagnostics only
            actual_fps = fps_counter.tick()
            
            # run detection if enabled
            detections: List[TraceData] = []
            if use_detection:
                frame, detections = detector.detect(frame, actual_fps, calibrator.m_per_px)

            # render all overlays (zone, FPS, warnings, etc.)
            out = renderer.render(
                frame,
                fps=actual_fps,
                detection_on=use_detection,
                detections=detections,
                zone_selector=zone
            )
            
            frame_delay_ms = int(1000 / nominal_fps)
            cv2.imshow("Live Feed", out)
            key = cv2.waitKey(frame_delay_ms) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                pause_for_frame_draw(grabber)
            elif key == ord('d'):
                use_detection = not use_detection
    finally:
        # cleanup
        grabber.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # p = argparse.ArgumentParser(description="Loiter/behaviour detection on RTSP or video file")
    # p.add_argument("--source",
    #     required=True,
    #     help="RTSP URL (e.g. rtsp://…) or path to local video file"
    # )
    # args = p.parse_args()
    index("./assets/video_test.MOV")
