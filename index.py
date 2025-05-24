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
from utils import save_video
from file_reader_thread import ThreadedFileReader
def index(source: str):
    grabber = ThreadedFileReader(src=source, queue_size=5).start()

    # wait for first frame
    first = None
    while first is None:
        first = grabber.get_frame()
        time.sleep(0.01)

    # calibration
    calibrator = CalibrationTool(first)
    print("Meters per pixel:", calibrator.m_per_px)

    # zone selection
    zone = ZoneSelector(first)

    # restart video from beginning
    grabber.reset()

    # wait again for first frame post-reset
    first = None
    while first is None:
        first = grabber.get_frame()
        time.sleep(0.01)

    # now create detector, renderer, etc., and proceed exactly as before...
    detector = DetectorService({
                        "human":  {"model_path": "./model/yolov8n.pt",
                                   "pose_model_path": "./model/yolov8m-pose.pt"},
                        "custom": {"model_path": "./model/custom.pt"}
                    }, zone_pts=zone.pts)
    renderer = OverlayRenderer(zone.pts)
    fps_counter = FPSCounter()
    use_detection = True
    prev_time = time.time()

  # 6) VideoWriter setup
    nominal_fps = grabber.cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(grabber.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(grabber.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter("./assets/output/output_video_1.mp4",
                              fourcc, nominal_fps, (width, height))

    # Precompute frame interval
    frame_interval = 1.0 / nominal_fps

    try:
        while True:
            loop_start = time.time()

            # 1) Grab the latest frame (or skip if none)
            frame = grabber.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            # 2) Compute actual FPS for diagnostics
            actual_fps = fps_counter.tick()

            # 3) Detection & tracking
            detections = []
            if use_detection:
                frame, detections = detector.detect(frame, actual_fps, calibrator.m_per_px)

            # 4) Render overlays
            rendered = renderer.render(
                frame,
                fps=actual_fps,
                detection_on=use_detection,
                detections=detections,
                zone_selector=zone
            )

            # 5) Write to file and display
            out.write(rendered)
            cv2.imshow("Live Feed", rendered)

            # 6) Handle key controls
            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord(' '): pause_for_frame_draw(grabber)
            elif key == ord('d'): use_detection = not use_detection

            # 7) Sleep the remainder of the frame interval
            elapsed = time.time() - loop_start
            to_sleep = frame_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    finally:
        grabber.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # p = argparse.ArgumentParser(description="Loiter/behaviour detection on RTSP or video file")
    # p.add_argument("--source",
    #     required=True,
    #     help="RTSP URL (e.g. rtsp://…) or path to local video file"
    # )
    # args = p.parse_args()
    index("./assets/in_yard.mov")
