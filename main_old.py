import cv2
import time
import logging
from ultralytics import YOLO
from threading import Lock
from utils import run_in_thread
from detect import HumanDetector, CustomDetector
import math
import numpy as np
from utils import draw_dashed_polygon
from config.env import settings
# Configure logging
logging.basicConfig(filename='stream.log', level=logging.INFO)

class VideoStreamer:
    def __init__(self, rtsp_url, use_detection=True):
        
        # for FPS calculation
        self.prev_time = time.time()
        self.fps = 0.0
        
        self.rtsp_url = rtsp_url
        self.use_detection = use_detection
        self.detector = HumanDetector(
            model_path='./model/yolov8n.pt',
            pose_model_path='./model/yolov8m-pose.pt'
        )
        # open camera
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {rtsp_url}")

        self.frame_data = {"frame": None}
        self.running = True
        self.lock = Lock()
        
        # grab one frame for calibration
        ret, calib_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame for calibration")

        # interactive calibration: click two points, then enter their real distance
        self.m_per_px = self._interactive_calibration(calib_frame)

        # pick a zone polygon for trespass detection
        ret, zone_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame for zone selection")
        self.zone_pts = self._interactive_zone(zone_frame)

        # Launch frame reader via reusable thread util
        run_in_thread(self.read_frames)

    def read_frames(self):
        """Continuously grab frames and store the latest one."""
        while self.running:
            if not self.cap.isOpened():
                time.sleep(1)
                continue
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame_data["frame"] = frame
            else:
                time.sleep(0.1)

    def get_fps(self):
        """Compute instantaneous FPS since last call."""
        curr_time = time.time()
        dt = curr_time - self.prev_time
        # protect against division by zero
        if dt > 0:
            self.fps = 1.0 / dt
        self.prev_time = curr_time
        return self.fps

    def toggle_detection(self):
        self.use_detection = not self.use_detection
        logging.info(f"Detection {'enabled' if self.use_detection else 'disabled'}.")



    def stream_loop(self):
        while True:
            with self.lock:
                frame = self.frame_data["frame"].copy() if self.frame_data["frame"] is not None else None

            if frame is None:
                time.sleep(0.05)
                continue
            
            # draw the zone
            overlay = frame.copy()
            cv2.fillPoly(overlay,
                         [np.array(self.zone_pts, dtype=np.int32)],
                         color=(255,255,255))      # white fill
            alpha = 0.25  # adjust transparency
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # --- draw the dashed border ---
            draw_dashed_polygon(
                frame,
                self.zone_pts,
                color=(0,128,0),   # dark-green dashes
                thickness=2,
                dash_length=15
            )
            
            # Run your detector if needed
            if self.use_detection:
                frame, trace_data = self.detector.detect_humans(frame, self.fps, self.m_per_px)
                self.detector.save_trace_json(trace_data, path='human_traces.json')

            # Get FPS and draw it
            fps = self.get_fps()
            cv2.putText(frame,
                        f"FPS: {fps:.1f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

            # Draw detection status
            status = "Detection: ON" if self.use_detection else "Detection: OFF"
            cv2.putText(frame,
                        status,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2)

            cv2.imshow("Live Feed", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.toggle_detection()

        self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
    
    def _interactive_calibration(self, frame: "np.ndarray") -> float:
        """
        Show `frame` and let the user click two points corresponding
        to a known real-world distance. Then prompt for that distance.
        Returns meters-per-pixel.
        """
        pts = []
        def on_mouse(evt, x, y, flags, _):
            if evt == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
                pts.append((x, y))

        cv2.namedWindow("Calibrate")
        cv2.setMouseCallback("Calibrate", on_mouse)
        print("Click two points in the frame to define a known distance, then press 'c'.")

        while True:
            disp = frame.copy()
            for p in pts:
                cv2.circle(disp, p, 5, (0,255,0), -1)
            if len(pts) == 2:
                cv2.line(disp, pts[0], pts[1], (0,255,0), 2)

            cv2.imshow("Calibrate", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(pts) == 2:
                break
        cv2.destroyWindow("Calibrate")

        # compute pixel distance
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        pixel_dist = math.hypot(dx, dy)

        real_dist = float(input("Enter real-world distance (meters) between the two points: "))
        m_per_px  = real_dist / pixel_dist
        print(f"Calibration complete: {m_per_px:.6f} m/px")
        return m_per_px

    def _interactive_zone(self, frame: "np.ndarray") -> list[tuple[int,int]]:
        """
        Let the user click out a polygon. Click vertices, then press 'z' to finish.
        Returns a list of (x,y) points.
        """
        pts = []
        def on_mouse(evt, x, y, flags, _):
            if evt == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))

        cv2.namedWindow("Define Zone")
        cv2.setMouseCallback("Define Zone", on_mouse)
        print("Click zone vertices, then press 'z' to finish.")

        while True:
            disp = frame.copy()
            # draw existing points/lines
            for p in pts:
                cv2.circle(disp, p, 4, (0, 0, 255), -1)
            if len(pts) > 1:
                cv2.polylines(disp, [np.array(pts)], isClosed=False, color=(0,0,255), thickness=2)

            cv2.imshow("Define Zone", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('z') and len(pts) >= 3:
                break

        cv2.destroyWindow("Define Zone")
        return pts


if __name__ == '__main__':
    rtsp_url = settings.RTSP_URL
    streamer = VideoStreamer(rtsp_url)
    streamer.stream_loop()
    
    print(streamer.m_per_px)

