import cv2
import time
from threading import Lock
from utils import run_in_thread

class FrameGrabber:
    def __init__(self, rtsp_url: str):
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {rtsp_url}")

        self.frame = None
        self.lock = Lock()
        self.running = False

    def start(self):
        """Begin background frame capture."""
        if not self.running:
            self.running = True
            run_in_thread(self._read_loop)

    def _read_loop(self):
        """Continuously read frames in a separate thread."""
        while self.running:
            ret, f = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            with self.lock:
                self.frame = f

    def get_frame(self):
        """Safely retrieve the latest frame copy (or None)."""
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        """Stop capture and release resources."""
        self.running = False
        if self.cap:
            self.cap.release()
