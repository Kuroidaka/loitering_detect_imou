import cv2
import time
from threading import Thread, Lock

class LocalFrameGrabber:
    """
    Continuously reads frames from a video source (RTSP URL or local file) in a background thread.

    Attributes:
        source (str): Path or URL of the video source.
        loop (bool): Whether to restart from the beginning when a file ends (unused here).
    """
    def __init__(self, source: str):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.frame = None
        self.lock = Lock()
        self.running = False
        self.thread = None

    def start(self):
        """Start the background thread to read frames."""
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._read_loop, daemon=True)
            self.thread.start()

    def _read_loop(self):
        """Internal loop: continuously read frames while running."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # If video file, could loop by resetting position
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def get_frame(self):
        """Retrieve the latest frame copy, or None if no frame is available."""
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def pause(self):
        """Pause the background frame-reading thread without releasing the capture."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def close(self):
        """Release the VideoCapture resource."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def stop(self):
        """Stop reading frames and release the capture."""
        self.pause()
        self.close()
