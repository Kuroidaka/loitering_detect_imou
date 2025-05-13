import cv2

class LocalFrameGrabber:
    """
    Reads frames from a video source (local file) without using threads.
    """

    def __init__(self, source: str):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def get_frame(self):
        """Read and return the next frame from the video, or None if end reached."""
        ret, frame = self.cap.read()
        if not ret:
            # Optional: loop the video if it's a file
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None
        return frame

    def pause(self):
        """Return the last read frame without reading a new one."""
        return self.current_frame


    def release(self):
        """Release the video capture object."""
        if self.cap:
            self.cap.release()
            self.cap = None
