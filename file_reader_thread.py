import cv2
import threading
import time
from queue import Queue, Empty
from typing import Optional

class ThreadedFileReader:
    """
    Background‐threaded video reader that can be reset to the beginning.
    .cap           : underlying VideoCapture
    .start()       : begin grabbing
    .get_frame()   : nonblocking fetch latest frame or None
    .reset()       : seek to frame 0, clear buffer, restart reader thread
    .release()     : stop grabbing, join thread, release cap
    """
    def __init__(self, src=0, queue_size=128):
        self.src        = src
        self.queue_size = queue_size
        self._init_capture()
        self._make_queue_and_lock()
        self._make_thread()

    def _init_capture(self):
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open source {self.src}")

    def _make_queue_and_lock(self):
        self.queue = Queue(maxsize=self.queue_size)
        self._stop_lock = threading.Lock()
        self.stopped = False

    def _make_thread(self):
        self._thr = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        """Kick off the reader thread."""
        self._thr.start()
        return self

    def _reader(self):
        while True:
            with self._stop_lock:
                if self.stopped:
                    break
            ret, frame = self.cap.read()
            if not ret:
                # EOF reached
                break

            if self.queue.full():
                try: self.queue.get_nowait()
                except Empty: pass
            self.queue.put(frame)
        # exit; mark stopped
        with self._stop_lock:
            self.stopped = True

    def get_frame(self) -> Optional[cv2.Mat]:
        """Nonblocking fetch of latest frame (or None)."""
        try:
            return self.queue.get_nowait()
        except Empty:
            return None

    def reset(self):
        """Seek back to the beginning, clear the buffer, and restart the reader."""
        # 1) stop the current thread
        with self._stop_lock:
            self.stopped = True
        if self._thr.is_alive():
            self._thr.join()

        # 2) clear any buffered frames
        with self.queue.mutex:
            self.queue.queue.clear()

        # 3) seek capture back to frame 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 4) restart the reader thread
        with self._stop_lock:
            self.stopped = False
        self._make_thread()
        self._thr.start()

    def release(self):
        """Stop reader and free resources."""
        with self._stop_lock:
            self.stopped = True
        if self._thr.is_alive():
            self._thr.join()
        self.cap.release()
