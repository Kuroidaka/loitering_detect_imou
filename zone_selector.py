import cv2
import numpy as np

class ZoneSelector:
    """
    Interactive polygon selector and point-in-polygon tester.
    """
    def __init__(self, frame: np.ndarray):
        # Runs UI and stores vertex list
        self.pts = self._interactive_zone(frame)

    def _interactive_zone(self, frame: np.ndarray) -> list[tuple[int,int]]:
        pts = []
        def on_mouse(evt, x, y, flags, _):
            if evt == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))

        cv2.namedWindow("Define Zone")
        cv2.setMouseCallback("Define Zone", on_mouse)
        print("Click zone vertices, then press 'z' to finish.")

        while True:
            disp = frame.copy()
            for p in pts:
                cv2.circle(disp, p, 4, (0, 0, 255), -1)
            if len(pts) > 1:
                cv2.polylines(
                    disp,
                    [np.array(pts, dtype=np.int32)],
                    isClosed=False,
                    color=(0,0,255),
                    thickness=2
                )

            cv2.imshow("Define Zone", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('z') and len(pts) >= 3:
                break

        cv2.destroyWindow("Define Zone")
        return pts

