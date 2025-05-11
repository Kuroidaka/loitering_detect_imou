import cv2
import math

class CalibrationTool:
    """
    Interactive two-point calibration to compute meters-per-pixel.
    """
    def __init__(self, frame):
        # Runs the UI and stores the result
        self.m_per_px = self._interactive_calibration(frame)

    def _interactive_calibration(self, frame):
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

        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        pixel_dist = math.hypot(dx, dy)

        real_dist = float(input("Enter real-world distance (meters) between the two points: "))
        m_per_px  = real_dist / pixel_dist
        print(f"Calibration complete: {m_per_px:.6f} m/px")
        return m_per_px
