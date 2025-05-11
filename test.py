import cv2

def calibrate_m_per_px(frame, real_dist_m):
    pts = []

    def click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x,y))
            cv2.circle(frame, (x,y), 5, (0,255,0), -1)
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", click)

    while len(pts) < 2:
        cv2.imshow("calibrate", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyWindow("calibrate")

    (x1,y1),(x2,y2) = pts
    pixel_dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return real_dist_m / pixel_dist

# usage inside VideoStreamer.__init__:
ret, frame = self.cap.read()
self.m_per_px = calibrate_m_per_px(frame, real_dist_m=1.2)
