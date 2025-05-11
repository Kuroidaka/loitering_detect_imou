import cv2
import time

def play_video(video_path: str):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Query the video’s nominal frame rate (unused for timing, but could be shown)
    nominal_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # For measuring actual playback FPS
    prev_time = time.time()

    # Read and display frames until the video ends or you press 'q'
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        # Compute instantaneous FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        fps = 1.0 / dt if dt > 0 else 0.0
        prev_time = curr_time

        # Overlay the FPS on the frame
        cv2.putText(frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

        # Show the frame
        cv2.imshow("Video Playback", frame)
        # Wait for the exact frame interval, exit on 'q'
        # We're still throttling to the nominal rate so playback speed matches the video
        delay_ms = int(1000 / nominal_fps)
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    play_video("./assets/video_test.MOV")
