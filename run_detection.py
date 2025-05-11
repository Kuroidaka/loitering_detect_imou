import cv2
import argparse
from detect.human import HumanDetector  # adjust import as needed

def main():
    parser = argparse.ArgumentParser(
        description="Detect and track humans in a video using YOLOv8 + ByteTrack"
    )
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path to save annotated output video (optional)"
    )
    parser.add_argument(
        "-m", "--model", default="yolov8n.pt",
        help="Path to YOLOv8 model file"
    )
    parser.add_argument(
        "--no-tracking", action="store_false", dest="enable_tracking",
        help="Disable tracking (only draw raw detections)"
    )
    args = parser.parse_args()

    # Initialize detector
    detector = HumanDetector(model_path=args.model, enable_tracking=args.enable_tracking)

    # Open video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Couldn't open video {args.input}")
        return

    # Prepare VideoWriter if requested
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"Error: Couldn't open writer for {args.output}")
            writer = None

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = detector.detect_humans(frame)

        # Show on screen
        cv2.imshow("Human Detection", annotated)
        if writer:
            writer.write(annotated)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


