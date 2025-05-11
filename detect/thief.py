import cv2
import time
import logging
from ultralytics import YOLO
import random

# Logging setup
logging.basicConfig(
    filename='stream.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class CustomDetector:
    def __init__(self,
            model_path,
            target_classes=None,
    ):
        # Load your fine-tuned model
        self.model = YOLO(model_path)
        # Get all class names from the model
        self.class_names = [n.lower() for n in self.model.names.values()]
        
        # If you only want to highlight a subset, pass them here; otherwise all
        if target_classes:
            self.targets = set(c.lower() for c in target_classes)
        else:
            self.targets = set(self.class_names)
        
        # Generate a random color for each class
        random.seed(42)
        self.colors = {
            name: (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            for name in self.class_names
        }
        
        logging.info(f"Initialized detector for classes: {self.targets}")

    def detect(self, frame):
        # Run inference
        results = self.model(frame, verbose=False)[0]
        boxes = results.boxes
        
        logging.info(f"Frame processed — {len(boxes)} total detections")
        
        for box in boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id].lower()
            conf  = float(box.conf[0])
            
            # Only draw boxes for your selected classes
            if label in self.targets:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = self.colors[label]
                
                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
                
                logging.info(f"  ▶ {label} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]")
        
        return frame
