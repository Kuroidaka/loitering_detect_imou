# detect.py

import cv2
import json
import logging
import numpy as np
from threading import Lock
from typing import List, Dict, Tuple

from ultralytics import YOLO
import supervision as sv

from common_model import TraceData
from .behavior import BehaviourDetector

logging.basicConfig(filename='stream.log', level=logging.INFO)

class HumanDetector:
    def __init__(
        self,
        model_path: str = '/model/yolov8s.pt',
        pose_model_path: str = '/model/yolov8m-pose.pt',
        enable_tracking: bool = True,
        enable_keypoints: bool = False
    ):
        self.enable_tracking  = enable_tracking
        self.enable_keypoints = enable_keypoints
        self._lock            = Lock()

        # Core detectors
        self.det_model  = YOLO(model_path)
        self.DETECT_CONF = 0.6

        self.pose_model = YOLO(pose_model_path) if enable_keypoints else None
        self.behaviour_detector = BehaviourDetector()

        # Tracking components
        self.smoother         = sv.DetectionsSmoother()
        self.tracker          = sv.ByteTrack()     if enable_tracking else None
        self.box_annotator    = sv.BoxAnnotator()  if enable_tracking else None
        self.label_annotator  = sv.LabelAnnotator(
            text_scale = 3,
            text_thickness = 2,    
        )if enable_tracking else None
        self.trace_annotator  = sv.TraceAnnotator(
            trace_length = 1000,
            thickness = 10,
            )if enable_tracking else None
        self.track_histories  = {}                 # tid → list of centroids

        # Keypoint annotators
        self.edge_annotator   = sv.EdgeAnnotator()  if enable_keypoints else None
        self.vertex_annotator = sv.VertexAnnotator()if enable_keypoints else None
    def detect_humans(
        self, frame: np.ndarray, fps: float, m_per_px: float
    ) -> Tuple[np.ndarray, List[TraceData]]:
        """
        Runs person detection → optional tracking → optional keypoints.
        Returns (annotated_frame, trace_data).
        """
        with self._lock:
            annotated   = frame.copy()
            trace_data: List[TraceData] = []

            # 1. Raw YOLO detections
            yolo_results = self.det_model(frame, classes=[0], conf=self.DETECT_CONF)[0]

            # 2. Either track+annotate or draw raw boxes
            if self.enable_tracking:
                dets_smoothed = self._get_smoothed_detections(yolo_results)
                tracked       = self._update_tracking(dets_smoothed)
                
                
                trace_data    = self._build_trace_data(tracked, yolo_results)


                # detect behaviours
                # trace_data = self.behaviour_detector.annotate_behaviour(trace_data)

                # annotated = self.behaviour_detector.draw_loiter_flags(annotated, trace_data, tracked.xyxy)
                
                annotated     = self._annotate_tracking(
                    annotated, tracked, yolo_results, trace_data, fps, m_per_px
                )

            else:
                annotated = self._draw_raw_boxes(annotated, yolo_results)

            # 3. Optionally overlay pose keypoints
            if self.enable_keypoints:
                annotated = self._annotate_keypoints(annotated, frame)

            return annotated, trace_data
    # ─── Helpers for detection & tracking ─────────────────────────────────────

    def _get_smoothed_detections(self, results) -> sv.Detections:
        dets = sv.Detections.from_ultralytics(results)
        return self.smoother.update_with_detections(dets)

    def _update_tracking(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)

    def _build_trace_data(
        self,
        tracked: sv.Detections,
        results
    ) -> List[TraceData]:
        """
        Build a list of TraceData instances from tracked results.
        """
        data: List[TraceData] = []
        for bbox, tid, cid, conf in zip(
            tracked.xyxy,
            tracked.tracker_id,
            tracked.class_id,
            tracked.confidence
        ):
            tid = int(tid)
            # record centroid
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            self.track_histories.setdefault(tid, []).append((cx, cy))
            # create TraceData object
            record = TraceData(
                id=tid,
                class_name=results.names[int(cid)],
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                confidence=float(conf),
                trace=self.track_histories.get(tid, [])
            )
            data.append(record)
        return data

    def _annotate_tracking(
        self,
        img: np.ndarray,
        tracked: sv.Detections,
        results,
        trace_data: List[TraceData],
        fps: float,
        m_per_px: float,
    ) -> np.ndarray:
        # 1) Build labels including speed
        labels = []
        for entry, cid in zip(trace_data, tracked.class_id):
        
            avg, mn = self.behaviour_detector.compute_speeds(entry.trace, fps=fps, m_per_px=m_per_px)
            print(f"Average speed = {avg:.2f} m/s, Minimum speed = {mn:.2f} m/s")
            speed = avg or 0.0
            labels.append(
                f"#{entry.id} {results.names[int(cid)]} {speed:.1f}m/s"
            )

        # 2) Draw boxes, then labels (with speed), then traces
        img = self.box_annotator.annotate(img, tracked)
        img = self.label_annotator.annotate(img, tracked, labels)
        img = self.trace_annotator.annotate(img, tracked)

        return img

    # ─── Helpers for raw drawing & keypoints ────────────────────────────────

    def _draw_raw_boxes(self, img: np.ndarray, results) -> np.ndarray:
        for box in results.boxes:
            if int(box.cls[0]) != 0:  # only person
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(
                img,
                f'Person {conf:.2f}',
                (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                1
            )
        return img

    def _annotate_keypoints(
        self,
        img: np.ndarray,
        frame: np.ndarray
    ) -> np.ndarray:
        pose_res = self.pose_model(frame)[0]
        kps      = sv.KeyPoints.from_ultralytics(pose_res)
        img      = self.edge_annotator.annotate(img, key_points=kps)
        return self.vertex_annotator.annotate(img, key_points=kps)

    # ─── JSON export ────────────────────────────────────────────────────────

    def save_trace_json(self, trace_data: List[dict], path='trace_data.json'):
        try:
            # turn each TraceData into a dict
            json_ready = [td.__dict__ for td in trace_data]
            with open(path, 'w') as f:
                json.dump(json_ready, f, indent=2)
            logging.info(f"Trace data written to {path}")
        except Exception as e:
            logging.error(f"Failed to write trace JSON: {e}")
