import numpy as np
from detect import HumanDetector, CustomDetector, BehaviourDetector
from typing import List, Tuple
from common_model import TraceData

from config import settings
class DetectorService:
    """
    Unified interface for multiple detection models.
    """
    def __init__(self, model_cfg: dict, zone_pts:list[tuple[int,int]]):
        # Human detector
        self.zone_pts = zone_pts
        human_cfg = model_cfg.get("human", {})
        self.human_detector = HumanDetector(
            model_path=human_cfg.get("model_path", ""),
            pose_model_path=human_cfg.get("pose_model_path", "")
        )
        
        self.behaviour = BehaviourDetector()

        # Custom detector (e.g., vehicle, object)
        # custom_cfg = model_cfg.get("custom", {})
        # self.custom_detector = CustomDetector(
        #     model_path=custom_cfg.get("model_path", "")
        # )

    def detect(self, frame: np.ndarray, fps: float, m_per_px: float) -> Tuple[np.ndarray, List[TraceData]]:
        """
        Runs all enabled detectors and returns a combined list of detection dicts.
        """
        # Human detection
        frame, human_data = self.human_detector.detect_humans(
            frame, fps, m_per_px, zone_pts = self.zone_pts
        )
        
        people = [d for d in human_data if d.class_name == "person"]

        if people:
            for person in people:
                human_data = self.behaviour.analyze_behaviour(
                    human_data,
                    fps=fps,
                    zone_pts=self.zone_pts,
                    v_min=settings.MIN_SPEED_TO_BE_LOITERING,        # e.g. 0.2 m/s
                    m_per_px=m_per_px   # your scene calibration
                )
        # Example: custom = self.custom_detector.detect_objects(frame)
        # Combine lists if you have multiple detectors:
        # return human_data + custom

        # Save traces for later analysis
        self.human_detector.save_trace_json(human_data, path='human_traces.json')
        return frame, human_data,
