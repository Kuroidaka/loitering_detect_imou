import numpy as np
from detect import HumanDetector, CustomDetector
from typing import List, Tuple
from common_model import TraceData

class DetectorService:
    """
    Unified interface for multiple detection models.
    """
    def __init__(self, model_cfg: dict):
        # Human detector
        human_cfg = model_cfg.get("human", {})
        self.human_detector = HumanDetector(
            model_path=human_cfg.get("model_path", ""),
            pose_model_path=human_cfg.get("pose_model_path", "")
        )
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
            frame, fps, m_per_px
        )
        # Example: custom = self.custom_detector.detect_objects(frame)
        # Combine lists if you have multiple detectors:
        # return human_data + custom

        # Save traces for later analysis
        self.human_detector.save_trace_json(human_data, path='human_traces.json')
        return frame, human_data
