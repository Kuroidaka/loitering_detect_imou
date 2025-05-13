import numpy as np
from detect import HumanDetector, CustomDetector, BehaviourDetector
from typing import List, Tuple
from common_model import TraceData
from utils import save_dicts_to_json
from dataclasses import asdict

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

    def detect(
        self,
        frame: np.ndarray,
        fps: float,
        m_per_px: float
    ) -> Tuple[np.ndarray, List[TraceData]]:
        """
        Runs all enabled detectors and returns (frame, updated TraceData list).
        """
        # 1) Run human detection to get a list of TraceData (with IDs)
        frame, human_data = self.human_detector.detect_humans(
            frame, fps, m_per_px
        )

        # 2) For each entry, if it's a person, analyze just that one
        result: List[TraceData] = []
        for entry in human_data:
            if entry.class_name == "person":
                updated = self.behaviour.analyze_behaviour(
                    entry,
                    fps=fps,
                    zone_pts=self.zone_pts,
                    v_min=settings.MIN_SPEED_TO_BE_LOITERING,
                    m_per_px=m_per_px
                )
                result.append(updated)
            else:
                result.append(entry)

        dicts = [asdict(e) for e in result]
        if dicts:
            dicts_no_trace = []
            for d in dicts:
                d.pop('trace', None)
                dicts_no_trace.append(d)
            save_dicts_to_json(dicts, "human_data.json")


        return frame, result