import cv2
from typing import List, Dict, Tuple
from common_model import TraceData

class BehaviourDetector:
    """
    Analyse tracked object traces to detect behaviors such as loitering.
    """
    def __init__(
        self,
        loiter_max_disp: int = 50,
        loiter_min_frames: int = 30
    ):
        """
        :param loiter_max_disp: Maximum pixel displacement to consider as loitering.
        :param loiter_min_frames: Number of recent frames to inspect for loitering.
        """
        self.loiter_max_disp = loiter_max_disp
        self.loiter_min_frames = loiter_min_frames

    def is_loitering(self, history: List[Tuple[int, int]]) -> bool:
        """
        Determine if the given centroid history indicates loitering:
        returns True if the last loiter_min_frames points lie within a box of size loiter_max_disp.
        """
        if len(history) < self.loiter_min_frames:
            return False

        window = history[-self.loiter_min_frames:]
        xs = [pt[0] for pt in window]
        ys = [pt[1] for pt in window]
        if (max(xs) - min(xs) <= self.loiter_max_disp and
            max(ys) - min(ys) <= self.loiter_max_disp):
            return True
        return False

    def annotate_behaviour(
        self,
        trace_data: List[TraceData]
    ) -> List[TraceData]:
        annotated:List[TraceData] = []
        for entry in trace_data:
            loiter = self.is_loitering(entry.trace)
            entry.loitering = loiter   # now a normal dict
            annotated.append(entry)
        return annotated

    def draw_loiter_flags(
        self,
        img: 'np.ndarray',
        trace_data: List[Dict],
        tracked_boxes: List[Tuple[float, float, float, float]]
    ) -> 'np.ndarray':
        """
        Draws red bounding boxes around objects flagged as loitering.
        :param img: The image to draw on.
        :param trace_data: List of behaviour-annotated entries.
        :param tracked_boxes: List of (x1,y1,x2,y2) for each tracked object, same order as trace_data.
        """
        for entry, box in zip(trace_data, tracked_boxes):
            if entry.loitering:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        return img
