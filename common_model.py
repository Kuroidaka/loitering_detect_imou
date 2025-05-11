import cv2
from typing import List, Dict, Tuple, Optional


from pydantic import BaseModel, Field
from typing import List, Tuple

class TraceData(BaseModel):
    """
    Schema for individual trace records of tracked humans.
    """
    id: int
    class_name: str = Field(..., alias="class")  # original key 'class'
    bbox: List[int]                              # [x1, y1, x2, y2]
    confidence: float
    trace: List[Tuple[int, int]]                 # list of centroids (cx, cy)
    avg_speed: Optional[float] = None
    min_speed: Optional[float] = None
    loitering: bool = False

    class Config:
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "id": 28,
                "class": "person",
                "bbox": [852, 4, 2298, 1285],
                "confidence": 0.9013750553131104,
                "trace": [[2238, 651], [2240, 650], ...],
                "loitering": False
            }
        }
