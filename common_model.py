from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class TraceData:
    id: int
    class_name: str  # manually map from 'class' if needed
    bbox: List[int]
    confidence: float
    avg_speed: Optional[float] = None
    min_speed: Optional[float] = None
    rule_loiter: bool = False
    raw_loiter_score: float = 0.0
    is_loitering: bool = False
    dwell_time: Optional[float] = None
    pause_duration: Optional[float] = None
    movement_radius: Optional[float] = None
    curl_count: Optional[int] = None
    path_variance: Optional[float] = None
    trace: List[Tuple[int, int]] = field(default_factory=list)