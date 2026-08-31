from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class TrackedObject(BaseModel):
    track_id: int
    class_name: str
    class_id: int
    bbox: BoundingBox
    timestamp_sec: float
    frame_index: int


class KeyframeMetadata(BaseModel):
    frame_index: int
    timestamp_sec: float
    timestamp_formatted: str
    motion_score: float
    detected_objects: List[str] = Field(default_factory=list)
    track_ids: List[int] = Field(default_factory=list)
    has_anomaly: bool = False
    anomaly_reason: Optional[str] = None
    frame_path: Optional[str] = None


class VideoSegment(BaseModel):
    segment_id: int
    start_sec: float
    end_sec: float
    keyframe_indices: List[int]
    activity_level: str  
    summary: Optional[str] = None


class ForensicEvidence(BaseModel):
    timestamp_sec: float
    timestamp_formatted: str
    evidence_type: str 
    description: str
    confidence: float
    snapshot_path: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None


class IncidentReport(BaseModel):
    case_title: str
    video_source: str
    query: str
    summary_findings: str
    evidence_items: List[ForensicEvidence] = Field(default_factory=list)
    timeline_timestamps: List[str] = Field(default_factory=list)
    verdict: str 
