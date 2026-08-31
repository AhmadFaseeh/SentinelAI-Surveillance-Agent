"""
SentinelAI: Forensic Investigation Tool Definitions
Tools used by the LangGraph agent for video exploration, tracking, OCR plate reading, and evidence synthesis.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2

from ..embeddings.siglip_embedder import MultimodalEmbedder
from ..embeddings.vector_store import VideoVectorStore
from ..perception.tracker import SurveillanceTracker
from ..ingestion.video_reader import VideoReader
from ..core.types import ForensicEvidence, IncidentReport
from ..core.logger import logger

try:
    import easyocr
    _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
except Exception:
    _ocr_reader = None


class SurveillanceTools:
    def __init__(
        self,
        embedder: MultimodalEmbedder,
        vector_store: VideoVectorStore,
        tracker: Optional[SurveillanceTracker] = None,
        video_reader: Optional[VideoReader] = None
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.tracker = tracker
        self.video_reader = video_reader
        self.ocr_reader = _ocr_reader

    def search_visual_scenes(
        self,
        query: str,
        top_k: int = 5,
        time_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches video keyframes using natural language semantic description.
        """
        logger.info(f"[bold yellow]Tool Call:[/bold yellow] search_visual_scenes('{query}', top_k={top_k})")
        query_emb = self.embedder.embed_text(query)[0]
        results = self.vector_store.search(
            query_embedding=query_emb,
            top_k=top_k,
            time_range=time_range
        )
        return results

    def read_license_plate_ocr(self, timestamp_sec: float) -> List[str]:
        """
        Reads visible text and vehicle license plates from the frame at timestamp.
        """
        logger.info(f"[bold yellow]Tool Call:[/bold yellow] read_license_plate_ocr(timestamp={timestamp_sec:.2f}s)")
        if not self.video_reader or self.ocr_reader is None:
            return []

        frame = self.video_reader.get_frame_at_timestamp(timestamp_sec)
        if frame is None:
            return []

        try:
            results = self.ocr_reader.readtext(frame)
            extracted_texts = [text for (_, text, conf) in results if conf > 0.35 and len(text.strip()) > 1]
            return extracted_texts
        except Exception as e:
            logger.warning(f"OCR reading error: {e}")
            return []

    def get_track_history(self, track_id: int) -> Dict[str, Any]:
        """
        Retrieves movement trajectory and dwell/loitering stats for an object ID.
        """
        logger.info(f"[bold yellow]Tool Call:[/bold yellow] get_track_history(track_id={track_id})")
        if not self.tracker:
            return {"error": "Tracker is not active for this video stream"}
        summary = self.tracker.get_track_summary(track_id)
        if summary is None:
            return {"error": f"Track ID {track_id} not found"}
        return summary

    def inspect_frame_details(self, timestamp_sec: float) -> Dict[str, Any]:
        """
        Inspects high-resolution frame at timestamp for forensic verification.
        """
        logger.info(f"[bold yellow]Tool Call:[/bold yellow] inspect_frame_details(timestamp={timestamp_sec:.2f}s)")
        if not self.video_reader:
            return {"error": "VideoReader instance not attached"}

        frame = self.video_reader.get_frame_at_timestamp(timestamp_sec)
        if frame is None:
            return {"error": f"Could not retrieve frame at timestamp {timestamp_sec}s"}

        return {
            "timestamp_sec": timestamp_sec,
            "timestamp_formatted": VideoReader.format_time(timestamp_sec),
            "frame_shape": list(frame.shape),
            "status": "Frame retrieved successfully"
        }
