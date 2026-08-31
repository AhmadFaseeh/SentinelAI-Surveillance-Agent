
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .video_reader import VideoReader
from .motion_detector import MotionDetector
from ..core.types import KeyframeMetadata, VideoSegment
from ..core.config import IngestionConfig, default_config
from ..core.logger import logger


class SceneSegmenter:
    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or default_config.ingestion
        self.motion_detector = MotionDetector(self.config)

    def process_video(
        self,
        video_path: str,
        output_snapshots_dir: Optional[str] = None
    ) -> Tuple[List[KeyframeMetadata], List[VideoSegment]]:
        snapshots_dir = Path(output_snapshots_dir) if output_snapshots_dir else None
        if snapshots_dir:
            snapshots_dir.mkdir(parents=True, exist_ok=True)

        keyframes: List[KeyframeMetadata] = []
        segments: List[VideoSegment] = []

        with VideoReader(video_path) as reader:
            metadata = reader.get_metadata()
            logger.info(f"Analyzing video motion stream @ {self.config.sample_fps} FPS...")

            current_segment_keyframes = []
            segment_id = 1
            segment_start_sec = 0.0
            in_active_segment = False
            consecutive_idle = 0

            total_samples = int(reader.duration_sec * self.config.sample_fps) + 1
            pbar = tqdm(
                reader.stream_sampled_frames(target_fps=self.config.sample_fps),
                total=total_samples,
                desc="Ingesting Video Frames",
                unit="frame"
            )

            for frame_idx, timestamp_sec, frame in pbar:
                is_motion, motion_score, motion_boxes = self.motion_detector.analyze_frame(frame)
                timestamp_formatted = VideoReader.format_time(timestamp_sec)

                frame_snapshot_path = None
                if snapshots_dir:
                    frame_snapshot_path = str(
                        snapshots_dir / f"frame_{frame_idx:06d}_{timestamp_sec:.2f}s.jpg"
                    )

                if is_motion or frame_idx == 0:
                    if snapshots_dir and frame_snapshot_path:
                        reader.save_snapshot(frame, frame_snapshot_path)

                    kf = KeyframeMetadata(
                        frame_index=frame_idx,
                        timestamp_sec=timestamp_sec,
                        timestamp_formatted=timestamp_formatted,
                        motion_score=motion_score,
                        frame_path=frame_snapshot_path
                    )
                    keyframes.append(kf)
                    current_segment_keyframes.append(frame_idx)

                    if not in_active_segment:
                        in_active_segment = True
                        segment_start_sec = timestamp_sec

                    consecutive_idle = 0
                else:
                    consecutive_idle += 1

                if in_active_segment and consecutive_idle >= self.config.consecutive_static_threshold:
                    segments.append(
                        VideoSegment(
                            segment_id=segment_id,
                            start_sec=segment_start_sec,
                            end_sec=timestamp_sec,
                            keyframe_indices=current_segment_keyframes.copy(),
                            activity_level="high" if len(current_segment_keyframes) > 10 else "medium",
                            summary=f"Activity detected between {VideoReader.format_time(segment_start_sec)} and {timestamp_formatted}"
                        )
                    )
                    segment_id += 1
                    current_segment_keyframes.clear()
                    in_active_segment = False

            if in_active_segment and current_segment_keyframes:
                segments.append(
                    VideoSegment(
                        segment_id=segment_id,
                        start_sec=segment_start_sec,
                        end_sec=reader.duration_sec,
                        keyframe_indices=current_segment_keyframes.copy(),
                        activity_level="high" if len(current_segment_keyframes) > 10 else "medium",
                        summary=f"Activity detected until end of video ({VideoReader.format_time(reader.duration_sec)})"
                    )
                )

        logger.info(
            f"[bold green]Ingestion Complete:[/bold green] Extracted {len(keyframes)} active keyframes "
            f"across {len(segments)} activity segments."
        )
        return keyframes, segments
