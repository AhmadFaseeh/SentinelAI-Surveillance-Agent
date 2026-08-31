"""
SentinelAI: End-to-End CLI Demonstration Script
Simulates full surveillance workflow:
1. Creates a synthetic CCTV surveillance footage with dynamic activity
2. Ingests & motion-filters video
3. Detects objects (YOLO) & extracts multimodal embeddings (SigLIP)
4. Indexes into ChromaDB vector store
5. Deploys LangGraph Forensic Agent to investigate a natural language query
6. Outputs structured forensic incident report
"""

import cv2
import numpy as np
from pathlib import Path
from rich.panel import Panel
from rich.table import Table

from src.core.logger import logger, console
from src.core.types import IncidentReport
from src.ingestion.video_reader import VideoReader
from src.ingestion.scene_segmenter import SceneSegmenter
from src.perception.detector import SurveillanceDetector
from src.perception.tracker import SurveillanceTracker
from src.embeddings.siglip_embedder import MultimodalEmbedder
from src.embeddings.vector_store import VideoVectorStore
from src.agent.tools import SurveillanceTools
from src.agent.graph import ForensicAgent
from src.llm.local_llm import LocalLLMClient


def generate_synthetic_cctv(output_path: str, duration_sec: int = 10, fps: int = 10):
    """Generates a synthetic surveillance video with background and moving entities."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    logger.info(f"Generating synthetic CCTV footage: {output_path} ({duration_sec}s @ {fps}fps)...")
    total_frames = duration_sec * fps

    for i in range(total_frames):
        # Surveillance background (dark asphalt road with lane lines)
        frame = np.full((height, width, 3), (35, 35, 40), dtype=np.uint8)
        cv2.line(frame, (0, 240), (640, 240), (100, 100, 100), 2)

        # Draw CCTV timestamp watermark
        timestamp_str = f"CAM_01 | 2026-08-31 20:30:{i // fps:02d}.{int((i % fps) * 100):02d}"
        cv2.putText(frame, timestamp_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Event: Blue Vehicle moves from left to right between sec 2 and sec 7 (frames 20 to 70)
        if 20 <= i <= 70:
            car_x = int(50 + (i - 20) * 10)
            cv2.rectangle(frame, (car_x, 260), (car_x + 120, 340), (200, 50, 20), -1)  # Blue-ish BGR
            cv2.putText(frame, "VEHICLE", (car_x + 10, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    logger.info("Synthetic CCTV video generation complete.")


def run_demo():
    console.print(Panel.fit(
        "[bold cyan]🛡️ SentinelAI: Autonomous Surveillance Forensic Agent Demo[/bold cyan]\n"
        "[green]100% Local & Offline CCTV Analysis Pipeline[/green]",
        border_style="cyan"
    ))

    # 1. Video Preparation
    data_dir = Path("./data/demo")
    data_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(data_dir / "cctv_sample_feed.mp4")
    snapshots_dir = str(data_dir / "snapshots")

    generate_synthetic_cctv(video_path, duration_sec=8, fps=10)

    # 2. Ingestion & Motion Filtering
    logger.info("\n[bold]Step 1:[/bold] Ingestion & Motion Filtering...")
    segmenter = SceneSegmenter()
    keyframes, segments = segmenter.process_video(
        video_path=video_path,
        output_snapshots_dir=snapshots_dir
    )

    # 3. Perception & Object Detection
    logger.info("\n[bold]Step 2:[/bold] Object Detection & Multimodal Embeddings...")
    detector = SurveillanceDetector()
    embedder = MultimodalEmbedder()
    vector_store = VideoVectorStore(collection_name="demo_surveillance")
    vector_store.clear()

    reader = VideoReader(video_path)
    images_for_embed = []
    for kf in keyframes:
        frame = reader.get_frame_at_timestamp(kf.timestamp_sec)
        if frame is not None:
            boxes = detector.detect_in_frame(frame)
            kf.detected_objects = [b.class_name for b in boxes]
            images_for_embed.append(VideoReader.bgr_to_pil(frame))

    # 4. Vector Storage
    if images_for_embed:
        embeddings = embedder.embed_image_batch(images_for_embed)
        vector_store.add_keyframes(keyframes, embeddings, video_id="cctv_sample")

    # 5. Agent Investigation
    logger.info("\n[bold]Step 3:[/bold] Deploying LangGraph Forensic Agent...")
    tools = SurveillanceTools(
        embedder=embedder,
        vector_store=vector_store,
        video_reader=reader
    )
    llm_client = LocalLLMClient()
    agent = ForensicAgent(tools=tools, llm_client=llm_client)

    test_query = "Vehicle moving on road"
    logger.info(f"Submitting Forensic Query: [bold yellow]'{test_query}'[/bold yellow]")
    report: IncidentReport = agent.run_investigation(query=test_query, video_source="cctv_sample_feed.mp4")

    # 6. Display Structured Findings
    console.print("\n")
    console.print(Panel(
        f"[bold]Case Title:[/bold] {report.case_title}\n"
        f"[bold]Verdict:[/bold] [bold green]{report.verdict}[/bold green]\n"
        f"[bold]Findings:[/bold] {report.summary_findings}",
        title="📋 Forensic Incident Report",
        border_style="green"
    ))

    table = Table(title="🔍 Discovered Evidence Timeline")
    table.add_column("Timestamp", style="cyan", no_wrap=True)
    table.add_column("Evidence Type", style="magenta")
    table.add_column("Confidence", style="green")
    table.add_column("Description", style="white")

    for ev in report.evidence_items:
        table.add_row(
            ev.timestamp_formatted,
            ev.evidence_type,
            f"{ev.confidence:.2f}",
            ev.description
        )

    console.print(table)
    reader.close()


if __name__ == "__main__":
    run_demo()
