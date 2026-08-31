from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional


class IngestionConfig(BaseModel):
    sample_fps: float = Field(default=2.0, description="Frames per second sampled when motion is detected")
    min_motion_area: int = Field(default=500, description="Minimum contour area for motion detection in pixels")
    motion_threshold: float = Field(default=0.015, description="Fraction of frame pixels changing to trigger active scene")
    consecutive_static_threshold: int = Field(default=10, description="Number of static frames before skipping scene")


class PerceptionConfig(BaseModel):
    yolo_model_name: str = Field(default="yolov8s.pt", description="YOLO model checkpoint name or path")
    confidence_threshold: float = Field(default=0.35, description="Confidence threshold for object detection")
    iou_threshold: float = Field(default=0.45, description="IoU threshold for NMS")
    track_classes: List[int] = Field(
        default=[0, 1, 2, 3, 5, 7, 24, 26, 28, 43],
        description="COCO class IDs of interest (0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 24: backpack, 26: handbag, 28: suitcase, 43: knife)"
    )


class EmbeddingConfig(BaseModel):
    clip_model_name: str = Field(default="ViT-B-16-SigLIP", description="OpenCLIP or HuggingFace model name")
    clip_pretrained: str = Field(default="webli", description="Pretrained weights tag")
    device: str = Field(default="cuda", description="Device for embedding computation ('cuda' or 'cpu')")
    chroma_db_dir: str = Field(default="./data/chroma_db", description="ChromaDB persistence directory")


class AgentConfig(BaseModel):
    llm_provider: str = Field(default="ollama", description="Local LLM provider ('ollama' or 'direct')")
    ollama_model: str = Field(default="qwen2.5:3b", description="Ollama model tag (e.g. qwen2.5:3b, llama3.2:3b)")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API endpoint")
    max_agent_iterations: int = Field(default=8, description="Maximum reasoning steps for investigation")


class AppConfig(BaseModel):
    app_name: str = "SentinelAI - CCTV Video Forensic Agent"
    version: str = "1.0.0"
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./data/reports")
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


default_config = AppConfig()
