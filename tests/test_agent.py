"""
Unit Tests for Detective Agent Workflow
"""

import pytest
import numpy as np

from src.embeddings.siglip_embedder import MultimodalEmbedder
from src.embeddings.vector_store import VideoVectorStore
from src.agent.tools import SurveillanceTools
from src.agent.graph import ForensicAgent
from src.llm.local_llm import LocalLLMClient
from src.core.types import KeyframeMetadata


def test_agent_investigation_cycle():
    store = VideoVectorStore(collection_name="test_agent_col", in_memory=True)
    embedder = MultimodalEmbedder()

    # Add sample indexed keyframe
    kf = KeyframeMetadata(
        frame_index=30,
        timestamp_sec=3.0,
        timestamp_formatted="00:03.000",
        motion_score=0.4,
        detected_objects=["car", "person"]
    )
    emb = embedder.embed_text("white car parked near gate")
    store.add_keyframes([kf], emb, video_id="sample_cam")

    tools = SurveillanceTools(embedder=embedder, vector_store=store)
    llm = LocalLLMClient()
    agent = ForensicAgent(tools=tools, llm_client=llm)

    report = agent.run_investigation(query="white car")
    assert report is not None
    assert report.verdict in ["CONFIRMED_EVENT", "INCONCLUSIVE", "NO_MATCH"]
    assert len(report.evidence_items) > 0
