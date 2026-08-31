"""
Unit Tests for Multimodal Embeddings & ChromaDB Vector Store
"""

import pytest
import numpy as np
from PIL import Image

from src.embeddings.siglip_embedder import MultimodalEmbedder
from src.embeddings.vector_store import VideoVectorStore
from src.core.types import KeyframeMetadata


def test_embedder_shapes():
    embedder = MultimodalEmbedder()

    # Text embedding
    text_emb = embedder.embed_text("person carrying bag")
    assert text_emb.shape[0] == 1
    assert text_emb.shape[1] > 0

    # Image embedding
    dummy_img = Image.new("RGB", (224, 224), color="blue")
    img_emb = embedder.embed_image(dummy_img)
    assert len(img_emb.shape) == 1
    assert img_emb.shape[0] == text_emb.shape[1]


def test_vector_store_indexing_and_search():
    store = VideoVectorStore(collection_name="test_collection", in_memory=True)

    # Create dummy keyframe
    kf = KeyframeMetadata(
        frame_index=15,
        timestamp_sec=1.5,
        timestamp_formatted="00:01.500",
        motion_score=0.25,
        detected_objects=["person"]
    )

    dummy_emb = np.random.randn(1, 512).astype(np.float32)
    dummy_emb = dummy_emb / np.linalg.norm(dummy_emb)

    store.add_keyframes([kf], dummy_emb, video_id="test_video")

    # Search with same vector
    results = store.search(dummy_emb[0], top_k=1)
    assert len(results) == 1
    assert results[0]["timestamp_sec"] == 1.5
