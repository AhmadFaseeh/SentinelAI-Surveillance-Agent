"""
SentinelAI: Vector Storage & Temporal Video Indexing
Manages ChromaDB collections for multimodal keyframe retrieval.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
import numpy as np

from ..core.types import KeyframeMetadata
from ..core.config import EmbeddingConfig, default_config
from ..core.logger import logger


class VideoVectorStore:
    def __init__(
        self,
        collection_name: str = "cctv_keyframes",
        config: Optional[EmbeddingConfig] = None,
        in_memory: bool = False
    ):
        self.config = config or default_config.embeddings
        self.in_memory = in_memory

        if in_memory:
            logger.info("Initializing In-Memory Ephemeral ChromaDB Vector Store...")
            self.client = chromadb.EphemeralClient()
        else:
            self.db_dir = Path(self.config.chroma_db_dir)
            self.db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing Persistent ChromaDB Vector Store at: [bold cyan]{self.db_dir}[/bold cyan]")
            self.client = chromadb.PersistentClient(path=str(self.db_dir))

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "SentinelAI Video Keyframes Vector Index", "hnsw:space": "cosine"}
        )

    def add_keyframes(
        self,
        keyframes: List[KeyframeMetadata],
        embeddings: np.ndarray,
        video_id: str
    ):
        """
        Inserts keyframe embeddings and rich metadata into ChromaDB.
        """
        if len(keyframes) == 0 or len(embeddings) == 0:
            return

        ids = []
        metadatas = []
        documents = []
        embed_list = embeddings.tolist()

        for idx, kf in enumerate(keyframes):
            doc_id = f"{video_id}_kf_{kf.frame_index}_{kf.timestamp_sec:.2f}"
            ids.append(doc_id)

            # Metadata for filtering
            meta = {
                "video_id": video_id,
                "frame_index": int(kf.frame_index),
                "timestamp_sec": float(kf.timestamp_sec),
                "timestamp_formatted": kf.timestamp_formatted,
                "motion_score": float(kf.motion_score),
                "detected_objects": ", ".join(kf.detected_objects) if kf.detected_objects else "none",
                "track_ids": ", ".join(map(str, kf.track_ids)) if kf.track_ids else "none",
                "frame_path": kf.frame_path or ""
            }
            metadatas.append(meta)

            # Context document
            doc_text = (
                f"Video: {video_id} at {kf.timestamp_formatted}. "
                f"Objects detected: {meta['detected_objects']}. "
                f"Motion score: {kf.motion_score:.3f}."
            )
            documents.append(doc_text)

        self.collection.upsert(
            ids=ids,
            embeddings=embed_list,
            metadatas=metadatas,
            documents=documents
        )
        logger.info(f"Successfully indexed [bold green]{len(ids)}[/bold green] keyframes into ChromaDB.")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        time_range: Optional[tuple] = None,
        class_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic cosine search with optional temporal or class filters.
        """
        where_clause = {}
        if time_range:
            min_sec, max_sec = time_range
            where_clause = {
                "$and": [
                    {"timestamp_sec": {"$gte": float(min_sec)}},
                    {"timestamp_sec": {"$lte": float(max_sec)}}
                ]
            }

        query_list = query_embedding.tolist()
        if isinstance(query_list[0], float):
            query_list = [query_list]

        results = self.collection.query(
            query_embeddings=query_list,
            n_results=top_k,
            where=where_clause if where_clause else None
        )

        matches = []
        if results and "metadatas" in results and len(results["metadatas"]) > 0:
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(metas)
            documents = results["documents"][0] if "documents" in results else [""] * len(metas)

            for meta, dist, doc in zip(metas, distances, documents):
                similarity = max(0.0, 1.0 - dist)
                matches.append({
                    "timestamp_sec": meta.get("timestamp_sec"),
                    "timestamp_formatted": meta.get("timestamp_formatted"),
                    "frame_index": meta.get("frame_index"),
                    "frame_path": meta.get("frame_path"),
                    "detected_objects": meta.get("detected_objects"),
                    "similarity_score": round(similarity, 4),
                    "document": doc
                })

        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches

    def clear(self):
        """Clears all vectors in current collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"description": "SentinelAI Video Keyframes Vector Index", "hnsw:space": "cosine"}
        )
