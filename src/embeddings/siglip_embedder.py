"""
SentinelAI: Multimodal SigLIP / CLIP Embedder Engine
Encodes surveillance frames and natural language queries into a unified latent vector space.
"""

import torch
from PIL import Image
import numpy as np
from typing import List, Union, Optional
import open_clip
import cv2

from ..core.config import EmbeddingConfig, default_config
from ..core.logger import logger


class MultimodalEmbedder:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or default_config.embeddings
        self.device = "cuda" if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"

        # Fast, standard CLIP ViT-B-32 by default
        model_name = "ViT-B-32"
        pretrained = "openai"

        logger.info(
            f"Loading Multimodal Vision Embedder: [bold cyan]{model_name}[/bold cyan] "
            f"(Pretrained: {pretrained}) on [bold green]{self.device}[/bold green]..."
        )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embeds a single query string or a list of query strings.
        Returns: normalized numpy array of embeddings (shape: [N, D]).
        """
        if isinstance(text, str):
            text = [text]

        tokens = self.tokenizer(text).to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    @torch.no_grad()
    def embed_image(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        """
        Embeds a single image (PIL Image, OpenCV numpy array, or file path).
        Returns: normalized 1D numpy array of embedding (shape: [D]).
        """
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

    @torch.no_grad()
    def embed_image_batch(self, images: List[Union[Image.Image, np.ndarray, str]]) -> np.ndarray:
        """
        Embeds a batch of images for efficient GPU/CPU acceleration.
        Returns: normalized numpy array (shape: [N, D]).
        """
        tensors = []
        for img in images:
            if isinstance(img, str):
                pil_img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                continue
            tensors.append(self.preprocess(pil_img))

        if not tensors:
            return np.empty((0, 512))

        batch_tensor = torch.stack(tensors).to(self.device)
        features = self.model.encode_image(batch_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()
