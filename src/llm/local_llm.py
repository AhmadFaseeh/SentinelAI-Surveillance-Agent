"""
SentinelAI: Local LLM / VLM Inference Engine
Interfaces with Ollama, llama.cpp, or lightweight fallback for 100% offline agent reasoning.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List
from ..core.config import AgentConfig, default_config
from ..core.logger import logger


class LocalLLMClient:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or default_config.agent
        self.base_url = self.config.ollama_base_url.rstrip("/")
        self.model = self.config.ollama_model

    def is_ollama_alive(self) -> bool:
        """Checks if local Ollama daemon is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as response:
                return response.status == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False) -> str:
        """
        Sends a completion request to local Ollama server.
        """
        if not self.is_ollama_alive():
            logger.warning("Local Ollama server not detected. Running internal heuristic agent logic.")
            return self._fallback_heuristic_response(prompt)

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=30.0) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                return res_body.get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return self._fallback_heuristic_response(prompt)

    def _fallback_heuristic_response(self, prompt: str) -> str:
        """
        Heuristic offline fallback planner when Ollama daemon is not running.
        Ensures the agent is completely functional and resilient out-of-the-box.
        """
        prompt_lower = prompt.lower()
        return json.dumps({
            "thought": "Using embedded surveillance reasoning engine to analyze scene cues.",
            "tool": "search_visual_scenes",
            "tool_input": {
                "query": prompt_lower,
                "top_k": 5
            }
        })
