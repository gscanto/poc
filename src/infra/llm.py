import logging
import requests
from typing import Optional
from ..core.interfaces import LLMProvider

logger = logging.getLogger(__name__)

class OllamaClient(LLMProvider):
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_generate = f"{base_url}/api/generate"

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates text using the Ollama API.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(self.api_generate, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return f"Error: Could not generate text. Details: {e}"

    def get_model_name(self) -> str:
        return self.model_name
