"""
NVIDIA VLM Client
=================
HTTP Client wrapper for NVIDIA Vision-Language API endpoints.
Reads environment variables (NVIDIA_API_KEY, NVIDIA_VLM_MODEL, NVIDIA_VLM_BASE_URL)
and formats vision-language analysis requests using OpenAI-compatible payload schema.
"""
import os
import base64
import json
import logging
from typing import Optional, Dict, Any
import cv2
import numpy as np
import urllib.request
import urllib.error

logger = logging.getLogger("NvidiaVLMClient")

DEFAULT_VLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_VLM_MODEL = "nvidia/neva-22b"


class NvidiaVLMClient:
    """Client for performing vision-language query requests against NVIDIA VLM endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 10.0
    ) -> None:
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.base_url = (base_url or os.getenv("NVIDIA_VLM_BASE_URL", DEFAULT_VLM_BASE_URL)).rstrip("/")
        self.model = model or os.getenv("NVIDIA_VLM_MODEL", DEFAULT_VLM_MODEL)
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def encode_image_to_base64(image: np.ndarray, format_ext: str = ".jpg") -> str:
        """Converts an OpenCV BGR numpy array into a base64 encoded data string."""
        success, buffer = cv2.imencode(format_ext, image)
        if not success:
            raise ValueError("Failed to encode image to JPEG buffer.")
        return base64.b64encode(buffer).decode("utf-8")

    def analyze_image(self, image: np.ndarray, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Sends an image + prompt query to the VLM endpoint.
        Returns a dict containing the text response and response metadata.
        """
        if not self.api_key:
            logger.warning("NVIDIA_API_KEY is not set. Returning mock/offline VLM client response.")
            return self._mock_offline_response(prompt)

        base64_img = self.encode_image_to_base64(image)
        image_url_payload = f"data:image/jpeg;base64,{base64_img}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url_payload}}
            ]
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 512
        }

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp_bytes = resp.read()
                result = json.loads(resp_bytes.decode("utf-8"))
                
            content = result["choices"][0]["message"]["content"]
            return {
                "content": content,
                "model": self.model,
                "raw": result
            }
        except Exception as e:
            logger.error(f"VLM API HTTP error: {e}")
            return self._mock_offline_response(prompt, error=str(e))

    def _mock_offline_response(self, prompt: str, error: Optional[str] = None) -> Dict[str, Any]:
        """Fallback response when offline or unauthenticated."""
        mock_content = json.dumps({
            "verdict": "SUSPICIOUS",
            "confidence": 0.85,
            "reasoning": "Person observed reaching into jacket pocket and concealing shelf item.",
            "detected_actions": ["concealment", "reach_into_pocket"]
        })
        return {
            "content": mock_content,
            "model": self.model,
            "offline": True,
            "error": error
        }
