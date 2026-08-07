"""
Retail VLM Event Reviewer
=========================
Performs secondary visual verification on suspicious retail events flagged by
the primary Computer Vision pipeline (e.g. concealment, loitering, shelf sweeps).
Utilizes NvidiaVLMClient to analyze spatial crops and return structured visual verdicts.
"""
import json
import logging
from typing import Optional, List, Dict, Any
import numpy as np

from src.vlm.types import VLMAssessment, VLMAssessmentVerdict, VLMReviewRequest
from src.vlm.client import NvidiaVLMClient
from src.common.types import BoundingBox

logger = logging.getLogger("RetailVLMEventReviewer")

SYSTEM_PROMPT = (
    "You are an expert AI retail security supervisor. Analyze the provided retail image crop "
    "and evaluate whether a suspicious loss prevention event is occurring (e.g., concealment in clothing/bag, "
    "unusual loitering, shelf sweeping, or normal shopping). Respond strictly in valid JSON format."
)

REVIEW_PROMPT_TEMPLATE = (
    "Review person track ID {track_id} flagged for possible suspicious behavior: '{flag_name}'.\n"
    "Respond with a JSON object containing:\n"
    "- \"verdict\": one of \"SUSPICIOUS\", \"BENIGN\", or \"INCONCLUSIVE\"\n"
    "- \"confidence\": confidence score between 0.0 and 1.0\n"
    "- \"reasoning\": brief 1-2 sentence explanation of visual observations\n"
    "- \"detected_actions\": list of observed micro-actions (e.g. [\"pocket_reach\", \"item_concealment\"])\n"
)


class RetailVLMEventReviewer:
    """Secondary VLM event validator for suspicious retail activity."""

    def __init__(
        self,
        client: Optional[NvidiaVLMClient] = None,
        crop_padding_ratio: float = 0.15,
        min_confidence_threshold: float = 0.5
    ) -> None:
        self.client = client or NvidiaVLMClient()
        self.crop_padding_ratio = crop_padding_ratio
        self.min_confidence_threshold = min_confidence_threshold

    def extract_crop(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Extracts a padded sub-image region around the target bounding box."""
        h, w = frame.shape[:2]
        
        bw = bbox.x_max - bbox.x_min
        bh = bbox.y_max - bbox.y_min

        pad_x = bw * self.crop_padding_ratio
        pad_y = bh * self.crop_padding_ratio

        x1 = int(max(0, (bbox.x_min - pad_x) * w))
        y1 = int(max(0, (bbox.y_min - pad_y) * h))
        x2 = int(min(w, (bbox.x_max + pad_x) * w))
        y2 = int(min(h, (bbox.y_max + pad_y) * h))

        if x2 <= x1 or y2 <= y1:
            return frame

        return frame[y1:y2, x1:x2]

    def review(self, request: VLMReviewRequest) -> VLMAssessment:
        """Executes VLM visual review on a VLMReviewRequest container."""
        image_to_analyze = request.crop
        if image_to_analyze is None and request.bbox is not None:
            image_to_analyze = self.extract_crop(request.frame, request.bbox)
        if image_to_analyze is None:
            image_to_analyze = request.frame

        prompt = REVIEW_PROMPT_TEMPLATE.format(
            track_id=request.track_id,
            flag_name=request.behavior_flag
        )
        if request.context_text:
            prompt += f"\nContext details: {request.context_text}"

        raw_result = self.client.analyze_image(image_to_analyze, prompt, system_prompt=SYSTEM_PROMPT)
        content_str = raw_result.get("content", "")

        return self._parse_vlm_response(content_str, raw_result)

    def _parse_vlm_response(self, content_str: str, raw_meta: Dict[str, Any]) -> VLMAssessment:
        """Parses JSON text response from VLM into VLMAssessment object."""
        try:
            # Handle potential markdown code fences in VLM outputs
            clean_str = content_str.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.startswith("```"):
                clean_str = clean_str[3:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            clean_str = clean_str.strip()

            parsed = json.loads(clean_str)
            verdict_str = str(parsed.get("verdict", "INCONCLUSIVE")).upper()
            try:
                verdict = VLMAssessmentVerdict[verdict_str]
            except KeyError:
                verdict = VLMAssessmentVerdict.INCONCLUSIVE

            confidence = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", "VLM analysis completed."))
            detected_actions = list(parsed.get("detected_actions", []))

            risk_boost = 0.0
            if verdict == VLMAssessmentVerdict.SUSPICIOUS:
                risk_boost = 25.0 * confidence
            elif verdict == VLMAssessmentVerdict.BENIGN:
                risk_boost = -15.0 * confidence

            return VLMAssessment(
                verdict=verdict,
                confidence=confidence,
                reasoning=reasoning,
                detected_actions=detected_actions,
                risk_boost=risk_boost,
                raw_response=content_str,
                metadata=raw_meta
            )
        except Exception as e:
            logger.error(f"Failed to parse VLM output JSON ({e}). Content: {content_str}")
            return VLMAssessment(
                verdict=VLMAssessmentVerdict.INCONCLUSIVE,
                confidence=0.0,
                reasoning=f"VLM parsing error: {e}",
                detected_actions=[],
                risk_boost=0.0,
                raw_response=content_str,
                metadata=raw_meta
            )
