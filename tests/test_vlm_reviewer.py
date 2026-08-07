"""
Unit and Integration Tests for VLM Reviewer Package
===================================================
Tests for NvidiaVLMClient, RetailVLMEventReviewer, assessment types,
and PipelineOrchestrator integration.
"""
import os
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.vlm.types import (
    VLMAssessmentVerdict,
    VLMAssessment,
    VLMReviewRequest
)
from src.vlm.client import NvidiaVLMClient
from src.vlm.reviewer import RetailVLMEventReviewer
from src.common.types import BoundingBox, DetectedObject, TrackedPerson, ClassLabel
from src.inference.orchestrator import PipelineOrchestrator
from src.inference.event_bus import EventBus
from src.behavior.types import BehaviorFlag


# ── Fixtures ──

@pytest.fixture
def mock_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    return BoundingBox(x_min=0.2, y_min=0.2, x_max=0.5, y_max=0.8)


@pytest.fixture
def mock_vlm_client():
    client = NvidiaVLMClient(api_key="mock_key", model="nvidia/neva-22b")
    return client


# ── 1. NvidiaVLMClient Tests ──

def test_vlm_client_init_defaults(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    client = NvidiaVLMClient()
    assert client.api_key == ""
    assert "nvidia.com" in client.base_url
    assert client.model == "nvidia/neva-22b"


def test_vlm_client_env_override(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "env_secret_key")
    monkeypatch.setenv("NVIDIA_VLM_MODEL", "custom-vlm-model")
    monkeypatch.setenv("NVIDIA_VLM_BASE_URL", "https://custom.vlm.api/v1")

    client = NvidiaVLMClient()
    assert client.api_key == "env_secret_key"
    assert client.model == "custom-vlm-model"
    assert client.base_url == "https://custom.vlm.api/v1"


def test_encode_image_to_base64(mock_frame):
    b64_str = NvidiaVLMClient.encode_image_to_base64(mock_frame)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0


def test_analyze_image_offline_fallback(mock_frame):
    client = NvidiaVLMClient(api_key="")  # No key -> offline response
    res = client.analyze_image(mock_frame, "Is person suspicious?")
    assert res.get("offline") is True
    assert "content" in res


def test_analyze_image_http_success(mock_frame):
    client = NvidiaVLMClient(api_key="test_key")

    mock_resp_data = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "verdict": "SUSPICIOUS",
                        "confidence": 0.9,
                        "reasoning": "Detected pocket concealment action.",
                        "detected_actions": ["concealment"]
                    })
                }
            }
        ]
    }

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_cm = MagicMock()
        mock_cm.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
        mock_cm.__enter__.return_value = mock_cm
        mock_urlopen.return_value = mock_cm

        res = client.analyze_image(mock_frame, "Check suspicious activity")
        assert "content" in res
        assert res["model"] == "nvidia/neva-22b"
        assert "SUSPICIOUS" in res["content"]


# ── 2. RetailVLMEventReviewer Tests ──

def test_extract_crop(mock_frame, sample_bbox):
    reviewer = RetailVLMEventReviewer()
    crop = reviewer.extract_crop(mock_frame, sample_bbox)
    assert isinstance(crop, np.ndarray)
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_parse_vlm_response_suspicious():
    reviewer = RetailVLMEventReviewer()
    raw_content = json.dumps({
        "verdict": "SUSPICIOUS",
        "confidence": 0.88,
        "reasoning": "Person placed item inside inner jacket pocket.",
        "detected_actions": ["concealment", "pocket_reach"]
    })

    assessment = reviewer._parse_vlm_response(raw_content, {})
    assert assessment.verdict == VLMAssessmentVerdict.SUSPICIOUS
    assert assessment.confidence == 0.88
    assert assessment.risk_boost > 0
    assert "concealment" in assessment.detected_actions


def test_parse_vlm_response_benign():
    reviewer = RetailVLMEventReviewer()
    raw_content = "```json\n" + json.dumps({
        "verdict": "BENIGN",
        "confidence": 0.95,
        "reasoning": "Customer put phone in pocket.",
        "detected_actions": ["use_phone"]
    }) + "\n```"

    assessment = reviewer._parse_vlm_response(raw_content, {})
    assert assessment.verdict == VLMAssessmentVerdict.BENIGN
    assert assessment.confidence == 0.95
    assert assessment.risk_boost < 0


def test_parse_vlm_response_invalid_json():
    reviewer = RetailVLMEventReviewer()
    assessment = reviewer._parse_vlm_response("Invalid response format", {})
    assert assessment.verdict == VLMAssessmentVerdict.INCONCLUSIVE
    assert assessment.confidence == 0.0


def test_review_request(mock_frame, sample_bbox, mock_vlm_client):
    reviewer = RetailVLMEventReviewer(client=mock_vlm_client)
    req = VLMReviewRequest(
        event_id="e1",
        track_id=42,
        behavior_flag="pocket_concealment",
        timestamp_ms=1000.0,
        frame=mock_frame,
        bbox=sample_bbox
    )

    assessment = reviewer.review(req)
    assert isinstance(assessment, VLMAssessment)
    assert assessment.verdict in VLMAssessmentVerdict


# ── 3. PipelineOrchestrator Integration Tests ──

def test_orchestrator_vlm_review_stage(mock_frame, sample_bbox):
    # Setup mocks for required orchestrator sub-components
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [
        DetectedObject(class_label=ClassLabel.PERSON, bbox=sample_bbox, confidence=0.9, track_id=1)
    ]

    mock_tracker = MagicMock()
    mock_person = TrackedPerson(track_id=1, bbox=sample_bbox, confidence=0.9)
    mock_tracker.track.return_value = ([mock_person], [])

    mock_assoc = MagicMock()
    mock_assoc.associate.return_value = {}

    mock_behavior = MagicMock()
    flag = BehaviorFlag(
        behavior_type="pocket_concealment",
        track_id=1,
        confidence=0.85,
        timestamp_ms=100.0
    )
    mock_behavior.analyze.return_value = [flag]

    mock_vlm = MagicMock(spec=RetailVLMEventReviewer)
    mock_assessment = VLMAssessment(
        verdict=VLMAssessmentVerdict.SUSPICIOUS,
        confidence=0.9,
        reasoning="Visual review confirmed suspicious pocket concealment action.",
        risk_boost=22.5
    )
    mock_vlm.review.return_value = mock_assessment

    event_bus = EventBus()
    published_events = []
    event_bus.subscribe("vlm_assessment_event", lambda ev: published_events.append(ev))

    orchestrator = PipelineOrchestrator(
        camera_id="test_cam",
        detector=mock_detector,
        tracker=mock_tracker,
        association_engine=mock_assoc,
        behavior_engine=mock_behavior,
        vlm_reviewer=mock_vlm,
        event_bus=event_bus
    )
    orchestrator.initialize()

    metadata, alerts = orchestrator.process_frame(mock_frame, 0, 0.0)

    # Verify VLM reviewer was called with VLMReviewRequest
    assert mock_vlm.review.called
    assert len(published_events) == 1
    assert published_events[0].verdict == VLMAssessmentVerdict.SUSPICIOUS
