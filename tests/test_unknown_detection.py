"""Tests for unknown product detector."""
import pytest
import tempfile
import os
import numpy as np
from src.product_recognition.unknown_detector import UnknownDetector
from src.product_recognition.types import RecognitionResult


def test_unknown_detector():
    with tempfile.TemporaryDirectory() as tmpdir:
        detector = UnknownDetector(tmpdir)
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        result = RecognitionResult(track_id=1, recognized=False, similarity=0.4)
        emb = np.zeros(384, dtype=np.float32)

        uid = detector.process_unknown(crop, result, emb)
        assert uid is not None
        assert os.path.exists(os.path.join(tmpdir, f"{uid}.jpg"))
        assert os.path.exists(os.path.join(tmpdir, f"{uid}.json"))
