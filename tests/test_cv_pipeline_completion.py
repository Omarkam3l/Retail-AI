"""Tests for CV Pipeline Completion features (track recovery, appearance matching, prevent duplicate pickup)."""
import pytest
import numpy as np
from typing import Dict, Any, List
from src.common.types import BoundingBox, DetectedObject, TrackedPerson, ClassLabel, FrameMetadata
from src.product_recognition.types import RecognitionResult, MatchResult
from src.association.recovery import AssociationRecoveryEngine
from src.association.matcher import SpatialMatcher, calculate_overlap_ratio


def test_overlap_ratio():
    # Large box containing a small box
    big = BoundingBox(0.0, 0.0, 1.0, 1.0)
    small = BoundingBox(0.1, 0.1, 0.3, 0.3)
    ratio = calculate_overlap_ratio(big, small)
    assert ratio == pytest.approx(1.0)


def test_track_recovery():
    recovery = AssociationRecoveryEngine(recovery_threshold=0.5, max_recovery_age_frames=10)
    emb = np.ones(384, dtype=np.float32)
    bbox = BoundingBox(0.1, 0.1, 0.3, 0.3)

    recovery.record_inactive(
        track_id=12,
        class_label=ClassLabel.SHELF_ITEM,
        bbox=bbox,
        embedding=emb,
        frame_index=10,
        timestamp_ms=1000.0
    )

    # Attempt recovery with same class and close spatial box
    obj = DetectedObject(ClassLabel.SHELF_ITEM, BoundingBox(0.12, 0.12, 0.32, 0.32), 0.9)
    recovered_id = recovery.attempt_recovery(obj, emb, frame_index=11, timestamp_ms=1050.0)
    assert recovered_id == 12


def test_appearance_aware_matching():
    matcher = SpatialMatcher(alpha=0.4, beta=0.6, gamma=0.0)
    persons = [TrackedPerson(track_id=1, bbox=BoundingBox(0.0, 0.0, 1.0, 1.0), confidence=0.9)]
    objects = [DetectedObject(ClassLabel.SHELF_ITEM, BoundingBox(0.4, 0.4, 0.6, 0.6), 0.9, track_id=2)]
    
    # Associate
    emb = np.ones(384, dtype=np.float32)
    object_embeddings = {2: emb}
    
    matches = matcher.match(persons, objects, object_embeddings)
    assert len(matches) == 1
    assert matches[0][0] == 1  # person 1
    assert matches[0][1] == 2  # object 2
