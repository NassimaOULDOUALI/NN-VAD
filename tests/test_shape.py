"""Minimal sanity tests (fill in as modules are implemented)."""
import numpy as np
from src.data.dataset import speaker_of


def test_speaker_of():
    assert speaker_of("103-1240-0001") == "103"
