"""Shared pytest fixtures."""
import os
import pytest

# Root of the provided corpus. Override with VAD_DATA_ROOT if needed.
DEFAULT_ROOT = os.environ.get("VAD_DATA_ROOT", "data/vad_data")


@pytest.fixture(scope="session")
def root():
    if not os.path.isdir(DEFAULT_ROOT):
        pytest.skip(f"provided corpus not found at {DEFAULT_ROOT} "
                    f"(set VAD_DATA_ROOT)")
    return DEFAULT_ROOT
