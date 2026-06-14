from pathlib import Path

import pytest

from qmc.sampler.direction_numbers import load_direction_numbers

# Path to the Joe-Kuo direction-number table, resolved relative to the repo
# root so tests run from anywhere.
DN_PATH = Path(__file__).resolve().parent.parent / "src" / "qmc" / "sampler" / "joe_kuo_new_100.txt"


@pytest.fixture(scope="session")
def dn_path() -> str:
    return str(DN_PATH)


@pytest.fixture(scope="session")
def v64(dn_path):
    """Direction numbers for 64 dimensions (loaded once per session)."""
    return load_direction_numbers(dn_path, max_dim=64)
