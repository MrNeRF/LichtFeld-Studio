# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pytest configuration and fixtures for lichtfeld module tests."""

import sys
from pathlib import Path

import pytest

# Find the build directory and add to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
SOURCE_MODULE_PATH = PROJECT_ROOT / "src" / "python"

# Add the source Python modules first so tests exercise the working tree.
if SOURCE_MODULE_PATH.exists():
    sys.path.insert(0, str(SOURCE_MODULE_PATH))

# Keep built modules available for compiled extensions like `lichtfeld`.
MODULE_PATH = BUILD_DIR / "src" / "python"
if MODULE_PATH.exists():
    sys.path.insert(1 if SOURCE_MODULE_PATH.exists() else 0, str(MODULE_PATH))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "integration: marks integration tests")


@pytest.fixture(scope="session")
def lf():
    """Import and return the lichtfeld module.

    This fixture is session-scoped so the module is only imported once.
    """
    try:
        import lichtfeld

        return lichtfeld
    except ImportError as e:
        pytest.skip(f"lichtfeld module not available: {e}")


@pytest.fixture
def test_data_dir():
    """Path to the data/ folder with real datasets."""
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        pytest.skip(f"Test data directory not found: {data_dir}")
    return data_dir


@pytest.fixture
def bicycle_dataset(test_data_dir):
    """Path to the bicycle dataset."""
    bicycle = test_data_dir / "bicycle"
    if not bicycle.exists():
        pytest.skip("bicycle dataset not available")
    return bicycle


@pytest.fixture
def benchmark_ply(test_data_dir):
    """Path to a benchmark PLY file if available."""
    ply_path = PROJECT_ROOT / "results" / "benchmark" / "bicycle" / "splat_30000.ply"
    if not ply_path.exists():
        pytest.skip(f"Benchmark PLY not available: {ply_path}")
    return ply_path


@pytest.fixture
def test_sog():
    """Path to test SOG file in tests/data."""
    sog_path = PROJECT_ROOT / "tests" / "data" / "bicycle_ref.sog"
    if not sog_path.exists():
        pytest.skip(f"Test SOG not available: {sog_path}")
    return sog_path


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def numpy():
    """Import numpy."""
    try:
        import numpy as np

        return np
    except ImportError:
        pytest.skip("numpy not available")


@pytest.fixture
def small_tensor(lf, numpy):
    """Create a small test tensor."""
    arr = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
    return lf.Tensor.from_numpy(arr)


@pytest.fixture
def gpu_available(lf):
    """Check if GPU is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        # If torch not available, try creating a CUDA tensor
        try:
            import numpy as np

            arr = np.array([1.0], dtype=np.float32)
            t = lf.Tensor.from_numpy(arr)
            t_cuda = t.cuda()
            return t_cuda.is_cuda
        except Exception:
            return False


@pytest.fixture(scope="session")
def torch():
    """Import PyTorch for interop tests."""
    try:
        import torch

        return torch
    except ImportError:
        pytest.skip("PyTorch not available")
