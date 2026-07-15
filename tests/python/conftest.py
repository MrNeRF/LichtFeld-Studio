# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pytest configuration and fixtures for lichtfeld module tests."""

import hashlib
import os
import sys
import tempfile
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


# The real, user-facing Asset Manager catalog. No test may ever write here.
PRODUCTION_ASSET_CATALOG_DIR = Path.home() / ".lichtfeld" / "asset_manager"


def _asset_catalog_fingerprint():
    """Cheap snapshot of the production catalog: library.json content plus the
    directory listing. Catches both mutated entries and stray new files."""
    if not PRODUCTION_ASSET_CATALOG_DIR.exists():
        return None
    library = PRODUCTION_ASSET_CATALOG_DIR / "library.json"
    library_hash = (
        hashlib.md5(library.read_bytes()).hexdigest() if library.is_file() else None
    )
    listing = sorted(p.name for p in PRODUCTION_ASSET_CATALOG_DIR.iterdir())
    return (library_hash, listing)


def pytest_configure(config):
    """Register custom markers and baseline-isolate the asset catalog."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "integration: marks integration tests")

    # Catch catalog resolution that happens before any function fixture runs
    # (import/collection time). Per-test fixtures override this with their own
    # temp dir; this baseline only guarantees nothing resolves to production.
    session_catalog = Path(tempfile.gettempdir()) / "lfs-test-asset-manager"
    os.environ.setdefault("LFS_ASSET_MANAGER_DIR", str(session_catalog))


@pytest.fixture(scope="session", autouse=True)
def guard_production_asset_catalog():
    """Fail the suite if any test mutates the production Asset Manager catalog.

    This is the enforcement behind "tests must never pollute production": even a
    future test that forgets to isolate the catalog trips this wire."""
    before = _asset_catalog_fingerprint()
    yield
    after = _asset_catalog_fingerprint()
    if after != before:
        pytest.fail(
            "A test mutated the production Asset Manager catalog at "
            f"{PRODUCTION_ASSET_CATALOG_DIR}. Tests must isolate it via the "
            "isolate_asset_manager_catalog fixture.\n"
            f"  before: {before}\n  after:  {after}",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def isolate_asset_manager_catalog(tmp_path, monkeypatch):
    """Redirect the Asset Manager catalog to a per-test temp directory.

    Code paths that resolve the catalog implicitly (e.g.
    register_catalog_asset_path -> load_asset_index) otherwise write into the
    user's real ~/.lichtfeld/asset_manager/library.json, leaving dead entries
    that point at deleted pytest tmp dirs. resolve_asset_manager_storage_path()
    reads this env var first on every call, so the redirect is binding-proof;
    pinning the legacy path to the temp dir suppresses the real-catalog copy.
    """
    catalog_dir = tmp_path / "asset_manager"
    monkeypatch.setenv("LFS_ASSET_MANAGER_DIR", str(catalog_dir))
    try:
        from lfs_plugins import asset_index

        monkeypatch.setattr(asset_index, "LEGACY_STORAGE_PATH", catalog_dir, raising=False)
        monkeypatch.setattr(
            asset_index, "LEGACY_LIBRARY_PATH", catalog_dir / "library.json", raising=False
        )
        monkeypatch.setattr(
            asset_index, "DEFAULT_LIBRARY_PATH", catalog_dir / "library.json", raising=False
        )
    except Exception:
        pass
    return catalog_dir


@pytest.fixture(autouse=True)
def isolate_lichtfeld_module_overrides():
    """Keep test-local module stubs from leaking into later test modules.

    Several UI tests replace ``lichtfeld`` and reload ``lfs_plugins`` against
    that stub. Restoring only ``lichtfeld`` leaves already-imported plugin
    modules holding the stub, making otherwise independent files fail according
    to collection order. Preserve the module graph seen at test entry and put it
    back after the test.
    """
    prefixes = ("lichtfeld", "lfs_plugins")

    def is_managed(name):
        return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)

    before = {name: module for name, module in sys.modules.items() if is_managed(name)}
    yield

    for name in [name for name in sys.modules if is_managed(name) and name not in before]:
        del sys.modules[name]
    sys.modules.update(before)


@pytest.fixture
def bypass_plugin_installer(monkeypatch):
    """Keep manager behavior tests independent from the bundled uv runtime."""
    from lfs_plugins.installer import PluginInstaller

    monkeypatch.setattr(
        PluginInstaller,
        "ensure_venv",
        lambda self, *args, **kwargs: True,
    )
    monkeypatch.setattr(
        PluginInstaller,
        "install_dependencies",
        lambda self, *args, **kwargs: True,
    )


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
