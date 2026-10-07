# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Subprocess coverage for orderly Vulkan Python interpreter shutdown."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _python_environment(lf):
    build_dir = Path(os.environ.get("LFS_TEST_BUILD_DIR", PROJECT_ROOT / "build"))
    module_dir = Path(lf.__file__).resolve().parent
    python_paths = [
        str(PROJECT_ROOT / "src" / "python"),
        str(module_dir),
        os.environ.get("PYTHONPATH", ""),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(path for path in python_paths if path)
    env["LFS_TEST_BUILD_DIR"] = str(build_dir)
    return env


def _assert_vulkan_selected(lf):
    probe = lf.Tensor.ones([1], device="gpu")
    if probe.backend != "vulkan":
        pytest.skip("the selected tensor backend is not Vulkan")


@pytest.mark.gpu
def test_vulkan_tensor_subprocess_exits_normally(lf):
    _assert_vulkan_selected(lf)
    script = (
        "import lichtfeld as lf; "
        "tensor = lf.Tensor.ones([4], device='gpu'); "
        "assert tensor.backend == 'vulkan'; "
        "assert tensor.sum().cpu().item() == 4.0"
    )
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        cwd=PROJECT_ROOT,
        env=_python_environment(lf),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Vulkan tensor subprocess exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_vulkan_load_subprocess_exits_normally(lf):
    _assert_vulkan_selected(lf)
    test_node = (
        "tests/python/test_io_load.py::"
        "TestLoadResult::test_load_dataset_returns_load_result"
    )
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-m", "pytest", test_node, "-ra", "-q"],
        cwd=PROJECT_ROOT,
        env=_python_environment(lf),
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, (
        f"Vulkan load subprocess exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
