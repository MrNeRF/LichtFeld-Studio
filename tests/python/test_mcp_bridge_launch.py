# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Launch arguments must respect the installed portable CLI contract."""

import importlib.util
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("portable", [False, True])
def test_bridge_keeps_development_only_flags_out_of_windows_portable_launch(
    tmp_path, monkeypatch, portable
):
    script = Path(__file__).resolve().parents[2] / "scripts" / "lichtfeld_mcp_bridge.py"
    spec = importlib.util.spec_from_file_location("bridge_launch_test", script)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    executable = tmp_path / "LichtFeld-Studio.exe"
    executable.touch()
    if portable:
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "bin.manifest").touch()
    monkeypatch.setenv("LFS_EXECUTABLE", str(executable))
    monkeypatch.setattr(bridge, "REPO_ROOT", tmp_path)
    expected = [str(executable)]
    if not (portable and os.name == "nt"):
        expected.append("--no-splash")
    assert bridge.pick_launch_command() == expected
