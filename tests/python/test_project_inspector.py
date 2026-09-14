"""Focused binding and CLI checks for native .licht inspection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _fixture() -> Path:
    configured = os.environ.get("LFS_PROJECT_INSPECT_FIXTURE")
    if configured:
        return Path(configured)
    for parent in Path(__file__).parents:
        candidate = parent / ".codex_tmp/am_concept/testhome/projects/project.licht"
        if candidate.is_file():
            return candidate
    return Path(__file__).parents[2] / "missing-project.licht"


@pytest.fixture(scope="module")
def native_io():
    try:
        from lichtfeld import io
    except ImportError as error:
        pytest.skip(f"native lichtfeld.io is unavailable: {error}")
    return io


def test_native_inspector_bindings(native_io):
    path = _fixture()
    if not path.is_file():
        pytest.skip(f"inspection fixture is unavailable: {path}")
    card = native_io.inspect_project_card(path)
    assert card.validation_scope == "head"
    assert card.physical_file_size == path.stat().st_size
    assert native_io.classify_project(path).state == card.open_state
    details = native_io.inspect_project_details(path)
    assert details.card.validation_scope == "structure"
    assert details.storage.physical_bytes == path.stat().st_size
    assert isinstance(native_io.project_storage_stats(path).dead_ratio, float)
    if card.has_preview:
        assert native_io.read_preview(path).startswith(b"\x89PNG")


def test_cli_uses_native_inspector():
    path = _fixture()
    if not path.is_file():
        pytest.skip(f"inspection fixture is unavailable: {path}")
    repo = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{repo / 'src/python'}:{repo / 'build/src/python'}"
    result = subprocess.run(
        [sys.executable, str(repo / "tools/inspect_licht.py"), str(path)],
        cwd=repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "validation scope: head" in result.stdout
    assert "Reclaimable" in result.stdout
