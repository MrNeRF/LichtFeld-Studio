"""Exercise closed-file project operations through the native Python bindings."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


def _fixture() -> Path:
    configured = os.environ.get("LFS_PROJECT_OPERATIONS_FIXTURE")
    if configured:
        return Path(configured)
    for parent in Path(__file__).parents:
        candidate = parent / ".codex_tmp/am_concept/testhome/projects/bonsai.licht"
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


def test_closed_file_operations(native_io, tmp_path):
    source = _fixture()
    if not source.is_file():
        pytest.skip(f"operations fixture is unavailable: {source}")
    path = tmp_path / "copy.licht"
    shutil.copy2(source, path)

    canceled = native_io.verify_project_file(path, cancel=lambda: True)
    assert canceled.status is native_io.ProjectVerificationStatus.CANCELED
    verified = native_io.verify_project_file(path)
    assert verified.status is native_io.ProjectVerificationStatus.VERIFIED

    restored = native_io.restore_save(path, 1, tmp_path / "restored.licht")
    assert restored.project_uuid != native_io.inspect_project_card(path).project_uuid

    png = b"\x89PNG\r\n\x1a\n"
    native_io.set_project_preview(path, png)
    native_io.set_project_license(path, "CC-BY-4.0", "Python test")
    titled = native_io.set_project_title(path, "Python operation copy")
    assert titled.title == "Python operation copy"
    assert native_io.inspect_project_details(path).license.identifier == "CC-BY-4.0"

    compacted = native_io.compact_project_file(path)
    assert "older save points" in compacted.diagnostic
    native_io.clear_project_license(path)
    assert native_io.inspect_project_details(path).license is None
