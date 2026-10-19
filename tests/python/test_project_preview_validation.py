# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Project previews accept PNG data and reject other nonempty payloads."""

import shutil
import struct
import zlib
from pathlib import Path

import pytest


def _png_1x1():
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def project_path(tmp_path):
    source = Path(__file__).parents[1] / "data" / "portable-sog.licht"
    if not source.is_file():
        pytest.skip(f"project fixture is unavailable: {source}")
    destination = tmp_path / "preview.licht"
    shutil.copyfile(source, destination)
    return destination


def _remove_existing_preview(native_io, project_path):
    native_io.reduce_size(
        project_path,
        {"drop_unbound_checkpoints": False, "drop_embedded_dataset": False, "drop_thumbnail": True},
    )


@pytest.fixture
def native_io():
    try:
        from lichtfeld import io
    except ImportError as error:
        pytest.skip(f"native lichtfeld.io is unavailable: {error}")
    return io


def test_set_project_preview_rejects_non_png_bytes(native_io, project_path):
    _remove_existing_preview(native_io, project_path)

    with pytest.raises(ValueError, match="png_bytes must contain a valid PNG image"):
        native_io.set_project_preview(project_path, b"not-png")

    assert not native_io.inspect_project_card(project_path).has_preview


def test_set_project_preview_accepts_png_and_empty_bytes_clear(native_io, project_path):
    _remove_existing_preview(native_io, project_path)
    png = _png_1x1()
    native_io.set_project_preview(project_path, png)
    assert native_io.read_preview(project_path) == png

    _remove_existing_preview(native_io, project_path)
    native_io.set_project_preview(project_path, b"")
    assert not native_io.inspect_project_card(project_path).has_preview
