# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Binding contracts; the round-trip case needs a running viewer.

These tests never start an export, load a backend, or save a project.
"""

import pytest


@pytest.mark.parametrize(
    "backend,preset,fallback",
    [
        ("", "native", "abort"),
        ("UPPERCASE", "quality", "abort"),
        (".example", "quality", "abort"),
        ("a" * 129, "quality", "abort"),
        ("example", "", "abort"),
        ("example", "bad/preset", "abort"),
        ("example", "a" * 129, "abort"),
        ("native", "quality", "abort"),
        ("example", "quality", "unknown"),
    ],
)
def test_invalid_selection_raises_value_error_before_accessing_viewer(lf, backend, preset, fallback):
    with pytest.raises(ValueError):
        lf.ui.set_video_reconstruction_selection(backend, preset, fallback)


def test_unavailable_viewer_is_reported_without_dispatching_export(lf):
    try:
        lf.ui.get_video_reconstruction_selection()
    except RuntimeError as error:
        assert str(error) == "Viewer is unavailable"
    else:
        pytest.skip("This case checks the no-viewer boundary")

    for operation in (
        lambda: lf.ui.set_video_reconstruction_selection("native", "native"),
        lf.ui.reset_video_reconstruction_selection,
        lambda: lf.ui.export_video(1920, 1080, 30, 18, "unused.mp4"),
    ):
        with pytest.raises(RuntimeError, match="Viewer is unavailable"):
            operation()


@pytest.mark.integration
def test_live_selection_round_trip_reset_and_invalid_update_are_atomic(lf):
    try:
        original = lf.ui.get_video_reconstruction_selection()
    except RuntimeError as error:
        if str(error) != "Viewer is unavailable":
            raise
        pytest.skip("Requires a running viewer; run from its Python console")

    try:
        # Metadata-only: an uninstalled backend is a valid saved request.
        lf.ui.set_video_reconstruction_selection("com.example.uninstalled", "quality")
        requested = {"backend_id": "com.example.uninstalled", "preset_id": "quality", "fallback": "abort"}
        assert lf.ui.get_video_reconstruction_selection() == requested
        with pytest.raises(ValueError):
            lf.ui.set_video_reconstruction_selection("native", "quality")
        assert lf.ui.get_video_reconstruction_selection() == requested
        lf.ui.set_video_reconstruction_selection("com.example.uninstalled", "quality", "native")
        assert lf.ui.get_video_reconstruction_selection() == {**requested, "fallback": "native"}
        lf.ui.reset_video_reconstruction_selection()
        assert lf.ui.get_video_reconstruction_selection() == {
            "backend_id": "native", "preset_id": "native", "fallback": "abort"
        }
    finally:
        lf.ui.set_video_reconstruction_selection(**original)
