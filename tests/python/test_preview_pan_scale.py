# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mouse movement and preview layout use different coordinate units."""

from types import SimpleNamespace

import pytest

from test_rmlui_image_sources import panel_modules


def event(**values):
    return SimpleNamespace(get_parameter=lambda key, default: str(values.get(key, default)))


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0])
@pytest.mark.parametrize("zoom,rotation", [(1.0, 0), (0.75, 1), (2.0, 3)])
def test_preview_drag_follows_physical_pointer(panel_modules, monkeypatch, scale, zoom, rotation):
    module, _ = panel_modules
    monkeypatch.setattr(module.lf.ui, "get_ui_scale", lambda: scale)
    panel = module.ImagePreviewPanel()
    panel._fit_to_window = False
    panel._zoom = zoom
    panel._rotation_quadrants = rotation
    panel._pan_x, panel._pan_y = 15, -7
    panel._on_img_mousedown(event(button=0, mouse_x=200, mouse_y=150))
    panel._on_img_mousemove(event(mouse_x=280, mouse_y=190))
    assert (panel._pan_x - 15) * scale == pytest.approx(80)
    assert (panel._pan_y + 7) * scale == pytest.approx(40)
    panel._on_img_mouseup(event())
    panel._on_img_mousemove(event(mouse_x=350, mouse_y=200))
    assert (panel._pan_x - 15) * scale == pytest.approx(80)


def test_fit_preview_does_not_pan(panel_modules):
    module, _ = panel_modules
    panel = module.ImagePreviewPanel()
    panel._on_img_mousedown(event(button=0, mouse_x=200, mouse_y=150))
    panel._on_img_mousemove(event(mouse_x=280, mouse_y=190))
    assert panel._pan_x == panel._pan_y == 0
