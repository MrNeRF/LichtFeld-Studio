# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Python API for lichtfeld.ui.screen."""

import pytest


def test_screen_module_is_registered(lf):
    screen = lf.ui.screen
    assert callable(screen.areas)
    assert callable(screen.editors)
    assert callable(screen.split)
    assert callable(screen.join)
    assert callable(screen.close)
    assert callable(screen.swap)
    assert callable(screen.set_editor)
    assert callable(screen.open_editor)
    assert callable(screen.close_editor)
    assert callable(screen.toggle_maximized)
    assert callable(screen.reset)
    assert callable(screen.active_view)
    assert callable(screen.set_active_view)
    assert callable(screen.view_command)
    assert callable(screen.view_camera)
    assert callable(screen.set_view_camera)
    assert callable(screen.view_settings)
    assert callable(screen.set_view_settings)


def test_screen_split_rejects_invalid_direction(lf):
    with pytest.raises(ValueError, match="direction"):
        lf.ui.screen.split(1, direction="diagonal")


def test_screen_split_rejects_invalid_factor(lf):
    with pytest.raises(ValueError, match="factor"):
        lf.ui.screen.split(1, factor=0.0)
    with pytest.raises(ValueError, match="factor"):
        lf.ui.screen.split(1, factor=1.0)


def test_set_view_settings_rejects_unknown_fields(lf):
    with pytest.raises(ValueError, match="Unknown view setting"):
        lf.ui.screen.set_view_settings(1, not_a_field=True)


def test_screen_queries_are_empty_without_a_live_viewer(lf):
    assert lf.ui.screen.areas() == []
    assert lf.ui.screen.editors() == []
    assert lf.ui.screen.active_view() == 0
    assert lf.ui.screen.split(1) == 0
    assert lf.ui.screen.join(1, 2) is False
    assert lf.ui.screen.close(1) is False
    assert lf.ui.screen.swap(1, 2) is False
    assert lf.ui.screen.set_editor(1, "console") is False
    assert lf.ui.screen.open_editor("console") == 0
    assert lf.ui.screen.close_editor("console") is False
    assert lf.ui.screen.toggle_maximized(1) is False
    assert lf.ui.screen.set_active_view(1) is False
    assert lf.ui.screen.view_command(1, "overlay:grid") is False
    assert lf.ui.screen.view_camera(1) == {}
    assert lf.ui.screen.view_settings(1) == {}
    lf.ui.screen.reset()


@pytest.mark.integration
def test_screen_layout_and_view_settings_round_trip(lf):
    areas = lf.ui.screen.areas()
    if not areas:
        pytest.skip("screen API requires an initialized LichtFeld GUI")

    views = [a for a in areas if a["is_view"]]
    assert views, "default screen has a 3D view"
    view = views[0]["id"]
    assert lf.ui.screen.active_view() == view

    editors = {e["id"]: e for e in lf.ui.screen.editors()}
    assert "view3d" in editors
    assert editors["view3d"]["multi_instance"] is True

    added = lf.ui.screen.split(view, direction="vertical", factor=0.4)
    assert added > 0
    after_split = lf.ui.screen.areas()
    assert len(after_split) == len(areas) + 1
    added_area = next(area for area in after_split if area["id"] == added)
    assert added_area["width"] > 0.0
    assert added_area["height"] > 0.0

    settings = lf.ui.screen.view_settings(view)
    assert "show_grid" in settings
    original_grid = settings["show_grid"]
    lf.ui.screen.set_view_settings(view, show_grid=not original_grid)
    assert lf.ui.screen.view_settings(view)["show_grid"] is (not original_grid)
    lf.ui.screen.set_view_settings(view, show_grid=original_grid)

    camera = lf.ui.screen.view_camera(view)
    assert len(camera["rotation"]) == 9
    assert len(camera["translation"]) == 3
    assert lf.ui.screen.set_view_camera(view, (4.0, 3.0, 4.0), (0.0, 0.0, 0.0))
    moved = lf.ui.screen.view_camera(view)
    assert moved["translation"] == pytest.approx((4.0, 3.0, 4.0), abs=1e-4)

    assert lf.ui.screen.view_command(view, "overlay:grid")
    assert lf.ui.screen.join(view, added) or lf.ui.screen.close(added)
    lf.ui.screen.reset()
    restored = lf.ui.screen.areas()
    assert any(a["editor"] == "view3d" for a in restored)
    assert any(a["editor"] == "scene" for a in restored)
