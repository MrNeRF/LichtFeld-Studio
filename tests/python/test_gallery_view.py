# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gallery camera framing and explicit removal of authored paths."""
import math
from types import SimpleNamespace

import pytest

from lfs_plugins.gallery_view import capture_camera_path, capture_view, restore_camera_path, restore_view, TONEMAPPING


def app(width=1600, height=900):
    calls = []
    settings = SimpleNamespace(apply_appearance_correction=True, equirectangular=True, orthographic=True,
                               background_color=(0, 0, 0), mip_filter=False, raster_backend="3dgs")
    def set_orthographic(enabled, extent_world=None):
        settings.orthographic = enabled
        if extent_world is not None:
            settings.extent = extent_world
    lf = SimpleNamespace(
        get_current_view=lambda: SimpleNamespace(width=width, height=height,
            ortho_view_extent_world=getattr(settings, "extent", 6.0)),
        get_render_settings=lambda: settings,
        set_camera=lambda *args: calls.append(("camera", args)),
        set_camera_fov=lambda fov: calls.append(("fov", fov)),
        set_orthographic=set_orthographic,
        is_orthographic=lambda: settings.orthographic,
        get_camera=lambda: SimpleNamespace(eye=(0, 1, 3), target=(0, 0, 0), up=(0, 1, 0), fov=60),
        ui=SimpleNamespace(clear_keyframes=lambda: calls.append(("clear_path",)),
                           set_camera_path=lambda path: calls.append(("path", path)) or True,
                           get_camera_path=lambda: None),
    )
    return lf, calls, settings


@pytest.mark.parametrize("width,height", [(1600, 900), (900, 1600), (1024, 1024)])
def test_portal_default_fov_preserves_the_longer_axis(width, height):
    lf, calls, _ = app(width, height)
    restore_view(lf, {"camera": {"position": [0, 1, 3], "target": [0, 0, 0], "fov": 60}})
    fov = next(call[1] for call in calls if call[0] == "fov")
    span = math.tan(math.radians(fov) / 2) * max(1, width / height)
    assert span == pytest.approx(math.tan(math.radians(60) / 2))


def test_native_vertical_fov_and_roll_are_restored_without_aspect_conversion():
    lf, calls, settings = app()
    restore_view(lf, {"verticalFov": True, "camera": {
        "position": [-2, 1, -6], "target": [0, 0, 0], "up": [-0.6, 0.8, 0], "fov": 50},
        "antialiasing": True, "renderMode": "3dgut"})
    assert ("camera", ((2., 1., 6.), (0., 0., 0.), (0.6, 0.8, 0.))) in calls
    assert ("fov", 50) in calls
    assert settings.mip_filter and settings.raster_backend == "3dgut"
    assert not settings.apply_appearance_correction and not settings.equirectangular
    assert not settings.orthographic


def test_explicit_empty_path_clears_old_path_but_absent_path_is_unchanged():
    lf, calls, _ = app()
    restore_view(lf, {"cameraPath": None})
    assert calls == [("clear_path",)]
    calls.clear()
    restore_view(lf, {})
    assert calls == []


def test_capture_camera_path_copies_authored_track_and_treats_missing_as_clear():
    lf, _, settings = app()
    settings.apply_appearance_correction = settings.equirectangular = False
    assert capture_camera_path(lf) is None
    authored = {"version": 1, "keyframes": [{"t": 0}], "duration": 2, "loopMode": "once", "playbackSpeed": 1}
    lf.ui.get_camera_path = lambda: authored
    captured = capture_camera_path(lf)
    assert captured == authored and captured is not authored
    authored["duration"] = 9
    assert captured["duration"] == 2
    assert capture_view(lf)["cameraPath"]["duration"] == 9


def test_restore_reports_lichtfeld_studio_when_native_path_is_rejected():
    lf, calls, _ = app()
    lf.ui.set_camera_path = lambda path: False
    with pytest.raises(ValueError, match="LichtFeld Studio could not restore this camera path"):
        restore_view(lf, {"cameraPath": {"version": 1, "keyframes": []}})
    assert calls == []


def test_restore_camera_path_does_not_reset_unrelated_view_settings():
    lf, calls, settings = app()
    settings.apply_appearance_correction = True
    settings.equirectangular = True
    settings.orthographic = True
    settings.background_color = (0.2, 0.3, 0.4)
    path = {"version": 1, "keyframes": [{"t": 0}], "duration": 3, "loopMode": "once", "playbackSpeed": 1}
    restore_camera_path(lf, path)
    assert calls == [("path", path)]
    assert settings.apply_appearance_correction and settings.equirectangular and settings.orthographic
    assert settings.background_color == (0.2, 0.3, 0.4)
    restore_camera_path(lf, None)
    assert ("clear_path",) in calls
    assert settings.background_color == (0.2, 0.3, 0.4)


def test_authored_solid_background_replaces_an_existing_environment_map():
    lf, _, settings = app()
    settings.environment_mode = "EQUIRECTANGULAR"
    settings.environment_map_path = "local-environment.hdr"
    restore_view(lf, {"background": [0.1, 0.2, 0.3]})
    assert settings.environment_mode == "SOLID_COLOR"
    assert settings.background_color == (0.1, 0.2, 0.3)
    assert settings.environment_map_path == "local-environment.hdr"


def test_missing_viewport_rejects_camera_restore_before_mutating_the_app():
    lf, calls, settings = app(0, 0)
    with pytest.raises(ValueError, match="Open a viewport"):
        restore_view(lf, {"camera": {"position": [0, 0, 1], "target": [0, 0, 0], "fov": 60}})
    assert calls == []
    assert settings.apply_appearance_correction and settings.equirectangular and settings.orthographic


def test_publish_rejects_equirectangular_without_changing_the_view():
    lf, calls, settings = app()
    settings.apply_appearance_correction = False
    settings.orthographic = False
    settings.equirectangular = True
    with pytest.raises(ValueError, match="Equirectangular"):
        capture_view(lf)
    assert calls == []
    assert settings.equirectangular


def test_orthographic_world_extent_roundtrips_across_viewport_sizes():
    source, _, settings = app(1600, 900)
    settings.apply_appearance_correction = settings.equirectangular = False
    settings.extent = 6.25
    view = capture_view(source)
    assert view["camera"]["projection"] == "orthographic"
    assert view["camera"]["orthoScale"] == 6.25
    target, _, restored = app(480, 640)
    restore_view(target, view)
    assert restored.orthographic and restored.extent == 6.25
    assert capture_view(target) == view


@pytest.mark.parametrize("extent", [None, True, 0, -1, "6", float("nan"), float("inf"), 10**1000])
def test_invalid_orthographic_extent_is_rejected_before_camera_changes(extent):
    lf, calls, _ = app()
    with pytest.raises(ValueError, match="projection"):
        restore_view(lf, {"camera": {"position": [0, 1, 3], "target": [0, 0, 0],
            "fov": 50, "projection": "orthographic", "orthoScale": extent}})
    assert calls == []


def test_orthographic_restore_requires_viewport_even_with_vertical_fov():
    lf, calls, _ = app(0, 0)
    with pytest.raises(ValueError, match="Open a viewport"):
        restore_view(lf, {"verticalFov": True, "camera": {"position": [0, 1, 3],
            "target": [0, 0, 0], "fov": 50, "projection": "orthographic", "orthoScale": 6}})
    assert calls == []


@pytest.mark.parametrize("fields", [{"position": [0, 0, 0]}, {"up": [0, 0, 0]},
    {"position": [float("nan"), 1, 3]}, {"target": [10**1000, 0, 0]},
    {"fov": float("nan")}, {"fov": True}, {"fov": 180}])
def test_malformed_remote_camera_does_not_mutate_the_view(fields):
    lf, calls, _ = app()
    with pytest.raises(ValueError, match="camera"):
        restore_view(lf, {"camera": {"position": [0, 1, 3], "target": [0, 0, 0], "fov": 50, **fields}})
    assert calls == []


@pytest.mark.parametrize("profile", ["studio", "standard"])
@pytest.mark.parametrize("tone", TONEMAPPING)
def test_color_roundtrip_keeps_tone_exposure_and_blending_order(profile, tone):
    lf, _, settings = app()
    restore_view(lf, {"renderProfile": profile, "tonemapping": tone, "exposure": 4.0})
    assert settings.splat_render_profile == profile
    assert settings.color_exposure == 4.0 and settings.color_tonemapping == tone
    captured = capture_view(lf)
    assert captured["renderProfile"] == profile
    assert captured["exposure"] == 4.0 and captured["tonemapping"] == tone


@pytest.mark.parametrize("view", [{"exposure": True}, {"exposure": 0}, {"exposure": 9},
                                 {"exposure": 10 ** 1000},
                                 {"exposure": float("nan")}, {"exposure": float("inf")},
                                 {"tonemapping": []}, {"tonemapping": "unknown"}, {"renderProfile": "unknown"}])
def test_bad_color_settings_do_not_partially_change_the_view(view):
    lf, calls, settings = app()
    with pytest.raises(ValueError, match="color settings"):
        restore_view(lf, view)
    assert calls == []
    assert settings.apply_appearance_correction and settings.equirectangular and settings.orthographic


def test_missing_color_values_follow_the_portal_profile_defaults():
    lf, _, settings = app()
    restore_view(lf, {})
    assert settings.color_exposure == 1 and settings.color_tonemapping == "linear"
    assert settings.splat_render_profile == "standard"
    restore_view(lf, {"renderProfile": "studio"})
    assert settings.color_exposure == 1 and settings.color_tonemapping == "none"
    assert settings.splat_render_profile == "studio"


@pytest.mark.parametrize("degree", range(4))
def test_overall_lighting_detail_roundtrips_separately_from_geometry(degree):
    lf, _, settings = app()
    restore_view(lf, {"shDegree": degree})
    assert settings.sh_degree == degree
    assert capture_view(lf)["shDegree"] == degree
    settings.sh_degree = str(degree)
    assert capture_view(lf)["shDegree"] == degree
    restore_view(lf, {})
    assert settings.sh_degree == 3


@pytest.mark.parametrize("degree", [None, True, -1, 4, 1.0, "1", float("nan"), []])
def test_invalid_lighting_detail_is_rejected_before_view_mutations(degree):
    lf, calls, settings = app()
    with pytest.raises(ValueError, match="lighting detail"):
        restore_view(lf, {"shDegree": degree})
    assert calls == []
    assert settings.orthographic and settings.apply_appearance_correction


def test_hdr_capture_and_restore_never_publish_the_private_path():
    lf, calls, settings = app()
    settings.apply_appearance_correction = settings.equirectangular = False
    settings.environment_mode = "EQUIRECTANGULAR"
    settings.environment_map_path = "/private/user/home/studio.exr"
    settings.environment_exposure = -1.5
    settings.environment_rotation_degrees = 124
    view = capture_view(lf)
    assert view["environment"] == {"exposure": -1.5, "rotation": 124}
    assert "private" not in str(view)
    restore_view(lf, view, environment_path="/retained/background.lfsenv")
    assert settings.environment_mode == "EQUIRECTANGULAR"
    assert settings.environment_map_path == "/retained/background.lfsenv"
    assert settings.environment_exposure == -1.5
    restore_view(lf, {})
    assert settings.environment_mode == "SOLID_COLOR"


@pytest.mark.parametrize("environment,path", [({"exposure": 0, "rotation": 0}, None),
    ({"exposure": float("nan"), "rotation": 0}, "asset"), ({"exposure": 7, "rotation": 0}, "asset"),
    ({"exposure": 0, "rotation": 361}, "asset"), ({"exposure": True, "rotation": 0}, "asset"),
    ({"exposure": 0, "rotation": 0, "path": "private"}, "asset")])
def test_invalid_hdr_restore_does_not_mutate_scene(environment, path):
    lf, calls, settings = app()
    before = vars(settings).copy()
    with pytest.raises(ValueError):
        restore_view(lf, {"environment": environment}, environment_path=path)
    assert vars(settings) == before and calls == []
