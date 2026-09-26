# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Authored view interchange. Invoke native accessors on the UI thread."""
from __future__ import annotations

import copy
import math

TONEMAPPING = ("none", "linear", "filmic", "hejl", "aces", "aces2", "neutral")


def viewer_vector(value):
    # Studio visualizer D=diag(1,-1,-1); gallery import rotates data about Z.
    x, y, z = value
    return [-float(x), float(y), -float(z)]


def capture_camera_path(lf):
    """Return the authored camera track, or None when portal playback should be cleared."""
    path = lf.ui.get_camera_path()
    if path is None:
        return None
    if not isinstance(path, dict):
        raise ValueError("LichtFeld Studio could not read this camera track.")
    return copy.deepcopy(path)


def restore_camera_path(lf, path):
    """Apply only the authored camera track. Other view settings stay unchanged."""
    if path is None:
        lf.ui.clear_keyframes()
        return
    if not isinstance(path, dict) or not lf.ui.set_camera_path(path):
        raise ValueError("LichtFeld Studio could not restore this camera path.")


def capture_view(lf):
    camera, settings = lf.get_camera(), lf.get_render_settings()
    if camera is None or settings is None:
        raise ValueError("Open a splat before capturing its gallery view.")
    if settings.apply_appearance_correction:
        raise ValueError("Appearance correction is not supported by gallery sync yet.")
    if settings.equirectangular:
        raise ValueError("Equirectangular views are not supported by gallery sync yet.")
    exposure = float(getattr(settings, "color_exposure", 1.0))
    tonemapping = str(getattr(settings, "color_tonemapping", "none"))
    profile = str(getattr(settings, "splat_render_profile", "studio"))
    sh_degree = getattr(settings, "sh_degree", 3)
    # Native dynamic settings expose this integral property as text.
    if isinstance(sh_degree, str) and sh_degree in ("0", "1", "2", "3"):
        sh_degree = int(sh_degree)
    if type(sh_degree) not in (int, float) or not 0 <= sh_degree <= 3 or int(sh_degree) != sh_degree:
        raise ValueError("Choose a supported lighting detail before publishing.")
    sh_degree = int(sh_degree)
    if not math.isfinite(exposure) or not 0.1 <= exposure <= 8 or tonemapping not in TONEMAPPING or profile not in ("studio", "standard"):
        raise ValueError("Choose a supported tone mapping and exposure between 0.1 and 8 before publishing.")
    view = {
        "camera": {"position": viewer_vector(camera.eye), "target": viewer_vector(camera.target),
            "up": viewer_vector(camera.up), "fov": float(camera.fov)},
        "verticalFov": True, "cameraPath": capture_camera_path(lf), "renderProfile": profile,
        "shDegree": sh_degree,
        "antialiasing": bool(settings.mip_filter),
        "renderMode": "3dgut" if str(settings.raster_backend) == "3dgut" else "3dgs",
        "background": list(settings.background_color), "exposure": exposure, "tonemapping": tonemapping,
    }
    if lf.is_orthographic():
        viewport = lf.get_current_view()
        extent = float(viewport.ortho_view_extent_world) if viewport is not None else 0.0
        if not math.isfinite(extent) or extent <= 0:
            raise ValueError("Open an orthographic viewport before publishing to the gallery.")
        view["camera"].update(projection="orthographic", orthoScale=extent)
    if getattr(settings, "environment_mode", "SOLID_COLOR") == "EQUIRECTANGULAR":
        exposure_ev = float(settings.environment_exposure)
        rotation = float(settings.environment_rotation_degrees)
        if not settings.environment_map_path or not -6 <= exposure_ev <= 6 or not -360 <= rotation <= 360:
            raise ValueError("Choose a valid HDR background before publishing.")
        view["environment"] = {"exposure": exposure_ev, "rotation": rotation}
    return view


def restore_view(lf, view, *, environment_path=None):
    environment = view.get("environment")
    if environment is not None:
        if (not isinstance(environment, dict) or environment.keys() != {"exposure", "rotation"}
                or any(type(environment[k]) not in (int, float) or not lo <= environment[k] <= hi
                       for k, lo, hi in (("exposure", -6, 6), ("rotation", -360, 360)))
                or not environment_path):
            raise ValueError("The gallery HDR background is invalid or has not been downloaded.")
    profile = view.get("renderProfile", "standard")
    exposure = view.get("exposure", 1.0)
    tonemapping = view.get("tonemapping", "none" if profile == "studio" else "linear")
    sh_degree = view.get("shDegree", 3)
    if type(sh_degree) is not int or not 0 <= sh_degree <= 3:
        raise ValueError("The gallery view contains unsupported lighting detail.")
    if (profile not in ("studio", "standard") or not isinstance(tonemapping, str) or
            tonemapping not in TONEMAPPING or isinstance(exposure, bool) or
            not isinstance(exposure, (int, float)) or not 0.1 <= exposure <= 8):
        raise ValueError("The gallery view contains unsupported color settings.")
    camera = view.get("camera")
    if camera is not None:
        if not isinstance(camera, dict):
            raise ValueError("The gallery view contains an invalid camera.")
        vectors = []
        for name in ("position", "target", "up"):
            value = camera.get(name, [0, 1, 0] if name == "up" else None)
            if not isinstance(value, (list, tuple)) or len(value) != 3 or any(
                    type(x) not in (int, float) or not -1.7976931348623157e308 <= x <= 1.7976931348623157e308 for x in value):
                raise ValueError("The gallery view contains an invalid camera vector.")
            vectors.append(value)
        position, target, up = vectors
        forward = [b-a for a, b in zip(position, target)]
        length, up_length = math.hypot(*forward), math.hypot(*up)
        if not math.isfinite(length) or length < 1e-6 or not math.isfinite(up_length) or up_length < 1e-6:
            raise ValueError("The gallery view contains an invalid camera direction.")
        dot = sum((x/length)*(y/up_length) for x, y in zip(forward, up))
        if abs(dot) >= math.sqrt(1-1e-8):
            raise ValueError("The gallery view contains an invalid camera orientation.")
        fov = camera.get("fov")
        if type(fov) not in (int, float) or not 1 <= fov <= 179:
            raise ValueError("The gallery view contains an invalid camera field of view.")
    vertical_fov = None
    projection = camera.get("projection", "perspective") if camera else "perspective"
    extent = camera.get("orthoScale") if camera else None
    if projection not in ("perspective", "orthographic") or (
            projection == "orthographic" and (type(extent) not in (int, float) or
            not 0 < extent <= 1.7976931348623157e308)) or (
            projection != "orthographic" and camera and "orthoScale" in camera):
        raise ValueError("The gallery view contains unsupported projection settings.")
    if projection == "orthographic":
        viewport = lf.get_current_view()
        if viewport is None or viewport.width <= 0 or viewport.height <= 0:
            raise ValueError("Open a viewport before restoring the gallery camera.")
    if camera:
        vertical_fov = float(camera["fov"])
        if not view.get("verticalFov", False):
            viewport = lf.get_current_view()
            if viewport is None or viewport.width <= 0 or viewport.height <= 0:
                raise ValueError("Open a viewport before restoring the gallery camera.")
            # Portal's default FOV spans the longer viewport dimension.
            aspect = max(1.0, viewport.width / viewport.height)
            vertical_fov = math.degrees(2 * math.atan(math.tan(math.radians(vertical_fov) / 2) / aspect))
        # Validate and apply the native extent before changing the camera pose.
        if projection == "orthographic":
            lf.set_orthographic(True, extent_world=extent)
        lf.set_camera(tuple(viewer_vector(camera["position"])), tuple(viewer_vector(camera["target"])),
            tuple(viewer_vector(camera.get("up", [0, 1, 0]))))
        lf.set_camera_fov(vertical_fov)
    # Fetch the settings snapshot after changing projection so subsequent
    # property writes cannot restore the previous orthographic flag.
    if projection != "orthographic":
        lf.set_orthographic(False)
    settings = lf.get_render_settings()
    settings.apply_appearance_correction = False
    settings.equirectangular = False
    settings.color_exposure = exposure
    settings.color_tonemapping = tonemapping
    settings.splat_render_profile = profile
    settings.sh_degree = sh_degree
    if "antialiasing" in view:
        settings.mip_filter = view["antialiasing"]
    if "renderMode" in view:
        settings.raster_backend = view["renderMode"]
    settings.environment_mode = "SOLID_COLOR"
    if "background" in view:
        settings.background_color = tuple(view["background"])
    if environment is not None:
        settings.environment_map_path = str(environment_path)
        settings.environment_exposure = environment["exposure"]
        settings.environment_rotation_degrees = environment["rotation"]
        settings.environment_mode = "EQUIRECTANGULAR"
    if "cameraPath" in view:
        restore_camera_path(lf, view["cameraPath"])
