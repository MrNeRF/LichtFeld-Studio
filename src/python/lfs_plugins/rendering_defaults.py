# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""User defaults for new projects; never implicitly modify a loaded project."""
import json
import math

import lichtfeld as lf
from . import rml_widgets as w

DEFAULTS = {
    "show_coord_axes": False, "show_pivot": False, "show_grid": True,
    "show_camera_frustums": False, "desaturate_unselected": False,
    "desaturate_cropping": False, "hide_outside_depth_box": False,
    "axes_size": 2.0, "grid_opacity": 0.5, "camera_frustum_scale": 0.25,
    "selection_color_preview": [0.0, 0.871, 0.298],
    "selection_color_center_marker": [0.0, 0.604, 0.733],
    "lod_page_pool_splats": 0, "lod_pool_vram_fraction": 0.15,
}
RANGES = {"axes_size": (0.5, 10), "grid_opacity": (0, 1),
          "camera_frustum_scale": (0.01, 10), "lod_page_pool_splats": (0, 100_000_000),
          "lod_pool_vram_fraction": (0.05, 0.9)}


def read_defaults():
    getter = getattr(lf.ui, "get_rendering_defaults_json", None)
    return json.loads(getter()) if getter else dict(DEFAULTS)


def vignette_value(prop):
    theme = getattr(lf.ui, "theme", lambda: None)()
    return getattr(getattr(theme, "vignette", None), prop,
                   {"enabled": False, "intensity": 0.3, "radius": 0.7, "softness": 0.5}[prop])


def set_vignette_value(prop, value, *, persist=True):
    if prop == "enabled":
        lf.ui.set_theme_vignette_enabled(bool(value))
    else:
        values = {key: vignette_value(key) for key in ("intensity", "radius", "softness")}
        values[prop] = float(value)
        args = (values["intensity"], values["radius"], values["softness"])
        if persist:
            lf.ui.set_theme_vignette_style(*args)
        else:
            lf.ui.set_theme_vignette_style(*args, persist=False)


class RenderingDefaultsSection:
    def __init__(self):
        self._values = read_defaults()
        self._handle = None
        self._error = ""
        # Text typed into a field; it is validated and saved only on Enter or blur.
        self._text = {}
        self._escape_revert = w.EscapeRevertController()

    def bind(self, model):
        for prop, default in DEFAULTS.items():
            def getter(p=prop, d=default):
                if p in self._text:
                    return self._text[p]
                value = self._values.get(p, d)
                return w.color_to_hex(value) if isinstance(d, list) else value
            if isinstance(default, bool):
                setter = lambda value, p=prop: self._set(p, value)
            else:
                setter = lambda value, p=prop: self._text.__setitem__(p, str(value))
            model.bind("default_" + prop, getter, setter)
        model.bind_func("rendering_defaults_error", lambda: self._error)
        for prop in ("enabled", "intensity", "radius", "softness"):
            model.bind("theme_vignette_" + prop,
                       lambda p=prop: vignette_value(p),
                       lambda value, p=prop: set_vignette_value(p, value))
        self._handle = model.get_handle()

    def mount(self, doc):
        for element in doc.query_selector_all('input[type="text"]'):
            prop = element.get_attribute("data-value", "")
            if not prop.startswith("default_") or prop[len("default_"):] not in self._text_props():
                continue
            w.bind_committed_text_input(
                element, prop[len("default_"):], escape_revert=self._escape_revert,
                capture=lambda: None, restore=lambda _snapshot, p=prop[len("default_"):]: self._discard(p),
                commit=self._commit)

    @staticmethod
    def _text_props():
        return {prop for prop, default in DEFAULTS.items() if not isinstance(default, bool)}

    def _discard(self, prop):
        if self._text.pop(prop, None) is not None and self._handle:
            self._handle.dirty("default_" + prop)

    def _commit(self, prop):
        text = self._text.pop(prop, None)
        if text is not None:
            self._set(prop, text)

    def _set(self, prop, value):
        try:
            default = DEFAULTS[prop]
            if isinstance(default, list):
                value = w.hex_to_color(str(value))
                if value is None:
                    raise ValueError("Invalid color")
                value = list(w.normalize_color(value))
            elif isinstance(default, bool):
                value = bool(value)
            else:
                value = float(str(value).replace(",", "").strip())
                low, high = RANGES[prop]
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError("Out of range")
                if isinstance(default, int):
                    if not value.is_integer():
                        raise ValueError("Integer required")
                    value = int(value)
            # Persist a patch, preserving other defaults and future fields.
            lf.ui.set_rendering_defaults_json(json.dumps({prop: value}))
            self._values = read_defaults()
            self._error = ""
        except (ValueError, TypeError, OverflowError):
            self._error = lf.ui.tr("preferences.viewport_defaults_invalid")
        if self._handle:
            self._handle.dirty("default_" + prop)
            self._handle.dirty("rendering_defaults_error")

    def sync(self):
        values = read_defaults()
        if values != self._values:
            self._values = values
            if self._handle:
                for prop in DEFAULTS:
                    self._handle.dirty("default_" + prop)

    def reset(self):
        lf.ui.set_rendering_defaults_json(json.dumps(DEFAULTS))
        self._error = ""
        if self._handle:
            self._handle.dirty("rendering_defaults_error")
        self.sync()
