# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""User defaults for new projects; never implicitly modify a loaded project."""
import json
import math

import lichtfeld as lf
from . import rml_widgets as w
from .default_reset import LabelDefaultReset

DEFAULTS = {
    "show_coord_axes": False, "show_pivot": False, "show_grid": True,
    "show_camera_frustums": False, "desaturate_unselected": False, "selection_outline": False,
    "selection_outline_color": [0.0, 0.604, 0.733],
    "selection_outline_opacity": 1.0,
    "selection_outline_width": 2.0,
    "selection_outline_threshold": 0.5,
    "selection_outline_holes": True,
    "desaturate_cropping": False, "hide_outside_depth_box": False,
    "axes_size": 2.0, "grid_opacity": 0.5, "camera_frustum_scale": 0.25,
    "selection_color_preview": [0.0, 0.871, 0.298],
    "selection_color_center_marker": [0.0, 0.604, 0.733],
    "lod_page_pool_splats": 0, "lod_pool_vram_fraction": 0.15,
}
COLOR_PROPS = tuple(prop for prop, default in DEFAULTS.items() if isinstance(default, list))
PICKER_CHANNELS = ("default_picker_r", "default_picker_g", "default_picker_b")


def swatch_binding(prop):
    return f"default_{prop}_swatch"

RANGES = {"selection_outline_opacity": (0, 1), "selection_outline_width": (1, 12), "selection_outline_threshold": (0.05, 0.95), "axes_size": (0.5, 10), "grid_opacity": (0, 1),
          "camera_frustum_scale": (0.01, 10), "lod_page_pool_splats": (0, 100_000_000),
          "lod_pool_vram_fraction": (0.05, 0.9)}


def read_defaults():
    getter = getattr(lf.ui, "get_rendering_defaults_json", None)
    return json.loads(getter()) if getter else dict(DEFAULTS)


def confirm_apply_defaults(on_applied=None):
    """Apply saved defaults to the open project only after explicit confirmation."""
    tr = lf.ui.tr
    apply_label = tr("common.apply")

    def apply_if_confirmed(button):
        if button == apply_label:
            lf.ui.apply_rendering_defaults()
            if on_applied:
                on_applied()

    lf.ui.confirm_dialog(tr("rendering_panel.apply_defaults"),
                         tr("rendering_panel.apply_defaults_confirm"),
                         [apply_label, tr("common.cancel")], apply_if_confirmed)


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
        # Colour whose picker is open; a drag previews as text and saves on release.
        self._color_editing = ""
        self._reveal_picker = False
        self._doc = None
        self._escape_revert = w.EscapeRevertController()
        self._default_reset = LabelDefaultReset(lambda prop: DEFAULTS.get(prop), self._apply_reset)

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
        for prop in COLOR_PROPS:
            model.bind_func(swatch_binding(prop), lambda p=prop: self._swatch_css(p))
        model.bind_func("default_color_editing", lambda: self._color_editing)
        for i, name in enumerate(PICKER_CHANNELS):
            model.bind_func(name, lambda i=i: float(self._color(self._color_editing)[i]) if self._color_editing else 0.0)
        model.bind_event("default_color_click", self._toggle_color_picker)
        model.bind_event("default_picker_change", self._preview_picked_color)
        model.bind_event("default_color_commit", lambda *_: self._commit(self._color_editing))
        for prop in ("enabled", "intensity", "radius", "softness"):
            model.bind("theme_vignette_" + prop,
                       lambda p=prop: vignette_value(p),
                       lambda value, p=prop: set_vignette_value(p, value))
        model.bind_event("apply_viewport_defaults", self._confirm_apply)
        self._handle = model.get_handle()

    def mount(self, doc):
        self._default_reset.mount(doc)
        self._doc = doc
        for element in doc.query_selector_all('input[type="text"]'):
            prop = element.get_attribute("data-value", "")
            if not prop.startswith("default_") or prop[len("default_"):] not in self._text_props():
                continue
            w.bind_committed_text_input(
                element, prop[len("default_"):], escape_revert=self._escape_revert,
                capture=lambda: None, restore=lambda _snapshot, p=prop[len("default_"):]: self._discard(p),
                commit=self._commit_text)

    @staticmethod
    def _text_props():
        return {prop for prop, default in DEFAULTS.items() if not isinstance(default, bool)}

    def _color(self, prop):
        pending = w.hex_to_color(self._text[prop]) if prop in self._text else None
        return pending or self._values.get(prop, DEFAULTS[prop])

    def _swatch_css(self, prop):
        color = self._color(prop)
        return "rgb({},{},{})".format(*(w.color_channel_byte(color, i) for i in range(3)))

    def _toggle_color_picker(self, _handle, _event, args):
        prop = str(args[0]) if args else ""
        self._commit(self._color_editing)
        self._color_editing = "" if prop == self._color_editing else prop
        self._reveal_picker = bool(self._color_editing)
        if self._handle:
            self._handle.dirty("default_color_editing")
        self._dirty_picker()

    def _dirty_picker(self):
        if self._handle:
            for name in PICKER_CHANNELS:
                self._handle.dirty(name)

    def _preview_picked_color(self, _handle, event, _args):
        prop = self._color_editing
        if not event or not prop:
            return
        self._text[prop] = w.color_to_hex(
            [float(event.get_parameter(channel, "0")) for channel in ("red", "green", "blue")])
        self._dirty_value(prop)

    def _dirty_value(self, prop):
        if not self._handle:
            return
        self._handle.dirty("default_" + prop)
        if prop in COLOR_PROPS:
            self._handle.dirty(swatch_binding(prop))

    def _discard(self, prop):
        if self._text.pop(prop, None) is not None:
            self._dirty_value(prop)

    def _commit(self, prop):
        text = self._text.pop(prop, None)
        if text is not None:
            self._set(prop, text)

    def _commit_text(self, prop):
        self._commit(prop)
        if prop == self._color_editing:
            self._dirty_picker()

    def _apply_reset(self, prop, value):
        self._discard(prop)
        self._set(prop, value)
        return not self._error

    def unmount(self):
        self._default_reset.unmount()
        self._doc = None

    def _set(self, prop, value):
        try:
            default = DEFAULTS[prop]
            if isinstance(default, list):
                value = value if isinstance(value, (tuple, list)) else w.hex_to_color(str(value))
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
        self._dirty_value(prop)
        if self._handle:
            self._handle.dirty("rendering_defaults_error")

    def sync(self):
        values = read_defaults()
        if values != self._values:
            self._values = values
            for prop in DEFAULTS:
                self._dirty_value(prop)
        # A picker opened near the bottom of the page scrolls into view once it is laid out.
        picker = self._doc.get_element_by_id("picker-" + self._color_editing) if self._reveal_picker and self._doc else None
        if picker is not None and picker.client_height > 0:
            picker.scroll_into_view(False)
            self._reveal_picker = False

    def _confirm_apply(self, *_):
        # A click can arrive before the focused field's blur listener commits.
        for prop in tuple(self._text):
            self._commit(prop)
            if self._error:
                return
        if not self._error:
            confirm_apply_defaults()

    def reset(self):
        lf.ui.set_rendering_defaults_json(json.dumps(DEFAULTS))
        self._error = ""
        if self._handle:
            self._handle.dirty("rendering_defaults_error")
        self.sync()
