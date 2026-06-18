# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewport image export controls for the viewport overlay."""

import math
import threading
import time

import lichtfeld as lf

from . import rml_widgets as w


_DEFAULT_FORMAT = "jpg"
_DEFAULT_RESOLUTION = "viewport"
_DEFAULT_CUSTOM_WIDTH = 1920
_DEFAULT_CUSTOM_HEIGHT = 1080
_CUSTOM_MIN = 1
_CUSTOM_MAX = 65536
_STATUS_FADE_SECONDS = 1.95
_RESOLUTION_HEIGHTS = {
    "viewport": 0,
    "1080p": 1080,
    "4k": 2160,
    "8k": 4320,
    "12k": 6480,
    "16k": 8640,
    "20k": 10800,
    "24k": 12960,
    "28k": 15120,
    "32k": 17280,
    "custom": -1,
}


def _ui_label(key: str, fallback: str) -> str:
    tr = getattr(lf.ui, "tr", None)
    if not callable(tr):
        return fallback
    try:
        value = tr(key)
    except Exception:
        return fallback
    if value and value != key:
        return value
    return fallback


def _parse_int(value, fallback):
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    return parsed


def _clamp_int(value, lower=_CUSTOM_MIN, upper=_CUSTOM_MAX):
    return min(max(int(value), lower), upper)


def _normalize_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "checked"}


def _normalize_format(value):
    value = str(value or "").strip().lower()
    if value in {"png"}:
        return "png"
    return "jpg"


def _normalize_resolution(value):
    value = str(value or "").strip().lower()
    return value if value in _RESOLUTION_HEIGHTS else _DEFAULT_RESOLUTION


class ViewportExportControlsController:
    _DIRTY_FIELDS = (
        "viewport_export_tool_label",
        "viewport_export_format_value",
        "viewport_export_resolution_value",
        "viewport_export_has_scene",
        "viewport_export_can_export",
        "viewport_export_show_transparency",
        "viewport_export_transparency",
        "viewport_export_show_custom_size",
        "viewport_export_custom_width_str",
        "viewport_export_custom_height_str",
        "viewport_export_export_label",
        "viewport_export_status_text",
        "viewport_export_has_status",
    )

    def __init__(self, on_visibility_changed=None):
        self._handle = None
        self._on_visibility_changed = on_visibility_changed
        self._visible = False
        self._has_scene = False
        self._is_exporting = False
        self._format = _DEFAULT_FORMAT
        self._resolution = _DEFAULT_RESOLUTION
        self._transparent = False
        self._custom_width = _DEFAULT_CUSTOM_WIDTH
        self._custom_height = _DEFAULT_CUSTOM_HEIGHT
        self._status_text = ""
        self._status_transient = False
        self._status_generation = 0
        self._status_animation_generation = 0
        self._status_deadline = 0.0
        self._status_timer = None
        self._current_doc = None
        self._last_state_key = None

    @property
    def visible(self):
        return self._visible

    def bind_model(self, model):
        model.bind_func(
            "viewport_export_tool_label",
            lambda: _ui_label("toolbar.viewport_export", "Viewport Export"),
        )
        model.bind_func("viewport_export_has_scene", lambda: self._has_scene)
        model.bind_func("viewport_export_can_export", lambda: self._has_scene and not self._is_exporting)
        model.bind_func("viewport_export_show_transparency", lambda: self._format == "png")
        model.bind_func("viewport_export_show_custom_size", lambda: self._resolution == "custom")
        model.bind_func("viewport_export_export_label", lambda: "Export")
        model.bind_func("viewport_export_status_text", lambda: self._status_text)
        model.bind_func("viewport_export_has_status", lambda: bool(self._status_text))
        model.bind(
            "viewport_export_format_value",
            lambda: self._format,
            self._set_format,
        )
        model.bind(
            "viewport_export_resolution_value",
            lambda: self._resolution,
            self._set_resolution,
        )
        model.bind(
            "viewport_export_transparency",
            lambda: self._transparent,
            self._set_transparent,
        )
        model.bind(
            "viewport_export_custom_width_str",
            lambda: str(self._custom_width),
            self._set_custom_width,
        )
        model.bind(
            "viewport_export_custom_height_str",
            lambda: str(self._custom_height),
            self._set_custom_height,
        )
        model.bind_event("viewport_export_action", self._on_action)

        self._handle = model.get_handle()

    def mount(self, doc):
        self._visible = False
        self._last_state_key = None
        self._status_text = ""
        self._status_transient = False
        self._status_deadline = 0.0
        self._status_generation += 1
        self._status_animation_generation = 0
        self._cancel_status_timer()
        self._current_doc = doc

        wrap = doc.get_element_by_id("viewport-export-block")
        if wrap:
            wrap.set_class("hidden", True)
        status = doc.get_element_by_id("viewport-export-status")
        if status:
            status.add_event_listener(
                "animationend",
                lambda _event, element=status: self._finish_status_animation(element),
            )
        self._sync_status_element(doc)

        for input_id in ("viewport-export-width", "viewport-export-height"):
            w.bind_select_all_on_focus(doc.get_element_by_id(input_id))

    def update(self, doc):
        dirty = False
        dirty_reasons = []
        self._current_doc = doc
        wrap = doc.get_element_by_id("viewport-export-block")
        if wrap:
            wrap.set_class("hidden", not self._visible)

        self._refresh_state()
        if self._expire_status_if_needed():
            dirty = True
            dirty_reasons.append("status-expired")
        status_dirty = self._sync_status_element(doc)
        if status_dirty:
            dirty_reasons.append("status")
        state_key = self._state_key()
        if state_key != self._last_state_key:
            self._last_state_key = state_key
            self._dirty_all()
            dirty = True
            dirty_reasons.append("state")
        if status_dirty:
            dirty = True

        return ",".join(dirty_reasons) if dirty else None

    def unmount(self):
        self._handle = None
        self._visible = False
        self._last_state_key = None
        self._status_transient = False
        self._status_deadline = 0.0
        self._status_generation += 1
        self._status_animation_generation = 0
        self._cancel_status_timer()
        self._current_doc = None

    def toggle(self, notify=True):
        self._set_visible(not self._visible, notify=notify)

    def close(self, notify=True):
        self._set_visible(False, notify=notify)

    def _set_visible(self, visible, notify=True):
        visible = bool(visible)
        if self._visible == visible:
            return
        self._visible = visible
        if self._visible:
            self._refresh_state()
        self._dirty_all()
        if not notify:
            return
        callback = self._on_visibility_changed
        if callable(callback):
            try:
                callback()
            except Exception:
                pass

    def _state_key(self):
        return (
            self._visible,
            self._has_scene,
            self._is_exporting,
            self._format,
            self._resolution,
            self._transparent,
            self._custom_width,
            self._custom_height,
            self._status_text,
            self._status_generation,
        )

    def _refresh_state(self):
        self._has_scene = self._scene_available()

    def _scene_available(self):
        getter = getattr(lf, "has_scene", None)
        if callable(getter):
            try:
                return bool(getter())
            except Exception:
                pass
        scene_getter = getattr(lf, "get_scene", None)
        if callable(scene_getter):
            try:
                return scene_getter() is not None
            except Exception:
                return False
        return False

    def _set_format(self, value):
        self._format = _normalize_format(value)
        if self._format != "png":
            self._transparent = False
        self._dirty_all()

    def _set_resolution(self, value):
        previous = self._resolution
        self._resolution = _normalize_resolution(value)
        if self._resolution == "custom" and previous != "custom":
            self._seed_custom_size_from_view()
        self._dirty_all()

    def _set_transparent(self, value):
        self._transparent = _normalize_bool(value) and self._format == "png"
        self._dirty_all()

    def _set_custom_width(self, value):
        self._custom_width = _clamp_int(_parse_int(value, self._custom_width))
        self._dirty_all()

    def _set_custom_height(self, value):
        self._custom_height = _clamp_int(_parse_int(value, self._custom_height))
        self._dirty_all()

    def _seed_custom_size_from_view(self):
        try:
            view = lf.get_current_view()
        except Exception:
            view = None
        width = int(getattr(view, "width", 0) or 0)
        height = int(getattr(view, "height", 0) or 0)
        if width > 0 and height > 0:
            self._custom_width = _clamp_int(width)
            self._custom_height = _clamp_int(height)

    def _export_dimensions(self):
        height = _RESOLUTION_HEIGHTS.get(self._resolution, 0)
        if height < 0:
            return self._custom_width, self._custom_height
        return 0, height

    def _on_action(self, handle, event, args):
        del handle, event
        action = str(args[0]) if args else "export"
        if action == "close":
            self.close()
            return
        if action == "export":
            self._export()

    def _export(self):
        self._refresh_state()
        if self._is_exporting:
            return
        if not self._has_scene:
            self._set_status("Load a scene first.")
            return

        extension = "png" if self._format == "png" else "jpg"
        dialog_name = "save_png_file_dialog" if self._format == "png" else "save_jpg_file_dialog"
        dialog = getattr(lf.ui, dialog_name, None)
        if not callable(dialog):
            self._report_error("Image save dialog is not available.")
            return

        try:
            path = dialog(f"viewport_export.{extension}")
        except Exception as exc:
            self._report_error(str(exc).strip() or "Could not open save dialog.")
            return
        if not path:
            return

        width, height = self._export_dimensions()
        self._is_exporting = True
        self._set_status("Exporting...", transient=False)
        self._dirty_all()
        try:
            result = lf.export_viewport_image(
                path,
                self._format,
                width,
                height,
                self._transparent and self._format == "png",
            )
            result_width = int(result.get("width", 0) or 0)
            result_height = int(result.get("height", 0) or 0)
            if result_width > 0 and result_height > 0:
                self._set_status(f"Saved {result_width} x {result_height}")
            else:
                self._set_status("Saved")
        except Exception as exc:
            self._set_status("Export failed.")
            self._report_error(str(exc).strip() or "Viewport export failed.")
        finally:
            self._is_exporting = False
            self._dirty_all()

    def _set_status(self, text, transient=True):
        self._status_text = str(text or "")
        self._status_transient = bool(transient and self._status_text)
        self._status_deadline = (
            time.monotonic() + _STATUS_FADE_SECONDS
            if self._status_transient
            else 0.0
        )
        self._status_generation += 1
        self._status_animation_generation = 0
        self._schedule_status_expiry()
        self._dirty_all()
        self._sync_status_element(self._current_doc)
        self._request_redraw()

    def _clear_status(self):
        self._status_text = ""
        self._status_transient = False
        self._status_deadline = 0.0
        self._status_generation += 1
        self._status_animation_generation = 0
        self._cancel_status_timer()

    def _expire_status_if_needed(self):
        if not self._status_transient or self._status_deadline <= 0.0:
            return False
        if time.monotonic() < self._status_deadline:
            return False
        self._clear_status()
        return True

    def _sync_status_element(self, doc):
        if doc is None or not hasattr(doc, "get_element_by_id"):
            return False

        element = doc.get_element_by_id("viewport-export-status")
        if not element:
            return False

        if not self._status_text:
            element.set_class("hidden", True)
            element.remove_property("opacity")
            return False

        element.set_text(self._status_text)
        element.set_class("hidden", False)

        if not self._status_transient:
            element.remove_property("opacity")
            return False

        if self._status_animation_generation == self._status_generation:
            return False

        generation = self._status_generation
        self._status_animation_generation = generation
        element.set_attribute("data-toast-generation", str(generation))
        element.set_property("opacity", "1")

        element.animate("opacity", "0", _STATUS_FADE_SECONDS, "quadratic-out", "1")
        return True

    def _finish_status_animation(self, element):
        if element.get_attribute("data-toast-generation", "") != str(self._status_generation):
            return
        self._clear_status()
        element.set_class("hidden", True)
        element.remove_property("opacity")
        self._dirty_all()
        self._request_redraw()

    def _schedule_status_expiry(self):
        self._cancel_status_timer()
        if not self._status_transient or self._status_deadline <= 0.0:
            return
        generation = self._status_generation
        delay = max(0.0, self._status_deadline - time.monotonic()) + 0.05
        timer = threading.Timer(delay, lambda: self._request_status_expiry_update(generation))
        timer.daemon = True
        self._status_timer = timer
        timer.start()

    def _cancel_status_timer(self):
        timer = self._status_timer
        self._status_timer = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

    def _request_status_expiry_update(self, generation):
        def request_update():
            if generation != self._status_generation:
                return
            if self._handle:
                try:
                    w.request_model_update(self._handle)
                except Exception:
                    pass
            self._request_redraw()

        try:
            scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
            if scheduler is None:
                scheduler = getattr(lf.ui, "_run_on_ui_thread", None)
            if callable(scheduler):
                scheduler(request_update)
                return
        except Exception:
            pass

        if threading.current_thread() is threading.main_thread():
            request_update()
        else:
            self._request_redraw()

    def _request_redraw(self):
        request_redraw = getattr(lf.ui, "request_redraw", None)
        if callable(request_redraw):
            try:
                request_redraw()
            except Exception:
                pass

    def _report_error(self, message):
        dialog = getattr(lf.ui, "message_dialog", None)
        if callable(dialog):
            try:
                dialog(
                    _ui_label("viewport_export.operation_failed", "Viewport Export Failed"),
                    message,
                    style="error",
                )
            except Exception:
                pass

    def _dirty_all(self):
        if not self._handle:
            return
        for field in self._DIRTY_FIELDS:
            self._handle.dirty(field)
