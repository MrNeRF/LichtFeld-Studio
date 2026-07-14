# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ground-truth comparison controls for the viewport overlay."""

import lichtfeld as lf


_DEFAULT_MODE = "rgb"


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


def _normalize_mode(value):
    value = str(value or "").strip().lower()
    if value in {"normal", "normals"}:
        return "normal"
    if value == "depth":
        return "depth"
    return _DEFAULT_MODE


class GTCompareControlsController:
    _DIRTY_FIELDS = (
        "gt_compare_tool_label",
        "gt_compare_mode_value",
    )

    def __init__(self):
        self._handle = None
        self._visible = False
        self._mode = _DEFAULT_MODE
        self._last_state_key = None

    @property
    def visible(self):
        return self._visible

    def bind_model(self, model):
        model.bind_func("gt_compare_tool_label", lambda: _ui_label("status_bar.gt_compare", "GT Compare"))
        model.bind(
            "gt_compare_mode_value",
            lambda: self._mode,
            self._set_mode,
        )
        self._handle = model.get_handle()

    def mount(self, doc):
        self._visible = False
        self._last_state_key = None
        wrap = doc.get_element_by_id("gt-compare-mode-block")
        if wrap:
            wrap.set_class("hidden", True)

    def update(self, doc):
        dirty = False
        dirty_reasons = []
        visible = self._gt_compare_active()
        wrap = doc.get_element_by_id("gt-compare-mode-block")
        if wrap:
            wrap.set_class("hidden", not visible)

        if visible != self._visible:
            self._visible = visible
            dirty = True
            dirty_reasons.append("visibility")

        if not visible:
            self._last_state_key = None
            return ",".join(dirty_reasons) if dirty else None

        self._mode = self._read_mode()
        state_key = self._mode
        if state_key != self._last_state_key:
            self._last_state_key = state_key
            self._dirty_all()
            dirty = True
            dirty_reasons.append("mode")
        return ",".join(dirty_reasons) if dirty else None

    def unmount(self):
        self._handle = None
        self._visible = False
        self._last_state_key = None

    def _gt_compare_active(self):
        getter = getattr(lf.ui, "get_split_view_mode", None)
        if not callable(getter):
            return False
        try:
            return getter() == "gt_comparison"
        except Exception:
            return False

    def _read_mode(self):
        getter = getattr(lf.ui, "get_gt_comparison_mode", None)
        if not callable(getter):
            return _DEFAULT_MODE
        try:
            return _normalize_mode(getter())
        except Exception:
            return _DEFAULT_MODE

    def _set_mode(self, value):
        mode = _normalize_mode(value)
        setter = getattr(lf.ui, "set_gt_comparison_mode", None)
        if callable(setter):
            try:
                setter(mode)
            except Exception:
                pass
            self._mode = self._read_mode()
        else:
            self._mode = _DEFAULT_MODE
        self._dirty_all()

    def _dirty_all(self):
        if not self._handle:
            return
        for field in self._DIRTY_FIELDS:
            self._handle.dirty(field)
