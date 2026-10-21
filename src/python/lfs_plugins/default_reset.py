# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Delegated label resets shared by retained-mode settings panels."""
import math
import weakref

import lichtfeld as lf


def values_match(left, right):
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(values_match(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8)
    return left == right


class LabelDefaultReset:
    """Resolve defaults on demand and let each panel enforce its edit contract."""

    def __init__(self, default_value, apply_value):
        self._default_value = self._weak_callback(default_value)
        self._apply_value = self._weak_callback(apply_value)
        self._doc = None
        self._owner_attribute = f"data-reset-owner-{id(self)}"
        self._generation = 0

    @staticmethod
    def _weak_callback(callback):
        if getattr(callback, "__self__", None) is None:
            return callback
        reference = weakref.WeakMethod(callback)

        def invoke(*args):
            method = reference()
            return method(*args) if method is not None else None

        return invoke

    def mount(self, doc):
        already_bound = doc.has_attribute(self._owner_attribute)
        self._doc = doc
        self._generation += 1
        doc.set_attribute(self._owner_attribute, str(self._generation))
        if not already_bound:
            doc.add_event_listener("dblclick", self._on_double_click)
            doc.add_event_listener("mouseover", self._on_hover)

    def unmount(self):
        # Listeners are owned by the document; never retain elements from old rows.
        self._doc = None

    def _label(self, event):
        element = event.target()
        while element is not None and element != self._doc:
            if element.has_attribute("data-reset-prop"):
                return element, element.get_attribute("data-reset-prop", "")
            if any(element.is_class_set(name) for name in (
                    "setting-row__label-col", "prop-label", "preferences-label", "control-label")):
                row = self._row(element)
                if row is not None and self._owns(row) and self._enabled(element):
                    # New registry-backed rows inherit the behavior. A panel's
                    # resolver must explicitly accept the inferred property.
                    for control in row.query_selector_all("input, select, .scrub-field"):
                        for attribute in ("data-pv-id", "data-prop", "data-value", "data-checked"):
                            prop = control.get_attribute(attribute, "")
                            if not prop or "." in prop:
                                continue
                            if prop.startswith("default_"):
                                prop = prop[len("default_"):]
                            if prop.endswith("_hex"):
                                prop = prop[:-4]
                            try:
                                value = self._default_value(prop)
                            except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
                                continue
                            if value is not None:
                                # A reactive row can be reused for another
                                # binding. Never cache an inferred property
                                # on its label or reset may target the old row.
                                return element, prop
            element = element.parent()
        return None

    @staticmethod
    def _row(label):
        element = label.parent()
        while element is not None:
            if element.has_attribute("data-reset-row") or any(element.is_class_set(name) for name in ("setting-row", "preferences-row")):
                return element
            element = element.parent()
        return None

    @staticmethod
    def _enabled(element):
        while element is not None:
            if (element.has_attribute("disabled") or element.has_attribute("readonly")
                    or element.get_attribute("aria-disabled", "") == "true"
                    or any(element.is_class_set(name) for name in ("disabled", "disabled-overlay", "hidden"))):
                return False
            element = element.parent()
        return True

    def _context(self, event):
        if self._doc is None:
            return None
        target = self._label(event)
        if target is None:
            return None
        label, prop = target
        if not self._enabled(label):
            return None
        row = self._row(label)
        if row is None:
            return None
        controls = row.query_selector_all("input, select, .scrub-field, .color-swatch")
        # A disabled input can sit beside an enabled label. Do not bypass its lock.
        if not controls or any(not self._enabled(control) for control in controls):
            return None
        if not self._owns(row):
            return None
        return label, row, controls, prop

    def _owns(self, element):
        while element is not None and not element.has_attribute(self._owner_attribute):
            element = element.parent()
        return element is not None and element.get_attribute(self._owner_attribute, "") == str(self._generation)

    def _on_double_click(self, event):
        if int(event.get_parameter("button", "0")) != 0:
            return
        context = self._context(event)
        if context is None:
            return
        label, row, controls, prop = context
        try:
            value = self._default_value(prop)
            if value is None or self._apply_value(prop, value) is not True:
                return
        except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
            return
        event.stop_propagation()
        # Animate only after the panel accepted and synchronized the value.
        self._flash(label, controls)

    def _on_hover(self, event):
        context = self._context(event)
        if context is None:
            return
        label, row, _controls, prop = context
        try:
            value = self._default_value(prop)
        except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
            return
        if value is None:
            return
        if isinstance(value, float):
            value = f"{value:g}"
        elif isinstance(value, bool):
            value = lf.ui.tr("common.yes" if value else "common.no")
        elif isinstance(value, int):
            value = f"{value:,}"
        hint = lf.ui.tr("common.reset_default_hint").replace("{value}", str(value))
        description = row.get_attribute("data-tooltip", "")
        # data-tooltip accepts localization keys; title accepts composed text.
        label.set_attribute("title", (lf.ui.tr(description) + "\n" if description else "") + hint)

    @staticmethod
    def _flash(label, controls):
        primary = lf.ui.theme().palette.primary
        r, g, b = (round(float(primary[i]) * 255) for i in range(3))
        # Animate paint properties only; no layout changes, blur or render passes.
        label.animate("color", label.get_property("color"), .24, "quadratic-out",
                      f"rgb({r},{g},{b})", True)
        for control in controls:
            control.animate("background-color", control.get_property("background-color"),
                            .24, "quadratic-out", f"rgba({r},{g},{b},0.22)", True)
