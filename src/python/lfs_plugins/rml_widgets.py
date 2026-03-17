# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""RmlUI widget builder helpers for constructing DOM subtrees
with correct CSS classes from components.rcss.

Usage in Panel.on_mount():

    from . import rml_widgets as w

    container = doc.get_element_by_id("settings")
    w.button(container, "start", "Start Training", style="success")
    w.slider(container, "lr", label="Learning Rate",
             min=0.0, max=1.0, step=0.01, value=0.3)
    w.checkbox(container, "enabled", label="Enable Feature", checked=True)
    w.select(container, "mode", label="Mode",
             options=[("auto", "Auto"), ("manual", "Manual")])
    w.collapsible(container, "advanced", title="Advanced Settings")
    w.progress(container, "prog", value=0.5, label="50%")
"""


def find_ancestor_with_attribute(element, attribute, stop=None):
    """Walk up the DOM tree looking for an element with the given attribute."""
    while element is not None and element != stop:
        if element.has_attribute(attribute):
            return element
        element = element.parent()
    return None


def _section_duration(height_px, duration):
    if duration is not None:
        return max(0.08, float(duration))
    height_px = max(0.0, float(height_px))
    return min(0.28, 0.16 + height_px / 2200.0)


def _apply_section_visual_state(expanded, header_element=None, arrow_element=None):
    if header_element:
        header_element.set_class("is-expanded", expanded)
        header_element.set_class("is-collapsed", not expanded)
    if arrow_element:
        arrow_element.set_text(chr(0x25B6))
        arrow_element.set_class("is-expanded", expanded)
        arrow_element.set_class("is-collapsed", not expanded)


def sync_section_state(content_element, expanded, header_element=None, arrow_element=None):
    """Apply the steady-state visual state for a collapsible section."""
    if not content_element:
        return

    _apply_section_visual_state(expanded, header_element, arrow_element)
    content_element.set_class("collapsed", not expanded)

    content_element.remove_property("max-height")
    content_element.remove_property("opacity")
    content_element.remove_property("pointer-events")


def animate_section_toggle(content_element, expanding, arrow_element=None,
                           duration=None, header_element=None):
    """Animate a section open/close with synchronized arrow and header state."""
    if not content_element:
        return

    _apply_section_visual_state(expanding, header_element, arrow_element)

    if expanding:
        content_element.set_class("collapsed", False)
        content_element.remove_property("pointer-events")

        current_h = max(content_element.client_height, 0)
        target_h = max(content_element.scroll_height, current_h)
        if target_h <= 0:
            sync_section_state(content_element, True, header_element, arrow_element)
            return
        duration = _section_duration(target_h, duration)
        fade_duration = max(0.1, min(0.18, duration * 0.7))
        content_element.animate("max-height", f"{target_h}px", duration, "cubic-out",
                                f"{current_h}px" if current_h > 0 else "0px",
                                remove_on_complete=True)
        content_element.animate("opacity", "1", fade_duration, "quadratic-out",
                                remove_on_complete=True)
        return

    content_element.set_property("pointer-events", "none")
    current_h = max(content_element.client_height, 0)
    target_h = max(content_element.scroll_height, current_h)
    if target_h <= 0:
        sync_section_state(content_element, False, header_element, arrow_element)
        return
    duration = _section_duration(target_h, duration)
    fade_duration = max(0.1, min(0.18, duration * 0.7))
    content_element.animate("max-height", "0px", duration, "cubic-in-out",
                            f"{target_h}px")
    content_element.animate("opacity", "0", fade_duration, "quadratic-out", "1")


def button(container, id, label, style="", disabled=False):
    """Create a styled button element.

    Args:
        style: One of "", "primary", "success", "warning", "error", "secondary".
    """
    btn = container.append_child("button")
    btn.set_id(id)
    classes = "btn"
    if style:
        classes += f" btn--{style}"
    btn.set_class_names(classes)
    btn.set_text(label)
    if disabled:
        btn.set_attribute("disabled", "disabled")
    return btn


def aligned_property_row(container, label="", control_classes="setting-row__control-col"):
    """Create a fixed label-left / control-right row.

    Returns:
        Tuple of (row_element, control_container).
    """
    row = container.append_child("div")
    row.set_class_names("setting-row setting-row--aligned")

    lbl = row.append_child("span")
    lbl.set_class_names("setting-row__label-col")
    if label:
        lbl.set_text(label)

    control = row.append_child("div")
    control.set_class_names(control_classes)
    return row, control


def aligned_checkbox_row(container, id, label="", checked=False, data_prop=""):
    """Create a fixed label-left / checkbox-right row."""
    row = container.append_child("div")
    row.set_class_names("setting-row setting-row--aligned")
    row.set_id(f"row-{id}")

    lbl = row.append_child("span")
    lbl.set_class_names("setting-row__label-col")
    lbl.set_id(f"text-{id}")
    if label:
        lbl.set_text(label)

    control = row.append_child("label")
    control.set_class_names("setting-row__control-col setting-row__control-col--checkbox")
    control.set_id(f"label-{id}")

    cb = control.append_child("input")
    cb.set_id(f"cb-{id}")
    cb.set_attribute("type", "checkbox")
    if data_prop:
        cb.set_attribute("data-prop", data_prop)
    if checked:
        cb.set_attribute("checked", "")
    return row, cb, control


def checkbox(container, id, label="", checked=False, data_prop=""):
    """Create a setting row with a labeled checkbox."""
    row, _cb, _control = aligned_checkbox_row(
        container,
        id,
        label=label,
        checked=checked,
        data_prop=data_prop,
    )
    return row


def slider(container, id, label="", min=0.0, max=1.0, step=0.01,
           value=None, data_prop=""):
    """Create a setting row with a range slider and value display."""
    row, control = aligned_property_row(
        container,
        label=label,
        control_classes="setting-row__control-col setting-row__control-col--slider",
    )

    inp = control.append_child("input")
    inp.set_id(f"slider-{id}")
    inp.set_attribute("type", "range")
    inp.set_class_names("setting-slider")
    inp.set_attribute("min", str(min))
    inp.set_attribute("max", str(max))
    inp.set_attribute("step", str(step))
    if data_prop:
        inp.set_attribute("data-prop", data_prop)
    if value is not None:
        inp.set_attribute("value", str(value))

    val_span = control.append_child("span")
    val_span.set_id(f"val-{id}")
    val_span.set_class_names("slider-value")
    if value is not None:
        val_span.set_text(f"{value:.3f}")

    return row


def select(container, id, label="", options=None, data_prop=""):
    """Create a setting row with a select dropdown.

    Args:
        options: List of (value, display_text) tuples.
    """
    row, control = aligned_property_row(
        container,
        label=label,
        control_classes="setting-row__control-col setting-row__control-col--fill",
    )

    sel = control.append_child("select")
    sel.set_id(f"sel-{id}")
    if data_prop:
        sel.set_attribute("data-prop", data_prop)

    if options:
        for val, text in options:
            opt = sel.append_child("option")
            opt.set_attribute("value", str(val))
            opt.set_text(text)

    return row


def collapsible(container, id, title="", open=True):
    """Create a collapsible section with header and content area.

    Returns (header_element, content_element) tuple.
    """
    header = container.append_child("div")
    header.set_class_names("section-header")
    header.set_id(f"hdr-{id}")
    header.set_attribute("data-section", id)

    arrow = header.append_child("span")
    arrow.set_class_names("section-arrow")
    arrow.set_id(f"arrow-{id}")
    arrow.set_text(chr(0x25B6))

    title_span = header.append_child("span")
    title_span.set_id(f"text-hdr-{id}")
    title_span.set_text(title)

    content = container.append_child("div")
    content.set_class_names("section-content")
    content.set_id(f"sec-{id}")
    sync_section_state(content, open, header, arrow)

    return header, content


def progress(container, id, value=0.0, label=""):
    """Create a progress bar with optional text overlay."""
    wrapper = container.append_child("div")
    wrapper.set_property("position", "relative")

    prog = wrapper.append_child("progress")
    prog.set_id(id)
    prog.set_attribute("value", str(value))
    prog.set_attribute("max", "1")

    if label:
        text = wrapper.append_child("span")
        text.set_id(f"{id}-text")
        text.set_class_names("progress__text")
        text.set_text(label)

    return wrapper


def color_swatch(container, id, r=0, g=0, b=0, data_prop=""):
    """Create a color swatch with RGB component displays."""
    row, control = aligned_property_row(
        container,
        control_classes="setting-row__control-col setting-row__control-col--color",
    )
    row.set_id(f"row-{id}")

    for ch, val in [("r", r), ("g", g), ("b", b)]:
        comp = control.append_child("span")
        comp.set_class_names("color-comp")
        comp.set_id(f"{ch}c-{id}")
        comp.set_text(f"{val:.0f}")

    swatch = control.append_child("div")
    swatch.set_class_names("color-swatch")
    swatch.set_id(f"swatch-{id}")
    swatch.set_property("background-color",
                        f"rgb({int(r)},{int(g)},{int(b)})")
    if data_prop:
        swatch.set_attribute("data-prop", data_prop)

    hex_input = control.append_child("input")
    hex_input.set_id(f"hex-{id}")
    hex_input.set_class_names("color-hex")
    hex_input.set_attribute("type", "text")
    if data_prop:
        hex_input.set_attribute("data-prop", data_prop)

    return row


def separator(container):
    """Create a horizontal separator line."""
    sep = container.append_child("div")
    sep.set_class_names("separator")
    return sep


def setting_row(container, label="", control_id=""):
    """Create an empty setting row with an optional label.

    Returns the row element. Caller adds controls to it.
    """
    row = container.append_child("div")
    row.set_class_names("setting-row")

    if label:
        lbl = row.append_child("span")
        lbl.set_class_names("prop-label")
        if control_id:
            lbl.set_id(f"label-{control_id}")
        lbl.set_text(label)

    return row


def number_input(container, id, label="", value="", data_prop="",
                 data_type="float", fmt="", min_val=None, max_val=None):
    """Create a setting row with a text input for numeric values.

    Args:
        data_type: "int" or "float" for validation.
        fmt: Python format string for display (e.g. "%.6f", "%d").
        min_val/max_val: Clamping bounds (None = unclamped).
    """
    row, control = aligned_property_row(
        container,
        label=label,
        control_classes="setting-row__control-col setting-row__control-col--fill",
    )

    inp = control.append_child("input")
    inp.set_id(f"num-{id}")
    inp.set_attribute("type", "text")
    inp.set_class_names("number-input")
    if data_prop:
        inp.set_attribute("data-prop", data_prop)
    if data_type:
        inp.set_attribute("data-type", data_type)
    if fmt:
        inp.set_attribute("data-fmt", fmt)
    if min_val is not None:
        inp.set_attribute("data-min", str(min_val))
    if max_val is not None:
        inp.set_attribute("data-max", str(max_val))
    if value != "":
        inp.set_attribute("value", str(value))

    return row


def icon_button(container, id, icon_src, selected=False,
                disabled=False, tooltip="", tooltip_key=""):
    """Create an icon button for toolbars.

    Args:
        icon_src: Path to icon image (relative to assets).
        tooltip_key: Locale key for tooltip (resolved via LocalizationManager).
    """
    btn = container.append_child("div")
    btn.set_id(id)
    btn.set_class_names("icon-btn")
    if selected:
        btn.set_class("selected", True)
    if disabled:
        btn.set_attribute("disabled", "disabled")

    img = btn.append_child("img")
    img.set_attribute("src", icon_src)

    if tooltip_key:
        btn.set_attribute("data-tooltip", tooltip_key)
    elif tooltip:
        btn.set_attribute("title", tooltip)

    return btn
