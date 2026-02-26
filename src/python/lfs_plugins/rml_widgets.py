# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""RmlUI widget builder helpers for constructing DOM subtrees
with correct CSS classes from components.rcss.

Usage in RmlPanel.on_load():

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
    btn.set_inner_rml(label)
    if disabled:
        btn.set_attribute("disabled", "disabled")
    return btn


def checkbox(container, id, label="", checked=False, data_prop=""):
    """Create a setting row with a labeled checkbox."""
    row = container.append_child("div")
    row.set_class_names("setting-row")
    row.set_id(f"row-{id}")

    lbl = row.append_child("label")
    lbl.set_class_names("setting-label")
    lbl.set_id(f"label-{id}")

    cb = lbl.append_child("input")
    cb.set_id(f"cb-{id}")
    cb.set_attribute("type", "checkbox")
    if data_prop:
        cb.set_attribute("data-prop", data_prop)
    if checked:
        cb.set_attribute("checked", "")

    if label:
        span = lbl.append_child("span")
        span.set_id(f"text-{id}")
        span.set_inner_rml(label)

    return row


def slider(container, id, label="", min=0.0, max=1.0, step=0.01,
           value=None, data_prop=""):
    """Create a setting row with a range slider and value display."""
    row = container.append_child("div")
    row.set_class_names("setting-row")

    inp = row.append_child("input")
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

    val_span = row.append_child("span")
    val_span.set_id(f"val-{id}")
    val_span.set_class_names("slider-value")
    if value is not None:
        val_span.set_inner_rml(f"{value:.3f}")

    if label:
        prop_lbl = row.append_child("span")
        prop_lbl.set_id(f"label-{id}")
        prop_lbl.set_class_names("prop-label")
        prop_lbl.set_inner_rml(label)

    return row


def select(container, id, label="", options=None, data_prop=""):
    """Create a setting row with a select dropdown.

    Args:
        options: List of (value, display_text) tuples.
    """
    row = container.append_child("div")
    row.set_class_names("setting-row")

    sel = row.append_child("select")
    sel.set_id(f"sel-{id}")
    if data_prop:
        sel.set_attribute("data-prop", data_prop)

    if options:
        for val, text in options:
            opt_rml = f'<option value="{val}">{text}</option>'
            sel.set_inner_rml(sel.get_inner_rml() + opt_rml)

    if label:
        prop_lbl = row.append_child("span")
        prop_lbl.set_id(f"label-{id}")
        prop_lbl.set_class_names("prop-label")
        prop_lbl.set_inner_rml(label)

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
    arrow.set_inner_rml("&#x25BC;" if open else "&#x25B6;")

    title_span = header.append_child("span")
    title_span.set_id(f"text-hdr-{id}")
    title_span.set_inner_rml(title)

    content = container.append_child("div")
    content.set_class_names("section-content")
    content.set_id(f"sec-{id}")
    if not open:
        content.set_class("collapsed", True)

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
        text.set_inner_rml(label)

    return wrapper


def color_swatch(container, id, r=0, g=0, b=0, data_prop=""):
    """Create a color swatch with RGB component displays."""
    row = container.append_child("div")
    row.set_class_names("setting-row")
    row.set_id(f"row-{id}")

    for ch, val in [("r", r), ("g", g), ("b", b)]:
        comp = row.append_child("span")
        comp.set_class_names("color-comp")
        comp.set_id(f"{ch}c-{id}")
        comp.set_inner_rml(f"{val:.0f}")

    swatch = row.append_child("div")
    swatch.set_class_names("color-swatch")
    swatch.set_id(f"swatch-{id}")
    swatch.set_property("background-color",
                        f"rgb({int(r)},{int(g)},{int(b)})")
    if data_prop:
        swatch.set_attribute("data-prop", data_prop)

    hex_input = row.append_child("input")
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
        lbl.set_inner_rml(label)

    return row


def number_input(container, id, label="", value="", data_prop="",
                 data_type="float", fmt="", min_val=None, max_val=None):
    """Create a setting row with a text input for numeric values.

    Args:
        data_type: "int" or "float" for validation.
        fmt: Python format string for display (e.g. "%.6f", "%d").
        min_val/max_val: Clamping bounds (None = unclamped).
    """
    row = container.append_child("div")
    row.set_class_names("setting-row")

    inp = row.append_child("input")
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

    if label:
        prop_lbl = row.append_child("span")
        prop_lbl.set_id(f"label-{id}")
        prop_lbl.set_class_names("prop-label")
        prop_lbl.set_inner_rml(label)

    return row


def icon_button(container, id, icon_src, selected=False,
                disabled=False, tooltip=""):
    """Create an icon button for toolbars.

    Args:
        icon_src: Path to icon image (relative to assets).
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

    if tooltip:
        btn.set_attribute("title", tooltip)

    return btn
