# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Escaped content and explicit actions for the shared project operation modal."""
from html import escape
from .project_inspector import LICENSES
from typing import Any, Callable


def form_content(kind: str, data: dict[str, Any], *, tr: Callable[[str], str],
                 confirm_label: str, busy: bool) -> tuple[str, list[dict[str, Any]]]:
    def text(value: Any) -> str:
        return escape(str(value if value is not None else ""), quote=True)

    def label(key: str) -> str:
        return text(tr(key))

    def form_row(key: str, control: str, name: str = "") -> str:
        caption = (f'<label class="modal-field-label" for="{text(name)}">{label(key)}</label>'
                   if name else f'<span class="modal-field-label">{label(key)}</span>')
        return f'<div class="modal-field">{caption}{control}</div>'

    def fact(key: str, value: Any, *, id: str = "") -> str:
        return form_row(key, f'<span id="{id}" class="modal-field-value" title="{text(value)}">{text(value)}</span>')

    def field(name: str, key: str) -> str:
        return form_row(key, f'<input id="{name}" name="{name}" type="text" value="{text(data.get(name, ""))}" />', name)

    def choice(name: str, key: str, options: list[tuple[str, str]]) -> str:
        selected = str(data.get(name, ""))
        items = ''.join(f'<option value="{text(value)}"{" selected" if value == selected else ""}>{text(caption)}</option>' for value, caption in options)
        return form_row(key, f'<select id="{name}" name="{name}">{items}</select>', name)

    def button(key: str, style: str = "secondary", disabled: bool = False) -> dict[str, Any]:
        return dict(label=tr(key), style=style, disabled=disabled)

    body = '<div class="project-form">'
    if kind not in {"license", "remove_content", "compact_content"}:
        body += fact("projects.property.path", data.get("path", ""))
    buttons = []
    if busy:
        body += f'<div class="modal-note">{label("projects.status.reading")}</div>'
    elif kind == "export_as":
        body += choice("format", "projects.dialog.format", [(value, value.upper()) for value in data.get("formats", [])])
        body += field("destination", "projects.dialog.choose_destination")
        buttons = [button("projects.dialog.choose_destination")]
    elif kind == "update_thumbnail":
        keys = {"viewport": "projects.dialog.current_viewport", "first_dataset": "projects.dialog.first_dataset_image", "first_embedded": "projects.dialog.first_embedded_image", "image_file": "projects.dialog.image_file"}
        sources = data.get("sources") or ["image_file"]
        body += choice("source", "projects.dialog.source", [(key, tr(keys[key])) for key in sources if key in keys])
        if data.get("gallery_cover_available"):
            body += (f'<label class="modal-check"><input name="use_gallery_cover" type="checkbox" value="yes"'
                     f'{" checked" if data.get("use_gallery_cover") else ""}{" disabled" if data.get("gallery_cover_blocked") else ""}/>'
                     f'<span>{label("projects.gallery.cover.use")}</span></label>')
            if data.get("gallery_cover_blocked"):
                body += f'<div class="modal-note">{label(data.get("gallery_cover_reason", "projects.gallery.eligibility.connect"))}</div>'
    elif kind == "license":
        selected = str(data.get("license_choice") or "CC-BY-4.0")
        options = ''.join(
            f'<option value="{text(identifier)}" title="{label("projects.license.meaning_" + key)}"{" selected" if selected == identifier else ""}>'
            f'<span class="modal-option-name">{label("projects.license." + key)}</span>'
            f'<span class="modal-option-meaning">{label("projects.license.meaning_" + key)}</span></option>'
            for identifier, key in LICENSES)
        body += form_row("projects.property.license", f'<select id="license_choice" name="license_choice">{options}</select>', "license_choice")
        meaning = next((key for identifier, key in LICENSES if identifier == selected), "custom")
        body += '<div class="modal-note">' + label("projects.license.meaning_" + meaning) + '</div>'
        if selected == "custom":
            body += field("license_name", "projects.license.name")
            body += form_row("projects.license.text", f'<textarea id="license_text" name="license_text" rows="1">{text(data.get("license_text", ""))}</textarea>', "license_text")
        if selected not in {"CC0-1.0", "LicenseRef-Proprietary"}:
            body += field("attribution", "projects.license.attribution")
    elif kind == "rename":
        body += field("name", "projects.property.display_name")
    elif kind == "repair":
        body += field("destination", "projects.dialog.choose_destination")
        buttons = [button("projects.dialog.choose_destination")]
    if data.get("message"):
        body += f'<div class="modal-note warning-text">{text(data["message"])}</div>'
    if not busy:
        buttons.insert(0, dict(label=confirm_label, style="warning" if kind in {"remove_content", "compact_content"} else "primary", disabled=bool(data.get("blocked"))))
    buttons.append(button("common.cancel"))
    return body + '</div>', buttons
