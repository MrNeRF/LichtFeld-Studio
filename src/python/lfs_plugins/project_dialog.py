# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Escaped content and explicit actions for the shared project operation modal."""
from html import escape
from .project_inspector import LICENSES
from typing import Any, Callable


def form_content(kind: str, data: dict[str, Any], *, tr: Callable[[str], str],
                 confirm_label: str, busy: bool, resumable: bool) -> tuple[str, list[dict[str, Any]]]:
    def text(value: Any) -> str:
        return escape(str(value if value is not None else ""), quote=True)

    def label(key: str) -> str:
        return text(tr(key))

    def fact(key: str, value: Any, *, id: str = "") -> str:
        return (f'<div class="modal-field"><span class="modal-field-label">{label(key)}</span>'
                f'<span id="{id}" class="modal-field-value" title="{text(value)}">{text(value)}</span></div>')

    def field(name: str, key: str) -> str:
        return (f'<label class="modal-field"><span class="modal-field-label">{label(key)}</span>'
                f'<input id="{name}" name="{name}" type="text" value="{text(data.get(name, ""))}" /></label>')

    def choice(name: str, key: str, options: list[tuple[str, str]]) -> str:
        selected = str(data.get(name, ""))
        items = ''.join(f'<option value="{text(value)}"{" selected" if value == selected else ""}>{text(caption)}</option>' for value, caption in options)
        return (f'<label class="modal-field"><span class="modal-field-label">{label(key)}</span>'
                f'<select id="{name}" name="{name}">{items}</select></label>')

    def button(key: str, style: str = "secondary", disabled: bool = False) -> dict[str, Any]:
        return dict(label=tr(key), style=style, disabled=disabled)

    body = '<div class="project-form">'
    if kind not in {"set_license", "remove_content", "compact_content"}:
        body += fact("projects.property.path", data.get("path", ""))
    buttons = []
    if busy:
        body += f'<div class="modal-note">{label("projects.status.reading")}</div>'
    elif kind == "save_history":
        headings = ["projects.dialog.save_kind", "projects.property.date", "projects.property.iteration", "projects.dialog.bytes_added"]
        body += '<div class="modal-history-head"><span class="modal-radio-space"></span>' + ''.join(f'<span class="modal-history-{i}">{label(key)}</span>' for i, key in enumerate(headings)) + '</div>'
        selected = int(data.get("generation") or 0)
        for row in data.get("rows", []):
            generation = int(row["generation"])
            kind_key = {"explicit": "common.save", "autosave": "projects.property.autosave", "recovered": "projects.dialog.recovered", "compaction": "projects.dialog.compaction", "compact": "projects.dialog.compaction"}.get(row["kind"], "common.save")
            values = [tr(kind_key), row["date"], row["iteration"], row["bytes_added"]]
            body += f'<label class="modal-history-row"><input id="save-{generation}" type="radio" name="generation" value="{generation}"{" checked" if generation == selected else ""} />'
            body += ''.join(f'<span class="modal-history-{i}" title="{text(value)}">{text(value)}</span>' for i, value in enumerate(values)) + '</label>'
        body += '<div class="modal-note">' + label("projects.dialog.recovery_note") + '</div>'
        if not resumable:
            body += '<div class="modal-note">' + label("projects.dialog.no_checkpoint") + '</div>'
        body += field("destination", "projects.dialog.choose_destination")
        buttons = [button("projects.action.open_as_new_project", "primary", not bool(data.get("rows"))),
                   button("projects.action.resume_from_here", "success", not resumable),
                   button("projects.dialog.choose_destination")]
    elif kind == "reduce_size":
        body += fact("projects.property.size", data.get("physical_size", ""))
        body += fact("projects.property.reclaimable", data.get("selected_reclaimable", ""))
        for row in data.get("checkpoints", []):
            value = f'{row["iteration"]} · {row["bytes"]}'
            if row["locked"]:
                value += ' · ' + tr("projects.dialog.checkpoint_kept")
            body += fact("projects.property.iteration", value)
        for name, key, allowed in (("drop_checkpoints", "projects.dialog.keep_latest_checkpoint", "drop_checkpoints_allowed"),
                                   ("drop_dataset", "projects.dialog.drop_embedded_dataset", "drop_dataset_allowed")):
            body += (f'<label class="modal-check"><input id="{name}" name="{name}" type="checkbox" value="yes"'
                     f'{" checked" if data.get(name) else ""}{" disabled" if not data.get(allowed) else ""}/><span>{label(key)}</span></label>')
        if not data.get("drop_dataset_allowed"):
            body += '<div class="modal-note">' + label("projects.dialog.dataset_kept") + '</div>'
        body += fact("projects.dialog.projected_size", data.get("selected_projected_size", ""))
        body += '<div class="modal-note">' + label("projects.dialog.recovery_note") + '</div>'
    elif kind == "export_as":
        body += choice("format", "projects.dialog.format", [(value, value.upper()) for value in data.get("formats", [])])
        body += field("destination", "projects.dialog.choose_destination")
        buttons = [button("projects.dialog.choose_destination")]
    elif kind == "update_thumbnail":
        keys = {"viewport": "projects.dialog.current_viewport", "first_dataset": "projects.dialog.first_dataset_image", "first_embedded": "projects.dialog.first_embedded_image", "image_file": "projects.dialog.image_file"}
        body += choice("source", "projects.dialog.source", [(key, tr(value)) for key, value in keys.items()])
        if data.get("gallery_cover_available"):
            body += (f'<label class="modal-check"><input name="use_gallery_cover" type="checkbox" value="yes"'
                     f'{" checked" if data.get("use_gallery_cover") else ""}{" disabled" if data.get("gallery_cover_blocked") else ""}/>'
                     f'<span>{label("projects.gallery.cover.use")}</span></label>')
            if data.get("gallery_cover_blocked"):
                body += f'<div class="modal-note">{label(data.get("gallery_cover_reason", "projects.gallery.eligibility.connect"))}</div>'
    elif kind == "set_license":
        selected = str(data.get("license_choice") or "CC-BY-4.0")
        options = ''.join(
            f'<option value="{text(identifier)}"{" selected" if selected == identifier else ""}>'
            f'<span class="license-option-name">{label("projects.license." + key)}</span>'
            f'<span class="license-option-meaning">{label("projects.license.meaning_" + key)}</span></option>'
            for identifier, key in LICENSES)
        body += (f'<label class="modal-field"><span class="modal-field-label">{label("projects.property.license")}</span>'
                 f'<select id="license_choice" class="license-choice" name="license_choice">{options}</select></label>')
        meaning = next((key for identifier, key in LICENSES if identifier == selected), "custom")
        body += '<div class="modal-note">' + label("projects.license.meaning_" + meaning) + '</div>'
        if selected == "custom":
            body += field("license_name", "projects.license.name")
            body += (f'<label class="modal-field modal-field--stacked"><span class="modal-field-label">{label("projects.license.text")}</span>'
                     f'<textarea id="license_text" name="license_text" rows="3">{text(data.get("license_text", ""))}</textarea></label>')
        if selected not in {"CC0-1.0", "LicenseRef-Proprietary"}:
            body += field("attribution", "projects.license.attribution")
    elif kind == "rename":
        body += field("name", "projects.property.display_name")
    elif kind == "repair":
        body += field("destination", "projects.dialog.choose_destination")
        buttons = [button("projects.dialog.choose_destination")]
    if data.get("message"):
        body += f'<div class="modal-note warning-text">{text(data["message"])}</div>'
    if kind != "save_history" and not busy:
        buttons.insert(0, dict(label=confirm_label, style="warning" if kind in {"reduce_size", "remove_content", "compact_content"} else "primary", disabled=bool(data.get("blocked"))))
    buttons.append(button("common.cancel"))
    return body + '</div>', buttons
