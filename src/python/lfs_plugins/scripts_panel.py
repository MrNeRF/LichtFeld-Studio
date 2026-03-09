# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Python scripts management panel."""

from pathlib import Path

import lichtfeld as lf
from .types import RmlPanel


class ScriptsPanel(RmlPanel):
    """Floating window for managing loaded Python scripts."""

    idname = "lfs.scripts"
    label = "Python Scripts"
    space = "FLOATING"
    order = 200
    options = {"DEFAULT_CLOSED"}
    rml_template = "rmlui/scripts_panel.rml"
    rml_height_mode = "content"
    initial_width = 520
    update_interval_ms = 200

    def __init__(self):
        self._handle = None
        self._has_scripts = False
        self._last_signature = None
        self._last_lang = ""

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("scripts_panel")
        if model is None:
            return

        tr = lf.ui.tr

        model.bind_func("panel_label", lambda: self.label)
        model.bind_func("actions_label", lambda: tr("scripts_panel.actions"))
        model.bind_func("reload_all_label", lambda: tr("scripts_panel.reload_all"))
        model.bind_func("clear_all_label", lambda: tr("scripts_panel.clear_all"))
        model.bind_func("no_scripts_loaded", lambda: tr("scripts_panel.no_scripts_loaded"))
        model.bind_func("load_hint", lambda: tr("scripts_panel.load_hint"))
        model.bind_func("loaded_scripts_label", lambda: tr("scripts_panel.loaded_scripts"))
        model.bind_func("reload_label", lambda: tr("scripts_panel.reload"))
        model.bind_func("show_empty_state", lambda: not self._has_scripts)
        model.bind_func("show_script_list", lambda: self._has_scripts)
        model.bind_func("has_scripts", lambda: self._has_scripts)

        model.bind_event("reload_all", self._on_reload_all)
        model.bind_event("clear_all", self._on_clear_all)
        model.bind_record_list("scripts")

        self._handle = model.get_handle()

    def on_load(self, doc):
        super().on_load(doc)
        self._last_lang = lf.ui.get_current_language()

        scripts_list = doc.get_element_by_id("scripts-list")
        if scripts_list:
            scripts_list.add_event_listener("click", self._on_scripts_click)
            scripts_list.add_event_listener("change", self._on_scripts_change)

        self._refresh_scripts(force=True)

    def on_update(self, doc):
        del doc
        dirty = False

        current_lang = lf.ui.get_current_language()
        if current_lang != self._last_lang:
            self._last_lang = current_lang
            self._refresh_scripts(force=True)
            self._dirty_model()
            dirty = True

        dirty |= self._refresh_scripts(force=False)
        return dirty

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for field in fields:
            self._handle.dirty(field)

    def _get_scripts(self):
        return list(lf.scripts.get_scripts())

    def _script_signature(self, scripts):
        return tuple(
            (
                script["path"],
                bool(script["enabled"]),
                bool(script["has_error"]),
                script["error_message"],
            )
            for script in scripts
        )

    def _build_script_records(self, scripts):
        tr = lf.ui.tr
        records = []
        for index, script in enumerate(scripts):
            path = script["path"]
            filename = Path(path).name or path
            has_error = bool(script["has_error"])
            records.append({
                "index": str(index),
                "filename": filename,
                "path_text": f"{tr('scripts_panel.path')}: {path}",
                "enabled": bool(script["enabled"]),
                "has_error": has_error,
                "error_text": (
                    f"{tr('scripts_panel.error')}: {script['error_message']}"
                    if has_error else ""
                ),
            })
        return records

    def _refresh_scripts(self, force=False):
        if not self._handle:
            return False

        scripts = self._get_scripts()
        signature = self._script_signature(scripts)
        if not force and signature == self._last_signature:
            return False

        had_scripts = self._has_scripts
        self._last_signature = signature
        self._has_scripts = bool(scripts)
        self._handle.update_record_list("scripts", self._build_script_records(scripts))
        if had_scripts != self._has_scripts:
            self._dirty_model("show_empty_state", "show_script_list", "has_scripts")
        return True

    def _find_with_attr(self, element, attr, stop=None):
        while element is not None and element != stop:
            if element.has_attribute(attr):
                return element
            element = element.parent()
        return None

    def _on_scripts_click(self, event):
        container = event.current_target()
        target = self._find_with_attr(event.target(), "data-reload-index", container)
        if target is None:
            return

        try:
            index = int(target.get_attribute("data-reload-index", ""))
        except (TypeError, ValueError):
            return

        scripts = self._get_scripts()
        if not (0 <= index < len(scripts)):
            return

        event.stop_propagation()
        self._reload_script(index, scripts[index])

    def _on_scripts_change(self, event):
        container = event.current_target()
        target = self._find_with_attr(event.target(), "data-script-index", container)
        if target is None:
            return

        try:
            index = int(target.get_attribute("data-script-index", ""))
        except (TypeError, ValueError):
            return

        enabled = target.has_attribute("checked")
        lf.scripts.set_script_enabled(index, enabled)
        self._refresh_scripts(force=True)

    def _reload_script(self, index, script):
        lf.scripts.set_script_error(index, "")
        if script["enabled"]:
            result = lf.scripts.run([script["path"]])
            if not result["success"]:
                lf.scripts.set_script_error(index, result["error"])
        self._refresh_scripts(force=True)

    def _reload_all(self):
        lf.scripts.clear_errors()
        enabled_paths = lf.scripts.get_enabled_paths()
        if enabled_paths:
            result = lf.scripts.run(enabled_paths)
            if not result["success"]:
                for index, script in enumerate(self._get_scripts()):
                    if script["enabled"]:
                        lf.scripts.set_script_error(index, result["error"])
        self._refresh_scripts(force=True)

    def _on_reload_all(self, _handle, _event, _args):
        self._reload_all()

    def _on_clear_all(self, _handle, _event, _args):
        lf.scripts.clear()
        self._refresh_scripts(force=True)
