# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Application-level appearance and language preferences."""

import lichtfeld as lf

from .keymap_bindings import KeymapBindingsSection
from .types import Panel

__lfs_panel_classes__ = ["PreferencesPanel"]
__lfs_panel_ids__ = ["lfs.preferences"]


class PreferencesPanel(Panel):
    """Floating home for application-level preferences."""

    id = "lfs.preferences"
    label = "Preferences"
    space = lf.ui.PanelSpace.FLOATING
    order = 100
    template = "rmlui/preferences.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    size = (780, 440)
    options = {lf.ui.PanelOption.DEFAULT_CLOSED}
    update_policy = "interval"
    update_interval_ms = 50

    SCALE_OPTIONS = (
        (0.0, "menu.view.ui_scale.auto"),
        (1.0, "100%"),
        (1.25, "125%"),
        (1.5, "150%"),
        (1.75, "175%"),
        (2.0, "200%"),
    )

    NAVIGATION_OPTIONS = (
        ("orbit", "preferences.navigation_orbit"),
        ("trackball", "preferences.navigation_trackball"),
        ("fpv", "preferences.navigation_fpv"),
        ("drone", "preferences.navigation_drone"),
    )

    EXPANDABLE_SECTIONS = (
        "language",
        "working_directory",
        "appearance",
        "scene_rendering",
        "navigation",
        "view_snap",
        "key_bindings",
        "interface",
        "file_associations",
        "mcp",
    )

    def __init__(self):
        self._handle = None
        self._scene_upscaler_catalog = []
        self._scene_upscaler_presets = {}
        self._keymap = KeymapBindingsSection()
        self._theme_catalog = []
        self._language_catalog = []
        self._last_state = None
        self._section = "general"
        self._expanded_sections = set(self.EXPANDABLE_SECTIONS)
        self._mcp_enabled = True
        self._mcp_expose_network = False
        self._mcp_port = "45677"
        self._mcp_applied_port = 45677
        self._mcp_request_logging = False
        self._mcp_safe_mode = False
        self._last_mcp_runtime_config = None
        self._working_directory = ""
        self._applied_working_directory = ""
        self._document = None
        self._file_associations = []

    def on_bind_model(self, ctx):
        self._read_mcp_preferences()
        self._read_working_directory()
        model = ctx.create_data_model("preferences")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("preferences.title"))
        model.bind_func("show_general", lambda: self._section == "general")
        model.bind_func("show_appearance", lambda: self._section == "appearance")
        model.bind_func("show_input", lambda: self._section == "input")
        model.bind_func("show_interface", lambda: self._section == "interface")
        model.bind_func("show_file_associations", self._show_file_associations)
        model.bind_func("has_file_associations", self._has_file_associations)
        model.bind_func("show_mcp", lambda: self._section == "mcp")
        model.bind_func("show_section_reset", lambda: True)
        model.bind_func("reset_section_label", self._reset_section_label)
        for section in self.EXPANDABLE_SECTIONS:
            model.bind_func(
                f"{section}_expanded",
                lambda section=section: section in self._expanded_sections,
            )
        model.bind("theme_idx", self._theme_index, self._set_theme_index)
        model.bind("scale_idx", self._scale_index, self._set_scale_index)
        model.bind(
            "scene_upscaler_idx",
            self._scene_upscaler_index,
            self._set_scene_upscaler_index,
        )
        model.bind(
            "scene_upscaler_preset_idx",
            self._scene_upscaler_preset_index,
            self._set_scene_upscaler_preset_index,
        )
        model.bind_func(
            "scene_upscaler_has_preset",
            lambda: len(self._scene_upscaler_presets.get(self._scene_upscaler(), ())) > 1,
        )
        model.bind("language_idx", self._language_index, self._set_language_index)
        model.bind("navigation_idx", self._navigation_index, self._set_navigation_index)
        model.bind("view_snap", lf.get_camera_view_snap_enabled, self._set_view_snap)
        model.bind("remember_navigation", lf.ui.remember_camera_navigation, self._set_remember_navigation)
        model.bind("remember_view_snap", lf.ui.remember_camera_view_snap, self._set_remember_view_snap)
        model.bind("mcp_enabled", lambda: self._mcp_enabled, self._set_mcp_enabled)
        model.bind("mcp_expose_network", lambda: self._mcp_expose_network, self._set_mcp_expose_network)
        model.bind("mcp_port", lambda: self._mcp_port, self._set_mcp_port)
        model.bind("working_directory", lambda: self._working_directory, self._set_working_directory_draft)
        model.bind_func("working_directory_hint", self._working_directory_hint)
        model.bind("mcp_request_logging", lambda: self._mcp_request_logging, self._set_mcp_request_logging)
        model.bind_func("mcp_safe_mode", lambda: self._mcp_safe_mode)
        model.bind_func("mcp_status", self._mcp_status_text)
        model.bind("mcp_endpoint_value", self._mcp_endpoint_text, lambda _value: None)
        model.bind_func("mcp_error", self._mcp_error_text)
        model.bind_func("mcp_has_error", lambda: bool(self._mcp_error_text()))
        model.bind_func("mcp_log_file", self._mcp_log_file_text)
        model.bind_func("mcp_has_log_file", lambda: bool(self._mcp_log_file_text()))
        model.bind_event("accept_and_close", self._on_accept_and_close)
        model.bind_event("reset_current_section", self._on_reset_current_section)
        model.bind_event("reset_all_settings", self._on_reset_all_settings)
        model.bind_event("show_general", lambda *_: self._set_section("general"))
        model.bind_event("show_appearance", lambda *_: self._set_section("appearance"))
        model.bind_event("show_input", lambda *_: self._set_section("input"))
        model.bind_event("show_interface", lambda *_: self._set_section("interface"))
        model.bind_event("show_file_associations", lambda *_: self._set_section("file_associations"))
        model.bind_event("show_mcp", lambda *_: self._set_section("mcp"))
        model.bind_event("set_file_association", self._on_set_file_association)
        model.bind_event("toggle_mcp_enabled", self._on_toggle_mcp_enabled)
        model.bind_event("mcp_port_change", self._on_mcp_port_change)
        model.bind_event("confirm_mcp_port", self._on_confirm_mcp_port)
        model.bind_event("working_directory_change", self._on_working_directory_change)
        model.bind_event("confirm_working_directory", self._on_confirm_working_directory)
        model.bind_event("browse_working_directory", self._on_browse_working_directory)
        model.bind_event("use_default_working_directory", self._on_use_default_working_directory)
        model.bind_event("open_mcp_log_folder", self._on_open_mcp_log_folder)
        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_record_list("themes")
        model.bind_record_list("scales")
        model.bind_record_list("scene_upscalers")
        model.bind_record_list("scene_upscaler_presets")
        model.bind_record_list("languages")
        model.bind_record_list("navigation_modes")
        model.bind_record_list("file_associations")
        self._handle = model.get_handle()
        self._keymap.bind(model)
        self._reload_file_associations()

    def on_mount(self, doc):
        # The title bar is a cancellation boundary for the drafted MCP port,
        # so it needs the specialized close handler instead of Panel's generic
        # visibility-only listener.
        close_btn = doc.get_element_by_id("close-btn") if doc else None
        if close_btn:
            close_btn.add_event_listener(
                "click", lambda _ev: self._on_close(None, None, None)
            )
        self._document = doc
        self._expanded_sections = set(self.EXPANDABLE_SECTIONS)
        self._dirty_expanded_sections()
        self._rebuild_records()
        self._load_mcp_preferences()
        self._consume_section_request()
        self._last_state = self._state()
        self._refresh_selection()
        self._keymap.on_mount(doc)

    def on_unmount(self, doc):
        self._keymap.on_unmount()
        self._document = None
        self._handle = None
        doc.remove_data_model("preferences")

    def on_update(self, doc):
        self._consume_section_request()
        self._sync_mcp_runtime()
        state = self._state()
        if state != self._last_state:
            self._last_state = state
            self._sync_scene_upscaler_preset_records()
            self._dirty_selection()
            self._dirty_mcp()
        self._keymap.on_update(doc)

    def _state(self):
        return (
            lf.ui.get_theme(),
            float(lf.ui.get_ui_scale_preference()),
            self._scene_upscaler(),
            self._scene_upscaler_preset(),
            lf.ui.get_current_language(),
            lf.get_camera_navigation_mode(),
            lf.get_camera_view_snap_enabled(),
            lf.ui.remember_camera_navigation(),
            lf.ui.remember_camera_view_snap(),
            self._mcp_status_signature(),
        )

    def _rebuild_records(self):
        self._sync_scene_reconstruction_catalog()
        self._theme_catalog = sorted(
            lf.ui.themes(),
            key=lambda theme: (theme.get("order", 0), theme.get("name", theme.get("id", ""))),
        )
        self._language_catalog = list(lf.ui.get_languages())
        if not self._handle:
            return
        self._handle.update_record_list(
            "themes",
            [
                {
                    "index": str(index),
                    "label": lf.ui.tr(theme.get("label_key") or theme.get("name") or theme["id"]),
                }
                for index, theme in enumerate(self._theme_catalog)
            ],
        )
        self._handle.update_record_list(
            "scales",
            [
                {
                    "index": str(index),
                    "label": lf.ui.tr(label) if scale == 0.0 else label,
                }
                for index, (scale, label) in enumerate(self.SCALE_OPTIONS)
            ],
        )
        self._handle.update_record_list(
            "scene_upscalers",
            [
                {"index": str(index), "label": lf.ui.tr(label_key)}
                for index, (_backend, label_key) in enumerate(self._scene_upscaler_catalog)
            ],
        )
        self._sync_scene_upscaler_preset_records()
        self._handle.update_record_list(
            "languages",
            [
                {"index": str(index), "label": name}
                for index, (_code, name) in enumerate(self._language_catalog)
            ],
        )
        self._handle.update_record_list(
            "navigation_modes",
            [
                {"index": str(index), "label": lf.ui.tr(label)}
                for index, (_mode, label) in enumerate(self.NAVIGATION_OPTIONS)
            ],
        )
        self._reload_file_associations()

    def _theme_index(self):
        current = lf.ui.get_theme()
        for index, theme in enumerate(self._theme_catalog):
            if theme["id"] == current:
                return str(index)
        return "0"

    def _set_theme_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self._theme_catalog):
            lf.ui.set_theme(self._theme_catalog[index]["id"])
            self._refresh_selection()

    def _scale_index(self):
        preference = float(lf.ui.get_ui_scale_preference())
        for index, (scale, _label) in enumerate(self.SCALE_OPTIONS):
            if abs(preference - scale) < 0.01:
                return str(index)
        return "0"

    def _set_scale_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self.SCALE_OPTIONS):
            lf.ui.set_ui_scale(self.SCALE_OPTIONS[index][0])
            self._refresh_selection()

    def _scene_upscaler(self):
        settings = lf.get_render_settings()
        backend = "native" if settings is None else str(settings.scene_upscaler)
        return (
            backend
            if any(item[0] == backend for item in self._scene_upscaler_catalog)
            else "native"
        )

    def _sync_scene_reconstruction_catalog(self):
        records = lf.ui.get_scene_reconstruction_options()
        backends = []
        presets = {}
        for record in records:
            backend_id = str(record["id"])
            label_key = str(record["label_key"])
            backend_presets = tuple(
                (str(preset["id"]), str(preset["label_key"]))
                for preset in record.get("presets", ())
            )
            if backend_id and label_key and backend_presets:
                backends.append((backend_id, label_key))
                presets[backend_id] = backend_presets
        self._scene_upscaler_catalog = backends
        self._scene_upscaler_presets = presets

    def _sync_scene_upscaler_preset_records(self):
        if not self._handle:
            return
        presets = self._scene_upscaler_presets.get(self._scene_upscaler(), ())
        selected_preset = self._scene_upscaler_preset()
        self._handle.update_record_list(
            "scene_upscaler_presets",
            [
                {
                    "index": str(index),
                    "label": lf.ui.tr(label_key),
                    "selected": preset == selected_preset,
                }
                for index, (preset, label_key) in enumerate(presets)
            ],
        )

    def _scene_upscaler_index(self):
        current = self._scene_upscaler()
        return next(
            (str(index) for index, (backend, _label) in enumerate(self._scene_upscaler_catalog)
             if backend == current),
            "0",
        )

    def _set_scene_upscaler_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if not 0 <= index < len(self._scene_upscaler_catalog):
            return
        settings = lf.get_render_settings()
        if settings is None:
            return
        backend = self._scene_upscaler_catalog[index][0]
        presets = self._scene_upscaler_presets.get(backend, ())
        default_preset = presets[0][0] if presets else "native"
        try:
            remembered_preset = str(lf.ui.get_scene_reconstruction_preset_preference(backend))
        except AttributeError:
            remembered_preset = default_preset
        preset = (
            remembered_preset
            if any(item[0] == remembered_preset for item in presets)
            else default_preset
        )
        self._apply_scene_reconstruction(backend, preset)
        self._sync_scene_upscaler_preset_records()
        self._refresh_selection()

    @staticmethod
    def _apply_scene_reconstruction(backend, preset):
        setter = getattr(lf.ui, "set_scene_reconstruction", None)
        if setter is not None:
            return setter(backend, preset)
        settings = lf.get_render_settings()
        if settings is None:
            return False
        settings.scene_upscaler = backend
        settings.scene_upscaler_preset = preset
        return True

    def _scene_upscaler_preset(self):
        backend = self._scene_upscaler()
        settings = lf.get_render_settings()
        presets = self._scene_upscaler_presets.get(backend, ())
        if not presets:
            return "native"
        preset = presets[0][0] if settings is None else str(settings.scene_upscaler_preset)
        return preset if any(item[0] == preset for item in presets) else presets[0][0]

    def _scene_upscaler_preset_index(self):
        current = self._scene_upscaler_preset()
        presets = self._scene_upscaler_presets.get(self._scene_upscaler(), ())
        return next(
            (str(index) for index, (preset, _label) in enumerate(presets)
             if preset == current),
            "0",
        )

    def _set_scene_upscaler_preset_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        backend = self._scene_upscaler()
        presets = self._scene_upscaler_presets.get(backend, ())
        if not 0 <= index < len(presets):
            return
        if len(presets) <= 1:
            return
        self._apply_scene_reconstruction(backend, presets[index][0])
        self._refresh_selection()

    def _language_index(self):
        current = lf.ui.get_current_language()
        for index, (code, _name) in enumerate(self._language_catalog):
            if code == current:
                return str(index)
        return "0"

    def _set_language_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self._language_catalog):
            language = self._language_catalog[index][0]
            if language == lf.ui.get_current_language():
                return
            lf.ui.set_language(language)
            self._refresh_selection()

    def _navigation_index(self):
        current = lf.get_camera_navigation_mode()
        for index, (mode, _label) in enumerate(self.NAVIGATION_OPTIONS):
            if mode == current:
                return str(index)
        return "0"

    def _set_navigation_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self.NAVIGATION_OPTIONS):
            lf.set_camera_navigation_mode(self.NAVIGATION_OPTIONS[index][0])
            self._refresh_selection()

    def _set_view_snap(self, enabled):
        lf.set_camera_view_snap_enabled(bool(enabled))
        self._refresh_selection()

    def _set_remember_navigation(self, enabled):
        lf.ui.set_remember_camera_navigation(bool(enabled))
        if enabled:
            lf.set_camera_navigation_mode(lf.get_camera_navigation_mode())
        self._refresh_selection()

    def _set_remember_view_snap(self, enabled):
        lf.ui.set_remember_camera_view_snap(bool(enabled))
        if enabled:
            lf.set_camera_view_snap_enabled(lf.get_camera_view_snap_enabled())
        self._refresh_selection()

    def _read_working_directory(self):
        stored = lf.ui.get_working_directory_preference()
        self._applied_working_directory = stored or lf.ui.get_default_working_directory()
        self._working_directory = self._applied_working_directory
        self._dirty_working_directory()

    def _set_working_directory_draft(self, value):
        self._working_directory = str(value).strip()
        self._dirty_working_directory()

    def _on_working_directory_change(self, _handle, event, args):
        if args:
            self._set_working_directory_draft(args[0])
        if event.get_bool_parameter("linebreak", False):
            self._commit_working_directory()

    def _on_confirm_working_directory(self, _handle, _event, _args):
        self._commit_working_directory()

    def _on_browse_working_directory(self, _handle, _event, _args):
        start = self._working_directory or lf.ui.get_default_working_directory()
        chosen = lf.ui.open_folder_dialog(
            lf.ui.tr("preferences.working_directory"), start)
        if not chosen:
            return
        self._working_directory = chosen
        self._commit_working_directory()

    def _on_use_default_working_directory(self, _handle, _event, _args):
        lf.ui.clear_working_directory()
        self._read_working_directory()

    def _working_directory_hint(self):
        path = lf.ui.get_temp_project_directory()
        template = lf.ui.tr("preferences.working_directory_hint") or "Temporary projects: {path}"
        return template.replace("{path}", path)

    def _commit_working_directory(self):
        draft = (self._working_directory or "").strip()
        default_path = lf.ui.get_default_working_directory()
        if not draft or draft == default_path:
            lf.ui.clear_working_directory()
            self._read_working_directory()
            return True
        error = lf.ui.set_working_directory(draft)
        if error:
            lf.ui.message_dialog(
                lf.ui.tr("preferences.working_directory"),
                error or lf.ui.tr("preferences.working_directory_invalid"),
                "error",
            )
            self._working_directory = self._applied_working_directory
            self._dirty_working_directory()
            return False
        self._read_working_directory()
        return True

    def _dirty_working_directory(self):
        if not self._handle:
            return
        self._handle.dirty("working_directory")
        self._handle.dirty("working_directory_hint")

    def _read_mcp_preferences(self):
        preferences = lf.ui.get_mcp_preferences()
        self._mcp_enabled = bool(preferences.get("enabled", True))
        self._mcp_expose_network = bool(preferences.get("expose_network", False))
        self._mcp_port = str(preferences.get("port", 45677))
        self._mcp_applied_port = int(self._mcp_port)
        self._mcp_request_logging = bool(preferences.get("request_logging", False))
        self._mcp_safe_mode = bool(preferences.get("safe_mode", False))
        self._last_mcp_runtime_config = self._mcp_runtime_config_signature()

    def _load_mcp_preferences(self):
        self._read_mcp_preferences()
        self._dirty_mcp()

    def _set_mcp_enabled(self, enabled):
        if self._mcp_safe_mode:
            return
        self._mcp_enabled = bool(enabled)
        self._apply_mcp_preferences()

    def _on_toggle_mcp_enabled(self, _handle, _event, _args):
        self._set_mcp_enabled(not self._mcp_enabled)

    def _set_mcp_expose_network(self, enabled):
        if self._mcp_safe_mode:
            return
        self._mcp_expose_network = bool(enabled)
        self._apply_mcp_preferences()

    def _set_mcp_port(self, value):
        if self._mcp_safe_mode:
            return
        self._mcp_port = str(value).strip()
        self._dirty_mcp()

    def _on_mcp_port_change(self, _handle, event, args):
        if args:
            self._set_mcp_port(args[0])
        if event.get_bool_parameter("linebreak", False):
            self._commit_mcp_port()

    def _on_confirm_mcp_port(self, _handle, _event, _args):
        self._commit_mcp_port()

    def _commit_mcp_port(self):
        port = self._validated_mcp_port()
        if port is None:
            self._dirty_mcp()
            return False
        if port == self._mcp_applied_port:
            status = lf.ui.get_mcp_status()
            if status.get("enabled") and status.get("phase") == "failed":
                return self._apply_mcp_preferences(port)
            return True
        return self._apply_mcp_preferences(port)

    def _set_mcp_request_logging(self, enabled):
        if self._mcp_safe_mode:
            return
        self._mcp_request_logging = bool(enabled)
        self._apply_mcp_preferences()

    def _apply_mcp_preferences(self, port=None):
        if self._mcp_safe_mode:
            return False
        port = self._mcp_applied_port if port is None else port
        accepted = lf.ui.set_mcp_preferences(
            self._mcp_enabled,
            self._mcp_expose_network,
            port,
            self._mcp_request_logging,
        )
        if not accepted:
            self._dirty_mcp()
            return False
        self._mcp_applied_port = port
        self._last_mcp_runtime_config = self._mcp_runtime_config_signature()
        self._dirty_mcp()
        return True

    def _validated_mcp_port(self):
        try:
            port = int(self._mcp_port)
        except (TypeError, ValueError):
            return None
        return port if 1 <= port <= 65535 else None

    def _mcp_status_signature(self):
        status = lf.ui.get_mcp_status()
        return (
            bool(status.get("enabled")),
            bool(status.get("running")),
            str(status.get("phase", "")),
            bool(status.get("expose_network")),
            int(status.get("port", 0)),
            int(status.get("request_count", 0)),
            int(status.get("success_count", 0)),
            int(status.get("error_count", 0)),
            bool(status.get("request_logging")),
            str(status.get("log_file", "")),
            str(status.get("error", "")),
            str(status.get("error_kind", "")),
            str(status.get("error_address", "")),
            int(status.get("error_port", 0)),
            bool(status.get("safe_mode", False)),
            tuple(str(endpoint) for endpoint in status.get("endpoints") or ()),
        )

    def _mcp_runtime_config_signature(self):
        status = lf.ui.get_mcp_status()
        return (
            bool(status.get("enabled")),
            bool(status.get("expose_network")),
            int(status.get("port", 45677)),
            bool(status.get("request_logging")),
            bool(status.get("safe_mode", False)),
        )

    def _sync_mcp_runtime(self):
        signature = self._mcp_runtime_config_signature()
        if signature == self._last_mcp_runtime_config:
            return
        draft_port = self._mcp_port
        has_unconfirmed_port = draft_port != str(self._mcp_applied_port)
        self._load_mcp_preferences()
        if has_unconfirmed_port:
            self._mcp_port = draft_port
            self._dirty_mcp()

    def _mcp_status_text(self):
        if self._validated_mcp_port() is None:
            return lf.ui.tr("preferences.mcp_status_error")
        status = lf.ui.get_mcp_status()
        phase = status.get("phase")
        if phase == "starting":
            return lf.ui.tr("preferences.mcp_status_starting")
        if phase == "stopping":
            return lf.ui.tr("preferences.mcp_status_stopping")
        if phase == "failed":
            return lf.ui.tr("preferences.mcp_status_error")
        if phase == "disabled" or not status.get("enabled"):
            return lf.ui.tr("preferences.mcp_status_off")
        if phase == "running" or status.get("running"):
            return lf.ui.tr("preferences.mcp_status_running")
        return lf.ui.tr("preferences.mcp_status_error")

    def _mcp_endpoint_text(self):
        status = lf.ui.get_mcp_status()
        if not status.get("running"):
            return lf.ui.tr("preferences.mcp_no_active_endpoint")
        endpoints = status.get("endpoints") or []
        if endpoints:
            return "\n".join(str(endpoint) for endpoint in endpoints)
        port = status.get("port", 45677)
        return f"http://127.0.0.1:{port}/mcp\nhttp://localhost:{port}/mcp"

    def _mcp_endpoint_rows(self):
        return min(10, max(2, len(self._mcp_endpoint_text().splitlines())))

    def _mcp_error_text(self):
        if self._validated_mcp_port() is None:
            return lf.ui.tr("preferences.mcp_invalid_port")
        status = lf.ui.get_mcp_status()
        error_kind = status.get("error_kind", "none")
        if error_kind == "invalid_port":
            return lf.ui.tr("preferences.mcp_invalid_port")
        if error_kind == "bind_failed":
            address = str(status.get("error_address", ""))
            port = int(status.get("error_port", 0))
            endpoint = f"{address}:{port}" if address and port else address
            return lf.ui.tr("preferences.mcp_bind_failed").format(
                endpoint=endpoint
            )
        return (
            lf.ui.tr("status_bar.mcp_error_detail")
            if error_kind not in ("", "none") or status.get("error")
            else ""
        )

    def _mcp_log_file_text(self):
        return str(lf.ui.get_mcp_status().get("log_file", ""))

    def _on_open_mcp_log_folder(self, _handle, _event, _args):
        lf.ui.open_url(lf.ui.get_mcp_log_directory())

    @staticmethod
    def _coerce_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _file_association_status_rows(self):
        getter = getattr(lf, "file_associations_status", None)
        if not callable(getter):
            return []
        rows = getter()
        if not rows:
            return []
        result = []
        for row in rows:
            extension = str(row.get("extension", "")).strip()
            if not extension:
                continue
            result.append(
                {
                    "extension": extension,
                    "registered": bool(row.get("registered", False)),
                    "label": extension,
                }
            )
        return result

    def _reload_file_associations(self):
        self._file_associations = self._file_association_status_rows()
        if self._handle:
            self._handle.update_record_list(
                "file_associations",
                [
                    {
                        "extension": row["extension"],
                        "registered": row["registered"],
                        "label": row["label"],
                    }
                    for row in self._file_associations
                ],
            )
            dirty = getattr(self._handle, "dirty", None)
            if callable(dirty):
                dirty("has_file_associations")
                dirty("show_file_associations")

    def _has_file_associations(self):
        return bool(self._file_associations)

    def _show_file_associations(self):
        return self._section == "file_associations" and self._has_file_associations()

    def _on_set_file_association(self, _handle, _event, args):
        if not args or len(args) < 2:
            return
        self._set_file_association(args[0], args[1])

    def _set_file_association(self, extension, enabled):
        setter = getattr(lf, "file_association_set", None)
        if not callable(setter):
            return False
        ok = bool(setter(str(extension), self._coerce_bool(enabled)))
        self._reload_file_associations()
        return ok

    def _consume_section_request(self):
        section = lf.ui.take_preferences_section_request()
        if section in ("general", "appearance", "input", "interface", "file_associations", "mcp"):
            if section == "file_associations" and not self._has_file_associations():
                return
            self._set_section(section)

    def _dirty_mcp(self):
        if not self._handle:
            return
        for name in (
            "mcp_enabled",
            "mcp_expose_network",
            "mcp_port",
            "mcp_request_logging",
            "mcp_safe_mode",
            "mcp_status",
            "mcp_endpoint_value",
            "mcp_error",
            "mcp_has_error",
            "mcp_log_file",
            "mcp_has_log_file",
        ):
            self._handle.dirty(name)
        if self._document:
            endpoint_list = self._document.get_element_by_id("mcp-endpoints")
            if endpoint_list:
                endpoint_list.set_attribute("rows", str(self._mcp_endpoint_rows()))

    def _on_close(self, _handle, _event, _args):
        # The floating-window title bar is cancellation: discard an unconfirmed
        # port draft while preserving settings that were already applied live.
        self._mcp_port = str(self._mcp_applied_port)
        self._working_directory = self._applied_working_directory
        self._dirty_mcp()
        self._dirty_working_directory()
        lf.ui.set_panel_enabled(self.id, False)

    def _on_accept_and_close(self, _handle, _event, _args):
        if not self._commit_mcp_port():
            return
        if not self._commit_working_directory():
            return
        lf.ui.set_panel_enabled(self.id, False)

    def _set_section(self, section):
        if self._section == section:
            return
        self._section = section
        if self._handle:
            for name in (
                "show_general",
                "show_appearance",
                "show_input",
                "show_interface",
                "show_file_associations",
                "has_file_associations",
                "show_mcp",
                "show_section_reset",
                "reset_section_label",
            ):
                self._handle.dirty(name)

    def _on_toggle_section(self, _handle, _event, args):
        if not args:
            return
        section = str(args[0])
        if section not in self.EXPANDABLE_SECTIONS:
            return
        if section in self._expanded_sections:
            self._expanded_sections.remove(section)
        else:
            self._expanded_sections.add(section)
        if self._handle:
            self._handle.dirty(f"{section}_expanded")

    def _dirty_expanded_sections(self):
        if not self._handle:
            return
        for section in self.EXPANDABLE_SECTIONS:
            self._handle.dirty(f"{section}_expanded")

    def _reset_section_label(self):
        return lf.ui.tr("preferences.reset_current_section")

    def _on_reset_current_section(self, _handle, _event, _args):
        reset_label = self._reset_section_label()
        section_name = lf.ui.tr(f"preferences.{self._section}")

        def _on_result(button):
            if button != reset_label:
                return
            error = self._reset_section()
            if error:
                lf.ui.message_dialog(
                    reset_label,
                    f"{lf.ui.tr('preferences.reset_section_failed')} {error}",
                    "error",
                )
                return
            lf.ui.message_dialog(
                reset_label,
                lf.ui.tr("preferences.reset_section_success"),
            )

        lf.ui.confirm_dialog(
            f"{reset_label}: {section_name}",
            f"{lf.ui.tr('preferences.reset_section_confirmation')} {section_name}.",
            [lf.ui.tr("common.cancel"), reset_label],
            _on_result,
        )

    def _on_reset_all_settings(self, _handle, _event, _args):
        reset_label = lf.ui.tr("preferences.reset_all_settings")

        def _on_result(button):
            if button != reset_label:
                return
            errors = [
                self._reset_section(section)
                for section in ("general", "appearance", "input", "interface", "mcp")
            ]
            errors.append(lf.ui.reset_window_state())
            error = next((item for item in errors if item), None)
            if error:
                lf.ui.message_dialog(
                    reset_label,
                    f"{lf.ui.tr('preferences.reset_section_failed')} {error}",
                    "error",
                )
                return
            self._refresh_selection()
            lf.ui.message_dialog(reset_label, lf.ui.tr("preferences.reset_all_settings_success"))

        lf.ui.confirm_dialog(
            reset_label,
            lf.ui.tr("preferences.reset_all_settings_confirmation"),
            [lf.ui.tr("common.cancel"), reset_label],
            _on_result,
        )

    def _reset_section(self, section=None):
        section = section or self._section
        if section == "general":
            lf.ui.set_language("en")
            lf.ui.clear_working_directory()
            self._read_working_directory()
        elif section == "appearance":
            lf.ui.set_theme("dark")
            lf.ui.set_ui_scale(0.0)
            lf.ui.set_scene_reconstruction("native", "native")
            lf.ui.reset_scene_reconstruction_preferences()
            self._sync_scene_upscaler_preset_records()
        elif section == "input":
            lf.ui.set_remember_camera_navigation(False)
            lf.ui.set_remember_camera_view_snap(False)
            lf.set_camera_navigation_mode("orbit")
            lf.set_camera_view_snap_enabled(False)
        elif section == "interface":
            return lf.ui.reset_layout()
        elif section == "mcp":
            if not self._mcp_safe_mode:
                lf.ui.set_mcp_preferences(True, False, 45677, False)
                self._load_mcp_preferences()
        self._refresh_selection()
        return None

    def _refresh_selection(self):
        self._last_state = self._state()
        self._dirty_selection()

    def _dirty_selection(self):
        if self._handle:
            self._handle.dirty("theme_idx")
            self._handle.dirty("scale_idx")
            self._handle.dirty("scene_upscaler_idx")
            self._handle.dirty("scene_upscaler_preset_idx")
            self._handle.dirty("scene_upscaler_has_preset")
            self._handle.dirty("language_idx")
            self._handle.dirty("navigation_idx")
            self._handle.dirty("view_snap")
            self._handle.dirty("remember_navigation")
            self._handle.dirty("remember_view_snap")
            self._dirty_mcp()
            self._dirty_working_directory()
