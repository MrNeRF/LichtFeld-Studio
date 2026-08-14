# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Application-level appearance and language preferences."""

import lichtfeld as lf

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

    SCENE_RENDER_SCALE_OPTIONS = (0.25, 0.33, 0.5, 0.67, 0.75, 1.0)
    DEFAULT_SCENE_UPSCALER_OPTIONS = (
        ("native", "preferences.scene_upscaler_native"),
        ("spatial", "preferences.scene_upscaler_spatial"),
        ("temporal", "preferences.scene_upscaler_temporal"),
    )
    TEMPORAL_QUALITY_OPTIONS = (
        ("performance", "preferences.temporal_quality_performance"),
        ("balanced", "preferences.temporal_quality_balanced"),
        ("quality", "preferences.temporal_quality_quality"),
    )

    NAVIGATION_OPTIONS = (
        ("orbit", "preferences.navigation_orbit"),
        ("trackball", "preferences.navigation_trackball"),
        ("fpv", "preferences.navigation_fpv"),
        ("drone", "preferences.navigation_drone"),
    )

    EXPANDABLE_SECTIONS = (
        "language",
        "appearance",
        "scene_rendering",
        "navigation",
        "view_snap",
        "interface",
        "mcp",
    )

    def __init__(self):
        self._handle = None
        self._scene_upscaler_catalog = list(self.DEFAULT_SCENE_UPSCALER_OPTIONS)
        self._scene_upscaler_recommended_scales = {}
        self._theme_catalog = []
        self._language_catalog = []
        self._scene_render_scale_catalog = []
        self._last_state = None
        self._section = "general"
        self._expanded_sections = set(self.EXPANDABLE_SECTIONS)
        self._mcp_enabled = True
        self._mcp_expose_network = False
        self._mcp_port = "45677"
        self._mcp_applied_port = 45677
        self._mcp_request_logging = False
        self._last_mcp_runtime_config = None
        self._document = None

    def on_bind_model(self, ctx):
        self._read_mcp_preferences()
        model = ctx.create_data_model("preferences")
        if model is None:
            return

        # A data model is recreated after the floating panel is remounted.  The
        # record list itself belongs to that model, so never reuse the previous
        # catalog's "already synchronized" marker across model lifetimes.
        self._scene_render_scale_catalog = []

        model.bind_func("panel_label", lambda: lf.ui.tr("preferences.title"))
        model.bind_func("show_general", lambda: self._section == "general")
        model.bind_func("show_appearance", lambda: self._section == "appearance")
        model.bind_func("show_input", lambda: self._section == "input")
        model.bind_func("show_interface", lambda: self._section == "interface")
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
        model.bind("scene_render_scale_idx", self._scene_render_scale_index, self._set_scene_render_scale_index)
        model.bind("viewer_splat_precision_idx", self._viewer_splat_precision_index, self._set_viewer_splat_precision_index)
        model.bind("scene_upscaler_idx", self._scene_upscaler_index, self._set_scene_upscaler_index)
        model.bind_func(
            "scene_upscaler_has_scale",
            lambda: self._scene_upscaler() in {"spatial", "temporal"},
        )
        model.bind_func(
            "scene_upscaler_has_quality",
            lambda: self._scene_upscaler() in {"temporal", "nvidia-dlss", "amd-fsr3"},
        )
        model.bind("temporal_quality_idx", self._temporal_quality_index, self._set_temporal_quality_index)
        model.bind("language_idx", self._language_index, self._set_language_index)
        model.bind("navigation_idx", self._navigation_index, self._set_navigation_index)
        model.bind("view_snap", lf.get_camera_view_snap_enabled, self._set_view_snap)
        model.bind("remember_navigation", lf.ui.remember_camera_navigation, self._set_remember_navigation)
        model.bind("remember_view_snap", lf.ui.remember_camera_view_snap, self._set_remember_view_snap)
        model.bind("mcp_enabled", lambda: self._mcp_enabled, self._set_mcp_enabled)
        model.bind("mcp_expose_network", lambda: self._mcp_expose_network, self._set_mcp_expose_network)
        model.bind("mcp_port", lambda: self._mcp_port, self._set_mcp_port)
        model.bind("mcp_request_logging", lambda: self._mcp_request_logging, self._set_mcp_request_logging)
        model.bind_func("mcp_status", self._mcp_status_text)
        model.bind("mcp_endpoint_value", self._mcp_endpoint_text, lambda _value: None)
        model.bind_func("mcp_error", self._mcp_error_text)
        model.bind_func("mcp_has_error", lambda: bool(self._mcp_error_text()))
        model.bind_func("mcp_log_file", self._mcp_log_file_text)
        model.bind_func("mcp_has_log_file", lambda: bool(self._mcp_log_file_text()))
        model.bind_event("close", self._on_close)
        model.bind_event("reset_current_section", self._on_reset_current_section)
        model.bind_event("reset_all_settings", self._on_reset_all_settings)
        model.bind_event("show_general", lambda *_: self._set_section("general"))
        model.bind_event("show_appearance", lambda *_: self._set_section("appearance"))
        model.bind_event("show_input", lambda *_: self._set_section("input"))
        model.bind_event("show_interface", lambda *_: self._set_section("interface"))
        model.bind_event("show_mcp", lambda *_: self._set_section("mcp"))
        model.bind_event("toggle_mcp_enabled", self._on_toggle_mcp_enabled)
        model.bind_event("mcp_port_change", self._on_mcp_port_change)
        model.bind_event("confirm_mcp_port", self._on_confirm_mcp_port)
        model.bind_event("open_mcp_log_folder", self._on_open_mcp_log_folder)
        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_record_list("themes")
        model.bind_record_list("scales")
        model.bind_record_list("scene_render_scales")
        model.bind_record_list("scene_upscalers")
        model.bind_record_list("temporal_qualities")
        model.bind_record_list("languages")
        model.bind_record_list("navigation_modes")
        self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._document = doc
        self._expanded_sections = set(self.EXPANDABLE_SECTIONS)
        self._dirty_expanded_sections()
        self._rebuild_records()
        self._load_mcp_preferences()
        self._consume_section_request()
        self._last_state = self._state()
        self._refresh_selection()

    def on_unmount(self, doc):
        self._document = None
        self._handle = None
        self._scene_render_scale_catalog = []
        doc.remove_data_model("preferences")

    def on_update(self, doc):
        self._consume_section_request()
        self._sync_mcp_runtime()
        self._sync_scene_upscaler_catalog()
        self._sync_scene_render_scale_records()
        state = self._state()
        if state == self._last_state:
            return
        self._last_state = state
        self._dirty_selection()

    def _state(self):
        self._sync_scene_upscaler_catalog()
        return (
            lf.ui.get_theme(),
            float(lf.ui.get_ui_scale_preference()),
            self._viewer_splat_precision_index(),
            self._scene_render_scale(),
            self._scene_upscaler(),
            tuple(value for value, _label_key in self._scene_upscaler_catalog),
            self._temporal_quality(),
            lf.ui.get_current_language(),
            lf.get_camera_navigation_mode(),
            lf.get_camera_view_snap_enabled(),
            lf.ui.remember_camera_navigation(),
            lf.ui.remember_camera_view_snap(),
            self._mcp_status_signature(),
        )

    def _rebuild_records(self):
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
        self._sync_scene_render_scale_records()
        self._sync_scene_upscaler_records()
        self._handle.update_record_list(
            "temporal_qualities",
            [
                {"index": str(index), "label": lf.ui.tr(label_key)}
                for index, (_value, label_key) in enumerate(self.TEMPORAL_QUALITY_OPTIONS)
            ],
        )
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

    def _scene_render_scale(self):
        backend = self._scene_upscaler()
        if backend == "native":
            return 1.0
        try:
            return max(0.25, min(1.0, float(lf.ui.get_scene_upscaler_scale(backend))))
        except AttributeError:
            settings = lf.get_render_settings()
            return 1.0 if settings is None else max(0.25, min(1.0, float(settings.scene_upscaler_scale)))

    def _viewer_splat_precision_index(self):
        try:
            return 1 if int(lf.ui.get_viewer_splat_precision()) == 32 else 0
        except (AttributeError, TypeError, ValueError):
            return 0

    def _set_viewer_splat_precision_index(self, value):
        try:
            lf.ui.set_viewer_splat_precision(32 if int(value) == 1 else 16)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            self._refresh_selection()
            return
        self._refresh_selection()

    def _scene_render_scale_options(self):
        current = self._scene_render_scale()
        options = list(self.SCENE_RENDER_SCALE_OPTIONS)
        if not any(abs(current - option) < 0.001 for option in options):
            options.append(current)
            options.sort()
        return options

    def _sync_scene_render_scale_records(self):
        options = self._scene_render_scale_options()
        if options == self._scene_render_scale_catalog:
            return
        self._scene_render_scale_catalog = options
        if self._handle:
            self._handle.update_record_list(
                "scene_render_scales",
                [
                    {"index": str(index), "label": f"{round(scale * 100):d}%"}
                    for index, scale in enumerate(options)
                ],
            )
            self._handle.dirty("scene_render_scale_idx")

    def _scene_render_scale_index(self):
        current = self._scene_render_scale()
        for index, scale in enumerate(self._scene_render_scale_catalog):
            if abs(current - scale) < 0.001:
                return str(index)
        return "0"

    def _set_scene_render_scale_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self._scene_render_scale_catalog):
            settings = lf.get_render_settings()
            if settings is not None:
                backend = self._scene_upscaler()
                if backend == "native":
                    return
                scale = self._scene_render_scale_catalog[index]
                settings.scene_upscaler_scale = scale
                try:
                    lf.ui.set_scene_upscaler_scale(backend, scale)
                except AttributeError:
                    pass
            self._refresh_selection()

    def _scene_upscaler(self):
        settings = lf.get_render_settings()
        if settings is None:
            return "native"
        backend = str(settings.scene_upscaler)
        return backend if any(value == backend for value, _ in self._scene_upscaler_catalog) else "native"

    def _sync_scene_upscaler_catalog(self):
        try:
            records = lf.get_scene_upscaler_options()
            catalog = [
                (str(record["id"]), str(record["label_key"]))
                for record in records
                if record.get("id") and record.get("label_key")
            ]
            self._scene_upscaler_recommended_scales = {
                str(record["id"]): tuple(float(scale) for scale in record.get("recommended_input_scales", ()))
                for record in records
                if record.get("id")
            }
        except (AttributeError, KeyError, TypeError):
            catalog = list(self.DEFAULT_SCENE_UPSCALER_OPTIONS)
            self._scene_upscaler_recommended_scales = {}
        if not catalog:
            catalog = [("native", "preferences.scene_upscaler_native")]
        self._scene_upscaler_catalog = catalog

    def _recommended_upscaler_scale(self, backend=None):
        backend = backend or self._scene_upscaler()
        if backend == "amd-fsr3":
            return {
                "performance": 0.5,
                "balanced": 1.0 / 1.7,
                "quality": 1.0 / 1.5,
            }.get(self._temporal_quality(), 1.0 / 1.7)
        if backend != "nvidia-dlss":
            return 0.0
        scales = self._scene_upscaler_recommended_scales.get(backend, ())
        quality_index = {"performance": 0, "balanced": 1, "quality": 2}.get(
            self._temporal_quality(), 1
        )
        if len(scales) <= quality_index:
            return 0.0
        scale = float(scales[quality_index])
        return scale if 0.25 <= scale <= 1.0 else 0.0

    def _sync_scene_upscaler_records(self):
        self._sync_scene_upscaler_catalog()
        self._handle.update_record_list(
            "scene_upscalers",
            [
                {"index": str(index), "label": lf.ui.tr(label_key)}
                for index, (_value, label_key) in enumerate(self._scene_upscaler_catalog)
            ],
        )

    def _scene_upscaler_index(self):
        current = self._scene_upscaler()
        for index, (value, _label_key) in enumerate(self._scene_upscaler_catalog):
            if value == current:
                return str(index)
        return "0"

    def _set_scene_upscaler_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self._scene_upscaler_catalog):
            settings = lf.get_render_settings()
            if settings is not None:
                backend = self._scene_upscaler_catalog[index][0]
                try:
                    scale = self._recommended_upscaler_scale(backend)
                    if scale <= 0.0:
                        scale = float(lf.ui.get_scene_upscaler_scale(backend))
                except AttributeError:
                    scale = 1.0 if backend == "native" else float(settings.scene_upscaler_scale)
                try:
                    quality_id = str(lf.ui.get_scene_upscaler_quality(backend))
                except AttributeError:
                    quality_id = "balanced"
                quality = {"performance": 0, "balanced": 1, "quality": 2}.get(quality_id, 1)
                try:
                    settings.set_scene_upscaler(backend, scale, quality)
                except AttributeError:
                    settings.scene_upscaler = backend
                    settings.scene_upscaler_scale = scale
                    settings.scene_temporal_quality = quality_id
                self._sync_scene_render_scale_records()
            self._refresh_selection()

    def _temporal_quality(self):
        settings = lf.get_render_settings()
        if settings is None:
            return "balanced"
        # Resource hot reload can update this Python panel before the rebuilt
        # extension module is loaded. Keep the whole Preferences document
        # mountable instead of leaving unrelated selects at their first item.
        quality = str(getattr(settings, "scene_temporal_quality", "balanced"))
        return quality if quality in {"performance", "balanced", "quality"} else "balanced"

    def _temporal_quality_index(self):
        current = self._temporal_quality()
        for index, (value, _label_key) in enumerate(self.TEMPORAL_QUALITY_OPTIONS):
            if value == current:
                return str(index)
        return "1"

    def _set_temporal_quality_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self.TEMPORAL_QUALITY_OPTIONS):
            settings = lf.get_render_settings()
            if settings is not None:
                try:
                    quality_id = self.TEMPORAL_QUALITY_OPTIONS[index][0]
                    settings.scene_temporal_quality = quality_id
                    lf.ui.set_scene_upscaler_quality(self._scene_upscaler(), quality_id)
                except AttributeError:
                    return
            self._refresh_selection()

    def _language_index(self):
        current = lf.ui.get_current_language()
        for index, (code, _name) in enumerate(self._language_catalog):
            if code == current:
                return str(index)
        for index, (code, _name) in enumerate(self._language_catalog):
            if code == "en":
                return str(index)
        return "0"

    def _set_language_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self._language_catalog):
            lf.ui.set_language(self._language_catalog[index][0])
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

    def _read_mcp_preferences(self):
        preferences = lf.ui.get_mcp_preferences()
        self._mcp_enabled = bool(preferences.get("enabled", True))
        self._mcp_expose_network = bool(preferences.get("expose_network", False))
        self._mcp_port = str(preferences.get("port", 45677))
        self._mcp_applied_port = int(self._mcp_port)
        self._mcp_request_logging = bool(preferences.get("request_logging", False))
        self._last_mcp_runtime_config = self._mcp_runtime_config_signature()

    def _load_mcp_preferences(self):
        self._read_mcp_preferences()
        self._dirty_mcp()

    def _set_mcp_enabled(self, enabled):
        self._mcp_enabled = bool(enabled)
        self._apply_mcp_preferences()

    def _on_toggle_mcp_enabled(self, _handle, _event, _args):
        self._set_mcp_enabled(not self._mcp_enabled)

    def _set_mcp_expose_network(self, enabled):
        self._mcp_expose_network = bool(enabled)
        self._apply_mcp_preferences()

    def _set_mcp_port(self, value):
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
            return True
        return self._apply_mcp_preferences(port)

    def _set_mcp_request_logging(self, enabled):
        self._mcp_request_logging = bool(enabled)
        self._apply_mcp_preferences()

    def _apply_mcp_preferences(self, port=None):
        port = self._mcp_applied_port if port is None else port
        lf.ui.set_mcp_preferences(self._mcp_enabled, self._mcp_expose_network, port,
                                  self._mcp_request_logging)
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
            bool(status.get("expose_network")),
            int(status.get("port", 0)),
            int(status.get("request_count", 0)),
            int(status.get("success_count", 0)),
            int(status.get("error_count", 0)),
            bool(status.get("request_logging")),
            str(status.get("log_file", "")),
            str(status.get("error", "")),
            tuple(str(endpoint) for endpoint in status.get("endpoints") or ()),
        )

    def _mcp_runtime_config_signature(self):
        status = lf.ui.get_mcp_status()
        return (
            bool(status.get("enabled")),
            bool(status.get("expose_network")),
            int(status.get("port", 45677)),
            bool(status.get("request_logging")),
        )

    def _sync_mcp_runtime(self):
        signature = self._mcp_runtime_config_signature()
        if signature == self._last_mcp_runtime_config:
            return
        self._load_mcp_preferences()

    def _mcp_status_text(self):
        if self._validated_mcp_port() is None:
            return lf.ui.tr("preferences.mcp_status_error")
        status = lf.ui.get_mcp_status()
        if not status.get("enabled"):
            return lf.ui.tr("preferences.mcp_status_off")
        if status.get("running"):
            return lf.ui.tr("preferences.mcp_status_running")
        return lf.ui.tr("preferences.mcp_status_error")

    def _mcp_endpoint_text(self):
        status = lf.ui.get_mcp_status()
        endpoints = status.get("endpoints") or []
        if endpoints:
            return "\n".join(str(endpoint) for endpoint in endpoints)
        port = status.get("port", 45677)
        if status.get("expose_network"):
            return f"http://0.0.0.0:{port}/mcp"
        return f"http://127.0.0.1:{port}/mcp\nhttp://localhost:{port}/mcp"

    def _mcp_endpoint_rows(self):
        return min(10, max(2, len(self._mcp_endpoint_text().splitlines())))

    def _mcp_error_text(self):
        if self._validated_mcp_port() is None:
            return lf.ui.tr("preferences.mcp_invalid_port")
        return str(lf.ui.get_mcp_status().get("error", ""))

    def _mcp_log_file_text(self):
        return str(lf.ui.get_mcp_status().get("log_file", ""))

    def _on_open_mcp_log_folder(self, _handle, _event, _args):
        lf.ui.open_url(lf.ui.get_mcp_log_directory())

    def _consume_section_request(self):
        section = lf.ui.take_preferences_section_request()
        if section in ("general", "appearance", "input", "interface", "mcp"):
            self._set_section(section)

    def _dirty_mcp(self):
        if not self._handle:
            return
        for name in ("mcp_enabled", "mcp_expose_network", "mcp_port", "mcp_request_logging", "mcp_status",
                     "mcp_endpoint_value", "mcp_error", "mcp_has_error", "mcp_log_file",
                     "mcp_has_log_file"):
            self._handle.dirty(name)
        if self._document:
            endpoint_list = self._document.get_element_by_id("mcp-endpoints")
            if endpoint_list:
                endpoint_list.set_attribute("rows", str(self._mcp_endpoint_rows()))

    def _on_close(self, _handle, _event, _args):
        if not self._commit_mcp_port():
            return
        lf.ui.set_panel_enabled(self.id, False)

    def _set_section(self, section):
        if self._section == section:
            return
        self._section = section
        if self._handle:
            for name in ("show_general", "show_appearance", "show_input", "show_interface", "show_mcp",
                          "show_section_reset", "reset_section_label"):
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
            errors = [self._reset_section(section) for section in ("general", "appearance", "input", "interface", "mcp")]
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
        elif section == "appearance":
            lf.ui.set_theme("dark")
            lf.ui.set_ui_scale(0.0)
            try:
                lf.ui.set_viewer_splat_precision(16)
            except (AttributeError, RuntimeError):
                pass
            settings = lf.get_render_settings()
            if settings is not None:
                settings.render_scale = 1.0
                settings.scene_upscaler_scale = 1.0
                settings.scene_upscaler = "native"
                try:
                    for backend, _label_key in self._scene_upscaler_catalog:
                        lf.ui.set_scene_upscaler_scale(backend, 1.0)
                        lf.ui.set_scene_upscaler_quality(backend, "balanced")
                except AttributeError:
                    pass
                try:
                    settings.scene_temporal_quality = "balanced"
                except AttributeError:
                    pass
        elif section == "input":
            lf.ui.set_remember_camera_navigation(False)
            lf.ui.set_remember_camera_view_snap(False)
            lf.set_camera_navigation_mode("orbit")
            lf.set_camera_view_snap_enabled(False)
        elif section == "interface":
            return lf.ui.reset_layout()
        elif section == "mcp":
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
            self._handle.dirty("scene_render_scale_idx")
            self._handle.dirty("viewer_splat_precision_idx")
            self._handle.dirty("scene_upscaler_idx")
            self._handle.dirty("scene_upscaler_has_scale")
            self._handle.dirty("scene_upscaler_has_quality")
            self._handle.dirty("temporal_quality_idx")
            self._handle.dirty("language_idx")
            self._handle.dirty("navigation_idx")
            self._handle.dirty("view_snap")
            self._handle.dirty("remember_navigation")
            self._handle.dirty("remember_view_snap")
            self._dirty_mcp()
