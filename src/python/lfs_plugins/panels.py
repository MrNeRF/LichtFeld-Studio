# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Built-in plugin panel registration and per-panel GUIL chrome hooks."""

from __future__ import annotations

import traceback
from dataclasses import dataclass


@dataclass(frozen=True)
class _PanelSpec:
    module_name: str
    class_name: str
    id: str
    label: str
    space: str
    order: int
    template: str
    height_mode: str
    size: tuple[int, int]
    options: tuple[str, ...] = ("DEFAULT_CLOSED",)
    update_policy: str = "interval"
    update_interval_ms: int = 100
    style: str = ""
    has_poll: bool = False
    has_draw: bool = False


# Keep the registration contract in one place. The implementation classes use
# panel_metadata() too, so registration cannot silently drift from the class.
PANEL_SPECS = {
    "new_project": _PanelSpec(
        "lfs_plugins.import_panels", "NewProjectPanel", "lfs.new_project",
        "New Project", "FLOATING", 11, "rmlui/new_project_panel.rml",
        "CONTENT", (560, 0), update_policy="dirty",
    ),
    "resume_checkpoint": _PanelSpec(
        "lfs_plugins.import_panels", "ResumeCheckpointPanel", "lfs.resume_checkpoint",
        "Resume Checkpoint", "FLOATING", 12, "rmlui/resume_checkpoint_panel.rml",
        "CONTENT", (580, 0), update_policy="dirty",
    ),
    "export": _PanelSpec(
        "lfs_plugins.export_panel", "ExportPanel", "lfs.export", "Export",
        "FLOATING", 10, "rmlui/export_panel.rml", "CONTENT", (320, 0),
        update_policy="dirty",
    ),
    "about": _PanelSpec(
        "lfs_plugins.about_panel", "AboutPanel", "lfs.about", "About",
        "FLOATING", 100, "rmlui/about.rml", "CONTENT", (400, 0),
    ),
    "account": _PanelSpec(
        "lfs_plugins.account_panel", "AccountPanel", "lfs.account", "Account",
        "FLOATING", 95, "rmlui/account_panel.rml", "CONTENT", (440, 0),
    ),
    "bug_report": _PanelSpec(
        "lfs_plugins.bug_report_panel", "BugReportPanel", "lfs.bug_report",
        "Report a bug", "FLOATING", 96, "rmlui/bug_report_panel.rml",
        "CONTENT", (520, 0),
    ),
    "getting_started": _PanelSpec(
        "lfs_plugins.getting_started_panel", "GettingStartedPanel",
        "lfs.getting_started", "Getting Started", "FLOATING", 99,
        "rmlui/getting_started.rml", "CONTENT", (560, 0), update_policy="dirty",
    ),
    "image_preview": _PanelSpec(
        "lfs_plugins.image_preview_panel", "ImagePreviewPanel", "lfs.image_preview",
        "Image Preview", "FLOATING", 98, "rmlui/image_preview.rml", "FILL",
        (900, 600), update_policy="dirty",
    ),
    "histogram": _PanelSpec(
        "lfs_plugins.histogram_panel", "HistogramPanel", "lfs.histogram", "Histogram",
        "BOTTOM_DOCK", 97, "rmlui/histogram_panel.rml", "FILL", (860, 660),
        update_policy="dirty", has_poll=True,
    ),
    "scripts": _PanelSpec(
        "lfs_plugins.scripts_panel", "ScriptsPanel", "lfs.scripts", "Python Scripts",
        "FLOATING", 200, "rmlui/scripts_panel.rml", "CONTENT", (520, 0),
        update_policy="dirty",
    ),
    "preferences": _PanelSpec(
        "lfs_plugins.preferences_panel", "PreferencesPanel", "lfs.preferences",
        "Preferences", "FLOATING", 100, "rmlui/preferences.rml", "FILL", (780, 440), update_policy="dirty",
    ),
    "mesh2splat": _PanelSpec(
        "lfs_plugins.mesh2splat_panel", "Mesh2SplatPanel", "native.mesh2splat",
        "Mesh to Splat", "FLOATING", 12, "rmlui/mesh2splat_panel.rml", "CONTENT",
        (420, 0), update_policy="dirty",
    ),
    "plugin_marketplace": _PanelSpec(
        "lfs_plugins.plugin_marketplace_panel", "PluginMarketplacePanel",
        "lfs.plugin_marketplace", "Plugin Marketplace", "FLOATING", 91,
        "rmlui/plugin_marketplace.rml", "FILL", (770, 560),
        update_policy="interval", update_interval_ms=250,
    ),
    "asset_manager": _PanelSpec(
        "lfs_plugins.asset_manager_panel", "AssetManagerPanel", "lfs.asset_manager",
        "Asset Manager", "LEFT_DOCK", 20, "rmlui/asset_manager.rml", "FILL",
        (980, 620), update_policy="dirty",
    ),
}

_PANEL_METADATA_FIELDS = (
    "id", "label", "space", "order", "template", "height_mode", "size",
    "options", "update_policy", "update_interval_ms", "style",
)


def panel_metadata(name, lf):
    """Resolve a shared panel spec into class attributes for a runtime."""
    spec = PANEL_SPECS[name]
    return {
        "id": spec.id,
        "label": spec.label,
        "space": getattr(lf.ui.PanelSpace, spec.space),
        "order": spec.order,
        "template": spec.template,
        "height_mode": getattr(lf.ui.PanelHeightMode, spec.height_mode),
        "size": spec.size,
        "options": {getattr(lf.ui.PanelOption, option) for option in spec.options},
        "update_policy": spec.update_policy,
        "update_interval_ms": spec.update_interval_ms,
        "style": spec.style,
    }


def panel_class(name):
    """Decorate an implementation class with the shared panel contract."""
    import lichtfeld as lf

    metadata = panel_metadata(name, lf)

    def decorate(cls):
        for field in _PANEL_METADATA_FIELDS:
            setattr(cls, field, metadata[field])
        return cls

    return decorate


def capture_panel_chrome(panel):
    """Return a JSON-serializable dict from ``panel.capture_chrome()``, or None."""
    hook = getattr(panel, "capture_chrome", None)
    if hook is None:
        return None
    payload = hook()
    return payload if isinstance(payload, dict) else None


def apply_panel_chrome(panel, payload):
    """Deliver a payload (dict or None) to ``panel.apply_chrome()`` if present."""
    hook = getattr(panel, "apply_chrome", None)
    if hook is None:
        return
    hook(payload if isinstance(payload, dict) else {})


def _register_lazy_panel(lf, name):
    """Register cheap metadata now and import the implementation on first use."""
    from importlib import import_module

    spec = PANEL_SPECS[name]
    metadata = panel_metadata(name, lf)

    class LazyPanel(lf.ui.Panel):
        _implementation = None
        _implementation_module = spec.module_name
        _implementation_name = spec.class_name

        _delegated_instance_methods = frozenset({
            "poll", "draw", "show", "on_bind_model", "on_mount", "on_unmount",
            "on_update", "on_scene_changed", "capture_chrome", "apply_chrome",
        })

        def _load(self):
            if self._implementation is None:
                module = import_module(self._implementation_module)
                self._implementation = getattr(module, self._implementation_name)()
            return self._implementation

        def __getattribute__(self, attribute):
            if attribute in object.__getattribute__(self, "_delegated_instance_methods"):
                return getattr(object.__getattribute__(self, "_load")(), attribute)
            return super().__getattribute__(attribute)

        def __getattr__(self, attribute):
            return getattr(self._load(), attribute)

    LazyPanel.__name__ = spec.class_name
    LazyPanel.__qualname__ = spec.class_name
    LazyPanel.__module__ = "lfs_plugins.panels"
    for field in _PANEL_METADATA_FIELDS:
        setattr(LazyPanel, field, metadata[field])
    if spec.has_poll:
        def poll(self, context):
            return self._load().poll(context)

        LazyPanel.poll = poll
    if spec.has_draw:
        def draw(self, layout):
            return self._load().draw(layout)

        LazyPanel.draw = draw
    lf.register_class(LazyPanel)
    lf.ui.set_panel_enabled(spec.id, False)
    return LazyPanel


def __getattr__(name):
    """Lazy re-export so importing panels.py never loads marketplace eagerly."""
    if name == "PluginMarketplacePanel":
        from .plugin_marketplace_panel import PluginMarketplacePanel

        return PluginMarketplacePanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build_builtin_panel_steps(lf):
    """Return ordered (name, callable) registration steps.

    Rendering must stay first. Each callable is one cohesive unit (imports +
    register/enable/hook calls for that feature). Callables close over ``lf``.
    """

    def rendering_panel():
        from .rendering_panel import RenderingPanel

        lf.register_class(RenderingPanel)

    def training_panel():
        from .training_panel import TrainingPanel

        lf.register_class(TrainingPanel)

    def import_panels():
        _register_lazy_panel(lf, "new_project")
        _register_lazy_panel(lf, "resume_checkpoint")

    def selection_groups():
        from . import selection_groups as selection_groups_mod

        selection_groups_mod.register()

    def operators():
        from . import operators as operators_mod

        operators_mod.register()

    def sequencer_ops():
        from . import sequencer_ops as sequencer_ops_mod

        sequencer_ops_mod.register()

    def tools():
        from . import tools as tools_mod

        tools_mod.register()

    def menus():
        from . import file_menu, edit_menu, tools_menu, view_menu, help_menu

        file_menu.register()
        edit_menu.register()
        tools_menu.register()
        view_menu.register()
        help_menu.register()

    def export_panel():
        _register_lazy_panel(lf, "export")

    def about_panel():
        _register_lazy_panel(lf, "about")

    def account_panel():
        _register_lazy_panel(lf, "account")

    def bug_report_panel():
        _register_lazy_panel(lf, "bug_report")

    def portal_account():
        # Account session validation is a startup service, not panel UI. Keep
        # it eager so opening Account later sees the same initialized state.
        from .portal_account import initialize_portal_account

        initialize_portal_account()

    def getting_started_panel():
        _register_lazy_panel(lf, "getting_started")

    def startup_recent_panel():
        from .startup_recent_panel import StartupRecentPanel

        lf.register_class(StartupRecentPanel)
        lf.ui.set_panel_enabled("lfs.startup_recent", False)

    def image_preview_panel():
        _register_lazy_panel(lf, "image_preview")

        def open_camera_preview(uid):
            from .image_preview_panel import open_camera_preview_by_uid
            return open_camera_preview_by_uid(uid)

        lf.ui.on_open_camera_preview(open_camera_preview)

    def histogram_panel():
        _register_lazy_panel(lf, "histogram")

    def scripts_panel():
        _register_lazy_panel(lf, "scripts")

    def preferences_panel():
        _register_lazy_panel(lf, "preferences")

    def mesh2splat_panel():
        _register_lazy_panel(lf, "mesh2splat")

    def plugin_marketplace_panel():
        _register_lazy_panel(lf, "plugin_marketplace")

    def asset_manager_panel():
        _register_lazy_panel(lf, "asset_manager")

    def overlays():
        from .overlays import register as register_overlays

        register_overlays()

    return [
        ("rendering_panel", rendering_panel),
        ("training_panel", training_panel),
        ("import_panels", import_panels),
        ("selection_groups", selection_groups),
        ("operators", operators),
        ("sequencer_ops", sequencer_ops),
        ("tools", tools),
        ("menus", menus),
        ("export_panel", export_panel),
        ("about_panel", about_panel),
        ("account_panel", account_panel),
        ("bug_report_panel", bug_report_panel),
        ("portal_account", portal_account),
        ("getting_started_panel", getting_started_panel),
        ("startup_recent_panel", startup_recent_panel),
        ("image_preview_panel", image_preview_panel),
        ("histogram_panel", histogram_panel),
        ("scripts_panel", scripts_panel),
        ("preferences_panel", preferences_panel),
        ("mesh2splat_panel", mesh2splat_panel),
        ("plugin_marketplace_panel", plugin_marketplace_panel),
        ("asset_manager_panel", asset_manager_panel),
        ("overlays", overlays),
    ]


def register_builtin_panels():
    """Initialize built-in plugin system panels.

    Returns True once the registration loop has run, even if individual steps
    fail. Returns False only when ``import lichtfeld`` fails or the step-loop
    machinery itself raises. This keeps the C++ side from retrying/double-
    registering and still allows the dev hot-reload watcher to start.
    """
    try:
        import lichtfeld as lf
    except Exception as e:
        print(f"[ERROR] register_builtin_panels failed: {e}")
        traceback.print_exc()
        return False

    try:
        steps = _build_builtin_panel_steps(lf)
        failed = []
        for name, step in steps:
            try:
                step()
            except Exception as e:
                lf.log.error(
                    f"register_builtin_panels: step '{name}' failed: {e}\n{traceback.format_exc()}"
                )
                failed.append(name)
        if failed:
            lf.log.error(
                f"register_builtin_panels: {len(failed)} step(s) failed: {', '.join(failed)}"
            )
        return True
    except Exception as e:
        lf.log.error(
            f"register_builtin_panels failed: {e}\n{traceback.format_exc()}"
        )
        return False
