# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Built-in plugin panel registration."""

import traceback


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
        from .import_panels import DatasetImportPanel, ResumeCheckpointPanel, URLImportPanel

        lf.register_class(DatasetImportPanel)
        lf.ui.set_panel_enabled("lfs.dataset_import", False)
        lf.register_class(ResumeCheckpointPanel)
        lf.ui.set_panel_enabled("lfs.resume_checkpoint", False)
        lf.register_class(URLImportPanel)
        lf.ui.set_panel_enabled("lfs.url_import", False)

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
        from .export_panel import ExportPanel

        lf.register_class(ExportPanel)
        lf.ui.set_panel_enabled("lfs.export", False)

    def about_panel():
        from .about_panel import AboutPanel

        lf.register_class(AboutPanel)
        lf.ui.set_panel_enabled("lfs.about", False)

    def account_panel():
        from .account_panel import AccountPanel

        lf.register_class(AccountPanel)
        lf.ui.set_panel_enabled("lfs.account", False)

    def bug_report_panel():
        from .bug_report_panel import BugReportPanel

        lf.register_class(BugReportPanel)
        lf.ui.set_panel_enabled("lfs.bug_report", False)

    def portal_account():
        from .portal_account import initialize_portal_account

        initialize_portal_account()

    def getting_started_panel():
        from .getting_started_panel import GettingStartedPanel

        lf.register_class(GettingStartedPanel)
        lf.ui.set_panel_enabled("lfs.getting_started", False)

    def image_preview_panel():
        from .image_preview_panel import ImagePreviewPanel, open_camera_preview_by_uid

        lf.register_class(ImagePreviewPanel)
        lf.ui.set_panel_enabled("lfs.image_preview", False)
        lf.ui.on_open_camera_preview(open_camera_preview_by_uid)

    def histogram_panel():
        from .histogram_panel import HistogramPanel

        lf.register_class(HistogramPanel)
        lf.ui.set_panel_enabled("lfs.histogram", False)

    def scripts_panel():
        from .scripts_panel import ScriptsPanel

        lf.register_class(ScriptsPanel)
        lf.ui.set_panel_enabled("lfs.scripts", False)

    def input_settings_panel():
        from .input_settings_panel import InputSettingsPanel

        lf.register_class(InputSettingsPanel)
        lf.ui.set_panel_enabled("lfs.input_settings", False)

    def mesh2splat_panel():
        from .mesh2splat_panel import Mesh2SplatPanel

        lf.register_class(Mesh2SplatPanel)
        lf.ui.set_panel_enabled("native.mesh2splat", False)

    def plugin_marketplace_panel():
        from .plugin_marketplace_panel import PluginMarketplacePanel

        lf.register_class(PluginMarketplacePanel)
        lf.ui.set_panel_enabled("lfs.plugin_marketplace", False)

    def asset_manager_panel():
        # Eager import to register save callbacks
        from . import asset_manager_integration  # noqa: F401
        from .asset_manager_panel import AssetManagerPanel

        lf.register_class(AssetManagerPanel)
        lf.ui.set_panel_enabled("lfs.asset_manager", False)

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
        ("image_preview_panel", image_preview_panel),
        ("histogram_panel", histogram_panel),
        ("scripts_panel", scripts_panel),
        ("input_settings_panel", input_settings_panel),
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
                    "register_builtin_panels: step '%s' failed: %s\n%s",
                    name,
                    e,
                    traceback.format_exc(),
                )
                failed.append(name)
        if failed:
            lf.log.error(
                "register_builtin_panels: %d step(s) failed: %s",
                len(failed),
                ", ".join(failed),
            )
        return True
    except Exception as e:
        lf.log.error(
            "register_builtin_panels failed: %s\n%s",
            e,
            traceback.format_exc(),
        )
        return False
