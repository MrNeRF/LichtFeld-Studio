# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Built-in plugin panel registration."""

from .plugin_marketplace_panel import PluginMarketplacePanel


def register_builtin_panels():
    """Initialize built-in plugin system panels."""
    try:
        import lichtfeld as lf

        # Main panel tabs (Rendering must be first)
        from .rendering_panel import RenderingPanel
        lf.register_class(RenderingPanel)

        from .training_panel import TrainingPanel
        lf.register_class(TrainingPanel)

        from .scene_panel import ScenePanel
        lf.register_class(ScenePanel)

        from .toolbar import GizmoToolbar, UtilityToolbar
        lf.register_class(UtilityToolbar)
        lf.register_class(GizmoToolbar)

        from . import selection_groups
        selection_groups.register()

        from . import transform_controls
        transform_controls.register()

        from . import operators
        operators.register()

        from . import sequencer_ops
        sequencer_ops.register()

        from . import tools
        tools.register()

        from . import file_menu, edit_menu, view_menu, help_menu
        file_menu.register()
        edit_menu.register()
        view_menu.register()
        help_menu.register()

        # Floating panels
        from .export_panel import ExportPanel
        lf.register_class(ExportPanel)
        lf.ui.set_panel_enabled("lfs.export", False)

        from .about_panel import AboutPanel
        lf.register_class(AboutPanel)
        lf.ui.set_panel_enabled("lfs.about", False)

        from .getting_started_panel import GettingStartedPanel
        lf.register_class(GettingStartedPanel)
        lf.ui.set_panel_enabled("lfs.getting_started", False)

        from .image_preview_panel import ImagePreviewPanel
        lf.register_class(ImagePreviewPanel)
        lf.ui.set_panel_enabled("lfs.image_preview", False)

        from .scripts_panel import ScriptsPanel
        lf.register_class(ScriptsPanel)
        lf.ui.set_panel_enabled("lfs.scripts", False)

        from .input_settings_panel import InputSettingsPanel
        lf.register_class(InputSettingsPanel)
        lf.ui.set_panel_enabled("lfs.input_settings", False)

        lf.register_class(PluginMarketplacePanel)
        lf.ui.set_panel_enabled("lfs.plugin_marketplace", False)

        # Status bar (must be registered last, always visible)
        from .status_bar_panel import StatusBarPanel
        lf.register_class(StatusBarPanel)

        # Viewport overlays
        from .overlays import register as register_overlays
        register_overlays()
    except Exception as e:
        import traceback
        print(f"[ERROR] register_builtin_panels failed: {e}")
        traceback.print_exc()
