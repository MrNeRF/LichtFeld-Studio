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

        from .import_panels import DatasetImportPanel, ResumeCheckpointPanel
        lf.register_class(DatasetImportPanel)
        lf.ui.set_panel_enabled("lfs.dataset_import", False)
        lf.register_class(ResumeCheckpointPanel)
        lf.ui.set_panel_enabled("lfs.resume_checkpoint", False)

        from . import selection_groups
        selection_groups.register()

        from . import operators
        operators.register()

        from . import sequencer_ops
        sequencer_ops.register()

        from . import tools
        tools.register()

        from . import file_menu, edit_menu, tools_menu, view_menu, help_menu
        file_menu.register()
        edit_menu.register()
        tools_menu.register()
        view_menu.register()
        help_menu.register()

        # Auxiliary panels
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

        from .image_preview_panel import open_camera_preview_by_uid
        lf.ui.on_open_camera_preview(open_camera_preview_by_uid)

        from .histogram_panel import HistogramPanel
        lf.register_class(HistogramPanel)
        lf.ui.set_panel_enabled("lfs.histogram", False)

        from .scripts_panel import ScriptsPanel
        lf.register_class(ScriptsPanel)
        lf.ui.set_panel_enabled("lfs.scripts", False)

        from .input_settings_panel import InputSettingsPanel
        lf.register_class(InputSettingsPanel)
        lf.ui.set_panel_enabled("lfs.input_settings", False)

        from .mesh2splat_panel import Mesh2SplatPanel
        lf.register_class(Mesh2SplatPanel)
        lf.ui.set_panel_enabled("native.mesh2splat", False)

        lf.register_class(PluginMarketplacePanel)
        lf.ui.set_panel_enabled("lfs.plugin_marketplace", False)

        # Viewport overlays
        from .overlays import register as register_overlays
        register_overlays()
        return True
    except Exception as e:
        import traceback
        print(f"[ERROR] register_builtin_panels failed: {e}")
        traceback.print_exc()
        return False
