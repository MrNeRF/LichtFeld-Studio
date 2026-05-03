# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""File menu implementation using Blender-style operators."""

import lichtfeld as lf
from .asset_manager_integration import register_catalog_asset_path
from .types import Operator
from .layouts.menus import register_menu, menu_operator, menu_separator
from .import_panels import open_dataset_import_panel, open_resume_checkpoint_panel

__lfs_menu_classes__ = ["FileMenu"]


class NewProjectOperator(Operator):
    label = "menu.file.new_project"
    description = "Clear the scene to start a new project"

    def execute(self, context) -> set:
        lf.new_project()
        return {"FINISHED"}


class ImportDatasetOperator(Operator):
    label = "menu.file.import_dataset"
    description = "Import a dataset folder"

    def execute(self, context) -> set:
        path = lf.ui.open_dataset_folder_dialog()
        if path:
            open_dataset_import_panel(path)
        return {"FINISHED"}


class ImportPlyOperator(Operator):
    label = "menu.file.import_ply"
    description = "Import a splat file"

    def execute(self, context) -> set:
        path = lf.ui.open_ply_file_dialog("")
        if path:
            register_catalog_asset_path(path, select=True)
            lf.load_file(path, is_dataset=False)
        return {"FINISHED"}


class ImportMeshOperator(Operator):
    label = "menu.file.import_mesh"
    description = "Import a 3D mesh file"

    def execute(self, context) -> set:
        path = lf.ui.open_mesh_file_dialog("")
        if path:
            register_catalog_asset_path(
                path,
                asset_type="mesh",
                role="reference",
                select=True,
            )
            lf.load_file(path, is_dataset=False)
        return {"FINISHED"}


class ImportCheckpointOperator(Operator):
    label = "menu.file.import_checkpoint"
    description = "Import a checkpoint file"

    def execute(self, context) -> set:
        path = lf.ui.open_checkpoint_file_dialog()
        if path:
            open_resume_checkpoint_panel(path)
        return {"FINISHED"}


class ImportConfigOperator(Operator):
    label = "menu.file.import_config"
    description = "Import a configuration file"

    def execute(self, context) -> set:
        path = lf.ui.open_json_file_dialog()
        if path:
            lf.load_config_file(path)
        return {"FINISHED"}


class ExportOperator(Operator):
    label = "menu.file.export"
    description = "Export the scene"

    def execute(self, context) -> set:
        lf.ui.set_panel_enabled("lfs.export", True)
        return {"FINISHED"}


class ExportConfigOperator(Operator):
    label = "menu.file.export_config"
    description = "Export the current configuration"

    def execute(self, context) -> set:
        path = lf.ui.save_json_file_dialog("config.json")
        if path:
            lf.save_config_file(path)
        return {"FINISHED"}


class Mesh2SplatOperator(Operator):
    label = "menu.file.mesh_to_splat"
    description = "Convert a mesh to Gaussian splats"

    def execute(self, context) -> set:
        lf.ui.set_panel_enabled("native.mesh2splat", True)
        return {"FINISHED"}


class ExtractVideoFramesOperator(Operator):
    label = "menu.file.extract_video_frames"
    description = "Extract frames from a video file"

    def execute(self, context) -> set:
        lf.ui.set_panel_enabled("native.video_extractor", True)
        return {"FINISHED"}


class ExitOperator(Operator):
    label = "menu.file.exit"
    description = "Exit the application"

    def execute(self, context) -> set:
        tr = lf.ui.tr
        lf.ui.set_exit_popup_open(True)

        def _on_result(button):
            lf.ui.set_exit_popup_open(False)
            if button == tr("exit_popup.exit"):
                lf.force_exit()

        lf.ui.confirm_dialog(
            tr("exit_popup.title"),
            tr("exit_popup.message") + "\n" + tr("exit_popup.unsaved_warning"),
            [tr("common.cancel"), tr("exit_popup.exit")],
            _on_result,
        )
        return {"FINISHED"}


def _on_show_dataset_load_popup(path: str):
    open_dataset_import_panel(path)


def _on_show_resume_checkpoint_popup(path: str):
    open_resume_checkpoint_panel(path)


@register_menu
class FileMenu:
    """File menu for the menu bar."""

    label = "menu.file"
    location = "MENU_BAR"
    order = 10

    def menu_items(self):
        return [
            menu_operator(NewProjectOperator),
            menu_separator(),
            menu_operator(ImportDatasetOperator),
            menu_operator(ImportPlyOperator),
            menu_operator(ImportMeshOperator),
            menu_operator(ImportCheckpointOperator),
            menu_operator(ImportConfigOperator),
            menu_separator(),
            menu_operator(ExportOperator),
            menu_operator(ExportConfigOperator),
            menu_separator(),
            menu_operator(Mesh2SplatOperator),
            menu_operator(ExtractVideoFramesOperator),
            menu_separator(),
            menu_operator(ExitOperator),
        ]


_operator_classes = [
    NewProjectOperator,
    ImportDatasetOperator,
    ImportPlyOperator,
    ImportMeshOperator,
    ImportCheckpointOperator,
    ImportConfigOperator,
    ExportOperator,
    ExportConfigOperator,
    Mesh2SplatOperator,
    ExtractVideoFramesOperator,
    ExitOperator,
]


def register():
    for cls in _operator_classes:
        lf.register_class(cls)

    lf.ui.on_show_dataset_load_popup(_on_show_dataset_load_popup)
    lf.ui.on_show_resume_checkpoint_popup(_on_show_resume_checkpoint_popup)
    lf.ui.on_request_exit(lambda: lf.ui.execute_operator(ExitOperator._class_id()))


def unregister():
    for cls in reversed(_operator_classes):
        lf.unregister_class(cls)
