# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""File menu implementation using Blender-style operators."""

import lichtfeld as lf
from .types import Operator
from .layouts.menus import register_menu
from .popups import ExitConfirmationPopup, SaveDirectoryPopup, ResumeCheckpointPopup

_exit_popup = ExitConfirmationPopup()
_save_dir_popup = SaveDirectoryPopup()
_resume_popup = ResumeCheckpointPopup()


class NewProjectOperator(Operator):
    label = "menu.file.new_project"
    description = "Clear the scene to start a new project"

    def execute(self, context) -> set:
        lf.clear_scene()
        return {"FINISHED"}


class ImportDatasetOperator(Operator):
    label = "menu.file.import_dataset"
    description = "Import a dataset folder"

    def execute(self, context) -> set:
        path = lf.ui.open_dataset_folder_dialog()
        if path:
            _save_dir_popup.show(path, on_confirm=_on_dataset_load)
        return {"FINISHED"}


class ImportPlyOperator(Operator):
    label = "menu.file.import_ply"
    description = "Import a PLY file"

    def execute(self, context) -> set:
        path = lf.ui.open_ply_file_dialog("")
        if path:
            lf.load_file(path, is_dataset=False)
        return {"FINISHED"}


class ImportMeshOperator(Operator):
    label = "menu.file.import_mesh"
    description = "Import a 3D mesh file"

    def execute(self, context) -> set:
        path = lf.ui.open_mesh_file_dialog("")
        if path:
            lf.load_file(path, is_dataset=False)
        return {"FINISHED"}


class ImportCheckpointOperator(Operator):
    label = "menu.file.import_checkpoint"
    description = "Import a checkpoint file"

    def execute(self, context) -> set:
        path = lf.ui.open_checkpoint_file_dialog()
        if path:
            _resume_popup.show(path, on_confirm=_on_checkpoint_load)
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
        _exit_popup.show(on_confirm=lf.force_exit)
        return {"FINISHED"}


def _on_dataset_load(params):
    lf.load_file(
        str(params.dataset_path),
        is_dataset=True,
        output_path=str(params.output_path),
        init_path=str(params.init_path) if params.init_path else "",
    )


def _on_checkpoint_load(params):
    lf.load_checkpoint_for_training(
        str(params.checkpoint_path), str(params.dataset_path), str(params.output_path)
    )


def _on_show_dataset_load_popup(path: str):
    _save_dir_popup.show(path, on_confirm=_on_dataset_load)


def _on_show_resume_checkpoint_popup(path: str):
    _resume_popup.show(path, on_confirm=_on_checkpoint_load)


def _draw_popups(layout):
    _exit_popup.draw(layout)
    _save_dir_popup.draw(layout)
    _resume_popup.draw(layout)


@register_menu
class FileMenu:
    """File menu for the menu bar."""

    label = "menu.file"
    location = "MENU_BAR"
    order = 10

    def draw(self, layout):
        layout.operator_(NewProjectOperator._class_id())
        layout.separator()
        layout.operator_(ImportDatasetOperator._class_id())
        layout.operator_(ImportPlyOperator._class_id())
        layout.operator_(ImportMeshOperator._class_id())
        layout.operator_(ImportCheckpointOperator._class_id())
        layout.operator_(ImportConfigOperator._class_id())
        layout.separator()
        layout.operator_(ExportOperator._class_id())
        layout.operator_(ExportConfigOperator._class_id())
        layout.separator()
        layout.operator_(Mesh2SplatOperator._class_id())
        layout.operator_(ExtractVideoFramesOperator._class_id())
        layout.separator()
        layout.operator_(ExitOperator._class_id())


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

    lf.ui.register_popup_draw_callback(_draw_popups)
    lf.ui.on_show_dataset_load_popup(_on_show_dataset_load_popup)
    lf.ui.on_show_resume_checkpoint_popup(_on_show_resume_checkpoint_popup)
    lf.ui.on_request_exit(lambda: lf.ui.execute_operator(ExitOperator._class_id()))


def unregister():
    lf.ui.unregister_popup_draw_callback(_draw_popups)

    for cls in reversed(_operator_classes):
        lf.unregister_class(cls)
