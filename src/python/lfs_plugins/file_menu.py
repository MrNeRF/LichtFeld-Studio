# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""File menu implementation using Blender-style operators."""

from pathlib import Path, PureWindowsPath

import lichtfeld as lf
from .asset_manager_integration import register_catalog_asset_path
from .types import Operator
from .layouts.menus import (
    menu_action,
    menu_operator,
    menu_separator,
    menu_submenu,
    menu_toggle,
    register_menu,
)
from .import_panels import open_dataset_import_panel, open_resume_checkpoint_panel
from .training_confirm import _project_has_path, confirm_discard_work_then

__lfs_menu_classes__ = ["FileMenu"]

class _ImportRejected(RuntimeError):
    def __init__(self, reason: str, message_key: str):
        super().__init__(reason)
        self.message_key = message_key


def _warn_import_failure(path: str, reason: str) -> None:
    message = f"Import rejected: path='{path}', reason='{reason}'"
    lf.log.warn(message)


def _show_import_failure(path: str, reason: str, message_key: str) -> None:
    _warn_import_failure(path, reason)
    message = lf.ui.tr(message_key).format(path=path, reason=reason)
    lf.ui.message_dialog(
        lf.ui.tr("menu.file.import_failed"), message, "error"
    )


def _run_import(path: str, callback) -> bool:
    try:
        callback()
        return True
    except Exception as exc:
        reason = str(exc).strip() or exc.__class__.__name__
        _show_import_failure(
            path,
            reason,
            getattr(
                exc,
                "message_key",
                "menu.file.import_failed_message",
            ),
        )
        return False


def _open_dataset_import_checked(path: str) -> None:
    if not lf.is_dataset_path(path):
        raise _ImportRejected(
            "dataset format was not recognized",
            "menu.file.dataset_not_recognized",
        )
    if not open_dataset_import_panel(path):
        raise RuntimeError("dataset import dialog is unavailable")


def _open_checkpoint_import_checked(path: str) -> None:
    if not lf.read_checkpoint_header(path):
        raise _ImportRejected(
            "checkpoint format was not recognized",
            "menu.file.checkpoint_not_recognized",
        )
    if not open_resume_checkpoint_panel(path):
        raise RuntimeError("checkpoint import dialog is unavailable")


def _offer_remove_missing_recent(path: str) -> None:
    tr = lf.ui.tr
    remove_label = tr("menu.file.remove_from_recent")

    def _on_result(button):
        if button == remove_label:
            lf.project_remove_recent_file(path)

    lf.ui.confirm_dialog(
        tr("menu.file.recent_missing_title"),
        tr("menu.file.recent_missing_message").format(path=path),
        [remove_label, tr("common.cancel")],
        _on_result,
    )


def _open_project(path: str, discard_changes: bool, stop_training: bool = False):
    if stop_training:
        return lf.project_open(path, discard_changes, True)
    return lf.project_open(path, discard_changes)


def _new_project(discard_changes: bool, stop_training: bool = False):
    if stop_training:
        return lf.new_project(discard_changes, True)
    return lf.new_project(discard_changes)


def _open_recent_checked(path: str, stop_training: bool = False) -> None:
    try:
        _open_project(path, True, stop_training)
    except FileNotFoundError:
        # NotFoundError subclasses FileNotFoundError (see startup_recent_panel).
        _offer_remove_missing_recent(path)
    except Exception as exc:
        message = str(exc).strip() or lf.ui.tr(
            "menu.file.recent_missing_message"
        ).format(path=path)
        lf.ui.message_dialog(
            lf.ui.tr("menu.file.open_project"), message, "error"
        )


def _open_recent_project(path: str) -> None:
    if not Path(path).is_file():
        _offer_remove_missing_recent(path)
        return
    confirm_discard_work_then(
        lf.ui.tr("menu.file.open_project"),
        lambda stop_training: _open_recent_checked(path, stop_training),
    )


def format_recent_project_entry(path: str, tr) -> tuple[str, str]:
    """Return the compact recent-project label and full-path tooltip."""
    windows_path = PureWindowsPath(path)
    display_path = windows_path if windows_path.drive or "\\" in path else Path(path)
    name = display_path.name or path
    anchor = display_path.anchor
    parent_parts = [
        part
        for part in display_path.parent.parts
        if part and part not in {anchor, "/", "\\"}
    ]
    parent = "/".join(parent_parts[-2:])
    if not parent:
        return name, path
    return tr("menu.file.recent_entry").format(name=name, parent=parent), path


class NewProjectOperator(Operator):
    label = "menu.file.new_project"
    description = "Clear the scene to start a new project"

    def execute(self, context) -> set:
        confirm_discard_work_then(
            lf.ui.tr("menu.file.new_project"),
            lambda stop_training: _new_project(True, stop_training),
        )
        return {"FINISHED"}


class OpenProjectOperator(Operator):
    label = "menu.file.open_project"
    description = "Open a LichtFeld project"

    def execute(self, context) -> set:
        confirm_discard_work_then(
            lf.ui.tr("menu.file.open_project"),
            lambda stop_training: _open_project("", True, stop_training),
        )
        return {"FINISHED"}


class SaveProjectOperator(Operator):
    label = "menu.file.save_project"
    description = "Save the active LichtFeld project"

    def execute(self, context) -> set:
        lf.project_save()
        return {"FINISHED"}


class SaveProjectAsOperator(Operator):
    label = "menu.file.save_project_as"
    description = "Save the active project to a new path"

    def execute(self, context) -> set:
        lf.project_save_as("")
        return {"FINISHED"}


class CompactProjectOperator(Operator):
    label = "menu.file.compact_project"
    description = "Reclaim dead bytes in the active LichtFeld project"

    def execute(self, context) -> set:
        lf.project_compact()
        return {"FINISHED"}


class ImportDatasetOperator(Operator):
    label = "menu.file.import_dataset"
    description = "Import a dataset folder"

    def execute(self, context) -> set:
        path = lf.ui.open_dataset_folder_dialog()
        if path and not _run_import(
            path, lambda: _open_dataset_import_checked(path)
        ):
            return {"CANCELLED"}
        return {"FINISHED"}


class ImportPlyOperator(Operator):
    label = "menu.file.import_ply"
    description = "Import a splat file"

    def execute(self, context) -> set:
        path = lf.ui.open_ply_file_dialog("")
        if path:
            def _load() -> None:
                register_catalog_asset_path(path, select=True)
                lf.load_file(path, is_dataset=False)

            if not _run_import(path, _load):
                return {"CANCELLED"}
        return {"FINISHED"}


class ImportMeshOperator(Operator):
    label = "menu.file.import_mesh"
    description = "Import a 3D mesh file"

    def execute(self, context) -> set:
        path = lf.ui.open_mesh_file_dialog("")
        if path:
            def _load() -> None:
                register_catalog_asset_path(
                    path,
                    asset_type="mesh",
                    role="reference",
                    select=True,
                )
                lf.load_file(path, is_dataset=False)

            if not _run_import(path, _load):
                return {"CANCELLED"}
        return {"FINISHED"}


class ImportCheckpointOperator(Operator):
    label = "menu.file.import_checkpoint"
    description = "Import a checkpoint file"

    def execute(self, context) -> set:
        path = lf.ui.open_checkpoint_file_dialog()
        if path and not _run_import(
            path, lambda: _open_checkpoint_import_checked(path)
        ):
            return {"CANCELLED"}
        return {"FINISHED"}


class ImportConfigOperator(Operator):
    label = "menu.file.import_config"
    description = "Import a configuration file"

    def execute(self, context) -> set:
        path = lf.ui.open_json_file_dialog()
        if path and not _run_import(path, lambda: lf.load_config_file(path)):
            return {"CANCELLED"}
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
        lf.request_exit()
        return {"FINISHED"}


def _show_exit_confirmation(training_in_progress: bool = False) -> None:
    tr = lf.ui.tr
    cancel_label = tr("common.cancel")
    lf.ui.set_exit_popup_open(True)

    if training_in_progress:
        stop_save_label = tr("exit_popup.stop_and_save")
        discard_label = tr("exit_popup.discard_and_exit")

        def _on_training_result(button):
            lf.ui.set_exit_popup_open(False)
            if button == stop_save_label:
                lf.stop_save_and_exit()
            elif button == discard_label:
                lf.force_exit()
            else:
                lf.cancel_exit()

        lf.ui.confirm_dialog(
            tr("exit_popup.training_title"),
            tr("exit_popup.training_message"),
            [stop_save_label, discard_label, cancel_label],
            _on_training_result,
        )
        return

    has_path = _project_has_path()
    save_label = (
        tr("common.save")
        if has_path
        else tr("menu.file.save_project_as")
    )
    discard_label = tr("exit_popup.discard")

    def _on_result(button):
        lf.ui.set_exit_popup_open(False)
        if button == save_label:
            if has_path:
                lf.save_and_exit()
            else:
                lf.save_as_and_exit()
        elif button == discard_label:
            lf.force_exit()
        else:
            lf.cancel_exit()

    lf.ui.confirm_dialog(
        tr("exit_popup.title"),
        tr("exit_popup.message") + "\n" + tr("exit_popup.unsaved_warning"),
        [save_label, discard_label, cancel_label],
        _on_result,
    )


def _show_project_switch_confirmation(
    new_project: bool, path: str
) -> None:
    if new_project:
        title = lf.ui.tr("menu.file.new_project")
        callback = lambda stop_training: _new_project(True, stop_training)
    else:
        title = lf.ui.tr("menu.file.open_project")
        callback = lambda stop_training: _open_project(
            path, True, stop_training
        )
    confirm_discard_work_then(title, callback)


def _show_stop_training_confirmation(
    new_project: bool, path: str, discard_changes: bool = False
) -> None:
    tr = lf.ui.tr
    yes_label = tr("common.yes")
    no_label = tr("common.no")

    def _on_result(button):
        if button != yes_label:
            return
        if new_project:
            _new_project(discard_changes, True)
        else:
            _open_project(path, discard_changes, True)

    lf.ui.confirm_dialog(
        tr("project_switch.stop_training_title"),
        tr("project_switch.stop_training_message"),
        [yes_label, no_label],
        _on_result,
    )


def _show_load_file_confirmation(paths, is_dataset: bool, replace: bool) -> None:
    title = lf.ui.tr(
        "load_dataset_popup.save_title" if is_dataset else "unsaved_work.title"
    )

    def _proceed(stop_training: bool) -> None:
        for i, path in enumerate(paths):
            lf.load_file(
                path,
                is_dataset=is_dataset,
                discard_changes=True,
                replace=(replace and i == 0),
                stop_training=stop_training,
            )

    confirm_discard_work_then(title, _proceed)


def _on_show_dataset_load_popup(path: str):
    open_dataset_import_panel(path)


def _on_show_resume_checkpoint_popup(path: str):
    open_resume_checkpoint_panel(path)


def _can_compact_project() -> bool:
    return _project_has_path()


@register_menu
class FileMenu:
    """File menu for the menu bar."""

    label = "menu.file"
    location = "MENU_BAR"
    order = 10

    def menu_items(self):
        recent_items = []
        for recent_path in lf.project_recent_files():
            path = str(recent_path)
            label, tooltip = format_recent_project_entry(path, lf.ui.tr)
            recent_items.append(
                menu_action(
                    label,
                    lambda selected=path: _open_recent_project(selected),
                    tooltip=tooltip,
                )
            )
        if not recent_items:
            recent_items.append(
                {
                    "type": "item",
                    "label": lf.ui.tr("menu.file.no_recent_projects"),
                    "callback": lambda: None,
                    "enabled": False,
                }
            )
        else:
            recent_items.append(menu_separator())
            recent_items.append(
                menu_action(
                    lf.ui.tr("menu.file.clear_recent_projects"),
                    lf.project_clear_recent_files,
                )
            )

        return [
            menu_operator(NewProjectOperator),
            menu_operator(OpenProjectOperator),
            menu_submenu(
                lf.ui.tr("menu.file.open_recent"),
                recent_items,
            ),
            menu_operator(
                SaveProjectOperator,
                shortcut="Ctrl+S",
            ),
            menu_operator(SaveProjectAsOperator),
            menu_operator(
                CompactProjectOperator,
                enabled=_can_compact_project(),
            ),
            menu_toggle(
                lf.ui.tr("menu.file.auto_save_on_close"),
                lambda: lf.project_set_auto_save_on_close(
                    not lf.project_auto_save_on_close_enabled()
                ),
                lf.project_auto_save_on_close_enabled(),
            ),
            menu_separator(),
            menu_submenu(
                lf.ui.tr("menu.file.import"),
                [
                    menu_operator(ImportDatasetOperator),
                    menu_operator(ImportPlyOperator),
                    menu_operator(ImportMeshOperator),
                    menu_operator(ImportCheckpointOperator),
                    menu_separator(),
                    menu_operator(ImportConfigOperator),
                ],
            ),
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
    OpenProjectOperator,
    SaveProjectOperator,
    SaveProjectAsOperator,
    CompactProjectOperator,
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
    lf.ui.on_request_exit(_show_exit_confirmation)
    lf.ui.on_project_switch_confirmation(
        _show_project_switch_confirmation
    )
    lf.ui.on_show_load_file_confirmation(
        _show_load_file_confirmation
    )
    lf.ui.on_stop_training_confirmation(
        _show_stop_training_confirmation
    )


def unregister():
    for cls in reversed(_operator_classes):
        lf.unregister_class(cls)
