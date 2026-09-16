# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""File menu implementation using Blender-style operators."""

from pathlib import Path, PureWindowsPath

import lichtfeld as lf
from .asset_index import display_name
from .types import Operator
from .layouts.menus import (
    menu_action,
    menu_operator,
    menu_separator,
    menu_submenu,
    menu_toggle,
    register_menu,
)
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
    from .import_panels import open_new_project_panel

    if not lf.is_dataset_path(path):
        raise _ImportRejected(
            "dataset format was not recognized",
            "menu.file.dataset_not_recognized",
        )
    if not open_new_project_panel(path):
        raise RuntimeError("dataset import dialog is unavailable")


def _open_checkpoint_import_checked(path: str) -> None:
    from .import_panels import open_resume_checkpoint_panel

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


def _open_project(
    path: str,
    discard_changes: bool,
    stop_training: bool = False,
    keep_asset_manager_open: bool = False,
):
    if keep_asset_manager_open:
        return lf.project_open(
            path, discard_changes, stop_training, True
        )
    if stop_training:
        return lf.project_open(path, discard_changes, True)
    return lf.project_open(path, discard_changes)


def open_project_with_confirmation(
    path: str,
    *,
    keep_asset_manager_open: bool = False,
) -> None:
    """Open a known project path through the standard project-switch flow."""
    title = lf.ui.tr("menu.file.open_project")

    def _open_checked(stop_training: bool) -> None:
        try:
            _open_project(
                path,
                True,
                stop_training,
                keep_asset_manager_open=keep_asset_manager_open,
            )
        except Exception as exc:
            message = str(exc).strip() or title
            lf.ui.message_dialog(title, message, "error")

    confirm_discard_work_then(title, _open_checked)


def _new_project(discard_changes: bool, stop_training: bool = False):
    if stop_training:
        return lf.new_project(discard_changes, True)
    return lf.new_project(discard_changes)


def _open_recent_checked(
    path: str,
    stop_training: bool = False,
    keep_asset_manager_open: bool = False,
) -> None:
    try:
        _open_project(
            path,
            True,
            stop_training,
            keep_asset_manager_open=keep_asset_manager_open,
        )
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


def open_recent_project_with_confirmation(
    path: str,
    *,
    keep_asset_manager_open: bool = False,
) -> None:
    path = str(path)
    if not Path(path).is_file():
        _offer_remove_missing_recent(path)
        return
    confirm_discard_work_then(
        lf.ui.tr("menu.file.open_project"),
        lambda stop_training: _open_recent_checked(
            path, stop_training, keep_asset_manager_open
        ),
    )


def _open_recent_project(path: str) -> None:
    open_recent_project_with_confirmation(path)


def format_recent_project_entry(path: str, tr) -> tuple[str, str]:
    """Return the compact recent-project label and full-path tooltip."""
    windows_path = PureWindowsPath(path)
    display_path = windows_path if windows_path.drive or "\\" in path else Path(path)
    name = display_name({"path": display_path.as_posix(), "name": "", "name_origin": "stem"}) or path
    anchor = display_path.anchor
    parent_parts = [
        part
        for part in display_path.parent.parts
        if part and part not in {anchor, "/", "\\"}
    ]
    parent = "/".join(parent_parts[-2:])
    if not parent:
        return name, path
    template = tr("menu.file.recent_entry")
    if chr(0x2014) in template or chr(0x2013) in template:
        template = "{name} ({parent})"
    return template.format(name=name, parent=parent), path


class NewProjectOperator(Operator):
    label = "menu.file.new_project"
    description = "Create a new project"

    def execute(self, context) -> set:
        from .import_panels import open_new_project_panel

        open_new_project_panel("")
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


class EmbedDatasetOperator(Operator):
    label = "menu.file.embed_dataset"
    description = "Copy the external dataset into the project"

    @classmethod
    def poll(cls, context) -> bool:
        return bool(getattr(lf, "project_can_embed_dataset", lambda: False)())

    def execute(self, context) -> set:
        lf.project_embed_dataset()
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
                lf.load_file(path, is_dataset=False)

            if not _run_import(path, _load):
                return {"CANCELLED"}
        return {"FINISHED"}


class ImportSsogOperator(Operator):
    label = "menu.file.import_ssog"
    description = "Import a SSOG folder containing lod-meta.json"

    def execute(self, context) -> set:
        path = lf.ui.open_folder_dialog()
        if not path:
            return {"CANCELLED"}
        if not _run_import(path, lambda: lf.load_file(path, is_dataset=False)):
            return {"CANCELLED"}
        return {"FINISHED"}


class ImportMeshOperator(Operator):
    label = "menu.file.import_mesh"
    description = "Import a 3D mesh file"

    def execute(self, context) -> set:
        path = lf.ui.open_mesh_file_dialog("")
        if path:
            def _load() -> None:
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
    new_project: bool,
    path: str,
    keep_asset_manager_open: bool = False,
    create_path: str = "",
    overwrite: bool = False,
) -> None:
    if new_project:
        title = lf.ui.tr("menu.file.new_project")
        if create_path:
            callback = lambda stop_training: lf.project_create(
                create_path,
                discard_changes=True,
                stop_training=stop_training,
                overwrite=overwrite,
            )
        else:
            callback = lambda stop_training: _new_project(True, stop_training)
    else:
        title = lf.ui.tr("menu.file.open_project")
        callback = lambda stop_training: _open_project(
            path, True, stop_training, keep_asset_manager_open
        )
    confirm_discard_work_then(title, callback)


def _show_stop_training_confirmation(
    new_project: bool,
    path: str,
    discard_changes: bool = False,
    keep_asset_manager_open: bool = False,
    create_path: str = "",
    overwrite: bool = False,
) -> None:
    tr = lf.ui.tr
    yes_label = tr("common.yes")
    no_label = tr("common.no")

    def _on_result(button):
        if button != yes_label:
            return
        if new_project and create_path:
            lf.project_create(
                create_path,
                discard_changes=True,
                stop_training=True,
                overwrite=overwrite,
            )
        elif new_project:
            _new_project(discard_changes, True)
        else:
            _open_project(
                path,
                discard_changes,
                True,
                keep_asset_manager_open,
            )

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


def _on_show_new_project_dialog(path: str):
    from .import_panels import open_new_project_panel

    open_new_project_panel(path)


def _on_show_resume_checkpoint_popup(path: str):
    from .import_panels import open_resume_checkpoint_panel

    open_resume_checkpoint_panel(path)


def _can_compact_project() -> bool:
    return _project_has_path()


def _publish_current_project_to_gallery() -> None:
    """Open the shared Gallery review for the active saved project."""
    from .gallery_messages import tr as gallery_tr

    title = lf.ui.tr("menu.file.publish_to_gallery")
    try:
        if not _project_has_path():
            raise ValueError(gallery_tr("error.save_first"))
        poll = lf.project_poll_write()
        raw_path = str(poll.get("path") or "")
        if not raw_path:
            raise ValueError(gallery_tr("error.save_first"))
        path = str(Path(raw_path).resolve())
        project_path = Path(path)
        if not project_path.is_file():
            raise FileNotFoundError(f"The saved project file was not found: {path}")

        card = lf.io.inspect_project_card(path)
        project_id = str(card.project_uuid)
        if not project_id:
            raise ValueError(gallery_tr("error.project_changed"))

        from .gallery_controller import get_gallery_controller
        from .gallery_file_panel import open_gallery_file_panel

        controller = get_gallery_controller()
        state = controller.snapshot()
        link = state.get("links", {}).get(project_id)
        scene = None
        if link:
            scene = next((row for row in state.get("scenes", [])
                          if row.get("id") == link.get("sceneId")), None)
            if scene is None:
                raise ValueError(gallery_tr("error.refresh"))
        linked_fields = ((link or {}).get("localFields") or (link or {}).get("sharedFields")
                         or scene or {})
        project_name = str(getattr(card, "title", None) or project_path.stem)
        asset = {
            "id": project_id,
            "path": path,
            "name": project_name,
            "commit_uuid": str(card.commit_uuid),
            "file_uuid": str(card.file_uuid),
            "file_size_bytes": int(card.physical_file_size),
            "has_preview": bool(card.has_preview),
            "exists": True,
            "status": "AVAILABLE",
            "publication": {},
        }
        fields = {
            "title": str(linked_fields.get("title") or project_name),
            "description": str(linked_fields.get("description") or ""),
            "visibility": linked_fields.get("visibility", "private"),
            "upload_format": controller.upload_format,
        }
        from .gallery_actions import gallery_quota
        quota_bytes, used_bytes, remaining_bytes = gallery_quota(state)
        quota = (gallery_tr("quota.used", used=f"{used_bytes / 1e9:.1f}",
                            quota=f"{quota_bytes / 1e9:g}")
                 if quota_bytes is not None else "")
        warning = (gallery_tr("quota.warning")
                   if remaining_bytes is not None and asset["file_size_bytes"] > remaining_bytes
                   else "")

        # Use the same primary verb as Asset Manager. A linked project may be
        # locally newer, remotely newer, divergent, or not checked; treating
        # every link as an update can bypass the corresponding Gallery action.
        publish_new = False
        action = "publish"
        if link:
            from .gallery_actions import gallery_actions
            from .gallery_controller import asset_sync_state

            facts = asset_sync_state(
                asset,
                link,
                scene,
                state.get("jobs", ()),
                checked=bool(state.get("checkedAt")),
                established=state.get("established", state.get("connected", True)),
            )
            for key in ("signed_in", "busy", "relink_required", "unsupported", "source_formats",
                        "quotaBytes", "usedBytes", "reservedBytes", "hdrBackgrounds"):
                if key in state:
                    facts[key] = state[key]
            primary = next((item for item in gallery_actions(asset, facts) if item["primary"]), None)
            if not primary:
                raise ValueError(gallery_tr("error.refresh"))
            if not primary["enabled"]:
                raise ValueError(primary["reason"] or gallery_tr("error.refresh"))

            action = primary["id"]
            details = {key: fields[key] for key in ("title", "description", "visibility")}
            if action == "check":
                controller.refresh()
                lf.ui.message_dialog(title, gallery_tr("error.refresh"), "info")
                return
            if action in ("resolve", "apply"):
                controller.resolve_asset(asset, details, apply_only=action == "apply")
                return
            if action in ("open", "copy"):
                # File → Publish remains a publishing workflow when the link
                # is already current: the review can still update its metadata.
                action = "update"
            if action == "publish_again":
                publish_new = True
                action = "publish"
            elif action not in ("publish", "update"):
                raise ValueError(gallery_tr("error.refresh"))

        open_gallery_file_panel(
            controller=controller,
            asset=asset,
            scene=scene,
            action=action,
            fields=fields,
            quota=quota,
            warning=warning,
            publish_new=publish_new,
            expected_project_path=path,
        )
    except Exception as exc:
        message = str(exc).strip() or title
        lf.ui.message_dialog(title, message, "error")


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
            menu_action(
                lf.ui.tr("menu.file.publish_to_gallery"),
                _publish_current_project_to_gallery,
                enabled=_can_compact_project(),
            ),
            menu_operator(EmbedDatasetOperator, enabled=bool(getattr(lf, "project_can_embed_dataset", lambda: False)())),
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
                    menu_operator(ImportSsogOperator),
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
    EmbedDatasetOperator,
    CompactProjectOperator,
    ImportDatasetOperator,
    ImportPlyOperator,
    ImportSsogOperator,
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

    lf.ui.on_show_new_project_dialog(_on_show_new_project_dialog)
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
