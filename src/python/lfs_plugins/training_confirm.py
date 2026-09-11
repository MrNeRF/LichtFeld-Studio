# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared save prompts, training-stop confirms, and project-path probes for plugin UI."""

import lichtfeld as lf


_bound_proceed_generation = 0


def _training_is_active() -> bool:
    probe = getattr(lf, "is_training_active", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        return False


def _project_is_dirty() -> bool:
    probe = getattr(lf, "project_is_dirty", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        return False


def _confirm_stop_training_then(callback) -> None:
    if not _training_is_active():
        callback(False)
        return

    tr = lf.ui.tr
    yes_label = tr("common.yes")
    no_label = tr("common.no")

    def _on_result(button):
        if button == yes_label:
            callback(True)

    lf.ui.confirm_dialog(
        tr("project_switch.stop_training_title"),
        tr("project_switch.stop_training_message"),
        [yes_label, no_label],
        _on_result,
    )


def _project_has_path() -> bool:
    probe = getattr(lf, "project_has_path", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        return False


def _save_titled_project() -> bool:
    save = getattr(lf, "project_save", None)
    if not callable(save):
        return False
    try:
        result = save(wait=True)
    except TypeError:
        result = save(True, False)
    except Exception:
        return False
    return result is not False


def _invoke_project_save_as():
    save_as = getattr(lf, "project_save_as", None)
    if not callable(save_as):
        return False
    try:
        return save_as("", wait=True)
    except TypeError:
        return save_as("")
    except Exception:
        return False


def _schedule_once_project_bound(on_bound) -> None:
    scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
    if not callable(scheduler):
        return
    global _bound_proceed_generation
    _bound_proceed_generation += 1
    generation = _bound_proceed_generation

    def _on_ui():
        if generation != _bound_proceed_generation:
            return
        if _project_has_path():
            on_bound()

    scheduler(_on_ui)


def confirm_discard_work_then(
    title,
    on_proceed,
    *,
    message_key="exit_popup.unsaved_warning",
    ask_stop_training=True,
) -> None:
    def _after_save():
        if ask_stop_training:
            _confirm_stop_training_then(on_proceed)
        else:
            on_proceed(False)

    if not _project_is_dirty():
        _after_save()
        return

    tr = lf.ui.tr
    save_label = (
        tr("common.save") if _project_has_path() else tr("menu.file.save_project_as")
    )
    continue_label = tr("unsaved_work.continue_without_saving")
    cancel_label = tr("common.cancel")

    def _on_result(button):
        if button == save_label:
            if _project_has_path():
                if not _save_titled_project():
                    return
                _after_save()
                return
            accepted = _invoke_project_save_as()
            if accepted is False:
                return
            if _project_has_path():
                _after_save()
                return
            _schedule_once_project_bound(_after_save)
        elif button == continue_label:
            _after_save()

    lf.ui.confirm_dialog(
        title,
        tr(message_key),
        [save_label, continue_label, cancel_label],
        _on_result,
    )
