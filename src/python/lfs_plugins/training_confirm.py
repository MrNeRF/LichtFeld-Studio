# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared training-stop confirms and project-path probes for plugin UI."""

import lichtfeld as lf


def _training_is_active() -> bool:
    probe = getattr(lf, "is_training_active", None)
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
