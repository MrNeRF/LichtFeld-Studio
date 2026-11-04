# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""File menu operators that open the New Project dialog on a chosen video or photo folder."""
import lichtfeld as lf

from ..types import Operator


class NewProjectFromVideoOperator(Operator):
    label = "menu.file.new_project_from_video"
    description = "Create a project from a video: extract frames, solve cameras, train"

    def execute(self, context) -> set:
        path = lf.ui.open_video_file_dialog()
        if not path:
            return {"CANCELLED"}
        from ..import_panels import open_new_project_panel

        open_new_project_panel(path)
        return {"FINISHED"}


class NewProjectFromPhotosOperator(Operator):
    label = "menu.file.new_project_from_photos"
    description = "Create a project from a folder of photos: solve cameras, train"

    def execute(self, context) -> set:
        path = lf.ui.open_folder_dialog(lf.ui.tr("new_project.choose_images"), "")
        if not path:
            return {"CANCELLED"}
        from ..import_panels import open_new_project_panel

        open_new_project_panel(path)
        return {"FINISHED"}
