"""Import progress overlay - modal dialog during dataset import."""

import lichtfeld as lf
from ..types import Panel

BACKDROP_FLAGS = (
    lf.ui.UILayout.WindowFlags.NoTitleBar
    | lf.ui.UILayout.WindowFlags.NoResize
    | lf.ui.UILayout.WindowFlags.NoMove
    | lf.ui.UILayout.WindowFlags.NoScrollbar
    | lf.ui.UILayout.WindowFlags.NoInputs
    | lf.ui.UILayout.WindowFlags.NoBackground
    | lf.ui.UILayout.WindowFlags.NoFocusOnAppearing
    | lf.ui.UILayout.WindowFlags.NoBringToFrontOnFocus
)

DIALOG_FLAGS = (
    lf.ui.UILayout.WindowFlags.NoTitleBar
    | lf.ui.UILayout.WindowFlags.NoResize
    | lf.ui.UILayout.WindowFlags.NoMove
    | lf.ui.UILayout.WindowFlags.NoScrollbar
    | lf.ui.UILayout.WindowFlags.NoCollapse
    | lf.ui.UILayout.WindowFlags.AlwaysAutoResize
)

OVERLAY_WIDTH = 400.0
BUTTON_WIDTH = 100.0
BUTTON_HEIGHT = 30.0
AUTO_DISMISS_SECONDS = 2.0
BACKDROP_COLOR = (0.0, 0.0, 0.0, 0.4)
VERTICAL_OFFSET = 0.4


class ImportProgressOverlay(Panel):
    """Modal overlay showing import progress and completion status."""

    label = "##ImportProgress"
    space = "VIEWPORT_OVERLAY"
    order = 160

    @classmethod
    def poll(cls, context):
        if not hasattr(lf.ui, "get_import_state"):
            return False
        state = lf.ui.get_import_state()
        is_active = state.get("active", False)
        show_completion = state.get("show_completion", False)

        if show_completion and not is_active:
            if state.get("success", False) and state.get("seconds_since_completion", 0.0) > AUTO_DISMISS_SECONDS:
                lf.ui.dismiss_import()
                return False

        return is_active or show_completion

    def draw(self, layout):
        state = lf.ui.get_import_state()
        is_active = state.get("active", False)
        show_completion = state.get("show_completion", False)

        if not is_active and not show_completion:
            return

        vp_x, vp_y = layout.get_viewport_pos()
        vp_w, vp_h = layout.get_viewport_size()
        scale = layout.get_dpi_scale()

        if is_active:
            layout.set_next_window_pos((vp_x, vp_y))
            layout.set_next_window_size((vp_w, vp_h))
            if layout.begin_window("##ImportBackdrop", BACKDROP_FLAGS):
                layout.draw_window_rect_filled(vp_x, vp_y, vp_x + vp_w, vp_y + vp_h, BACKDROP_COLOR)
            layout.end_window()

        overlay_width = OVERLAY_WIDTH * scale
        btn_width = BUTTON_WIDTH * scale
        btn_height = BUTTON_HEIGHT * scale

        layout.set_next_window_pos((vp_x + (vp_w - overlay_width) * 0.5, vp_y + vp_h * VERTICAL_OFFSET))
        layout.set_next_window_size((overlay_width, 0))
        layout.push_modal_style()

        if layout.begin_window("##ImportProgressWin", DIALOG_FLAGS):
            self._draw_content(layout, state, is_active, show_completion, overlay_width, btn_width, btn_height, scale)

        layout.end_window()
        layout.pop_modal_style()

    def _draw_content(self, layout, state, is_active, show_completion, overlay_width, btn_width, btn_height, scale):
        dataset_type = state.get("dataset_type", "dataset")
        path_str = state.get("path", "")
        success = state.get("success", False)
        error = state.get("error", "")
        num_images = state.get("num_images", 0)
        num_points = state.get("num_points", 0)

        if show_completion and not is_active:
            if success:
                layout.text_colored(lf.ui.tr("progress.import_complete_title"), (0.4, 0.9, 0.4, 1.0))
            else:
                layout.text_colored(lf.ui.tr("progress.import_failed_title"), (1.0, 0.4, 0.4, 1.0))
        else:
            dtype = dataset_type if dataset_type else "dataset"
            layout.label(lf.ui.tr("progress.importing").replace("%s", dtype))

        layout.spacing()
        if path_str:
            layout.text_colored(f"Path: {path_str}", (0.7, 0.7, 0.7, 1.0))
        layout.spacing()

        progress = state.get("progress", 0.0)
        layout.progress_bar(progress, "", -1)

        if is_active:
            layout.label(f"{progress * 100:.0f}%")
            layout.same_line()
            layout.label(state.get("stage", ""))

        if show_completion and (num_images > 0 or num_points > 0):
            layout.spacing()
            layout.text_colored(f"{num_images} images, {num_points} points", (0.5, 0.8, 0.5, 1.0))

        if error:
            layout.spacing()
            layout.text_colored(error, (1.0, 0.4, 0.4, 1.0))

        if show_completion and not is_active:
            layout.spacing()
            layout.set_cursor_pos_x((overlay_width - btn_width) * 0.5 - 20 * scale)
            if layout.button_styled(lf.ui.tr("common.ok"), "primary", (btn_width, btn_height)):
                lf.ui.dismiss_import()
