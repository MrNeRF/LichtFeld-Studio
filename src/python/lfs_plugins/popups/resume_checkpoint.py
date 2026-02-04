# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resume checkpoint popup for checkpoint import configuration."""

from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckpointLoadParams:
    """Parameters for loading a checkpoint."""

    checkpoint_path: Path
    dataset_path: Path
    output_path: Path


class ResumeCheckpointPopup:
    """Popup for configuring checkpoint resume."""

    POPUP_WIDTH = 580
    INPUT_WIDTH = 400
    BUTTON_WIDTH = 100
    BUTTON_SPACING = 8

    def __init__(self):
        self._open = False
        self._pending_open = False
        self._checkpoint_path = ""
        self._header = None
        self._stored_dataset_path = ""
        self._dataset_path = ""
        self._output_path = ""
        self._dataset_valid = False
        self._on_confirm: Optional[Callable[[CheckpointLoadParams], None]] = None

    @property
    def is_open(self) -> bool:
        return self._open

    def show(self, checkpoint_path: str, on_confirm: Optional[Callable[[CheckpointLoadParams], None]] = None):
        """Show the popup with the given checkpoint path."""
        import lichtfeld as lf

        self._checkpoint_path = checkpoint_path
        self._header = lf.read_checkpoint_header(checkpoint_path)
        if not self._header:
            return

        params = lf.read_checkpoint_params(checkpoint_path)
        if not params:
            return

        self._stored_dataset_path = params.dataset_path
        self._dataset_path = self._stored_dataset_path
        self._output_path = params.output_path

        # Check if stored dataset path is valid
        self._dataset_valid = self._validate_dataset(self._dataset_path)

        self._on_confirm = on_confirm
        self._pending_open = True

    def _validate_dataset(self, path: str) -> bool:
        """Check if the path is a valid dataset directory."""
        from pathlib import Path

        p = Path(path)
        return p.is_dir()

    def draw(self, layout):
        """Draw the popup. Called every frame."""
        import lichtfeld as lf
        tr = lf.ui.tr

        if not self._pending_open and not self._open:
            return

        scale = layout.get_dpi_scale()

        if self._pending_open:
            layout.set_next_window_pos_center()
            layout.set_next_window_size((self.POPUP_WIDTH * scale, 0))
            layout.open_popup(tr("resume_checkpoint_popup.title"))
            self._pending_open = False
            self._open = True

        layout.push_modal_style()

        if layout.begin_popup_modal(tr("resume_checkpoint_popup.title")):
            header = self._header

            # Header
            layout.text_colored(tr("resume_checkpoint_popup.checkpoint"), (0.3, 0.7, 1.0, 1.0))
            layout.same_line()
            layout.text_colored("|", (0.5, 0.5, 0.5, 1.0))
            layout.same_line()
            layout.label(tr("resume_checkpoint_popup.configure_paths"))

            layout.spacing()
            layout.separator()
            layout.spacing()

            # Checkpoint info
            checkpoint_name = Path(self._checkpoint_path).name
            layout.text_colored(tr("resume_checkpoint_popup.file"), (0.6, 0.6, 0.6, 1.0))
            layout.same_line()
            layout.label(checkpoint_name)
            layout.same_line()
            layout.text_colored(
                f"(iter {header.iteration}, {header.num_gaussians} gaussians)", (0.6, 0.6, 0.6, 1.0)
            )

            layout.spacing()

            # Stored path (read-only)
            layout.text_colored(tr("resume_checkpoint_popup.stored_path"), (0.6, 0.6, 0.6, 1.0))
            layout.same_line()
            if not self._dataset_valid and self._dataset_path == self._stored_dataset_path:
                layout.text_colored(self._stored_dataset_path, (0.9, 0.3, 0.3, 1.0))
                layout.same_line()
                layout.text_colored(tr("resume_checkpoint_popup.not_found"), (0.9, 0.3, 0.3, 1.0))
            else:
                layout.label(self._stored_dataset_path)

            layout.spacing()
            layout.separator()
            layout.spacing()

            # Dataset path (editable)
            layout.text_colored(tr("resume_checkpoint_popup.dataset_path"), (0.6, 0.6, 0.6, 1.0))
            layout.set_next_item_width(self.INPUT_WIDTH * scale)
            changed, self._dataset_path = layout.input_text("##dataset_path", self._dataset_path)
            if changed:
                self._dataset_valid = self._validate_dataset(self._dataset_path)

            layout.same_line()
            if layout.button(tr("common.browse") + "##dataset"):
                path = lf.ui.open_dataset_folder_dialog()
                if path:
                    self._dataset_path = path
                    self._dataset_valid = self._validate_dataset(path)

            layout.same_line()
            if self._dataset_valid:
                layout.text_colored("[OK]", (0.3, 0.9, 0.3, 1.0))
            else:
                layout.text_colored(tr("resume_checkpoint_popup.invalid"), (0.9, 0.3, 0.3, 1.0))

            layout.spacing()

            # Output path
            layout.text_colored(tr("resume_checkpoint_popup.output_path"), (0.6, 0.6, 0.6, 1.0))
            layout.set_next_item_width(self.INPUT_WIDTH * scale)
            _, self._output_path = layout.input_text("##output_path", self._output_path)
            layout.same_line()
            if layout.button(tr("common.browse") + "##output"):
                path = lf.ui.open_dataset_folder_dialog()
                if path:
                    self._output_path = path

            layout.spacing()
            layout.text_wrapped(tr("resume_checkpoint_popup.help_text"))
            layout.spacing()
            layout.separator()
            layout.spacing()

            avail_width = layout.get_content_region_avail()[0]
            btn_width = self.BUTTON_WIDTH * scale
            btn_spacing = self.BUTTON_SPACING * scale
            total_width = btn_width * 2 + btn_spacing
            layout.set_cursor_pos_x(layout.get_cursor_pos()[0] + avail_width - total_width)

            if layout.button_styled(tr("common.cancel"), "secondary", (btn_width, 0)) or lf.ui.is_key_pressed(lf.ui.Key.ESCAPE):
                self._open = False
                layout.close_current_popup()

            layout.same_line(0, btn_spacing)

            if not self._dataset_valid:
                layout.begin_disabled()

            if layout.button_styled(tr("common.load"), "success", (btn_width, 0)) or (
                self._dataset_valid and lf.ui.is_key_pressed(lf.ui.Key.ENTER)
            ):
                self._open = False
                layout.close_current_popup()
                if self._on_confirm:
                    params = CheckpointLoadParams(
                        checkpoint_path=Path(self._checkpoint_path),
                        dataset_path=Path(self._dataset_path),
                        output_path=Path(self._output_path),
                    )
                    self._on_confirm(params)

            if not self._dataset_valid:
                layout.end_disabled()

            layout.end_popup_modal()
        else:
            self._open = False

        layout.pop_modal_style()
