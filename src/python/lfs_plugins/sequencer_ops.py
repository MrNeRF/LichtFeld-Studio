# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sequencer operators for keyframe manipulation."""

import lichtfeld as lf

from .layouts.menus import keymap_shortcut
from .types import Operator


class AddKeyframeOperator(Operator):
    """Add a keyframe at the current camera position."""

    label = "sequencer.add_keyframe_here"
    shortcut = ""

    def execute(self, context):
        lf.ui.add_keyframe()
        return {"FINISHED"}


class UpdateKeyframeOperator(Operator):
    """Update selected keyframe to current camera position."""

    label = "sequencer.update_to_current_view"
    shortcut = ""

    def execute(self, context):
        lf.ui.update_keyframe()
        return {"FINISHED"}


class PlayPauseOperator(Operator):
    """Toggle sequencer playback."""

    # Operator metadata is exposed to non-GUI callers verbatim. Keep this as a
    # stable product label until the operator registry gains localized labels.
    label = "Play/Pause"
    shortcut = ""

    def execute(self, context):
        lf.ui.play_pause()
        return {"FINISHED"}


def register():
    AddKeyframeOperator.shortcut = keymap_shortcut(lf.keymap.Action.SEQUENCER_ADD_KEYFRAME)
    UpdateKeyframeOperator.shortcut = keymap_shortcut(lf.keymap.Action.SEQUENCER_UPDATE_KEYFRAME)
    PlayPauseOperator.shortcut = keymap_shortcut(lf.keymap.Action.SEQUENCER_PLAY_PAUSE)
    lf.register_class(AddKeyframeOperator)
    lf.register_class(UpdateKeyframeOperator)
    lf.register_class(PlayPauseOperator)


def unregister():
    lf.unregister_class(AddKeyframeOperator)
    lf.unregister_class(UpdateKeyframeOperator)
    lf.unregister_class(PlayPauseOperator)
