# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sequencer operators for keyframe manipulation."""

import lichtfeld as lf

from .types import Operator


def _shortcut(action, fallback):
    try:
        shortcut = lf.keymap.get_trigger_description(action, lf.keymap.ToolMode.GLOBAL)
        return "" if shortcut == "Unbound" else shortcut
    except (AttributeError, RuntimeError, TypeError):
        return fallback


class AddKeyframeOperator(Operator):
    """Add a keyframe at the current camera position."""

    label = "Add Keyframe Here"
    shortcut = ""

    def execute(self, context):
        lf.ui.add_keyframe()
        return {"FINISHED"}


class UpdateKeyframeOperator(Operator):
    """Update selected keyframe to current camera position."""

    label = "Update to Current View"
    shortcut = ""

    def execute(self, context):
        lf.ui.update_keyframe()
        return {"FINISHED"}


class PlayPauseOperator(Operator):
    """Toggle sequencer playback."""

    label = "Play/Pause"
    shortcut = ""

    def execute(self, context):
        lf.ui.play_pause()
        return {"FINISHED"}


def register():
    AddKeyframeOperator.shortcut = _shortcut(lf.keymap.Action.SEQUENCER_ADD_KEYFRAME, "K")
    UpdateKeyframeOperator.shortcut = _shortcut(lf.keymap.Action.SEQUENCER_UPDATE_KEYFRAME, "U")
    PlayPauseOperator.shortcut = _shortcut(lf.keymap.Action.SEQUENCER_PLAY_PAUSE, "Space")
    lf.register_class(AddKeyframeOperator)
    lf.register_class(UpdateKeyframeOperator)
    lf.register_class(PlayPauseOperator)


def unregister():
    lf.unregister_class(AddKeyframeOperator)
    lf.unregister_class(UpdateKeyframeOperator)
    lf.unregister_class(PlayPauseOperator)
