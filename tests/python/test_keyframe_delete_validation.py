# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""First-key deletion is a valid command; missing keys remain harmless."""


def test_delete_first_keyframe_is_a_valid_command(lf):
    # No scene is required to validate the Python command boundary. Native
    # Sequencer and MCP tests verify real deletion and undo on populated paths.
    for _ in range(2):
        lf.ui.delete_keyframe(0)
