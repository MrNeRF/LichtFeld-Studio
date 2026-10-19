# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Input validation for protected sequencer key deletion."""

import pytest


def test_delete_protected_first_keyframe_raises(lf):
    for _ in range(2):
        with pytest.raises(ValueError, match="first keyframe cannot be deleted"):
            lf.ui.delete_keyframe(0)
