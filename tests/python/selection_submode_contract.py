# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Selection submode contract shared by the pytest suite and the embedded SelectionSubmodeTest.

The embedded interpreter has no pytest, so this module must not import it.
"""

SELECTION_MODES = ("centers", "rectangle", "polygon", "lasso", "rings", "color", "box", "sphere")


def check_selection_submode_follows_native_mode(lf):
    original = lf.ui.get_selection_submode()
    try:
        for expected, mode in enumerate(SELECTION_MODES):
            lf.ui.set_selection_mode(mode)
            assert lf.ui.get_selection_submode() == expected
            assert lf.ui.context().selection_submode == expected
    finally:
        lf.ui.set_selection_mode(SELECTION_MODES[original])
