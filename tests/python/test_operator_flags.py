# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
from pathlib import Path


def test_operator_flags_or_can_be_assigned_to_descriptor():
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "build" / "src" / "python"))
    import lichtfeld as lf

    combined = lf.ops.OperatorFlags.UNDO | lf.ops.OperatorFlags.REGISTER
    descriptor = lf.ops.OperatorDescriptor()
    descriptor.flags = combined
    assert descriptor.flags == combined
