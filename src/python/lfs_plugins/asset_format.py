# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared adaptive byte display for local assets and Gallery transfers."""


def format_size(value):
    import lichtfeld as lf
    try:
        size = max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        size = 0
    for divisor, unit in ((1000**3, "gb"), (1000**2, "mb"), (1000, "kb"), (1, "b")):
        if size >= divisor or divisor == 1:
            amount = size / divisor
            # Round before selecting precision, so 9.99 KB becomes 10 KB.
            digits = 1 if round(amount, 1) < 10 else 0
            return f"{amount:.{digits}f} {lf.ui.tr('projects.unit.' + unit)}"
