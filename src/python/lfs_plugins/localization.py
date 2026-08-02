# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small helpers for localized messages whose grammar depends on a count."""


def plural_form(language: str, count: int) -> str:
    """Return the plural form supported by the shipped locale catalogs."""
    if language == "pl":
        absolute_count = abs(count)
        if absolute_count == 1:
            return "one"
        if 2 <= absolute_count % 10 <= 4 and not 12 <= absolute_count % 100 <= 14:
            return "few"
        return "other"
    return "one" if abs(count) == 1 else "other"


def localized_count(key: str, count: int) -> str:
    """Format a count-sensitive localization key using the active language."""
    import lichtfeld as lf

    form = plural_form(lf.ui.get_current_language(), count)
    return lf.ui.tr(f"{key}.{form}").format(count=count)
