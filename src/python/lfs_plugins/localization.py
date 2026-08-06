# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Localized count helpers and the central extension point for grammar rules.

Locale JSON files keep their text declarative as ``<base>.one``, ``<base>.few``,
and ``<base>.other``. This module selects the form needed by the active language.
Polish currently uses its three shipped forms; other shipped languages use the
``one``/``other`` subset. Add future language-specific grammar here and cover its
boundary values in ``test_localization_contracts.py`` before using it in UI code.
"""


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
    return safe_format(lf.ui.tr(f"{key}.{form}"), count=count)


def safe_format(text: str, *args: object, **values: object) -> str:
    """Format translator-controlled text without allowing malformed braces to escape."""
    try:
        return text.format(*args, **values)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return text
