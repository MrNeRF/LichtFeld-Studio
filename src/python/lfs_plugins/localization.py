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


def localized_count(
    key: str,
    count: int,
    *,
    plural_count: int | None = None,
    **values: object,
) -> str:
    """Format a count-sensitive localization key using the active language."""
    import lichtfeld as lf

    form = plural_form(
        lf.ui.get_current_language(), count if plural_count is None else plural_count
    )
    return safe_format(lf.ui.tr(f"{key}.{form}"), count=count, **values)


def _grouped(value: object) -> object:
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else value


def safe_format(text: str, *args: object, **values: object) -> str:
    """Format translator-controlled text without allowing malformed braces to escape.

    Integers are shown with digit groups, 1234 as 1,234.
    """
    try:
        return text.format(*(_grouped(arg) for arg in args), **{key: _grouped(value) for key, value in values.items()})
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return text
