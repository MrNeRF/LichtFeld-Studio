# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Localized text for modules that must also work without the app (tests, scripts).

Mirrors the plugin installer: look the key up through lf.ui.tr when lichtfeld is
importable, otherwise fall back to the English text. The key is the contract
that tests/python/test_localization_contracts.py checks against every locale.
"""
from __future__ import annotations

from ..localization import safe_format


def _tr(key: str, fallback: str, **values: object) -> str:
    try:
        import lichtfeld as lf

        text = lf.ui.tr(key)
        if text and text != key:
            return safe_format(text, **values)
    except Exception:  # noqa: BLE001 - no app, or the UI module is unavailable
        pass
    return safe_format(fallback, **values)
