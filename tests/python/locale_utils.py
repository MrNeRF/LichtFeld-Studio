# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers for asserting that UI code requests translation keys that exist.

Panels resolve labels through ``lf.ui.tr(key)``. Tests stub ``tr`` as the
identity function, so a typo'd or removed key is invisible to them even though
it renders the raw key in the shipped UI. Checking requested keys against the
English locale turns that silent failure into a test failure.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

LOCALE_EN = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "visualizer"
    / "gui"
    / "resources"
    / "locales"
    / "en.json"
)


@lru_cache(maxsize=1)
def _locale_data() -> dict:
    return json.loads(LOCALE_EN.read_text(encoding="utf-8"))


def locale_key_exists(key: str) -> bool:
    """Whether a dotted translation key resolves in the English locale.

    The locale file mixes nested sections with literal dotted leaf names, for
    example ``{"export": {"format.ply_standard": "..."}}``, so every split
    point has to be tried rather than assuming a single shape.
    """

    def resolve(node, remainder: str) -> bool:
        if not isinstance(node, dict):
            return False
        if remainder in node:
            return True
        head, sep, tail = remainder.partition(".")
        while sep:
            if head in node and resolve(node[head], tail):
                return True
            next_head, sep, tail = tail.partition(".")
            head = f"{head}.{next_head}"
        return False

    return resolve(_locale_data(), key)


def missing_locale_keys(keys) -> list[str]:
    """The subset of ``keys`` that do not resolve, preserving order."""
    seen: set[str] = set()
    missing: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        if not locale_key_exists(key):
            missing.append(key)
    return missing
