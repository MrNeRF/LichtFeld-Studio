# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Marketplace defaults from the canonical application preferences store."""

import lichtfeld as lf

DEFAULTS = {"view": "list", "git_checkout": False}


def read_preferences():
    values = DEFAULTS.copy()
    getter = getattr(lf.ui, "get_marketplace_preferences", None)
    raw = getter() if getter else {}
    if isinstance(raw, dict):
        if raw.get("view") in ("card", "list"):
            values["view"] = raw["view"]
        if isinstance(raw.get("git_checkout"), bool):
            values["git_checkout"] = raw["git_checkout"]
    return values


def set_preference(key, value):
    if key == "view" and value in ("card", "list"):
        lf.ui.set_marketplace_view(value)
    elif key == "git_checkout" and isinstance(value, bool):
        lf.ui.set_marketplace_git_checkout(value)
    else:
        raise ValueError("Unsupported Marketplace preference")


def reset_preferences():
    lf.ui.reset_marketplace_preferences()
