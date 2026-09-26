# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager shortcut adapter for the native keymap registry.

Older native modules have no gallery Action entries. Their scoped defaults remain
usable until rebuilt; a conflicting native binding takes precedence.
"""
from .rml_keys import KI_RETURN, KI_C

# (native action, command, RML key, native key, modifiers, display)
SHORTCUTS = (
    ("ASSET_GALLERY_PRIMARY", "primary", KI_RETURN, 257, 2, "Ctrl+Enter"),
    ("ASSET_GALLERY_COPY_LINK", "copy", KI_C, 67, 3, "Ctrl+Shift+C"),
    ("ASSET_REFRESH", "refresh_scope", 111, 294, 0, "F5"),
)


def shortcut_command(keymap, key, ctrl=False, shift=False, alt=False, meta=False):
    modifiers = (2 if ctrl else 0) | (1 if shift else 0) | (4 if alt else 0) | (8 if meta else 0)
    native_key = (257 if key == KI_RETURN else 290 + key - 107 if 107 <= key <= 130
                  else 65 + key - 12 if 12 <= key <= 37 else 48 + key - 2 if 2 <= key <= 11 else None)
    if keymap is not None and native_key is not None:
        if keymap.is_capturing():
            return None
        action = keymap.get_action_for_key(keymap.ToolMode.GLOBAL, native_key, modifiers)
        for name, command, *_ in SHORTCUTS:
            registered = getattr(keymap.Action, name, None)
            if registered is not None and action == registered:
                return command
        if action != keymap.Action.NONE:
            return None
        # A rebuilt registry owns rebindings, including explicitly unbound keys.
        if all(hasattr(keymap.Action, item[0]) for item in SHORTCUTS):
            return None
    return next((command for _, command, rml_key, _, mods, _ in SHORTCUTS
                 if key == rml_key and modifiers == mods), None)
