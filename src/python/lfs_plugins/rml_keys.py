# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Centralized RmlUI key identifier constants (Rml::Input::KeyIdentifier)."""

KI_SPACE = 1
KI_A = 12
KI_C = 14
KI_1 = 3
KI_F = 17
KI_I = 20
KI_M = 24
KI_R = 29
KI_T = 31
KI_OEM_PLUS = 39
KI_OEM_MINUS = 41
KI_ADD = 63
KI_SUBTRACT = 65
KI_RETURN = 72
KI_TAB = 70
KI_ESCAPE = 81
KI_END = 88
KI_HOME = 89
KI_LEFT = 90
KI_UP = 91
KI_RIGHT = 92
KI_DOWN = 93
KI_DELETE = 99
KI_F2 = 108


def native_key(key):
    """Translate RmlUI key identifiers to the native keymap's AppKey codes."""
    if 2 <= key <= 11:
        return 48 + key - 2
    if 12 <= key <= 37:
        return 65 + key - 12
    if 51 <= key <= 60:
        return 320 + key - 51
    if 107 <= key <= 130:
        return 290 + key - 107
    return {
        1: 32, 38: 59, 39: 61, 40: 44, 41: 45, 42: 46, 43: 47, 44: 96,
        45: 91, 46: 92, 47: 93, 48: 39, 61: 335, 62: 332, 63: 334,
        65: 333, 66: 330, 67: 331, 68: 336, 69: 259, 70: 258, 72: 257,
        73: 284, 74: 280, 81: 256, 86: 266, 87: 267, 88: 269, 89: 268,
        90: 263, 91: 265, 92: 262, 93: 264, 97: 283, 98: 260, 99: 261,
        103: 348, 131: 282, 132: 281,
    }.get(key)
