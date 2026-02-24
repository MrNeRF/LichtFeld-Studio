/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::vis::input {

    // Application-level key codes. Numeric values match GLFW for backward
    // compatibility with saved input profiles (JSON with integer key codes).
    constexpr int KEY_UNKNOWN = -1;
    constexpr int KEY_SPACE = 32;
    constexpr int KEY_APOSTROPHE = 39;
    constexpr int KEY_COMMA = 44;
    constexpr int KEY_MINUS = 45;
    constexpr int KEY_PERIOD = 46;
    constexpr int KEY_SLASH = 47;
    constexpr int KEY_0 = 48;
    constexpr int KEY_1 = 49;
    constexpr int KEY_2 = 50;
    constexpr int KEY_3 = 51;
    constexpr int KEY_4 = 52;
    constexpr int KEY_5 = 53;
    constexpr int KEY_6 = 54;
    constexpr int KEY_7 = 55;
    constexpr int KEY_8 = 56;
    constexpr int KEY_9 = 57;
    constexpr int KEY_SEMICOLON = 59;
    constexpr int KEY_EQUAL = 61;
    constexpr int KEY_A = 65;
    constexpr int KEY_B = 66;
    constexpr int KEY_C = 67;
    constexpr int KEY_D = 68;
    constexpr int KEY_E = 69;
    constexpr int KEY_F = 70;
    constexpr int KEY_G = 71;
    constexpr int KEY_H = 72;
    constexpr int KEY_I = 73;
    constexpr int KEY_J = 74;
    constexpr int KEY_K = 75;
    constexpr int KEY_L = 76;
    constexpr int KEY_M = 77;
    constexpr int KEY_N = 78;
    constexpr int KEY_O = 79;
    constexpr int KEY_P = 80;
    constexpr int KEY_Q = 81;
    constexpr int KEY_R = 82;
    constexpr int KEY_S = 83;
    constexpr int KEY_T = 84;
    constexpr int KEY_U = 85;
    constexpr int KEY_V = 86;
    constexpr int KEY_W = 87;
    constexpr int KEY_X = 88;
    constexpr int KEY_Y = 89;
    constexpr int KEY_Z = 90;
    constexpr int KEY_LEFT_BRACKET = 91;
    constexpr int KEY_BACKSLASH = 92;
    constexpr int KEY_RIGHT_BRACKET = 93;
    constexpr int KEY_GRAVE_ACCENT = 96;
    constexpr int KEY_ESCAPE = 256;
    constexpr int KEY_ENTER = 257;
    constexpr int KEY_TAB = 258;
    constexpr int KEY_BACKSPACE = 259;
    constexpr int KEY_INSERT = 260;
    constexpr int KEY_DELETE = 261;
    constexpr int KEY_RIGHT = 262;
    constexpr int KEY_LEFT = 263;
    constexpr int KEY_DOWN = 264;
    constexpr int KEY_UP = 265;
    constexpr int KEY_PAGE_UP = 266;
    constexpr int KEY_PAGE_DOWN = 267;
    constexpr int KEY_HOME = 268;
    constexpr int KEY_END = 269;
    constexpr int KEY_CAPS_LOCK = 280;
    constexpr int KEY_SCROLL_LOCK = 281;
    constexpr int KEY_NUM_LOCK = 282;
    constexpr int KEY_PRINT_SCREEN = 283;
    constexpr int KEY_PAUSE = 284;
    constexpr int KEY_F1 = 290;
    constexpr int KEY_F2 = 291;
    constexpr int KEY_F3 = 292;
    constexpr int KEY_F4 = 293;
    constexpr int KEY_F5 = 294;
    constexpr int KEY_F6 = 295;
    constexpr int KEY_F7 = 296;
    constexpr int KEY_F8 = 297;
    constexpr int KEY_F9 = 298;
    constexpr int KEY_F10 = 299;
    constexpr int KEY_F11 = 300;
    constexpr int KEY_F12 = 301;
    constexpr int KEY_KP_0 = 320;
    constexpr int KEY_KP_1 = 321;
    constexpr int KEY_KP_2 = 322;
    constexpr int KEY_KP_3 = 323;
    constexpr int KEY_KP_4 = 324;
    constexpr int KEY_KP_5 = 325;
    constexpr int KEY_KP_6 = 326;
    constexpr int KEY_KP_7 = 327;
    constexpr int KEY_KP_8 = 328;
    constexpr int KEY_KP_9 = 329;
    constexpr int KEY_KP_DECIMAL = 330;
    constexpr int KEY_KP_DIVIDE = 331;
    constexpr int KEY_KP_MULTIPLY = 332;
    constexpr int KEY_KP_SUBTRACT = 333;
    constexpr int KEY_KP_ADD = 334;
    constexpr int KEY_KP_ENTER = 335;
    constexpr int KEY_KP_EQUAL = 336;
    constexpr int KEY_LEFT_SHIFT = 340;
    constexpr int KEY_LEFT_CONTROL = 341;
    constexpr int KEY_LEFT_ALT = 342;
    constexpr int KEY_LEFT_SUPER = 343;
    constexpr int KEY_RIGHT_SHIFT = 344;
    constexpr int KEY_RIGHT_CONTROL = 345;
    constexpr int KEY_RIGHT_ALT = 346;
    constexpr int KEY_RIGHT_SUPER = 347;
    constexpr int KEY_MENU = 348;

    constexpr int KEYMOD_NONE = 0;
    constexpr int KEYMOD_SHIFT = 0x0001;
    constexpr int KEYMOD_CTRL = 0x0002;
    constexpr int KEYMOD_ALT = 0x0004;
    constexpr int KEYMOD_SUPER = 0x0008;

    enum class AppMouseButton : int {
        LEFT = 0,
        RIGHT = 1,
        MIDDLE = 2,
    };

    constexpr int ACTION_RELEASE = 0;
    constexpr int ACTION_PRESS = 1;
    constexpr int ACTION_REPEAT = 2;

} // namespace lfs::vis::input
