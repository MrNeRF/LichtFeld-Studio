/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace lfs::vis {

    using DirtyMask = uint32_t;

    namespace DirtyFlag {
        constexpr DirtyMask CAMERA = 1 << 0;
        constexpr DirtyMask SPLATS = 1 << 1;
        constexpr DirtyMask MESH = 1 << 2;
        constexpr DirtyMask VIEWPORT = 1 << 3;
        constexpr DirtyMask OVERLAY = 1 << 4;
        constexpr DirtyMask PPISP = 1 << 5;
        constexpr DirtyMask SELECTION = 1 << 6;
        constexpr DirtyMask BACKGROUND = 1 << 7;
        constexpr DirtyMask SPLIT_VIEW = 1 << 8;
        constexpr DirtyMask LAST_ = SPLIT_VIEW;
        constexpr DirtyMask ALL = (LAST_ << 1) - 1;
        static_assert(ALL == 0x1FF, "Update LAST_ when adding new flags");
    } // namespace DirtyFlag

} // namespace lfs::vis
