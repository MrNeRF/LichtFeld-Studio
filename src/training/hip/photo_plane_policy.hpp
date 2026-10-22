/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>
#include <climits>
#include <string_view>

namespace lfs::training::hip {
// Conservative admission for the measured Float32 fused case. Byte targets,
// decoupled/masked loss, smaller images and unvalidated devices keep the
// canonical backward. Use wide arithmetic before forming any dispatch grid.
inline bool photo_plane_backward_enabled(unsigned path, bool byte_target,
                                         int n, int c, int h, int w,
                                         std::string_view architecture,
                                         bool disabled) {
    return !disabled && path==2 && !byte_target && architecture=="gfx1150" &&
           n>0 && c>0 && h>0 && w>0 && w<=INT_MAX-31 &&
           uint64_t(h)*uint64_t(w)>=1024*1024 &&
           uint64_t(n)*uint64_t(c)<=65535 &&
           (uint64_t(h)+7)/8<=65535;
}
} // namespace lfs::training::hip
