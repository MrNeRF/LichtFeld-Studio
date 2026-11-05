/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>

namespace lfs::vis::gui {
    enum class PreviewTextureStatus : std::uint8_t {
        Unknown,
        Pending,
        Ready,
        Error
    };
} // namespace lfs::vis::gui
