/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <chrono>

namespace lfs::vis::gui::vram_hud_sample_policy {

    inline constexpr auto kSampleInterval = std::chrono::milliseconds(250);

    [[nodiscard]] inline bool shouldPublish(bool already_published,
                                            bool expanded_changed,
                                            std::chrono::steady_clock::time_point now,
                                            std::chrono::steady_clock::time_point last_sample) noexcept {
        return !already_published || expanded_changed || now - last_sample >= kSampleInterval;
    }

} // namespace lfs::vis::gui::vram_hud_sample_policy
