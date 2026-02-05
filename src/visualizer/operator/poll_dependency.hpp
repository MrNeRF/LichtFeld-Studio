/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace lfs::vis::op {

    enum class PollDependency : uint8_t {
        NONE = 0,
        SELECTION = 1 << 0,
        TRAINING = 1 << 1,
        SCENE = 1 << 2,
        ALL = SELECTION | TRAINING | SCENE
    };

    constexpr PollDependency operator|(PollDependency a, PollDependency b) {
        return static_cast<PollDependency>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
    }

    constexpr PollDependency operator&(PollDependency a, PollDependency b) {
        return static_cast<PollDependency>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
    }

} // namespace lfs::vis::op
