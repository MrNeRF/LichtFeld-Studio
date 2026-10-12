/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"

#include <cstdint>
#include <functional>

namespace lfs::vis::detail {

    inline bool dispatch_camera_training_notice(const std::uint32_t mutation_flags,
                                                const bool training_active,
                                                const std::function<void()>& post_notice) {
        constexpr auto camera_training_changed =
            static_cast<std::uint32_t>(core::Scene::MutationType::CAMERA_TRAINING_CHANGED);
        if (!training_active || !(mutation_flags & camera_training_changed) || !post_notice)
            return false;
        post_notice();
        return true;
    }

} // namespace lfs::vis::detail
