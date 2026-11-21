/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include <algorithm>

#include <cstdint>
#include <functional>

namespace lfs::vis::detail {

    inline bool dispatch_unused_camera_mask_notice(const core::Scene& scene,
                                                   const core::param::MaskMode mode,
                                                   const std::function<void()>& post_notice) {
        if (mode != core::param::MaskMode::None || !post_notice)
            return false;
        if (!std::ranges::any_of(scene.getAllCameras(), [](const auto& camera) {
                return camera && camera->mask_asset_uuid().has_value();
            }))
            return false;
        post_notice();
        return true;
    }

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
