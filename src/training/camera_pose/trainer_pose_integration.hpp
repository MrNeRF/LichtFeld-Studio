/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/parameters.hpp"
#include <string>

namespace lfs::training::camera_pose {
    // Empty means supported by the initial raw-RGB Trainer integration.
    [[nodiscard]] std::string trainer_pose_incompatibility(const lfs::core::param::OptimizationParameters& params);
} // namespace lfs::training::camera_pose
