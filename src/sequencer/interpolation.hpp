/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "keyframe.hpp"
#include <cstddef>
#include <span>
#include <vector>

namespace lfs::sequencer {

    inline constexpr size_t MAX_GENERATED_PATH_SAMPLES = 1'000'000;

    // Map t in [0,1] to eased t
    [[nodiscard]] float applyEasing(float t, EasingType easing);

    // Catmull-Rom spline: t in [0,1] interpolates between p1 and p2
    [[nodiscard]] glm::vec3 catmullRom(
        const glm::vec3& p0, const glm::vec3& p1,
        const glm::vec3& p2, const glm::vec3& p3,
        float t);

    // Interpolate camera state through keyframe spline
    [[nodiscard]] CameraState interpolateSpline(
        std::span<const Keyframe> keyframes,
        float time);

    // Generate path points for visualization
    [[nodiscard]] std::vector<glm::vec3> generatePathPoints(
        std::span<const Keyframe> keyframes,
        int samples_per_segment = 20);

} // namespace lfs::sequencer
