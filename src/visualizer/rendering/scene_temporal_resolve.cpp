/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/scene_temporal_resolve.hpp"

#include <algorithm>
#include <cmath>

namespace lfs::vis {
    namespace {
        bool finite(const glm::vec2 value) {
            return std::isfinite(value.x) && std::isfinite(value.y);
        }

        bool finite(const glm::vec4 value) {
            return std::isfinite(value.x) && std::isfinite(value.y) &&
                   std::isfinite(value.z) && std::isfinite(value.w);
        }
    } // namespace

    SceneTemporalResolveSettings sceneTemporalQualitySettings(
        const SceneTemporalQuality quality) {
        switch (quality) {
        case SceneTemporalQuality::Performance:
            return {.history_weight = 0.75f,
                    .depth_threshold = 0.02f,
                    .motion_rejection_pixels = 96.0f};
        case SceneTemporalQuality::Quality:
            return {.history_weight = 0.95f,
                    .depth_threshold = 0.005f,
                    .motion_rejection_pixels = 192.0f};
        case SceneTemporalQuality::Balanced:
        default:
            return {};
        }
    }

    SceneTemporalResolveResult resolveSceneTemporalSample(
        const SceneTemporalResolveSample& sample,
        const SceneTemporalResolveSettings& settings) {
        SceneTemporalResolveResult result{.color = sample.current};
        if (!finite(sample.current)) {
            result.color = {};
            result.rejection = SceneHistoryRejection::InvalidColor;
            return result;
        }
        if (!sample.history_valid) {
            result.rejection = SceneHistoryRejection::NoHistory;
            return result;
        }
        if (!finite(sample.motion_pixels) || sample.output_extent.x <= 0 ||
            sample.output_extent.y <= 0 ||
            glm::length(sample.motion_pixels) >
                std::max(0.0f, settings.motion_rejection_pixels)) {
            result.rejection = SceneHistoryRejection::InvalidMotion;
            return result;
        }

        const glm::vec2 previous_pixel = sample.current_pixel + sample.motion_pixels;
        result.previous_uv = (previous_pixel + glm::vec2(0.5f)) /
                             glm::vec2(sample.output_extent);
        if (!finite(result.previous_uv) || result.previous_uv.x < 0.0f ||
            result.previous_uv.y < 0.0f || result.previous_uv.x > 1.0f ||
            result.previous_uv.y > 1.0f) {
            result.rejection = SceneHistoryRejection::OutsideHistory;
            return result;
        }
        if (sample.depth_available &&
            (!std::isfinite(sample.current_depth) || !std::isfinite(sample.history_depth) ||
             std::abs(sample.current_depth - sample.history_depth) >
                 std::max(0.0f, settings.depth_threshold))) {
            result.rejection = SceneHistoryRejection::Disocclusion;
            return result;
        }
        if (!finite(sample.history) || !finite(sample.neighborhood_min) ||
            !finite(sample.neighborhood_max)) {
            result.rejection = SceneHistoryRejection::InvalidColor;
            return result;
        }

        const glm::vec4 lower = glm::min(sample.neighborhood_min, sample.neighborhood_max);
        const glm::vec4 upper = glm::max(sample.neighborhood_min, sample.neighborhood_max);
        const glm::vec4 clamped_history = glm::clamp(sample.history, lower, upper);
        const float configured_weight = std::clamp(settings.history_weight, 0.0f, 1.0f);
        const float motion_confidence = settings.motion_rejection_pixels > 0.0f
                                            ? std::clamp(1.0f - glm::length(sample.motion_pixels) /
                                                                    settings.motion_rejection_pixels,
                                                         0.0f,
                                                         1.0f)
                                            : 0.0f;
        result.effective_history_weight = configured_weight * motion_confidence;
        result.color = glm::mix(sample.current, clamped_history,
                                result.effective_history_weight);
        return result;
    }

} // namespace lfs::vis
