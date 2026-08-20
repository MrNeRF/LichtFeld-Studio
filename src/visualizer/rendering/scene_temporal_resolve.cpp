/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/scene_temporal_resolve.hpp"

#include <algorithm>
#include <cmath>

namespace lfs::vis {
    namespace {
        bool finite(const glm::vec2 value) noexcept {
            return std::isfinite(value.x) && std::isfinite(value.y);
        }

        bool finite(const glm::vec3 value) noexcept {
            return std::isfinite(value.x) && std::isfinite(value.y) &&
                   std::isfinite(value.z);
        }

        bool finite(const glm::vec4 value) noexcept {
            return finite(glm::vec3(value)) && std::isfinite(value.w);
        }
    } // namespace

    SceneTemporalResolveSettings sceneTemporalQualitySettings(
        const SceneTemporalQuality quality) noexcept {
        switch (quality) {
        case SceneTemporalQuality::Performance:
            return {
                .history_weight = 0.75f,
                .depth_relative_threshold = 0.02f,
                .depth_absolute_threshold = 2e-4f,
                .motion_rejection_pixels = 96.0f,
            };
        case SceneTemporalQuality::Quality:
            return {
                .history_weight = 0.95f,
                .depth_relative_threshold = 0.005f,
                .depth_absolute_threshold = 5e-5f,
                .motion_rejection_pixels = 192.0f,
            };
        case SceneTemporalQuality::Balanced:
        default:
            return {};
        }
    }

    SceneTemporalResolveResult resolveSceneTemporalSample(
        const SceneTemporalResolveSample& sample,
        const SceneTemporalResolveSettings& settings) noexcept {
        SceneTemporalResolveResult result{.color = sample.current};
        if (!finite(sample.current) || !finite(sample.current_pixel_center) ||
            sample.motion_extent.x <= 0 || sample.motion_extent.y <= 0 ||
            sample.output_extent.x <= 0 || sample.output_extent.y <= 0) {
            result.color = {};
            result.rejection = SceneHistoryRejection::InvalidCurrent;
            return result;
        }
        if (!sample.history_valid) {
            result.rejection = SceneHistoryRejection::NoHistory;
            return result;
        }

        const float motion_limit = std::max(0.0f, settings.motion_rejection_pixels);
        if (!finite(sample.current_to_previous_pixels)) {
            result.rejection = SceneHistoryRejection::InvalidMotion;
            return result;
        }
        const float motion_length = glm::length(sample.current_to_previous_pixels);
        if (!std::isfinite(motion_length) || motion_length > motion_limit) {
            result.rejection = SceneHistoryRejection::InvalidMotion;
            return result;
        }

        const glm::vec2 current_uv = sample.current_pixel_center /
                                     glm::vec2(sample.output_extent);
        result.previous_uv = current_uv + sample.current_to_previous_pixels /
                                              glm::vec2(sample.motion_extent);
        const glm::vec2 minimum_uv = 0.5f / glm::vec2(sample.output_extent);
        const glm::vec2 maximum_uv = 1.0f - minimum_uv;
        if (!finite(result.previous_uv) || result.previous_uv.x < minimum_uv.x ||
            result.previous_uv.y < minimum_uv.y || result.previous_uv.x > maximum_uv.x ||
            result.previous_uv.y > maximum_uv.y) {
            result.rejection = SceneHistoryRejection::OutsideHistory;
            return result;
        }

        if (sample.depth_available) {
            if (!std::isfinite(sample.current_linear_depth) ||
                !std::isfinite(sample.history_linear_depth) ||
                sample.current_linear_depth <= 0.0f || sample.history_linear_depth <= 0.0f) {
                result.rejection = SceneHistoryRejection::Disocclusion;
                return result;
            }
            const float relative = std::max(0.0f, settings.depth_relative_threshold);
            const float absolute = std::max(0.0f, settings.depth_absolute_threshold);
            const float threshold = std::max(absolute,
                                             relative * sample.current_linear_depth);
            if (std::abs(sample.current_linear_depth - sample.history_linear_depth) > threshold) {
                result.rejection = SceneHistoryRejection::Disocclusion;
                return result;
            }
        }

        if (!finite(sample.history) || !finite(sample.neighborhood_min) ||
            !finite(sample.neighborhood_max)) {
            result.rejection = SceneHistoryRejection::InvalidHistory;
            return result;
        }
        const glm::vec3 lower = glm::min(sample.neighborhood_min, sample.neighborhood_max);
        const glm::vec3 upper = glm::max(sample.neighborhood_min, sample.neighborhood_max);
        const glm::vec3 clamped_history = glm::clamp(glm::vec3(sample.history), lower, upper);
        const float configured_weight = std::clamp(settings.history_weight, 0.0f, 1.0f);
        const float motion_confidence = motion_limit > 0.0f
                                            ? std::clamp(1.0f - motion_length / motion_limit,
                                                         0.0f,
                                                         1.0f)
                                            : 1.0f;
        result.effective_history_weight = configured_weight * motion_confidence;
        result.color = glm::vec4(glm::mix(glm::vec3(sample.current),
                                          clamped_history,
                                          result.effective_history_weight),
                                 sample.current.a);
        return result;
    }

} // namespace lfs::vis
