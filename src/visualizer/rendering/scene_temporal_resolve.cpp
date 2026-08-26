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

    glm::vec2 sceneTemporalJitterPixels(const glm::vec2 jitter_ndc,
                                        const glm::ivec2 render_extent,
                                        const bool flip_y) noexcept {
        if (!finite(jitter_ndc) || render_extent.x <= 0 || render_extent.y <= 0)
            return {};
        // Projection jitter is expressed in OpenGL NDC (+Y up); temporal
        // sampling addresses the current image from its top-left (+Y down).
        glm::vec2 result{
            jitter_ndc.x * 0.5f * static_cast<float>(render_extent.x),
            -jitter_ndc.y * 0.5f * static_cast<float>(render_extent.y)};
        if (flip_y)
            result.y = -result.y;
        return result;
    }

    float sceneTemporalHistoryWeight(const float configured_weight,
                                     const std::uint64_t accumulated_frames) noexcept {
        if (!std::isfinite(configured_weight) || accumulated_frames == 0)
            return 0.0f;
        const double frame_count = static_cast<double>(accumulated_frames);
        const float uniform_accumulation =
            static_cast<float>(frame_count / (frame_count + 1.0));
        return std::min(std::clamp(configured_weight, 0.0f, 1.0f), uniform_accumulation);
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
        if (!finite(sample.neighborhood_min) || !finite(sample.neighborhood_max) ||
            !finite(sample.neighborhood_cross_sum)) {
            result.rejection = SceneHistoryRejection::InvalidCurrent;
            return result;
        }
        const glm::vec3 lower = glm::min(sample.neighborhood_min, sample.neighborhood_max);
        const glm::vec3 upper = glm::max(sample.neighborhood_min, sample.neighborhood_max);
        const float current_sharpness = std::clamp(settings.current_sharpness, 0.0f, 0.25f);
        const glm::vec3 reconstructed_current = glm::clamp(
            glm::vec3(sample.current) * (1.0f + 4.0f * current_sharpness) -
                sample.neighborhood_cross_sum * current_sharpness,
            lower,
            upper);
        result.color = glm::vec4(reconstructed_current, sample.current.a);
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

        const glm::vec2 render_extent(sample.motion_extent);
        result.current_render_uv =
            sample.current_pixel_center / glm::vec2(sample.output_extent) +
            sample.current_jitter_pixels / render_extent;
        result.previous_uv =
            sample.current_pixel_center / glm::vec2(sample.output_extent) +
            sample.current_to_previous_pixels / render_extent;
        result.previous_render_uv =
            result.previous_uv + sample.previous_jitter_pixels / render_extent;
        const glm::vec2 minimum_uv = 0.5f / glm::vec2(sample.output_extent);
        const glm::vec2 maximum_uv = 1.0f - minimum_uv;
        if (!finite(result.previous_uv) || result.previous_uv.x < minimum_uv.x ||
            result.previous_uv.y < minimum_uv.y || result.previous_uv.x > maximum_uv.x ||
            result.previous_uv.y > maximum_uv.y) {
            result.rejection = SceneHistoryRejection::OutsideHistory;
            return result;
        }

        if (sample.depth_available) {
            const glm::vec2 render_minimum_uv = 0.5f / render_extent;
            const glm::vec2 render_maximum_uv = 1.0f - render_minimum_uv;
            if (result.previous_render_uv.x < render_minimum_uv.x ||
                result.previous_render_uv.y < render_minimum_uv.y ||
                result.previous_render_uv.x > render_maximum_uv.x ||
                result.previous_render_uv.y > render_maximum_uv.y) {
                result.rejection = SceneHistoryRejection::Disocclusion;
                return result;
            }
            if (!std::isfinite(sample.current_linear_depth) ||
                !std::isfinite(sample.history_linear_depth) ||
                !std::isfinite(sample.depth_far_plane) ||
                sample.current_linear_depth <= 0.0f || sample.history_linear_depth <= 0.0f ||
                sample.depth_far_plane <= 0.0f ||
                sample.current_linear_depth >= sample.depth_far_plane ||
                sample.history_linear_depth >= sample.depth_far_plane) {
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

        if (!finite(sample.history)) {
            result.rejection = SceneHistoryRejection::InvalidHistory;
            return result;
        }
        const glm::vec3 clamped_history = glm::clamp(glm::vec3(sample.history), lower, upper);
        const float configured_weight = std::clamp(settings.history_weight, 0.0f, 1.0f);
        const float geometric_motion_length = glm::length(sample.current_to_previous_pixels);
        const float confidence_span = std::max(0.0f, settings.motion_confidence_pixels);
        const float motion_confidence =
            motion_limit > 0.0f && confidence_span > 0.0f &&
                    std::isfinite(geometric_motion_length)
                ? std::clamp(1.0f - geometric_motion_length / confidence_span, 0.0f, 1.0f)
                : 0.0f;
        result.effective_history_weight = configured_weight * motion_confidence;
        result.color = glm::vec4(glm::mix(reconstructed_current,
                                          clamped_history,
                                          result.effective_history_weight),
                                 sample.current.a);
        return result;
    }

} // namespace lfs::vis
