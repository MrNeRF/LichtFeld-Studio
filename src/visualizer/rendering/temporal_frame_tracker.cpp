/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/temporal_frame_tracker.hpp"

#include <cmath>

namespace lfs::vis {
    namespace {
        constexpr float EPSILON = 1e-6f;

        bool finite(const lfs::rendering::FrameView& view, const glm::vec2 jitter,
                    const float scale) {
            const auto finite_vec3 = [](const glm::vec3& value) {
                return std::isfinite(value.x) && std::isfinite(value.y) &&
                       std::isfinite(value.z);
            };
            bool valid = view.size.x > 0 && view.size.y > 0 && finite_vec3(view.translation) &&
                         std::isfinite(view.focal_length_mm) && std::isfinite(view.near_plane) &&
                         std::isfinite(view.far_plane) && std::isfinite(view.ortho_scale) &&
                         std::isfinite(jitter.x) && std::isfinite(jitter.y) &&
                         std::isfinite(scale);
            for (int column = 0; column < 3; ++column)
                valid = valid && finite_vec3(view.rotation[column]);
            return valid;
        }

        bool different(const float lhs, const float rhs) {
            return std::abs(lhs - rhs) > EPSILON;
        }

        bool projectionChanged(const lfs::rendering::FrameView& lhs,
                               const lfs::rendering::FrameView& rhs) {
            if (lhs.orthographic != rhs.orthographic ||
                different(lhs.focal_length_mm, rhs.focal_length_mm) ||
                different(lhs.near_plane, rhs.near_plane) ||
                different(lhs.far_plane, rhs.far_plane) ||
                different(lhs.ortho_scale, rhs.ortho_scale) ||
                lhs.intrinsics_override.has_value() != rhs.intrinsics_override.has_value())
                return true;
            if (!lhs.intrinsics_override)
                return false;
            const auto& a = *lhs.intrinsics_override;
            const auto& b = *rhs.intrinsics_override;
            return different(a.focal_x, b.focal_x) || different(a.focal_y, b.focal_y) ||
                   different(a.center_x, b.center_x) || different(a.center_y, b.center_y);
        }

        float halton(std::uint64_t index, const std::uint64_t base) {
            float result = 0.0f;
            float fraction = 1.0f;
            while (index > 0) {
                fraction /= static_cast<float>(base);
                result += fraction * static_cast<float>(index % base);
                index /= base;
            }
            return result;
        }
    } // namespace

    glm::vec2 temporalJitterPixels(const std::uint64_t sequence) {
        const std::uint64_t sample = sequence + 1;
        return {halton(sample, 2) - 0.5f, halton(sample, 3) - 0.5f};
    }

    glm::vec2 temporalJitterNdc(const std::uint64_t sequence,
                                const glm::ivec2 render_size) {
        if (render_size.x <= 0 || render_size.y <= 0) {
            return {0.0f, 0.0f};
        }
        const glm::vec2 pixel = temporalJitterPixels(sequence);
        return {2.0f * pixel.x / static_cast<float>(render_size.x),
                2.0f * pixel.y / static_cast<float>(render_size.y)};
    }

    glm::mat4 applySceneProjectionJitter(const glm::mat4& projection,
                                         const glm::vec2 jitter_ndc) {
        if (!std::isfinite(jitter_ndc.x) || !std::isfinite(jitter_ndc.y)) {
            return projection;
        }
        glm::mat4 jittered = projection;
        for (int column = 0; column < 4; ++column) {
            jittered[column][0] += jitter_ndc.x * projection[column][3];
            jittered[column][1] += jitter_ndc.y * projection[column][3];
        }
        return jittered;
    }

    lfs::rendering::FrameView applySceneViewJitter(
        const lfs::rendering::FrameView& view, const glm::vec2 jitter_pixels) {
        if (!std::isfinite(jitter_pixels.x) || !std::isfinite(jitter_pixels.y) ||
            view.orthographic || jitter_pixels == glm::vec2(0.0f)) {
            return view;
        }

        lfs::rendering::FrameView jittered = view;
        const glm::ivec2 camera_size =
            view.subregion_full_size.x > 0 && view.subregion_full_size.y > 0
                ? view.subregion_full_size
                : view.size;
        if (camera_size.x <= 0 || camera_size.y <= 0) {
            return view;
        }

        if (!jittered.intrinsics_override) {
            const auto [focal_x, focal_y] = lfs::rendering::computePixelFocalLengths(
                camera_size, view.focal_length_mm);
            jittered.intrinsics_override = lfs::rendering::CameraIntrinsics{
                .focal_x = focal_x,
                .focal_y = focal_y,
                .center_x = static_cast<float>(camera_size.x) * 0.5f,
                .center_y = static_cast<float>(camera_size.y) * 0.5f,
            };
        }
        jittered.intrinsics_override->center_x += jitter_pixels.x;
        jittered.intrinsics_override->center_y += jitter_pixels.y;
        return jittered;
    }

    TemporalProjectionPair makeTemporalProjectionPair(
        const TemporalFrameState& state,
        const glm::mat4& current_projection,
        const glm::mat4& previous_projection) {
        return {
            .current = applySceneProjectionJitter(current_projection, state.current_jitter),
            .previous = applySceneProjectionJitter(previous_projection, state.previous_jitter),
        };
    }

    std::size_t TemporalFrameTracker::index(const TemporalViewId id) {
        return static_cast<std::size_t>(id);
    }

    TemporalFrameState TemporalFrameTracker::prepare(const TemporalViewId id,
                                                     const TemporalFrameInput& input) const {
        const auto& entry = entries_.at(index(id));
        TemporalFrameState result{.current = input.view,
                                  .previous = input.view,
                                  .current_jitter = input.jitter,
                                  .previous_jitter = input.jitter,
                                  .sequence = entry.sequence,
                                  .reset_reasons = entry.pending_reset};
        if (!finite(input.view, input.jitter, input.render_scale)) {
            result.reset_reasons |= TemporalResetReason::InvalidInput;
            return result;
        }
        if (!entry.committed) {
            result.reset_reasons |= TemporalResetReason::FirstFrame;
            return result;
        }

        const auto& previous = *entry.committed;
        result.previous = previous.view;
        result.previous_jitter = previous.jitter;
        if (input.camera_cut)
            result.reset_reasons |= TemporalResetReason::CameraCut;
        if (input.view.size != previous.view.size)
            result.reset_reasons |= TemporalResetReason::RenderSize;
        if (different(input.render_scale, previous.render_scale))
            result.reset_reasons |= TemporalResetReason::RenderScale;
        if (projectionChanged(input.view, previous.view))
            result.reset_reasons |= TemporalResetReason::Projection;
        if (input.scene_generation != previous.scene_generation)
            result.reset_reasons |= TemporalResetReason::Scene;
        if (input.backend_key != previous.backend_key)
            result.reset_reasons |= TemporalResetReason::Backend;
        result.history_valid = result.reset_reasons == TemporalResetReason::None;
        if (!result.history_valid) {
            result.previous = input.view;
            result.previous_jitter = input.jitter;
        }
        return result;
    }

    void TemporalFrameTracker::commit(const TemporalViewId id, const TemporalFrameInput& input) {
        auto& entry = entries_.at(index(id));
        if (!finite(input.view, input.jitter, input.render_scale)) {
            entry.committed.reset();
            entry.pending_reset = TemporalResetReason::InvalidInput;
            return;
        }
        entry.committed = input;
        ++entry.sequence;
        entry.pending_reset = TemporalResetReason::None;
    }

    void TemporalFrameTracker::reset(const TemporalViewId id, const TemporalResetReason reason) {
        auto& entry = entries_.at(index(id));
        entry.committed.reset();
        entry.pending_reset = reason;
    }

    void TemporalFrameTracker::resetAll(const TemporalResetReason reason) {
        for (auto& entry : entries_) {
            entry.committed.reset();
            entry.pending_reset = reason;
        }
    }
} // namespace lfs::vis
