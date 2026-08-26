/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/temporal_frame_tracker.hpp"

#include <cmath>

namespace lfs::vis {
    namespace {
        constexpr float EPSILON = 1e-6f;

        bool finiteFrameView(const lfs::rendering::FrameView& view,
                             const glm::vec2 jitter,
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

        std::optional<glm::mat4> projectionFromView(const lfs::rendering::FrameView& view) {
            if (view.size.x <= 0 || view.size.y <= 0 || view.intrinsics_override.has_value() ||
                !std::isfinite(view.near_plane) || !std::isfinite(view.far_plane) ||
                view.near_plane <= 0.0f || view.far_plane <= view.near_plane) {
                return std::nullopt;
            }
            return lfs::rendering::createProjectionMatrixFromFocal(
                view.size,
                view.focal_length_mm,
                view.orthographic,
                view.ortho_scale,
                view.near_plane,
                view.far_plane);
        }

        std::optional<TemporalProjectionPair> makeTemporalViewProjectionPairWithJitters(
            const TemporalFrameState& state,
            const glm::vec2 current_jitter,
            const glm::vec2 previous_jitter) {
            const auto current_projection = projectionFromView(state.current);
            const auto previous_projection = projectionFromView(state.previous);
            if (!current_projection || !previous_projection) {
                return std::nullopt;
            }
            TemporalFrameState jittered = state;
            jittered.current_jitter = current_jitter;
            jittered.previous_jitter = previous_jitter;
            const auto projections = makeTemporalProjectionPair(
                jittered, *current_projection, *previous_projection);
            return TemporalProjectionPair{
                .current = projections.current * state.current.getViewMatrix(),
                .previous = projections.previous * state.previous.getViewMatrix(),
            };
        }
    } // namespace

    bool validTemporalFrameInput(const TemporalFrameInput& input) noexcept {
        return input.output_extent.x > 0 && input.output_extent.y > 0 &&
               finiteFrameView(input.view, input.jitter, input.render_scale);
    }

    glm::vec2 temporalJitterPixels(const std::uint64_t sequence) {
        const std::uint64_t sample = sequence + 1;
        return {halton(sample, 2) - 0.5f, halton(sample, 3) - 0.5f};
    }

    glm::vec2 temporalJitterNdc(const glm::vec2 jitter_pixels,
                                const glm::ivec2 render_size) {
        if (render_size.x <= 0 || render_size.y <= 0) {
            return {0.0f, 0.0f};
        }
        if (!std::isfinite(jitter_pixels.x) || !std::isfinite(jitter_pixels.y)) {
            return {0.0f, 0.0f};
        }
        // FrameView intrinsics use top-left image coordinates (+Y down), while
        // projection matrices use OpenGL NDC (+Y up). Keep the conversion at
        // this boundary so the jitter applied by the raster and the matrices
        // consumed by motion reprojection describe the same sample position.
        return {2.0f * jitter_pixels.x / static_cast<float>(render_size.x),
                -2.0f * jitter_pixels.y / static_cast<float>(render_size.y)};
    }

    glm::vec2 temporalJitterNdc(const std::uint64_t sequence,
                                const glm::ivec2 render_size) {
        return temporalJitterNdc(temporalJitterPixels(sequence), render_size);
    }

    void TemporalConvergenceController::prepare(const bool enabled,
                                                const bool restart,
                                                const bool allow_settle) {
        enabled_ = enabled;
        if (!enabled_) {
            sequence_ = 0;
            remaining_ = 0;
        } else if (!allow_settle) {
            // Volatile sources such as live training previews and streamed LOD
            // transitions invalidate history again before a settle burst can
            // become useful. Keep the continuous jitter sequence, but do not
            // enqueue follow-up renders that would only contend for the GPU.
            remaining_ = 0;
        } else if (restart) {
            remaining_ = SAMPLE_COUNT;
        }
    }

    glm::vec2 TemporalConvergenceController::jitter() const {
        return enabled_ ? temporalJitterPixels(sequence_) : glm::vec2(0.0f);
    }

    bool TemporalConvergenceController::completeSuccessfulFrame() {
        if (!enabled_) {
            return false;
        }
        ++sequence_;
        if (remaining_ > 0) {
            --remaining_;
        }
        return remaining_ > 0;
    }

    void TemporalConvergenceController::cancelSettle() {
        remaining_ = 0;
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

    std::optional<TemporalProjectionPair> makeTemporalViewProjectionPair(
        const TemporalFrameState& state) {
        return makeTemporalViewProjectionPairWithJitters(
            state, state.current_jitter, state.previous_jitter);
    }

    std::optional<TemporalProjectionPair> makeTemporalMotionViewProjectionPair(
        const TemporalFrameState& state) {
        return makeTemporalViewProjectionPairWithJitters(state, glm::vec2(0.0f), glm::vec2(0.0f));
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
        if (!validTemporalFrameInput(input)) {
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
        if (input.output_extent != previous.output_extent)
            result.reset_reasons |= TemporalResetReason::OutputExtent;
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
            result.sequence = 0;
        }
        return result;
    }

    void TemporalFrameTracker::commit(const TemporalViewId id, const TemporalFrameInput& input) {
        auto& entry = entries_.at(index(id));
        if (!validTemporalFrameInput(input)) {
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
        entry.sequence = 0;
        entry.pending_reset = reason;
    }

    void TemporalFrameTracker::resetAll(const TemporalResetReason reason) {
        for (auto& entry : entries_) {
            entry.committed.reset();
            entry.sequence = 0;
            entry.pending_reset = reason;
        }
    }
} // namespace lfs::vis
