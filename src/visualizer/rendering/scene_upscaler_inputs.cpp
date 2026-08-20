/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/scene_upscaler_inputs.hpp"

namespace lfs::vis {

    SceneUpscalerInputValidation validateSceneUpscalerInputs(
        const SceneTemporalPlan& plan, const SceneUpscalerInputs& inputs) {
        SceneUpscalerInputValidation result;
        if (!plan.valid()) {
            result.issues |= SceneUpscalerInputIssue::InvalidPlan;
            return result;
        }
        if (!plan.active()) {
            return result;
        }

        if (plan.requirements.depth) {
            if (!inputs.depth.available() || !inputs.depth.valid()) {
                result.issues |= SceneUpscalerInputIssue::MissingDepth;
            } else if (!inputs.depth.matchesRenderExtent(plan.render_extent)) {
                result.issues |= SceneUpscalerInputIssue::DepthExtent;
            }
        }
        if (plan.requirements.motion) {
            if (!inputs.motion.available() || !inputs.motion.valid()) {
                result.issues |= SceneUpscalerInputIssue::MissingMotion;
            } else {
                if (!inputs.motion.matchesRenderExtent(plan.render_extent.x,
                                                       plan.render_extent.y)) {
                    result.issues |= SceneUpscalerInputIssue::MotionExtent;
                }
                if (inputs.motion.direction != SceneMotionDirection::CurrentToPrevious) {
                    result.issues |= SceneUpscalerInputIssue::MotionDirection;
                }
                if (plan.requirements.jitter &&
                    inputs.motion.includes_jitter != inputs.jitter_applied) {
                    result.issues |= SceneUpscalerInputIssue::MotionJitterMismatch;
                }
            }
        }
        if (plan.requirements.jitter && !inputs.jitter_applied) {
            result.issues |= SceneUpscalerInputIssue::MissingJitter;
        }
        if (plan.requirements.history_color && inputs.history_expected) {
            if (!inputs.history.available() || !inputs.history.valid()) {
                result.issues |= SceneUpscalerInputIssue::MissingHistory;
            } else if (!inputs.history.matches(plan)) {
                result.issues |= SceneUpscalerInputIssue::HistoryExtent;
            }
        }
        return result;
    }

} // namespace lfs::vis
