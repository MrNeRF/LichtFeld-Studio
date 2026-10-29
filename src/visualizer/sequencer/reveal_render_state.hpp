/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"
#include "rendering/rendering.hpp"
#include "scene/visible_splat_bounds.hpp"
#include "sequencer/timeline.hpp"
#include "shaders/shader_service.hpp"
#include <algorithm>
#include <cmath>
#include <optional>

namespace lfs::vis {
    inline std::optional<shaders::CustomRevealContext>
    makeCustomRevealContext(const std::optional<sequencer::RevealClip>& clip, const float time) {
        if (!clip || clip->effect != sequencer::RevealEffect::Custom || !clip->shader_id || time >= clip->end)
            return std::nullopt;
        return shaders::CustomRevealContext{*clip->shader_id, clip->progress(time), clip->softness};
    }

    // The reveal of `clip` at `time` over geometry bounded by `lo` and `hi`.
    inline std::optional<rendering::GaussianRevealState> makeRevealRenderState(
        const glm::vec3& lo, const glm::vec3& hi, const sequencer::RevealClip& clip, const float time) {
        if (clip.effect == sequencer::RevealEffect::Custom || time >= clip.end)
            return std::nullopt;
        // The spread starts at the scene's world origin, independently of the
        // positions or transforms of the visible splats.
        const float center_x = 0.0f;
        const float center_z = 0.0f;
        const float radius = glm::length(glm::vec2(std::max(std::abs(lo.x), std::abs(hi.x)),
                                                   std::max(std::abs(lo.z), std::abs(hi.z))));
        return rendering::GaussianRevealState{clip.progress(time), clip.softness,
                                              center_x, center_z, radius, static_cast<int>(clip.effect), lo.y, hi.y};
    }

    inline std::optional<rendering::GaussianRevealState> makeRevealRenderState(
        const core::Scene& scene, const sequencer::RevealClip& clip, const float time) {
        glm::vec3 lo, hi;
        if (!visibleSplatBounds(scene, lo, hi))
            return std::nullopt;
        return makeRevealRenderState(lo, hi, clip, time);
    }
} // namespace lfs::vis
