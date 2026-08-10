/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "rendering/scene_depth_contract.hpp"
#include "rendering/scene_motion_contract.hpp"
#include "rendering/scene_temporal_plan.hpp"

#include <cstdint>

namespace lfs::vis {

    enum class SceneUpscalerInputIssue : std::uint32_t {
        None = 0,
        InvalidPlan = 1u << 0u,
        MissingDepth = 1u << 1u,
        DepthExtent = 1u << 2u,
        MissingMotion = 1u << 3u,
        MotionExtent = 1u << 4u,
        MotionDirection = 1u << 5u,
        MissingJitter = 1u << 6u,
        MissingHistory = 1u << 7u,
        HistoryExtent = 1u << 8u,
        MissingReactiveMask = 1u << 9u,
        MissingExposure = 1u << 10u,
        MotionJitterMismatch = 1u << 11u,
    };

    [[nodiscard]] constexpr SceneUpscalerInputIssue operator|(const SceneUpscalerInputIssue lhs,
                                                              const SceneUpscalerInputIssue rhs) {
        return static_cast<SceneUpscalerInputIssue>(static_cast<std::uint32_t>(lhs) |
                                                    static_cast<std::uint32_t>(rhs));
    }

    constexpr SceneUpscalerInputIssue& operator|=(SceneUpscalerInputIssue& lhs,
                                                  const SceneUpscalerInputIssue rhs) {
        lhs = lhs | rhs;
        return lhs;
    }

    [[nodiscard]] constexpr bool hasSceneUpscalerInputIssue(const SceneUpscalerInputIssue value,
                                                            const SceneUpscalerInputIssue issue) {
        return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(issue)) != 0;
    }

    struct SceneUpscalerInputs {
        SceneDepthContract depth;
        SceneMotionContract motion;
        SceneHistoryContract history;
        bool jitter_applied = false;
        bool reactive_mask_available = false;
        bool exposure_available = false;
    };

    struct SceneUpscalerInputValidation {
        SceneUpscalerInputIssue issues = SceneUpscalerInputIssue::None;

        [[nodiscard]] constexpr bool valid() const {
            return issues == SceneUpscalerInputIssue::None;
        }
    };

    [[nodiscard]] LFS_VIS_API SceneUpscalerInputValidation validateSceneUpscalerInputs(
        const SceneTemporalPlan& plan, const SceneUpscalerInputs& inputs);

} // namespace lfs::vis
