/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "rendering/scene_temporal_plan.hpp"

namespace lfs::vis {

    struct SceneTemporalRequest {
        TemporalViewId view = TemporalViewId::Main;
        SceneUpscalerRequirements requirements;
        TemporalFrameInput frame;
        glm::ivec2 render_extent{0, 0};
        glm::ivec2 output_extent{0, 0};
    };

    struct PreparedSceneTemporalFrame {
        TemporalViewId view = TemporalViewId::Main;
        SceneTemporalPlan plan;
        TemporalFrameState frame;
        SceneHistoryContract history;

        [[nodiscard]] constexpr bool active() const {
            return plan.active() && plan.valid();
        }

        [[nodiscard]] constexpr bool temporal() const {
            return active() && plan.temporal();
        }
    };

    class LFS_VIS_API SceneTemporalCoordinator {
    public:
        [[nodiscard]] PreparedSceneTemporalFrame prepare(const SceneTemporalRequest& request);
        [[nodiscard]] bool commit(const SceneTemporalRequest& request,
                                  const PreparedSceneTemporalFrame& prepared,
                                  SceneHistoryStorage color_storage = SceneHistoryStorage::None,
                                  SceneHistoryStorage depth_storage = SceneHistoryStorage::None);
        void reset(TemporalViewId view,
                   TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void resetAll(TemporalResetReason reason = TemporalResetReason::HistoryDisabled);

    private:
        TemporalFrameTracker frames_;
        SceneHistoryTracker histories_;
    };

} // namespace lfs::vis
