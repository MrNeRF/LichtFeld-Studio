/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "rendering/scene_temporal_plan.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace lfs::vis {

    struct SceneTemporalRequest {
        TemporalViewId view = TemporalViewId::Main;
        SceneTemporalRequirements requirements;
        TemporalFrameInput frame;
        glm::ivec2 render_extent{0, 0};
        glm::ivec2 output_extent{0, 0};
    };

    struct PreparedSceneTemporalFrame {
        TemporalViewId view = TemporalViewId::Main;
        SceneTemporalPlan plan;
        TemporalFrameState frame;
        SceneHistoryContract history;
        std::uint64_t ticket = 0;

        [[nodiscard]] constexpr bool active() const noexcept {
            return ticket != 0 && plan.active() && plan.valid();
        }
    };

    class LFS_VIS_API SceneTemporalCoordinator {
    public:
        [[nodiscard]] PreparedSceneTemporalFrame prepare(const SceneTemporalRequest& request);
        [[nodiscard]] bool commit(
            const PreparedSceneTemporalFrame& prepared,
            SceneHistoryStorage color_storage = SceneHistoryStorage::None,
            SceneHistoryStorage depth_storage = SceneHistoryStorage::None);
        void discard(const PreparedSceneTemporalFrame& prepared,
                     TemporalResetReason reason = TemporalResetReason::ResolveFailure);
        void reset(TemporalViewId view,
                   TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void resetAll(TemporalResetReason reason = TemporalResetReason::HistoryDisabled);

    private:
        struct PendingFrame {
            std::uint64_t ticket = 0;
            TemporalFrameInput input;
            SceneTemporalPlan plan;
            TemporalFrameState frame;
        };

        [[nodiscard]] static constexpr std::size_t index(const TemporalViewId view) noexcept {
            return static_cast<std::size_t>(view);
        }

        [[nodiscard]] std::uint64_t nextTicket() noexcept;

        TemporalFrameTracker frames_;
        SceneHistoryTracker histories_;
        std::array<std::optional<PendingFrame>,
                   static_cast<std::size_t>(TemporalViewId::Count)>
            pending_{};
        std::uint64_t next_ticket_ = 1;
    };

} // namespace lfs::vis
