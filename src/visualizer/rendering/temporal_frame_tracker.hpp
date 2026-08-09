/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "rendering/frame_contract.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace lfs::vis {

    enum class TemporalViewId : std::uint8_t { Main,
                                               SplitLeft,
                                               SplitRight,
                                               Count };

    enum class TemporalResetReason : std::uint32_t {
        None = 0,
        FirstFrame = 1u << 0u,
        CameraCut = 1u << 1u,
        RenderSize = 1u << 2u,
        RenderScale = 1u << 3u,
        Projection = 1u << 4u,
        Scene = 1u << 5u,
        Backend = 1u << 6u,
        HistoryDisabled = 1u << 7u,
        InvalidInput = 1u << 8u,
    };

    [[nodiscard]] constexpr TemporalResetReason operator|(const TemporalResetReason lhs,
                                                          const TemporalResetReason rhs) {
        return static_cast<TemporalResetReason>(static_cast<std::uint32_t>(lhs) |
                                                static_cast<std::uint32_t>(rhs));
    }

    constexpr TemporalResetReason& operator|=(TemporalResetReason& lhs,
                                              const TemporalResetReason rhs) {
        lhs = lhs | rhs;
        return lhs;
    }

    [[nodiscard]] constexpr bool hasTemporalResetReason(const TemporalResetReason value,
                                                        const TemporalResetReason reason) {
        return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(reason)) != 0;
    }

    struct TemporalFrameInput {
        lfs::rendering::FrameView view;
        float render_scale = 1.0f;
        std::uint64_t scene_generation = 0;
        std::uint64_t backend_key = 0;
        bool camera_cut = false;
    };

    struct TemporalFrameState {
        lfs::rendering::FrameView current;
        lfs::rendering::FrameView previous;
        glm::vec2 current_jitter{0.0f};
        glm::vec2 previous_jitter{0.0f};
        std::uint64_t sequence = 0;
        TemporalResetReason reset_reasons = TemporalResetReason::FirstFrame;
        bool history_valid = false;
    };

    class LFS_VIS_API TemporalFrameTracker {
    public:
        [[nodiscard]] TemporalFrameState prepare(TemporalViewId id,
                                                 const TemporalFrameInput& input) const;
        void commit(TemporalViewId id, const TemporalFrameInput& input);
        void reset(TemporalViewId id,
                   TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void resetAll(TemporalResetReason reason = TemporalResetReason::HistoryDisabled);

    private:
        struct Entry {
            std::optional<TemporalFrameInput> committed;
            std::uint64_t sequence = 0;
            TemporalResetReason pending_reset = TemporalResetReason::None;
        };

        [[nodiscard]] static std::size_t index(TemporalViewId id);
        std::array<Entry, static_cast<std::size_t>(TemporalViewId::Count)> entries_{};
    };

} // namespace lfs::vis
