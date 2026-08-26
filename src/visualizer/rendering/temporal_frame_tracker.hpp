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
        Quality = 1u << 9u,
        Requested = 1u << 10u,
        RuntimeUnavailable = 1u << 11u,
        ResolveFailure = 1u << 12u,
        OutputExtent = 1u << 13u,
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

    [[nodiscard]] constexpr std::uint32_t temporalResetReasonMask(
        const TemporalResetReason value) {
        return static_cast<std::uint32_t>(value);
    }

    struct TemporalFrameInput {
        lfs::rendering::FrameView view;
        glm::ivec2 output_extent{0, 0};
        glm::vec2 jitter{0.0f};
        float render_scale = 1.0f;
        std::uint64_t scene_generation = 0;
        std::uint64_t backend_key = 0;
        bool camera_cut = false;
    };

    [[nodiscard]] LFS_VIS_API bool validTemporalFrameInput(
        const TemporalFrameInput& input) noexcept;

    [[nodiscard]] LFS_VIS_API glm::vec2 temporalJitterPixels(std::uint64_t sequence);
    [[nodiscard]] LFS_VIS_API glm::vec2 temporalJitterNdc(glm::vec2 jitter_pixels,
                                                          glm::ivec2 render_size);
    [[nodiscard]] LFS_VIS_API glm::vec2 temporalJitterNdc(std::uint64_t sequence,
                                                          glm::ivec2 render_size);

    class LFS_VIS_API TemporalConvergenceController {
    public:
        static constexpr std::uint32_t SAMPLE_COUNT = 8;

        void prepare(bool enabled, bool restart, bool allow_settle = true);
        [[nodiscard]] glm::vec2 jitter() const;
        [[nodiscard]] bool completeSuccessfulFrame();
        void cancelSettle();
        [[nodiscard]] std::uint64_t sequence() const { return sequence_; }
        [[nodiscard]] std::uint32_t remaining() const { return remaining_; }
        [[nodiscard]] bool enabled() const { return enabled_; }

    private:
        bool enabled_ = false;
        std::uint64_t sequence_ = 0;
        std::uint32_t remaining_ = 0;
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

    struct TemporalProjectionPair {
        glm::mat4 current{1.0f};
        glm::mat4 previous{1.0f};
    };

    [[nodiscard]] LFS_VIS_API glm::mat4 applySceneProjectionJitter(
        const glm::mat4& projection, glm::vec2 jitter_ndc);
    [[nodiscard]] LFS_VIS_API lfs::rendering::FrameView applySceneViewJitter(
        const lfs::rendering::FrameView& view, glm::vec2 jitter_pixels);
    [[nodiscard]] LFS_VIS_API TemporalProjectionPair makeTemporalProjectionPair(
        const TemporalFrameState& state,
        const glm::mat4& current_projection,
        const glm::mat4& previous_projection);
    [[nodiscard]] LFS_VIS_API std::optional<TemporalProjectionPair>
    makeTemporalViewProjectionPair(const TemporalFrameState& state);
    [[nodiscard]] LFS_VIS_API std::optional<TemporalProjectionPair>
    makeTemporalMotionViewProjectionPair(const TemporalFrameState& state);

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
