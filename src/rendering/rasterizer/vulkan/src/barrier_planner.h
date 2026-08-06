/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "buffer.h"

#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::rendering::vulkan {

    // Pure host-side hazard planner for buffer barriers (epic #1496).
    // Vulkan headers provide types only — no Vulkan calls.

    enum class BufferUse : uint8_t {
        ComputeRead,
        ComputeWrite,
        ComputeReadWrite,
        TransferRead,
        TransferWrite,
        IndirectRead,    // DRAW_INDIRECT stage, INDIRECT_COMMAND_READ access
        HostRead,        // host readback (barrier only; fence/timeline wait is separate)
        ConditionalRead, // conditional-rendering predicate consumption
    };

    struct Scope {
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 access = VK_ACCESS_2_NONE;

        friend bool operator==(const Scope& a, const Scope& b) noexcept {
            return a.stage == b.stage && a.access == b.access;
        }
    };

    struct DeclaredAccess {
        const _VulkanBuffer* buffer = nullptr;
        BufferUse use = BufferUse::ComputeRead;
    };

    class BufferBarrierPlanner {
    public:
        struct Stats {
            uint64_t barriers_emitted = 0;
            uint64_t accesses_elided = 0;
            uint64_t conservative_fallbacks = 0;
        };

        explicit BufferBarrierPlanner(uint32_t queue_family_index = 0);

        // Register a tracked buffer (owned allocation or adopted external parent).
        void track(VkBuffer buffer);
        // Drop tracking; handle reuse must not inherit state.
        void forget(VkBuffer buffer);
        // Legacy-path hook: next access takes the conservative row.
        void invalidate(VkBuffer buffer);
        // Batch-boundary reset (beginCommandBatch): conservative writer + reuse-barrier visibility.
        void onBatchBegin();
        void reset();

        // Merge accesses per VkBuffer, hazard-check once against pre-plan state, then update state.
        [[nodiscard]] std::vector<VkBufferMemoryBarrier2> plan(std::span<const DeclaredAccess> accesses);

        [[nodiscard]] Stats stats() const;

    private:
        struct BufferState {
            Scope writer; // last unsynchronized writer ({NONE,NONE} if none)
            VkPipelineStageFlags2 reader_stages = VK_PIPELINE_STAGE_2_NONE;
            VkPipelineStageFlags2 visible_stages = VK_PIPELINE_STAGE_2_NONE;
            VkAccessFlags2 visible_access = VK_ACCESS_2_NONE;
        };

        struct MergedAccess {
            VkBuffer handle = VK_NULL_HANDLE;
            Scope scope;
            bool is_write = false;
        };

        [[nodiscard]] static Scope scopeFor(BufferUse use) noexcept;
        [[nodiscard]] static Scope conservativeWriterScope() noexcept;
        [[nodiscard]] static Scope reuseBarrierDstScope() noexcept;
        [[nodiscard]] static bool isWriteUse(BufferUse use) noexcept;
        [[nodiscard]] static bool isVisible(const Scope& a, const BufferState& state) noexcept;
        [[nodiscard]] static bool hasWriter(const BufferState& state) noexcept;

        [[nodiscard]] VkBufferMemoryBarrier2 makeBarrier(VkBuffer buffer, Scope src, Scope dst) const;

        uint32_t queue_family_index_ = 0;
        std::unordered_set<VkBuffer> tracked_;
        std::unordered_set<VkBuffer> force_conservative_;
        std::unordered_map<VkBuffer, BufferState> states_;
        Stats stats_{};
    };

} // namespace lfs::rendering::vulkan
