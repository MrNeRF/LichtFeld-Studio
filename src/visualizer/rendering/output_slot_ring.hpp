/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "render_target_id.hpp"
#include "window/vulkan_context.hpp"
#include <unordered_map>
#include <vector>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <string_view>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    // Per (logical × ring) image payload + bookkeeping. Slot images are
    // non-owning copies; OutputImagePool owns destruction via the serials.
    struct OutputImageSlot {
        VulkanContext::ExternalImage image{};
        VulkanContext::ExternalImage depth_image{};
        glm::ivec2 size{0, 0};       // valid/logical extent
        glm::ivec2 alloc_size{0, 0}; // allocated (ceil64) extent
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        // Content identity for consumers (compose may overwrite). Not a tracker key.
        std::uint64_t generation = 0;
        // Resource identity for VulkanImageBarrierTracker. Set from the color
        // pool acquisition serial when the VkImages are (re)acquired.
        std::uint64_t image_generation = 0;
        // Pool acquisition serials (0 = none).
        std::uint64_t color_pool_serial = 0;
        std::uint64_t depth_pool_serial = 0;
        // Timeline value signalled by the compute submission that produced
        // this exact ring image. Graphics-queue readbacks wait this value.
        std::uint64_t completion_value = 0;
    };

    // Each target owns three images. Submission indices remain unique even after
    // release, so a late readback cannot pin another target's transient storage.
    class LFS_VIS_API OutputSlotRing {
    public:
        static constexpr std::size_t kFrameRingSize = 3;
        using TimelineCompleteFn = std::function<bool(std::uint64_t)>;
        using TimelineWaitFn = std::function<lfs::Status(std::uint64_t)>;
        using PerSlotFn = std::function<void(OutputImageSlot&)>;
        struct Column {
            std::array<OutputImageSlot, kFrameRingSize> slots{};
            std::size_t base = 0;
            std::size_t cursor = 0;
            std::size_t latest = 0;
            std::uint64_t generation = 0;
        };
        using Table = std::unordered_map<RenderTargetId, Column, RenderTargetIdHash>;
        [[nodiscard]] std::size_t acquire(RenderTargetId target);
        [[nodiscard]] std::size_t acquireTransient(const TimelineCompleteFn& complete);
        [[nodiscard]] lfs::Status waitUntilReusable(std::size_t cell, std::string_view reason,
                                                    const TimelineCompleteFn& complete,
                                                    const TimelineWaitFn& wait);
        void publishCompletion(std::size_t cell, std::uint64_t value) noexcept;
        void clearSlotCompletion(RenderTargetId target, std::size_t cell);
        void markLatest(RenderTargetId target, std::size_t cell);
        [[nodiscard]] std::uint64_t bumpGeneration(RenderTargetId target);
        [[nodiscard]] OutputImageSlot& slotAt(RenderTargetId target, std::size_t cell);
        [[nodiscard]] const OutputImageSlot& slotAt(RenderTargetId target, std::size_t cell) const;
        [[nodiscard]] std::size_t latestRingSlot(RenderTargetId target) const;
        [[nodiscard]] OutputImageSlot& latestSlot(RenderTargetId target);
        [[nodiscard]] const OutputImageSlot& latestSlot(RenderTargetId target) const;
        void clearLogical(RenderTargetId target, const PerSlotFn& release);
        bool releaseRenderTarget(RenderTargetId target, const PerSlotFn& release);
        [[nodiscard]] bool released(RenderTargetId target) const { return released_.contains(target); }
        [[nodiscard]] bool contains(RenderTargetId target) const { return slots_.contains(target); }
        [[nodiscard]] std::size_t submissionCount() const { return completions_.size(); }
        void reset() noexcept;
        [[nodiscard]] std::uint64_t ringCompletionValue(std::size_t cell) const noexcept;
        [[nodiscard]] std::uint64_t generation(RenderTargetId target) const noexcept;
        [[nodiscard]] const Table& table() const noexcept { return slots_; }

    private:
        Table slots_;
        std::unordered_set<RenderTargetId, RenderTargetIdHash> released_;
        std::vector<std::uint64_t> completions_;
        std::vector<std::size_t> transient_cells_;
    };
} // namespace lfs::vis
