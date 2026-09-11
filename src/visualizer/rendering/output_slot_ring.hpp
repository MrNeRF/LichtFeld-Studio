/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "window/vulkan_context.hpp"

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

    // Host bookkeeping for the 3-deep frame ring × 4 logical output slots.
    // GPU-free: timeline complete/wait are injected; no Vulkan/CUDA calls.
    class LFS_VIS_API OutputSlotRing {
    public:
        static constexpr std::size_t kOutputSlotCount = 4;
        static constexpr std::size_t kFrameRingSize = 3;

        // Poll whether a timeline value has already retired. May throw.
        using TimelineCompleteFn = std::function<bool(std::uint64_t value)>;
        // Bounded wait for a timeline value. Success means Ready; failure
        // carries the diagnostic and MUST leave the ring watermark intact.
        using TimelineWaitFn = std::function<lfs::Status(std::uint64_t value)>;
        // Renderer-side pool release + barrier forget for one slot cell.
        using PerSlotFn = std::function<void(OutputImageSlot&)>;

        // Round-robin acquire of the next ring index.
        [[nodiscard]] std::size_t acquire() noexcept;

        // Wait until ring index `ring_slot` may be reused.
        // - watermark 0 → free (no-op success)
        // - complete_fn true → zero watermark, success
        // - else wait_fn; zero watermark only on Ready (wait_fn success)
        // Never manufactures a free slot on non-Ready.
        [[nodiscard]] lfs::Status waitUntilReusable(std::size_t ring_slot,
                                                    std::string_view reason,
                                                    const TimelineCompleteFn& complete_fn,
                                                    const TimelineWaitFn& wait_fn);

        // Publish the per-ring reuse watermark after a successful submit.
        void publishCompletion(std::size_t ring_slot, std::uint64_t value) noexcept;

        // Clear the per-image completion value (compose start / resize path).
        void clearSlotCompletion(std::size_t logical, std::size_t ring);

        // Mark `ring` as the published "latest" for `logical`.
        void markLatest(std::size_t logical, std::size_t ring);

        // Bump and return the monotonic content generation for `logical`.
        [[nodiscard]] std::uint64_t bumpGeneration(std::size_t logical);

        [[nodiscard]] OutputImageSlot& slotAt(std::size_t logical, std::size_t ring);
        [[nodiscard]] const OutputImageSlot& slotAt(std::size_t logical, std::size_t ring) const;

        // Throws std::out_of_range if the stored latest index is out of range
        // (same contract as the historical latestOutputRingSlot).
        [[nodiscard]] std::size_t latestRingSlot(std::size_t logical) const;

        [[nodiscard]] OutputImageSlot& latestSlot(std::size_t logical);
        [[nodiscard]] const OutputImageSlot& latestSlot(std::size_t logical) const;

        // Invoke per_slot_fn for every ring cell in the logical column, then
        // zero those cells and the latest/generation for that logical index.
        void clearLogical(std::size_t logical, const PerSlotFn& per_slot_fn);

        // Zero every table, cursor, and watermark.
        void reset() noexcept;

        [[nodiscard]] std::uint64_t ringCompletionValue(std::size_t ring_slot) const noexcept;
        [[nodiscard]] std::uint64_t generation(std::size_t logical) const noexcept;
        [[nodiscard]] std::size_t nextRingSlot() const noexcept { return next_ring_slot_; }

        [[nodiscard]] const std::array<std::array<OutputImageSlot, kFrameRingSize>, kOutputSlotCount>&
        table() const noexcept {
            return slots_;
        }
        [[nodiscard]] std::array<std::array<OutputImageSlot, kFrameRingSize>, kOutputSlotCount>&
        table() noexcept {
            return slots_;
        }

    private:
        void checkLogical(std::size_t logical, std::string_view what) const;
        void checkRing(std::size_t ring, std::string_view what) const;

        std::array<std::array<OutputImageSlot, kFrameRingSize>, kOutputSlotCount> slots_{};
        std::array<std::uint64_t, kFrameRingSize> ring_completion_values_{};
        std::size_t next_ring_slot_ = 0;
        std::array<std::size_t, kOutputSlotCount> latest_output_ring_slot_{};
        std::array<std::uint64_t, kOutputSlotCount> output_generations_{};
    };

} // namespace lfs::vis
