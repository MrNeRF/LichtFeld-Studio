/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "output_slot_ring.hpp"

#include <format>
#include <stdexcept>

namespace lfs::vis {

    void OutputSlotRing::checkLogical(const std::size_t logical, const std::string_view what) const {
        if (logical >= kOutputSlotCount) [[unlikely]] {
            throw std::out_of_range(std::format(
                "OutputSlotRing {}: logical slot out of range (logical={}, count={})",
                what,
                logical,
                kOutputSlotCount));
        }
    }

    void OutputSlotRing::checkRing(const std::size_t ring, const std::string_view what) const {
        if (ring >= kFrameRingSize) [[unlikely]] {
            throw std::out_of_range(std::format(
                "OutputSlotRing {}: ring slot out of range (ring={}, size={})",
                what,
                ring,
                kFrameRingSize));
        }
    }

    std::size_t OutputSlotRing::acquire() noexcept {
        const std::size_t slot = next_ring_slot_;
        next_ring_slot_ = (next_ring_slot_ + 1) % kFrameRingSize;
        return slot;
    }

    lfs::Status OutputSlotRing::waitUntilReusable(const std::size_t ring_slot,
                                                  const std::string_view reason,
                                                  const TimelineCompleteFn& complete_fn,
                                                  const TimelineWaitFn& wait_fn) {
        if (ring_slot >= kFrameRingSize) {
            return {};
        }
        const std::uint64_t value = ring_completion_values_[ring_slot];
        if (value == 0) {
            return {};
        }
        try {
            if (complete_fn && complete_fn(value)) {
                ring_completion_values_[ring_slot] = 0;
                return {};
            }
        } catch (const std::exception& e) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = std::format("VkSplat {} ring-slot status failed: {}",
                                            reason,
                                            e.what()),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }

        if (!wait_fn) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = std::format(
                    "VkSplat {} ring-slot wait has no wait function (value={}, slot={})",
                    reason,
                    value,
                    ring_slot),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }

        // Non-Ready leaves ring_completion_values_[slot] unchanged
        // (no manufactured free slot).
        try {
            auto wait_status = wait_fn(value);
            if (!wait_status) {
                return wait_status;
            }
        } catch (const std::exception& e) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = std::format("VkSplat {} ring-slot wait failed: {}",
                                            reason,
                                            e.what()),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
        ring_completion_values_[ring_slot] = 0;
        return {};
    }

    void OutputSlotRing::publishCompletion(const std::size_t ring_slot,
                                           const std::uint64_t value) noexcept {
        if (ring_slot < kFrameRingSize) {
            ring_completion_values_[ring_slot] = value;
        }
    }

    void OutputSlotRing::clearSlotCompletion(const std::size_t logical, const std::size_t ring) {
        slotAt(logical, ring).completion_value = 0;
    }

    void OutputSlotRing::markLatest(const std::size_t logical, const std::size_t ring) {
        checkLogical(logical, "markLatest");
        checkRing(ring, "markLatest");
        latest_output_ring_slot_[logical] = ring;
    }

    std::uint64_t OutputSlotRing::bumpGeneration(const std::size_t logical) {
        checkLogical(logical, "bumpGeneration");
        return ++output_generations_[logical];
    }

    OutputImageSlot& OutputSlotRing::slotAt(const std::size_t logical, const std::size_t ring) {
        checkLogical(logical, "slotAt");
        checkRing(ring, "slotAt");
        return slots_[logical][ring];
    }

    const OutputImageSlot& OutputSlotRing::slotAt(const std::size_t logical,
                                                  const std::size_t ring) const {
        checkLogical(logical, "slotAt");
        checkRing(ring, "slotAt");
        return slots_[logical][ring];
    }

    std::size_t OutputSlotRing::latestRingSlot(const std::size_t logical) const {
        checkLogical(logical, "latestRingSlot");
        const std::size_t ring_slot = latest_output_ring_slot_[logical];
        if (ring_slot >= kFrameRingSize) [[unlikely]] {
            throw std::out_of_range(std::format(
                "VkSplat latest output ring slot is outside the ring "
                "(output_index={}, observed_ring_slot={}, ring_size={})",
                logical,
                ring_slot,
                kFrameRingSize));
        }
        return ring_slot;
    }

    OutputImageSlot& OutputSlotRing::latestSlot(const std::size_t logical) {
        return slotAt(logical, latestRingSlot(logical));
    }

    const OutputImageSlot& OutputSlotRing::latestSlot(const std::size_t logical) const {
        return slotAt(logical, latestRingSlot(logical));
    }

    void OutputSlotRing::clearLogical(const std::size_t logical, const PerSlotFn& per_slot_fn) {
        checkLogical(logical, "clearLogical");
        for (auto& slot : slots_[logical]) {
            if (per_slot_fn) {
                per_slot_fn(slot);
            }
            slot = {};
        }
        latest_output_ring_slot_[logical] = 0;
        output_generations_[logical] = 0;
    }

    void OutputSlotRing::reset() noexcept {
        slots_ = {};
        ring_completion_values_ = {};
        next_ring_slot_ = 0;
        latest_output_ring_slot_ = {};
        output_generations_ = {};
    }

    std::uint64_t OutputSlotRing::ringCompletionValue(const std::size_t ring_slot) const noexcept {
        if (ring_slot >= kFrameRingSize) {
            return 0;
        }
        return ring_completion_values_[ring_slot];
    }

    std::uint64_t OutputSlotRing::generation(const std::size_t logical) const noexcept {
        if (logical >= kOutputSlotCount) {
            return 0;
        }
        return output_generations_[logical];
    }

} // namespace lfs::vis
