/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "output_slot_ring.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace lfs::vis {

    OutputSlotRing::OutputSlotRing() {
        // Keep the historical numeric adapter alive while callers migrate to
        // stable view keys. These columns are intentionally never retired.
        (void)registerKey(kLegacyMainOutputKey);
        (void)registerKey(kLegacySplitLeftOutputKey);
        (void)registerKey(kLegacySplitRightOutputKey);
        (void)registerKey(kLegacyPreviewOutputKey);
    }

    OutputSlotRing::LogicalIndex OutputSlotRing::registerKey(const ViewOutputKey key) {
        if (!key.valid()) {
            throw std::invalid_argument("OutputSlotRing registerKey: key zero is invalid");
        }

        if (const auto existing = logical_by_key_.find(key); existing != logical_by_key_.end()) {
            return existing->second;
        }

        const bool append = free_logical_indices_.empty();
        const LogicalIndex logical = append ? slots_.size() : free_logical_indices_.back();

        // Reserve storage before publishing the map entry, so allocation failure
        // cannot leave the registry and column arrays out of sync.
        logical_by_key_.reserve(logical_by_key_.size() + 1);
        if (append) {
            slots_.reserve(slots_.size() + 1);
            latest_output_ring_slot_.reserve(latest_output_ring_slot_.size() + 1);
            output_generations_.reserve(output_generations_.size() + 1);
            keys_by_logical_.reserve(keys_by_logical_.size() + 1);
        }
        const auto [inserted, did_insert] = logical_by_key_.emplace(key, logical);
        if (!did_insert) {
            return inserted->second;
        }

        if (append) {
            slots_.emplace_back();
            latest_output_ring_slot_.push_back(0);
            output_generations_.push_back(0);
            keys_by_logical_.push_back(key);
        } else {
            free_logical_indices_.pop_back();
            slots_[logical] = {};
            latest_output_ring_slot_[logical] = 0;
            output_generations_[logical] = 0;
            keys_by_logical_[logical] = key;
        }
        ++active_logical_count_;
        return logical;
    }

    OutputSlotRing::LogicalIndex OutputSlotRing::registerSceneView(const ViewId view_id) {
        const auto key = sceneOutputKey(view_id);
        if (view_id == 0) {
            throw std::invalid_argument("OutputSlotRing registerSceneView: view id zero is invalid");
        }
        return registerKey(key);
    }

    std::optional<OutputSlotRing::LogicalIndex> OutputSlotRing::findKey(
        const ViewOutputKey key) const noexcept {
        if (!key.valid()) {
            return std::nullopt;
        }
        const auto found = logical_by_key_.find(key);
        return found == logical_by_key_.end() ? std::nullopt
                                              : std::optional<LogicalIndex>(found->second);
    }

    std::optional<ViewOutputKey> OutputSlotRing::keyAt(const LogicalIndex logical) const noexcept {
        if (logical >= keys_by_logical_.size() || !keys_by_logical_[logical].valid()) {
            return std::nullopt;
        }
        return keys_by_logical_[logical];
    }

    bool OutputSlotRing::retireKey(const ViewOutputKey key, const PerSlotFn& per_slot_fn) {
        const auto found = logical_by_key_.find(key);
        if (found == logical_by_key_.end() || isReservedLegacyOutputKey(key) || !per_slot_fn) {
            return false;
        }

        // Reserve before invoking renderer callbacks. Once callbacks succeed,
        // all remaining retirement operations are nonallocating.
        free_logical_indices_.reserve(free_logical_indices_.size() + 1);
        const LogicalIndex logical = found->second;
        // Run the renderer callback for every cell before clearing any cell.
        // A throwing callback leaves the key live and does not clear cells.
        for (auto& slot : slots_[logical]) {
            per_slot_fn(slot);
        }
        for (auto& slot : slots_[logical]) {
            slot = {};
        }

        logical_by_key_.erase(found);
        keys_by_logical_[logical] = kInvalidViewOutputKey;
        latest_output_ring_slot_[logical] = 0;
        output_generations_[logical] = 0;
        free_logical_indices_.push_back(logical);
        --active_logical_count_;
        return true;
    }

    void OutputSlotRing::checkLogical(const std::size_t logical, const std::string_view what) const {
        if (logical >= slots_.size()) [[unlikely]] {
            throw std::out_of_range(std::format(
                "OutputSlotRing {}: logical slot out of range (logical={}, count={})",
                what,
                logical,
                slots_.size()));
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
        for (auto& column : slots_) {
            column = {};
        }
        ring_completion_values_ = {};
        next_ring_slot_ = 0;
        std::fill(latest_output_ring_slot_.begin(), latest_output_ring_slot_.end(), 0);
        std::fill(output_generations_.begin(), output_generations_.end(), 0);
    }

    std::uint64_t OutputSlotRing::ringCompletionValue(const std::size_t ring_slot) const noexcept {
        if (ring_slot >= kFrameRingSize) {
            return 0;
        }
        return ring_completion_values_[ring_slot];
    }

    std::uint64_t OutputSlotRing::generation(const std::size_t logical) const noexcept {
        if (logical >= output_generations_.size()) {
            return 0;
        }
        return output_generations_[logical];
    }

} // namespace lfs::vis
