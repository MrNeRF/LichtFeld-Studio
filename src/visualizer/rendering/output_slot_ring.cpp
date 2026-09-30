/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "output_slot_ring.hpp"

#include <format>
#include <stdexcept>

namespace lfs::vis {

    std::size_t OutputSlotRing::acquire(RenderTargetId target) {
        if (!target.valid() || released(target))
            throw std::invalid_argument("Invalid or released render target");
        auto [it, inserted] = slots_.try_emplace(target);
        auto& column = it->second;
        if (inserted) {
            column.base = completions_.size();
            column.latest = column.base;
            completions_.resize(column.base + kFrameRingSize);
        }
        const auto cell = column.base + column.cursor;
        column.cursor = (column.cursor + 1) % kFrameRingSize;
        return cell;
    }
    std::size_t OutputSlotRing::acquireTransient(const TimelineCompleteFn& complete) {
        for (auto cell : transient_cells_) {
            if (!completions_[cell] || complete(completions_[cell]))
                return cell;
        }
        const auto cell = completions_.size();
        completions_.push_back(0);
        transient_cells_.push_back(cell);
        return cell;
    }
    lfs::Status OutputSlotRing::waitUntilReusable(const std::size_t ring_slot,
                                                  const std::string_view reason,
                                                  const TimelineCompleteFn& complete_fn,
                                                  const TimelineWaitFn& wait_fn) {
        if (ring_slot >= completions_.size()) {
            return {};
        }
        const std::uint64_t value = completions_[ring_slot];
        if (value == 0) {
            return {};
        }
        try {
            if (complete_fn && complete_fn(value)) {
                completions_[ring_slot] = 0;
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

        // Non-Ready leaves completions_[slot] unchanged
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
        completions_[ring_slot] = 0;
        return {};
    }

    void OutputSlotRing::publishCompletion(std::size_t cell, std::uint64_t value) noexcept {
        if (cell < completions_.size())
            completions_[cell] = value;
    }
    void OutputSlotRing::clearSlotCompletion(RenderTargetId target, std::size_t cell) {
        slotAt(target, cell).completion_value = 0;
    }
    void OutputSlotRing::markLatest(RenderTargetId target, std::size_t cell) {
        if (released(target))
            return;
        (void)slotAt(target, cell);
        slots_.at(target).latest = cell;
    }
    std::uint64_t OutputSlotRing::bumpGeneration(RenderTargetId target) {
        if (released(target))
            return 0;
        return ++slots_.at(target).generation;
    }
    OutputImageSlot& OutputSlotRing::slotAt(RenderTargetId target, std::size_t cell) {
        auto& column = slots_.at(target);
        return column.slots.at(cell - column.base);
    }
    const OutputImageSlot& OutputSlotRing::slotAt(RenderTargetId target, std::size_t cell) const {
        const auto& column = slots_.at(target);
        return column.slots.at(cell - column.base);
    }
    std::size_t OutputSlotRing::latestRingSlot(RenderTargetId target) const {
        return slots_.at(target).latest;
    }
    OutputImageSlot& OutputSlotRing::latestSlot(RenderTargetId target) {
        return slotAt(target, latestRingSlot(target));
    }
    const OutputImageSlot& OutputSlotRing::latestSlot(RenderTargetId target) const {
        static const OutputImageSlot empty;
        return contains(target) ? slotAt(target, latestRingSlot(target)) : empty;
    }
    void OutputSlotRing::clearLogical(RenderTargetId target, const PerSlotFn& release) {
        auto it = slots_.find(target);
        if (it == slots_.end())
            return;
        for (auto& slot : it->second.slots) {
            if (release)
                release(slot);
            slot = {};
        }
    }
    bool OutputSlotRing::releaseRenderTarget(RenderTargetId target, const PerSlotFn& release) {
        if (!target.valid() || released(target))
            return false;
        clearLogical(target, release);
        slots_.erase(target);
        released_.insert(target);
        return true;
    }
    void OutputSlotRing::reset() noexcept {
        slots_.clear();
        completions_.clear();
        transient_cells_.clear();
    }
    std::uint64_t OutputSlotRing::ringCompletionValue(std::size_t cell) const noexcept {
        return cell < completions_.size() ? completions_[cell] : 0;
    }
    std::uint64_t OutputSlotRing::generation(RenderTargetId target) const noexcept {
        auto it = slots_.find(target);
        return it == slots_.end() ? 0 : it->second.generation;
    }
} // namespace lfs::vis
