/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "render_target_id.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::vis {
    // Each target owns three images. Submission indices remain unique even after
    // release, so a late readback cannot pin another target's transient storage.
    template <typename Payload>
    class BasicOutputSlotRing {
    public:
        static constexpr std::size_t kFrameRingSize = 3;
        using TimelineCompleteFn = std::function<bool(std::uint64_t)>;
        using TimelineWaitFn = std::function<lfs::Status(std::uint64_t)>;
        using PerSlotFn = std::function<void(Payload&)>;
        struct Column {
            std::array<Payload, kFrameRingSize> slots{};
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
        [[nodiscard]] Payload& slotAt(RenderTargetId target, std::size_t cell);
        [[nodiscard]] const Payload& slotAt(RenderTargetId target, std::size_t cell) const;
        [[nodiscard]] std::size_t latestRingSlot(RenderTargetId target) const;
        [[nodiscard]] Payload& latestSlot(RenderTargetId target);
        [[nodiscard]] const Payload& latestSlot(RenderTargetId target) const;
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

    template <typename Payload>
    std::size_t BasicOutputSlotRing<Payload>::acquire(RenderTargetId target) {
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
    template <typename Payload>
    std::size_t BasicOutputSlotRing<Payload>::acquireTransient(const TimelineCompleteFn& complete) {
        for (auto cell : transient_cells_) {
            if (!completions_[cell] || complete(completions_[cell]))
                return cell;
        }
        const auto cell = completions_.size();
        completions_.push_back(0);
        transient_cells_.push_back(cell);
        return cell;
    }
    template <typename Payload>
    lfs::Status BasicOutputSlotRing<Payload>::waitUntilReusable(const std::size_t ring_slot,
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

    template <typename Payload>
    void BasicOutputSlotRing<Payload>::publishCompletion(std::size_t cell, std::uint64_t value) noexcept {
        if (cell < completions_.size())
            completions_[cell] = value;
    }
    template <typename Payload>
    void BasicOutputSlotRing<Payload>::clearSlotCompletion(RenderTargetId target, std::size_t cell) {
        slotAt(target, cell).completion_value = 0;
    }
    template <typename Payload>
    void BasicOutputSlotRing<Payload>::markLatest(RenderTargetId target, std::size_t cell) {
        if (released(target))
            return;
        (void)slotAt(target, cell);
        slots_.at(target).latest = cell;
    }
    template <typename Payload>
    std::uint64_t BasicOutputSlotRing<Payload>::bumpGeneration(RenderTargetId target) {
        if (released(target))
            return 0;
        return ++slots_.at(target).generation;
    }
    template <typename Payload>
    Payload& BasicOutputSlotRing<Payload>::slotAt(RenderTargetId target, std::size_t cell) {
        auto& column = slots_.at(target);
        return column.slots.at(cell - column.base);
    }
    template <typename Payload>
    const Payload& BasicOutputSlotRing<Payload>::slotAt(RenderTargetId target, std::size_t cell) const {
        const auto& column = slots_.at(target);
        return column.slots.at(cell - column.base);
    }
    template <typename Payload>
    std::size_t BasicOutputSlotRing<Payload>::latestRingSlot(RenderTargetId target) const {
        return slots_.at(target).latest;
    }
    template <typename Payload>
    Payload& BasicOutputSlotRing<Payload>::latestSlot(RenderTargetId target) {
        return slotAt(target, latestRingSlot(target));
    }
    template <typename Payload>
    const Payload& BasicOutputSlotRing<Payload>::latestSlot(RenderTargetId target) const {
        static const Payload empty;
        return contains(target) ? slotAt(target, latestRingSlot(target)) : empty;
    }
    template <typename Payload>
    void BasicOutputSlotRing<Payload>::clearLogical(RenderTargetId target, const PerSlotFn& release) {
        auto it = slots_.find(target);
        if (it == slots_.end())
            return;
        for (auto& slot : it->second.slots) {
            if (release)
                release(slot);
            slot = {};
        }
    }
    template <typename Payload>
    bool BasicOutputSlotRing<Payload>::releaseRenderTarget(RenderTargetId target, const PerSlotFn& release) {
        if (!target.valid() || released(target))
            return false;
        clearLogical(target, release);
        slots_.erase(target);
        released_.insert(target);
        return true;
    }
    template <typename Payload>
    void BasicOutputSlotRing<Payload>::reset() noexcept {
        slots_.clear();
        completions_.clear();
        transient_cells_.clear();
    }
    template <typename Payload>
    std::uint64_t BasicOutputSlotRing<Payload>::ringCompletionValue(std::size_t cell) const noexcept {
        return cell < completions_.size() ? completions_[cell] : 0;
    }
    template <typename Payload>
    std::uint64_t BasicOutputSlotRing<Payload>::generation(RenderTargetId target) const noexcept {
        auto it = slots_.find(target);
        return it == slots_.end() ? 0 : it->second.generation;
    }
} // namespace lfs::vis
