/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <core/tensor_vulkan_interop.hpp>
#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace lfs::vis {
    // Publications are immutable snapshots. Invalidation drops only our
    // reference; submitted frames retain their own leases until completion.
    // The renderer owns content invalidation and bypasses this cache while
    // training can modify raw native pointers.
    class HipInputPublications {
        using Slot = std::array<std::optional<core::TensorVulkanBuffer>, 8>;
        std::vector<Slot> slots_;
        std::uint64_t budget_;
        std::uint64_t cached_bytes_ = 0;

        static std::uint64_t charge(const core::TensorVulkanBuffer& buffer) {
            return buffer.bytes < 256 ? 256 : buffer.bytes;
        }
        void drop(Slot& slot) {
            for (const auto& entry : slot)
                if (entry)
                    cached_bytes_ -= charge(*entry);
            slot = {};
        }

    public:
        // Bound retained publication payloads across all view targets. This is
        // a cache limit, not a replacement for the driver's memory budget.
        explicit HipInputPublications(std::uint64_t budget = 256ull * 1024 * 1024) : budget_(budget) {}
        void resize(std::size_t count) {
            for (auto i = count; i < slots_.size(); ++i)
                drop(slots_[i]);
            slots_.resize(count);
        }
        void clear() {
            slots_.clear();
            cached_bytes_ = 0;
        }
        void invalidate() {
            for (auto& slot : slots_)
                drop(slot);
        }
        void invalidate(std::size_t slot) { drop(slots_.at(slot)); }

        template <class Load>
        std::optional<core::TensorVulkanBuffer> get(std::size_t slot, std::size_t field, Load&& load) {
            auto& entry = slots_.at(slot).at(field);
            if (!entry) {
                auto loaded = load();
                if (!loaded || charge(*loaded) > budget_ - cached_bytes_)
                    return loaded;
                cached_bytes_ += charge(*loaded);
                entry = std::move(loaded);
            }
            return entry;
        }
    };
} // namespace lfs::vis
