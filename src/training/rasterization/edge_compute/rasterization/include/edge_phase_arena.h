/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>

namespace edge_compute::rasterization {

    // A small forward-only phase view over the shared raster arena. Keeping
    // the phase cursor separate makes the V-sized geometry and exact I-sized
    // sort allocation one contiguous, 256-byte-aligned forward workspace.
    class EdgePhaseArena {
    public:
        explicit EdgePhaseArena(std::function<char*(size_t)> backing)
            : backing_(std::move(backing)) {}

        void begin(size_t minimum_bytes) {
            if (started_)
                throw std::logic_error("EDGE forward phase may only be begun once");
            capacity_ = align_size(minimum_bytes);
            base_ = capacity_ > 0 ? backing_(capacity_) : nullptr;
            if (capacity_ > 0 && base_ == nullptr)
                throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate EDGE phase workspace");
            started_ = true;
        }

        char* allocate(size_t bytes) {
            LFS_ASSERT_MSG(started_, "EDGE phase allocation requires an active phase");
            if (bytes == 0)
                return nullptr;
            const size_t aligned = align_size(bytes);
            if (cursor_ > capacity_ || aligned > capacity_ - cursor_) {
                if (cursor_ > std::numeric_limits<size_t>::max() - aligned)
                    throw std::overflow_error("EDGE phase workspace size overflow");
                const size_t required = align_size(cursor_ + aligned);
                const size_t extension = required > capacity_ ? required - capacity_ : 0;
                char* next = extension > 0 ? backing_(extension) : nullptr;
                if (extension > 0 && next == nullptr)
                    throw std::runtime_error("OUT_OF_MEMORY: Failed to extend EDGE phase workspace");
                if (base_ != nullptr && next != base_ + capacity_)
                    throw std::runtime_error("EDGE phase workspace extension was not contiguous");
                if (base_ == nullptr)
                    base_ = next;
                capacity_ = required;
            }
            char* result = base_ + cursor_;
            cursor_ += aligned;
            return result;
        }

        std::function<char*(size_t)> allocator() {
            return [this](size_t bytes) { return allocate(bytes); };
        }

    private:
        static size_t align_size(size_t bytes) {
            constexpr size_t alignment = 256;
            if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1))
                throw std::overflow_error("EDGE phase workspace alignment overflow");
            return (bytes + alignment - 1) & ~(alignment - 1);
        }

        std::function<char*(size_t)> backing_;
        char* base_ = nullptr;
        size_t capacity_ = 0;
        size_t cursor_ = 0;
        bool started_ = false;
    };

} // namespace edge_compute::rasterization
