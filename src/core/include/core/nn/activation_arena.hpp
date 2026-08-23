/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

namespace lfs::core::nn {

    // Bump allocator for NN activations. First forward records the high-water
    // mark via the pool; end() then commits one persistent buffer. Later
    // forwards reset the bump pointer and allocate views over that buffer.
    class LFS_CORE_API ActivationArena {
    public:
        void begin(cudaStream_t stream);
        void end();
        [[nodiscard]] std::size_t mark() const { return used_; }
        void rewind(std::size_t to);

        // Returns a pointer from the committed buffer, or nullptr while
        // recording (caller must fall back to the pool).
        void* try_alloc(std::size_t bytes);

        [[nodiscard]] std::shared_ptr<void> owner() const { return owner_; }
        [[nodiscard]] std::size_t high_water() const { return high_water_; }
        [[nodiscard]] std::size_t capacity() const { return cap_; }

        static ActivationArena* current() noexcept;
        static void bind(ActivationArena* arena) noexcept;

    private:
        void commit();

        std::shared_ptr<void> owner_;
        char* base_ = nullptr;
        std::size_t cap_ = 0;
        std::size_t used_ = 0;
        std::size_t high_water_ = 0;
        cudaStream_t stream_ = nullptr;
    };

    class LFS_CORE_API ActivationArenaGuard {
    public:
        explicit ActivationArenaGuard(ActivationArena& arena);
        ~ActivationArenaGuard();
        ActivationArenaGuard(const ActivationArenaGuard&) = delete;
        ActivationArenaGuard& operator=(const ActivationArenaGuard&) = delete;

    private:
        ActivationArena* prev_ = nullptr;
    };

} // namespace lfs::core::nn
