/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/activation_arena.hpp"

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "internal/memory_pool.hpp"

namespace lfs::core::nn {
    namespace {

        thread_local ActivationArena* t_current = nullptr;

        constexpr std::size_t kAlign = 256;

        std::size_t align_up(const std::size_t bytes) {
            return (bytes + (kAlign - 1)) & ~(kAlign - 1);
        }

    } // namespace

    ActivationArena* ActivationArena::current() noexcept {
        return t_current;
    }

    void ActivationArena::bind(ActivationArena* arena) noexcept {
        t_current = arena;
    }

    void ActivationArena::begin(const cudaStream_t stream) {
        stream_ = stream;
        used_ = 0;
    }

    void ActivationArena::rewind(const std::size_t to) {
        used_ = to;
    }

    void ActivationArena::end() {
        if (used_ > high_water_) {
            high_water_ = used_;
        }
        if (cap_ == 0 && high_water_ > 0) {
            lfs::core::CudaMemoryPool::instance().trim();
            commit();
        }
    }

    void ActivationArena::commit() {
        std::size_t bytes = align_up(high_water_ < kAlign ? kAlign : high_water_);
        void* ptr = allocate_cuda_storage(bytes, stream_);
        LFS_ASSERT_MSG(ptr != nullptr, "activation arena failed to allocate");
        const cudaStream_t s = stream_;
        owner_ = std::shared_ptr<void>(ptr, [s](void* p) { safe_cuda_pool_deallocate(p, s); });
        base_ = static_cast<char*>(ptr);
        cap_ = bytes;
        used_ = 0;
    }

    void* ActivationArena::try_alloc(const std::size_t bytes) {
        if (bytes == 0) {
            return nullptr;
        }
        const std::size_t aligned = align_up(bytes);
        if (cap_ == 0) {
            used_ += aligned;
            if (used_ > high_water_) {
                high_water_ = used_;
            }
            return nullptr;
        }
        if (used_ + aligned > cap_) {
            used_ += aligned;
            if (used_ > high_water_) {
                high_water_ = used_;
            }
            return nullptr;
        }
        void* ptr = base_ + used_;
        used_ += aligned;
        return ptr;
    }

    ActivationArenaGuard::ActivationArenaGuard(ActivationArena& arena)
        : prev_(ActivationArena::current()) {
        ActivationArena::bind(&arena);
    }

    ActivationArenaGuard::~ActivationArenaGuard() {
        ActivationArena::bind(prev_);
    }

} // namespace lfs::core::nn
