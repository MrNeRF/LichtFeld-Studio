/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda/memory_arena.hpp"
#include "core/cuda_error.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training {

    // Borrows the rasterizer arena's committed memory between training frames,
    // when no rasterizer or viewer frame owns it. Takes up to `wanted` bytes
    // without growing the arena, or nothing when a frame is active or fewer
    // than `minimum` bytes are committed. All use must be enqueued on `stream`
    // and finish before destruction, whose end_frame orders the next frame
    // after that work.
    class IdleArenaScratch {
    public:
        IdleArenaScratch(const std::size_t wanted, const std::size_t minimum, cudaStream_t stream)
            : stream_(stream) {
            auto* arena = core::GlobalArenaManager::instance().try_get_arena();
            if (arena == nullptr || wanted == 0) {
                return;
            }
            const std::size_t bytes = std::min(wanted, arena->get_memory_info().arena_capacity);
            if (bytes == 0 || bytes < minimum) {
                return;
            }
            const auto frame = arena->try_begin_frame(stream);
            if (!frame) {
                return;
            }
            arena_ = arena;
            frame_ = *frame;
            data_ = arena_->get_allocator(frame_, "training.idle_scratch")(bytes);
            bytes_ = data_ != nullptr ? bytes : 0;
        }

        ~IdleArenaScratch() {
            if (arena_ != nullptr) {
                arena_->end_frame(frame_, stream_);
            }
        }

        IdleArenaScratch(const IdleArenaScratch&) = delete;
        IdleArenaScratch& operator=(const IdleArenaScratch&) = delete;

        [[nodiscard]] char* data() const { return data_; }
        [[nodiscard]] std::size_t capacity() const { return bytes_; }

        // The first `bytes` of the borrow, cleared on the scratch stream, or
        // null when the borrow is smaller.
        [[nodiscard]] void* zeroed(const std::size_t bytes) const {
            if (bytes == 0 || bytes > bytes_) {
                return nullptr;
            }
            LFS_CUDA_CHECK(cudaMemsetAsync(data_, 0, bytes, stream_));
            return data_;
        }

    private:
        core::RasterizerMemoryArena* arena_ = nullptr;
        cudaStream_t stream_ = nullptr;
        std::uint64_t frame_ = 0;
        char* data_ = nullptr;
        std::size_t bytes_ = 0;
    };

} // namespace lfs::training
