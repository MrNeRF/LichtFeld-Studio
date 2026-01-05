/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/cuda_stream_pool.hpp"
#include "internal/cuda_stream_context.hpp"

namespace lfs::core {

    PooledStreamGuard::PooledStreamGuard(bool high_priority)
        : prev_stream_(CUDAStreamContext::instance().getCurrentStream()) {

        if (high_priority) {
            stream_ = CUDAStreamPool::instance().acquire_high_priority();
        } else {
            stream_ = CUDAStreamPool::instance().acquire();
        }

        if (stream_) {
            CUDAStreamContext::instance().setCurrentStream(stream_);
        }
    }

    PooledStreamGuard::~PooledStreamGuard() {
        CUDAStreamContext::instance().setCurrentStream(prev_stream_);
    }

} // namespace lfs::core
