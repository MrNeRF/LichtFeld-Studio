/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Common.h"

#ifdef LFS_ENABLE_CUDA_FAILURE_INJECTION
#include <atomic>
#endif

namespace gsplat_lfs {

#ifdef LFS_ENABLE_CUDA_FAILURE_INJECTION
    namespace {
        std::atomic_bool force_cuda_allocation_failure{false};
    }

    void set_cuda_allocation_failure_for_testing(const bool fail) {
        force_cuda_allocation_failure.store(fail, std::memory_order_release);
    }

    bool cuda_allocation_failure_is_forced() {
        return force_cuda_allocation_failure.load(std::memory_order_acquire);
    }
#else
    void set_cuda_allocation_failure_for_testing(const bool) {}
#endif

} // namespace gsplat_lfs
