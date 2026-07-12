/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Common.h"

#include <atomic>

namespace gsplat_lfs {

    namespace {
        std::atomic_bool force_cuda_allocation_failure{false};
    }

    void set_cuda_allocation_failure_for_testing(const bool fail) {
        force_cuda_allocation_failure.store(fail, std::memory_order_relaxed);
    }

    bool cuda_allocation_failure_is_forced() {
        return force_cuda_allocation_failure.load(std::memory_order_relaxed);
    }

} // namespace gsplat_lfs
