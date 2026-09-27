/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/session_cuda.hpp"

#include <cuda_profiler_api.h>

namespace lfs::training {
    namespace {

        void profile(const bool start) {
            if (start)
                cudaProfilerStart();
            else
                cudaProfilerStop();
        }

    } // namespace

    const lfs::gpu_ops::SessionOps& cuda_session_ops() {
        static const lfs::gpu_ops::SessionOps ops{.profile = profile};
        return ops;
    }

} // namespace lfs::training
