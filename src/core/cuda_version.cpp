/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_version.hpp"
#include <cuda_runtime.h>

namespace lfs::core {

    CudaVersionInfo check_cuda_version() {
        CudaVersionInfo info;

        if (cudaDriverGetVersion(&info.driver_version) != cudaSuccess) {
            info.query_failed = true;
            return info;
        }

        info.major = info.driver_version / 1000;
        info.minor = (info.driver_version % 1000) / 10;
        info.supported = info.driver_version >= MIN_CUDA_VERSION;

        return info;
    }

} // namespace lfs::core
