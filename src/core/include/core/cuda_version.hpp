/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::core {

    // CUDA 12.8 minimum (NVIDIA driver 570+)
    // Format: major * 1000 + minor * 10
    constexpr int MIN_CUDA_VERSION = 12080;

    struct CudaVersionInfo {
        int driver_version = 0;
        int major = 0;
        int minor = 0;
        bool supported = false;
        bool query_failed = false;
    };

    CudaVersionInfo check_cuda_version();

} // namespace lfs::core
