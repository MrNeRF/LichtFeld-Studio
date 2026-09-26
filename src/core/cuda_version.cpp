/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_version.hpp"
#if LFS_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace lfs::core {

    CudaVersionInfo check_cuda_version() {
        CudaVersionInfo info;

#if LFS_HAS_CUDA
        if (cudaDriverGetVersion(&info.driver_version) != cudaSuccess) {
            info.query_failed = true;
            return info;
        }

        info.major = info.driver_version / 1000;
        info.minor = (info.driver_version % 1000) / 10;
        info.supported = info.driver_version >= MIN_CUDA_VERSION;
#else
        info.query_failed = true;
#endif

        return info;
    }

    std::string get_pytorch_cuda_tag(const std::string& version_hint) {
        // Explicit version mapping
        if (version_hint == "12.8")
            return "cu128";
        if (version_hint == "12.4")
            return "cu124";
        if (version_hint == "12.1")
            return "cu121";
        if (version_hint == "11.8")
            return "cu118";

        // Already in tag format
        if (version_hint == "cu128" || version_hint == "cu124" ||
            version_hint == "cu121" || version_hint == "cu118") {
            return version_hint;
        }

        // Auto-detect from system
        const auto info = check_cuda_version();
        if (info.query_failed) {
            return "cu128"; // Default to latest
        }

        if (info.major >= 12) {
            if (info.minor >= 8)
                return "cu128";
            if (info.minor >= 4)
                return "cu124";
            return "cu121";
        }

        if (info.major == 11 && info.minor >= 8) {
            return "cu118";
        }

        return "cu118"; // Fallback for older CUDA
    }

} // namespace lfs::core
