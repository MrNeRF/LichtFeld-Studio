/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"
#include <cstdint>
#include <optional>
namespace lfs::core {
    struct TensorExportCapabilities {
        bool dlpack = false;
    };
    // Numeric device types are the DLPack protocol values; no DLPack headers
    // or GPU SDK are needed by consumers of this Tensor contract.
    struct TensorExportDevice {
        int32_t type = 1;
        int32_t ordinal = 0;
    };
    LFS_CORE_API TensorExportCapabilities gpu_backend_export_capabilities(GpuBackend);
    LFS_CORE_API TensorExportDevice tensor_dlpack_device(const Tensor&);
    // nullopt synchronizes for an unspecified consumer; -1 skips synchronization.
    // Other integers retain the DLPack legacy/per-thread/native stream encoding.
    LFS_CORE_API Tensor prepare_tensor_dlpack_export(const Tensor&, std::optional<int64_t> consumer_stream);
} // namespace lfs::core
