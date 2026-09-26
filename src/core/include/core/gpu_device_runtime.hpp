/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor_fwd.hpp"

#include <optional>
namespace lfs::core {

    LFS_CORE_API void gpu_device_barrier(GpuBackend backend);
    LFS_CORE_API int gpu_device_count(GpuBackend backend);
    // Allocation ownership is not currently available through the public tensor metadata.
    LFS_CORE_API std::optional<std::size_t> reserved_allocation_bytes(const Tensor& tensor);

} // namespace lfs::core
