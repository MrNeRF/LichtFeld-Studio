/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_cuda_interop.hpp"
#include "native_pressure.hpp"
#include <stdexcept>
namespace lfs::core {
    void* allocate_cuda_storage(const size_t bytes,
                                const cudaStream_t stream,
                                const CudaStorageMode mode,
                                const char* const label,
                                const char* const operation) {

        throw std::runtime_error("CUDA support is not compiled into this build");
    }

} // namespace lfs::core
namespace lfs::core::internal {
    void record_pressure_breadcrumb() {}
} // namespace lfs::core::internal
