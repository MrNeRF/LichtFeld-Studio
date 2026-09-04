/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../gpu_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "../../internal/tensor_ops.hpp"
#include "core/assert.hpp"

namespace lfs::core::internal {
    namespace {
        template <class T>
        T* cuda_pointer(const StorageRef storage) {
            LFS_ASSERT_MSG(storage.backend == GpuBackend::CUDA,
                           "CUDA scan/sort adapter received non-CUDA storage");
            return reinterpret_cast<T*>(
                static_cast<unsigned char*>(storage.data) + storage.byte_offset);
        }
    } // namespace

    void CudaBackendOps::cumsum(
        const StorageRef data, const StridedLayout& layout, const int dim,
        const ExecContext context) {
        tensor_ops::launch_cumsum(
            cuda_pointer<void>(data), layout.dims.data(), layout.rank,
            dim, data.dtype, context.cuda_stream);
    }

    void CudaBackendOps::sort_1d(
        const StorageRef values, const StorageRef indices, const size_t count,
        const SortProgram& program, const ExecContext context) {
        tensor_ops::launch_sort_1d(
            cuda_pointer<float>(values), cuda_pointer<int64_t>(indices),
            count, program.descending, context.cuda_stream);
    }

    void CudaBackendOps::sort_2d(
        const StorageRef values, const StorageRef indices,
        const SortProgram& program, const ExecContext context) {
        tensor_ops::launch_sort_2d(
            cuda_pointer<float>(values), cuda_pointer<int64_t>(indices),
            program.outer_size, program.dim_size, program.inner_size,
            program.dim, program.descending, context.cuda_stream);
    }

} // namespace lfs::core::internal
