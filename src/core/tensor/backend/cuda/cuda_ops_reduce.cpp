/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../gpu_backend_ops.hpp"

#include "../../internal/memory_pool.hpp"
#include "../../internal/tensor_impl.hpp"
#include "../../internal/tensor_ops.hpp"
#include "core/assert.hpp"
#include "core/cuda_error.hpp"

#include <cuda_runtime.h>
#include <type_traits>

namespace lfs::core::internal {
    namespace {
        template <class T>
        T* cuda_pointer(const StorageRef storage) {
            LFS_ASSERT_MSG(storage.backend == GpuBackend::CUDA,
                           "CUDA reduction adapter received non-CUDA storage");
            return reinterpret_cast<T*>(
                static_cast<unsigned char*>(storage.data) + storage.byte_offset);
        }

        template <class T>
        const T* cuda_const_pointer(const StorageRef storage) {
            return cuda_pointer<const T>(storage);
        }

        void prime_stream_polling(const ExecContext context) {
            // Keep blocking scalar reductions on the active polling path without
            // rediscovering pointer ownership for every internal launch.
            const cudaError_t status = cudaStreamQuery(context.cuda_stream);
            if (status != cudaSuccess && status != cudaErrorNotReady) {
                LFS_CUDA_CHECK(status);
            }
        }

        template <class T>
        size_t count_nonzero(const StorageRef input, const size_t count,
                             const ExecContext context) {
            size_t result = 0;
            size_t* device_result = static_cast<size_t*>(
                CudaMemoryPool::instance().allocate(sizeof(size_t), context.cuda_stream));
            LFS_ASSERT_MSG(device_result != nullptr,
                           "count_nonzero failed to allocate pooled device counter");
            LFS_CUDA_CHECK(cudaMemsetAsync(
                device_result, 0, sizeof(size_t), context.cuda_stream));
            if constexpr (std::is_same_v<T, unsigned char>) {
                tensor_ops::launch_count_nonzero_bool(
                    cuda_const_pointer<unsigned char>(input), device_result,
                    count, context.cuda_stream);
            } else {
                tensor_ops::launch_count_nonzero_float(
                    cuda_const_pointer<float>(input), device_result,
                    count, context.cuda_stream);
            }
            LFS_CUDA_CHECK(cudaStreamSynchronize(context.cuda_stream));
            LFS_CUDA_CHECK(cudaMemcpy(
                &result, device_result, sizeof(size_t), cudaMemcpyDeviceToHost));
            CudaMemoryPool::instance().deallocate(device_result, context.cuda_stream);
            return result;
        }
    } // namespace

    float CudaBackendOps::sum_scalar(
        const StorageRef input, const size_t count, const ExecContext context) {
        prime_stream_polling(context);
        return tensor_ops::direct_sum_scalar(
            cuda_const_pointer<float>(input), count, context.cuda_stream);
    }

    float CudaBackendOps::mean_scalar(
        const StorageRef input, const size_t count, const ExecContext context) {
        prime_stream_polling(context);
        return tensor_ops::direct_mean_scalar(
            cuda_const_pointer<float>(input), count, context.cuda_stream);
    }

    float CudaBackendOps::max_scalar(
        const StorageRef input, const size_t count, const ExecContext context) {
        prime_stream_polling(context);
        return tensor_ops::direct_max_scalar(
            cuda_const_pointer<float>(input), count, context.cuda_stream);
    }

    float CudaBackendOps::min_scalar(
        const StorageRef input, const size_t count, const ExecContext context) {
        prime_stream_polling(context);
        return tensor_ops::direct_min_scalar(
            cuda_const_pointer<float>(input), count, context.cuda_stream);
    }

    void CudaBackendOps::reduce(
        const StorageRef input, const StorageRef output,
        const StridedLayout& input_layout, const ReduceProgram& program,
        const ExecContext context) {
        LFS_ASSERT_MSG(program.axis_count <= MAX_TENSOR_RANK,
                       "reduction axis count exceeds MAX_TENSOR_RANK");
        tensor_ops::launch_reduce_op(
            cuda_const_pointer<void>(input), cuda_pointer<void>(output),
            input_layout.dims.data(), input_layout.rank,
            program.axes.data(), program.axis_count, program.keepdim, program.op,
            input.dtype, program.result_dtype, context.cuda_stream);
    }

    void CudaBackendOps::column_reduce(
        const StorageRef input, const StorageRef output, const size_t rows,
        const size_t columns, const ReduceProgram& program,
        const ExecContext context) {
        tensor_ops::launch_column_reduce(
            cuda_const_pointer<float>(input), cuda_pointer<float>(output),
            rows, columns, program.op, context.cuda_stream);
    }

    void CudaBackendOps::strided_reduce(
        const StorageRef input, const StorageRef output, const size_t outer_size,
        const size_t reduce_size, const size_t inner_size,
        const ReduceProgram& program,
        const ExecContext context) {
        tensor_ops::launch_strided_reduce_fast(
            cuda_const_pointer<float>(input), cuda_pointer<float>(output),
            outer_size, reduce_size, inner_size, program.op, context.cuda_stream);
    }

    void CudaBackendOps::fused_transform_reduce(
        const StorageRef input, const StorageRef output, const size_t count,
        const tensor_ops::FusedPointwiseOpChain& chain,
        const ReduceProgram& program,
        const ExecContext context) {
        tensor_ops::launch_fused_transform_reduce(
            cuda_const_pointer<float>(input), cuda_pointer<float>(output), count,
            chain, program.op, context.cuda_stream);
    }

    void CudaBackendOps::fused_segmented_transform_reduce(
        const StorageRef input, const StorageRef output, const size_t segment_count,
        const size_t segment_size, const tensor_ops::FusedPointwiseOpChain& chain,
        const ReduceProgram& program, const ExecContext context) {
        tensor_ops::launch_fused_segmented_transform_reduce(
            cuda_const_pointer<float>(input), cuda_pointer<float>(output),
            segment_count, segment_size, chain, program.op, context.cuda_stream);
    }

    size_t CudaBackendOps::count_nonzero_bool(
        const StorageRef input, const size_t count, const ExecContext context) {
        return count_nonzero<unsigned char>(input, count, context);
    }

    size_t CudaBackendOps::count_nonzero_float(
        const StorageRef input, const size_t count, const ExecContext context) {
        return count_nonzero<float>(input, count, context);
    }

    bool CudaBackendOps::has_nan(
        const StorageRef input, const size_t count, const ExecContext context) {
        prime_stream_polling(context);
        return tensor_ops::has_nan_gpu(
            cuda_const_pointer<float>(input), count, context.cuda_stream);
    }

    bool CudaBackendOps::has_inf(
        const StorageRef input, const size_t count, const ExecContext context) {
        prime_stream_polling(context);
        return tensor_ops::has_inf_gpu(
            cuda_const_pointer<float>(input), count, context.cuda_stream);
    }

} // namespace lfs::core::internal
