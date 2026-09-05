/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "../../internal/tensor_nn_ops.hpp"
#include "../../internal/tensor_ops.hpp"
#include "core/assert.hpp"

namespace lfs::core::internal {
    namespace {
        template <class T>
        T* cuda_pointer(const StorageRef storage) {
            LFS_ASSERT_MSG(storage.backend == GpuBackend::CUDA,
                           "CUDA matrix/NN adapter received non-CUDA storage");
            return reinterpret_cast<T*>(
                static_cast<unsigned char*>(storage.data) + storage.byte_offset);
        }

        template <class T>
        const T* cuda_const_pointer(const StorageRef storage) {
            return cuda_pointer<const T>(storage);
        }
    } // namespace

    void CudaBackendOps::sgemm(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const GemmProgram& program, const ExecContext context) {
        LFS_FACADE_TRACE(sgemm);
        tensor_ops::launch_sgemm(
            cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
            cuda_pointer<float>(output), program.m, program.n, program.k,
            context.cuda_stream);
    }

    void CudaBackendOps::sgemm_tn(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const GemmProgram& program, const ExecContext context) {
        LFS_FACADE_TRACE(sgemm_tn);
        tensor_ops::launch_sgemm_tn(
            cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
            cuda_pointer<float>(output), program.m, program.n, program.k,
            context.cuda_stream);
    }

    void CudaBackendOps::sgemm_batched(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const GemmProgram& program, const ExecContext context) {
        LFS_FACADE_TRACE(sgemm_batched);
        tensor_ops::launch_sgemm_batched(
            cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
            cuda_pointer<float>(output), program.batch, program.m, program.n,
            program.k, context.cuda_stream);
    }

    void CudaBackendOps::sgemm_bias_relu(
        const StorageRef lhs, const StorageRef rhs, const StorageRef bias,
        const StorageRef output, const GemmProgram& program,
        const ExecContext context) {
        LFS_FACADE_TRACE(sgemm_bias_relu);
        tensor_ops::launch_sgemm_bias_relu(
            cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
            cuda_const_pointer<float>(bias), cuda_pointer<float>(output),
            program.m, program.n, program.k, context.cuda_stream);
    }

    void CudaBackendOps::dot_product(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const size_t count, const ExecContext context) {
        LFS_FACADE_TRACE(dot_product);
        tensor_ops::launch_dot_product(
            cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
            cuda_pointer<float>(output), count, context.cuda_stream);
    }

    void CudaBackendOps::diag(
        const StorageRef diagonal, const StorageRef output, const size_t count,
        const ExecContext context) {
        LFS_FACADE_TRACE(diag);
        tensor_ops::launch_diag(
            cuda_const_pointer<float>(diagonal), cuda_pointer<float>(output),
            count, context.cuda_stream);
    }

    void CudaBackendOps::eye(
        const StorageRef output, const size_t rows, const size_t columns,
        const ExecContext context) {
        LFS_FACADE_TRACE(eye);
        tensor_ops::launch_eye(
            cuda_pointer<float>(output), rows, columns, context.cuda_stream);
    }

    void CudaBackendOps::cdist(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const size_t lhs_rows, const size_t rhs_rows, const size_t columns,
        const float p, const ExecContext context) {
        LFS_FACADE_TRACE(cdist);
        tensor_ops::launch_cdist(
            cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
            cuda_pointer<float>(output), lhs_rows, rhs_rows, columns, p,
            context.cuda_stream);
    }

    void CudaBackendOps::max_pool2d(
        const StorageRef input, const StorageRef output,
        const PoolProgram& program, const ExecContext context) {
        LFS_FACADE_TRACE(max_pool2d);
        tensor_ops::launch_max_pool2d(
            cuda_const_pointer<float>(input), cuda_pointer<float>(output),
            program.batch, program.channels, program.input_height, program.input_width,
            program.output_height, program.output_width, program.kernel_size,
            program.stride, program.padding, context.cuda_stream);
    }

    void CudaBackendOps::adaptive_avg_pool2d(
        const StorageRef input, const StorageRef output,
        const PoolProgram& program, const ExecContext context) {
        LFS_FACADE_TRACE(adaptive_avg_pool2d);
        tensor_ops::launch_adaptive_avg_pool2d(
            cuda_const_pointer<float>(input), cuda_pointer<float>(output),
            program.batch, program.channels, program.input_height, program.input_width,
            program.output_height, program.output_width, context.cuda_stream);
    }

    void CudaBackendOps::bias_add(
        const StorageRef input, const StorageRef bias, const StorageRef output,
        const int count, const int channels, const int spatial_size,
        const ExecContext context) {
        LFS_FACADE_TRACE(bias_add);
        tensor_ops::launch_bias_add(
            cuda_const_pointer<float>(input), cuda_const_pointer<float>(bias),
            cuda_pointer<float>(output), count, channels, spatial_size,
            context.cuda_stream);
    }

    void CudaBackendOps::bias_relu(
        const StorageRef input, const StorageRef bias, const StorageRef output,
        const int count, const int channels, const int spatial_size,
        const ExecContext context) {
        LFS_FACADE_TRACE(bias_relu);
        tensor_ops::launch_bias_relu(
            cuda_const_pointer<float>(input), cuda_const_pointer<float>(bias),
            cuda_pointer<float>(output), count, channels, spatial_size,
            context.cuda_stream);
    }

    void CudaBackendOps::relu(
        const StorageRef input, const StorageRef output, const int count,
        const ExecContext context) {
        LFS_FACADE_TRACE(relu);
        tensor_ops::launch_relu(
            cuda_const_pointer<float>(input), cuda_pointer<float>(output),
            count, context.cuda_stream);
    }

} // namespace lfs::core::internal
