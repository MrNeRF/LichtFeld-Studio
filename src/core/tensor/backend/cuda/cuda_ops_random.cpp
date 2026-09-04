/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../gpu_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "../../internal/tensor_ops.hpp"
#include "core/assert.hpp"
#include "core/cuda_error.hpp"

#include <cuda_runtime.h>

namespace lfs::core::internal {
    namespace {
        template <class T>
        T* cuda_pointer(const StorageRef storage) {
            LFS_ASSERT_MSG(storage.backend == GpuBackend::CUDA,
                           "CUDA random adapter received non-CUDA storage");
            return reinterpret_cast<T*>(
                static_cast<unsigned char*>(storage.data) + storage.byte_offset);
        }

        template <class T>
        const T* cuda_const_pointer(const StorageRef storage) {
            return cuda_pointer<const T>(storage);
        }
    } // namespace

    void CudaBackendOps::uniform(
        const StorageRef output, const RandomProgram& program,
        const ExecContext context) {
        tensor_ops::launch_uniform(
            cuda_pointer<float>(output), program.count, program.first,
            program.second, program.seed, context.cuda_stream);
    }

    void CudaBackendOps::bernoulli(
        const StorageRef output, const RandomProgram& program,
        const ExecContext context) {
        tensor_ops::launch_bernoulli(
            cuda_pointer<float>(output), program.count, program.first,
            program.seed, context.cuda_stream);
    }

    void CudaBackendOps::randint(
        const StorageRef output, const RandomProgram& program,
        const ExecContext context) {
        tensor_ops::launch_randint(
            cuda_pointer<int>(output), program.count, program.low, program.high,
            program.seed, context.cuda_stream);
    }

    void CudaBackendOps::multinomial(
        const StorageRef weights, const StorageRef output,
        const RandomProgram& program, const ExecContext context) {
        tensor_ops::launch_multinomial(
            cuda_const_pointer<float>(weights), cuda_pointer<int64_t>(output),
            program.count, program.sample_count, program.replacement,
            program.seed, context.cuda_stream);
    }

    void CudaBackendOps::normal(
        const StorageRef output, const StorageRef odd_count_scratch,
        const RandomProgram& program, const ExecContext context) {
        if (program.count % 2 == 0) {
            RandomGenerator::instance().generate_cuda_normal(
                cuda_pointer<float>(output), program.count, program.first,
                program.second, context.cuda_stream);
            return;
        }

        RandomGenerator::instance().generate_cuda_normal(
            cuda_pointer<float>(odd_count_scratch), program.count + 1,
            program.first, program.second, context.cuda_stream);
        LFS_CUDA_CHECK(cudaMemcpyAsync(
            cuda_pointer<float>(output), cuda_pointer<float>(odd_count_scratch),
            program.count * sizeof(float), cudaMemcpyDeviceToDevice,
            context.cuda_stream));
        LFS_CUDA_CHECK(cudaStreamSynchronize(context.cuda_stream));
    }

} // namespace lfs::core::internal
