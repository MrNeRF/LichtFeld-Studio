/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "core/assert.hpp"
#include "kernels/tensor_histogram.hpp"

namespace lfs::core::internal {
    void CudaBackendOps::histogram_u8(const StorageRef values, const StorageRef counts,
                                      const size_t size, const ExecContext context) {
        LFS_FACADE_TRACE(histogram_u8);
        LFS_ASSERT_MSG(values.backend == GpuBackend::CUDA && counts.backend == GpuBackend::CUDA,
                       "CUDA histogram requires CUDA storage");
        const auto* input = static_cast<const uint8_t*>(values.data) + values.byte_offset;
        auto* output = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(counts.data) + counts.byte_offset);
        tensor_ops::launch_histogram_u8(input, output, size, context.cuda_stream);
    }
} // namespace lfs::core::internal
