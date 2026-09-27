/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "core/shared_image_ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_stream_context.hpp"

namespace lfs::core::internal {
    Tensor CudaBackendOps::image_undistort(const Tensor& input, const UndistortParams& params, bool mask, ExecContext context) {
        LFS_FACADE_TRACE(image_undistort);
        const CUDAStreamGuard guard(context.cuda_stream);
        return shared_image_ops(GpuBackend::CUDA)->undistort(input, params, mask);
    }
    Tensor CudaBackendOps::image_resize_prior(const Tensor& input, int height, int width, bool normal, ExecContext context) {
        LFS_FACADE_TRACE(image_resize_prior);
        const CUDAStreamGuard guard(context.cuda_stream);
        return shared_image_ops(GpuBackend::CUDA)->resize(input, height, width, normal ? gpu_ops::Resample::NormalPrior : gpu_ops::Resample::DepthPrior, 2);
    }
} // namespace lfs::core::internal
