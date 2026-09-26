/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_stream_context.hpp"

namespace lfs::core::internal {
    Tensor CudaBackendOps::image_undistort(const Tensor& input, const UndistortParams& params, bool mask, ExecContext context) {
        LFS_FACADE_TRACE(image_undistort);
        const CUDAStreamGuard guard(context.cuda_stream);
        return mask ? core::undistort_mask(input, params, context.cuda_stream) : core::undistort_image(input, params, context.cuda_stream);
    }
    Tensor CudaBackendOps::image_resize_prior(const Tensor& input, int height, int width, bool normal, ExecContext context) {
        LFS_FACADE_TRACE(image_resize_prior);
        const CUDAStreamGuard guard(context.cuda_stream);
        return normal ? core::resize_normal_prior(input, height, width, context.cuda_stream) : core::resize_depth_prior(input, height, width, context.cuda_stream);
    }
} // namespace lfs::core::internal
