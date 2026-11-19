/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/shared_image_ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_stream_context.hpp"

namespace lfs::core::internal {
    Tensor CudaBackendOps::image_warp_region(const Tensor& input, const UndistortParams& params, int x, int y,
                                             int width, int height, ExecContext context) {
        return cuda::undistort_image_region(input, params, x, y, width, height, context.cuda_stream);
    }

    Tensor CudaBackendOps::image_warp(const Tensor& input, const UndistortParams& p, int mode, bool inverse,
                                      Tensor* validity, ExecContext execution) {
        const GpuBackendScope scope(GpuBackend::CUDA);
        const auto stream = execution.cuda_stream;
        switch (mode) {
        case 0:
            if (inverse) {
                Tensor unused;
                return cuda::distort_image_to_source(input, p, validity ? *validity : unused, stream);
            }
            return cuda::undistort_image(input, p, stream);
        case 1: return inverse ? cuda::distort_mask_to_source_area(input, p, stream) : cuda::undistort_mask_area(input, p, stream);
        case 2: return inverse ? cuda::distort_depth_to_source_area(input, p, stream) : cuda::undistort_depth_area(input, p, stream);
        case 3: return inverse ? cuda::distort_normal_to_source_area(input, p, stream) : cuda::undistort_normal_area(input, p, stream);
        case 4: return cuda::inverse_distortion_sample_map(p, stream);
        default: throw std::invalid_argument("Unknown image warp mode");
        }
    }

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
