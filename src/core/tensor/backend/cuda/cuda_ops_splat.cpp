/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "core/cuda/splat_transform.hpp"
namespace lfs::core::internal {
    void CudaBackendOps::affine_splat_geometry(StorageRef scales, StorageRef rotations, StorageRef out_scales, StorageRef out_rotations,
                                               const splat_transform::LinearTransform& linear, size_t n, ExecContext context) {
        LFS_FACADE_TRACE(affine_splat_geometry);
        const auto data = [](StorageRef s) { return reinterpret_cast<float*>(static_cast<char*>(s.data) + s.byte_offset); };
        cuda::transform_splat_geometry(linear, data(scales), data(rotations), data(out_scales), data(out_rotations), n, context.cuda_stream);
    }
} // namespace lfs::core::internal
