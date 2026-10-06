/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "mesh_mask.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include <stdexcept>
namespace lfs::training {
    namespace {
        auto mesh_ops() {
            const auto* ops = training_ops(core::default_gpu_backend()).masks;
            if (!ops || !ops->mesh_coverage)
                throw std::runtime_error("Evaluation mesh coverage is unavailable on the selected backend");
            return ops->mesh_coverage;
        }
    } // namespace
    core::Tensor rasterize_mesh_coverage(const core::Tensor& vertices, const core::Tensor& indices,
                                         const MeshMaskCamera& camera, float z_near, void*) {
        return mesh_ops()(vertices, indices, camera, {}, nullptr, z_near);
    }
    core::Tensor rasterize_mesh_coverage(const core::Tensor& vertices, const core::Tensor& indices,
                                         const MeshMaskCamera& camera, const core::Tensor& samples,
                                         const core::UndistortParams& distortion, float z_near, void*) {
        return mesh_ops()(vertices, indices, camera, samples, &distortion, z_near);
    }
    core::Tensor splat_point_coverage(const core::Tensor& means, const MeshMaskCamera& camera,
                                      int radius, int close, const core::UndistortParams* distortion, void*) {
        LFS_ASSERT_MSG(means.is_valid() && means.ndim() == 2 && means.shape()[1] == 3 &&
                           means.device() == core::Device::GPU && means.dtype() == core::DataType::Float32 &&
                           camera.width > 0 && camera.height > 0 && radius >= 0 && close >= 0,
                       "Point coverage requires GPU Float32 [N,3], positive dimensions and nonnegative radii");
        const auto* ops = training_ops(core::default_gpu_backend()).masks;
        if (!ops || !ops->point_coverage)
            throw std::runtime_error("Evaluation point coverage is unavailable on the selected backend");
        auto mask = ops->point_coverage(means, camera, radius, distortion);
        for (const bool dilate : {true, false}) {
            for (const int axis : {1, 0}) {
                auto output = mask.clone();
                const int extent = static_cast<int>(mask.shape()[axis]);
                for (int offset = 1; offset <= std::min(close, extent - 1); ++offset) {
                    for (const int direction : {-1, 1}) {
                        const int dst = direction > 0 ? offset : 0;
                        const int src = direction > 0 ? 0 : offset;
                        auto target = output.slice(axis, dst, dst + extent - offset);
                        const auto shifted = mask.slice(axis, src, src + extent - offset);
                        target.copy_(dilate ? target.maximum(shifted) : target.minimum(shifted));
                    }
                }
                mask = std::move(output);
            }
        }
        return mask;
    }
} // namespace lfs::training
