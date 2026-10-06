/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"
#include "lfs/training/ops/masks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

// Host side of the Vulkan and Metal mesh coverage kernels; mesh_mask.cu is the reference.
namespace lfs::training {

    inline constexpr uint32_t kMeshMaskThreads = 256;
    inline constexpr uint32_t kMeshMaskMaxPrepareGroups = 65535;
    inline constexpr uint32_t kMeshMaskMaxLargeGroups = 2048;

    // 4-byte scalars only, so the Slang and MSL kernels read the same layout.
    struct MeshMaskCameraBlock {
        float rows[12];
        float fx, fy, cx, cy;
        int32_t width, height;
        float guard_min_x, guard_max_x, guard_min_y, guard_max_y;
        int32_t distorted, model, num_distortion;
        float distortion[12];
        float src_fx, src_fy, src_cx, src_cy, dst_fx, dst_fy;
    };
    static_assert(sizeof(MeshMaskCameraBlock) == 43 * 4);

    inline void validate_mesh_coverage(const core::Tensor& vertices, const core::Tensor& indices,
                                       const gpu_ops::MeshMaskCamera& camera, const core::Tensor* samples,
                                       const core::UndistortParams* distortion, const float z_near) {
        using core::DataType;
        using core::Device;
        LFS_ASSERT_MSG(vertices.is_valid() && vertices.ndim() == 2 && vertices.shape()[1] == 3 &&
                           vertices.device() == Device::GPU && vertices.dtype() == DataType::Float32,
                       "Mesh mask vertices must be GPU Float32 [V,3]");
        LFS_ASSERT_MSG(indices.is_valid() && indices.ndim() == 2 && indices.shape()[1] == 3 &&
                           indices.device() == Device::GPU && indices.dtype() == DataType::Int32,
                       "Mesh mask indices must be GPU Int32 [F,3]");
        LFS_ASSERT_MSG(vertices.shape()[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                           indices.shape()[0] <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "Mesh mask vertex or face count exceeds Int32 range");
        LFS_ASSERT_MSG(camera.width > 0 && camera.height > 0 &&
                           static_cast<int64_t>(camera.width) * camera.height <= std::numeric_limits<int>::max(),
                       "Mesh mask output dimensions must be positive and fit Int32 indexing");
        LFS_ASSERT_MSG(std::isfinite(camera.fx) && camera.fx > 0.0f && std::isfinite(camera.fy) &&
                           camera.fy > 0.0f && std::isfinite(camera.cx) && std::isfinite(camera.cy),
                       "Mesh mask intrinsics must be finite with positive focal lengths");
        LFS_ASSERT_MSG(std::isfinite(z_near) && z_near > 0.0f, "Mesh mask near plane must be positive and finite");
        LFS_ASSERT_MSG((samples == nullptr) == (distortion == nullptr),
                       "Mesh mask sample map and distortion parameters must be provided together");
        if (!samples)
            return;
        LFS_ASSERT_MSG(samples->is_valid() && samples->device() == Device::GPU &&
                           samples->dtype() == DataType::Float32 && samples->ndim() == 3 &&
                           samples->shape()[0] == static_cast<size_t>(camera.height) &&
                           samples->shape()[1] == static_cast<size_t>(camera.width) && samples->shape()[2] == 2,
                       "Mesh mask inverse sample map must be GPU Float32 [H,W,2]");
        LFS_ASSERT_MSG(distortion->src_width == camera.width && distortion->src_height == camera.height,
                       "Mesh mask distortion source dimensions must match the output");
        LFS_ASSERT_MSG(distortion->src_width > 0 && distortion->src_height > 0 && distortion->dst_width > 0 &&
                           distortion->dst_height > 0 && std::isfinite(distortion->src_fx) &&
                           distortion->src_fx > 0.0f && std::isfinite(distortion->src_fy) &&
                           distortion->src_fy > 0.0f && std::isfinite(distortion->dst_fx) &&
                           distortion->dst_fx > 0.0f && std::isfinite(distortion->dst_fy) && distortion->dst_fy > 0.0f,
                       "Mesh mask distortion dimensions and focal lengths must be positive");
    }

    // Matches pack_camera in mesh_mask.cu, including the guard band that reaches wide-angle samples.
    inline MeshMaskCameraBlock pack_mesh_mask_camera(const gpu_ops::MeshMaskCamera& camera,
                                                     const core::Tensor* samples,
                                                     const core::UndistortParams* distortion) {
        MeshMaskCameraBlock packed{};
        std::copy(camera.world_to_camera.begin(), camera.world_to_camera.end(), packed.rows);
        packed.fx = camera.fx;
        packed.fy = camera.fy;
        packed.cx = camera.cx;
        packed.cy = camera.cy;
        packed.width = camera.width;
        packed.height = camera.height;
        packed.guard_min_x = std::numeric_limits<float>::infinity();
        packed.guard_max_x = -std::numeric_limits<float>::infinity();
        packed.guard_min_y = std::numeric_limits<float>::infinity();
        packed.guard_max_y = -std::numeric_limits<float>::infinity();
        const auto expand_guard = [&](const float fx, const float fy, const float cx, const float cy,
                                      const int width, const int height) {
            packed.guard_min_x = std::min(packed.guard_min_x, (-static_cast<float>(width) - cx) / fx);
            packed.guard_max_x = std::max(packed.guard_max_x, (2.0f * width - cx) / fx);
            packed.guard_min_y = std::min(packed.guard_min_y, (-static_cast<float>(height) - cy) / fy);
            packed.guard_max_y = std::max(packed.guard_max_y, (2.0f * height - cy) / fy);
        };
        expand_guard(camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height);
        if (!distortion)
            return packed;

        packed.distorted = 1;
        packed.model = static_cast<int32_t>(distortion->model_type);
        packed.num_distortion = distortion->num_distortion;
        std::copy(std::begin(distortion->distortion), std::end(distortion->distortion), packed.distortion);
        packed.src_fx = distortion->src_fx;
        packed.src_fy = distortion->src_fy;
        packed.src_cx = distortion->src_cx;
        packed.src_cy = distortion->src_cy;
        packed.dst_fx = distortion->dst_fx;
        packed.dst_fy = distortion->dst_fy;
        expand_guard(distortion->dst_fx, distortion->dst_fy, distortion->dst_cx, distortion->dst_cy,
                     distortion->dst_width, distortion->dst_height);
        expand_guard(distortion->src_fx, distortion->src_fy, distortion->src_cx, distortion->src_cy,
                     distortion->src_width, distortion->src_height);
        // Wide-angle rays can land far outside every frame; the guard must reach them.
        if (!samples)
            return packed;
        const auto magnitude = samples->abs();
        const auto extent = magnitude.masked_fill(magnitude.ne(magnitude), 0.0f).max({0, 1}).cpu();
        const float extent_x = extent.ptr<float>()[0];
        const float extent_y = extent.ptr<float>()[1];
        packed.guard_min_x = std::min(packed.guard_min_x, -2.0f * extent_x);
        packed.guard_max_x = std::max(packed.guard_max_x, 2.0f * extent_x);
        packed.guard_min_y = std::min(packed.guard_min_y, -2.0f * extent_y);
        packed.guard_max_y = std::max(packed.guard_max_y, 2.0f * extent_y);
        return packed;
    }

} // namespace lfs::training
