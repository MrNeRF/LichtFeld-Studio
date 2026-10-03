/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda/undistort/undistort.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cuda_runtime.h>

namespace lfs::training {

    struct MeshMaskCamera {
        std::array<float, 12> world_to_camera{
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f};
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        int width = 0;
        int height = 0;
    };

    [[nodiscard]] lfs::core::Tensor rasterize_mesh_coverage(
        const lfs::core::Tensor& vertices,
        const lfs::core::Tensor& indices,
        const MeshMaskCamera& camera,
        float z_near,
        cudaStream_t stream = nullptr);

    [[nodiscard]] lfs::core::Tensor rasterize_mesh_coverage(
        const lfs::core::Tensor& vertices,
        const lfs::core::Tensor& indices,
        const MeshMaskCamera& camera,
        const lfs::core::Tensor& inverse_sample_map,
        const lfs::core::UndistortParams& distortion,
        float z_near,
        cudaStream_t stream = nullptr);

} // namespace lfs::training
