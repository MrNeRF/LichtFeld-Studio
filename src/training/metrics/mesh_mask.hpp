/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "lfs/training/ops/masks.hpp"

#include <array>

namespace lfs::training {

    using MeshMaskCamera = lfs::gpu_ops::MeshMaskCamera;

    [[nodiscard]] lfs::core::Tensor rasterize_mesh_coverage(
        const lfs::core::Tensor& vertices,
        const lfs::core::Tensor& indices,
        const MeshMaskCamera& camera,
        float z_near,
        void* stream = nullptr);

    [[nodiscard]] lfs::core::Tensor rasterize_mesh_coverage(
        const lfs::core::Tensor& vertices,
        const lfs::core::Tensor& indices,
        const MeshMaskCamera& camera,
        const lfs::core::Tensor& inverse_sample_map,
        const lfs::core::UndistortParams& distortion,
        float z_near,
        void* stream = nullptr);

    [[nodiscard]] core::Tensor splat_point_coverage(const core::Tensor& means, const MeshMaskCamera& camera,
                                                    int radius, int close, const core::UndistortParams* distortion = nullptr,
                                                    void* stream = nullptr);

} // namespace lfs::training
