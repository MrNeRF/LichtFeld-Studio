/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace lfs::training::kernels {

    void launch_roi_weight_map(
        const float* world_to_camera, // [1,4,4] or [4,4] CUDA, row-major
        const float* camera_position, // [3] CUDA
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height,
        const glm::mat4& world_to_cropbox,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        bool inverse,
        float outside_weight,
        float* output, // [H,W] CUDA
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
