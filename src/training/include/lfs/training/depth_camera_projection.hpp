/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/tensor.hpp"
#include "lfs/training/ops/geometry_types.hpp"

#include <algorithm>
#include <cstddef>

namespace lfs::training {

    // The camera's lens as the GPU projection models it; pinhole once undistortion is prepared.
    // Header-only so preprocessing has it in builds without the trainer.
    [[nodiscard]] inline kernels::DepthCameraProjection depth_camera_projection(const lfs::core::Camera& cam) {
        kernels::DepthCameraProjection p;
        if (cam.is_undistort_prepared())
            return p;
        p.model = static_cast<int>(cam.camera_model_type());
        auto copy = [](const lfs::core::Tensor& tensor, float* out, size_t count) {
            if (!tensor.is_valid() || !tensor.numel())
                return;
            const auto host = tensor.cpu().contiguous();
            std::copy_n(host.ptr<float>(), std::min(count, host.numel()), out);
        };
        copy(cam.radial_distortion(), p.radial, 6);
        if (cam.camera_model_type() == lfs::core::CameraModelType::THIN_PRISM_FISHEYE)
            copy(cam.tangential_distortion(), p.thin_prism, 4);
        else
            copy(cam.tangential_distortion(), p.tangential, 2);
        return p;
    }

} // namespace lfs::training
