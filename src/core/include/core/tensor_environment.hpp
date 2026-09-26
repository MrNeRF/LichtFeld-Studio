/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"
namespace lfs::core {
    struct EnvironmentCompositeParams {
        float rotation[9]; // camera rotation, glm::mat3 memory order (column-major)
        int full_width = 0;
        int full_height = 0;
        int band_width = 0;
        int band_height = 0;
        int y_offset = 0;
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
        int equirect_view = 0;
        float exposure_factor = 1.0f;
        float env_rotation_radians = 0.0f;
        int env_width = 0;
        int env_height = 0;
    };

    // RGB is float CHW, alpha is float HW, environment is float HWC. Returns u8 HWC.
    LFS_CORE_API Tensor environment_composite(const Tensor& rgb, const Tensor& alpha, const Tensor& environment,
                                              const EnvironmentCompositeParams& params);
} // namespace lfs::core
