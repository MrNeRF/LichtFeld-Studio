/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"

#include <array>
#include <cstdint>

namespace lfs::gpu_ops {

    struct RoiParams {
        HW image;
        Intrinsics intrinsics;
        // Column-major, matching the scene crop-box transform.
        std::array<float, 16> world_to_cropbox;
        std::array<float, 3> minimum, maximum;
        float outside_weight;
        bool inverse;
    };

    struct TrainingImageOps {
        void (*heatmap)(In loss, Out latest, Out ema, int slot, float ema_alpha);
        void (*roi)(In view, In camera_position, Out weights, const RoiParams&);
        void (*resize_background)(In source, Out destination);
        void (*random_background)(Out destination, uint64_t seed);
        void (*canny)(In image, Out edges);
        void (*normalize_scalar)(Out values, In scalar, float skip_below);
        // Optional upload/layout specialization; callers retain their default path.
        Tensor (*upload_image_chw)(In cpu_hwc) = nullptr;
        Tensor (*quantize_to_8bit_grid)(In image) = nullptr;
        Tensor (*quantize_to_grid)(In image, float levels) = nullptr;
        Tensor (*flip_error_map)(In reference, In test, float pixels_per_degree) = nullptr;
        Tensor (*flip_error_image)(In error_map) = nullptr;
    };

} // namespace lfs::gpu_ops
