/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"
#include <array>

namespace lfs::gpu_ops {
    struct PPISPAdamUpdateParams {
        float lr, beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps;
    };
    struct PPISPAdamGroup {
        Out parameter, moment1, moment2;
        In gradient;
    };
    struct PPISPInputs {
        In exposure, vignetting, color, crf;
    };
    struct PPISPOutputs {
        Out exposure, vignetting, color, crf;
    };
    struct PPISPRegion {
        int y_offset, full_height;
        int cameras, frames, camera_index, frame_index;
        int x_offset = 0, full_width = 0;
    };
    struct PPISPOps {
        void (*forward)(const PPISPInputs&, In rgb, Out corrected, const PPISPRegion&);
        void (*backward)(const PPISPInputs&, In rgb, In grad_output,
                         const PPISPOutputs& gradients, Out grad_rgb,
                         int cameras, int frames, int camera_index, int frame_index);
        void (*adam)(const PPISPAdamGroup&, const PPISPAdamUpdateParams&);
        // An invalid parameter binding skips that group.
        void (*adam_batch)(const std::array<PPISPAdamGroup, 4>&, const PPISPAdamUpdateParams&);
        // Invalid gradient or loss bindings skip that output.
        void (*vignetting_regularization)(In parameters, Out gradient, Out loss,
                                          float center, float channel, float non_positive);
        void (*project_mean)(Out exposure, Out color);
        void (*initialize)(const PPISPOutputs&);
        // backward may write grad_rgb over grad_output.
        bool in_place = false;
    };
    struct ControllerOps {
        // Invalid features initializes the prior slot and waits for the copy.
        void (*prepare_input)(In features, Out fc_input, float exposure_prior);
        // Invalid grad_input skips SGEMM and ReLU for the first layer.
        void (*backward_layer)(In grad_output, In activation, In weight,
                               Out weight_gradient, Out bias_gradient, Out grad_input);
    };
} // namespace lfs::gpu_ops
