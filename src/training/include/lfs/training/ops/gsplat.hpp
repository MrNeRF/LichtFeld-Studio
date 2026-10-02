/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_types.h"
#include "lfs/training/ops/raster.hpp"

namespace lfs::gpu_ops {
    enum class GsplatRenderMode { RGB = 0,
                                  D = 1,
                                  ED = 2,
                                  RGB_D = 3,
                                  RGB_ED = 4 };

    struct GsplatParams {
        HW full_image;
        Intrinsics intrinsics;
        int tile_x = 0, tile_y = 0, tile_w = 0, tile_h = 0;
        ShParams sh;
        core::CameraModelType camera_model = core::CameraModelType::PINHOLE;
        GsplatRenderMode render_mode = GsplatRenderMode::RGB;
        float scaling_modifier = 1.f;
        bool antialiased = false;
    };

    // Resolve each gradient at its existing accumulation point. In particular,
    // SH-rest storage stays unallocated until the first active SH backward.
    struct GsplatGradients {
        void* owner;
        Out (*get)(void*, AdamSlot);
    };

    struct GsplatSaved {
        State backend;
    };

    struct GsplatRasterOps {
        State (*create)();
        RasterResult (*forward)(GsplatSaved&, const SplatInputs&, In view,
                                In radial, In tangential, In bg_color, In bg_image,
                                const GsplatParams&, const RenderOutputs&);
        void (*backward)(GsplatSaved&, In grad_image, In grad_alpha,
                         const GsplatGradients&, Out densification, In error_map, In edge_map,
                         Out edge_scores, Out max_screen_share);
        void (*release)(GsplatSaved&) noexcept;

        // Rasterizer-owned VRAM rows. Image and alpha are the bound outputs.
        void (*record_vram)(
            const GsplatSaved&, In image, In alpha,
            In gt_tile, In bg_tile, In error_map);
        // Drops cached camera and output tensors. True when every cache is gone.
        bool (*release_caches)(GsplatSaved&) noexcept;
    };
} // namespace lfs::gpu_ops
