/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"

#include <cstddef>

namespace lfs::gpu_ops {

    enum class PhotoPath {
        L1,
        SSIM,
        Fused,
        Decoupled,
        MaskedFused,
        MaskedDecoupled
    };

    struct PhotoParams {
        PhotoPath path = PhotoPath::Fused;
        float ssim_weight = 0.f;
        bool valid_padding = true;
        float denominator = 0.f;
    };

    struct PhotoSaved {
        Tensor ssim_map, cs_map;
        State backend;
    };

    struct PhotoWorkspaceBytes {
        size_t required = 0;
        size_t allocated = 0;
        size_t error_map = 0;
    };

    struct PhotometricOps {
        State (*create)();

        // Binds the caller's outputs to the workspace views this pass produces.
        void (*evaluate)(
            PhotoSaved&, In corrected, In raw, In target, In mask,
            const PhotoParams&, Out loss, Out grad_corrected, Out grad_raw);

        // Allocating metric reduction. maps publishes the per-pixel maps on PhotoSaved.
        Tensor (*metric)(
            PhotoSaved&, In predicted, In target,
            bool maps, bool valid_padding);

        void (*error_map)(
            PhotoSaved&, In predicted, In target,
            Out error, bool contrast_structure_only);

        void (*map_to_error)(In map, Out error);

        PhotoWorkspaceBytes (*workspace_bytes)(const PhotoSaved&);
        void (*shrink_to_required)(PhotoSaved&);
        void (*reset)(PhotoSaved&);

        // Null when evaluate writes grad_raw. Otherwise the decoupled paths leave grad_raw empty, and this adds the
        // raw-render gradient of the last evaluate into grad_image after the appearance backward, then releases it;
        // an empty grad_image only releases it.
        void (*add_raw_gradient)(PhotoSaved&, Out grad_image) = nullptr;
    };

} // namespace lfs::gpu_ops
