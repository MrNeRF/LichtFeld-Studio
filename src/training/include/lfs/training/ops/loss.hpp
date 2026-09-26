/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <memory>

namespace lfs::gpu_ops {

    using Tensor = core::Tensor;
    using In = const Tensor&;
    using Out = Tensor&;

    // Concrete members exist only in the backend implementation.
    struct BackendState {
        virtual ~BackendState() = default;
    };
    using State = std::unique_ptr<BackendState>;

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
    };

    struct PhotoSaved {
        Tensor ssim_map, cs_map;
        State backend;
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
    };

} // namespace lfs::gpu_ops
