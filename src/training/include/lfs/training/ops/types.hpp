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

    struct HW {
        int h = 0;
        int w = 0;
    };

    struct Intrinsics {
        float fx = 0.f;
        float fy = 0.f;
        float cx = 0.f;
        float cy = 0.f;
    };

} // namespace lfs::gpu_ops
