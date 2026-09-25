/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace lfs::gpu_ops {

    using Tensor = core::Tensor;
    using In = const Tensor&;
    using Out = Tensor&;

    struct HW {
        int h, w;
    };
    struct Intrinsics {
        float fx, fy, cx, cy;
    };

    enum class Layout { HW,
                        HWC,
                        CHW };
    enum class ShStorage { Float32,
                           IeeeFloat16,
                           Q16 };

    // Concrete members exist only in the backend implementation.
    struct BackendState {
        virtual ~BackendState() = default;
    };
    using State = std::unique_ptr<BackendState>;

} // namespace lfs::gpu_ops
