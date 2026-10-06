/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "graphics_external_tensor.hpp"

#include "core/services.hpp"
#include "window/graphics_context.hpp"
#include "window/window_manager.hpp"

namespace lfs::vis {

    lfs::core::SplatTensorAllocator makeViewerSplatTensorAllocator(
        const bool preserve_float_shN) {
        auto* const window = services().windowOrNull();
        auto* const context = window ? window->getGraphicsContext() : nullptr;
        return context ? context->splatTensorAllocator(preserve_float_shN)
                       : lfs::core::SplatTensorAllocator{};
    }

} // namespace lfs::vis
