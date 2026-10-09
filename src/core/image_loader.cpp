/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_loader.hpp"
#include <cassert>

namespace lfs::core {

    static ImageLoadFunc g_image_loader;

    void set_image_loader(ImageLoadFunc fn) {
        assert(fn);
        g_image_loader = std::move(fn);
    }

    bool has_image_loader() noexcept {
        return static_cast<bool>(g_image_loader);
    }

    Tensor load_image_cached(const ImageLoadParams& params) {
        assert(g_image_loader);
        return g_image_loader(params);
    }

} // namespace lfs::core
