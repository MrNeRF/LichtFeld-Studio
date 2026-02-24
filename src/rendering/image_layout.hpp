/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <cassert>

namespace lfs::rendering {

    using Tensor = lfs::core::Tensor;

    enum class ImageLayout { HWC,
                             CHW,
                             Unknown };

    inline ImageLayout detectImageLayout(const Tensor& image) {
        assert(image.ndim() == 3);
        const bool last_is_channel = (image.size(2) == 3 || image.size(2) == 4);
        const bool first_is_channel = (image.size(0) == 3 || image.size(0) == 4);

        if (first_is_channel && !last_is_channel)
            return ImageLayout::CHW;
        if (last_is_channel && !first_is_channel)
            return ImageLayout::HWC;
        // Ambiguous (e.g. 3x3x3): prefer CHW since rasterizer outputs CHW
        if (first_is_channel && last_is_channel)
            return ImageLayout::CHW;
        return ImageLayout::Unknown;
    }

    inline int imageHeight(const Tensor& image, ImageLayout layout) {
        assert(layout != ImageLayout::Unknown);
        return static_cast<int>(layout == ImageLayout::HWC ? image.size(0) : image.size(1));
    }

    inline int imageWidth(const Tensor& image, ImageLayout layout) {
        assert(layout != ImageLayout::Unknown);
        return static_cast<int>(layout == ImageLayout::HWC ? image.size(1) : image.size(2));
    }

    inline int imageChannels(const Tensor& image, ImageLayout layout) {
        assert(layout != ImageLayout::Unknown);
        return static_cast<int>(layout == ImageLayout::HWC ? image.size(2) : image.size(0));
    }

} // namespace lfs::rendering
