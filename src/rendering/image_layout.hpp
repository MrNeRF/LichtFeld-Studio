/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <cassert>
#include <vector>

namespace lfs::rendering {

    using Tensor = lfs::core::Tensor;

    enum class ImageLayout { HWC,
                             CHW,
                             Unknown };

    inline ImageLayout detectImageLayout(const Tensor& image) {
        assert(image.ndim() == 3);
        const auto is_channel = [](const auto dim) { return dim == 1 || dim == 3 || dim == 4; };
        const bool last_is_channel = is_channel(image.size(2));
        const bool first_is_channel = is_channel(image.size(0));

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

    inline Tensor flipImageVertical(const Tensor& image, ImageLayout layout) {
        assert(layout != ImageLayout::Unknown);
        if (!image.is_valid()) {
            return {};
        }

        const int height = imageHeight(image, layout);
        std::vector<int> row_indices(static_cast<size_t>(height));
        for (int row = 0; row < height; ++row) {
            row_indices[static_cast<size_t>(row)] = height - 1 - row;
        }

        const Tensor indices = Tensor::from_vector(
            row_indices, {static_cast<size_t>(height)}, image.device());
        const int dim = (layout == ImageLayout::HWC) ? 0 : 1;
        return image.index_select(dim, indices).contiguous();
    }

} // namespace lfs::rendering
