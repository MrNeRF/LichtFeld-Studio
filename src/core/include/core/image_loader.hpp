/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include <filesystem>
#include <functional>

namespace lfs::core {

    struct ImageLoadParams {
        std::filesystem::path path;
        int resize_factor = 1;
        int max_width = 0;
        void* stream = nullptr;
        bool output_uint8 = false;
    };

    using ImageLoadFunc = std::function<Tensor(const ImageLoadParams&)>;

    LFS_CORE_API void set_image_loader(ImageLoadFunc fn);
    LFS_CORE_API Tensor load_image_cached(const ImageLoadParams& params);

} // namespace lfs::core
