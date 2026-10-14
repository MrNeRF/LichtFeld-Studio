/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <filesystem>
#include <span>
#include <string>

namespace lfs::training {

    struct ImageCaption {
        int x = 0; // left edge of the tile the caption names
        std::string text;
    };

    [[nodiscard]] std::filesystem::path default_caption_font();

    // Height of the caption band above tiles that are tile_height pixels tall.
    [[nodiscard]] int caption_band_height(int tile_height);

    // Captions on a dark band, [3, height, width] Float32 on the GPU. Without a readable font the band stays empty
    // and keeps its size, so saved images keep their layout.
    [[nodiscard]] lfs::core::Tensor caption_band(std::span<const ImageCaption> captions, int width, int height,
                                                 const std::filesystem::path& font = default_caption_font());

} // namespace lfs::training
