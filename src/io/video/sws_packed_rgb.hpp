/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

extern "C" {
#include <libswscale/swscale.h>
}

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace lfs::io {

    // swscale's vectorized RGB writers store whole SIMD blocks, so a row whose width is not a
    // multiple of the block size runs past `width * 3` bytes. Convert into rows with an aligned
    // stride and spare bytes after the last row, then copy the pixels into the tightly packed
    // `destination` (exactly width * height * 3 bytes). `scratch` is reused across calls.
    [[nodiscard]] inline int scaleToPackedRgb24(SwsContext* const context, const uint8_t* const source_data[],
                                                const int source_linesize[], const int source_height,
                                                uint8_t* const destination, const int width, const int height,
                                                std::vector<uint8_t>& scratch) {
        constexpr std::size_t kAlignment = 64;
        const std::size_t packed = static_cast<std::size_t>(width) * 3;
        const std::size_t stride = (packed + kAlignment - 1) / kAlignment * kAlignment;
        scratch.resize(stride * static_cast<std::size_t>(height) + 2 * kAlignment);
        const auto address = reinterpret_cast<std::uintptr_t>(scratch.data());
        uint8_t* const rows = scratch.data() + ((kAlignment - address % kAlignment) % kAlignment);

        uint8_t* dst_data[4] = {rows, nullptr, nullptr, nullptr};
        int dst_linesize[4] = {static_cast<int>(stride), 0, 0, 0};
        const int converted = sws_scale(context, source_data, source_linesize, 0, source_height, dst_data, dst_linesize);
        if (converted <= 0)
            return converted;
        for (int y = 0; y < height; ++y)
            std::memcpy(destination + static_cast<std::size_t>(y) * packed, rows + static_cast<std::size_t>(y) * stride,
                        packed);
        return converted;
    }

} // namespace lfs::io
