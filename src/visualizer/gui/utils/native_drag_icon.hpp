/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <core/executable_path.hpp>
#include <gdk-pixbuf/gdk-pixbuf.h>

namespace lfs::vis::gui {
    // Caller owns the returned reference. Use the existing packaged branding.
    inline GdkPixbuf* loadNativeDragIcon(int size, const std::filesystem::path& path = lfs::core::getAssetsDir() / "lichtfeld-icon.png") {
        if (auto* image = gdk_pixbuf_new_from_file_at_scale(path.c_str(), size, size, true, nullptr))
            return image;
        // Missing branding must not break a transfer: a plain file thumbnail.
        auto* image = gdk_pixbuf_new(GDK_COLORSPACE_RGB, true, 8, size, size);
        if (!image)
            return nullptr;
        gdk_pixbuf_fill(image, 0);
        auto* pixels = gdk_pixbuf_get_pixels(image);
        const auto stride = gdk_pixbuf_get_rowstride(image);
        for (int y = size / 8; y < size * 7 / 8; ++y)
            for (int x = size / 4; x < size * 3 / 4; ++x) {
                auto* pixel = pixels + y * stride + x * 4;
                const bool border = x == size / 4 || x == size * 3 / 4 - 1 || y == size / 8 || y == size * 7 / 8 - 1;
                pixel[0] = pixel[1] = pixel[2] = border ? 85 : 245;
                pixel[3] = 255;
            }
        return image;
    }
} // namespace lfs::vis::gui
