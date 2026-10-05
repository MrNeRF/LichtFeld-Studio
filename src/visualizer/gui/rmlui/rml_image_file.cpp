/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/rmlui/rml_image_file.hpp"

#include "core/path_utils.hpp"
#include "internal/resource_paths.hpp"

#include <filesystem>
#include <memory>
#include <stb_image.h>
#include <string_view>

namespace lfs::vis::gui {
    namespace {
        std::optional<RmlImageFile> decode(const std::string& path) {
            int width = 0, height = 0, channels = 0;
            std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
                stbi_load(path.c_str(), &width, &height, &channels, 4), stbi_image_free);
            if (!pixels)
                return std::nullopt;
            const std::size_t count = static_cast<std::size_t>(width) * height;
            RmlImageFile image{.path = path, .width = width, .height = height,
                               .rgba = {pixels.get(), pixels.get() + count * 4}};
            for (std::size_t i = 0; i < count; ++i) {
                auto* p = image.rgba.data() + i * 4;
                if (channels == 1) {
                    p[1] = p[2] = p[3] = p[0];
                } else {
                    const unsigned int alpha = p[3];
                    p[0] = static_cast<std::uint8_t>((p[0] * alpha + 127) / 255);
                    p[1] = static_cast<std::uint8_t>((p[1] * alpha + 127) / 255);
                    p[2] = static_cast<std::uint8_t>((p[2] * alpha + 127) / 255);
                }
            }
            return image;
        }

        std::optional<RmlImageFile> decodeAsset(std::string name) {
            while (name.starts_with("../"))
                name.erase(0, 3);
            while (name.starts_with("./"))
                name.erase(0, 2);
            if (name.empty())
                return std::nullopt;
            try {
                const auto path = getAssetPath(name);
                if (std::filesystem::exists(path))
                    return decode(lfs::core::path_to_utf8(path));
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): optional asset fallbacks deliberately ignore path lookup failures.
            }
            return std::nullopt;
        }
    } // namespace

    std::optional<RmlImageFile> loadRmlImageFile(const std::string& source) {
        if (auto image = decode(source))
            return image;
        if (auto image = decodeAsset(source))
            return image;
        const std::string_view view(source);
        for (const std::string_view segment : {std::string_view("rmlui/icon/"), std::string_view("/icon/")}) {
            if (const auto pos = view.find(segment); pos != std::string_view::npos) {
                if (auto image = decodeAsset("icon/" + std::string(view.substr(pos + segment.size()))))
                    return image;
            }
        }
#ifndef _WIN32
        if (!source.empty() && source[0] != '/' && source.find("://") == std::string::npos) {
            const std::string absolute_source = "/" + source;
            if (std::filesystem::exists(absolute_source))
                return decode(absolute_source);
        }
#endif
        return std::nullopt;
    }
} // namespace lfs::vis::gui
