/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "texture_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <cassert>
#include <limits>
#include <stb_image.h>

namespace lfs::io::mesh {

    const TextureData* TextureLoader::load_from_file(const std::filesystem::path& path) {
        auto key = lfs::core::path_to_utf8(path);
        if (auto it = cache_.find(key); it != cache_.end()) {
            return &it->second;
        }

        TextureData tex;
        auto* data = stbi_load(key.c_str(), &tex.width, &tex.height, &tex.channels, 4);
        if (!data) {
            LOG_ERROR("Failed to load texture: {}", key);
            return nullptr;
        }

        tex.channels = 4;
        const size_t byte_count = static_cast<size_t>(tex.width) * tex.height * tex.channels;
        tex.pixels.assign(data, data + byte_count);
        stbi_image_free(data);

        LOG_INFO("Loaded texture: {} ({}x{}, {} ch)", key, tex.width, tex.height, tex.channels);
        auto [it, _] = cache_.emplace(std::move(key), std::move(tex));
        return &it->second;
    }

    TextureData TextureLoader::load_from_memory(const uint8_t* data, size_t size) {
        assert(data && size > 0);
        assert(size <= static_cast<size_t>(std::numeric_limits<int>::max()));

        TextureData tex;
        auto* pixels = stbi_load_from_memory(data, static_cast<int>(size),
                                             &tex.width, &tex.height, &tex.channels, 4);
        if (!pixels) {
            LOG_ERROR("Failed to decode embedded texture");
            return {};
        }

        tex.channels = 4;
        const size_t byte_count = static_cast<size_t>(tex.width) * tex.height * tex.channels;
        tex.pixels.assign(pixels, pixels + byte_count);
        stbi_image_free(pixels);

        return tex;
    }

    void TextureLoader::clear_cache() {
        cache_.clear();
    }

} // namespace lfs::io::mesh
