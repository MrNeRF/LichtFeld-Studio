/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::io::mesh {

    struct TextureData {
        std::vector<uint8_t> pixels;
        int width = 0;
        int height = 0;
        int channels = 0;
    };

    class TextureLoader {
    public:
        const TextureData* load_from_file(const std::filesystem::path& path);
        TextureData load_from_memory(const uint8_t* data, size_t size);
        void clear_cache();

    private:
        std::unordered_map<std::string, TextureData> cache_;
    };

} // namespace lfs::io::mesh
