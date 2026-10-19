/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/path_utils.hpp"
#include <stb_image.h>

namespace lfs::core::stb_file {
    // Keep the pipeline's FILE-based approach: stb's filename API has a fixed
    // 1024-unit Windows buffer even when the filename has an extended prefix.
    class StbFile {
    public:
        explicit StbFile(const std::filesystem::path& path) : file_(open_file(path, "rb")) {}
        ~StbFile() {
            if (file_)
                std::fclose(file_);
        }
        StbFile(const StbFile&) = delete;
        StbFile& operator=(const StbFile&) = delete;
        std::FILE* get() const { return file_; }

    private:
        std::FILE* file_;
    };

    inline bool stb_is_16_bit(const std::filesystem::path& path) {
        const StbFile file(path);
        return file.get() && stbi_is_16_bit_from_file(file.get());
    }

    inline stbi_uc* stb_load(const std::filesystem::path& path, int* w, int* h, int* channels, const int desired) {
        const StbFile file(path);
        return file.get() ? stbi_load_from_file(file.get(), w, h, channels, desired) : nullptr;
    }

    inline stbi_us* stb_load_16(const std::filesystem::path& path, int* w, int* h, int* channels, const int desired) {
        const StbFile file(path);
        return file.get() ? stbi_load_from_file_16(file.get(), w, h, channels, desired) : nullptr;
    }

    inline bool stb_info(const std::filesystem::path& path, int* w, int* h, int* channels) {
        const StbFile file(path);
        return file.get() && stbi_info_from_file(file.get(), w, h, channels);
    }
} // namespace lfs::core::stb_file
