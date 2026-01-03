/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace lfs::io {

    namespace fs = std::filesystem;

    // Safe filesystem operations that don't throw
    inline bool safe_exists(const fs::path& path) {
        std::error_code ec;
        return fs::exists(path, ec);
    }

    inline bool safe_is_directory(const fs::path& path) {
        std::error_code ec;
        return fs::is_directory(path, ec);
    }

    // Case-insensitive file finding
    inline fs::path find_file_ci(const fs::path& dir, const std::string& target) {
        if (!safe_exists(dir) || !safe_is_directory(dir))
            return {};

        std::string target_lower = target;
        std::transform(target_lower.begin(), target_lower.end(), target_lower.begin(), ::tolower);

        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec)
                break;
            if (entry.is_regular_file()) {
                std::string name = entry.path().filename().string();
                std::transform(name.begin(), name.end(), name.begin(), ::tolower);
                if (name == target_lower) {
                    return entry.path();
                }
            }
        }
        return {};
    }

    // Find file in multiple locations (case-insensitive)
    inline fs::path find_file_in_paths(const std::vector<fs::path>& search_paths,
                                       const std::string& filename) {
        for (const auto& dir : search_paths) {
            if (auto found = find_file_ci(dir, filename); !found.empty()) {
                return found;
            }
        }
        return {};
    }

    // Get standard COLMAP search paths for a base directory
    inline std::vector<fs::path> get_colmap_search_paths(const fs::path& base) {
        return {
            base / "sparse" / "0", // Standard COLMAP
            base / "sparse",       // Alternative COLMAP
            base                   // Reality Capture / flat structure
        };
    }

    // Check if a file has an image extension
    inline bool is_image_file(const fs::path& path) {
        static const std::vector<std::string> image_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"};

        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        return std::find(image_extensions.begin(), image_extensions.end(), ext) != image_extensions.end();
    }

    inline std::string strip_extension(const std::string& filename) {
        auto last_dot = filename.find_last_of('.');
        if (last_dot == std::string::npos) {
            return filename; // No extension found
        }
        return filename.substr(0, last_dot);
    }

    struct DatasetInfo {
        fs::path base_path;
        fs::path images_path;
        fs::path sparse_path;
        fs::path masks_path;
        bool has_masks = false;
        int image_count = 0;
        int mask_count = 0;
    };

    inline DatasetInfo detect_dataset_info(const fs::path& base_path) {
        static constexpr const char* const IMAGE_FOLDERS[] = {"images", "images_4", "images_2", "images_8", "input", "rgb"};
        static constexpr const char* const MASK_FOLDERS[] = {"masks", "mask", "dynamic_masks"};

        DatasetInfo info;
        info.base_path = base_path;

        for (const auto* name : IMAGE_FOLDERS) {
            if (safe_is_directory(base_path / name)) {
                info.images_path = base_path / name;
                break;
            }
        }
        if (info.images_path.empty()) {
            info.images_path = base_path / "images";
        }

        if (safe_is_directory(info.images_path)) {
            std::error_code ec;
            for (const auto& entry : fs::directory_iterator(info.images_path, ec)) {
                if (!ec && entry.is_regular_file() && is_image_file(entry.path())) {
                    ++info.image_count;
                }
            }
        }

        if (safe_is_directory(base_path / "sparse" / "0")) {
            info.sparse_path = base_path / "sparse" / "0";
        } else if (safe_is_directory(base_path / "sparse")) {
            info.sparse_path = base_path / "sparse";
        } else {
            info.sparse_path = base_path / "sparse" / "0";
        }

        for (const auto* name : MASK_FOLDERS) {
            if (safe_is_directory(base_path / name)) {
                info.masks_path = base_path / name;
                info.has_masks = true;
                std::error_code ec;
                for (const auto& entry : fs::directory_iterator(info.masks_path, ec)) {
                    if (!ec && entry.is_regular_file() && is_image_file(entry.path())) {
                        ++info.mask_count;
                    }
                }
                break;
            }
        }

        return info;
    }

} // namespace lfs::io