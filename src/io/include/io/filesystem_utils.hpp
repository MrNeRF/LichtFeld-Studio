/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/path_utils.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::io {

    namespace fs = std::filesystem;

    namespace detail {

        inline void ascii_lower_inplace(std::string& value) {
            for (char& ch : value) {
                const unsigned char uch = static_cast<unsigned char>(ch);
                if (uch >= 'A' && uch <= 'Z') {
                    ch = static_cast<char>(uch - 'A' + 'a');
                }
            }
        }

        inline std::string normalize_lookup_key(std::string value) {
            std::replace(value.begin(), value.end(), '\\', '/');
            ascii_lower_inplace(value);
            return value;
        }

        inline std::string normalize_lookup_key(const fs::path& value) {
            return normalize_lookup_key(lfs::core::path_to_utf8(value.lexically_normal()));
        }

    } // namespace detail

    inline constexpr std::array<const char*, 4> MASK_SEARCH_FOLDERS = {
        "masks",
        "mask",
        "segmentation",
        "dynamic_masks",
    };

    inline constexpr std::array<const char*, 4> MASK_SEARCH_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".mask.png",
    };

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

        std::string target_lower = detail::normalize_lookup_key(target);

        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec)
                break;
            if (entry.is_regular_file()) {
                std::string name = detail::normalize_lookup_key(entry.path().filename());
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

    // Recursive file index with exact relative-path matching and a basename
    // fallback when that basename is unique under the indexed root.
    class RecursiveFileCache {
    public:
        explicit RecursiveFileCache(const fs::path& root_path) {
            if (!safe_is_directory(root_path))
                return;

            std::error_code ec;
            for (fs::recursive_directory_iterator it(
                     root_path,
                     fs::directory_options::skip_permission_denied,
                     ec),
                 end;
                 !ec && it != end;
                 it.increment(ec)) {
                const auto& entry = *it;
                std::error_code file_ec;
                if (!entry.is_regular_file(file_ec) || file_ec)
                    continue;

                const fs::path rel = entry.path().lexically_relative(root_path);
                if (rel.empty())
                    continue;

                const std::string rel_key = detail::normalize_lookup_key(rel);
                exact_entries_.emplace(rel_key, entry.path());

                const std::string basename_key =
                    detail::normalize_lookup_key(entry.path().filename());
                if (auto [it_basename, inserted] =
                        basename_entries_.emplace(basename_key, entry.path());
                    !inserted && it_basename->second != entry.path()) {
                    ambiguous_basenames_.insert(basename_key);
                }
            }
        }

        fs::path find(const fs::path& relative_or_name) const {
            if (relative_or_name.empty())
                return {};

            const std::string exact_key =
                detail::normalize_lookup_key(relative_or_name);
            if (auto it = exact_entries_.find(exact_key);
                it != exact_entries_.end()) {
                return it->second;
            }

            const std::string basename_key =
                detail::normalize_lookup_key(relative_or_name.filename());
            if (ambiguous_basenames_.contains(basename_key))
                return {};

            if (auto it = basename_entries_.find(basename_key);
                it != basename_entries_.end()) {
                return it->second;
            }

            return {};
        }

    private:
        std::unordered_map<std::string, fs::path> exact_entries_;
        std::unordered_map<std::string, fs::path> basename_entries_;
        std::unordered_set<std::string> ambiguous_basenames_;
    };

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
        detail::ascii_lower_inplace(ext);

        return std::find(image_extensions.begin(), image_extensions.end(), ext) != image_extensions.end();
    }

    inline std::string strip_extension(const std::string& filename) {
        auto last_dot = filename.find_last_of('.');
        if (last_dot == std::string::npos) {
            return filename; // No extension found
        }
        return filename.substr(0, last_dot);
    }

    // Pre-scanned directory cache for fast case-insensitive mask lookups.
    // Avoids repeated directory scans for every image.
    class MaskDirCache {
    public:
        explicit MaskDirCache(const fs::path& base_path) {
            for (const auto* folder : MASK_SEARCH_FOLDERS) {
                const fs::path mask_dir = base_path / folder;
                if (!safe_is_directory(mask_dir))
                    continue;

                DirectoryIndex dir_index;
                std::error_code ec;
                for (fs::recursive_directory_iterator it(
                         mask_dir,
                         fs::directory_options::skip_permission_denied,
                         ec),
                     end;
                     !ec && it != end;
                     it.increment(ec)) {
                    const auto& entry = *it;
                    std::error_code file_ec;
                    if (!entry.is_regular_file(file_ec) || file_ec)
                        continue;
                    fs::path rel = entry.path().lexically_relative(mask_dir);
                    if (rel.empty())
                        continue;
                    dir_index.entries.emplace(detail::normalize_lookup_key(rel), entry.path());
                }
                dir_indices_.push_back(std::move(dir_index));
            }
        }

        fs::path find(const std::string& image_name) const {
            if (dir_indices_.empty())
                return {};

            const std::vector<std::string> lookup_keys = build_lookup_keys(image_name);

            for (const auto& dir_index : dir_indices_) {
                for (const auto& key : lookup_keys) {
                    if (auto it = dir_index.entries.find(key); it != dir_index.entries.end()) {
                        return it->second;
                    }
                }
            }
            return {};
        }

    private:
        struct DirectoryIndex {
            std::unordered_map<std::string, fs::path> entries;
        };

        static std::vector<std::string> build_lookup_keys(const std::string& image_name) {
            const fs::path img_path = lfs::core::utf8_to_path(image_name);
            const fs::path stem_path = img_path.parent_path() / img_path.stem();

            std::vector<std::string> keys;
            keys.reserve(1 + 2 * MASK_SEARCH_EXTENSIONS.size());
            keys.push_back(detail::normalize_lookup_key(img_path));

            for (const auto* ext : MASK_SEARCH_EXTENSIONS) {
                fs::path target = stem_path;
                target += ext;
                keys.push_back(detail::normalize_lookup_key(target));
            }

            for (const auto* ext : MASK_SEARCH_EXTENSIONS) {
                fs::path target = img_path;
                target += ext;
                keys.push_back(detail::normalize_lookup_key(target));
            }

            return keys;
        }

        std::vector<DirectoryIndex> dir_indices_;
    };

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

        DatasetInfo info;
        info.base_path = base_path;

        for (const auto* name : IMAGE_FOLDERS) {
            if (safe_is_directory(base_path / name)) {
                info.images_path = base_path / name;
                break;
            }
        }
        if (info.images_path.empty()) {
            bool has_colmap_in_root = !find_file_ci(base_path, "cameras.bin").empty() ||
                                      !find_file_ci(base_path, "cameras.txt").empty();
            if (has_colmap_in_root) {
                std::error_code ec;
                for (const auto& entry : fs::directory_iterator(base_path, ec)) {
                    if (!ec && entry.is_regular_file() && is_image_file(entry.path())) {
                        info.images_path = base_path;
                        break;
                    }
                }
            }
            if (info.images_path.empty()) {
                info.images_path = base_path / "images";
            }
        }

        if (safe_is_directory(info.images_path)) {
            std::error_code ec;
            for (const auto& entry : fs::directory_iterator(info.images_path, ec)) {
                if (!ec && entry.is_regular_file() && is_image_file(entry.path())) {
                    ++info.image_count;
                }
            }
        }

        for (const auto& sp : get_colmap_search_paths(base_path)) {
            if (!find_file_ci(sp, "cameras.bin").empty() || !find_file_ci(sp, "cameras.txt").empty()) {
                info.sparse_path = sp;
                break;
            }
        }
        if (info.sparse_path.empty()) {
            info.sparse_path = base_path / "sparse" / "0";
        }

        for (const auto* name : MASK_SEARCH_FOLDERS) {
            if (safe_is_directory(base_path / name)) {
                info.masks_path = base_path / name;
                info.has_masks = true;
                std::error_code ec;
                for (fs::recursive_directory_iterator it(
                         info.masks_path,
                         fs::directory_options::skip_permission_denied,
                         ec),
                     end;
                     !ec && it != end;
                     it.increment(ec)) {
                    std::error_code file_ec;
                    if (!it->is_regular_file(file_ec) || file_ec)
                        continue;
                    if (is_image_file(it->path())) {
                        ++info.mask_count;
                    }
                }
                break;
            }
        }

        return info;
    }

} // namespace lfs::io
