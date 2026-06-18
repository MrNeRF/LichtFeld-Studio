/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/path_utils.hpp"
#include "io/loader.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <functional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::io {

    namespace fs = std::filesystem;

    namespace detail {
        inline constexpr size_t CANCEL_POLL_INTERVAL = 64;

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

        inline void throw_if_scan_cancel_requested(const CancelCallback& cancel_requested,
                                                   const std::string_view message) {
            if (cancel_requested && cancel_requested()) {
                throw LoadCancelledError(std::string(message));
            }
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

    inline constexpr std::array<const char*, 2> DEPTH_SEARCH_FOLDERS = {
        "depth",
        "depths",
    };

    inline constexpr std::array<const char*, 6> DEPTH_SEARCH_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".depth.png",
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
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code file_ec;
            if (!it->is_regular_file(file_ec) || file_ec)
                continue;

            std::string name = detail::normalize_lookup_key(it->path().filename());
            if (name == target_lower) {
                return it->path();
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

    enum class LookupStatus {
        NotFound,
        Found,
        Ambiguous,
    };

    struct FileLookupResult {
        LookupStatus status = LookupStatus::NotFound;
        fs::path path;

        [[nodiscard]] bool found() const {
            return status == LookupStatus::Found && !path.empty();
        }

        [[nodiscard]] bool ambiguous() const {
            return status == LookupStatus::Ambiguous;
        }
    };

    // Recursive file index with exact relative-path matching and a basename
    // fallback when that basename is unique under the indexed root.
    class RecursiveFileCache {
    public:
        explicit RecursiveFileCache(const fs::path& root_path,
                                    const CancelCallback& cancel_requested = nullptr) {
            if (!safe_is_directory(root_path))
                return;

            std::error_code ec;
            size_t scanned_entries = 0;
            for (fs::recursive_directory_iterator it(
                     root_path,
                     fs::directory_options::skip_permission_denied,
                     ec),
                 end;
                 !ec && it != end;
                 it.increment(ec)) {
                if ((scanned_entries % detail::CANCEL_POLL_INTERVAL) == 0) {
                    detail::throw_if_scan_cancel_requested(cancel_requested,
                                                           "Filesystem scan cancelled");
                }
                ++scanned_entries;

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

        [[nodiscard]] FileLookupResult lookup(const fs::path& relative_or_name) const {
            if (relative_or_name.empty())
                return {};

            const std::string exact_key =
                detail::normalize_lookup_key(relative_or_name);
            if (auto it = exact_entries_.find(exact_key);
                it != exact_entries_.end()) {
                return FileLookupResult{LookupStatus::Found, it->second};
            }

            const std::string basename_key =
                detail::normalize_lookup_key(relative_or_name.filename());
            if (ambiguous_basenames_.contains(basename_key))
                return FileLookupResult{LookupStatus::Ambiguous, {}};

            if (auto it = basename_entries_.find(basename_key);
                it != basename_entries_.end()) {
                return FileLookupResult{LookupStatus::Found, it->second};
            }

            return {};
        }

        [[nodiscard]] fs::path find(const fs::path& relative_or_name) const {
            if (auto result = lookup(relative_or_name); result.found()) {
                return result.path;
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
        explicit MaskDirCache(const fs::path& base_path,
                              const CancelCallback& cancel_requested = nullptr) {
            for (const auto* folder : MASK_SEARCH_FOLDERS) {
                detail::throw_if_scan_cancel_requested(cancel_requested,
                                                       "Mask directory scan cancelled");
                const fs::path mask_dir = base_path / folder;
                if (!safe_is_directory(mask_dir))
                    continue;

                dir_indices_.emplace_back(mask_dir, cancel_requested);
            }
        }

        [[nodiscard]] FileLookupResult lookup(const std::string& image_name) const {
            if (dir_indices_.empty())
                return {};

            const std::vector<fs::path> lookup_keys = build_lookup_keys(image_name);
            bool saw_ambiguous_match = false;

            for (const auto& dir_index : dir_indices_) {
                for (const auto& key : lookup_keys) {
                    if (auto result = dir_index.lookup(key); result.found()) {
                        return result;
                    } else if (result.ambiguous()) {
                        saw_ambiguous_match = true;
                    }
                }
            }

            if (saw_ambiguous_match) {
                return FileLookupResult{LookupStatus::Ambiguous, {}};
            }

            return {};
        }

        [[nodiscard]] fs::path find(const std::string& image_name) const {
            if (auto result = lookup(image_name); result.found()) {
                return result.path;
            }
            return {};
        }

    private:
        static std::vector<fs::path> build_lookup_keys(const std::string& image_name) {
            const fs::path img_path = lfs::core::utf8_to_path(image_name);
            const fs::path stem_path = img_path.parent_path() / img_path.stem();

            std::vector<fs::path> keys;
            std::unordered_set<std::string> seen_keys;
            keys.reserve(1 + 2 * MASK_SEARCH_EXTENSIONS.size());

            auto append_key = [&](const fs::path& key) {
                const std::string normalized_key = detail::normalize_lookup_key(key);
                if (seen_keys.insert(normalized_key).second) {
                    keys.push_back(key);
                }
            };

            append_key(img_path);

            for (const auto* ext : MASK_SEARCH_EXTENSIONS) {
                fs::path target = stem_path;
                target += ext;
                append_key(target);
            }

            for (const auto* ext : MASK_SEARCH_EXTENSIONS) {
                fs::path target = img_path;
                target += ext;
                append_key(target);
            }

            return keys;
        }

        std::vector<RecursiveFileCache> dir_indices_;
    };

    // Pre-scanned directory cache for fast case-insensitive depth-map lookups.
    class DepthDirCache {
    public:
        explicit DepthDirCache(const fs::path& base_path,
                               const CancelCallback& cancel_requested = nullptr) {
            for (const auto* folder : DEPTH_SEARCH_FOLDERS) {
                detail::throw_if_scan_cancel_requested(cancel_requested,
                                                       "Depth directory scan cancelled");
                const fs::path depth_dir = base_path / folder;
                if (!safe_is_directory(depth_dir))
                    continue;

                dir_indices_.emplace_back(depth_dir, cancel_requested);
            }
        }

        [[nodiscard]] FileLookupResult lookup(const std::string& image_name) const {
            if (dir_indices_.empty())
                return {};

            const std::vector<fs::path> lookup_keys = build_lookup_keys(image_name);
            bool saw_ambiguous_match = false;

            for (const auto& dir_index : dir_indices_) {
                for (const auto& key : lookup_keys) {
                    if (auto result = dir_index.lookup(key); result.found()) {
                        return result;
                    } else if (result.ambiguous()) {
                        saw_ambiguous_match = true;
                    }
                }
            }

            if (saw_ambiguous_match) {
                return FileLookupResult{LookupStatus::Ambiguous, {}};
            }

            return {};
        }

        [[nodiscard]] fs::path find(const std::string& image_name) const {
            if (auto result = lookup(image_name); result.found()) {
                return result.path;
            }
            return {};
        }

    private:
        static std::vector<fs::path> build_lookup_keys(const std::string& image_name) {
            const fs::path img_path = lfs::core::utf8_to_path(image_name);
            const fs::path stem_path = img_path.parent_path() / img_path.stem();

            std::vector<fs::path> keys;
            std::unordered_set<std::string> seen_keys;
            keys.reserve(1 + 2 * DEPTH_SEARCH_EXTENSIONS.size());

            auto append_key = [&](const fs::path& key) {
                const std::string normalized_key = detail::normalize_lookup_key(key);
                if (seen_keys.insert(normalized_key).second) {
                    keys.push_back(key);
                }
            };

            append_key(img_path);

            for (const auto* ext : DEPTH_SEARCH_EXTENSIONS) {
                fs::path target = stem_path;
                target += ext;
                append_key(target);
            }

            for (const auto* ext : DEPTH_SEARCH_EXTENSIONS) {
                fs::path target = img_path;
                target += ext;
                append_key(target);
            }

            return keys;
        }

        std::vector<RecursiveFileCache> dir_indices_;
    };

    inline bool paths_equivalent_or_lexically_equal(const fs::path& lhs, const fs::path& rhs) {
        if (lhs.empty() || rhs.empty()) {
            return false;
        }

        std::error_code ec;
        if (fs::equivalent(lhs, rhs, ec)) {
            return true;
        }
        if (ec) {
            return lhs.lexically_normal() == rhs.lexically_normal();
        }

        return false;
    }

    inline int count_image_files(const fs::path& root_path, const bool recursive) {
        if (!safe_is_directory(root_path)) {
            return 0;
        }

        int count = 0;
        std::error_code ec;

        if (recursive) {
            for (fs::recursive_directory_iterator it(
                     root_path,
                     fs::directory_options::skip_permission_denied,
                     ec),
                 end;
                 !ec && it != end;
                 it.increment(ec)) {
                std::error_code file_ec;
                if (!it->is_regular_file(file_ec) || file_ec)
                    continue;
                if (is_image_file(it->path())) {
                    ++count;
                }
            }
            return count;
        }

        for (fs::directory_iterator it(root_path, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code file_ec;
            if (!it->is_regular_file(file_ec) || file_ec)
                continue;
            if (is_image_file(it->path())) {
                ++count;
            }
        }

        return count;
    }

    struct DatasetInfo {
        fs::path base_path;
        fs::path images_path;
        fs::path sparse_path;
        fs::path masks_path;
        fs::path depths_path;
        bool has_masks = false;
        bool has_depths = false;
        int image_count = 0;
        int mask_count = 0;
        int depth_count = 0;
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
            if (has_colmap_in_root && count_image_files(base_path, false) > 0) {
                info.images_path = base_path;
            }
            if (info.images_path.empty()) {
                info.images_path = base_path / "images";
            }
        }

        if (safe_is_directory(info.images_path)) {
            const bool recursive_image_scan =
                !paths_equivalent_or_lexically_equal(info.images_path, base_path);
            info.image_count = count_image_files(info.images_path, recursive_image_scan);
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
                info.mask_count = count_image_files(info.masks_path, true);
                break;
            }
        }

        for (const auto* name : DEPTH_SEARCH_FOLDERS) {
            if (safe_is_directory(base_path / name)) {
                info.depths_path = base_path / name;
                info.has_depths = true;
                info.depth_count = count_image_files(info.depths_path, true);
                break;
            }
        }

        return info;
    }

} // namespace lfs::io
