/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/path_utils.hpp"
#include "io/loader.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <functional>
#include <span>
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

    inline constexpr std::array<const char*, 2> NORMAL_SEARCH_FOLDERS = {
        "normal",
        "normals",
    };

    inline constexpr std::array<const char*, 3> NORMAL_SEARCH_EXTENSIONS = {
        ".png",
        ".tif",
        ".tiff",
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

                if (const std::string digit_key =
                        trailing_digit_run(entry.path().stem().string());
                    !digit_key.empty()) {
                    if (auto [it_digits, inserted] =
                            digit_entries_.emplace(digit_key, entry.path());
                        !inserted && it_digits->second != entry.path()) {
                        ambiguous_digits_.insert(digit_key);
                    }
                }
            }
        }

        // Trailing digit run of a filename stem ("RENDER_0042" -> "0042").
        // Runs shorter than two digits are ignored to avoid accidental pairing.
        [[nodiscard]] static std::string trailing_digit_run(const std::string& stem) {
            size_t end = stem.size();
            size_t begin = end;
            while (begin > 0 && std::isdigit(static_cast<unsigned char>(stem[begin - 1]))) {
                --begin;
            }
            if (end - begin < 2) {
                return {};
            }
            return stem.substr(begin, end - begin);
        }

        // Frame-index fallback for sidecar files whose names share only the
        // numeric suffix with the image (e.g. RENDER_0042.png / DEPTH_0042.png).
        [[nodiscard]] FileLookupResult lookup_by_digit_suffix(const std::string& digit_key) const {
            if (digit_key.empty()) {
                return {};
            }
            if (ambiguous_digits_.contains(digit_key)) {
                return FileLookupResult{LookupStatus::Ambiguous, {}};
            }
            if (auto it = digit_entries_.find(digit_key); it != digit_entries_.end()) {
                return FileLookupResult{LookupStatus::Found, it->second};
            }
            return {};
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
        std::unordered_map<std::string, fs::path> digit_entries_;
        std::unordered_set<std::string> ambiguous_digits_;
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

    // Pre-scanned directory cache for fast case-insensitive sidecar lookups
    // (depth maps, normal maps) with a trailing-frame-number fallback.
    class SidecarDirCache {
    public:
        SidecarDirCache(const fs::path& base_path,
                        std::span<const char* const> search_folders,
                        std::span<const char* const> search_extensions,
                        const CancelCallback& cancel_requested = nullptr)
            : extensions_(search_extensions) {
            for (const auto* folder : search_folders) {
                detail::throw_if_scan_cancel_requested(cancel_requested,
                                                       "Sidecar directory scan cancelled");
                const fs::path sidecar_dir = base_path / folder;
                if (!safe_is_directory(sidecar_dir))
                    continue;

                dir_indices_.emplace_back(sidecar_dir, cancel_requested);
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

            // Frame-number fallback. Ambiguity here means the convention does
            // not apply (e.g. several sidecar kinds per frame), so it degrades
            // to "no match" instead of failing the dataset.
            const std::string digit_key = RecursiveFileCache::trailing_digit_run(
                lfs::core::utf8_to_path(image_name).stem().string());
            for (const auto& dir_index : dir_indices_) {
                if (auto result = dir_index.lookup_by_digit_suffix(digit_key); result.found()) {
                    return result;
                }
            }

            if (saw_ambiguous_match) {
                return FileLookupResult{LookupStatus::Ambiguous, {}};
            }

            return {};
        }

        [[nodiscard]] bool has_dirs() const { return !dir_indices_.empty(); }

        [[nodiscard]] fs::path find(const std::string& image_name) const {
            if (auto result = lookup(image_name); result.found()) {
                return result.path;
            }
            return {};
        }

    private:
        std::vector<fs::path> build_lookup_keys(const std::string& image_name) const {
            const fs::path img_path = lfs::core::utf8_to_path(image_name);
            const fs::path stem_path = img_path.parent_path() / img_path.stem();

            std::vector<fs::path> keys;
            std::unordered_set<std::string> seen_keys;
            keys.reserve(1 + 2 * extensions_.size());

            auto append_key = [&](const fs::path& key) {
                const std::string normalized_key = detail::normalize_lookup_key(key);
                if (seen_keys.insert(normalized_key).second) {
                    keys.push_back(key);
                }
            };

            append_key(img_path);

            for (const auto* ext : extensions_) {
                fs::path target = stem_path;
                target += ext;
                append_key(target);
            }

            for (const auto* ext : extensions_) {
                fs::path target = img_path;
                target += ext;
                append_key(target);
            }

            return keys;
        }

        std::span<const char* const> extensions_;
        std::vector<RecursiveFileCache> dir_indices_;
    };

    // Pre-scanned directory cache for fast case-insensitive depth-map lookups.
    class DepthDirCache : public SidecarDirCache {
    public:
        explicit DepthDirCache(const fs::path& base_path,
                               const CancelCallback& cancel_requested = nullptr)
            : SidecarDirCache(base_path, DEPTH_SEARCH_FOLDERS, DEPTH_SEARCH_EXTENSIONS,
                              cancel_requested) {}

        [[nodiscard]] bool has_depth_dirs() const { return has_dirs(); }
    };

    // Pre-scanned directory cache for fast case-insensitive normal-map lookups.
    class NormalDirCache : public SidecarDirCache {
    public:
        explicit NormalDirCache(const fs::path& base_path,
                                const CancelCallback& cancel_requested = nullptr)
            : SidecarDirCache(base_path, NORMAL_SEARCH_FOLDERS, NORMAL_SEARCH_EXTENSIONS,
                              cancel_requested) {}

        [[nodiscard]] bool has_normal_dirs() const { return has_dirs(); }
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
        fs::path normals_path;
        bool has_masks = false;
        bool has_depths = false;
        bool has_normals = false;
        int image_count = 0;
        int mask_count = 0;
        int depth_count = 0;
        int normal_count = 0;
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

        for (const auto* name : NORMAL_SEARCH_FOLDERS) {
            if (safe_is_directory(base_path / name)) {
                info.normals_path = base_path / name;
                info.has_normals = true;
                info.normal_count = count_image_files(info.normals_path, true);
                break;
            }
        }

        return info;
    }

} // namespace lfs::io
