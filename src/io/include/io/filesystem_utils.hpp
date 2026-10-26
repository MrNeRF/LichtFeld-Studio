/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "io/loader.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::io {

    inline constexpr std::array<std::string_view, 2> TRANSFORMS_DATASET_MARKERS = {
        "transforms.json", "transforms_train.json"};

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

        inline std::string normalize_lookup_key(const std::filesystem::path& value) {
            return normalize_lookup_key(lfs::core::path_to_utf8(value.lexically_normal()));
        }

        inline std::string raw_lookup_key(const std::filesystem::path& value) {
            const auto utf8 = value.generic_u8string();
            return {utf8.begin(), utf8.end()};
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

    // Priors may use any resolution, with up to 1% aspect-ratio rounding error.
    [[nodiscard]] inline bool sidecar_dimensions_match_contract(
        const int sidecar_width,
        const int sidecar_height,
        const int requested_width,
        const int requested_height) noexcept {
        if (requested_width <= 0 || requested_height <= 0 || sidecar_width <= 0 || sidecar_height <= 0) {
            return false;
        }
        const double ratio = (static_cast<double>(sidecar_width) * requested_height) /
                             (static_cast<double>(sidecar_height) * requested_width);
        return std::abs(ratio - 1.0) <= 0.01;
    }

    // Safe filesystem operations that don't throw
    inline bool safe_exists(const std::filesystem::path& path) {
        std::error_code ec;
        return lfs::core::native_fs::exists(path, ec);
    }

    inline bool safe_is_directory(const std::filesystem::path& path) {
        std::error_code ec;
        return lfs::core::native_fs::is_directory(path, ec);
    }

    inline bool safe_is_regular_file(const std::filesystem::path& path) {
        std::error_code ec;
        return lfs::core::native_fs::is_regular_file(path, ec);
    }

    // Case-insensitive file finding
    inline std::filesystem::path find_file_ci(const std::filesystem::path& dir, const std::string& target) {
        if (!safe_exists(dir) || !safe_is_directory(dir))
            return {};

        std::string target_lower = detail::normalize_lookup_key(target);

        std::error_code ec;
        for (std::filesystem::directory_iterator it(lfs::core::native_fs::traversal_path(dir), ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code file_ec;
            if (!it->is_regular_file(file_ec) || file_ec)
                continue;

            std::string name = detail::normalize_lookup_key(it->path().filename());
            if (name == target_lower) {
                return lfs::core::native_fs::entry_path(dir, it->path());
            }
        }
        return {};
    }

    // Find file in multiple locations (case-insensitive)
    inline std::filesystem::path find_file_in_paths(const std::vector<std::filesystem::path>& search_paths,
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
        std::filesystem::path path;

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
        // Optional observer receives each regular file's path relative to root_path
        // during the existing filesystem walk.
        explicit RecursiveFileCache(const std::filesystem::path& root_path,
                                    const CancelCallback& cancel_requested = nullptr,
                                    const std::function<void(const std::filesystem::path&)>& on_file = nullptr) {
            if (!safe_is_directory(root_path))
                return;

            std::error_code ec;
            size_t scanned_entries = 0;
            const auto traversal_root = lfs::core::native_fs::traversal_path(root_path);
            for (std::filesystem::recursive_directory_iterator it(lfs::core::native_fs::traversal_path(traversal_root),
                                                                  std::filesystem::directory_options::skip_permission_denied,
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

                const std::filesystem::path rel = entry.path().lexically_relative(traversal_root);
                if (rel.empty())
                    continue;
                const std::filesystem::path file = root_path / rel;

                if (on_file)
                    on_file(rel);

                raw_entries_.emplace(detail::raw_lookup_key(rel), file);

                const std::string rel_key = detail::normalize_lookup_key(rel);
                if (auto [it_exact, inserted] = exact_entries_.emplace(rel_key, file);
                    !inserted && it_exact->second != file) {
                    ambiguous_exact_.insert(rel_key);
                }

                const std::string basename_key =
                    detail::normalize_lookup_key(entry.path().filename());
                if (auto [it_basename, inserted] =
                        basename_entries_.emplace(basename_key, file);
                    !inserted && it_basename->second != file) {
                    ambiguous_basenames_.insert(basename_key);
                }

                if (const std::string digit_key =
                        trailing_digit_run(lfs::core::path_to_utf8(entry.path().stem()));
                    !digit_key.empty()) {
                    if (auto [it_digits, inserted] =
                            digit_entries_.emplace(digit_key, file);
                        !inserted && it_digits->second != file) {
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

        [[nodiscard]] FileLookupResult lookup_exact(const std::filesystem::path& relative_or_name) const {
            if (relative_or_name.empty())
                return {};

            if (auto it = raw_entries_.find(detail::raw_lookup_key(relative_or_name));
                it != raw_entries_.end()) {
                return FileLookupResult{LookupStatus::Found, it->second};
            }

            const std::string exact_key =
                detail::normalize_lookup_key(relative_or_name);
            if (ambiguous_exact_.contains(exact_key))
                return FileLookupResult{LookupStatus::Ambiguous, {}};
            if (auto it = exact_entries_.find(exact_key);
                it != exact_entries_.end()) {
                return FileLookupResult{LookupStatus::Found, it->second};
            }

            return {};
        }

        [[nodiscard]] FileLookupResult lookup(const std::filesystem::path& relative_or_name) const {
            if (relative_or_name.empty())
                return {};
            if (auto result = lookup_exact(relative_or_name); result.status != LookupStatus::NotFound)
                return result;

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

        [[nodiscard]] std::filesystem::path find(const std::filesystem::path& relative_or_name) const {
            if (auto result = lookup(relative_or_name); result.found()) {
                return result.path;
            }
            return {};
        }

    private:
        std::unordered_map<std::string, std::filesystem::path> raw_entries_;
        std::unordered_map<std::string, std::filesystem::path> exact_entries_;
        std::unordered_map<std::string, std::filesystem::path> basename_entries_;
        std::unordered_set<std::string> ambiguous_exact_;
        std::unordered_set<std::string> ambiguous_basenames_;
        std::unordered_map<std::string, std::filesystem::path> digit_entries_;
        std::unordered_set<std::string> ambiguous_digits_;
    };

    // Get standard COLMAP search paths for a base directory
    inline std::vector<std::filesystem::path> get_colmap_search_paths(const std::filesystem::path& base) {
        return {
            base / "sparse" / "0", // Standard COLMAP
            base / "sparse",       // Alternative COLMAP
            base                   // Reality Capture / flat structure
        };
    }

    // Check if a file has an image extension
    inline bool is_image_file(const std::filesystem::path& path) {
        static const std::vector<std::string> image_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"};

        std::string ext = lfs::core::path_to_utf8(path.extension());
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
        // Indexes one explicit mask folder instead of the dataset's mask sub-folders.
        [[nodiscard]] static MaskDirCache for_folder(const std::filesystem::path& mask_dir) {
            MaskDirCache cache;
            if (safe_is_directory(mask_dir))
                cache.dir_indices_.emplace_back(mask_dir, nullptr);
            return cache;
        }

        explicit MaskDirCache(const std::filesystem::path& base_path,
                              const CancelCallback& cancel_requested = nullptr) {
            for (const auto* folder : MASK_SEARCH_FOLDERS) {
                detail::throw_if_scan_cancel_requested(cancel_requested,
                                                       "Mask directory scan cancelled");
                const std::filesystem::path mask_dir = base_path / folder;
                if (!safe_is_directory(mask_dir))
                    continue;

                dir_indices_.emplace_back(mask_dir, cancel_requested);
            }
        }

        [[nodiscard]] FileLookupResult lookup(const std::string& image_name) const {
            if (dir_indices_.empty())
                return {};

            const std::vector<std::filesystem::path> lookup_keys = build_lookup_keys(image_name);
            bool saw_ambiguous_match = false;

            // Check every relative-path/extension candidate before considering a
            // basename match, which may belong to a different camera subfolder.
            for (const auto& dir_index : dir_indices_) {
                for (const auto& key : lookup_keys) {
                    if (auto result = dir_index.lookup_exact(key); result.found()) {
                        return result;
                    } else if (result.ambiguous()) {
                        saw_ambiguous_match = true;
                    }
                }
            }
            if (saw_ambiguous_match)
                return FileLookupResult{LookupStatus::Ambiguous, {}};

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

        [[nodiscard]] std::filesystem::path find(const std::string& image_name) const {
            if (auto result = lookup(image_name); result.found()) {
                return result.path;
            }
            return {};
        }

    private:
        MaskDirCache() = default;

        static std::vector<std::filesystem::path> build_lookup_keys(const std::string& image_name) {
            const std::filesystem::path img_path = lfs::core::utf8_to_path(image_name);
            const std::filesystem::path stem_path = img_path.parent_path() / img_path.stem();

            std::vector<std::filesystem::path> keys;
            std::unordered_set<std::string> seen_keys;
            keys.reserve(1 + 2 * MASK_SEARCH_EXTENSIONS.size());

            auto append_key = [&](const std::filesystem::path& key) {
                const std::string normalized_key = detail::normalize_lookup_key(key);
                if (seen_keys.insert(normalized_key).second) {
                    keys.push_back(key);
                }
            };

            append_key(img_path);

            for (const auto* ext : MASK_SEARCH_EXTENSIONS) {
                std::filesystem::path target = stem_path;
                target += ext;
                append_key(target);
            }

            for (const auto* ext : MASK_SEARCH_EXTENSIONS) {
                std::filesystem::path target = img_path;
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
        SidecarDirCache(const std::filesystem::path& base_path,
                        std::span<const char* const> search_folders,
                        std::span<const char* const> search_extensions,
                        const CancelCallback& cancel_requested = nullptr)
            : extensions_(search_extensions) {
            for (const auto* folder : search_folders) {
                detail::throw_if_scan_cancel_requested(cancel_requested,
                                                       "Sidecar directory scan cancelled");
                const std::filesystem::path sidecar_dir = base_path / folder;
                if (!safe_is_directory(sidecar_dir))
                    continue;

                dir_indices_.emplace_back(sidecar_dir, cancel_requested);
            }
        }

        [[nodiscard]] FileLookupResult lookup(const std::string& image_name) const {
            if (dir_indices_.empty())
                return {};

            const std::vector<std::filesystem::path> lookup_keys = build_lookup_keys(image_name);
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
                lfs::core::path_to_utf8(lfs::core::utf8_to_path(image_name).stem()));
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

        [[nodiscard]] std::filesystem::path find(const std::string& image_name) const {
            if (auto result = lookup(image_name); result.found()) {
                return result.path;
            }
            return {};
        }

    private:
        std::vector<std::filesystem::path> build_lookup_keys(const std::string& image_name) const {
            const std::filesystem::path img_path = lfs::core::utf8_to_path(image_name);
            const std::filesystem::path stem_path = img_path.parent_path() / img_path.stem();

            std::vector<std::filesystem::path> keys;
            std::unordered_set<std::string> seen_keys;
            keys.reserve(1 + 2 * extensions_.size());

            auto append_key = [&](const std::filesystem::path& key) {
                const std::string normalized_key = detail::normalize_lookup_key(key);
                if (seen_keys.insert(normalized_key).second) {
                    keys.push_back(key);
                }
            };

            append_key(img_path);

            for (const auto* ext : extensions_) {
                std::filesystem::path target = stem_path;
                target += ext;
                append_key(target);
            }

            for (const auto* ext : extensions_) {
                std::filesystem::path target = img_path;
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
        explicit DepthDirCache(const std::filesystem::path& base_path,
                               const CancelCallback& cancel_requested = nullptr)
            : SidecarDirCache(base_path, DEPTH_SEARCH_FOLDERS, DEPTH_SEARCH_EXTENSIONS,
                              cancel_requested) {}

        [[nodiscard]] bool has_depth_dirs() const { return has_dirs(); }
    };

    // Pre-scanned directory cache for fast case-insensitive normal-map lookups.
    class NormalDirCache : public SidecarDirCache {
    public:
        explicit NormalDirCache(const std::filesystem::path& base_path,
                                const CancelCallback& cancel_requested = nullptr)
            : SidecarDirCache(base_path, NORMAL_SEARCH_FOLDERS, NORMAL_SEARCH_EXTENSIONS,
                              cancel_requested) {}

        [[nodiscard]] bool has_normal_dirs() const { return has_dirs(); }
    };

    inline bool paths_equivalent_or_lexically_equal(const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
        if (lhs.empty() || rhs.empty()) {
            return false;
        }

        std::error_code ec;
        if (lfs::core::native_fs::equivalent(lhs, rhs, ec)) {
            return true;
        }
        if (ec) {
            return lhs.lexically_normal() == rhs.lexically_normal();
        }

        return false;
    }

    inline int count_image_files(const std::filesystem::path& root_path, const bool recursive) {
        if (!safe_is_directory(root_path)) {
            return 0;
        }

        int count = 0;
        std::error_code ec;

        if (recursive) {
            for (std::filesystem::recursive_directory_iterator it(lfs::core::native_fs::traversal_path(root_path),
                                                                  std::filesystem::directory_options::skip_permission_denied,
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

        for (std::filesystem::directory_iterator it(lfs::core::native_fs::traversal_path(root_path), ec), end; !ec && it != end; it.increment(ec)) {
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
        std::filesystem::path base_path;
        std::filesystem::path images_path;
        std::filesystem::path sparse_path;
        std::filesystem::path masks_path;
        std::filesystem::path depths_path;
        std::filesystem::path normals_path;
        bool has_masks = false;
        bool has_depths = false;
        bool has_normals = false;
        int image_count = 0;
        int mask_count = 0;
        int depth_count = 0;
        int normal_count = 0;
    };

    // A given images_path (absolute, or relative to base_path) replaces the search, as for images kept outside the
    // dataset folder.
    inline DatasetInfo detect_dataset_info(const std::filesystem::path& base_path,
                                           const std::filesystem::path& images_path = {}) {
        static constexpr const char* const IMAGE_FOLDERS[] = {"images", "images_4", "images_2", "images_8", "input", "rgb"};

        DatasetInfo info;
        info.base_path = base_path;

        if (!images_path.empty())
            info.images_path = images_path.is_absolute() ? images_path : base_path / images_path;
        for (const auto* name : IMAGE_FOLDERS) {
            if (!info.images_path.empty())
                break;
            if (safe_is_directory(base_path / name))
                info.images_path = base_path / name;
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
