/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/sidecar_discovery.hpp"

#include "core/camera.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/source_site.hpp"
#include "io/filesystem_utils.hpp"

#include <filesystem>
#include <format>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace lfs::io {

    namespace {

        [[nodiscard]] bool needs_mask_discovery(const lfs::core::Camera& cam) {
            return !cam.has_mask() &&
                   !cam.mask_asset_uuid().has_value() &&
                   (!cam.image_name().empty() || !cam.image_path().empty());
        }

        [[nodiscard]] std::string format_ambiguous_mask_message(
            const lfs::core::Camera& cam,
            const std::filesystem::path& dataset_root) {
            const std::string display_name = !cam.image_name().empty()
                                                 ? cam.image_name()
                                                 : lfs::core::path_to_utf8(cam.image_path());
            return std::format(
                "Mask for image '{}' is ambiguous across the dataset mask folders ({}). "
                "Keep mask files in the same relative subdirectories as the images or rename them uniquely.",
                display_name,
                lfs::core::path_to_utf8(dataset_root));
        }

        [[nodiscard]] std::vector<std::filesystem::path> build_base_stamp_dirs(
            const std::filesystem::path& norm_root,
            const std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
            const std::vector<std::filesystem::path>& discovered_dirs) {
            std::vector<std::filesystem::path> dirs;
            std::unordered_set<std::string> seen;

            const auto append_dir = [&](const std::filesystem::path& dir) {
                if (dir.empty()) {
                    return;
                }
                const std::string key = detail::normalize_lookup_key(dir);
                if (seen.insert(key).second) {
                    dirs.push_back(dir);
                }
            };

            append_dir(norm_root);
            for (const auto* folder : MASK_SEARCH_FOLDERS) {
                append_dir(norm_root / folder);
            }
            for (const auto& dir : discovered_dirs) {
                append_dir(dir);
            }
            for (const auto& cam : cameras) {
                if (cam && !cam->mask_path().empty()) {
                    append_dir(cam->mask_path().parent_path());
                }
            }
            return dirs;
        }

    } // namespace

    void MaskDiscoveryCache::clear() noexcept {
        dataset_root_.clear();
        dir_stamps_.clear();
        camera_states_.clear();
    }

    bool MaskDiscoveryCache::matches(
        const std::filesystem::path& norm_root,
        const std::span<const std::shared_ptr<lfs::core::Camera>> cameras) const {
        if (dataset_root_.empty() || dataset_root_ != norm_root || dir_stamps_.empty()) {
            return false;
        }

        for (const auto& stamp : dir_stamps_) {
            const bool exists = safe_is_directory(stamp.path);
            if (exists != stamp.exists) {
                return false;
            }
            if (exists) {
                std::error_code ec;
                const auto mtime = lfs::core::native_fs::last_write_time(stamp.path, ec);
                if (ec || mtime != stamp.mtime) {
                    return false;
                }
            }
        }

        for (const auto& cam : cameras) {
            if (!cam) {
                continue;
            }
            const auto it = camera_states_.find(cam.get());
            if (it == camera_states_.end()) {
                return false;
            }
            const auto locked = it->second.camera.lock();
            if (locked.get() != cam.get()) {
                return false;
            }
            if (it->second.mask_path != cam->mask_path() ||
                it->second.has_in_memory != cam->has_in_memory_mask() ||
                it->second.asset_uuid != cam->mask_asset_uuid()) {
                return false;
            }
        }

        return true;
    }

    void MaskDiscoveryCache::update(
        std::filesystem::path norm_root,
        std::vector<DirStamp> dir_stamps,
        const std::span<const std::shared_ptr<lfs::core::Camera>> cameras) {
        dataset_root_ = std::move(norm_root);
        dir_stamps_ = std::move(dir_stamps);
        camera_states_.clear();
        camera_states_.reserve(cameras.size());
        for (const auto& cam : cameras) {
            if (!cam) {
                continue;
            }
            camera_states_.emplace(
                cam.get(),
                CameraState{
                    .camera = cam,
                    .mask_path = cam->mask_path(),
                    .has_in_memory = cam->has_in_memory_mask(),
                    .asset_uuid = cam->mask_asset_uuid(),
                });
        }
    }

    lfs::Result<MaskAssociationResult> associate_dataset_masks(
        const std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
        const std::filesystem::path& dataset_path,
        const MaskAssociationOptions& options,
        MaskDiscoveryCache* const cache) {
        MaskAssociationResult result;
        if (cameras.empty()) {
            return result;
        }

        const std::filesystem::path dataset_root = resolve_dataset_root(dataset_path);
        if (dataset_root.empty() || !safe_is_directory(dataset_root)) {
            if (cache) {
                cache->clear();
            }
            return result;
        }

        if (cache && cache->matches(dataset_root, cameras)) {
            return result;
        }

        std::vector<size_t> missing_indices;
        for (size_t i = 0; i < cameras.size(); ++i) {
            if (cameras[i] && needs_mask_discovery(*cameras[i])) {
                missing_indices.push_back(i);
            }
        }

        std::vector<std::filesystem::path> discovered_dirs;
        if (!missing_indices.empty()) {
            MaskDirCache mask_cache(
                dataset_root,
                options.cancel_requested,
                [&](const std::filesystem::path& dir) {
                    if (cache) {
                        discovered_dirs.push_back(dir);
                    }
                });

            if (mask_cache.has_mask_dirs()) {
                bool warned_ambiguous = false;
                for (size_t step = 0; step < missing_indices.size(); ++step) {
                    if ((step % detail::CANCEL_POLL_INTERVAL) == 0 &&
                        options.cancel_requested &&
                        options.cancel_requested()) {
                        return lfs::make_error(lfs::ErrorInit{
                            .code = lfs::ErrorCode::Cancelled,
                            .domain = lfs::ErrorDomain::IO,
                            .severity = lfs::Severity::Warning,
                            .user_message = "Mask discovery cancelled",
                            .detail = "Mask discovery cancelled",
                            .detection = LFS_SOURCE_SITE_CURRENT(),
                        });
                    }

                    const auto& cam = cameras[missing_indices[step]];
                    if (auto lookup = lookup_dataset_mask_normalized(
                            mask_cache, cam->image_path(), cam->image_name(), dataset_root);
                        lookup.found()) {
                        cam->set_mask_path(std::move(lookup.path));
                        ++result.masks_bound;
                    } else if (lookup.ambiguous()) {
                        const std::string msg = format_ambiguous_mask_message(*cam, dataset_root);
                        if (options.fail_on_ambiguous_mask) {
                            return lfs::make_error(lfs::ErrorInit{
                                .code = lfs::ErrorCode::InvalidArgument,
                                .domain = lfs::ErrorDomain::IO,
                                .user_message = msg,
                                .detail = msg,
                                .detection = LFS_SOURCE_SITE_CURRENT(),
                            });
                        }
                        ++result.ambiguous_masks;
                        if (!warned_ambiguous) {
                            LOG_WARN("{}", msg);
                            warned_ambiguous = true;
                        }
                    }
                }
            }
        }

        if (cache) {
            const auto stamp_dirs = build_base_stamp_dirs(dataset_root, cameras, discovered_dirs);
            std::vector<MaskDiscoveryCache::DirStamp> dir_stamps;
            dir_stamps.reserve(stamp_dirs.size());
            for (const auto& dir : stamp_dirs) {
                MaskDiscoveryCache::DirStamp stamp;
                stamp.path = dir;
                stamp.exists = safe_is_directory(dir);
                if (stamp.exists) {
                    std::error_code ec;
                    const auto mtime = lfs::core::native_fs::last_write_time(dir, ec);
                    if (!ec) {
                        stamp.mtime = mtime;
                    } else {
                        stamp.exists = false;
                    }
                }
                dir_stamps.push_back(std::move(stamp));
            }
            cache->update(dataset_root, std::move(dir_stamps), cameras);
        }

        return result;
    }

} // namespace lfs::io
