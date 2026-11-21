/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/uuid.hpp"
#include "io/loader.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace lfs::core {
    class Camera;
} // namespace lfs::core

namespace lfs::io {

    struct MaskAssociationOptions {
        bool fail_on_ambiguous_mask = true;
        CancelCallback cancel_requested = {};
    };

    struct MaskAssociationResult {
        size_t masks_bound = 0;
        size_t ambiguous_masks = 0;
    };

    class LFS_IO_API MaskDiscoveryCache {
    public:
        void clear() noexcept;

    private:
        struct DirStamp {
            std::filesystem::path path;
            bool exists = false;
            std::filesystem::file_time_type mtime{};
        };

        struct CameraState {
            std::weak_ptr<const lfs::core::Camera> camera;
            std::filesystem::path mask_path;
            bool has_in_memory = false;
            std::optional<lfs::core::Uuid> asset_uuid;
        };

        [[nodiscard]] bool matches(
            const std::filesystem::path& norm_root,
            std::span<const std::shared_ptr<lfs::core::Camera>> cameras) const;

        void update(
            std::filesystem::path norm_root,
            std::vector<DirStamp> dir_stamps,
            std::span<const std::shared_ptr<lfs::core::Camera>> cameras);

        std::filesystem::path dataset_root_;
        std::vector<DirStamp> dir_stamps_;
        std::unordered_map<const lfs::core::Camera*, CameraState> camera_states_;

        friend LFS_IO_API lfs::Result<MaskAssociationResult> associate_dataset_masks(
            std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
            const std::filesystem::path& dataset_path,
            const MaskAssociationOptions& options,
            MaskDiscoveryCache* cache);
    };

    // Discovers mask files under dataset_path for cameras that do not currently
    // have a valid file, in-memory mask, or project mask asset bound.
    // When cache is provided, repeats with unchanged mask folder mtimes and
    // camera mask bindings perform no per-camera file stats or directory walks.
    [[nodiscard]] LFS_IO_API lfs::Result<MaskAssociationResult> associate_dataset_masks(
        std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
        const std::filesystem::path& dataset_path,
        const MaskAssociationOptions& options = {},
        MaskDiscoveryCache* cache = nullptr);

} // namespace lfs::io
