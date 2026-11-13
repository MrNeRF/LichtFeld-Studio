/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "io/loader.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

namespace lfs::core {
    class Camera;
} // namespace lfs::core

namespace lfs::io {

    struct SidecarAssociationOptions {
        bool masks = false;
        bool depths = false;
        bool normals = false;
        // When true, ambiguous sidecar matches fail with a descriptive error
        // (matching dataset load behavior when the feature is active). When
        // false, ambiguity is logged once and skipped.
        bool fail_on_ambiguous_mask = true;
        bool fail_on_ambiguous_depth = false;
        bool fail_on_ambiguous_normal = false;
        // When true and camera images are readable on disk, sidecars that fail
        // the size/channel contract are left unbound so auto-generate can
        // regenerate and overwrite them.
        bool skip_mismatched_depths = false;
        bool skip_mismatched_normals = false;
        CancelCallback cancel_requested = {};
    };

    struct SidecarAssociationResult {
        size_t masks_bound = 0;
        size_t depths_bound = 0;
        size_t normals_bound = 0;
        size_t ambiguous_masks = 0;
        size_t ambiguous_depths = 0;
        size_t ambiguous_normals = 0;

        [[nodiscard]] size_t total_bound() const noexcept {
            return masks_bound + depths_bound + normals_bound;
        }
    };

    // Resolves the dataset root directory from a directory path or a dataset
    // manifest file (such as transforms.json). Returns an empty path if the
    // input does not resolve to an existing directory.
    [[nodiscard]] LFS_IO_API std::filesystem::path resolve_dataset_root(
        const std::filesystem::path& data_path);

    // Discovers sidecar files (masks, depths, normals) under dataset_path for
    // cameras that do not currently have a valid file/in-memory asset bound.
    // Existing valid paths, in-memory masks, and project mask assets are kept
    // untouched.
    [[nodiscard]] LFS_IO_API lfs::Result<SidecarAssociationResult>
    associate_dataset_sidecars(
        std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
        const std::filesystem::path& dataset_path,
        const SidecarAssociationOptions& options);

} // namespace lfs::io
