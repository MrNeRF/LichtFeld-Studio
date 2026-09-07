/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "io/project_chapters.hpp"
#include "io/project_document.hpp"

#include <filesystem>
#include <functional>
#include <optional>

namespace lfs::io::project {

    // Digest of one dataset file, matching EmbeddedDatasetEntry::xxh3_128.
    [[nodiscard]] LFS_IO_API lfs::Result<Hash128>
    hash_dataset_file(const std::filesystem::path& path);

    // Per-user cache folder that receives this project's extracted dataset.
    [[nodiscard]] LFS_IO_API lfs::Result<std::filesystem::path>
    embedded_dataset_cache_dir(const ProjectDocument& document);

    // Unpack the complete embedded dataset into cache_dir and publish a
    // ".complete" marker. Files already present with a matching hash are kept.
    // Returns nullopt when the document carries no complete embedded dataset.
    // progress may return false to cancel.
    [[nodiscard]] LFS_IO_API lfs::Result<std::optional<std::filesystem::path>>
    extract_embedded_dataset(const ProjectDocument& document,
                             const std::filesystem::path& cache_dir,
                             const std::function<bool(float)>& progress = {});

    // Extract into the per-user cache unless the external dataset folder
    // recorded in REFS still exists, then unpacking is not necessary and images
    // can be read from REFS. Returns the extracted root, or nullopt when nothing
    // needed extracting.
    [[nodiscard]] LFS_IO_API lfs::Result<std::optional<std::filesystem::path>>
    extract_embedded_dataset_if_needed(
        const ProjectDocument& document,
        const std::function<bool(float)>& progress = {});

} // namespace lfs::io::project
