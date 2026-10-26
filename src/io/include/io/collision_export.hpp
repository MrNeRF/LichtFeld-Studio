/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include "io/export_progress.hpp"
#include <filesystem>
namespace lfs::io {
    // Mixed blocks cannot collapse. An octree with M leaves needs at least
    // ceil((M-1)/7) internal nodes; solid records are intentionally not counted.
    LFS_IO_API void checkCollisionExportMixedBlocks(uint64_t mixed);

    LFS_IO_API core::CollisionSource loadCollisionSource(const std::filesystem::path&,
                                                         std::stop_token = {}, ExportProgressCallback = {});
    // Callers approve replacement of BOTH files before calling. Publication rolls
    // back ordinary failures, but is not crash-atomic or atomic for concurrent readers.
    LFS_IO_API void exportCollision(const core::CollisionVolume&, const std::filesystem::path&,
                                    const glm::mat4& transform = glm::mat4(1),
                                    std::stop_token = {}, ExportProgressCallback = {});
} // namespace lfs::io
