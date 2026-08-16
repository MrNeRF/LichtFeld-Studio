/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file checkpoint.hpp
 * @brief Embedded checkpoint serialization and legacy .resume import
 *
 * Format types and read-only functions live in core/checkpoint_format.hpp.
 * This header provides embedded serialization and import-only loading that
 * depend on training types (IStrategy, BilateralGrid, PPISP).
 */

#include "core/checkpoint_format.hpp"
#include "core/error.hpp"
#include "core/parameters.hpp"
#include <cstdint>
#include <expected>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::training {

    class IStrategy;
    class BilateralGrid;
    class PPISP;
    class PPISPControllerPool;
    class ADMMSparsityOptimizer;

    struct CheckpointStreamResult {
        lfs::core::CheckpointHeader header;
        std::uint64_t bytes = 0;
    };

    // Serialize the exact LFKP stream to a seekable destination for embedding
    // in a .licht CKPT chapter.
    [[nodiscard]] lfs::Result<CheckpointStreamResult>
    serialize_checkpoint(
        std::ostream& destination,
        int iteration,
        const IStrategy& strategy,
        const lfs::core::param::TrainingParameters& params,
        const BilateralGrid* bilateral_grid,
        const PPISP* ppisp,
        const PPISPControllerPool* ppisp_controller_pool,
        const ADMMSparsityOptimizer* sparsity_optimizer);

    /// Import a standalone legacy checkpoint.
    std::expected<int, std::string> load_checkpoint(
        const std::filesystem::path& path,
        IStrategy& strategy,
        lfs::core::param::TrainingParameters& params,
        BilateralGrid* bilateral_grid,
        PPISP* ppisp,
        PPISPControllerPool* ppisp_controller_pool,
        ADMMSparsityOptimizer* sparsity_optimizer,
        lfs::core::SplatTensorAllocator tensor_allocator = {});
    using CheckpointLoadResult = decltype(load_checkpoint(
        std::filesystem::path{},
        std::declval<IStrategy&>(),
        std::declval<
            lfs::core::param::TrainingParameters&>(),
        nullptr, nullptr, nullptr, nullptr));

    /// Load a complete checkpoint from a bounded, seekable CKPT stream.
    CheckpointLoadResult load_checkpoint(
        std::istream& source,
        std::uint64_t source_bytes,
        IStrategy& strategy,
        lfs::core::param::TrainingParameters& params,
        BilateralGrid* bilateral_grid,
        PPISP* ppisp,
        PPISPControllerPool* ppisp_controller_pool,
        ADMMSparsityOptimizer* sparsity_optimizer,
        lfs::core::SplatTensorAllocator tensor_allocator = {},
        std::string_view source_name = "embedded CKPT");

} // namespace lfs::training
