/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file checkpoint.hpp
 * @brief Training checkpoint save/load (.resume files)
 *
 * Format types and read-only functions live in core/checkpoint_format.hpp.
 * This header provides save/load that depend on training types (IStrategy, BilateralGrid, PPISP).
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

    inline constexpr std::string_view kCheckpointDirectoryName = "checkpoints";
    inline constexpr std::string_view kCheckpointFilename = "checkpoint.resume";

    [[nodiscard]] inline std::filesystem::path checkpoint_directory(
        const std::filesystem::path& output_path) {
        return output_path / kCheckpointDirectoryName;
    }

    [[nodiscard]] inline std::filesystem::path checkpoint_output_path(
        const std::filesystem::path& output_path) {
        return checkpoint_directory(output_path) / kCheckpointFilename;
    }

    class IStrategy;
    class BilateralGrid;
    class PPISP;
    class PPISPControllerPool;
    class ADMMSparsityOptimizer;

    struct CheckpointStreamResult {
        lfs::core::CheckpointHeader header;
        std::uint64_t bytes = 0;
    };

    // Serialize the exact LFKP stream to a seekable destination. This is shared
    // by the legacy .resume writer and the .licht snapshot service so the CKPT
    // chapter is byte-for-byte the existing checkpoint format.
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

    /// Save complete training checkpoint
    std::expected<void, std::string> save_checkpoint(
        const std::filesystem::path& path,
        int iteration,
        const IStrategy& strategy,
        const lfs::core::param::TrainingParameters& params,
        const BilateralGrid* bilateral_grid,
        const PPISP* ppisp,
        const PPISPControllerPool* ppisp_controller_pool,
        const ADMMSparsityOptimizer* sparsity_optimizer);

    /// Load complete training checkpoint (strategy + optional appearance components)
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
