/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "io/error.hpp"
#include <filesystem>
#include <functional>
#include <future>
#include <string>
#include <vector>

namespace lfs::io {

    using lfs::core::PointCloud;
    using lfs::core::SplatData;

    // Progress callback for export operations (returns false to cancel)
    using ExportProgressCallback = std::function<bool(float progress, const std::string& stage)>;

    // ============================================================================
    // PLY Export
    // ============================================================================

    struct PlyAttributeBlock {
        // Per-vertex data shaped [N] or [N,C]. The leading dimension must match the exported
        // vertex count. When saving SplatData with a deleted mask, exporters accept either
        // raw-count rows or already-filtered visible-count rows. Values are exported as float32.
        lfs::core::Tensor values;
        // Final PLY property names, one per exported column after [N,C] expansion.
        // names.size() must match the exported column count. Names must be non-empty tokens
        // without whitespace/control characters, must be unique within the block and across the
        // final vertex schema, and must not collide with built-in Gaussian/color property ids.
        std::vector<std::string> names;
    };

    struct PlySaveOptions {
        std::filesystem::path output_path;
        bool binary = true;
        bool async = false;
        ExportProgressCallback progress_callback = nullptr;
        // Additional per-vertex float properties appended after the built-in PLY schema.
        std::vector<PlyAttributeBlock> extra_attributes;
    };

    /**
     * @brief Save SplatData to PLY file
     * @return Result<void> - success or Error with details
     * @note When async=true, returns immediately. Use returned future to check result.
     */
    [[nodiscard]] LFS_IO_API Result<void> save_ply(const SplatData& splat_data, const PlySaveOptions& options);
    [[nodiscard]] LFS_IO_API Result<void> save_ply(const PointCloud& point_cloud, const PlySaveOptions& options);

    LFS_IO_API PointCloud to_point_cloud(const SplatData& splat_data);
    LFS_IO_API std::vector<std::string> get_ply_attribute_names(const SplatData& splat_data);

    // ============================================================================
    // SOG Export (SuperSplat format)
    // ============================================================================

    struct SogSaveOptions {
        std::filesystem::path output_path;
        int kmeans_iterations = 10;
        bool use_gpu = true;
        ExportProgressCallback progress_callback = nullptr;
    };

    /**
     * @brief Save SplatData to SOG (SuperSplat) format
     * @return Result<void> - success or Error with details (disk space, encoding, archive errors)
     */
    [[nodiscard]] LFS_IO_API Result<void> save_sog(const SplatData& splat_data, const SogSaveOptions& options);

    // ============================================================================
    // HTML Viewer Export
    // ============================================================================

    using HtmlProgressCallback = std::function<void(float progress, const std::string& stage)>;

    struct HtmlExportOptions {
        std::filesystem::path output_path;
        int kmeans_iterations = 10;
        HtmlProgressCallback progress_callback = nullptr;
    };

    /**
     * @brief Export SplatData as standalone HTML viewer
     * @return Result<void> - success or Error with details
     */
    [[nodiscard]] LFS_IO_API Result<void> export_html(const SplatData& splat_data, const HtmlExportOptions& options);

    // ============================================================================
    // SPZ Export (Niantic compressed format)
    // ============================================================================

    struct SpzSaveOptions {
        std::filesystem::path output_path;
    };

    /**
     * @brief Save SplatData to SPZ (Niantic compressed gaussian splat) format
     * @return Result<void> - success or Error with details
     */
    [[nodiscard]] LFS_IO_API Result<void> save_spz(const SplatData& splat_data, const SpzSaveOptions& options);

    // ============================================================================
    // USD Export (OpenUSD Gaussian ParticleField)
    // ============================================================================

    struct UsdSaveOptions {
        std::filesystem::path output_path;
    };

    /**
     * @brief Save SplatData to OpenUSD Gaussian ParticleField format
     * @return Result<void> - success or Error with details
     */
    [[nodiscard]] LFS_IO_API Result<void> save_usd(const SplatData& splat_data, const UsdSaveOptions& options);

} // namespace lfs::io
