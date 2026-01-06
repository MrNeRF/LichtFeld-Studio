/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

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

    struct PlySaveOptions {
        std::filesystem::path output_path;
        bool binary = true;
        bool async = false;
        ExportProgressCallback progress_callback = nullptr;
    };

    /**
     * @brief Save SplatData to PLY file
     * @return Result<void> - success or Error with details
     * @note When async=true, returns immediately. Use returned future to check result.
     */
    [[nodiscard]] Result<void> save_ply(const SplatData& splat_data, const PlySaveOptions& options);
    [[nodiscard]] Result<void> save_ply(const PointCloud& point_cloud, const PlySaveOptions& options);

    PointCloud to_point_cloud(const SplatData& splat_data);
    std::vector<std::string> get_ply_attribute_names(const SplatData& splat_data);

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
    [[nodiscard]] Result<void> save_sog(const SplatData& splat_data, const SogSaveOptions& options);

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
    [[nodiscard]] Result<void> export_html(const SplatData& splat_data, const HtmlExportOptions& options);

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
    [[nodiscard]] Result<void> save_spz(const SplatData& splat_data, const SpzSaveOptions& options);

} // namespace lfs::io
