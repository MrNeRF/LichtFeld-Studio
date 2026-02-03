/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "io/error.hpp"
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

// Forward declarations only - hide implementation details
namespace lfs::core {
    class SplatData;
    struct PointCloud;
} // namespace lfs::core

namespace lfs::training {
    class CameraDataset;
} // namespace lfs::training

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::PointCloud;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    // Progress callback type
    using ProgressCallback = std::function<void(float percentage, const std::string& message)>;

    // Dataset type enum
    enum class DatasetType {
        Unknown,
        COLMAP,
        Transforms
    };

    // Public types that clients need
    struct LoadOptions {
        int resize_factor = -1;
        int max_width = 3840;
        std::string images_folder = "images";
        bool validate_only = false;
        ProgressCallback progress = nullptr;
    };

    struct LoadedScene {
        std::shared_ptr<lfs::training::CameraDataset> cameras;
        std::shared_ptr<PointCloud> point_cloud;
    };

    struct LoadResult {
        std::variant<std::shared_ptr<SplatData>, LoadedScene> data;
        Tensor scene_center;
        bool images_have_alpha = false;
        std::string loader_used;
        std::chrono::milliseconds load_time{0};
        std::vector<std::string> warnings;
    };

    /**
     * @brief Main loader interface - the ONLY public API for the loader module
     *
     * This class provides a clean facade over all loading functionality.
     * All implementation details are hidden behind this interface.
     */
    class Loader {
    public:
        /**
         * @brief Create a loader instance
         */
        static std::unique_ptr<Loader> create();

        /**
         * @brief Quick check if path contains a dataset (vs single file like PLY)
         * @param path Directory or file to check
         * @return true if dataset, false if single file or not loadable
         */
        static bool isDatasetPath(const std::filesystem::path& path);

        /**
         * @brief Determine the type of dataset at the given path
         * @param path Directory or file to check
         * @return DatasetType enum value
         */
        static DatasetType getDatasetType(const std::filesystem::path& path);

        /**
         * @brief Load data from any supported format
         * @param path File or directory to load
         * @param options Loading options
         * @return LoadResult on success, Error on failure (path not found, invalid dataset, etc.)
         */
        [[nodiscard]] virtual Result<LoadResult> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {}) = 0;

        /**
         * @brief Check if a path can be loaded
         * @param path File or directory to check
         * @return true if the path can be loaded
         */
        virtual bool canLoad(const std::filesystem::path& path) const = 0;

        /**
         * @brief Get list of supported formats
         * @return Human-readable list of supported formats
         */
        virtual std::vector<std::string> getSupportedFormats() const = 0;

        /**
         * @brief Get list of supported file extensions
         * @return List of extensions (e.g., ".ply", ".json")
         */
        virtual std::vector<std::string> getSupportedExtensions() const = 0;

        virtual ~Loader() = default;
    };

    // PLY point cloud utilities

    /// Check if PLY contains Gaussian splat properties (opacity, scaling, rotation)
    /// Returns false for simple point clouds (xyz + colors only)
    bool is_gaussian_splat_ply(const std::filesystem::path& filepath);

    /// Load PLY as simple point cloud (xyz + optional colors)
    /// Use this for PLY files that are NOT Gaussian splats
    std::expected<PointCloud, std::string> load_ply_point_cloud(const std::filesystem::path& filepath);

} // namespace lfs::io