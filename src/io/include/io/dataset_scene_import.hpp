/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/mesh_data.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/uuid.hpp"
#include "io/loader.hpp"
#include "io/project_recovery.hpp"
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace lfs::training {
    [[nodiscard]] inline lfs::io::CentralizeDataset parse_centralize(const std::string& value) {
        if (value == "by_pointcloud")
            return lfs::io::CentralizeDataset::ByPointCloud;
        if (value == "by_cameras")
            return lfs::io::CentralizeDataset::ByCameras;
        return lfs::io::CentralizeDataset::Off;
    }

    [[nodiscard]] lfs::io::Result<lfs::io::LoadResult> loadTrainingData(
        const lfs::core::param::TrainingParameters& params);

    std::expected<void, std::string> loadTrainingDataIntoScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene);

    /// Parses a mesh init file ahead of installing the dataset, so neither the preview nor training
    /// start parse it again; does nothing for other init files. Load errors surface in the preview.
    void preloadInitMesh(const lfs::core::param::TrainingParameters& params);

    /// Hands the init mesh kept by preloadInitMesh or the dataset preview to training start; null
    /// when none is kept for `init_file` or the file was written since it was parsed.
    [[nodiscard]] std::shared_ptr<lfs::core::MeshData> takePendingInitMesh(const std::filesystem::path& init_file);

    // Reload the points used to initialize training for evaluation after resume.
    [[nodiscard]] lfs::Result<std::shared_ptr<lfs::core::PointCloud>> loadInitialPointCloud(
        const lfs::core::param::TrainingParameters& params, const glm::vec3& training_origin);

    // Replaces parameter tensors only; the SplatData object stays.
    std::expected<void, std::string> migrateTrainingModelToAllocator(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::SplatData& model,
        const lfs::core::SplatTensorAllocator& tensor_allocator,
        bool force_reallocation = false);

    std::expected<void, std::string> validateDatasetPath(
        const lfs::core::param::TrainingParameters& params);

    std::expected<void, std::string> applyLoadResultToScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::io::LoadResult&& load_result);
} // namespace lfs::training
