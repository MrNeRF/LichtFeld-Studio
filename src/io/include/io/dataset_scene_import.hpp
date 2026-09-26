/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

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

    std::expected<void, std::string> loadTrainingDataIntoScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene);

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
