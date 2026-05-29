/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "io/loader.hpp"
#include <expected>
#include <string>

namespace lfs::core {
    class Scene;
}

namespace lfs::training {
    /**
     * @brief Load training data into Scene
     *
     * This is the unified loading path for both headless and GUI modes.
     * Loads cameras and point cloud into Scene. SplatData creation is deferred
     * until initializeTrainingModel() is called.
     *
     * The point cloud is added as a POINTCLOUD node, allowing the user to:
     * - View the point cloud before training
     * - Apply a CropBox to filter points before training
     *
     * After calling this, call initializeTrainingModel() then create a Trainer with: Trainer(scene)
     *
     * @param params Training parameters (including data path)
     * @param scene Scene to populate with training data
     * @return Error message on failure
     */
    std::expected<void, std::string> loadTrainingDataIntoScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene);

    /**
     * @brief Initialize training model from point cloud
     *
     * Called when training starts. Creates SplatData from the POINTCLOUD node,
     * optionally filtering by any CropBox attached to the point cloud.
     *
     * The POINTCLOUD node is replaced with a SPLAT node containing the initialized model.
     *
     * @param params Training parameters
     * @param scene Scene containing the POINTCLOUD node
     * @return Error message on failure
     */
    std::expected<void, std::string> initializeTrainingModel(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::core::SplatTensorAllocator tensor_allocator = {});

    /**
     * @brief Copy an existing training model into caller-provided tensor storage.
     *
     * GUI/Vulkan training uses this to repair loaded or restored scene-owned
     * SplatData before VkSplat renders it. It preserves the SplatData object and
     * swaps only its parameter tensors, so strategies that hold a SplatData
     * reference remain valid.
     */
    std::expected<void, std::string> migrateTrainingModelToAllocator(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::SplatData& model,
        const lfs::core::SplatTensorAllocator& tensor_allocator,
        bool force_reallocation = false);

    /**
     * @brief Validate dataset path without loading data
     *
     * Checks that the dataset path exists and has the required structure.
     * Does not load any data into memory.
     *
     * @param params Training parameters (including data path)
     * @return Error message on failure
     */
    std::expected<void, std::string> validateDatasetPath(
        const lfs::core::param::TrainingParameters& params);

    /// Apply pre-loaded data to scene (for async loading)
    std::expected<void, std::string> applyLoadResultToScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::io::LoadResult&& load_result);
} // namespace lfs::training
