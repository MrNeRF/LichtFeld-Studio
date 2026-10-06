/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
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

namespace lfs::io::project {
    class ProjectDocument;
}

namespace lfs::training {
    class Trainer;

    /// Scene-graph payload produced by prepareTrainingModel(). Heavy tensor work
    /// is finished; the owner thread still has to replace the POINTCLOUD node.
    struct TrainingModelGraphInstall {
        std::unique_ptr<lfs::core::SplatData> model;
        lfs::core::NodeId parent_id = lfs::core::NULL_NODE;
        lfs::core::NodeId point_cloud_node_id = lfs::core::NULL_NODE;
        glm::mat4 node_transform{1.0f};
        // Removing the point cloud node drops the scene's initial point cloud; the points eval mask still needs it.
        bool keep_initial_point_cloud = false;
        bool has_preserved_cropbox = false;
        lfs::core::CropBoxData preserved_cropbox_data{};
        glm::mat4 preserved_cropbox_transform{1.0f};
    };

    /// Owner-thread snapshot of the graph inputs initializeTrainingModel needs.
    /// The point cloud is copied so the worker can prepare without walking nodes_.
    struct TrainingModelGraphCapture {
        lfs::core::NodeId point_cloud_node_id = lfs::core::NULL_NODE;
        lfs::core::NodeId parent_id = lfs::core::NULL_NODE;
        glm::mat4 node_transform{1.0f};
        std::optional<lfs::core::PointCloud> point_cloud;
        lfs::core::CropBoxData preserved_cropbox_data{};
        glm::mat4 preserved_cropbox_transform{1.0f};
        bool has_preserved_cropbox = false;
        lfs::core::SplatData* training_model = nullptr;
        lfs::core::Tensor scene_center;
        glm::vec3 training_data_origin{0.0f};
    };

    [[nodiscard]] TrainingModelGraphCapture captureTrainingModelGraph(lfs::core::Scene& scene);

    /// Result of streaming an embedded project CKPT into a Trainer on an
    /// already-hydrated scene (no scene clear). Caller owns TrainerManager
    /// install (setTrainer / setTrainerFromCheckpoint).
    struct ProjectCheckpointTrainer {
        std::unique_ptr<Trainer> trainer;
        int iteration = 0;
    };

    // Headless sessions have no project lifecycle; the application grants the
    // trainer its standing save destination and triggers here. source_path may
    // seed a fresh destination only when it contains the dataset being trained;
    // recovered sources must remain alive until the trainer finishes saving.
    void grant_headless_project_saves(
        Trainer& trainer,
        const lfs::core::param::TrainingParameters& params,
        const std::filesystem::path& destination = {},
        std::optional<std::filesystem::path> source_path = std::nullopt);

    /// Rows of model without its frozen --add-splat ranges and soft-deleted rows;
    /// nullopt when model has no frozen ranges.
    [[nodiscard]] lfs::Result<std::optional<lfs::core::SplatData>> exclude_frozen_rows(
        const lfs::core::SplatData& model);

    /// Write `--export` formats next to project.licht after a terminal project
    /// save. No-op when `params.export_formats` is empty. Frozen --add-splat rows
    /// are left out when --exclude-export was given or restored from the checkpoint.
    [[nodiscard]] lfs::Status export_final_splats(
        const Trainer& trainer,
        const lfs::core::param::TrainingParameters& params);

    /// Construct + initialize Trainer on the live scene, then stream the
    /// document's CKPT payload via load_checkpoint(istream). Does not clear
    /// the scene and does not install into TrainerManager.
    [[nodiscard]] std::expected<ProjectCheckpointTrainer, std::string>
    installTrainerFromProjectCheckpoint(
        lfs::core::Scene& scene,
        const lfs::io::project::ProjectDocument& document,
        const lfs::core::Uuid& checkpoint_uuid,
        const lfs::core::param::TrainingParameters& params,
        std::string_view source_name,
        int expected_iteration,
        const std::optional<lfs::io::project::RecoverySession>&
            recovery_session = std::nullopt,
        lfs::core::SplatTensorAllocator tensor_allocator = {});

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

    /// Points a model trained with these parameters started from, in the training frame at
    /// `training_origin`: the --init file when one was given, otherwise the dataset's sparse points.
    /// A resumed project does not keep them; the evaluation points mask reloads them here.
    [[nodiscard]] lfs::Result<std::shared_ptr<lfs::core::PointCloud>> loadInitialPointCloud(
        const lfs::core::param::TrainingParameters& params, const glm::vec3& training_origin);

    /**
     * @brief Initialize training model from point cloud
     *
     * Called when training starts. If a Gaussian-splat init file is set, that
     * file is loaded as the training model. Otherwise creates SplatData from the
     * POINTCLOUD node, optionally filtering by any CropBox attached to the point cloud.
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

    /// Build the training splat without mutating the scene graph. A nullopt
    /// result means an existing training model was updated in place. A populated
    /// install payload must be applied with installTrainingModel() on the thread
    /// that owns the scene graph (the viewer thread in the GUI).
    [[nodiscard]] std::expected<std::optional<TrainingModelGraphInstall>, std::string>
    prepareTrainingModel(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::core::SplatTensorAllocator tensor_allocator = {},
        const TrainingModelGraphCapture* graph_capture = nullptr);

    /// Replace the POINTCLOUD node with the prepared training model. Must run on
    /// the scene-owner thread; it is the only phase that erases/inserts nodes.
    [[nodiscard]] std::expected<void, std::string> installTrainingModel(
        lfs::core::Scene& scene,
        TrainingModelGraphInstall&& install);

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
