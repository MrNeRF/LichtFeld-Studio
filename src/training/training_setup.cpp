/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_setup.hpp"
#include "core/error.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/provenance.hpp"
#include "core/scene.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/shareable_allocation_limit.hpp"
#include "core/source_site.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "dataset.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "io/project_document.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "normal_auto_generate.hpp"
#include "trainer.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <variant>

#include "io/dataset_scene_import_internal.hpp"
namespace lfs::training {
    namespace {
        void randomChoosePointCloud(lfs::core::PointCloud& point_cloud,
                                    const int target_count,
                                    const int seed = 0) {
            const int64_t source_count = point_cloud.size();
            if (target_count <= 0 || source_count <= 0 ||
                static_cast<int64_t>(target_count) >= source_count) {
                return;
            }

            std::vector<int> all_indices(static_cast<std::size_t>(source_count));
            std::iota(all_indices.begin(), all_indices.end(), 0);
            std::mt19937 rng(seed);
            std::shuffle(all_indices.begin(), all_indices.end(), rng);
            std::vector<int> selected_indices(
                all_indices.begin(),
                all_indices.begin() + target_count);

            auto select_rows = [&](lfs::core::Tensor& tensor) {
                if (!tensor.is_valid() || tensor.numel() == 0 || tensor.ndim() == 0 ||
                    static_cast<int64_t>(tensor.size(0)) != source_count) {
                    return;
                }
                auto indices = lfs::core::Tensor::from_vector(
                    selected_indices,
                    lfs::core::TensorShape({static_cast<std::size_t>(target_count)}),
                    tensor.device());
                tensor = tensor.index_select(0, indices).contiguous();
            };

            select_rows(point_cloud.means);
            select_rows(point_cloud.colors);
            select_rows(point_cloud.normals);
            select_rows(point_cloud.sh0);
            select_rows(point_cloud.shN);
            select_rows(point_cloud.opacity);
            select_rows(point_cloud.scaling);
            select_rows(point_cloud.rotation);
        }

        std::optional<float> computeSceneScaleFromPositions(
            const lfs::core::Tensor& positions,
            const lfs::core::Tensor& scene_center) {
            if (!positions.is_valid() || positions.ndim() != 2 ||
                positions.size(0) == 0 || positions.size(1) < 3 ||
                !scene_center.is_valid() || scene_center.numel() < 3) {
                return std::nullopt;
            }

            const auto center = scene_center.to(positions.device());
            const auto dists = positions.sub(center).norm(2.0f, {1}, false);
            if (!dists.is_valid() || dists.size(0) == 0) {
                return std::nullopt;
            }

            const auto sorted_dists = dists.sort(0, false);
            return sorted_dists.first[dists.size(0) / 2].item();
        }

        void recomputeInitSplatSceneScale(
            lfs::core::SplatData& model,
            const lfs::core::Tensor& scene_center,
            const std::filesystem::path& init_file) {
            const auto scene_scale = computeSceneScaleFromPositions(model.means_raw(), scene_center);
            if (!scene_scale) {
                LOG_WARN("Could not compute scene scale for init splat {}; keeping {}",
                         lfs::core::path_to_utf8(init_file.filename()),
                         model.get_scene_scale());
                return;
            }

            const float previous_scale = model.get_scene_scale();
            model.set_scene_scale(*scene_scale);
            LOG_INFO("Computed init scene scale from {}: {} -> {}",
                     lfs::core::path_to_utf8(init_file.filename()),
                     previous_scale,
                     *scene_scale);
        }

        std::optional<std::filesystem::path> gaussianSplatInitPath(
            const lfs::core::param::TrainingParameters& params) {
            if (!params.init_path.has_value() || params.init_path->empty()) {
                return std::nullopt;
            }
            const std::filesystem::path init_file = lfs::core::utf8_to_path(*params.init_path);
            if (isPlainPointCloudPly(init_file)) {
                return std::nullopt;
            }
            return init_file;
        }

        TrainingModelGraphInstall makeGraphInstall(const TrainingModelGraphCapture& context,
                                                   std::unique_ptr<lfs::core::SplatData> model) {
            TrainingModelGraphInstall install;
            install.model = std::move(model);
            install.parent_id = context.parent_id;
            install.point_cloud_node_id = context.point_cloud_node_id;
            install.node_transform = context.node_transform;
            install.has_preserved_cropbox = context.has_preserved_cropbox;
            install.preserved_cropbox_data = context.preserved_cropbox_data;
            install.preserved_cropbox_transform = context.preserved_cropbox_transform;
            return install;
        }

        std::expected<TrainingModelGraphInstall, std::string> loadGaussianInitModel(
            const lfs::core::param::TrainingParameters& params,
            lfs::core::Scene& scene,
            const std::filesystem::path& init_file,
            const TrainingModelGraphCapture* graph_capture) {
            auto loader = lfs::io::Loader::create();
            auto init_result = loader->load(init_file);
            if (!init_result) {
                return std::unexpected(std::string(initFileError(std::format(
                                                                     "Failed to load '{}': {}",
                                                                     lfs::core::path_to_utf8(init_file),
                                                                     init_result.error().format()))
                                                       .user_message()));
            }

            auto* splat_ptr = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&init_result->data);
            if (!splat_ptr || !*splat_ptr) {
                return std::unexpected(std::string(initFileError(std::format(
                                                                     "'{}': invalid SplatData",
                                                                     lfs::core::path_to_utf8(init_file)))
                                                       .user_message()));
            }

            auto model = std::make_unique<lfs::core::SplatData>(std::move(**splat_ptr));
            centerInitializationMeans(model->means(), graph_capture
                                                          ? graph_capture->training_data_origin
                                                          : scene.getTrainingDataOrigin());
            const lfs::core::Tensor scene_center =
                graph_capture
                    ? graph_capture->scene_center
                    : scene.getSceneCenter();
            recomputeInitSplatSceneScale(*model, scene_center, init_file);
            applyTrainingSHDegree(*model, params.optimization.sh_degree);
            LOG_INFO("Loaded {} gaussians from {} (sh={})",
                     model->size(),
                     lfs::core::path_to_utf8(init_file.filename()),
                     model->get_max_sh_degree());

            TrainingModelGraphCapture context =
                graph_capture ? *graph_capture : captureTrainingModelGraph(scene);
            context.has_preserved_cropbox = false;
            return makeGraphInstall(context, std::move(model));
        }

        std::expected<std::unique_ptr<lfs::core::SplatData>, std::string> loadAddedSplat(
            const std::filesystem::path& path,
            const int target_degree) {
            auto loader = lfs::io::Loader::create();
            auto load_result = loader->load(path);
            if (!load_result) {
                return std::unexpected(std::format("Failed to load added splat '{}': {}",
                                                   lfs::core::path_to_utf8(path),
                                                   load_result.error().format()));
            }

            auto* splat_ptr = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&load_result->data);
            if (!splat_ptr || !*splat_ptr) {
                return std::unexpected(std::format("'{}' is not a supported splat file",
                                                   lfs::core::path_to_utf8(path)));
            }

            auto model = std::make_unique<lfs::core::SplatData>(std::move(**splat_ptr));
            applyTrainingSHDegree(*model, target_degree);
            LOG_INFO("Loaded added splat {}: {} Gaussians (sh={})",
                     lfs::core::path_to_utf8(path.filename()),
                     model->size(),
                     model->get_max_sh_degree());
            return std::move(model);
        }

        std::expected<void, std::string> appendAddedSplats(
            const lfs::core::param::TrainingParameters& params,
            lfs::core::SplatData& model,
            const glm::vec3& dataset_origin) {
            if (params.add_splat_paths.empty()) {
                return {};
            }

            applyTrainingSHDegree(model, params.optimization.sh_degree);

            const size_t base_count = static_cast<size_t>(model.size());
            size_t added_count = 0;
            size_t frozen_count = 0;
            std::vector<lfs::core::SplatData::FrozenRange> frozen_ranges = model.frozen_ranges();
            std::vector<std::unique_ptr<lfs::core::SplatData>> owned_added_splats;
            owned_added_splats.reserve(params.add_splat_paths.size());

            std::vector<std::pair<const lfs::core::SplatData*, glm::mat4>> splats;
            splats.reserve(params.add_splat_paths.size() + 1);
            splats.emplace_back(&model, glm::mat4{1.0f});

            for (size_t i = 0; i < params.add_splat_paths.size(); ++i) {
                const auto& path = params.add_splat_paths[i];
                auto added = loadAddedSplat(path, params.optimization.sh_degree);
                if (!added) {
                    return std::unexpected(added.error());
                }
                centerInitializationMeans((*added)->means(), dataset_origin);

                const size_t count = static_cast<size_t>((*added)->size());
                if (i < params.add_splat_freeze.size() && params.add_splat_freeze[i] && count > 0) {
                    frozen_ranges.push_back({base_count + added_count, count});
                    frozen_count += count;
                }
                added_count += count;
                splats.emplace_back(added->get(), glm::mat4{1.0f});
                owned_added_splats.push_back(std::move(*added));
            }

            const size_t merged_count = base_count + added_count;
            const int max_cap = params.optimization.max_cap;
            if (max_cap > 0 && merged_count > static_cast<size_t>(max_cap)) {
                return std::unexpected(std::format(
                    "Added splats contain {} Gaussians for a total of {}, exceeding --max-cap {}. "
                    "Increase --max-cap or add fewer splats.",
                    added_count, merged_count, max_cap));
            }

            auto merged = lfs::core::Scene::mergeSplatsWithTransforms(splats);
            if (!merged) {
                return std::unexpected("Failed to merge added splats into training model");
            }

            // Keep the base model scene scale so means LR remains tied to the dataset scale.
            const float scene_scale = model.get_scene_scale();
            lfs::core::SplatData merged_with_base_scale(
                merged->get_max_sh_degree(),
                std::move(merged->means_raw()),
                std::move(merged->sh0_raw()),
                std::move(merged->shN_raw()),
                std::move(merged->scaling_raw()),
                std::move(merged->rotation_raw()),
                std::move(merged->opacity_raw()),
                scene_scale,
                lfs::core::SplatData::ShNLayout::Swizzled);
            merged_with_base_scale.set_active_sh_degree(merged->get_active_sh_degree());
            applyTrainingSHDegree(merged_with_base_scale, params.optimization.sh_degree);
            merged_with_base_scale.set_frozen_ranges(std::move(frozen_ranges));
            model = std::move(merged_with_base_scale);

            LOG_INFO("Added {} splat file{} to training model: {} + {} -> {} Gaussians",
                     params.add_splat_paths.size(),
                     params.add_splat_paths.size() == 1 ? "" : "s",
                     base_count,
                     added_count,
                     model.size());
            if (frozen_count > 0) {
                LOG_INFO("Marked {} added Gaussian{} as frozen",
                         frozen_count,
                         frozen_count == 1 ? "" : "s");
            }
            return {};
        }

    } // namespace

    TrainingModelGraphCapture captureTrainingModelGraph(lfs::core::Scene& scene) {
        TrainingModelGraphCapture context;
        context.training_model = scene.getTrainingModel();
        context.scene_center = scene.getSceneCenter();
        context.training_data_origin = scene.getTrainingDataOrigin();
        for (const auto* node : scene.getNodes()) {
            if (!node || node->type != lfs::core::NodeType::POINTCLOUD || !node->point_cloud) {
                continue;
            }
            context.point_cloud_node_id = node->id;
            context.parent_id = node->parent_id;
            context.node_transform = node->transform();
            context.point_cloud = *node->point_cloud;
            break;
        }
        if (context.point_cloud_node_id == lfs::core::NULL_NODE) {
            return context;
        }

        const lfs::core::NodeId cropbox_id = scene.getCropBoxForSplat(context.point_cloud_node_id);
        if (cropbox_id == lfs::core::NULL_NODE) {
            return context;
        }
        const auto* cropbox_node = scene.getNodeById(cropbox_id);
        if (!cropbox_node || !cropbox_node->cropbox) {
            return context;
        }
        context.preserved_cropbox_data = *cropbox_node->cropbox;
        context.preserved_cropbox_transform =
            cropbox_node->parent_id == context.point_cloud_node_id
                ? cropbox_node->transform()
                : glm::inverse(scene.getWorldTransform(context.point_cloud_node_id)) *
                      scene.getWorldTransform(cropbox_id);
        context.has_preserved_cropbox = true;
        return context;
    }

    std::expected<std::optional<TrainingModelGraphInstall>, std::string> prepareTrainingModel(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::core::SplatTensorAllocator tensor_allocator,
        const TrainingModelGraphCapture* graph_capture) {

        if (const auto unavailable = unavailable_training_reason(
                params, lfs::core::default_gpu_backend(), training_loader_dependencies(params))) {
            return std::unexpected(*unavailable);
        }

        const glm::vec3 dataset_origin = graph_capture
                                             ? graph_capture->training_data_origin
                                             : scene.getTrainingDataOrigin();
        const auto finalize_new_model = [&](lfs::core::SplatData& model)
            -> std::expected<void, std::string> {
            applyTrainingSHDegree(model, params.optimization.sh_degree);
            if (auto result = appendAddedSplats(params, model, dataset_origin); !result) {
                return result;
            }
            if (auto result = migrateTrainingModelToAllocator(params, model, tensor_allocator); !result) {
                return result;
            }
            return {};
        };

        const bool has_training_model =
            graph_capture ? graph_capture->training_model != nullptr : scene.getTrainingModel() != nullptr;
        if (!has_training_model) {
            if (const auto init_file = gaussianSplatInitPath(params)) {
                auto loaded = loadGaussianInitModel(params, scene, *init_file, graph_capture);
                if (!loaded) {
                    return std::unexpected(std::move(loaded.error()));
                }
                if (auto result = appendAddedSplats(params, *loaded->model, dataset_origin); !result) {
                    return std::unexpected(std::move(result.error()));
                }
                const int max_cap = params.optimization.max_cap;
                if (max_cap > 0 && loaded->model->size() > max_cap) {
                    LOG_WARN("Max cap ({}) is less than initial splat count ({}), randomly selecting {} splats",
                             max_cap, loaded->model->size(), max_cap);
                    lfs::core::random_choose(*loaded->model, max_cap);
                }
                if (auto result = migrateTrainingModelToAllocator(
                        params, *loaded->model, tensor_allocator);
                    !result) {
                    return std::unexpected(std::move(result.error()));
                }
                return std::optional<TrainingModelGraphInstall>{std::move(*loaded)};
            }
        }

        if (auto* model = graph_capture ? graph_capture->training_model : scene.getTrainingModel()) {
            applyTrainingSHDegree(*model, params.optimization.sh_degree);
            if (auto result = appendAddedSplats(params, *model, dataset_origin); !result) {
                return std::unexpected(std::move(result.error()));
            }

            const int max_cap = params.optimization.max_cap;
            if (max_cap > 0 && model->size() > max_cap) {
                LOG_WARN("Max cap ({}) is less than initial splat count ({}), randomly selecting {} splats",
                         max_cap, model->size(), max_cap);
                lfs::core::random_choose(*model, max_cap);
            }

            if (auto result = migrateTrainingModelToAllocator(params, *model, tensor_allocator); !result) {
                return std::unexpected(std::move(result.error()));
            }
            if (!graph_capture) {
                scene.syncTrainingModelTopology(static_cast<size_t>(model->size()));
                scene.notifyMutation(lfs::core::Scene::MutationType::MODEL_CHANGED);
            }
            return std::optional<TrainingModelGraphInstall>{};
        }

        const TrainingModelGraphCapture owned_capture =
            graph_capture ? TrainingModelGraphCapture{} : captureTrainingModelGraph(scene);
        const TrainingModelGraphCapture& context = graph_capture ? *graph_capture : owned_capture;
        const lfs::core::PointCloud* point_cloud =
            context.point_cloud ? &*context.point_cloud : nullptr;
        lfs::core::PointCloud point_cloud_to_use;
        const int max_cap = params.optimization.max_cap;

        if (point_cloud && point_cloud->size() > 0) {
            // An enabled crop box previews the region and supplies ROI weights.
            // Only an explicit Apply removes seed points before training.
            point_cloud_to_use = *point_cloud;
            if (max_cap > 0) {
                point_cloud_to_use.means = point_cloud_to_use.means.cpu();
                point_cloud_to_use.colors = point_cloud_to_use.colors.cpu();
            }
        } else {
            LOG_INFO("No point cloud provided, using random initialization");
            point_cloud_to_use = *createRandomPointCloud();
        }

        if (!params.optimization.random && max_cap > 0 &&
            point_cloud_to_use.size() > static_cast<int64_t>(max_cap)) {
            LOG_WARN("Max cap ({}) is less than initial point count ({}), "
                     "sampling point cloud before training tensor allocation",
                     max_cap, point_cloud_to_use.size());
            randomChoosePointCloud(point_cloud_to_use, max_cap);
        }

        lfs::core::Tensor scene_center = context.scene_center;
        if (!scene_center.is_valid() || scene_center.numel() == 0) {
            LOG_WARN("No scene center from loader, computing from point cloud");
            if (point_cloud_to_use.size() > 0) {
                auto means_cpu = point_cloud_to_use.means.cpu();
                auto mean = means_cpu.mean({0});
                scene_center = max_cap > 0 ? mean : mean.gpu();
            } else {
                scene_center = lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU);
            }
        } else {
            scene_center = max_cap > 0 ? scene_center.cpu() : scene_center.gpu();
        }

        auto splat_result = lfs::core::init_model_from_pointcloud(
            params, scene_center, point_cloud_to_use, max_cap, tensor_allocator);

        if (!splat_result) {
            return std::unexpected(std::format("Failed to initialize model: {}", splat_result.error()));
        }

        if (max_cap > 0 && max_cap < static_cast<int>(splat_result->size())) {
            LOG_WARN("Max cap ({}) is less than initial splat count ({}), randomly selecting {} splats",
                     max_cap, splat_result->size(), max_cap);
            lfs::core::random_choose(*splat_result, max_cap);
        }

        auto model = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
        if (auto result = finalize_new_model(*model); !result) {
            return std::unexpected(std::move(result.error()));
        }
        if (params.init_path.has_value() && !params.init_path->empty()) {
            LOG_INFO("Init {} gaussians from {} (sh={})",
                     model->size(),
                     lfs::core::path_to_utf8(lfs::core::utf8_to_path(*params.init_path).filename()),
                     model->get_max_sh_degree());
        } else {
            LOG_INFO("Created training model with {} gaussians", model->size());
        }
        return std::optional<TrainingModelGraphInstall>{
            makeGraphInstall(context, std::move(model))};
    }

    std::expected<void, std::string> installTrainingModel(
        lfs::core::Scene& scene,
        TrainingModelGraphInstall&& install) {
        if (!install.model) {
            return std::unexpected("Training model install is missing splat data");
        }

        if (install.point_cloud_node_id != lfs::core::NULL_NODE) {
            if (const auto* pc_node = scene.getNodeById(install.point_cloud_node_id)) {
                scene.removeNode(pc_node->name, false);
            }
        }

        const lfs::core::NodeId model_id =
            scene.addSplat("Model", std::move(install.model), install.parent_id);
        if (model_id == lfs::core::NULL_NODE) {
            return std::unexpected("Failed to add training model to scene");
        }
        if (install.node_transform != glm::mat4{1.0f}) {
            scene.setNodeTransform(model_id, install.node_transform);
        }
        scene.setTrainingModelNode(model_id);
        if (install.has_preserved_cropbox) {
            const lfs::core::NodeId model_cropbox_id = scene.addCropBox("Model_cropbox", model_id);
            if (model_cropbox_id != lfs::core::NULL_NODE) {
                scene.setCropBoxData(model_cropbox_id, install.preserved_cropbox_data);
                scene.setNodeTransform(model_cropbox_id, install.preserved_cropbox_transform);
            }
        }
        return {};
    }

    std::expected<void, std::string> initializeTrainingModel(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::core::SplatTensorAllocator tensor_allocator) {
        auto prepared = prepareTrainingModel(params, scene, std::move(tensor_allocator));
        if (!prepared) {
            return std::unexpected(std::move(prepared.error()));
        }
        if (!*prepared) {
            return {};
        }
        return installTrainingModel(scene, std::move(**prepared));
    }

    std::expected<ProjectCheckpointTrainer, std::string>
    installTrainerFromProjectCheckpoint(
        lfs::core::Scene& scene,
        const lfs::io::project::ProjectDocument& document,
        const lfs::core::Uuid& checkpoint_uuid,
        const lfs::core::param::TrainingParameters& params,
        const std::string_view source_name,
        const int expected_iteration,
        const std::optional<lfs::io::project::RecoverySession>&
            recovery_session,
        lfs::core::SplatTensorAllocator tensor_allocator) {
        if (const auto unavailable = unavailable_training_reason(
                params, lfs::core::default_gpu_backend(), training_loader_dependencies(params))) {
            return std::unexpected(*unavailable);
        }
        if (params.dataset.data_path.empty()) {
            return std::unexpected(
                "Project checkpoint has no dataset path");
        }
        if (!std::filesystem::exists(
                params.dataset.data_path)) {
            return std::unexpected(std::format(
                "Dataset path does not exist: {}",
                lfs::core::path_to_utf8(
                    params.dataset.data_path)));
        }

        auto trainer = std::make_unique<Trainer>(scene);
        if (recovery_session) {
            trainer->set_recovery_session(*recovery_session);
        }
        if (!params.python_scripts.empty()) {
            trainer->set_python_scripts(params.python_scripts);
        }
        if (tensor_allocator) {
            trainer->setSplatTensorAllocator(
                std::move(tensor_allocator));
        }
        lfs::core::SplatData preloaded_model;
        lfs::core::SplatData* preloaded_model_ptr = nullptr;
        if (auto* hydrated = scene.getTrainingModel();
            hydrated && hydrated->size() > 0) {
            preloaded_model = hydrated->clone();
            preloaded_model.set_frozen_ranges(hydrated->frozen_ranges());
            preloaded_model_ptr = &preloaded_model;
        }
        if (const auto initialized =
                trainer->initialize(params);
            !initialized) {
            return std::unexpected(std::format(
                "Failed to initialize trainer from project: {}",
                initialized.error()));
        }
        const auto* checkpoint =
            document.find_checkpoint(checkpoint_uuid);
        if (!checkpoint) {
            return std::unexpected(
                "Project CKPT handle disappeared");
        }
        std::optional<CheckpointLoadResult> restored;
        auto visited = checkpoint->visit_stream(
            [&](std::istream& source,
                const std::uint64_t bytes)
                -> lfs::Result<void> {
                restored = trainer->load_checkpoint(
                    source, bytes, source_name,
                    preloaded_model_ptr);
                return {};
            });
        if (!visited) {
            return std::unexpected(std::format(
                "Failed to stream project CKPT: {}",
                lfs::format_for_developer(visited.error())));
        }
        if (!restored || !*restored) {
            return std::unexpected(std::format(
                "Failed to restore project trainer state: {}",
                restored ? restored->error()
                         : "CKPT visitor did not run"));
        }
        const int restored_iteration = **restored;
        if (restored_iteration != expected_iteration ||
            trainer->get_current_iteration() !=
                expected_iteration) {
            return std::unexpected(std::format(
                "Project resume iteration mismatch: "
                "display={} trainer={} expected={}",
                restored_iteration,
                trainer->get_current_iteration(),
                expected_iteration));
        }
        return ProjectCheckpointTrainer{
            .trainer = std::move(trainer),
            .iteration = restored_iteration,
        };
    }

    void grant_headless_project_saves(
        Trainer& trainer,
        const lfs::core::param::TrainingParameters& params,
        const std::filesystem::path& destination,
        std::optional<std::filesystem::path> source_path) {
        if (destination.empty() &&
            params.dataset.output_path.empty()) {
            LOG_WARN(
                "Headless project saves not granted: no output path is set");
            return;
        }
        trainer.set_live_project_snapshot(
            destination.empty()
                ? params.dataset.output_path / "project.licht"
                : destination,
            {}, std::move(source_path));
        trainer.set_trainer_project_save_policy({
            .on_completion = true,
            .on_stop_or_error = true,
            .at_step_boundaries = true,
        });
    }

    namespace {
        const char* final_export_extension(const lfs::core::param::OutputFormat format) {
            using lfs::core::param::OutputFormat;
            switch (format) {
            case OutputFormat::PLY: return ".ply";
            case OutputFormat::SOG: return ".sog";
            case OutputFormat::SSOG: return ".ssog";
            case OutputFormat::SPZ: return ".spz";
            case OutputFormat::GLB: return ".glb";
            case OutputFormat::HTML: return ".html";
            case OutputFormat::USD: return ".usd";
            case OutputFormat::USDA: return ".usda";
            case OutputFormat::USDC: return ".usdc";
            case OutputFormat::RAD: return ".rad";
            }
            return ".ply";
        }

        lfs::io::Result<void> save_final_splat(const lfs::core::SplatData& splat,
                                               const std::filesystem::path& output,
                                               const lfs::core::param::OutputFormat format,
                                               const lfs::core::ProvenanceStamp& provenance,
                                               const lfs::core::param::TrainingParameters& params) {
            using lfs::core::param::OutputFormat;
            switch (format) {
            case OutputFormat::PLY:
                return lfs::io::save_ply(splat, {.output_path = output, .binary = true, .provenance = provenance});
            case OutputFormat::SSOG:
                return lfs::io::save_ssog(splat, {.output_path = output,
                                                  .lod_levels = params.lod_levels,
                                                  .lod_ratio = params.lod_ratio,
                                                  .chunk_count_k = params.lod_chunk_count,
                                                  .chunk_extent = params.lod_chunk_extent,
                                                  .chunk_min_k = params.lod_chunk_min,
                                                  .kmeans_iterations = params.sog_iterations,
                                                  .provenance = provenance});
            case OutputFormat::SOG:
                return lfs::io::save_sog(splat, {.output_path = output, .kmeans_iterations = 10, .provenance = provenance});
            case OutputFormat::SPZ:
                return lfs::io::save_spz(splat, {.output_path = output, .version = 4, .provenance = provenance});
            case OutputFormat::GLB:
                return lfs::io::save_spz(splat, {.output_path = output, .provenance = provenance, .glb = true});
            case OutputFormat::HTML:
                return lfs::io::export_html(splat, {.output_path = output, .kmeans_iterations = 10, .provenance = provenance});
            case OutputFormat::USD:
            case OutputFormat::USDA:
            case OutputFormat::USDC:
                return lfs::io::save_usd(splat, {.output_path = output, .provenance = provenance});
            case OutputFormat::RAD:
                return lfs::io::save_rad(splat, {.output_path = output, .provenance = provenance});
            }
            return lfs::io::save_ply(splat, {.output_path = output, .binary = true, .provenance = provenance});
        }
    } // namespace

    void export_final_splats(const Trainer& trainer,
                             const lfs::core::param::TrainingParameters& params) {
        if (params.export_formats.empty()) {
            return;
        }
        const auto& model = trainer.get_strategy().get_model();
        const std::filesystem::path out_dir = params.dataset.output_path;
        const std::string stem = params.dataset.output_name.empty()
                                     ? std::format("splat_{}", trainer.get_current_iteration())
                                     : params.dataset.output_name;

        lfs::core::ProvenanceStamp stamp = params.include_provenance
                                               ? lfs::core::make_provenance_stamp()
                                               : lfs::core::make_minimal_provenance_stamp();
        if (params.include_provenance) {
            stamp.iteration = trainer.get_current_iteration();
            stamp.strategy = params.optimization.strategy;
        }

        for (const auto format : params.export_formats) {
            const std::filesystem::path path = out_dir / (stem + final_export_extension(format));
            if (const auto result = save_final_splat(model, path, format, stamp, params); !result) {
                LOG_ERROR("Failed to export final splat to {}: {}",
                          lfs::core::path_to_utf8(path), result.error().message);
            } else {
                LOG_INFO("Exported final splat: {}", lfs::core::path_to_utf8(path));
            }
        }
    }

} // namespace lfs::training
