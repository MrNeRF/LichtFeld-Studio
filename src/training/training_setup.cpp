/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_setup.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "dataset.hpp"
#include "io/loader.hpp"
#include "visualizer/scene/scene.hpp"
#include <format>

namespace lfs::training {

    namespace {
        constexpr size_t SH_CHANNELS = 3;

        void truncateSHDegree(lfs::core::SplatData& splat, const int target_degree) {
            if (target_degree < 0 || target_degree >= splat.get_max_sh_degree())
                return;

            if (target_degree == 0) {
                splat.shN() = lfs::core::Tensor{};
            } else {
                const size_t keep = static_cast<size_t>((target_degree + 1) * (target_degree + 1) - 1);
                auto& shN = splat.shN();
                if (shN.is_valid() && shN.ndim() >= 2 && shN.shape()[1] > keep) {
                    const auto slice_end = static_cast<int64_t>(shN.ndim() == 3 ? keep : keep * SH_CHANNELS);
                    shN = shN.slice(1, 0, slice_end).contiguous();
                }
            }
            splat.set_max_sh_degree(target_degree);
            splat.set_active_sh_degree(target_degree);
        }
    } // namespace

    std::expected<void, std::string> loadTrainingDataIntoScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::vis::Scene& scene) {

        // 1. Create loader
        auto data_loader = lfs::io::Loader::create();

        // 2. Set up load options with progress callback that emits events
        const auto& data_path = params.dataset.data_path;
        lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .validate_only = false,
            .progress = [&data_path](float percentage, const std::string& message) {
                LOG_DEBUG("[{:5.1f}%] {}", percentage, message);
                lfs::core::events::state::DatasetLoadProgress{
                    .path = data_path,
                    .progress = percentage,
                    .step = message}
                    .emit();
            }};

        // 3. Load the dataset
        LOG_INFO("Loading dataset from: {}", lfs::core::path_to_utf8(params.dataset.data_path));
        auto load_result = data_loader->load(params.dataset.data_path, load_options);
        if (!load_result) {
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error().format()));
        }

        LOG_INFO("Dataset loaded successfully using {} loader", load_result->loader_used);

        // 4. Handle the loaded data
        return std::visit([&](auto&& data) -> std::expected<void, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                // Direct PLY load - add to scene as splat node
                auto model = std::make_unique<lfs::core::SplatData>(std::move(*data));
                scene.addSplat("loaded_model", std::move(model));
                scene.setTrainingModelNode("loaded_model");
                LOG_INFO("Loaded PLY directly into scene");
                return {};

            } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                // Full scene data - store cameras and point cloud (defer SplatData creation)

                // Store cameras and scene center in Scene
                scene.setTrainCameras(data.cameras);
                scene.setInitialPointCloud(data.point_cloud);
                scene.setSceneCenter(load_result->scene_center);

                // Create dataset hierarchy:
                // [Dataset] bicycle
                // ├── PointCloud (54,275) - or Model if init_ply is used
                // │   └── CropBox (optional, can be added later)
                // ├── Cameras
                // │   ├── Training (185)
                // │   │   ├── cam_00001
                // │   │   └── ...
                // │   └── Validation (9)
                // │       └── ...
                // └── Images (194)
                //     ├── _DSC8679.JPG
                //     └── ...

                // Get dataset name from path
                std::string dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.filename());
                if (dataset_name.empty()) {
                    dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.parent_path().filename());
                }
                if (dataset_name.empty()) {
                    dataset_name = "Dataset";
                }

                // Add dataset root node
                auto dataset_id = scene.addDataset(dataset_name);

                // Load from --init file
                if (params.init_path.has_value()) {
                    // Use utf8_to_path for proper Unicode handling on Windows
                    const std::filesystem::path init_file = lfs::core::utf8_to_path(params.init_path.value());
                    const auto ext = init_file.extension().string();

                    // Point cloud PLY: initialize Gaussians from xyz+colors
                    if (ext == ".ply" && !lfs::io::is_gaussian_splat_ply(init_file)) {
                        auto pc_result = lfs::io::load_ply_point_cloud(init_file);
                        if (!pc_result) {
                            return std::unexpected(std::format("Failed to load '{}': {}",
                                                               lfs::core::path_to_utf8(init_file), pc_result.error()));
                        }

                        // Use scene_center from loader (camera centroid) for correct scene_scale
                        auto splat_result = lfs::core::init_model_from_pointcloud(
                            params, load_result->scene_center, *pc_result, static_cast<int>(pc_result->size()));

                        if (!splat_result) {
                            return std::unexpected(std::format("Init failed: {}", splat_result.error()));
                        }

                        auto model = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
                        LOG_INFO("Initialized {} Gaussians from {} (sh={})",
                                 model->size(), lfs::core::path_to_utf8(init_file.filename()), model->get_max_sh_degree());
                        scene.addSplat("Model", std::move(model), dataset_id);
                        scene.setTrainingModelNode("Model");
                    } else {
                        // Gaussian splat file
                        auto loader = lfs::io::Loader::create();
                        auto load_result = loader->load(init_file);

                        if (!load_result) {
                            return std::unexpected(std::format("Failed to load '{}': {}",
                                                               lfs::core::path_to_utf8(init_file), load_result.error().format()));
                        }

                        try {
                            auto splat_data = std::move(*std::get<std::shared_ptr<lfs::core::SplatData>>(load_result->data));
                            auto model = std::make_unique<lfs::core::SplatData>(std::move(splat_data));

                            const int target_sh = params.optimization.sh_degree;
                            if (target_sh >= 0 && target_sh < model->get_max_sh_degree()) {
                                LOG_INFO("Truncating SH: {} -> {}", model->get_max_sh_degree(), target_sh);
                                truncateSHDegree(*model, target_sh);
                            }

                            LOG_INFO("Loaded {} Gaussians from {} (sh={})",
                                     model->size(), lfs::core::path_to_utf8(init_file.filename()), model->get_max_sh_degree());
                            scene.addSplat("Model", std::move(model), dataset_id);
                            scene.setTrainingModelNode("Model");
                        } catch (const std::bad_variant_access&) {
                            return std::unexpected(std::format("'{}': invalid SplatData", lfs::core::path_to_utf8(init_file)));
                        }
                    }
                } else {
                    // Add point cloud as a POINTCLOUD node (defer SplatData creation until training starts)
                    // This allows the user to apply crop before training
                    if (data.point_cloud && data.point_cloud->size() > 0) {
                        LOG_INFO("Adding point cloud to scene ({} points) - SplatData will be created when training starts",
                                 data.point_cloud->size());
                        scene.addPointCloud("PointCloud", data.point_cloud, dataset_id);
                    } else {
                        // No point cloud - will use random initialization when training starts
                        LOG_INFO("No point cloud provided - random initialization will be used when training starts");
                    }
                }

                // Get camera info for train/val splits
                const auto& cameras = data.cameras->get_cameras();
                const bool enable_eval = params.optimization.enable_eval;
                const int test_every = params.dataset.test_every;

                // Count train/val cameras and masks
                size_t train_count = 0;
                size_t val_count = 0;
                size_t mask_count = 0;
                for (size_t i = 0; i < cameras.size(); ++i) {
                    if (enable_eval && (i % test_every) == 0) {
                        val_count++;
                    } else {
                        train_count++;
                    }
                    if (cameras[i]->has_mask()) {
                        mask_count++;
                    }
                }

                // Add Cameras group
                auto cameras_group_id = scene.addGroup("Cameras", dataset_id);

                // Add Training cameras group
                auto train_cameras_id = scene.addCameraGroup(
                    std::format("Training ({})", train_count),
                    cameras_group_id,
                    train_count);

                // Add individual training camera nodes (with mask_path if available)
                for (size_t i = 0; i < cameras.size(); ++i) {
                    if (!enable_eval || (i % test_every) != 0) { // Training camera (all if no eval)
                        scene.addCamera(cameras[i]->image_name(), train_cameras_id,
                                        static_cast<int>(i), cameras[i]->uid(),
                                        lfs::core::path_to_utf8(cameras[i]->image_path()),
                                        lfs::core::path_to_utf8(cameras[i]->mask_path()));
                    }
                }

                // Add Validation cameras group only if eval is enabled
                if (enable_eval && val_count > 0) {
                    auto val_cameras_id = scene.addCameraGroup(
                        std::format("Validation ({})", val_count),
                        cameras_group_id,
                        val_count);

                    // Add individual validation camera nodes (with mask_path if available)
                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if ((i % test_every) == 0) { // Validation camera
                            scene.addCamera(cameras[i]->image_name(), val_cameras_id,
                                            static_cast<int>(i), cameras[i]->uid(),
                                            lfs::core::path_to_utf8(cameras[i]->image_path()),
                                            lfs::core::path_to_utf8(cameras[i]->mask_path()));
                        }
                    }
                }

                LOG_INFO("Loaded dataset '{}' into scene: {} train{} cameras{}",
                         dataset_name, train_count,
                         enable_eval ? std::format(" + {} val", val_count) : "",
                         mask_count > 0 ? std::format(" ({} with masks)", mask_count) : "");
                return {};

            } else {
                return std::unexpected("Unknown data type returned from loader");
            }
        },
                          load_result->data);
    }

    std::expected<void, std::string> initializeTrainingModel(
        const lfs::core::param::TrainingParameters& params,
        lfs::vis::Scene& scene) {

        // Skip if training model already exists (e.g., from --init)
        if (scene.getTrainingModel() != nullptr) {
            return {};
        }

        // Find POINTCLOUD node
        lfs::vis::NodeId point_cloud_node_id = lfs::vis::NULL_NODE;
        lfs::vis::NodeId parent_id = lfs::vis::NULL_NODE;
        const lfs::core::PointCloud* point_cloud = nullptr;

        for (const auto* node : scene.getNodes()) {
            if (node->type == lfs::vis::NodeType::POINTCLOUD && node->point_cloud) {
                point_cloud_node_id = node->id;
                parent_id = node->parent_id;
                point_cloud = node->point_cloud.get();
                break;
            }
        }

        lfs::core::PointCloud point_cloud_to_use;
        const int max_cap = params.optimization.max_cap;

        if (point_cloud && point_cloud->size() > 0) {
            // Check for enabled CropBox
            const lfs::vis::CropBoxData* cropbox_data = nullptr;
            lfs::vis::NodeId cropbox_id = lfs::vis::NULL_NODE;

            if (point_cloud_node_id != lfs::vis::NULL_NODE) {
                cropbox_id = scene.getCropBoxForSplat(point_cloud_node_id);
                if (cropbox_id != lfs::vis::NULL_NODE) {
                    cropbox_data = scene.getCropBoxData(cropbox_id);
                }
            }

            if (cropbox_data && cropbox_data->enabled) {
                // Filter points by CropBox
                const glm::mat4 world_to_cropbox = glm::inverse(scene.getWorldTransform(cropbox_id));
                const auto& means = point_cloud->means;
                const auto& colors = point_cloud->colors;
                const size_t num_points = point_cloud->size();

                auto means_cpu = means.cpu();
                auto colors_cpu = colors.cpu();
                const float* means_ptr = means_cpu.ptr<float>();
                const uint8_t* colors_ptr = colors_cpu.ptr<uint8_t>();

                std::vector<float> filtered_means;
                std::vector<uint8_t> filtered_colors;
                filtered_means.reserve(num_points * 3);
                filtered_colors.reserve(num_points * 3);

                for (size_t i = 0; i < num_points; ++i) {
                    const glm::vec3 pos(means_ptr[i * 3], means_ptr[i * 3 + 1], means_ptr[i * 3 + 2]);
                    const glm::vec4 local_pos = world_to_cropbox * glm::vec4(pos, 1.0f);
                    const glm::vec3 local = glm::vec3(local_pos) / local_pos.w;

                    bool inside = local.x >= cropbox_data->min.x && local.x <= cropbox_data->max.x &&
                                  local.y >= cropbox_data->min.y && local.y <= cropbox_data->max.y &&
                                  local.z >= cropbox_data->min.z && local.z <= cropbox_data->max.z;

                    if (cropbox_data->inverse)
                        inside = !inside;

                    if (inside) {
                        filtered_means.push_back(means_ptr[i * 3]);
                        filtered_means.push_back(means_ptr[i * 3 + 1]);
                        filtered_means.push_back(means_ptr[i * 3 + 2]);
                        filtered_colors.push_back(colors_ptr[i * 3]);
                        filtered_colors.push_back(colors_ptr[i * 3 + 1]);
                        filtered_colors.push_back(colors_ptr[i * 3 + 2]);
                    }
                }

                const size_t filtered_count = filtered_means.size() / 3;
                LOG_INFO("CropBox filtering: {} -> {} points", num_points, filtered_count);

                if (filtered_count == 0) {
                    return std::unexpected("CropBox filtered out all points");
                }

                auto filtered_means_tensor = lfs::core::Tensor::from_vector(
                    filtered_means, {filtered_count, 3}, lfs::core::Device::CPU);
                auto filtered_colors_tensor = lfs::core::Tensor::zeros(
                    {filtered_count, 3}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                std::memcpy(filtered_colors_tensor.data_ptr(), filtered_colors.data(),
                            filtered_colors.size() * sizeof(uint8_t));

                point_cloud_to_use = lfs::core::PointCloud(filtered_means_tensor, filtered_colors_tensor);
            } else {
                // No CropBox or not enabled - use full point cloud
                point_cloud_to_use = *point_cloud;
                if (max_cap > 0) {
                    point_cloud_to_use.means = point_cloud_to_use.means.cpu();
                    point_cloud_to_use.colors = point_cloud_to_use.colors.cpu();
                }
            }
        } else {
            // No point cloud - use random initialization
            LOG_INFO("No point cloud provided, using random initialization");
            constexpr size_t NUM_INIT_GAUSSIANS = 10000;
            auto positions = lfs::core::Tensor::rand({NUM_INIT_GAUSSIANS, 3}, lfs::core::Device::CPU);
            positions = positions * 2.0f - 1.0f;
            auto colors = lfs::core::Tensor::randint({NUM_INIT_GAUSSIANS, 3}, 0, 256,
                                                     lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            point_cloud_to_use = lfs::core::PointCloud(positions, colors);
        }

        // Use scene center from loader (camera centroid) if available, otherwise compute from point cloud
        lfs::core::Tensor scene_center = scene.getSceneCenter();
        if (!scene_center.is_valid() || scene_center.numel() == 0) {
            // Fallback: compute from point cloud (less accurate for scene_scale)
            LOG_WARN("No scene center from loader, computing from point cloud (may affect densification)");
            if (point_cloud_to_use.size() > 0) {
                auto means_cpu = point_cloud_to_use.means.cpu();
                auto mean = means_cpu.mean({0});
                scene_center = max_cap > 0 ? mean : mean.cuda();
            } else {
                scene_center = lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU);
            }
        } else {
            // Ensure scene_center is on the correct device
            scene_center = max_cap > 0 ? scene_center.cpu() : scene_center.cuda();
        }

        // Initialize SplatData from point cloud
        auto splat_result = lfs::core::init_model_from_pointcloud(
            params, scene_center, point_cloud_to_use, max_cap);

        if (!splat_result) {
            return std::unexpected(std::format("Failed to initialize model: {}", splat_result.error()));
        }

        // Apply max_cap if needed
        if (max_cap > 0 && max_cap < static_cast<int>(splat_result->size())) {
            LOG_WARN("Max cap ({}) is less than initial splat count ({}), randomly selecting {} splats",
                     max_cap, splat_result->size(), max_cap);
            lfs::core::random_choose(*splat_result, max_cap);
        }

        // Remove the POINTCLOUD node and its children (e.g., CropBox)
        if (point_cloud_node_id != lfs::vis::NULL_NODE) {
            if (const auto* pc_node = scene.getNodeById(point_cloud_node_id)) {
                scene.removeNode(pc_node->name, false);
            }
        }

        // Add the new model
        auto model = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
        LOG_INFO("Created training model with {} gaussians", model->size());
        scene.addSplat("Model", std::move(model), parent_id);
        scene.setTrainingModelNode("Model");

        return {};
    }

    std::expected<void, std::string> validateDatasetPath(
        const lfs::core::param::TrainingParameters& params) {

        auto data_loader = lfs::io::Loader::create();

        lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .validate_only = true};

        auto result = data_loader->load(params.dataset.data_path, load_options);
        if (!result) {
            return std::unexpected(result.error().format());
        }
        return {};
    }

    std::expected<void, std::string> applyLoadResultToScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::vis::Scene& scene,
        lfs::io::LoadResult&& load_result) {

        return std::visit([&](auto&& data) -> std::expected<void, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                auto model = std::make_unique<lfs::core::SplatData>(std::move(*data));
                scene.addSplat("loaded_model", std::move(model));
                scene.setTrainingModelNode("loaded_model");
                return {};

            } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                scene.setTrainCameras(data.cameras);
                scene.setInitialPointCloud(data.point_cloud);
                scene.setSceneCenter(load_result.scene_center);

                std::string dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.filename());
                if (dataset_name.empty()) {
                    dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.parent_path().filename());
                }
                if (dataset_name.empty()) {
                    dataset_name = "Dataset";
                }

                const auto dataset_id = scene.addDataset(dataset_name);

                // Handle --init file
                if (params.init_path.has_value()) {
                    // Use utf8_to_path for proper Unicode handling on Windows
                    const std::filesystem::path init_file = lfs::core::utf8_to_path(params.init_path.value());
                    const auto ext = init_file.extension().string();

                    if (ext == ".ply" && !lfs::io::is_gaussian_splat_ply(init_file)) {
                        auto pc_result = lfs::io::load_ply_point_cloud(init_file);
                        if (!pc_result) {
                            return std::unexpected(std::format("Failed to load '{}': {}",
                                                               lfs::core::path_to_utf8(init_file), pc_result.error()));
                        }

                        auto splat_result = lfs::core::init_model_from_pointcloud(
                            params, load_result.scene_center, *pc_result, static_cast<int>(pc_result->size()));
                        if (!splat_result) {
                            return std::unexpected(std::format("Init failed: {}", splat_result.error()));
                        }

                        auto model = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
                        LOG_INFO("Init {} gaussians from {} (sh={})",
                                 model->size(), lfs::core::path_to_utf8(init_file.filename()), model->get_max_sh_degree());
                        scene.addSplat("Model", std::move(model), dataset_id);
                        scene.setTrainingModelNode("Model");
                    } else {
                        auto loader = lfs::io::Loader::create();
                        auto init_result = loader->load(init_file);
                        if (!init_result) {
                            return std::unexpected(std::format("Failed to load '{}': {}",
                                                               lfs::core::path_to_utf8(init_file), init_result.error().format()));
                        }

                        try {
                            auto splat_data = std::move(*std::get<std::shared_ptr<lfs::core::SplatData>>(init_result->data));
                            auto model = std::make_unique<lfs::core::SplatData>(std::move(splat_data));

                            const int target_sh = params.optimization.sh_degree;
                            if (target_sh >= 0 && target_sh < model->get_max_sh_degree()) {
                                truncateSHDegree(*model, target_sh);
                            }

                            LOG_INFO("Loaded {} gaussians from {} (sh={})",
                                     model->size(), lfs::core::path_to_utf8(init_file.filename()), model->get_max_sh_degree());
                            scene.addSplat("Model", std::move(model), dataset_id);
                            scene.setTrainingModelNode("Model");
                        } catch (const std::bad_variant_access&) {
                            return std::unexpected(std::format("'{}': invalid SplatData", lfs::core::path_to_utf8(init_file)));
                        }
                    }
                } else if (data.point_cloud && data.point_cloud->size() > 0) {
                    scene.addPointCloud("PointCloud", data.point_cloud, dataset_id);
                }

                // Build camera hierarchy
                const auto& cameras = data.cameras->get_cameras();
                const bool enable_eval = params.optimization.enable_eval;
                const int test_every = params.dataset.test_every;

                size_t train_count = 0, val_count = 0, mask_count = 0;
                for (size_t i = 0; i < cameras.size(); ++i) {
                    const bool is_val = enable_eval && (i % test_every) == 0;
                    is_val ? ++val_count : ++train_count;
                    if (cameras[i]->has_mask())
                        ++mask_count;
                }

                const auto cameras_group_id = scene.addGroup("Cameras", dataset_id);
                const auto train_cameras_id = scene.addCameraGroup(
                    std::format("Training ({})", train_count), cameras_group_id, train_count);

                for (size_t i = 0; i < cameras.size(); ++i) {
                    if (!enable_eval || (i % test_every) != 0) {
                        scene.addCamera(cameras[i]->image_name(), train_cameras_id,
                                        static_cast<int>(i), cameras[i]->uid(),
                                        lfs::core::path_to_utf8(cameras[i]->image_path()),
                                        lfs::core::path_to_utf8(cameras[i]->mask_path()));
                    }
                }

                if (enable_eval && val_count > 0) {
                    const auto val_cameras_id = scene.addCameraGroup(
                        std::format("Validation ({})", val_count), cameras_group_id, val_count);
                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if ((i % test_every) == 0) {
                            scene.addCamera(cameras[i]->image_name(), val_cameras_id,
                                            static_cast<int>(i), cameras[i]->uid(),
                                            lfs::core::path_to_utf8(cameras[i]->image_path()),
                                            lfs::core::path_to_utf8(cameras[i]->mask_path()));
                        }
                    }
                }

                LOG_INFO("Dataset '{}': {} train{} cameras{}",
                         dataset_name, train_count,
                         enable_eval ? std::format(" + {} val", val_count) : "",
                         mask_count > 0 ? std::format(" ({} masked)", mask_count) : "");
                return {};

            } else {
                return std::unexpected("Unknown data type from loader");
            }
        },
                          load_result.data);
    }

} // namespace lfs::training
