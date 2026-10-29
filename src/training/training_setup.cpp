/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#define GLM_ENABLE_EXPERIMENTAL

#include "training_setup.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/error.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/provenance.hpp"
#include "core/scene.hpp"
#include "core/sh_value_quant.hpp"
#include "core/shareable_allocation_limit.hpp"
#include "core/source_site.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "dataset.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "io/project_document.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "normal_auto_generate.hpp"
#include "rendering/mesh2splat.hpp"
#include "strategies/strategy_utils.hpp"
#include "trainer.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <variant>

namespace lfs::training {

    namespace {
        // The model local training continues from. A transient training model
        // is only a preview (e.g. a model trained elsewhere) that is never
        // saved, so training starts from the dataset instead.
        lfs::core::SplatData* localTrainingModel(lfs::core::Scene& scene) {
            const auto* node = scene.getNodeByUuid(scene.getTrainingModelNodeUuid());
            return node && node->transient ? nullptr : scene.getTrainingModel();
        }

        std::shared_ptr<lfs::core::PointCloud> createRandomPointCloud(const lfs::core::param::TrainingParameters& params) {
            const auto N = static_cast<size_t>(params.optimization.init_num_pts);
            const float extent = params.optimization.init_extent;
            auto positions = (lfs::core::Tensor::rand({N, 3}, lfs::core::Device::CPU) * 2.0f - 1.0f) * extent;
            auto colors = lfs::core::Tensor::randint({N, 3}, 0, 256, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            return std::make_shared<lfs::core::PointCloud>(positions, colors);
        }

        int effectiveMinTrackLengthForLoad(const lfs::core::param::TrainingParameters& params) {
            if (params.dataset.min_track_length > 0 &&
                params.init_path.has_value() &&
                !params.init_path->empty()) {
                LOG_WARN(
                    "min-track-length cannot be used with --init-ply; COLMAP sparse point filtering will not be applied because initialization uses '{}'",
                    *params.init_path);
                return 0;
            }
            return params.dataset.min_track_length;
        }

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

        lfs::io::CentralizeDataset parse_centralize(const std::string& s) {
            if (s == "by_pointcloud")
                return lfs::io::CentralizeDataset::ByPointCloud;
            if (s == "by_cameras")
                return lfs::io::CentralizeDataset::ByCameras;
            return lfs::io::CentralizeDataset::Off;
        }

        void applyTrainingSHDegree(lfs::core::SplatData& splat, const int target_degree) {
            const int before = splat.get_max_sh_degree();
            if (splat.set_sh_degree(target_degree)) {
                LOG_INFO("Adjusted training model SH degree: {} -> {}", before, splat.get_max_sh_degree());
            }
            if (splat.get_max_sh_degree() > 0 && splat.get_active_sh_degree() != 0) {
                splat.set_active_sh_degree(0);
            }
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

        constexpr float kShC0 = 0.28209479177387814f;

        lfs::Error initFileError(std::string message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Training,
                .user_message = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        bool isPlainPointCloudPly(const std::filesystem::path& path) {
            return lfs::core::path_to_utf8(path.extension()) == ".ply" && !lfs::io::is_gaussian_splat_ply(path) &&
                   !lfs::io::is_mesh_file(path);
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

        std::shared_ptr<lfs::core::PointCloud> pointCloudPreviewFromSplat(
            const lfs::core::SplatData& splat) {
            const auto n = static_cast<size_t>(splat.size());
            auto means = splat.means_raw().is_valid()
                             ? splat.means_raw().cpu().contiguous()
                             : lfs::core::Tensor::zeros({n, 3}, lfs::core::Device::CPU);
            auto colors = lfs::core::Tensor::zeros({n, 3}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            if (n > 0 && colors.data_ptr() != nullptr) {
                std::memset(colors.data_ptr(), 255, n * 3);
            }

            const auto& sh0 = splat.sh0_raw();
            if (n > 0 && sh0.is_valid() && sh0.numel() > 0) {
                try {
                    auto rgb = (sh0.cpu().slice(1, 0, 1).squeeze(1) * kShC0 + 0.5f)
                                   .clamp(0.0f, 1.0f)
                                   .contiguous();
                    if (rgb.is_valid() && rgb.ndim() == 2 &&
                        static_cast<size_t>(rgb.size(0)) == n && rgb.size(1) == 3 &&
                        rgb.dtype() == lfs::core::DataType::Float32 && rgb.ptr<float>() != nullptr) {
                        const float* src = rgb.ptr<float>();
                        auto* dst = static_cast<uint8_t*>(colors.data_ptr());
                        for (size_t i = 0; i < n * 3; ++i) {
                            const float scaled = src[i] * 255.0f;
                            dst[i] = static_cast<uint8_t>(
                                scaled < 0.0f ? 0.0f : (scaled > 255.0f ? 255.0f : scaled + 0.5f));
                        }
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Could not derive PointCloud colors from SH0: {}", e.what());
                }
            }

            return std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors));
        }

        // Mesh colors are linear, the point cloud stores display values: same 2.2 encode as the mesh viewer.
        std::shared_ptr<lfs::core::PointCloud> pointCloudPreviewFromMesh(const lfs::core::MeshData& mesh) {
            assert(mesh.vertices.ndim() == 2 && mesh.vertices.size(1) == 3);
            const auto n = static_cast<size_t>(mesh.vertex_count());
            auto means = mesh.vertices.cpu().contiguous();
            auto colors = mesh.has_colors()
                              ? (mesh.colors.cpu().slice(1, 0, 3).clamp(0.0f, 1.0f).pow(1.0f / 2.2f) * 255.0f + 0.5f)
                                    .to(lfs::core::DataType::UInt8)
                                    .contiguous()
                              : lfs::core::Tensor::full({n, 3}, 255.0f, lfs::core::Device::CPU,
                                                        lfs::core::DataType::UInt8);
            assert(colors.size(0) == n && colors.size(1) == 3);
            return std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors));
        }

        // Mesh2Splat emits one splat per texel of a square raster over the mesh bounds, so the
        // count grows with the square of the resolution. Refitting the resolution keeps the
        // whole surface covered within the budget, where dropping random splats leaves holes.
        lfs::Result<std::unique_ptr<lfs::core::SplatData>> splatsFromInitMesh(
            const lfs::core::MeshData& mesh,
            const std::filesystem::path& init_file,
            const int resolution,
            const int64_t budget) {
            lfs::core::Mesh2SplatOptions options;
            options.resolution_target = resolution;
            auto converted = lfs::rendering::mesh_to_splat(mesh, options);
            while (converted && budget > 0 && static_cast<int64_t>((*converted)->size()) > budget &&
                   options.resolution_target > lfs::core::Mesh2SplatOptions::kMinResolution) {
                const double shrink = std::sqrt(static_cast<double>(budget) / static_cast<double>((*converted)->size()));
                const int fitted_resolution = std::clamp(static_cast<int>(options.resolution_target * shrink),
                                                         lfs::core::Mesh2SplatOptions::kMinResolution,
                                                         options.resolution_target - 1);
                LOG_INFO("Mesh init: {} splats at resolution {} exceed the budget of {}, converting at resolution {}",
                         (*converted)->size(), options.resolution_target, budget, fitted_resolution);
                options.resolution_target = fitted_resolution;
                converted.value().reset();
                converted = lfs::rendering::mesh_to_splat(mesh, options);
            }
            if (!converted) {
                return initFileError(std::format("Failed to convert mesh '{}' to splats: {}",
                                                 lfs::core::path_to_utf8(init_file), converted.error()));
            }
            LOG_INFO("Mesh init: {} splats from {} triangles in {} (resolution {})",
                     (*converted)->size(), mesh.face_count(), lfs::core::path_to_utf8(init_file.filename()),
                     options.resolution_target);
            return std::move(*converted);
        }

        void warnIfMeshResolutionIgnored(const lfs::core::param::OptimizationParameters& opt) {
            if (opt.init_mesh_resolution != lfs::core::Mesh2SplatOptions{}.resolution_target) {
                LOG_WARN("init_mesh_resolution {} is ignored: it only applies when the init file is a mesh",
                         opt.init_mesh_resolution);
            }
        }

        // The init mesh is parsed once, by the import worker or the preview, and kept until training
        // start converts it; a changed write time means the file was edited in between.
        class PendingInitMesh {
        public:
            void keep(const std::filesystem::path& path, std::shared_ptr<lfs::core::MeshData> mesh) {
                std::lock_guard lock(mutex_);
                path_ = path;
                write_time_ = writeTime(path);
                mesh_ = std::move(mesh);
            }

            void clear() { keep({}, nullptr); }

            [[nodiscard]] std::shared_ptr<lfs::core::MeshData> peek(const std::filesystem::path& path) {
                std::lock_guard lock(mutex_);
                return matches(path) ? mesh_ : nullptr;
            }

            [[nodiscard]] std::shared_ptr<lfs::core::MeshData> take(const std::filesystem::path& path) {
                std::lock_guard lock(mutex_);
                const bool match = matches(path);
                auto mesh = std::move(mesh_);
                return match ? mesh : nullptr;
            }

        private:
            [[nodiscard]] bool matches(const std::filesystem::path& path) const {
                const auto now = writeTime(path);
                return mesh_ && path == path_ && (!now || now == write_time_);
            }

            static std::optional<std::filesystem::file_time_type> writeTime(const std::filesystem::path& path) {
                std::error_code error;
                const auto time = lfs::core::native_fs::last_write_time(path, error);
                return error ? std::nullopt : std::optional{time};
            }

            std::mutex mutex_;
            std::filesystem::path path_;
            std::optional<std::filesystem::file_time_type> write_time_;
            std::shared_ptr<lfs::core::MeshData> mesh_;
        };

        PendingInitMesh& pendingInitMesh() {
            static PendingInitMesh pending;
            return pending;
        }

        lfs::Result<std::shared_ptr<lfs::core::PointCloud>>
        loadInitReplacementPointCloud(const std::filesystem::path& init_file, const bool keep_mesh = false) {
            const auto filename = lfs::core::path_to_utf8(init_file.filename());
            const auto path_utf8 = lfs::core::path_to_utf8(init_file);

            if (const auto pending = keep_mesh ? pendingInitMesh().peek(init_file) : nullptr) {
                LOG_INFO("Init {} points from mesh vertices of {} (preview)", pending->vertex_count(), filename);
                return pointCloudPreviewFromMesh(*pending);
            }

            if (isPlainPointCloudPly(init_file)) {
                auto pc_result = lfs::io::load_ply_point_cloud(init_file);
                if (!pc_result) {
                    return initFileError(
                        std::format("Failed to load '{}': {}", path_utf8, pc_result.error()));
                }
                if (pc_result->size() <= 0) {
                    return initFileError(std::format("'{}' contains no points", path_utf8));
                }
                LOG_INFO("Init {} points from {} (replaces points3D)", pc_result->size(), filename);
                return std::make_shared<lfs::core::PointCloud>(std::move(*pc_result));
            }

            auto loader = lfs::io::Loader::create();
            auto init_result = loader->load(init_file);
            if (!init_result) {
                return initFileError(std::format(
                    "Failed to load '{}': {}", path_utf8, init_result.error().format()));
            }

            if (const auto* mesh_ptr = std::get_if<std::shared_ptr<lfs::core::MeshData>>(&init_result->data);
                mesh_ptr && *mesh_ptr) {
                auto preview = pointCloudPreviewFromMesh(**mesh_ptr);
                if (keep_mesh)
                    pendingInitMesh().keep(init_file, *mesh_ptr);
                LOG_INFO("Init {} points from mesh vertices of {} (preview)", preview->size(), filename);
                return preview;
            }

            auto* splat_ptr = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&init_result->data);
            if (!splat_ptr || !*splat_ptr) {
                return initFileError(std::format("'{}': invalid SplatData", path_utf8));
            }

            auto preview = pointCloudPreviewFromSplat(**splat_ptr);
            if (!preview || preview->size() <= 0) {
                return initFileError(std::format("'{}' contains no points", path_utf8));
            }
            LOG_INFO("Init {} points from splat {} (preview)", preview->size(), filename);
            return preview;
        }

        glm::vec3 centralizedDatasetOrigin(const std::optional<lfs::io::ImportGeoreference>& georeference) {
            if (!georeference) {
                return glm::vec3{0.0f};
            }
            using Provenance = lfs::io::ImportWorldOriginProvenance;
            if (georeference->world_origin_provenance != Provenance::CentralizeByCameras &&
                georeference->world_origin_provenance != Provenance::CentralizeByPointCloud) {
                return glm::vec3{0.0f};
            }
            const auto& origin = georeference->world_origin;
            return {static_cast<float>(origin[0]), static_cast<float>(origin[1]), static_cast<float>(origin[2])};
        }

        void centerInitializationMeans(lfs::core::Tensor& means, const glm::vec3& origin) {
            if (origin == glm::vec3{0.0f}) {
                return;
            }
            const auto shift = lfs::core::Tensor::from_vector(
                std::vector<float>{origin.x, origin.y, origin.z}, {3}, means.device());
            means = means - shift;
        }

        lfs::Result<void> attachDatasetPointCloud(
            const lfs::core::param::TrainingParameters& params,
            lfs::core::Scene& scene,
            const lfs::core::NodeId dataset_id,
            const lfs::io::LoadedScene& data,
            const bool verbose) {
            std::shared_ptr<lfs::core::PointCloud> point_cloud;
            if (!params.init_path.has_value() || params.init_path->empty()) {
                pendingInitMesh().clear();
            }
            if (params.init_path.has_value() && !params.init_path->empty()) {
                auto loaded = loadInitReplacementPointCloud(lfs::core::utf8_to_path(*params.init_path), true);
                if (!loaded) {
                    return lfs::Status::failure(loaded.error());
                }
                point_cloud = std::move(*loaded);
                centerInitializationMeans(point_cloud->means, scene.getTrainingDataOrigin());
            } else if (data.point_cloud && data.point_cloud->size() > 0) {
                point_cloud = data.point_cloud;
            } else {
                LOG_WARN("No sparse points available; using random initialization with {} points and extent {}",
                         params.optimization.init_num_pts, params.optimization.init_extent);
                point_cloud = createRandomPointCloud(params);
                if (verbose) {
                    LOG_INFO("Adding {} random points to scene", point_cloud->size());
                }
            }

            scene.setInitialPointCloud(point_cloud);
            scene.addPointCloud("PointCloud", point_cloud, dataset_id);
            return {};
        }

        bool evalMaskNeedsInitialPoints(const lfs::core::param::TrainingParameters& params) {
            return params.optimization.enable_eval &&
                   lfs::core::param::is_eval_mask_points(params.optimization.eval_mask);
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
            install.preserved_crop_polygon = context.preserved_crop_polygon;
            install.preserved_crop_polygon_transform = context.preserved_crop_polygon_transform;
            return install;
        }

        std::expected<TrainingModelGraphInstall, std::string> loadGaussianInitModel(
            const lfs::core::param::TrainingParameters& params,
            lfs::core::Scene& scene,
            const std::filesystem::path& init_file,
            const TrainingModelGraphCapture* graph_capture,
            const int64_t added_splat_count) {
            std::unique_ptr<lfs::core::SplatData> model;
            auto mesh = pendingInitMesh().take(init_file);
            if (!mesh) {
                auto loader = lfs::io::Loader::create();
                auto init_result = loader->load(init_file);
                if (!init_result) {
                    return std::unexpected(std::string(initFileError(std::format(
                                                                         "Failed to load '{}': {}",
                                                                         lfs::core::path_to_utf8(init_file),
                                                                         init_result.error().format()))
                                                           .user_message()));
                }
                if (auto* mesh_ptr = std::get_if<std::shared_ptr<lfs::core::MeshData>>(&init_result->data);
                    mesh_ptr && *mesh_ptr) {
                    mesh = std::move(*mesh_ptr);
                } else {
                    auto* splat_ptr = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&init_result->data);
                    if (!splat_ptr || !*splat_ptr) {
                        return std::unexpected(std::string(initFileError(std::format(
                                                                             "'{}': invalid SplatData",
                                                                             lfs::core::path_to_utf8(init_file)))
                                                               .user_message()));
                    }
                    warnIfMeshResolutionIgnored(params.optimization);
                    model = std::make_unique<lfs::core::SplatData>(std::move(**splat_ptr));
                }
            }
            if (mesh) {
                const int max_cap = params.optimization.max_cap;
                auto converted = splatsFromInitMesh(*mesh, init_file, params.optimization.init_mesh_resolution,
                                                    max_cap > 0 ? max_cap - added_splat_count : 0);
                if (!converted) {
                    return std::unexpected(std::string(converted.error().user_message()));
                }
                model = std::move(*converted);
            }

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

            return makeGraphInstall(graph_capture ? *graph_capture : captureTrainingModelGraph(scene),
                                    std::move(model));
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

        lfs::Result<std::vector<std::unique_ptr<lfs::core::SplatData>>> loadAddedSplats(
            const lfs::core::param::TrainingParameters& params) {
            std::vector<std::unique_ptr<lfs::core::SplatData>> added;
            if (params.add_splats_applied)
                return added;
            for (const auto& path : params.add_splat_paths) {
                auto splat = loadAddedSplat(path, params.optimization.sh_degree);
                if (!splat)
                    return initFileError(std::move(splat.error()));
                added.push_back(std::move(*splat));
            }
            return added;
        }

        std::expected<void, std::string> appendAddedSplats(
            const lfs::core::param::TrainingParameters& params,
            lfs::core::SplatData& model,
            const glm::vec3& dataset_origin,
            std::vector<std::unique_ptr<lfs::core::SplatData>> preloaded = {}) {
            if (params.add_splat_paths.empty() || params.add_splats_applied) {
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
                std::unique_ptr<lfs::core::SplatData> added;
                if (i < preloaded.size()) {
                    added = std::move(preloaded[i]);
                } else {
                    auto loaded = loadAddedSplat(params.add_splat_paths[i], params.optimization.sh_degree);
                    if (!loaded) {
                        return std::unexpected(loaded.error());
                    }
                    added = std::move(*loaded);
                }
                centerInitializationMeans(added->means(), dataset_origin);

                const size_t count = static_cast<size_t>(added->size());
                if (i < params.add_splat_freeze.size() && params.add_splat_freeze[i] && count > 0) {
                    frozen_ranges.push_back({base_count + added_count, count});
                    frozen_count += count;
                }
                added_count += count;
                splats.emplace_back(added.get(), glm::mat4{1.0f});
                owned_added_splats.push_back(std::move(added));
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

        [[nodiscard]] bool isAllocatorBackedTrainingTensorReady(const lfs::core::Tensor& tensor,
                                                                const size_t required_capacity) {
            if (!tensor.is_valid() || tensor.numel() == 0) {
                return required_capacity == 0;
            }
            if (!tensor.is_external_storage() || tensor.capacity() < required_capacity) {
                return false;
            }
            const auto kind = tensor.external_storage_kind();
            return kind == "vulkan_external_buffer" || kind == "splat.exportable";
        }

        std::expected<void, std::string> migrateTrainingModelToAllocatorImpl(
            const lfs::core::param::TrainingParameters& params,
            lfs::core::SplatData& model,
            const lfs::core::SplatTensorAllocator& tensor_allocator,
            const bool force_reallocation) {
            if (!tensor_allocator) {
                return {};
            }

            const size_t n = static_cast<size_t>(model.size());

            // exportable blocks commit live-N + headroom, not max_cap.
            // Requiring capacity >= max_cap made readiness always fail (allocator
            // clamps to the committed row budget), so every strategy step rebuilt
            // SplatData and wiped capacity_ensure → densify abort.
            //
            // Kind note: CUDA-only views use "splat.exportable"; the GUI interop
            // allocator tags the same clamped VMM block as "vulkan_external_buffer".
            // Treat both as live-N when committed capacity is below max_cap.
            const size_t configured_max =
                params.optimization.max_cap > 0
                    ? std::max<size_t>(static_cast<size_t>(params.optimization.max_cap), n)
                    : 0;
            const auto means_kind =
                model.means_raw().is_valid() ? model.means_raw().external_storage_kind()
                                             : std::string{};
            const bool exportable_or_interop =
                means_kind == "splat.exportable" || means_kind == "vulkan_external_buffer";
            const bool exportable_live_n =
                model.means_raw().is_valid() && model.means_raw().is_external_storage() &&
                exportable_or_interop && model.means_raw().capacity() > 0 &&
                (configured_max == 0 || model.means_raw().capacity() < configured_max);
            // For exportable live-N: stay on the current committed capacity (growth
            // is capacity_ensure's job). For other external kinds: target max_cap.
            const size_t target_capacity =
                exportable_live_n
                    ? std::max<size_t>(model.means_raw().capacity(), n)
                    : (configured_max > 0
                           ? configured_max
                           : std::max<size_t>(model.means_raw().capacity(), n));
            const auto layout_rest = static_cast<std::uint32_t>(model.max_sh_coeffs_rest());
            // Exportable/GUI path stores pad-dropped q16 codes; headless non-exportable
            // may still be float4-swizzle until quant. Size the readiness check from
            const bool shN_is_q16 = model.shN_value_quantized();
            // Single-buffer design: q16 is steady-state (always-commit). Densify
            // may hold a barrier-transient float workspace outside the exportable
            // block under render_mutex exclusive. Treat that float as migrate-ready
            // so ensureModel never remigrates/re-encodes mid-barrier.
            // commit restores q16 before the barrier ends.
            const bool shN_float_densify_workspace =
                layout_rest > 0 && model.shN_raw().is_valid() &&
                model.shN_raw().dtype() == lfs::core::DataType::Float32 && !shN_is_q16;
            const size_t target_shN_capacity =
                layout_rest == 0
                    ? 0
                    : (shN_is_q16
                           ? lfs::core::sh_value_quant::sh_value_u16_count(target_capacity, layout_rest)
                           : lfs::core::sh_swizzled_float_count(target_capacity, layout_rest));
            const size_t target_bounds_capacity =
                shN_is_q16 ? lfs::core::sh_value_quant::n_bounds_for_prims(target_capacity) * 2u
                           : 0;

            const bool shN_ready =
                target_shN_capacity == 0 || shN_float_densify_workspace ||
                isAllocatorBackedTrainingTensorReady(model.shN_raw(), target_shN_capacity);
            const bool bounds_ready =
                target_bounds_capacity == 0 || shN_float_densify_workspace ||
                isAllocatorBackedTrainingTensorReady(model.shN_value_bounds(),
                                                     target_bounds_capacity);

            const bool already_allocator_backed =
                isAllocatorBackedTrainingTensorReady(model.means_raw(), target_capacity) &&
                isAllocatorBackedTrainingTensorReady(model.sh0_raw(), target_capacity) &&
                isAllocatorBackedTrainingTensorReady(model.scaling_raw(), target_capacity) &&
                isAllocatorBackedTrainingTensorReady(model.rotation_raw(), target_capacity) &&
                isAllocatorBackedTrainingTensorReady(model.opacity_raw(), target_capacity) &&
                shN_ready && bounds_ready;
            if (already_allocator_backed && !force_reallocation) {
                model.set_tensor_allocator(tensor_allocator);
                return {};
            }

            try {
                // model = move(migrated) replaces private hooks; transfer densify grow.
                auto capacity_ensure = model.release_capacity_ensure();

                const int max_sh = model.get_max_sh_degree();
                const int active_sh = model.get_active_sh_degree();
                const float scene_scale = model.get_scene_scale();
                auto frozen_ranges = model.frozen_ranges();
                lfs::core::Tensor deleted = model.has_deleted_mask() ? model.deleted() : lfs::core::Tensor{};
                lfs::core::Tensor densification_info = model._densification_info;
                lfs::core::Tensor max_screen_share = model._max_screen_share;

                const auto copy_param =
                    [&](const lfs::core::Tensor& source,
                        const lfs::core::TensorShape& shape,
                        const size_t capacity,
                        const std::string_view name) -> lfs::core::Tensor {
                    lfs::core::Tensor source_cuda = source.device() == lfs::core::Device::CUDA
                                                        ? source
                                                        : source.cuda();
                    if (!source_cuda.is_contiguous()) {
                        source_cuda = source_cuda.contiguous();
                    }
                    lfs::core::Tensor dst = tensor_allocator(
                        shape,
                        capacity,
                        source_cuda.dtype(),
                        name);
                    dst.set_name(std::string{name});
                    dst.copy_from(source_cuda);
                    return dst;
                };

                lfs::core::Tensor means = copy_param(
                    model.means_raw(), model.means_raw().shape(), target_capacity, "SplatData.means");
                lfs::core::Tensor sh0 = copy_param(
                    model.sh0_raw(), model.sh0_raw().shape(), target_capacity, "SplatData.sh0");
                lfs::core::Tensor scaling = copy_param(
                    model.scaling_raw(), model.scaling_raw().shape(), target_capacity, "SplatData.scaling");
                lfs::core::Tensor rotation = copy_param(
                    model.rotation_raw(), model.rotation_raw().shape(), target_capacity, "SplatData.rotation");
                lfs::core::Tensor opacity = copy_param(
                    model.opacity_raw(), model.opacity_raw().shape(), target_capacity, "SplatData.opacity");

                lfs::core::Tensor shN;
                lfs::core::Tensor shN_bounds;
                bool need_q16_encode = false;
                if (layout_rest > 0 && model.shN_raw().is_valid() && model.shN_raw().numel() > 0) {
                    const size_t float_cap =
                        lfs::core::sh_swizzled_float_count(target_capacity, layout_rest);
                    const size_t q16_cap =
                        lfs::core::sh_value_quant::sh_value_u16_count(target_capacity, layout_rest);
                    const size_t bounds_cap =
                        lfs::core::sh_value_quant::n_bounds_for_prims(target_capacity) * 2u;

                    if (model.shN_value_quantized()) {
                        // Pad-dropped q16 codes + bounds → exportable/view target.
                        shN = copy_param(
                            model.shN_raw(), model.shN_raw().shape(), q16_cap, "SplatData.shN");
                        if (model.shN_value_bounds().is_valid() &&
                            model.shN_value_bounds().numel() > 0) {
                            shN_bounds = copy_param(
                                model.shN_value_bounds(),
                                model.shN_value_bounds().shape(),
                                bounds_cap,
                                "SplatData.shN_value_bounds");
                        }
                    } else {
                        // Try float topology install. Real SplatExportableStorage
                        // forces Float16 and clamps capacity to q16 cells — and rejects
                        // the swizzled float shape outright when live-N cells exceed the
                        // pad-dropped region — detect that and re-encode instead of
                        // bitcasting float into half.
                        lfs::core::Tensor src_float = model.shN_raw();
                        if (src_float.device() != lfs::core::Device::CUDA) {
                            src_float = src_float.cuda();
                        }
                        if (!src_float.is_contiguous()) {
                            src_float = src_float.contiguous();
                        }
                        lfs::core::Tensor installed;
                        try {
                            installed = tensor_allocator(src_float.shape(),
                                                         float_cap,
                                                         lfs::core::DataType::Float32,
                                                         "SplatData.shN");
                        } catch (const lfs::core::ShareableAllocationLimitError& error) {
                            LOG_INFO("Float shN install rejected by shareable allocation limit ({}); "
                                     "re-encoding to q16 instead",
                                     error.what());
                        } catch (const std::exception& error) {
                            // Exportable q16 region rejects the swizzled float shape when
                            // live-N cells exceed the pad-dropped capacity; same fallback
                            // as the Float16/clamp detection below.
                            LOG_DEBUG("Float shN install rejected by allocator ({}); "
                                      "re-encoding to q16 instead",
                                      error.what());
                        }
                        const bool landed_in_q16_exportable =
                            !installed.is_valid() ||
                            installed.dtype() == lfs::core::DataType::Float16 ||
                            installed.capacity() < float_cap;
                        if (landed_in_q16_exportable) {
                            shN = std::move(src_float);
                            need_q16_encode = true;
                        } else {
                            installed.set_name("SplatData.shN");
                            installed.copy_from(src_float);
                            shN = std::move(installed);
                        }
                    }
                }

                lfs::core::SplatData migrated(max_sh,
                                              std::move(means),
                                              std::move(sh0),
                                              std::move(shN),
                                              std::move(scaling),
                                              std::move(rotation),
                                              std::move(opacity),
                                              scene_scale,
                                              lfs::core::SplatData::ShNLayout::Swizzled);
                migrated.set_active_sh_degree(active_sh, std::move(shN_bounds));
                if (deleted.is_valid()) {
                    migrated.deleted() = std::move(deleted);
                    migrated.refresh_deleted_count();
                }
                if (densification_info.is_valid()) {
                    migrated._densification_info = std::move(densification_info);
                }
                if (max_screen_share.is_valid()) {
                    migrated._max_screen_share = std::move(max_screen_share);
                }
                migrated.set_frozen_ranges(std::move(frozen_ranges));
                model = std::move(migrated);
                model.set_tensor_allocator(tensor_allocator);
                if (capacity_ensure) {
                    model.set_capacity_ensure(std::move(capacity_ensure));
                }
                // Encode float/f16 rest into exportable q16 when the target block
                // is pad-dropped (probe above detected Float16-clamped ShN).
                // Single-buffer: cold migrate and any force rebuild land q16.
                if (need_q16_encode && model.shN_raw().is_valid() &&
                    !model.shN_value_quantized()) {
                    (void)lfs::training::sh_value::apply_shN_value_quant(model);
                }
                lfs::core::Tensor::trim_memory_pool();
            } catch (const std::exception& e) {
                return std::unexpected(std::format(
                    "Failed to migrate training SplatData to Vulkan-external storage: {}",
                    e.what()));
            }

            return {};
        }
    } // namespace

    std::expected<void, std::string> migrateTrainingModelToAllocator(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::SplatData& model,
        const lfs::core::SplatTensorAllocator& tensor_allocator,
        const bool force_reallocation) {
        return migrateTrainingModelToAllocatorImpl(params, model, tensor_allocator, force_reallocation);
    }

    lfs::io::Result<lfs::io::LoadResult> loadTrainingData(
        const lfs::core::param::TrainingParameters& params) {

        auto data_loader = lfs::io::Loader::create();

        const auto& data_path = params.dataset.data_path;
        lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .min_track_length = effectiveMinTrackLengthForLoad(params),
            .validate_only = false,
            .load_masks = params.optimization.mask_mode != lfs::core::param::MaskMode::None,
            .load_depths = params.optimization.use_depth_loss &&
                           params.optimization.depth_loss_weight > 0.0f,
            .load_normals = training_normal_priors_enabled(params.optimization) ||
                            params.optimization.enable_eval,
            .normal_auto_generate = params.optimization.normal_auto_generate,
            .depth_auto_generate = params.optimization.depth_auto_generate,
            .centralize = parse_centralize(params.dataset.centralize_dataset),
            .progress = [&data_path](float percentage, const std::string& message) {
                LOG_DEBUG("[{:5.1f}%] {}", percentage, message);
                lfs::core::events::state::DatasetLoadProgress{
                    .path = data_path,
                    .progress = percentage,
                    .step = message}
                    .emit();
            }};

        LOG_INFO("Loading dataset from: {}", lfs::core::path_to_utf8(params.dataset.data_path));
        auto result = data_loader->load(params.dataset.data_path, load_options);
        return result;
    }

    std::expected<void, std::string> loadTrainingDataIntoScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene) {
        auto load_result = loadTrainingData(params);
        if (!load_result) {
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error().format()));
        }

        return std::visit([&](auto&& data) -> std::expected<void, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                auto model = std::make_unique<lfs::core::SplatData>(std::move(*data));
                applyTrainingSHDegree(*model, params.optimization.sh_degree);
                const auto model_id = scene.addSplat("loaded_model", std::move(model));
                if (model_id == lfs::core::NULL_NODE) {
                    return std::unexpected("Failed to add loaded training model to scene");
                }
                scene.setTrainingModelNode(model_id);
                LOG_INFO("Loaded PLY directly into scene");
                return {};

            } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                scene.setSceneCenter(load_result->scene_center);
                scene.setTrainingDataOrigin(centralizedDatasetOrigin(load_result->georeference));
                scene.setImagesHaveAlpha(load_result->images_have_alpha);

                // Build dataset hierarchy in scene graph
                std::string dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.filename());
                if (dataset_name.empty()) {
                    dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.parent_path().filename());
                }
                if (dataset_name.empty()) {
                    dataset_name = "Dataset";
                }

                const auto dataset_id = scene.addDataset(dataset_name);

                if (auto attached = attachDatasetPointCloud(params, scene, dataset_id, data, true);
                    !attached) {
                    return std::unexpected(std::string(attached.error().user_message()));
                }

                auto cameras = data.cameras;
                const bool hold_out = params.optimization.holds_out_eval_images();
                const int test_every = params.dataset.test_every;
                if (hold_out && !params.resume_checkpoint) {
                    std::sort(cameras.begin(), cameras.end(), [](const auto& lhs, const auto& rhs) {
                        if (lhs->image_name() != rhs->image_name())
                            return lhs->image_name() < rhs->image_name();
                        return lhs->uid() < rhs->uid();
                    });
                }

                size_t train_count = 0;
                size_t val_count = 0;
                size_t mask_count = 0;
                for (size_t i = 0; i < cameras.size(); ++i) {
                    const bool is_eval = hold_out && (i % test_every) == 0;
                    cameras[i]->set_split(is_eval ? lfs::core::CameraSplit::Eval : lfs::core::CameraSplit::Train);
                    if (is_eval) {
                        val_count++;
                    } else {
                        train_count++;
                    }
                    if (cameras[i]->has_mask()) {
                        mask_count++;
                    }
                }

                const auto cameras_group_id = scene.addGroup("Cameras", dataset_id);

                const auto train_cameras_id = scene.addCameraGroup(
                    "Training",
                    cameras_group_id,
                    train_count);

                for (size_t i = 0; i < cameras.size(); ++i) {
                    if (!hold_out || (i % test_every) != 0) {
                        scene.addCamera(cameras[i]->image_name(), train_cameras_id, cameras[i]);
                    }
                }

                if (hold_out && val_count > 0) {
                    const auto val_cameras_id = scene.addCameraGroup(
                        "Validation",
                        cameras_group_id,
                        val_count);

                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if ((i % test_every) == 0) {
                            scene.addCamera(cameras[i]->image_name(), val_cameras_id, cameras[i]);
                        }
                    }
                }

                LOG_INFO("Loaded dataset '{}' into scene: {} train{} cameras{}",
                         dataset_name, train_count,
                         hold_out ? std::format(" + {} val", val_count) : "",
                         mask_count > 0 ? std::format(" ({} with masks)", mask_count) : "");
                return {};

            } else if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::MeshData>>) {
                assert(data && "MeshData must not be null");
                std::string mesh_name = lfs::core::path_to_utf8(params.dataset.data_path.stem());
                if (mesh_name.empty())
                    mesh_name = "mesh";
                scene.addMesh(mesh_name, data);
                LOG_INFO("Loaded mesh '{}' into scene", mesh_name);
                return {};

            } else {
                return std::unexpected("Unknown data type returned from loader");
            }
        },
                          load_result->data);
    }

    void preloadInitMesh(const lfs::core::param::TrainingParameters& params) {
        if (!params.init_path.has_value() || params.init_path->empty()) {
            return;
        }
        const auto init_file = lfs::core::utf8_to_path(*params.init_path);
        if (!lfs::io::is_mesh_file(init_file)) {
            return;
        }
        auto loaded = lfs::io::Loader::create()->load(init_file);
        if (auto* mesh = loaded ? std::get_if<std::shared_ptr<lfs::core::MeshData>>(&loaded->data) : nullptr;
            mesh && *mesh) {
            pendingInitMesh().keep(init_file, std::move(*mesh));
        }
    }

    lfs::Result<std::shared_ptr<lfs::core::PointCloud>> loadInitialPointCloud(
        const lfs::core::param::TrainingParameters& params, const glm::vec3& training_origin) {
        if (params.init_path.has_value() && !params.init_path->empty()) {
            auto loaded = loadInitReplacementPointCloud(lfs::core::utf8_to_path(*params.init_path));
            if (loaded)
                centerInitializationMeans((*loaded)->means, training_origin);
            return loaded;
        }

        const lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .min_track_length = effectiveMinTrackLengthForLoad(params),
            .centralize = parse_centralize(params.dataset.centralize_dataset)};
        const auto data_path = lfs::core::path_to_utf8(params.dataset.data_path);
        auto load_result = lfs::io::Loader::create()->load(params.dataset.data_path, load_options);
        if (!load_result)
            return initFileError(std::format("Failed to load '{}': {}", data_path, load_result.error().format()));
        const auto* loaded = std::get_if<lfs::io::LoadedScene>(&load_result->data);
        if (!loaded || !loaded->point_cloud || loaded->point_cloud->size() <= 0)
            return initFileError(std::format("'{}' has no sparse points and no --init file was given", data_path));
        return loaded->point_cloud;
    }

    TrainingModelGraphCapture captureTrainingModelGraph(lfs::core::Scene& scene) {
        TrainingModelGraphCapture context;
        context.training_model = localTrainingModel(scene);
        context.scene_center = scene.getSceneCenter();
        context.training_data_origin = scene.getTrainingDataOrigin();
        if (const auto* node = scene.getDatasetPointCloudNode()) {
            context.point_cloud_node_id = node->id;
            context.parent_id = node->parent_id;
            context.node_transform = node->transform();
            context.point_cloud = *node->point_cloud;
        }
        if (context.point_cloud_node_id == lfs::core::NULL_NODE) {
            return context;
        }

        const auto local_to_point_cloud = [&](const lfs::core::SceneNode& volume) {
            return volume.parent_id == context.point_cloud_node_id
                       ? volume.transform()
                       : glm::inverse(scene.getWorldTransform(context.point_cloud_node_id)) *
                             scene.getWorldTransform(volume.id);
        };
        if (const auto* polygon_node = scene.getNodeById(scene.getCropPolygonForSplat(context.point_cloud_node_id));
            polygon_node && polygon_node->crop_polygon) {
            context.preserved_crop_polygon = *polygon_node->crop_polygon;
            context.preserved_crop_polygon_transform = local_to_point_cloud(*polygon_node);
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
        context.preserved_cropbox_transform = local_to_point_cloud(*cropbox_node);
        context.has_preserved_cropbox = true;
        return context;
    }

    std::expected<std::optional<TrainingModelGraphInstall>, std::string> prepareTrainingModel(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::core::SplatTensorAllocator tensor_allocator,
        const TrainingModelGraphCapture* graph_capture) {

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
            graph_capture ? graph_capture->training_model != nullptr : localTrainingModel(scene) != nullptr;
        if (!has_training_model) {
            if (const auto init_file = gaussianSplatInitPath(params)) {
                auto added = loadAddedSplats(params);
                if (!added) {
                    return std::unexpected(std::string(added.error().user_message()));
                }
                int64_t added_count = 0;
                for (const auto& splat : *added)
                    added_count += static_cast<int64_t>(splat->size());
                auto loaded = loadGaussianInitModel(params, scene, *init_file, graph_capture, added_count);
                if (!loaded) {
                    return std::unexpected(std::move(loaded.error()));
                }
                if (auto result = appendAddedSplats(params, *loaded->model, dataset_origin, std::move(*added)); !result) {
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
                loaded->keep_initial_point_cloud = evalMaskNeedsInitialPoints(params);
                return std::optional<TrainingModelGraphInstall>{std::move(*loaded)};
            }
        }

        if (auto* model = graph_capture ? graph_capture->training_model : localTrainingModel(scene)) {
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
        warnIfMeshResolutionIgnored(params.optimization);
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
            LOG_WARN("No sparse points available; using random initialization with {} points and extent {}",
                     params.optimization.init_num_pts, params.optimization.init_extent);
            point_cloud_to_use = *createRandomPointCloud(params);
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
                scene_center = max_cap > 0 ? mean : mean.cuda();
            } else {
                scene_center = lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU);
            }
        } else {
            scene_center = max_cap > 0 ? scene_center.cpu() : scene_center.cuda();
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
        auto install = makeGraphInstall(context, std::move(model));
        install.keep_initial_point_cloud = evalMaskNeedsInitialPoints(params);
        return std::optional<TrainingModelGraphInstall>{std::move(install)};
    }

    std::expected<void, std::string> installTrainingModel(
        lfs::core::Scene& scene,
        TrainingModelGraphInstall&& install) {
        if (!install.model) {
            return std::unexpected("Training model install is missing splat data");
        }

        if (install.point_cloud_node_id != lfs::core::NULL_NODE) {
            if (const auto* pc_node = scene.getNodeById(install.point_cloud_node_id)) {
                auto initial_point_cloud = install.keep_initial_point_cloud ? scene.getInitialPointCloud() : nullptr;
                scene.removeNode(pc_node->name, false);
                if (initial_point_cloud)
                    scene.setInitialPointCloud(std::move(initial_point_cloud));
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
        if (install.preserved_crop_polygon) {
            static_cast<void>(scene.addCropPolygon("Model_crop_polygon", model_id, *install.preserved_crop_polygon,
                                                   install.preserved_crop_polygon_transform));
        }
        return {};
    }

    lfs::Status applyTrainingCropBox(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene) {
        if (!params.cropbox) {
            return {};
        }
        const auto* model = scene.getNodeByUuid(scene.getTrainingModelNodeUuid());
        if (!model) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "--cropbox needs the training model in the scene",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
        lfs::core::NodeId cropbox_id = scene.getCropBoxForSplat(model->id);
        if (cropbox_id == lfs::core::NULL_NODE) {
            cropbox_id = scene.addCropBox("Model_cropbox", model->id);
        }
        if (cropbox_id == lfs::core::NULL_NODE) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "Failed to add the --cropbox region of interest to the training model",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
        const auto& box = *params.cropbox;
        lfs::core::CropBoxData data = scene.getCropBoxData(cropbox_id) ? *scene.getCropBoxData(cropbox_id)
                                                                       : lfs::core::CropBoxData{};
        data.min = {box[0], box[1], box[2]};
        data.max = {box[3], box[4], box[5]};
        data.inverse = false;
        data.enabled = true;
        LOG_INFO("Region of interest: crop box ({}, {}, {}) to ({}, {}, {})", box[0], box[1], box[2], box[3], box[4],
                 box[5]);
        glm::mat4 transform{1.0f};
        if (params.cropbox_rotation) {
            // Rx * Ry * Rz about the dataset's axes, the frame of the corners, turning the box about its centre.
            const auto& degrees = *params.cropbox_rotation;
            transform = glm::translate(glm::mat4{1.0f}, (data.min + data.max) * 0.5f) *
                        glm::eulerAngleXYZ(glm::radians(degrees[0]), glm::radians(degrees[1]), glm::radians(degrees[2]));
            data.max = (data.max - data.min) * 0.5f;
            data.min = -data.max;
            LOG_INFO("Region of interest: crop box turned by ({}, {}, {}) degrees about its centre", degrees[0],
                     degrees[1], degrees[2]);
        }
        scene.setCropBoxData(cropbox_id, data);
        scene.setNodeTransform(cropbox_id, transform);
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
        if (auto installed = installTrainingModel(scene, std::move(**prepared)); !installed) {
            return installed;
        }
        if (auto applied = applyTrainingCropBox(params, scene); !applied) {
            return std::unexpected(std::string(applied.error().user_message()));
        }
        return {};
    }

    std::expected<void, std::string> validateDatasetPath(
        const lfs::core::param::TrainingParameters& params) {

        auto data_loader = lfs::io::Loader::create();

        lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .min_track_length = params.dataset.min_track_length,
            .validate_only = true,
            .load_masks = params.optimization.mask_mode != lfs::core::param::MaskMode::None,
            .load_depths = params.optimization.use_depth_loss &&
                           params.optimization.depth_loss_weight > 0.0f,
            .load_normals = training_normal_priors_enabled(params.optimization),
            .normal_auto_generate = params.optimization.normal_auto_generate,
            .depth_auto_generate = params.optimization.depth_auto_generate};

        auto result = data_loader->load(params.dataset.data_path, load_options);
        if (!result) {
            return std::unexpected(result.error().format());
        }
        return {};
    }

    std::expected<void, std::string> applyLoadResultToScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::io::LoadResult&& load_result) {

        return std::visit([&](auto&& data) -> std::expected<void, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                auto model = std::make_unique<lfs::core::SplatData>(std::move(*data));
                applyTrainingSHDegree(*model, params.optimization.sh_degree);
                const auto model_id = scene.addSplat("loaded_model", std::move(model));
                if (model_id == lfs::core::NULL_NODE) {
                    return std::unexpected("Failed to add loaded training model to scene");
                }
                scene.setTrainingModelNode(model_id);
                return {};

            } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                scene.setSceneCenter(load_result.scene_center);
                scene.setTrainingDataOrigin(centralizedDatasetOrigin(load_result.georeference));
                scene.setImagesHaveAlpha(load_result.images_have_alpha);

                std::string dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.filename());
                if (dataset_name.empty()) {
                    dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.parent_path().filename());
                }
                if (dataset_name.empty()) {
                    dataset_name = "Dataset";
                }

                const auto dataset_id = scene.addDataset(dataset_name);

                if (auto attached = attachDatasetPointCloud(params, scene, dataset_id, data, false);
                    !attached) {
                    return std::unexpected(std::string(attached.error().user_message()));
                }

                auto cameras = data.cameras;
                const bool hold_out = params.optimization.holds_out_eval_images();
                const int test_every = params.dataset.test_every;
                if (hold_out && !params.resume_checkpoint) {
                    std::sort(cameras.begin(), cameras.end(), [](const auto& lhs, const auto& rhs) {
                        if (lhs->image_name() != rhs->image_name())
                            return lhs->image_name() < rhs->image_name();
                        return lhs->uid() < rhs->uid();
                    });
                }

                size_t train_count = 0, val_count = 0, mask_count = 0;
                for (size_t i = 0; i < cameras.size(); ++i) {
                    const bool is_val = hold_out && (i % test_every) == 0;
                    cameras[i]->set_split(is_val ? lfs::core::CameraSplit::Eval : lfs::core::CameraSplit::Train);
                    is_val ? ++val_count : ++train_count;
                    if (cameras[i]->has_mask())
                        ++mask_count;
                }

                const auto cameras_group_id = scene.addGroup("Cameras", dataset_id);
                const auto train_cameras_id = scene.addCameraGroup(
                    "Training", cameras_group_id, train_count);

                for (size_t i = 0; i < cameras.size(); ++i) {
                    if (!hold_out || (i % test_every) != 0) {
                        scene.addCamera(cameras[i]->image_name(), train_cameras_id, cameras[i]);
                    }
                }

                if (hold_out && val_count > 0) {
                    const auto val_cameras_id = scene.addCameraGroup(
                        "Validation", cameras_group_id, val_count);
                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if ((i % test_every) == 0) {
                            scene.addCamera(cameras[i]->image_name(), val_cameras_id, cameras[i]);
                        }
                    }
                }

                LOG_INFO("Dataset '{}': {} train{} cameras{}",
                         dataset_name, train_count,
                         hold_out ? std::format(" + {} val", val_count) : "",
                         mask_count > 0 ? std::format(" ({} masked)", mask_count) : "");
                return {};

            } else if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::MeshData>>) {
                assert(data && "MeshData must not be null");
                std::string mesh_name = lfs::core::path_to_utf8(params.dataset.data_path.stem());
                if (mesh_name.empty())
                    mesh_name = "mesh";
                scene.addMesh(mesh_name, data);
                LOG_INFO("Loaded mesh '{}' into scene", mesh_name);
                return {};

            } else {
                return std::unexpected("Unknown data type from loader");
            }
        },
                          load_result.data);
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
        if (params.dataset.data_path.empty()) {
            return std::unexpected(
                "Project checkpoint has no dataset path");
        }
        if (!lfs::core::native_fs::exists(
                params.dataset.data_path)) {
            return std::unexpected(std::format(
                "Dataset path does not exist: {}",
                lfs::core::path_to_utf8(
                    params.dataset.data_path)));
        }

        auto trainer = std::make_unique<Trainer>(scene);
        auto source_nodes = document.scene_graph().nodes();
        if (!source_nodes) {
            return std::unexpected(std::format(
                "Failed to read project scene graph for snapshot bindings: {}",
                lfs::format_for_developer(source_nodes.error())));
        }
        lfs::io::project::ScenePayloadBindings snapshot_bindings;
        const auto training_uuid = scene.getTrainingModelNodeUuid();
        for (const auto& node : *source_nodes) {
            if (node.uuid != training_uuid && node.payload) {
                snapshot_bindings.emplace(node.uuid, *node.payload);
            }
        }
        trainer->set_project_snapshot_payload_bindings(
            std::move(snapshot_bindings));
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
        // A metadata save can change PRMS after the last CKPT was captured.
        // Explicit command-line overrides still take precedence on resume.
        auto resumed_params = trainer->getParams();
        if (!params.overrides.has_optimization_key("keep_all_checkpoints")) {
            auto retention = document.parameters().checkpoint_retention(
                resumed_params.optimization.strategy, resumed_params.optimization.keep_all_checkpoints);
            if (!retention)
                return std::unexpected(lfs::format_for_developer(retention.error()));
            resumed_params.optimization.keep_all_checkpoints = *retention;
            trainer->setParams(resumed_params);
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
                ? params.dataset.project_file()
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
            case OutputFormat::HTML: {
                std::optional<glm::vec3> bg_color;
                if (params.optimization.bg_mode == lfs::core::param::BackgroundMode::SolidColor) {
                    const auto& train_bg = params.optimization.bg_color;
                    bg_color = glm::vec3(train_bg[0], train_bg[1], train_bg[2]);
                }
                return lfs::io::export_html(splat, {.output_path = output,
                                                    .kmeans_iterations = 10,
                                                    .provenance = provenance,
                                                    .background_color = bg_color});
            }
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

    lfs::Result<std::optional<lfs::core::SplatData>> exclude_frozen_rows(const lfs::core::SplatData& model) {
        if (!model.has_frozen_ranges())
            return std::optional<lfs::core::SplatData>{};
        const auto rows = static_cast<size_t>(model.size());
        auto keep = make_frozen_mask(model, rows, model.means().device()).logical_not();
        assert(keep.ndim() == 1 && keep.numel() == rows);
        if (model.deleted_mask_matches_size())
            keep = keep.logical_and(model.deleted().logical_not());
        auto kept = lfs::core::extract_by_mask(model, keep);
        if (kept.size() == 0) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "Final export failed: every row is a frozen added splat.",
                .detail = "--exclude-export leaves no rows to export",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        return std::optional<lfs::core::SplatData>{std::move(kept)};
    }

    lfs::Status export_final_splats(const Trainer& trainer,
                                    const lfs::core::param::TrainingParameters& params) {
        if (params.export_formats.empty()) {
            return {};
        }
        const auto& trained = trainer.get_strategy().get_model();
        std::optional<lfs::core::SplatData> without_frozen;
        if (params.exclude_frozen_add_splats_from_export ||
            trainer.getParams().exclude_frozen_add_splats_from_export) {
            auto kept = exclude_frozen_rows(trained);
            if (!kept)
                return lfs::Status::failure(std::move(kept).error());
            without_frozen = std::move(*kept);
        }
        const auto& unfrozen = without_frozen ? *without_frozen : trained;
        // A stopped run keeps every splat so it can resume; its exports still hold only the crop box.
        std::optional<lfs::core::SplatData> cropped;
        if (const auto cropbox = trainer.export_cropbox()) {
            cropped = crop_copy_to_training_cropbox(unfrozen, *cropbox);
            if (cropped && cropped->size() == 0) {
                return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::FailedPrecondition,
                    .domain = lfs::ErrorDomain::Training,
                    .user_message = "Final export failed: no splat lies inside the crop box or polygon.",
                    .detail = "--crop leaves no rows to export",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
            if (cropped)
                LOG_INFO("Exporting the splats inside the crop box: {} of {}", cropped->size(), unfrozen.visible_count());
        }
        const auto& model = cropped ? *cropped : unfrozen;
        const std::filesystem::path out_dir = params.dataset.output_path;
        const std::string stem = params.dataset.output_name.empty()
                                     ? std::format("splat_{}", trainer.get_current_iteration())
                                     : params.dataset.output_stem();

        lfs::core::ProvenanceStamp stamp = params.include_provenance
                                               ? lfs::core::make_provenance_stamp()
                                               : lfs::core::make_minimal_provenance_stamp();
        if (params.include_provenance) {
            stamp.iteration = trainer.get_current_iteration();
            stamp.strategy = params.optimization.strategy;
        }

        std::string failed;
        for (const auto format : params.export_formats) {
            const std::filesystem::path path = out_dir / (stem + final_export_extension(format));
            const std::string file = lfs::core::path_to_utf8(path);
            try {
                if (const auto result = save_final_splat(model, path, format, stamp, params); !result) {
                    LOG_ERROR("Failed to export final splat to {}: {}", file, result.error().message);
                    failed += failed.empty() ? file : ", " + file;
                    continue;
                }
                LOG_INFO("Exported final splat: {}", file);
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to export final splat to {}: {}", file, e.what());
                failed += failed.empty() ? file : ", " + file;
            }
        }
        if (failed.empty()) {
            return {};
        }
        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Internal,
            .domain = lfs::ErrorDomain::IO,
            .user_message = "Final export failed.",
            .detail = std::format("Final export failed for {}", failed),
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
    }

} // namespace lfs::training
