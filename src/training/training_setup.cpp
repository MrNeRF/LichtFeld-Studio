/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_setup.hpp"
#include "core/camera.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "dataset.hpp"
#include "io/loader.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

namespace lfs::training {

    namespace {
        std::shared_ptr<lfs::core::PointCloud> createRandomPointCloud() {
            constexpr size_t N = 10000;
            auto positions = lfs::core::Tensor::rand({N, 3}, lfs::core::Device::CPU) * 2.0f - 1.0f;
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

        constexpr int kSphericalUndistortViewCount = 6;

        // Cube faces, widened from 90 to 96 degrees.
        //
        // All views expanded from one panorama share that panorama's camera centre,
        // so overlapping views carry no extra parallax: they resample the same rays
        // twice. Redundant coverage is therefore pure cost, and worse, it is uneven
        // cost -- an icosahedral 12-view layout at 90 degrees covers 25% of the
        // sphere once, 50% twice and 25% three times, which weights the photometric
        // loss by up to 3x purely as an artefact of the view geometry.
        //
        // Six faces tile the sphere exactly. The extra 3 degrees on each edge is a
        // deliberate seam margin so SSIM windows and densification see real context
        // across face boundaries, and it costs only ~12% angular overcoverage.
        constexpr float kSphericalUndistortFovDegrees = 96.0f;

        struct SphericalViewDefinition {
            std::string_view suffix;
            std::array<float, 3> forward;
        };

        constexpr std::array<SphericalViewDefinition, kSphericalUndistortViewCount> kSphericalUndistortViews{{
            {"cube_px", {1.f, 0.f, 0.f}},
            {"cube_nx", {-1.f, 0.f, 0.f}},
            {"cube_py", {0.f, 1.f, 0.f}},
            {"cube_ny", {0.f, -1.f, 0.f}},
            {"cube_pz", {0.f, 0.f, 1.f}},
            {"cube_nz", {0.f, 0.f, -1.f}},
        }};

        using Vec3 = std::array<float, 3>;

        [[nodiscard]] Vec3 cross(const Vec3& a, const Vec3& b) {
            return {
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]};
        }

        [[nodiscard]] float length(const Vec3& v) {
            return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        }

        [[nodiscard]] Vec3 normalize(const Vec3& v) {
            const float len = length(v);
            if (len <= 0.0f)
                return {0.0f, 0.0f, 1.0f};
            return {v[0] / len, v[1] / len, v[2] / len};
        }

        [[nodiscard]] std::array<float, 9> makeViewFromForward(const Vec3& forward_in) {
            const Vec3 forward = normalize(forward_in);
            constexpr Vec3 world_up{0.0f, 1.0f, 0.0f};
            constexpr Vec3 fallback_up{0.0f, 0.0f, 1.0f};

            Vec3 right = cross(world_up, forward);
            if (length(right) < 1e-5f)
                right = cross(fallback_up, forward);
            right = normalize(right);
            const Vec3 up = cross(forward, right);

            return {
                right[0], right[1], right[2],
                up[0], up[1], up[2],
                forward[0], forward[1], forward[2]};
        }

        [[nodiscard]] std::string sanitizeFilenameComponent(const std::string_view value) {
            std::string out;
            out.reserve(value.size());
            for (const unsigned char ch : value) {
                if (std::isalnum(ch) || ch == '-' || ch == '_') {
                    out.push_back(static_cast<char>(ch));
                } else {
                    out.push_back('_');
                }
            }
            while (!out.empty() && out.back() == '_')
                out.pop_back();
            return out.empty() ? "camera" : out;
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Camera> makeSphericalViewCamera(
            const lfs::core::Camera& source,
            const SphericalViewDefinition& view,
            const int view_size,
            const int source_width,
            const int source_height,
            const int uid,
            const std::string& image_name,
            const bool has_alpha) {
            const auto pano_to_view = makeViewFromForward(view.forward);
            auto R_cpu = source.R().cpu();
            if (!R_cpu.is_contiguous())
                R_cpu = R_cpu.contiguous();
            auto T_cpu = source.T().cpu();
            if (!T_cpu.is_contiguous())
                T_cpu = T_cpu.contiguous();

            auto R_acc = R_cpu.accessor<float, 2>();
            auto T_acc = T_cpu.accessor<float, 1>();

            std::vector<float> R_face(9, 0.0f);
            std::vector<float> T_face(3, 0.0f);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    for (int k = 0; k < 3; ++k)
                        R_face[row * 3 + col] += pano_to_view[row * 3 + k] * R_acc(k, col);
                }
                for (int k = 0; k < 3; ++k)
                    T_face[row] += pano_to_view[row * 3 + k] * T_acc(k);
            }

            const float focal = lfs::core::cube_face_focal(kSphericalUndistortFovDegrees, view_size);
            const float center = 0.5f * static_cast<float>(view_size);
            auto camera = std::make_shared<lfs::core::Camera>(
                lfs::core::Tensor::from_vector(R_face, {3, 3}, lfs::core::Device::CPU),
                lfs::core::Tensor::from_vector(T_face, {3}, lfs::core::Device::CPU),
                focal,
                focal,
                center,
                center,
                lfs::core::Tensor::empty({0}, lfs::core::Device::CPU),
                lfs::core::Tensor::empty({0}, lfs::core::Device::CPU),
                lfs::core::CameraModelType::PINHOLE,
                image_name,
                source.image_path(),
                source.has_mask() ? source.mask_path() : std::filesystem::path{},
                view_size,
                view_size,
                uid,
                source.camera_id(),
                source.has_depth() ? source.depth_path() : std::filesystem::path{});
            camera->set_has_alpha(has_alpha);
            camera->set_split(source.split());
            camera->set_cube_face_projection(lfs::core::CubeFaceProjection{
                .pano_to_face = pano_to_view,
                .face_size = view_size,
                .source_width = source_width,
                .source_height = source_height,
                .fov_degrees = kSphericalUndistortFovDegrees});
            return camera;
        }

        [[nodiscard]] std::expected<std::vector<std::shared_ptr<lfs::core::Camera>>, std::string>
        expandEquirectangularCamerasForUndistortImpl(
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
            int next_uid = 0;
            const bool has_passthrough_cameras = std::any_of(cameras.begin(), cameras.end(), [](const auto& camera) {
                return camera &&
                       camera->camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR;
            });
            if (has_passthrough_cameras) {
                for (const auto& camera : cameras) {
                    if (camera)
                        next_uid = std::max(next_uid, camera->uid() + 1);
                }
            }

            std::vector<std::shared_ptr<lfs::core::Camera>> expanded;
            expanded.reserve(cameras.size() * kSphericalUndistortViews.size());

            size_t panoramas = 0;
            size_t virtual_views = 0;
            size_t projected_depth_maps = 0;
            int min_face_size = std::numeric_limits<int>::max();
            int max_face_size = 0;

            for (const auto& camera : cameras) {
                if (!camera ||
                    camera->camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR) {
                    expanded.push_back(camera);
                    if (!has_passthrough_cameras && camera)
                        next_uid = std::max(next_uid, camera->uid() + 1);
                    continue;
                }

                ++panoramas;
                int source_width = 0;
                int source_height = 0;
                int source_channels = 0;
                try {
                    std::tie(source_width, source_height, source_channels) =
                        lfs::core::get_image_info(camera->image_path());
                } catch (const std::exception& e) {
                    return std::unexpected(std::format("Failed to inspect panorama '{}': {}",
                                                       lfs::core::path_to_utf8(camera->image_path()), e.what()));
                }
                if (source_width <= 0 || source_height <= 0 || source_channels <= 0) {
                    return std::unexpected(std::format("Failed to inspect panorama '{}'",
                                                       lfs::core::path_to_utf8(camera->image_path())));
                }

                const int view_size = lfs::core::cube_face_size_for_panorama(
                    source_width, source_height, kSphericalUndistortFovDegrees);
                min_face_size = std::min(min_face_size, view_size);
                max_face_size = std::max(max_face_size, view_size);

                const std::string stem = sanitizeFilenameComponent(
                    std::filesystem::path(camera->image_name()).stem().string());
                const std::string base_name = std::format("{:06d}_{}", camera->uid(), stem);

                if (camera->has_depth())
                    ++projected_depth_maps;

                for (size_t view_index = 0; view_index < kSphericalUndistortViews.size(); ++view_index) {
                    const auto& view = kSphericalUndistortViews[view_index];
                    const std::string image_name =
                        std::format("{}_{}.png", base_name, view.suffix);
                    expanded.push_back(makeSphericalViewCamera(
                        *camera,
                        view,
                        view_size,
                        source_width,
                        source_height,
                        next_uid++,
                        image_name,
                        camera->has_alpha()));
                    ++virtual_views;
                }
            }

            if (panoramas > 0) {
                LOG_INFO("Undistort expanded {} equirectangular panorama{} into {} internal spherical pinhole views "
                         "({} views/panorama, {:.0f} deg FOV, face size {} px auto, sampled internally from source panoramas)",
                         panoramas,
                         panoramas == 1 ? "" : "s",
                         virtual_views,
                         kSphericalUndistortViews.size(),
                         kSphericalUndistortFovDegrees,
                         min_face_size == max_face_size ? std::to_string(max_face_size)
                                                        : std::format("{}-{}", min_face_size, max_face_size));
                if (projected_depth_maps > 0) {
                    LOG_INFO("Projected {} panorama depth map{} onto the generated faces "
                             "(nearest sampling, radial distance converted to camera-space z)",
                             projected_depth_maps,
                             projected_depth_maps == 1 ? "" : "s");
                }
            }

            return expanded;
        }

        void applyTrainingSHDegree(lfs::core::SplatData& splat, const int target_degree) {
            const int before = splat.get_max_sh_degree();
            if (splat.set_sh_degree(target_degree)) {
                LOG_INFO("Adjusted training model SH degree: {} -> {}", before, splat.get_max_sh_degree());
            }
            if (splat.get_max_sh_degree() > 0 && splat.get_active_sh_degree() != 0) {
                const int active_before = splat.get_active_sh_degree();
                splat.set_active_sh_degree(0);
                LOG_INFO("Training SH schedule active degree: {} -> 0 (max {})",
                         active_before, splat.get_max_sh_degree());
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
            lfs::core::SplatData& model) {
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
            const size_t target_capacity =
                params.optimization.max_cap > 0
                    ? std::max<size_t>(static_cast<size_t>(params.optimization.max_cap), n)
                    : std::max<size_t>(model.means_raw().capacity(), n);
            const auto layout_rest = static_cast<std::uint32_t>(model.max_sh_coeffs_rest());
            const size_t target_shN_capacity =
                layout_rest > 0 ? lfs::core::sh_swizzled_float_count(target_capacity, layout_rest) : 0;

            const bool already_allocator_backed =
                isAllocatorBackedTrainingTensorReady(model.means_raw(), target_capacity) &&
                isAllocatorBackedTrainingTensorReady(model.sh0_raw(), target_capacity) &&
                isAllocatorBackedTrainingTensorReady(model.scaling_raw(), target_capacity) &&
                isAllocatorBackedTrainingTensorReady(model.rotation_raw(), target_capacity) &&
                isAllocatorBackedTrainingTensorReady(model.opacity_raw(), target_capacity) &&
                (target_shN_capacity == 0 ||
                 isAllocatorBackedTrainingTensorReady(model.shN_raw(), target_shN_capacity));
            if (already_allocator_backed && !force_reallocation) {
                model.set_tensor_allocator(tensor_allocator);
                return {};
            }

            try {
                const int max_sh = model.get_max_sh_degree();
                const int active_sh = model.get_active_sh_degree();
                const float scene_scale = model.get_scene_scale();
                auto frozen_ranges = model.frozen_ranges();
                lfs::core::Tensor deleted = model.has_deleted_mask() ? model.deleted() : lfs::core::Tensor{};
                lfs::core::Tensor densification_info = model._densification_info;

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
                if (target_shN_capacity > 0 && model.shN_raw().is_valid() && model.shN_raw().numel() > 0) {
                    shN = copy_param(
                        model.shN_raw(), model.shN_raw().shape(), target_shN_capacity, "SplatData.shN");
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
                migrated.set_active_sh_degree(active_sh);
                if (deleted.is_valid()) {
                    migrated.deleted() = std::move(deleted);
                }
                if (densification_info.is_valid()) {
                    migrated._densification_info = std::move(densification_info);
                }
                migrated.set_frozen_ranges(std::move(frozen_ranges));
                model = std::move(migrated);
                model.set_tensor_allocator(tensor_allocator);
                lfs::core::Tensor::trim_memory_pool();

                LOG_INFO("Migrated training SplatData tensors to Vulkan-external storage "
                         "(gaussians={}, capacity={}, shN_capacity_floats={})",
                         n,
                         target_capacity,
                         target_shN_capacity);
            } catch (const std::exception& e) {
                return std::unexpected(std::format(
                    "Failed to migrate training SplatData to Vulkan-external storage: {}",
                    e.what()));
            }

            return {};
        }
    } // namespace

    std::expected<std::vector<std::shared_ptr<lfs::core::Camera>>, std::string>
    expandEquirectangularCamerasForUndistort(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
        return expandEquirectangularCamerasForUndistortImpl(cameras);
    }

    std::expected<void, std::string> migrateTrainingModelToAllocator(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::SplatData& model,
        const lfs::core::SplatTensorAllocator& tensor_allocator,
        const bool force_reallocation) {
        return migrateTrainingModelToAllocatorImpl(params, model, tensor_allocator, force_reallocation);
    }

    std::expected<void, std::string> loadTrainingDataIntoScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene) {

        auto data_loader = lfs::io::Loader::create();

        const auto& data_path = params.dataset.data_path;
        lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .min_track_length = effectiveMinTrackLengthForLoad(params),
            .validate_only = false,
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
        auto load_result = data_loader->load(params.dataset.data_path, load_options);
        if (!load_result) {
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error().format()));
        }

        LOG_INFO("Dataset loaded successfully using {} loader", load_result->loader_used);

        return std::visit([&](auto&& data) -> std::expected<void, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                auto model = std::make_unique<lfs::core::SplatData>(std::move(*data));
                applyTrainingSHDegree(*model, params.optimization.sh_degree);
                scene.addSplat("loaded_model", std::move(model));
                scene.setTrainingModelNode("loaded_model");
                LOG_INFO("Loaded PLY directly into scene");
                return {};

            } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                scene.setInitialPointCloud(data.point_cloud);
                scene.setSceneCenter(load_result->scene_center);
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

                if (params.init_path.has_value()) {
                    const std::filesystem::path init_file = lfs::core::utf8_to_path(params.init_path.value());
                    const auto ext = init_file.extension().string();

                    if (ext == ".ply" && !lfs::io::is_gaussian_splat_ply(init_file)) {
                        auto pc_result = lfs::io::load_ply_point_cloud(init_file);
                        if (!pc_result) {
                            return std::unexpected(std::format("Failed to load '{}': {}",
                                                               lfs::core::path_to_utf8(init_file), pc_result.error()));
                        }

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
                        auto loader = lfs::io::Loader::create();
                        auto init_result = loader->load(init_file);

                        if (!init_result) {
                            return std::unexpected(std::format("Failed to load '{}': {}",
                                                               lfs::core::path_to_utf8(init_file), init_result.error().format()));
                        }

                        try {
                            auto splat_data = std::move(*std::get<std::shared_ptr<lfs::core::SplatData>>(init_result->data));
                            auto model = std::make_unique<lfs::core::SplatData>(std::move(splat_data));

                            recomputeInitSplatSceneScale(*model, load_result->scene_center, init_file);
                            applyTrainingSHDegree(*model, params.optimization.sh_degree);

                            LOG_INFO("Loaded {} Gaussians from {} (sh={})",
                                     model->size(), lfs::core::path_to_utf8(init_file.filename()), model->get_max_sh_degree());
                            scene.addSplat("Model", std::move(model), dataset_id);
                            scene.setTrainingModelNode("Model");
                        } catch (const std::bad_variant_access&) {
                            return std::unexpected(std::format("'{}': invalid SplatData", lfs::core::path_to_utf8(init_file)));
                        }
                    }
                } else {
                    if (data.point_cloud && data.point_cloud->size() > 0) {
                        LOG_INFO("Adding {} points to scene", data.point_cloud->size());
                        scene.addPointCloud("PointCloud", data.point_cloud, dataset_id);
                    } else {
                        LOG_INFO("No point cloud, using random initialization");
                        auto pc = createRandomPointCloud();
                        LOG_INFO("Adding {} random points to scene", pc->size());
                        scene.addPointCloud("PointCloud", pc, dataset_id);
                    }
                }

                const bool enable_eval = params.optimization.enable_eval;
                const int test_every = params.dataset.test_every;

                for (size_t i = 0; i < data.cameras.size(); ++i) {
                    const bool is_eval = enable_eval && (i % test_every) == 0;
                    data.cameras[i]->set_split(is_eval ? lfs::core::CameraSplit::Eval : lfs::core::CameraSplit::Train);
                }

                std::vector<std::shared_ptr<lfs::core::Camera>> undistorted_cameras;
                const auto* cameras_ptr = &data.cameras;
                if (params.optimization.undistort) {
                    auto expanded_cameras = expandEquirectangularCamerasForUndistort(data.cameras);
                    if (!expanded_cameras)
                        return std::unexpected(expanded_cameras.error());
                    undistorted_cameras = std::move(*expanded_cameras);
                    cameras_ptr = &undistorted_cameras;
                }
                const auto& cameras = *cameras_ptr;

                size_t train_count = 0;
                size_t val_count = 0;
                size_t mask_count = 0;
                for (const auto& camera : cameras) {
                    const bool is_eval = enable_eval && camera->split() == lfs::core::CameraSplit::Eval;
                    if (is_eval) {
                        val_count++;
                    } else {
                        train_count++;
                    }
                    if (camera->has_mask()) {
                        mask_count++;
                    }
                }

                const auto cameras_group_id = scene.addGroup("Cameras", dataset_id);

                const auto train_cameras_id = scene.addCameraGroup(
                    "Training",
                    cameras_group_id,
                    train_count);

                for (size_t i = 0; i < cameras.size(); ++i) {
                    if (!enable_eval || cameras[i]->split() != lfs::core::CameraSplit::Eval) {
                        scene.addCamera(cameras[i]->image_name(), train_cameras_id, cameras[i]);
                    }
                }

                if (enable_eval && val_count > 0) {
                    const auto val_cameras_id = scene.addCameraGroup(
                        "Validation",
                        cameras_group_id,
                        val_count);

                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if (cameras[i]->split() == lfs::core::CameraSplit::Eval) {
                            scene.addCamera(cameras[i]->image_name(), val_cameras_id, cameras[i]);
                        }
                    }
                }

                LOG_INFO("Loaded dataset '{}' into scene: {} train{} cameras{}",
                         dataset_name, train_count,
                         enable_eval ? std::format(" + {} val", val_count) : "",
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

    std::expected<void, std::string> initializeTrainingModel(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::core::SplatTensorAllocator tensor_allocator) {

        if (auto* model = scene.getTrainingModel()) {
            applyTrainingSHDegree(*model, params.optimization.sh_degree);
            if (auto result = appendAddedSplats(params, *model); !result) {
                return result;
            }

            const int max_cap = params.optimization.max_cap;
            if (max_cap > 0 && model->size() > max_cap) {
                LOG_WARN("Max cap ({}) is less than initial splat count ({}), randomly selecting {} splats",
                         max_cap, model->size(), max_cap);
                lfs::core::random_choose(*model, max_cap);
            }

            if (auto result = migrateTrainingModelToAllocator(params, *model, tensor_allocator); !result) {
                return result;
            }
            scene.syncTrainingModelTopology(static_cast<size_t>(model->size()));
            scene.notifyMutation(lfs::core::Scene::MutationType::MODEL_CHANGED);
            return {};
        }

        lfs::core::NodeId point_cloud_node_id = lfs::core::NULL_NODE;
        lfs::core::NodeId parent_id = lfs::core::NULL_NODE;
        const lfs::core::PointCloud* point_cloud = nullptr;
        glm::mat4 node_transform{1.0f};
        lfs::core::CropBoxData preserved_cropbox_data;
        glm::mat4 preserved_cropbox_transform{1.0f};
        bool has_preserved_cropbox = false;

        for (const auto* node : scene.getNodes()) {
            if (node->type == lfs::core::NodeType::POINTCLOUD && node->point_cloud) {
                point_cloud_node_id = node->id;
                parent_id = node->parent_id;
                node_transform = node->transform();
                point_cloud = node->point_cloud.get();
                break;
            }
        }

        lfs::core::PointCloud point_cloud_to_use;
        const int max_cap = params.optimization.max_cap;

        if (point_cloud && point_cloud->size() > 0) {
            const lfs::core::CropBoxData* cropbox_data = nullptr;
            lfs::core::NodeId cropbox_id = lfs::core::NULL_NODE;

            if (point_cloud_node_id != lfs::core::NULL_NODE) {
                cropbox_id = scene.getCropBoxForSplat(point_cloud_node_id);
                if (cropbox_id != lfs::core::NULL_NODE) {
                    cropbox_data = scene.getCropBoxData(cropbox_id);
                    if (const auto* cropbox_node = scene.getNodeById(cropbox_id);
                        cropbox_node && cropbox_node->cropbox) {
                        preserved_cropbox_data = *cropbox_node->cropbox;
                        preserved_cropbox_transform =
                            cropbox_node->parent_id == point_cloud_node_id
                                ? cropbox_node->transform()
                                : glm::inverse(scene.getWorldTransform(point_cloud_node_id)) *
                                      scene.getWorldTransform(cropbox_id);
                        has_preserved_cropbox = true;
                    }
                }
            }

            if (cropbox_data && cropbox_data->enabled) {
                const glm::mat4 pointcloud_to_cropbox =
                    has_preserved_cropbox ? glm::inverse(preserved_cropbox_transform)
                                          : glm::inverse(scene.getWorldTransform(cropbox_id));
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
                    const glm::vec4 local_pos = pointcloud_to_cropbox * glm::vec4(pos, 1.0f);
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
                point_cloud_to_use = *point_cloud;
                if (max_cap > 0) {
                    point_cloud_to_use.means = point_cloud_to_use.means.cpu();
                    point_cloud_to_use.colors = point_cloud_to_use.colors.cpu();
                }
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

        lfs::core::Tensor scene_center = scene.getSceneCenter();
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

        if (point_cloud_node_id != lfs::core::NULL_NODE) {
            if (const auto* pc_node = scene.getNodeById(point_cloud_node_id)) {
                scene.removeNode(pc_node->name, false);
            }
        }

        auto model = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
        applyTrainingSHDegree(*model, params.optimization.sh_degree);
        if (auto result = appendAddedSplats(params, *model); !result) {
            return result;
        }
        if (auto result = migrateTrainingModelToAllocator(params, *model, tensor_allocator); !result) {
            return result;
        }
        LOG_INFO("Created training model with {} gaussians", model->size());
        const lfs::core::NodeId model_id = scene.addSplat("Model", std::move(model), parent_id);
        if (node_transform != glm::mat4{1.0f}) {
            scene.setNodeTransform("Model", node_transform);
        }
        scene.setTrainingModelNode("Model");
        if (has_preserved_cropbox && model_id != lfs::core::NULL_NODE) {
            const lfs::core::NodeId model_cropbox_id = scene.addCropBox("Model_cropbox", model_id);
            if (model_cropbox_id != lfs::core::NULL_NODE) {
                scene.setCropBoxData(model_cropbox_id, preserved_cropbox_data);
                scene.setNodeTransform("Model_cropbox", preserved_cropbox_transform);
            }
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
            .validate_only = true};

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
                scene.addSplat("loaded_model", std::move(model));
                scene.setTrainingModelNode("loaded_model");
                return {};

            } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                scene.setInitialPointCloud(data.point_cloud);
                scene.setSceneCenter(load_result.scene_center);
                scene.setImagesHaveAlpha(load_result.images_have_alpha);

                std::string dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.filename());
                if (dataset_name.empty()) {
                    dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.parent_path().filename());
                }
                if (dataset_name.empty()) {
                    dataset_name = "Dataset";
                }

                const auto dataset_id = scene.addDataset(dataset_name);

                if (params.init_path.has_value()) {
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

                            recomputeInitSplatSceneScale(*model, load_result.scene_center, init_file);
                            applyTrainingSHDegree(*model, params.optimization.sh_degree);

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
                } else {
                    scene.addPointCloud("PointCloud", createRandomPointCloud(), dataset_id);
                }

                const bool enable_eval = params.optimization.enable_eval;
                const int test_every = params.dataset.test_every;

                for (size_t i = 0; i < data.cameras.size(); ++i) {
                    const bool is_val = enable_eval && (i % test_every) == 0;
                    data.cameras[i]->set_split(is_val ? lfs::core::CameraSplit::Eval : lfs::core::CameraSplit::Train);
                }

                std::vector<std::shared_ptr<lfs::core::Camera>> undistorted_cameras;
                const auto* cameras_ptr = &data.cameras;
                if (params.optimization.undistort) {
                    auto expanded_cameras = expandEquirectangularCamerasForUndistort(data.cameras);
                    if (!expanded_cameras)
                        return std::unexpected(expanded_cameras.error());
                    undistorted_cameras = std::move(*expanded_cameras);
                    cameras_ptr = &undistorted_cameras;
                }
                const auto& cameras = *cameras_ptr;

                size_t train_count = 0, val_count = 0, mask_count = 0;
                for (const auto& camera : cameras) {
                    const bool is_val = enable_eval && camera->split() == lfs::core::CameraSplit::Eval;
                    is_val ? ++val_count : ++train_count;
                    if (camera->has_mask())
                        ++mask_count;
                }

                const auto cameras_group_id = scene.addGroup("Cameras", dataset_id);
                const auto train_cameras_id = scene.addCameraGroup(
                    "Training", cameras_group_id, train_count);

                for (size_t i = 0; i < cameras.size(); ++i) {
                    if (!enable_eval || cameras[i]->split() != lfs::core::CameraSplit::Eval) {
                        scene.addCamera(cameras[i]->image_name(), train_cameras_id, cameras[i]);
                    }
                }

                if (enable_eval && val_count > 0) {
                    const auto val_cameras_id = scene.addCameraGroup(
                        "Validation", cameras_group_id, val_count);
                    for (size_t i = 0; i < cameras.size(); ++i) {
                        if (cameras[i]->split() == lfs::core::CameraSplit::Eval) {
                            scene.addCamera(cameras[i]->image_name(), val_cameras_id, cameras[i]);
                        }
                    }
                }

                LOG_INFO("Dataset '{}': {} train{} cameras{}",
                         dataset_name, train_count,
                         enable_eval ? std::format(" + {} val", val_count) : "",
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

} // namespace lfs::training
