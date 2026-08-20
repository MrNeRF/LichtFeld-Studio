/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "environment_image.hpp"
#include "environment_math.hpp"
#include "image_layout.hpp"
#include "point_cloud_raster.cuh"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering.hpp"
#include "screen_overlay_renderer.hpp"
#include <OpenImageIO/imageio.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <format>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <limits>
#include <mutex>
#include <vector>

namespace lfs::rendering {

    namespace {
        struct RasterImageResult {
            Tensor image;
            Tensor depth;
            bool valid = false;
            bool flip_y = false;
            float far_plane = DEFAULT_FAR_PLANE;
            bool orthographic = false;
        };

        struct EnvironmentImageCache {
            std::mutex mutex;
            std::shared_ptr<const EnvironmentImage> image;
        };

        [[nodiscard]] EnvironmentImageCache& environmentImageCache() {
            static EnvironmentImageCache cache;
            return cache;
        }

        [[nodiscard]] std::filesystem::path resolveEnvironmentPath(const std::filesystem::path& requested) {
            if (requested.empty() || requested.is_absolute()) {
                return requested;
            }
            if (std::filesystem::exists(requested)) {
                return requested;
            }

            const std::array candidates{
                lfs::core::getAssetsDir() / requested,
                lfs::core::getExecutableDir() / requested,
                lfs::core::getExecutableDir() / "assets" / requested,
            };
            for (const auto& candidate : candidates) {
                if (std::filesystem::exists(candidate)) {
                    return candidate;
                }
            }
            return lfs::core::getAssetsDir() / requested;
        }

    } // namespace

    std::expected<std::shared_ptr<const EnvironmentImage>, std::string>
    loadEnvironmentImageShared(const std::filesystem::path& environment_path) {
        const auto resolved_path = resolveEnvironmentPath(environment_path);
        auto& cache = environmentImageCache();
        std::lock_guard lock(cache.mutex);
        if (cache.image && cache.image->valid() && cache.image->path == resolved_path) {
            return cache.image;
        }

        cache.image.reset();
        if (resolved_path.empty()) {
            return std::unexpected("Environment map path is empty");
        }
        if (!std::filesystem::exists(resolved_path)) {
            return std::unexpected(std::format("Environment map not found: {}", resolved_path.string()));
        }

        const std::string path_utf8 = lfs::core::path_to_utf8(resolved_path);
        std::unique_ptr<OIIO::ImageInput> input(OIIO::ImageInput::open(path_utf8));
        if (!input) {
            return std::unexpected(std::format("Failed to open environment map {}: {}",
                                               path_utf8,
                                               OIIO::geterror()));
        }

        const auto& spec = input->spec();
        if (spec.width <= 0 || spec.height <= 0 || spec.nchannels <= 0) {
            input->close();
            return std::unexpected(std::format("Invalid environment map dimensions for {}", path_utf8));
        }

        const int read_channels = spec.nchannels >= 3 ? 3 : 1;
        std::vector<float> source_pixels(
            static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) *
            static_cast<size_t>(read_channels));
        if (!input->read_image(0, 0, 0, read_channels, OIIO::TypeDesc::FLOAT, source_pixels.data())) {
            const std::string error =
                std::format("Failed to read environment map {}: {}", path_utf8, input->geterror());
            input->close();
            return std::unexpected(error);
        }
        input->close();

        auto image = std::make_shared<EnvironmentImage>();
        image->path = resolved_path;
        image->width = spec.width;
        image->height = spec.height;
        if (read_channels == 3) {
            image->pixels = std::move(source_pixels);
        } else {
            const size_t pixel_count =
                static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height);
            image->pixels.resize(pixel_count * 3u);
            for (size_t pixel = 0; pixel < pixel_count; ++pixel) {
                const float value = source_pixels[pixel];
                image->pixels[pixel * 3u + 0u] = value;
                image->pixels[pixel * 3u + 1u] = value;
                image->pixels[pixel * 3u + 2u] = value;
            }
        }

        cache.image = image;
        LOG_INFO("Loaded tensor environment map {}", resolved_path.string());
        return image;
    }

    std::expected<EnvironmentImage, std::string> loadEnvironmentImage(
        const std::filesystem::path& environment_path) {
        auto image = loadEnvironmentImageShared(environment_path);
        if (!image) {
            return std::unexpected(image.error());
        }
        return **image;
    }

    void releaseEnvironmentImageCache() {
        auto& cache = environmentImageCache();
        std::lock_guard lock(cache.mutex);
        cache.image.reset();
    }

    namespace {

        [[nodiscard]] glm::vec3 sampleEnvironmentBilinear(const EnvironmentImage& image,
                                                          const float u,
                                                          const float v) {
            if (!image.valid()) {
                return glm::vec3(0.0f);
            }
            const auto fetch = [&](const int px, const int py) -> envmath::Vec3 {
                const size_t index =
                    (static_cast<size_t>(py) * static_cast<size_t>(image.width) + static_cast<size_t>(px)) * 3u;
                return {image.pixels[index + 0], image.pixels[index + 1], image.pixels[index + 2]};
            };
            const auto color = envmath::sampleEnvironmentBilinear(fetch, u, v, image.width, image.height);
            return {color.x, color.y, color.z};
        }

        [[nodiscard]] glm::vec3 environmentDirectionForPixel(
            const FrameView& frame_view,
            const int x,
            const int y,
            const bool equirectangular_view) {
            const float width = static_cast<float>(std::max(frame_view.size.x, 1));
            const float height = static_cast<float>(std::max(frame_view.size.y, 1));

            float focal_x = 0.0f;
            float focal_y = 0.0f;
            float center_x = width * 0.5f;
            float center_y = height * 0.5f;
            if (frame_view.intrinsics_override.has_value() && !frame_view.orthographic) {
                const auto& intrinsics = *frame_view.intrinsics_override;
                focal_x = intrinsics.focal_x;
                focal_y = intrinsics.focal_y;
                center_x = intrinsics.center_x;
                center_y = intrinsics.center_y;
            } else {
                const auto focal = computePixelFocalLengths(frame_view.size, frame_view.focal_length_mm);
                focal_x = focal.first;
                focal_y = focal.second;
            }

            const auto dir = envmath::environmentWorldDirection(
                static_cast<float>(x),
                static_cast<float>(y),
                width,
                height,
                equirectangular_view,
                focal_x,
                focal_y,
                center_x,
                center_y,
                &frame_view.rotation[0][0]);
            return {dir.x, dir.y, dir.z};
        }

        Result<std::vector<float>> renderEnvironmentBackground(
            const VideoCompositeFrameRequest& request,
            const int width,
            const int height) {
            const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
            std::vector<float> image(3 * pixel_count, 0.0f);

            if (!request.environment.enabled) {
                for (size_t i = 0; i < pixel_count; ++i) {
                    image[i] = request.background_color.r;
                    image[pixel_count + i] = request.background_color.g;
                    image[2 * pixel_count + i] = request.background_color.b;
                }
                return image;
            }

            auto environment = loadEnvironmentImageShared(request.environment.map_path);
            if (!environment) {
                return std::unexpected(environment.error());
            }

            const float exposure = std::exp2(request.environment.exposure);
            const float rotation = glm::radians(request.environment.rotation_degrees);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    glm::vec3 world_dir = environmentDirectionForPixel(
                        request.frame_view, x, y, request.environment.equirectangular);
                    const auto rotated = envmath::rotateAroundY({world_dir.x, world_dir.y, world_dir.z}, rotation);
                    const auto uv = envmath::equirectUvForDirection(envmath::normalized(rotated));

                    const glm::vec3 hdr = sampleEnvironmentBilinear(**environment, uv.u, uv.v);
                    const auto color = envmath::shadeEnvironmentRadiance({hdr.x, hdr.y, hdr.z}, exposure);

                    const size_t pixel = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                    image[pixel] = color.x;
                    image[pixel_count + pixel] = color.y;
                    image[2 * pixel_count + pixel] = color.z;
                }
            }
            return image;
        }

        [[nodiscard]] FrameMetadata makePointCloudFrameMetadata(
            const RasterImageResult& result) {
            return FrameMetadata{
                .depth_panels = {FramePanelMetadata{
                    .depth = result.depth.is_valid() ? std::make_shared<Tensor>(result.depth) : nullptr,
                    .start_position = 0.0f,
                    .end_position = 1.0f,
                }},
                .depth_panel_count = 1,
                .valid = result.valid,
                .flip_y = result.flip_y,
                .far_plane = result.far_plane,
                .orthographic = result.orthographic};
        }

        [[nodiscard]] std::optional<glm::mat4> cameraVisualizerTransform(
            const lfs::core::Camera& camera,
            const glm::mat4& scene_transform) {
            auto rotation_tensor = camera.R();
            auto translation_tensor = camera.T();
            if (!rotation_tensor.is_valid() || !translation_tensor.is_valid()) {
                return std::nullopt;
            }
            if (rotation_tensor.device() != lfs::core::Device::CPU) {
                rotation_tensor = rotation_tensor.cpu();
            }
            if (translation_tensor.device() != lfs::core::Device::CPU) {
                translation_tensor = translation_tensor.cpu();
            }
            if (rotation_tensor.dtype() != lfs::core::DataType::Float32 ||
                translation_tensor.dtype() != lfs::core::DataType::Float32 ||
                rotation_tensor.numel() < 9 || translation_tensor.numel() < 3) {
                return std::nullopt;
            }

            glm::mat4 world_to_camera(1.0f);
            const float* const rotation = rotation_tensor.ptr<float>();
            const float* const translation = translation_tensor.ptr<float>();
            if (!rotation || !translation) {
                return std::nullopt;
            }
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    world_to_camera[col][row] = rotation[row * 3 + col];
                }
                world_to_camera[3][row] = translation[row];
            }

            return scene_transform * glm::inverse(world_to_camera) * DATA_TO_VISUALIZER_CAMERA_AXES_4;
        }

        [[nodiscard]] std::vector<glm::vec3> cameraFrustumWorldPoints(
            const lfs::core::Camera& camera,
            const glm::mat4& visualizer_camera_to_world,
            const float scale) {
            std::vector<glm::vec3> points;
            // Picking must follow the calibrated frustum, not the resolution
            // selected for loading training images. Undistortion updates these
            // calibration dimensions and FoVy together; training downscaling
            // only changes the operational decode dimensions.
            const int calibration_width = camera.camera_width();
            const int calibration_height = camera.camera_height();
            if (calibration_width <= 0 || calibration_height <= 0 || scale <= 0.0f) {
                return points;
            }

            const bool equirectangular =
                camera.camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR;
            if (equirectangular) {
                constexpr int SEGMENTS = 48;
                points.reserve(SEGMENTS * 3);
                for (int circle = 0; circle < 3; ++circle) {
                    for (int i = 0; i < SEGMENTS; ++i) {
                        const float a = static_cast<float>(i) / static_cast<float>(SEGMENTS) *
                                        2.0f * glm::pi<float>();
                        glm::vec3 local(0.0f);
                        if (circle == 0) {
                            local = {std::cos(a), std::sin(a), 0.0f};
                        } else if (circle == 1) {
                            local = {std::cos(a), 0.0f, std::sin(a)};
                        } else {
                            local = {0.0f, std::cos(a), std::sin(a)};
                        }
                        points.push_back(glm::vec3(
                            visualizer_camera_to_world * glm::vec4(local * scale, 1.0f)));
                    }
                }
                return points;
            }

            if (camera.FoVy() <= 0.0f) {
                return points;
            }

            const float aspect = static_cast<float>(calibration_width) /
                                 static_cast<float>(calibration_height);
            const float fov_y = camera.FoVy();
            const float half_height = std::tan(fov_y * 0.5f) * scale;
            const float half_width = half_height * aspect;

            const std::array local_points{
                glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3(-half_width, half_height, -scale),
                glm::vec3(half_width, half_height, -scale),
                glm::vec3(half_width, -half_height, -scale),
                glm::vec3(-half_width, -half_height, -scale),
            };
            points.reserve(local_points.size());
            for (const glm::vec3& local : local_points) {
                points.push_back(glm::vec3(visualizer_camera_to_world * glm::vec4(local, 1.0f)));
            }
            return points;
        }

        [[nodiscard]] float pointSegmentDistance(
            const glm::vec2& point,
            const glm::vec2& a,
            const glm::vec2& b) {
            const glm::vec2 ab = b - a;
            const float denom = glm::dot(ab, ab);
            if (denom <= 1e-6f) {
                return glm::length(point - a);
            }
            const float t = std::clamp(glm::dot(point - a, ab) / denom, 0.0f, 1.0f);
            return glm::length(point - (a + ab * t));
        }

        [[nodiscard]] std::optional<glm::vec2> projectFrustumPoint(
            const glm::vec3& world_point,
            const CameraFrustumPickRequest& request) {
            const auto projected = projectWorldPoint(
                request.viewport.rotation,
                request.viewport.translation,
                request.viewport.size,
                world_point,
                request.viewport.focal_length_mm,
                request.viewport.orthographic,
                request.viewport.ortho_scale);
            if (!projected) {
                return std::nullopt;
            }

            const float scale_x = request.viewport_size.x /
                                  static_cast<float>(std::max(request.viewport.size.x, 1));
            const float scale_y = request.viewport_size.y /
                                  static_cast<float>(std::max(request.viewport.size.y, 1));
            return glm::vec2(
                request.viewport_pos.x + projected->x * scale_x,
                request.viewport_pos.y + projected->y * scale_y);
        }

        Result<RasterImageResult> renderSoftwarePointCloud(
            const Tensor& positions_source,
            const Tensor& colors_source,
            const PointCloudRenderRequest& request,
            const Tensor* const deleted_mask_source) {
            if (request.frame_view.size.x <= 0 || request.frame_view.size.y <= 0) {
                return std::unexpected("Invalid viewport dimensions");
            }
            const auto width_pixels = static_cast<std::size_t>(request.frame_view.size.x);
            const auto height_pixels = static_cast<std::size_t>(request.frame_view.size.y);
            if (width_pixels > std::numeric_limits<std::size_t>::max() / height_pixels) {
                return std::unexpected("Viewport dimensions overflow pixel count");
            }
            if (!positions_source.is_valid() || positions_source.ndim() != 2 || positions_source.size(1) != 3) {
                return std::unexpected("Point cloud positions must have shape [N, 3]");
            }
            if (!colors_source.is_valid() || colors_source.ndim() != 2 || colors_source.size(1) != 3 ||
                colors_source.size(0) != positions_source.size(0)) {
                return std::unexpected("Point cloud colors must have shape [N, 3]");
            }
            if (deleted_mask_source && deleted_mask_source->is_valid() &&
                deleted_mask_source->numel() != positions_source.size(0)) {
                return std::unexpected("Point cloud deleted mask must match point count");
            }

            Tensor positions_cuda = positions_source;
            if (positions_cuda.device() != lfs::core::Device::CUDA) {
                positions_cuda = positions_cuda.cuda();
            }
            positions_cuda = positions_cuda.contiguous();

            Tensor colors_cuda = colors_source;
            if (colors_cuda.dtype() == lfs::core::DataType::UInt8) {
                colors_cuda = colors_cuda.to(lfs::core::DataType::Float32) / 255.0f;
            }
            if (colors_cuda.dtype() != lfs::core::DataType::Float32) {
                colors_cuda = colors_cuda.to(lfs::core::DataType::Float32);
            }
            if (colors_cuda.device() != lfs::core::Device::CUDA) {
                colors_cuda = colors_cuda.cuda();
            }
            colors_cuda = colors_cuda.contiguous();

            Tensor transform_indices_cuda;
            const std::int32_t* transform_indices_ptr = nullptr;
            if (request.scene.transform_indices && request.scene.transform_indices->is_valid() &&
                request.scene.transform_indices->numel() == positions_source.size(0)) {
                transform_indices_cuda = *request.scene.transform_indices;
                if (transform_indices_cuda.dtype() != lfs::core::DataType::Int32) {
                    transform_indices_cuda = transform_indices_cuda.to(lfs::core::DataType::Int32);
                }
                if (transform_indices_cuda.device() != lfs::core::Device::CUDA) {
                    transform_indices_cuda = transform_indices_cuda.cuda();
                }
                transform_indices_cuda = transform_indices_cuda.contiguous();
                transform_indices_ptr = transform_indices_cuda.ptr<std::int32_t>();
            }

            const std::vector<glm::mat4>* const transforms_ptr = request.scene.model_transforms;
            const std::vector<glm::mat4> empty_transforms;
            const auto& transforms = transforms_ptr ? *transforms_ptr : empty_transforms;

            Tensor transforms_cuda;
            const float* transforms_device = nullptr;
            if (!transforms.empty()) {
                std::vector<float> transforms_host(transforms.size() * 16);
                for (size_t i = 0; i < transforms.size(); ++i) {
                    const float* m = glm::value_ptr(transforms[i]);
                    std::copy(m, m + 16, transforms_host.begin() + static_cast<std::ptrdiff_t>(i * 16));
                }
                transforms_cuda = Tensor::from_vector(
                                      transforms_host,
                                      {transforms.size(), static_cast<size_t>(16)},
                                      lfs::core::Device::CPU)
                                      .cuda()
                                      .contiguous();
                transforms_device = transforms_cuda.ptr<float>();
            }

            Tensor visibility_cuda;
            const std::uint8_t* visibility_device = nullptr;
            if (!request.scene.node_visibility_mask.empty()) {
                std::vector<int> mask_host(request.scene.node_visibility_mask.size());
                for (size_t i = 0; i < mask_host.size(); ++i) {
                    mask_host[i] = request.scene.node_visibility_mask[i] ? 1 : 0;
                }
                visibility_cuda = Tensor::from_vector(
                                      mask_host,
                                      {mask_host.size()},
                                      lfs::core::Device::CPU)
                                      .cuda()
                                      .to(lfs::core::DataType::UInt8)
                                      .contiguous();
                visibility_device = visibility_cuda.ptr<std::uint8_t>();
            }

            Tensor deleted_mask_cuda;
            const bool* deleted_mask_device = nullptr;
            if (deleted_mask_source && deleted_mask_source->is_valid()) {
                deleted_mask_cuda = *deleted_mask_source;
                if (deleted_mask_cuda.dtype() != lfs::core::DataType::Bool) {
                    deleted_mask_cuda = deleted_mask_cuda.to(lfs::core::DataType::Bool);
                }
                if (deleted_mask_cuda.device() != lfs::core::Device::CUDA) {
                    deleted_mask_cuda = deleted_mask_cuda.cuda();
                }
                deleted_mask_cuda = deleted_mask_cuda.contiguous();
                deleted_mask_device = deleted_mask_cuda.ptr<bool>();
            }

            const glm::mat4 view = request.frame_view.getViewMatrix();
            const glm::mat4 projection = createProjectionMatrix(
                request.frame_view.size,
                focalLengthToVFov(request.frame_view.focal_length_mm),
                request.frame_view.orthographic,
                request.frame_view.ortho_scale,
                request.frame_view.near_plane,
                request.frame_view.far_plane);
            const glm::mat4 view_proj = projection * view;

            const int width = request.frame_view.size.x;
            const int height = request.frame_view.size.y;
            const int channels = request.transparent_background ? 4 : 3;

            Tensor image_tensor = Tensor::empty(
                {static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            Tensor depth_tensor = Tensor::empty(
                {static_cast<size_t>(1), static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA, lfs::core::DataType::Float32);

            lfs::core::pin_operands({&positions_cuda, &colors_cuda});
            pcraster::LaunchParams params{};
            params.positions = positions_cuda.ptr<float>();
            params.colors = colors_cuda.ptr<float>();
            params.transforms = transforms_device;
            params.transform_indices = transform_indices_ptr;
            params.visibility_mask = visibility_device;
            params.deleted_mask = deleted_mask_device;
            params.n_points = static_cast<std::size_t>(positions_source.size(0));
            params.n_transforms = static_cast<int>(transforms.size());
            params.n_visibility = static_cast<int>(request.scene.node_visibility_mask.size());
            params.has_crop = request.filters.crop_box.has_value();
            if (params.has_crop) {
                const auto& crop = *request.filters.crop_box;
                std::copy_n(glm::value_ptr(crop.transform), 16, params.crop.to_local);
                params.crop.min[0] = crop.min.x;
                params.crop.min[1] = crop.min.y;
                params.crop.min[2] = crop.min.z;
                params.crop.max[0] = crop.max.x;
                params.crop.max[1] = crop.max.y;
                params.crop.max[2] = crop.max.z;
                params.crop.inverse = request.filters.crop_inverse;
                params.crop.desaturate = request.filters.crop_desaturate;
            }
            params.has_crop_ellipsoid = request.filters.crop_ellipsoid.has_value();
            if (params.has_crop_ellipsoid) {
                const auto& ellipsoid = *request.filters.crop_ellipsoid;
                std::copy_n(glm::value_ptr(ellipsoid.transform), 16, params.crop_ellipsoid.to_local);
                params.crop_ellipsoid.radii[0] = ellipsoid.radii.x;
                params.crop_ellipsoid.radii[1] = ellipsoid.radii.y;
                params.crop_ellipsoid.radii[2] = ellipsoid.radii.z;
                params.crop_ellipsoid.inverse = request.filters.crop_inverse;
                params.crop_ellipsoid.desaturate = request.filters.crop_desaturate;
            }
            std::copy_n(glm::value_ptr(view), 16, params.view);
            std::copy_n(glm::value_ptr(view_proj), 16, params.view_proj);
            params.width = width;
            params.height = height;
            params.channels = channels;
            params.equirectangular = request.render.equirectangular;
            params.orthographic = request.frame_view.orthographic;
            params.ortho_scale = request.frame_view.ortho_scale;
            params.focal_y = lfs::core::fov2focal(
                focalLengthToVFovRad(request.frame_view.focal_length_mm),
                request.frame_view.size.y);
            params.voxel_size = request.render.voxel_size;
            params.scaling_modifier = request.render.scaling_modifier;
            params.far_plane = request.frame_view.far_plane;
            params.bg_r = request.frame_view.background_color.r;
            params.bg_g = request.frame_view.background_color.g;
            params.bg_b = request.frame_view.background_color.b;
            params.bg_a = 1.0f;
            params.transparent_background = request.transparent_background;
            params.image = image_tensor.ptr<float>();
            params.depth = depth_tensor.ptr<float>();
            params.stream = image_tensor.stream();

            if (const cudaError_t status = pcraster::launchPointCloudRaster(params);
                status != cudaSuccess) {
                return std::unexpected(std::format("Point cloud rasterization failed: {}",
                                                   cudaGetErrorString(status)));
            }

            return RasterImageResult{
                .image = std::move(image_tensor),
                .depth = std::move(depth_tensor),
                .valid = true,
                .far_plane = request.frame_view.far_plane,
                .orthographic = request.frame_view.orthographic};
        }

        [[nodiscard]] Result<Tensor> toCpuChwFloatTensor(const Tensor& image) {
            if (!image.is_valid() || image.ndim() != 3) {
                return std::unexpected("Invalid image tensor");
            }
            const auto layout = detectImageLayout(image);
            if (layout == ImageLayout::Unknown) {
                return std::unexpected("Unsupported image tensor layout");
            }
            Tensor formatted = image;
            if (formatted.dtype() == lfs::core::DataType::UInt8) {
                formatted = formatted.to(lfs::core::DataType::Float32) / 255.0f;
            } else if (formatted.dtype() != lfs::core::DataType::Float32) {
                formatted = formatted.to(lfs::core::DataType::Float32);
            }
            if (layout == ImageLayout::HWC) {
                formatted = formatted.permute({2, 0, 1}).contiguous();
            }
            return formatted.cpu().contiguous();
        }

        Result<Tensor> composeVideoFrame(
            const std::shared_ptr<lfs::core::Tensor>& primary_image,
            const FrameMetadata* primary_metadata,
            const VideoCompositeFrameRequest& request) {
            const int width = request.frame_view.size.x > 0 ? request.frame_view.size.x : request.viewport.size.x;
            const int height = request.frame_view.size.y > 0 ? request.frame_view.size.y : request.viewport.size.y;
            if (width <= 0 || height <= 0) {
                return std::unexpected("Invalid video composite dimensions");
            }

            const size_t pixel_count = static_cast<size_t>(width) * height;
            auto background = renderEnvironmentBackground(request, width, height);
            if (!background) {
                return std::unexpected(background.error());
            }
            std::vector<float> image = std::move(*background);
            std::vector<float> depth(pixel_count, request.frame_view.far_plane);

            if (primary_image && primary_image->is_valid()) {
                auto cpu_image = toCpuChwFloatTensor(*primary_image);
                if (!cpu_image) {
                    return std::unexpected(cpu_image.error());
                }
                const auto& img = *cpu_image;
                const auto layout = detectImageLayout(img);
                const int src_w = imageWidth(img, layout);
                const int src_h = imageHeight(img, layout);
                const int channels = imageChannels(img, layout);
                const float* src = img.ptr<float>();
                for (int y = 0; y < height; ++y) {
                    const int sy = std::clamp(static_cast<int>(
                                                  static_cast<float>(y) * src_h / std::max(height, 1)),
                                              0, src_h - 1);
                    for (int x = 0; x < width; ++x) {
                        const int sx = std::clamp(static_cast<int>(
                                                      static_cast<float>(x) * src_w / std::max(width, 1)),
                                                  0, src_w - 1);
                        const size_t dst = static_cast<size_t>(y) * width + x;
                        const size_t src_pixel = static_cast<size_t>(sy) * src_w + sx;
                        const float src_r = src[src_pixel];
                        const float src_g = src[static_cast<size_t>(1) * src_h * src_w + src_pixel];
                        const float src_b = src[static_cast<size_t>(2) * src_h * src_w + src_pixel];
                        if (channels == 4) {
                            const float alpha = src[static_cast<size_t>(3) * src_h * src_w + src_pixel];
                            image[dst] = glm::mix(image[dst], src_r, alpha);
                            image[pixel_count + dst] = glm::mix(image[pixel_count + dst], src_g, alpha);
                            image[2 * pixel_count + dst] = glm::mix(image[2 * pixel_count + dst], src_b, alpha);
                        } else {
                            image[dst] = src_r;
                            image[pixel_count + dst] = src_g;
                            image[2 * pixel_count + dst] = src_b;
                        }
                    }
                }

                if (primary_metadata && primary_metadata->primaryDepth() &&
                    primary_metadata->primaryDepth()->is_valid()) {
                    Tensor depth_cpu = primary_metadata->primaryDepth()->cpu().contiguous();
                    if (depth_cpu.ndim() == 3 && depth_cpu.dtype() == lfs::core::DataType::Float32) {
                        const int depth_h = static_cast<int>(depth_cpu.size(1));
                        const int depth_w = static_cast<int>(depth_cpu.size(2));
                        const float* depth_src = depth_cpu.ptr<float>();
                        for (int y = 0; y < height; ++y) {
                            const int sy = std::clamp(static_cast<int>(
                                                          static_cast<float>(y) * depth_h / std::max(height, 1)),
                                                      0, depth_h - 1);
                            for (int x = 0; x < width; ++x) {
                                const int sx = std::clamp(static_cast<int>(
                                                              static_cast<float>(x) * depth_w / std::max(width, 1)),
                                                          0, depth_w - 1);
                                depth[static_cast<size_t>(y) * width + x] =
                                    depth_src[static_cast<size_t>(sy) * depth_w + sx];
                            }
                        }
                    }
                }
            }

            if (request.prerendered_meshes != nullptr) {
                const auto& mesh_layer = *request.prerendered_meshes;
                if (!mesh_layer.rgba.is_valid() || !mesh_layer.view_depth.is_valid()) {
                    return std::unexpected("Pre-rendered mesh layer is invalid");
                }
                Tensor mesh_rgba = mesh_layer.rgba.cpu().contiguous();
                Tensor mesh_depth = mesh_layer.view_depth.cpu().contiguous();
                if (mesh_rgba.dtype() != lfs::core::DataType::Float32 ||
                    mesh_rgba.ndim() != 3 || mesh_rgba.size(0) != 4u ||
                    mesh_rgba.size(1) != static_cast<size_t>(height) ||
                    mesh_rgba.size(2) != static_cast<size_t>(width) ||
                    mesh_depth.dtype() != lfs::core::DataType::Float32 ||
                    mesh_depth.ndim() != 2 ||
                    mesh_depth.size(0) != static_cast<size_t>(height) ||
                    mesh_depth.size(1) != static_cast<size_t>(width)) {
                    return std::unexpected("Pre-rendered mesh layer dimensions must match the composite frame");
                }

                const float* rgba = mesh_rgba.ptr<float>();
                const float* view_depth = mesh_depth.ptr<float>();
                for (size_t pixel = 0; pixel < pixel_count; ++pixel) {
                    if (rgba[3u * pixel_count + pixel] > 0.0f &&
                        view_depth[pixel] < depth[pixel]) {
                        image[pixel] = rgba[pixel];
                        image[pixel_count + pixel] = rgba[pixel_count + pixel];
                        image[2u * pixel_count + pixel] = rgba[2u * pixel_count + pixel];
                        depth[pixel] = view_depth[pixel];
                    }
                }
                return Tensor::from_vector(
                           image,
                           {static_cast<size_t>(3), static_cast<size_t>(height), static_cast<size_t>(width)},
                           lfs::core::Device::CPU)
                    .cuda();
            }

            return Tensor::from_vector(
                       image,
                       {static_cast<size_t>(3), static_cast<size_t>(height), static_cast<size_t>(width)},
                       lfs::core::Device::CPU)
                .cuda();
        }
    } // namespace

    class UtilityRenderingEngine final : public RenderingEngine {
    public:
        ~UtilityRenderingEngine() override {
            shutdown();
        }

        Result<void> initialize() override {
            initialized_ = true;
            return {};
        }

        void shutdown() override {
            initialized_ = false;
        }

        bool isInitialized() const override {
            return initialized_;
        }

        Result<GpuFrame> renderPointCloudGpuFrame(
            const lfs::core::SplatData& splat_data,
            const PointCloudRenderRequest& request) override {
            auto image_result = renderPointCloudImage(splat_data, request);
            if (!image_result || !image_result->image) {
                return std::unexpected(image_result ? "Point-cloud GPU-frame render returned no image"
                                                    : image_result.error());
            }
            return cacheTensorFrame(image_result->image, image_result->metadata, request.frame_view.size);
        }

        Result<PointCloudImageResult> renderPointCloudImage(
            const lfs::core::SplatData& splat_data,
            const PointCloudRenderRequest& request) override {
            constexpr float SH_C0 = 0.28209479177387814f;
            Tensor colors;
            try {
                colors = (splat_data.sh0_raw().slice(1, 0, 1).squeeze(1) * SH_C0 + 0.5f).clamp(0.0f, 1.0f);
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Failed to derive point colors from SH data: {}", e.what()));
            }

            auto result = renderSoftwarePointCloud(
                splat_data.get_means(),
                colors,
                request,
                splat_data.has_deleted_mask() ? &splat_data.deleted() : nullptr);
            if (!result) {
                return std::unexpected(result.error());
            }

            return PointCloudImageResult{
                .image = std::make_shared<Tensor>(std::move(result->image)),
                .metadata = makePointCloudFrameMetadata(*result)};
        }

        Result<PointCloudImageResult> renderPointCloudImage(
            const lfs::core::PointCloud& point_cloud,
            const PointCloudRenderRequest& request) override {
            auto result = renderSoftwarePointCloud(point_cloud.means, point_cloud.colors, request, nullptr);
            if (!result) {
                return std::unexpected(result.error());
            }

            return PointCloudImageResult{
                .image = std::make_shared<Tensor>(std::move(result->image)),
                .metadata = makePointCloudFrameMetadata(*result)};
        }

        Result<GpuFrame> renderPointCloudGpuFrame(
            const lfs::core::PointCloud& point_cloud,
            const PointCloudRenderRequest& request) override {
            auto image_result = renderPointCloudImage(point_cloud, request);
            if (!image_result || !image_result->image) {
                return std::unexpected(image_result ? "Raw point-cloud GPU-frame render returned no image"
                                                    : image_result.error());
            }
            return cacheTensorFrame(image_result->image, image_result->metadata, request.frame_view.size);
        }

        Result<GpuFrame> materializeGpuFrame(
            const std::shared_ptr<lfs::core::Tensor>& image,
            const FrameMetadata& metadata,
            const glm::ivec2& viewport_size) override {
            if (!image || !image->is_valid()) {
                return std::unexpected("Cannot materialize an empty tensor frame");
            }
            return cacheTensorFrame(image, metadata, viewport_size);
        }

        Result<std::shared_ptr<lfs::core::Tensor>> readbackGpuFrameColor(
            const GpuFrame& frame) override {
            if (!frame.valid() || frame.color.id != cached_tensor_frame_id_ || !cached_tensor_frame_image_) {
                return std::unexpected("Tensor-backed GPU frame is no longer available");
            }
            return cached_tensor_frame_image_;
        }

        Result<lfs::core::Tensor> renderVideoCompositeFrame(
            const std::optional<GpuFrame>& primary_frame,
            const VideoCompositeFrameRequest& request) override {
            std::shared_ptr<lfs::core::Tensor> primary_image;
            const FrameMetadata* primary_metadata = nullptr;
            if (primary_frame) {
                auto image = readbackGpuFrameColor(*primary_frame);
                if (image && *image) {
                    primary_image = *image;
                    primary_metadata = &cached_tensor_frame_metadata_;
                } else {
                    return std::unexpected(image.error());
                }
            }

            auto composite = composeVideoFrame(primary_image, primary_metadata, request);
            if (!composite) {
                return std::unexpected(composite.error());
            }
            return std::move(*composite);
        }

        Result<int> pickCameraFrustum(
            const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
            const CameraFrustumPickRequest& request) override {
            if (cameras.empty() || request.viewport_size.x <= 0.0f || request.viewport_size.y <= 0.0f ||
                request.viewport.size.x <= 0 || request.viewport.size.y <= 0) {
                return -1;
            }

            constexpr float HIT_RADIUS_PIXELS = 12.0f;
            int best_uid = -1;
            float best_score = HIT_RADIUS_PIXELS;
            float best_depth = std::numeric_limits<float>::max();
            const glm::vec2 mouse = request.mouse_pos;
            const glm::vec3 viewer_position = request.viewport.translation;

            for (size_t i = 0; i < cameras.size(); ++i) {
                const auto& camera = cameras[i];
                if (!camera) {
                    continue;
                }
                glm::mat4 scene_transform = request.scene_transform;
                if (i < request.scene_transforms.size()) {
                    scene_transform = request.scene_transforms[i];
                }

                const auto transform = cameraVisualizerTransform(*camera, scene_transform);
                if (!transform) {
                    continue;
                }
                const auto points = cameraFrustumWorldPoints(*camera, *transform, request.scale);
                if (points.empty()) {
                    continue;
                }

                float camera_best = HIT_RADIUS_PIXELS;
                if (camera->camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR) {
                    constexpr int SEGMENTS = 48;
                    for (int circle = 0; circle < 3; ++circle) {
                        const int offset = circle * SEGMENTS;
                        if (offset + SEGMENTS > static_cast<int>(points.size())) {
                            break;
                        }
                        for (int segment = 0; segment < SEGMENTS; ++segment) {
                            const auto a = projectFrustumPoint(points[offset + segment], request);
                            const auto b = projectFrustumPoint(points[offset + ((segment + 1) % SEGMENTS)], request);
                            if (!a || !b) {
                                continue;
                            }
                            camera_best = std::min(camera_best, pointSegmentDistance(mouse, *a, *b));
                        }
                    }
                } else if (points.size() >= 5) {
                    constexpr std::array<std::pair<int, int>, 8> EDGES{{
                        {0, 1},
                        {0, 2},
                        {0, 3},
                        {0, 4},
                        {1, 2},
                        {2, 3},
                        {3, 4},
                        {4, 1},
                    }};
                    for (const auto& [a_index, b_index] : EDGES) {
                        const auto a = projectFrustumPoint(points[static_cast<size_t>(a_index)], request);
                        const auto b = projectFrustumPoint(points[static_cast<size_t>(b_index)], request);
                        if (!a || !b) {
                            continue;
                        }
                        camera_best = std::min(camera_best, pointSegmentDistance(mouse, *a, *b));
                    }
                }

                const glm::vec3 camera_position = glm::vec3((*transform)[3]);
                const float depth = glm::length(camera_position - viewer_position);
                if (camera_best < best_score ||
                    (std::abs(camera_best - best_score) <= 1e-3f && depth < best_depth)) {
                    best_score = camera_best;
                    best_depth = depth;
                    best_uid = camera->uid();
                }
            }

            if (best_score < HIT_RADIUS_PIXELS)
                return best_uid;
            return -1;
        }

        ScreenOverlayRenderer* getScreenOverlayRenderer() override {
            return &overlay_renderer_;
        }

    private:
        GpuFrame cacheTensorFrame(std::shared_ptr<lfs::core::Tensor> image,
                                  const FrameMetadata& metadata,
                                  const glm::ivec2& viewport_size) {
            cached_tensor_frame_image_ = std::move(image);
            cached_tensor_frame_metadata_ = metadata;
            cached_tensor_frame_id_ = next_tensor_frame_id_++;
            if (cached_tensor_frame_id_ == 0) {
                cached_tensor_frame_id_ = next_tensor_frame_id_++;
            }

            return GpuFrame{
                .color =
                    {.id = cached_tensor_frame_id_,
                     .size = viewport_size,
                     .texcoord_scale = glm::vec2(1.0f)},
                .near_plane = metadata.near_plane,
                .far_plane = metadata.far_plane,
                .orthographic = metadata.orthographic};
        }

        bool initialized_ = false;
        unsigned int next_tensor_frame_id_ = 1;
        unsigned int cached_tensor_frame_id_ = 0;
        std::shared_ptr<lfs::core::Tensor> cached_tensor_frame_image_;
        FrameMetadata cached_tensor_frame_metadata_{};
        ScreenOverlayRenderer overlay_renderer_;
    };

    std::unique_ptr<RenderingEngine> RenderingEngine::create() {
        LOG_DEBUG("Creating utility RenderingEngine instance");
        return std::make_unique<UtilityRenderingEngine>();
    }

} // namespace lfs::rendering
