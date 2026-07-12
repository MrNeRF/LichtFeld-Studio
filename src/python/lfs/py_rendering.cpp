/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_rendering.hpp"
#include "core/checkpoint_format.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/property_registry.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "io/loader.hpp"
#include "py_scene.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/render_constants.hpp"
#include "rendering/rendering.hpp"
#include "scene/scene_render_state.hpp"
#include "visualizer/internal/viewport.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/post_work_utils.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/rendering/viewport_appearance_correction.hpp"
#include "visualizer/visualizer.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <functional>
#include <numbers>
#include <variant>

#include <glm/glm.hpp>

namespace nb = nanobind;

namespace lfs::python {

    namespace {
        constexpr std::uintmax_t MAX_RENDERED_ASSET_PREVIEW_BYTES =
            2ull * 1024ull * 1024ull * 1024ull;

        enum class PreviewReadback {
            FloatRgb,
            UInt8Rgb,
        };

        [[nodiscard]] std::optional<core::Tensor> viewportRenderImageHwc(
            const std::optional<vis::ViewportRender>& render,
            const bool clone_for_async) {
            if (!render || !render->image)
                return std::nullopt;

            auto image = clone_for_async ? render->image->clone() : *render->image;
            const auto layout = rendering::detectImageLayout(image);
            if (layout == rendering::ImageLayout::Unknown)
                return std::nullopt;
            if (layout == rendering::ImageLayout::CHW) {
                image = image.permute({1, 2, 0});
            } else {
                image = image.contiguous();
            }
            return image;
        }

        [[nodiscard]] std::optional<PyViewportRender> toPyViewportRender(
            const std::optional<vis::ViewportRender>& render,
            const bool clone_for_async) {
            auto image = viewportRenderImageHwc(render, clone_for_async);
            if (!image)
                return std::nullopt;

            std::optional<PyTensor> screen_pos;
            if (render->screen_positions) {
                auto screen_positions = clone_for_async
                                            ? render->screen_positions->clone()
                                            : *render->screen_positions;
                screen_pos = PyTensor(std::move(screen_positions), true);
            }

            return PyViewportRender{
                .image = PyTensor(std::move(*image), true),
                .screen_positions = std::move(screen_pos),
            };
        }

        [[nodiscard]] std::optional<vis::ViewportRender> captureViewportRenderThreadSafe() {
            auto invoke_capture = []() -> std::optional<vis::ViewportRender> {
                return vis::capture_viewport_render();
            };

            auto* const viewer = get_visualizer();
            if (!viewer || viewer->isOnViewerThread()) {
                return invoke_capture();
            }
            if (!viewer->acceptsPostedWork()) {
                return std::nullopt;
            }

            nb::gil_scoped_release release;
            return vis::post_work_and_wait(
                [viewer](vis::Visualizer::WorkItem work) { return viewer->postWork(std::move(work)); },
                invoke_capture,
                []() -> std::optional<vis::ViewportRender> { return std::nullopt; });
        }

        [[nodiscard]] core::PointCloud pointCloudFromMesh(const core::MeshData& mesh) {
            core::Tensor colors;
            const auto vertex_count = static_cast<std::size_t>(mesh.vertex_count());
            if (mesh.colors.is_valid() && mesh.colors.ndim() == 2 &&
                mesh.colors.size(0) == mesh.vertex_count() && mesh.colors.size(1) >= 3) {
                colors = mesh.colors.slice(1, 0, 3).contiguous();
            } else {
                colors = core::Tensor::full(
                    {vertex_count, static_cast<std::size_t>(3)},
                    0.72f,
                    mesh.vertices.device(),
                    core::DataType::Float32);
            }
            return core::PointCloud(mesh.vertices, std::move(colors));
        }

        [[nodiscard]] std::optional<core::Tensor> renderSplatAssetPreview(
            vis::RenderingManager& rendering_manager,
            core::SplatData& splat,
            const int width,
            const int height,
            const float focal_length_mm,
            const glm::mat3* custom_rotation = nullptr,
            const glm::vec3* custom_translation = nullptr) {
            if (splat.size() == 0) {
                return std::nullopt;
            }

            Viewport preview_viewport(
                static_cast<std::size_t>(width),
                static_cast<std::size_t>(height));
            if (custom_rotation && custom_translation) {
                preview_viewport.setViewMatrix(*custom_rotation, *custom_translation);
            } else {
                preview_viewport.camera.resetToHome();
            }

            vis::SceneRenderState scene_state;
            scene_state.combined_model = &splat;
            scene_state.model_transforms = {
                rendering::dataWorldTransformToVisualizerWorld(glm::mat4(1.0f)),
            };
            scene_state.transform_indices = std::make_shared<core::Tensor>(
                core::Tensor::zeros(
                    {static_cast<std::size_t>(splat.size())},
                    core::Device::CUDA,
                    core::DataType::Int32));
            scene_state.node_visibility_mask = {true};
            scene_state.selected_node_mask = {true};
            scene_state.visible_splat_count = 1;

            const auto image = rendering_manager.renderPreviewImage(
                splat,
                std::move(scene_state),
                preview_viewport.getRotationMatrix(),
                preview_viewport.getTranslation(),
                focal_length_mm,
                width,
                height);
            if (!image || !image->is_valid()) {
                return std::nullopt;
            }
            return image->clone();
        }

        [[nodiscard]] std::optional<core::Tensor> renderPointCloudAssetPreview(
            vis::RenderingManager& rendering_manager,
            const core::PointCloud& point_cloud,
            const int width,
            const int height,
            const float focal_length_mm,
            const glm::mat3* custom_rotation = nullptr,
            const glm::vec3* custom_translation = nullptr) {
            if (point_cloud.size() == 0) {
                return std::nullopt;
            }

            auto* const engine = rendering_manager.getRenderingEngine();
            if (!engine) {
                return std::nullopt;
            }

            Viewport preview_viewport(
                static_cast<std::size_t>(width),
                static_cast<std::size_t>(height));
            if (custom_rotation && custom_translation) {
                preview_viewport.setViewMatrix(*custom_rotation, *custom_translation);
            } else {
                preview_viewport.camera.resetToHome();
            }

            std::vector<glm::mat4> model_transforms{
                rendering::dataWorldTransformToVisualizerWorld(glm::mat4(1.0f)),
            };
            auto transform_indices = std::make_shared<core::Tensor>(
                core::Tensor::zeros(
                    {static_cast<std::size_t>(point_cloud.size())},
                    core::Device::CUDA,
                    core::DataType::Int32));

            rendering::PointCloudRenderRequest request{};
            request.frame_view.rotation = preview_viewport.getRotationMatrix();
            request.frame_view.translation = preview_viewport.getTranslation();
            request.frame_view.size = {width, height};
            request.frame_view.focal_length_mm = focal_length_mm;
            request.render.scaling_modifier = 1.0f;
            request.render.voxel_size = 0.01f;
            request.render.equirectangular = false;
            request.scene.model_transforms = &model_transforms;
            request.scene.transform_indices = std::move(transform_indices);
            request.scene.node_visibility_mask = {true};
            request.transparent_background = false;

            auto render_result = engine->renderPointCloudImage(point_cloud, request);
            if (!render_result || !render_result->image || !render_result->image->is_valid()) {
                if (!render_result) {
                    LOG_DEBUG("Point-cloud asset preview render failed: {}", render_result.error());
                }
                return std::nullopt;
            }
            return render_result->image->clone();
        }

        [[nodiscard]] std::optional<core::Tensor> renderAssetPreviewOnViewerThread(
            const std::string& path,
            const int width,
            const int height,
            const float focal_length_mm,
            const glm::mat3* rotation = nullptr,
            const glm::vec3* translation = nullptr) {
            if (width <= 0 || height <= 0) {
                return std::nullopt;
            }

            auto* const viewer = get_visualizer();
            auto* const rendering_manager = viewer ? viewer->getRenderingManager() : nullptr;
            if (!rendering_manager) {
                return std::nullopt;
            }

            try {
                auto loader = io::Loader::create();
                if (!loader) {
                    return std::nullopt;
                }

                io::LoadOptions options;
                options.resize_factor = -1;
                options.max_width = 0;
                options.images_folder = "images";
                options.validate_only = false;
                options.splat_tensor_allocator = rendering_manager->makeSplatTensorAllocator();

                const auto asset_path = core::utf8_to_path(path);
                std::error_code file_size_error;
                if (std::filesystem::is_regular_file(asset_path, file_size_error)) {
                    const auto file_size = std::filesystem::file_size(asset_path, file_size_error);
                    if (!file_size_error && file_size > MAX_RENDERED_ASSET_PREVIEW_BYTES) {
                        LOG_DEBUG("Skipping rendered asset preview for '{}' ({} MB exceeds {} MB budget)",
                                  path,
                                  file_size / (1024 * 1024),
                                  MAX_RENDERED_ASSET_PREVIEW_BYTES / (1024 * 1024));
                        return std::nullopt;
                    }
                }
                auto ext = asset_path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".ckpt" || ext == ".resume") {
                    auto checkpoint_result = core::load_checkpoint_splat_data(
                        asset_path,
                        rendering_manager->makeSplatTensorAllocator());
                    if (!checkpoint_result) {
                        LOG_DEBUG(
                            "Checkpoint asset preview load failed for '{}': {}",
                            path,
                            checkpoint_result.error());
                        return std::nullopt;
                    }
                    return renderSplatAssetPreview(
                        *rendering_manager,
                        *checkpoint_result,
                        width,
                        height,
                        focal_length_mm,
                        rotation,
                        translation);
                }

                auto load_result = loader->load(asset_path, options);
                if (!load_result) {
                    LOG_DEBUG("Asset preview load failed for '{}': {}", path, load_result.error().format());
                    return std::nullopt;
                }

                const auto* loaded_splat = std::get_if<std::shared_ptr<core::SplatData>>(&load_result->data);
                if (loaded_splat && *loaded_splat) {
                    return renderSplatAssetPreview(
                        *rendering_manager,
                        **loaded_splat,
                        width,
                        height,
                        focal_length_mm,
                        rotation,
                        translation);
                }

                const auto* loaded_scene = std::get_if<io::LoadedScene>(&load_result->data);
                if (loaded_scene && loaded_scene->point_cloud) {
                    return renderPointCloudAssetPreview(
                        *rendering_manager,
                        *loaded_scene->point_cloud,
                        width,
                        height,
                        focal_length_mm,
                        rotation,
                        translation);
                }

                const auto* loaded_mesh = std::get_if<std::shared_ptr<core::MeshData>>(&load_result->data);
                if (loaded_mesh && *loaded_mesh && (*loaded_mesh)->vertex_count() > 0) {
                    auto mesh_points = pointCloudFromMesh(**loaded_mesh);
                    return renderPointCloudAssetPreview(
                        *rendering_manager,
                        mesh_points,
                        width,
                        height,
                        focal_length_mm,
                        rotation,
                        translation);
                }
            } catch (const std::exception& e) {
                LOG_DEBUG("Asset preview render failed for '{}': {}", path, e.what());
            } catch (...) {
                LOG_DEBUG("Asset preview render failed for '{}': unknown error", path);
            }
            return std::nullopt;
        }

        [[nodiscard]] std::optional<core::Tensor> renderAssetPreviewThreadSafe(
            const std::string& path,
            const int width,
            const int height,
            const float focal_length_mm,
            const glm::mat3* rotation = nullptr,
            const glm::vec3* translation = nullptr) {
            auto invoke_render = [&]() -> std::optional<core::Tensor> {
                return renderAssetPreviewOnViewerThread(path, width, height, focal_length_mm, rotation, translation);
            };

            auto* const viewer = get_visualizer();
            if (!viewer || viewer->isOnViewerThread()) {
                return invoke_render();
            }
            if (!viewer->acceptsPostedWork()) {
                return std::nullopt;
            }

            nb::gil_scoped_release release;
            return vis::post_work_and_wait(
                [viewer](vis::Visualizer::WorkItem work) { return viewer->postWork(std::move(work)); },
                invoke_render,
                []() -> std::optional<core::Tensor> { return std::nullopt; });
        }

        [[nodiscard]] std::optional<glm::mat3> tensorToVisualizerRotation(const PyTensor& py_tensor) {
            const auto& tensor = py_tensor.tensor();
            if (!tensor.is_valid() || tensor.ndim() != 2 || tensor.size(0) != 3 || tensor.size(1) != 3) {
                return std::nullopt;
            }

            try {
                const auto cpu = tensor.to(core::Device::CPU).to(core::DataType::Float32).contiguous();
                const auto* const data = static_cast<const float*>(cpu.data_ptr());
                if (!data) {
                    return std::nullopt;
                }

                glm::mat3 rotation{};
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        const float value = data[row * 3 + col];
                        if (!std::isfinite(value)) {
                            return std::nullopt;
                        }
                        rotation[col][row] = value;
                    }
                }
                return rotation;
            } catch (const std::exception& e) {
                LOG_DEBUG("render_view rotation conversion failed: {}", e.what());
            } catch (...) {
                LOG_DEBUG("render_view rotation conversion failed: unknown error");
            }
            return std::nullopt;
        }

        [[nodiscard]] std::optional<glm::vec3> tensorToVisualizerTranslation(const PyTensor& py_tensor) {
            const auto& tensor = py_tensor.tensor();
            if (!tensor.is_valid() || tensor.ndim() != 1 || tensor.size(0) != 3) {
                return std::nullopt;
            }

            try {
                const auto cpu = tensor.to(core::Device::CPU).to(core::DataType::Float32).contiguous();
                const auto* const data = static_cast<const float*>(cpu.data_ptr());
                if (!data) {
                    return std::nullopt;
                }

                const glm::vec3 translation{data[0], data[1], data[2]};
                if (!std::isfinite(translation.x) || !std::isfinite(translation.y) || !std::isfinite(translation.z)) {
                    return std::nullopt;
                }
                return translation;
            } catch (const std::exception& e) {
                LOG_DEBUG("render_view translation conversion failed: {}", e.what());
            } catch (...) {
                LOG_DEBUG("render_view translation conversion failed: unknown error");
            }
            return std::nullopt;
        }

        [[nodiscard]] std::optional<core::Tensor> renderViewOnViewerThread(
            const glm::mat3& rotation,
            const glm::vec3& translation,
            const int width,
            const int height,
            const float fov_degrees,
            const PreviewReadback readback,
            const std::optional<glm::vec3>& background_color_override,
            const std::optional<bool> orthographic_override = std::nullopt,
            const std::optional<float> ortho_scale_override = std::nullopt) {
            if (width <= 0 || height <= 0 || !std::isfinite(fov_degrees) || fov_degrees <= 0.0f) {
                return std::nullopt;
            }

            auto* const viewer = get_visualizer();
            auto* const rendering_manager = viewer ? viewer->getRenderingManager() : nullptr;
            auto* const scene_manager = viewer ? viewer->getSceneManager() : nullptr;
            if (!rendering_manager || !scene_manager) {
                return std::nullopt;
            }

            std::shared_ptr<core::Tensor> image;
            if (readback == PreviewReadback::UInt8Rgb) {
                image = rendering_manager->renderPreviewImageRgb8(
                    scene_manager,
                    rotation,
                    translation,
                    lfs::rendering::vFovToFocalLength(fov_degrees),
                    width,
                    height,
                    background_color_override,
                    orthographic_override,
                    ortho_scale_override);
            } else {
                image = rendering_manager->renderPreviewImage(
                    scene_manager,
                    rotation,
                    translation,
                    lfs::rendering::vFovToFocalLength(fov_degrees),
                    width,
                    height,
                    background_color_override,
                    orthographic_override,
                    ortho_scale_override);
            }
            if (readback != PreviewReadback::FloatRgb) {
                rendering_manager->releasePreviewImageResources();
            }
            if (!image || !image->is_valid()) {
                return std::nullopt;
            }
            return *image;
        }

        [[nodiscard]] std::optional<core::Tensor> renderViewThreadSafe(
            const glm::mat3& rotation,
            const glm::vec3& translation,
            const int width,
            const int height,
            const float fov_degrees,
            const PreviewReadback readback,
            const std::optional<glm::vec3>& background_color_override,
            const std::optional<bool> orthographic_override = std::nullopt,
            const std::optional<float> ortho_scale_override = std::nullopt) {
            auto invoke_render = [&]() -> std::optional<core::Tensor> {
                return renderViewOnViewerThread(
                    rotation,
                    translation,
                    width,
                    height,
                    fov_degrees,
                    readback,
                    background_color_override,
                    orthographic_override,
                    ortho_scale_override);
            };

            auto* const viewer = get_visualizer();
            if (!viewer || viewer->isOnViewerThread()) {
                return invoke_render();
            }
            if (!viewer->acceptsPostedWork()) {
                return std::nullopt;
            }

            nb::gil_scoped_release release;
            return vis::post_work_and_wait(
                [viewer](vis::Visualizer::WorkItem work) { return viewer->postWork(std::move(work)); },
                invoke_render,
                []() -> std::optional<core::Tensor> { return std::nullopt; });
        }

        [[nodiscard]] std::optional<std::pair<core::Tensor, core::Tensor>> renderViewAndDepthOnViewerThread(
            const glm::mat3& rotation,
            const glm::vec3& translation,
            const int width,
            const int height,
            const float fov_degrees,
            const bool expected_depth) {
            if (width <= 0 || height <= 0 || !std::isfinite(fov_degrees) || fov_degrees <= 0.0f) {
                return std::nullopt;
            }
            auto* const viewer = get_visualizer();
            auto* const rendering_manager = viewer ? viewer->getRenderingManager() : nullptr;
            auto* const scene_manager = viewer ? viewer->getSceneManager() : nullptr;
            if (!rendering_manager || !scene_manager) {
                return std::nullopt;
            }
            auto rgbd = rendering_manager->renderPreviewImageAndDepth(
                scene_manager,
                rotation,
                translation,
                lfs::rendering::vFovToFocalLength(fov_degrees),
                width,
                height,
                expected_depth,
                std::nullopt);
            if (!rgbd.image || !rgbd.depth || !rgbd.image->is_valid() || !rgbd.depth->is_valid()) {
                return std::nullopt;
            }
            auto image = *rgbd.image;
            auto depth = *rgbd.depth;
            if (image.device() != core::Device::CPU) {
                image = image.cpu();
            }
            if (depth.device() != core::Device::CPU) {
                depth = depth.cpu();
            }
            return std::make_pair(image.contiguous(), depth.contiguous());
        }

        [[nodiscard]] std::optional<std::pair<core::Tensor, core::Tensor>> renderViewAndDepthThreadSafe(
            const glm::mat3& rotation,
            const glm::vec3& translation,
            const int width,
            const int height,
            const float fov_degrees,
            const bool expected_depth) {
            auto invoke_render = [&]() {
                return renderViewAndDepthOnViewerThread(rotation, translation, width, height, fov_degrees, expected_depth);
            };

            auto* const viewer = get_visualizer();
            if (!viewer || viewer->isOnViewerThread()) {
                return invoke_render();
            }
            if (!viewer->acceptsPostedWork()) {
                return std::nullopt;
            }

            nb::gil_scoped_release release;
            using Result = std::optional<std::pair<core::Tensor, core::Tensor>>;
            return vis::post_work_and_wait(
                [viewer](vis::Visualizer::WorkItem work) { return viewer->postWork(std::move(work)); },
                invoke_render,
                []() -> Result { return std::nullopt; });
        }
    } // namespace

    void set_render_scene_context(core::Scene* scene) {
        set_scene_for_python(scene);
    }

    core::Scene* get_render_scene() {
        if (auto* app_scene = get_application_scene()) {
            return app_scene;
        }
        return get_scene_for_python();
    }

    void register_render_settings_properties() {
        using namespace core::prop;
        using Proxy = vis::RenderSettingsProxy;

        PropertyGroup group;
        group.id = "render_settings";
        group.name = "Render Settings";

        auto add_color3 = [&](std::array<float, 3> Proxy::*member, const std::string& id, const std::string& name,
                              const std::string& desc, std::array<double, 3> default_val) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Color3;
            meta.default_vec3 = default_val;
            meta.min_value = 0.0;
            meta.max_value = 1.0;
            meta.getter = [member](const PropertyObjectRef& ref) -> std::any {
                const auto& arr = static_cast<const Proxy*>(ref.ptr)->*member;
                return std::array<float, 3>{arr[0], arr[1], arr[2]};
            };
            meta.setter = [member](PropertyObjectRef& ref, const std::any& val) {
                static_cast<Proxy*>(ref.ptr)->*member = std::any_cast<std::array<float, 3>>(val);
            };
            group.properties.push_back(std::move(meta));
        };

        auto add_bool = [&](bool Proxy::*member, const std::string& id, const std::string& name, const std::string& desc,
                            bool default_val) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Bool;
            meta.default_value = default_val ? 1.0 : 0.0;
            meta.getter = [member](const PropertyObjectRef& ref) -> std::any {
                return static_cast<const Proxy*>(ref.ptr)->*member;
            };
            meta.setter = [member](PropertyObjectRef& ref, const std::any& val) {
                static_cast<Proxy*>(ref.ptr)->*member = std::any_cast<bool>(val);
            };
            group.properties.push_back(std::move(meta));
        };

        auto add_float = [&](float Proxy::*member, const std::string& id, const std::string& name,
                             const std::string& desc, double default_val, double min_val, double max_val) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Float;
            meta.default_value = default_val;
            meta.min_value = min_val;
            meta.max_value = max_val;
            meta.getter = [member](const PropertyObjectRef& ref) -> std::any {
                return static_cast<const Proxy*>(ref.ptr)->*member;
            };
            meta.setter = [member](PropertyObjectRef& ref, const std::any& val) {
                static_cast<Proxy*>(ref.ptr)->*member = std::any_cast<float>(val);
            };
            group.properties.push_back(std::move(meta));
        };

        auto add_int_enum = [&](int Proxy::*member, const std::string& id, const std::string& name,
                                const std::string& desc, std::vector<EnumItem> items, int default_idx) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Enum;
            meta.enum_items = items;
            meta.default_enum = default_idx;
            meta.getter = [member, items](const PropertyObjectRef& ref) -> std::any {
                int val = static_cast<const Proxy*>(ref.ptr)->*member;
                for (const auto& item : items) {
                    if (item.value == val) {
                        return item.identifier;
                    }
                }
                return std::to_string(val);
            };
            meta.setter = [member, items](PropertyObjectRef& ref, const std::any& val) {
                if (val.type() == typeid(int)) {
                    static_cast<Proxy*>(ref.ptr)->*member = std::any_cast<int>(val);
                } else if (val.type() == typeid(std::string)) {
                    std::string id_str = std::any_cast<std::string>(val);
                    for (const auto& item : items) {
                        if (item.identifier == id_str) {
                            static_cast<Proxy*>(ref.ptr)->*member = item.value;
                            return;
                        }
                    }
                    static_cast<Proxy*>(ref.ptr)->*member = std::stoi(id_str);
                }
            };
            group.properties.push_back(std::move(meta));
        };

        auto add_string = [&](std::string Proxy::*member, const std::string& id, const std::string& name,
                              const std::string& desc, const std::string& default_val) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::String;
            meta.default_string = default_val;
            meta.getter = [member](const PropertyObjectRef& ref) -> std::any {
                return static_cast<const Proxy*>(ref.ptr)->*member;
            };
            meta.setter = [member](PropertyObjectRef& ref, const std::any& val) {
                static_cast<Proxy*>(ref.ptr)->*member = std::any_cast<std::string>(val);
            };
            group.properties.push_back(std::move(meta));
        };

        // Background
        add_color3(&Proxy::background_color, "background_color", "Color", "Viewport background color", {0.0, 0.0, 0.0});
        add_int_enum(&Proxy::environment_mode, "environment_mode", "Environment", "Viewport background mode",
                     {{"Solid Color", "SOLID_COLOR", 0}, {"Equirectangular HDRI", "EQUIRECTANGULAR", 1}}, 0);
        add_string(&Proxy::environment_map_path, "environment_map_path", "Preset",
                   "Relative asset path or absolute HDRI path for the environment background",
                   std::string(vis::kDefaultEnvironmentMapPath));
        add_float(&Proxy::environment_exposure, "environment_exposure", "Exposure", "Environment exposure in EV", 0.0, -6.0, 6.0);
        add_float(&Proxy::environment_rotation_degrees, "environment_rotation_degrees", "Rotation",
                  "Environment yaw rotation in degrees", 0.0, -180.0, 180.0);

        // Coordinate Axes
        add_bool(&Proxy::show_coord_axes, "show_coord_axes", "Show Coordinate Axes",
                 "Display coordinate axes in viewport", false);
        add_float(&Proxy::axes_size, "axes_size", "Size", "Size of coordinate axes", 2.0, 0.5, 10.0);

        // Pivot
        add_bool(&Proxy::show_pivot, "show_pivot", "Show Pivot", "Display pivot point", false);

        // Grid
        add_bool(&Proxy::show_grid, "show_grid", "Show Grid", "Display grid in viewport", true);
        add_int_enum(&Proxy::grid_plane, "grid_plane", "Plane", "Grid orientation",
                     {{"YZ (Right)", "0", 0}, {"XZ (Top)", "1", 1}, {"XY (Front)", "2", 2}}, 1);
        add_float(&Proxy::grid_opacity, "grid_opacity", "Grid Opacity", "Grid transparency", 0.5, 0.0, 1.0);

        // Camera Frustums
        add_bool(&Proxy::show_camera_frustums, "show_camera_frustums", "Camera Frustums",
                 "Show camera frustum wireframes", false);
        add_float(&Proxy::camera_frustum_scale, "camera_frustum_scale", "Scale", "Camera frustum display scale", 0.25,
                  0.01, 10.0);

        // Point Cloud Mode
        add_bool(&Proxy::point_cloud_mode, "point_cloud_mode", "Point Cloud Mode",
                 "Render as point cloud instead of splats", false);
        add_float(&Proxy::voxel_size, "voxel_size", "Point Size", "Point size in point cloud mode", 0.01, 0.001, 0.1);

        // Selection Colors
        add_color3(&Proxy::selection_color_committed, "selection_color_committed", "Committed",
                   "Committed selection color", {0.859, 0.325, 0.325});
        add_color3(&Proxy::selection_color_preview, "selection_color_preview", "Preview", "Preview selection color",
                   {0.0, 0.871, 0.298});
        add_color3(&Proxy::selection_color_center_marker, "selection_color_center_marker", "Center Marker",
                   "Selection center marker color", {0.0, 0.604, 0.733});

        // Desaturation
        add_bool(&Proxy::desaturate_unselected, "desaturate_unselected", "Desaturate Unselected",
                 "Desaturate unselected PLYs when one is selected", false);
        add_bool(&Proxy::desaturate_cropping, "desaturate_cropping", "Desaturate Cropping",
                 "Dim outside crop area instead of hiding", true);
        add_bool(&Proxy::hide_outside_depth_box, "hide_outside_depth_box", "Hide Outside Depth Box",
                 "Hide Gaussians outside the selection depth box", false);

        // View Settings
        add_float(&Proxy::focal_length_mm, "focal_length_mm", "Focal Length", "Focal length in mm", 35.0, 10.0, 200.0);
        add_int_enum(&Proxy::sh_degree, "sh_degree", "SH Degree", "Spherical harmonics degree",
                     {{"0", "0", 0}, {"1", "1", 1}, {"2", "2", 2}, {"3", "3", 3}}, 3);
        add_bool(&Proxy::equirectangular, "equirectangular", "Equirectangular", "Equirectangular projection mode",
                 false);
        add_int_enum(&Proxy::raster_backend, "raster_backend", "Raster Backend", "Gaussian rasterization backend",
                     {{"3DGS", "3dgs", 2},
                      {"3DGUT", "3dgut", 3}},
                     2);
        add_bool(&Proxy::mip_filter, "mip_filter", "Mip Filter", "Enable mip-map filtering", false);
        add_float(&Proxy::render_scale, "render_scale", "Render Scale", "Render resolution scale", 1.0, 0.25, 1.0);
        add_float(&Proxy::depth_view_min, "depth_view_min", "Depth Near", "Depth-map visualization near range",
                  lfs::rendering::DEFAULT_DEPTH_VIEW_MIN, 0.0, lfs::rendering::MAX_DEPTH_VIEW_DISTANCE);
        add_float(&Proxy::depth_view_max, "depth_view_max", "Depth Far", "Depth-map visualization far range",
                  lfs::rendering::DEFAULT_DEPTH_VIEW_MAX, lfs::rendering::DEFAULT_DEPTH_VIEW_MIN,
                  lfs::rendering::MAX_DEPTH_VIEW_DISTANCE);
        add_int_enum(&Proxy::depth_visualization_mode, "depth_visualization_mode", "Depth Mode",
                     "Depth-map visualization mode",
                     {{"Color", "palette", 0}, {"Gray", "gray", 1}}, 0);
        add_int_enum(&Proxy::gt_comparison_mode, "gt_comparison_mode", "GT Compare",
                     "Ground-truth comparison payload",
                     {{"RGB", "rgb", 0}, {"Normal", "normal", 1}, {"Depth", "depth", 2}}, 0);
        add_int_enum(&Proxy::camera_metrics_mode, "camera_metrics_mode", "Camera Metrics",
                     "Compute metrics when jumping to a source camera",
                     {{"Off", "OFF", 0}, {"PSNR", "PSNR", 1}, {"PSNR + SSIM", "PSNR_SSIM", 2}}, 0);

        add_bool(&Proxy::lod_enabled, "lod_enabled", "Enable LOD",
                 "Enable hierarchical level-of-detail rendering", false);
        add_bool(&Proxy::lod_debug_colors, "lod_debug_colors", "Debug Colors",
                 "Color splats by their LOD level for debugging", false);
        add_float(&Proxy::lod_max_splats, "lod_max_splats", "LOD Budget",
                  "Maximum number of splats in the dynamic LOD cut",
                  static_cast<double>(vis::DEFAULT_LOD_MAX_SPLATS), 1.0, 10000000.0);
        add_float(&Proxy::lod_page_pool_splats, "lod_page_pool_splats", "LOD Cache Budget",
                  "VRAM page-pool budget in splats for streamed RAD scenes (0 = auto)",
                  static_cast<double>(vis::DEFAULT_LOD_PAGE_POOL_SPLATS), 0.0, 100000000.0);
        add_float(&Proxy::lod_pool_vram_fraction, "lod_pool_vram_fraction", "LOD Pool VRAM Fraction",
                  "Share of free VRAM granted to the out-of-core LOD page pool",
                  static_cast<double>(vis::DEFAULT_LOD_POOL_VRAM_FRACTION), 0.05, 0.9);
        add_float(&Proxy::lod_fade_frames, "lod_fade_frames", "LOD Fade Frames",
                  "Frames a newly streamed LOD page fades in over (0 = instant)",
                  static_cast<double>(vis::DEFAULT_LOD_FADE_FRAMES), 0.0, 240.0);
        add_float(&Proxy::lod_render_scale, "lod_render_scale", "Render Scale",
                  "Quality multiplier: effective splat target = LOD Budget x Render Scale", vis::DEFAULT_LOD_RENDER_SCALE, 0.1, 5.0);
        add_float(&Proxy::lod_cone_foveation, "lod_cone_foveation", "Cone Foveation",
                  "Peripheral LOD penalty factor (1.0 = no penalty)", vis::DEFAULT_LOD_CONE_FOVEATION, 0.1, 2.0);
        add_float(&Proxy::lod_cone_inner_degrees, "lod_cone_inner_degrees", "Cone Inner",
                  "Inner cone angle in degrees (no penalty inside this angle)", vis::DEFAULT_LOD_CONE_INNER_DEGREES, 0.0, 180.0);
        add_float(&Proxy::lod_cone_outer_degrees, "lod_cone_outer_degrees", "Cone Outer",
                  "Outer cone angle in degrees (full penalty beyond this angle)", vis::DEFAULT_LOD_CONE_OUTER_DEGREES, 0.0, 180.0);

        add_bool(&Proxy::apply_appearance_correction, "apply_appearance_correction", "Appearance Correction",
                 "Enable PPISP appearance correction", false);
        add_int_enum(&Proxy::ppisp_mode, "ppisp_mode", "Mode", "PPISP correction mode",
                     {{"Manual", "MANUAL", 0}, {"Auto", "AUTO", 1}}, 1);

        using PPISP = vis::PPISPOverrides;
        const auto add_ppisp_float = [&](float PPISP::*member, const char* id, const char* name,
                                         const char* desc, double def, double min_v, double max_v) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Float;
            meta.default_value = def;
            meta.min_value = min_v;
            meta.max_value = max_v;
            meta.getter = [member](const PropertyObjectRef& ref) -> std::any {
                return static_cast<const Proxy*>(ref.ptr)->ppisp.*member;
            };
            meta.setter = [member](PropertyObjectRef& ref, const std::any& val) {
                static_cast<Proxy*>(ref.ptr)->ppisp.*member = std::any_cast<float>(val);
            };
            group.properties.push_back(std::move(meta));
        };

        const auto add_ppisp_bool = [&](bool PPISP::*member, const char* id, const char* name,
                                        const char* desc, bool def) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Bool;
            meta.default_value = def ? 1.0 : 0.0;
            meta.getter = [member](const PropertyObjectRef& ref) -> std::any {
                return static_cast<const Proxy*>(ref.ptr)->ppisp.*member;
            };
            meta.setter = [member](PropertyObjectRef& ref, const std::any& val) {
                static_cast<Proxy*>(ref.ptr)->ppisp.*member = std::any_cast<bool>(val);
            };
            group.properties.push_back(std::move(meta));
        };

        add_ppisp_float(&PPISP::exposure_offset, "ppisp_exposure", "Exposure", "Exposure offset (EV)", 0.0, -3.0, 3.0);
        add_ppisp_bool(&PPISP::vignette_enabled, "ppisp_vignette_enabled", "Vignette", "Enable vignette correction", true);
        add_ppisp_float(&PPISP::vignette_strength, "ppisp_vignette_strength", "Vignette Strength", "Vignette strength", 1.0, 0.0, 2.0);
        add_ppisp_float(&PPISP::wb_temperature, "ppisp_wb_temperature", "Temperature", "White balance temperature", 0.0, -1.0, 1.0);
        add_ppisp_float(&PPISP::wb_tint, "ppisp_wb_tint", "Tint", "White balance tint", 0.0, -1.0, 1.0);
        add_ppisp_float(&PPISP::gamma_multiplier, "ppisp_gamma_multiplier", "Gamma", "Gamma multiplier", 1.0, 0.5, 2.5);
        add_ppisp_float(&PPISP::gamma_red, "ppisp_gamma_red", "Gamma Red", "Red gamma offset", 0.0, -0.5, 0.5);
        add_ppisp_float(&PPISP::gamma_green, "ppisp_gamma_green", "Gamma Green", "Green gamma offset", 0.0, -0.5, 0.5);
        add_ppisp_float(&PPISP::gamma_blue, "ppisp_gamma_blue", "Gamma Blue", "Blue gamma offset", 0.0, -0.5, 0.5);
        add_ppisp_float(&PPISP::crf_toe, "ppisp_crf_toe", "Toe", "Shadow compression", 0.0, -1.0, 1.0);
        add_ppisp_float(&PPISP::crf_shoulder, "ppisp_crf_shoulder", "Shoulder", "Highlight roll-off", 0.0, -1.0, 1.0);
        add_ppisp_float(&PPISP::color_red_x, "ppisp_color_red_x", "Red X", "Red chromaticity X", 0.0, -0.5, 0.5);
        add_ppisp_float(&PPISP::color_red_y, "ppisp_color_red_y", "Red Y", "Red chromaticity Y", 0.0, -0.5, 0.5);
        add_ppisp_float(&PPISP::color_green_x, "ppisp_color_green_x", "Green X", "Green chromaticity X", 0.0, -0.5, 0.5);
        add_ppisp_float(&PPISP::color_green_y, "ppisp_color_green_y", "Green Y", "Green chromaticity Y", 0.0, -0.5, 0.5);
        add_ppisp_float(&PPISP::color_blue_x, "ppisp_color_blue_x", "Blue X", "Blue chromaticity X", 0.0, -0.5, 0.5);
        add_ppisp_float(&PPISP::color_blue_y, "ppisp_color_blue_y", "Blue Y", "Blue chromaticity Y", 0.0, -0.5, 0.5);

        add_bool(&Proxy::mesh_wireframe, "mesh_wireframe", "Wireframe Overlay", "Show wireframe on meshes", false);
        add_color3(&Proxy::mesh_wireframe_color, "mesh_wireframe_color", "Wireframe Color", "Mesh wireframe color",
                   {0.2, 0.2, 0.2});
        add_float(&Proxy::mesh_wireframe_width, "mesh_wireframe_width", "Wireframe Width", "Wireframe line width", 1.0,
                  0.5, 5.0);
        add_float(&Proxy::mesh_light_intensity, "mesh_light_intensity", "Light Intensity", "Mesh light intensity", 0.7,
                  0.0, 5.0);
        add_float(&Proxy::mesh_ambient, "mesh_ambient", "Ambient", "Mesh ambient light", 0.4, 0.0, 1.0);
        add_bool(&Proxy::mesh_backface_culling, "mesh_backface_culling", "Backface Culling", "Cull mesh back faces",
                 true);
        add_bool(&Proxy::mesh_shadow_enabled, "mesh_shadow_enabled", "Shadows", "Enable shadow mapping for meshes",
                 false);
        add_int_enum(&Proxy::mesh_shadow_resolution, "mesh_shadow_resolution", "Shadow Resolution",
                     "Shadow map resolution",
                     {{"512", "512", 512},
                      {"1024", "1024", 1024},
                      {"2048", "2048", 2048},
                      {"4096", "4096", 4096}},
                     2);

        PropertyRegistry::instance().register_group(std::move(group));
    }

    PyRenderSettings::PyRenderSettings(vis::RenderSettingsProxy settings)
        : settings_(std::move(settings)),
          prop_(&settings_, "render_settings") {}

    void PyRenderSettings::set(const std::string& name, nb::object value) {
        prop_.setattr(name, value);
        if (name == "raster_backend") {
            const auto backend = static_cast<rendering::GaussianRasterBackend>(settings_.raster_backend);
            settings_.raster_backend =
                static_cast<int>(rendering::normalizeViewerRasterBackend(backend, settings_.gut));
            settings_.gut = rendering::isGutBackend(
                static_cast<rendering::GaussianRasterBackend>(settings_.raster_backend));
        }
        vis::update_render_settings(settings_);
        request_redraw();
    }

    void PyRenderSettings::prop_setattr(const std::string& name, nb::object value) {
        set(name, value);
    }

    nb::dict PyRenderSettings::get_all_properties() const {
        nb::dict result;
        const auto* group = core::prop::PropertyRegistry::instance().get_group(prop_.group_id());
        if (!group) {
            return result;
        }

        nb::module_ props_module = nb::module_::import_("lfs_plugins.props");

        for (const auto& meta : group->properties) {
            nb::object prop_obj;

            switch (meta.type) {
            case core::prop::PropType::Float: {
                nb::object cls = props_module.attr("FloatProperty");
                prop_obj = cls(
                    nb::arg("default") = static_cast<float>(meta.default_value),
                    nb::arg("min") = static_cast<float>(meta.min_value),
                    nb::arg("max") = static_cast<float>(meta.max_value),
                    nb::arg("step") = static_cast<float>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case core::prop::PropType::Int: {
                nb::object cls = props_module.attr("IntProperty");
                prop_obj = cls(
                    nb::arg("default") = static_cast<int>(meta.default_value),
                    nb::arg("min") = static_cast<int>(meta.min_value),
                    nb::arg("max") = static_cast<int>(meta.max_value),
                    nb::arg("step") = static_cast<int>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case core::prop::PropType::Bool: {
                nb::object cls = props_module.attr("BoolProperty");
                prop_obj = cls(
                    nb::arg("default") = meta.default_value != 0.0,
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case core::prop::PropType::String: {
                nb::object cls = props_module.attr("StringProperty");
                prop_obj = cls(
                    nb::arg("default") = meta.default_string,
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case core::prop::PropType::Enum: {
                nb::object cls = props_module.attr("EnumProperty");
                nb::list items;
                std::string default_id;
                for (size_t i = 0; i < meta.enum_items.size(); ++i) {
                    const auto& item = meta.enum_items[i];
                    items.append(nb::make_tuple(item.identifier, item.name, ""));
                    if (static_cast<int>(i) == meta.default_enum) {
                        default_id = item.identifier;
                    }
                }
                prop_obj = cls(
                    nb::arg("items") = items,
                    nb::arg("default") = default_id,
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case core::prop::PropType::Vec3:
            case core::prop::PropType::Color3: {
                nb::object cls = props_module.attr("FloatVectorProperty");
                std::string subtype = (meta.type == core::prop::PropType::Color3) ? "COLOR" : "";
                prop_obj = cls(
                    nb::arg("default") = nb::make_tuple(
                        static_cast<float>(meta.default_vec3[0]),
                        static_cast<float>(meta.default_vec3[1]),
                        static_cast<float>(meta.default_vec3[2])),
                    nb::arg("size") = 3,
                    nb::arg("min") = static_cast<float>(meta.min_value),
                    nb::arg("max") = static_cast<float>(meta.max_value),
                    nb::arg("subtype") = subtype,
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            default:
                continue;
            }

            result[meta.id.c_str()] = prop_obj;
        }

        return result;
    }

    namespace {
        [[nodiscard]] std::optional<vis::SplitViewPanelId> parsePanelArg(const std::string& panel) {
            if (panel.empty() || panel == "main")
                return std::nullopt;
            if (panel == "left")
                return vis::SplitViewPanelId::Left;
            if (panel == "right")
                return vis::SplitViewPanelId::Right;
            throw std::invalid_argument("panel must be 'main', 'left', or 'right'");
        }

        [[nodiscard]] std::optional<vis::ViewInfo> viewInfoForPanelArg(const std::string& panel) {
            const auto panel_id = parsePanelArg(panel);
            return panel_id ? vis::get_view_info_for_panel(*panel_id)
                            : vis::get_current_view_info();
        }

        [[nodiscard]] std::string normalizeExportImageFormat(std::string format,
                                                             const std::filesystem::path& path) {
            if (format.empty()) {
                format = path.extension().string();
            }
            if (!format.empty() && format.front() == '.') {
                format.erase(format.begin());
            }
            std::transform(format.begin(), format.end(), format.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });

            if (format == "jpg" || format == "jpeg")
                return "jpg";
            if (format == "png")
                return "png";
            throw std::invalid_argument("format must be 'jpg' or 'png'");
        }

        [[nodiscard]] std::filesystem::path imageExportPathForFormat(std::filesystem::path path,
                                                                     const std::string& format) {
            if (path.empty()) {
                throw std::invalid_argument("export path is empty");
            }

            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });

            if (format == "png") {
                if (ext != ".png") {
                    path.replace_extension(".png");
                }
                return path;
            }

            if (ext != ".jpg" && ext != ".jpeg") {
                path.replace_extension(".jpg");
            }
            return path;
        }

        [[nodiscard]] glm::mat3 viewInfoRotationMatrix(const vis::ViewInfo& view_info) {
            glm::mat3 rotation{};
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    rotation[col][row] = view_info.rotation[static_cast<std::size_t>(row * 3 + col)];
                }
            }
            return rotation;
        }

        [[nodiscard]] std::optional<float> scaledViewInfoOrthoScale(const vis::ViewInfo& view_info,
                                                                    const int target_height) {
            if (!view_info.orthographic) {
                return std::nullopt;
            }
            if (view_info.height <= 0 || target_height <= 0 ||
                !std::isfinite(view_info.ortho_scale) || view_info.ortho_scale <= 0.0f) {
                return std::nullopt;
            }

            const double scale = static_cast<double>(view_info.ortho_scale) *
                                 static_cast<double>(target_height) /
                                 static_cast<double>(view_info.height);
            if (!std::isfinite(scale) || scale <= 0.0) {
                return std::nullopt;
            }
            return static_cast<float>(scale);
        }

        [[nodiscard]] core::Tensor toU8Hwc(core::Tensor image) {
            if (!image.is_valid() || image.ndim() != 3) {
                throw std::runtime_error("viewport export expected a 3D image tensor");
            }
            if (image.shape()[0] <= 4 && image.shape()[2] > 4) {
                image = image.permute({1, 2, 0}).contiguous();
            }
            image = image.to(core::Device::CPU);
            if (image.dtype() != core::DataType::UInt8) {
                image = (image.to(core::DataType::Float32).clamp(0, 1) * 255.0f)
                            .to(core::DataType::UInt8);
            }
            image = image.contiguous();
            if (image.ndim() != 3 || image.shape()[2] < 1 || image.shape()[2] > 4) {
                throw std::runtime_error("viewport export image channels must be in [1..4]");
            }
            return image;
        }

        [[nodiscard]] core::Tensor renderCurrentViewRgb8(const vis::ViewInfo& view_info,
                                                         const int width,
                                                         const int height,
                                                         const std::optional<glm::vec3>& background_color_override) {
            auto image = renderViewThreadSafe(
                viewInfoRotationMatrix(view_info),
                glm::vec3{view_info.translation[0], view_info.translation[1], view_info.translation[2]},
                width,
                height,
                view_info.fov,
                PreviewReadback::UInt8Rgb,
                background_color_override,
                view_info.orthographic,
                scaledViewInfoOrthoScale(view_info, height));
            if (!image || !image->is_valid()) {
                throw std::runtime_error("viewport export render failed");
            }
            return toU8Hwc(std::move(*image));
        }

        using ExportImageResult = std::expected<core::Tensor, std::string>;

        [[nodiscard]] ExportImageResult runExportOnViewerThread(
            std::function<ExportImageResult()> invoke_render) {
            auto* const viewer = get_visualizer();
            if (!viewer || viewer->isOnViewerThread()) {
                return invoke_render();
            }
            if (!viewer->acceptsPostedWork()) {
                return std::unexpected("viewer is not accepting export work");
            }

            nb::gil_scoped_release release;
            return vis::post_work_and_wait(
                [viewer](vis::Visualizer::WorkItem work) { return viewer->postWork(std::move(work)); },
                std::move(invoke_render),
                []() -> ExportImageResult {
                    return std::unexpected("viewport export was cancelled");
                });
        }

        [[nodiscard]] core::Tensor renderCurrentViewExport(const vis::ViewInfo& view_info,
                                                           const int width,
                                                           const int height,
                                                           const vis::ExportPostProcessMode mode) {
            auto result = runExportOnViewerThread([&view_info, width, height, mode]() -> ExportImageResult {
                auto* const viewer = get_visualizer();
                auto* const rendering_manager = viewer ? viewer->getRenderingManager() : nullptr;
                auto* const scene_manager = viewer ? viewer->getSceneManager() : nullptr;
                if (!rendering_manager || !scene_manager) {
                    return std::unexpected("no active viewer is available");
                }
                const vis::RenderingManager::ExportImageRequest request{
                    .rotation = viewInfoRotationMatrix(view_info),
                    .translation = {view_info.translation[0],
                                    view_info.translation[1],
                                    view_info.translation[2]},
                    .focal_length_mm = lfs::rendering::vFovToFocalLength(view_info.fov),
                    .width = width,
                    .height = height,
                    .orthographic_override = view_info.orthographic,
                    .ortho_scale_override = scaledViewInfoOrthoScale(view_info, height),
                    .mode = mode,
                };
                return rendering_manager->renderExportImage(scene_manager, request);
            });
            if (!result) {
                throw std::runtime_error("viewport export failed: " + result.error());
            }
            return std::move(*result);
        }

        // Post-process for images assembled outside renderExportImage (the BW2A
        // transparent fallback): applies the same PPISP correction path.
        [[nodiscard]] core::Tensor applyExportPostProcessThreadSafe(core::Tensor image,
                                                                    const vis::ExportPostProcessMode mode) {
            auto result = runExportOnViewerThread(
                [image = std::move(image), mode]() mutable -> ExportImageResult {
                    auto* const viewer = get_visualizer();
                    auto* const rendering_manager = viewer ? viewer->getRenderingManager() : nullptr;
                    auto* const scene_manager = viewer ? viewer->getSceneManager() : nullptr;
                    if (!rendering_manager || !scene_manager) {
                        return std::unexpected("no active viewer is available");
                    }
                    const auto view_info = viewInfoForPanelArg("main");
                    const vis::ExportPostProcessView view{
                        .rotation = view_info ? viewInfoRotationMatrix(*view_info) : glm::mat3{1.0f},
                        .focal_length_mm =
                            view_info ? lfs::rendering::vFovToFocalLength(view_info->fov) : 0.0f,
                        .equirectangular_view = rendering_manager->getSettings().equirectangular,
                        .controller_predict_size =
                            view_info ? glm::ivec2{view_info->width, view_info->height} : glm::ivec2{0, 0},
                    };
                    return vis::applyExportPostProcess(std::move(image),
                                                       scene_manager,
                                                       rendering_manager->getSettings(),
                                                       rendering_manager->getCurrentCameraId(),
                                                       mode,
                                                       view);
                });
            if (!result) {
                throw std::runtime_error("viewport export post-process failed: " + result.error());
            }
            return std::move(*result);
        }

        [[nodiscard]] core::Tensor recoverAlphaRgba(core::Tensor black_rgb, core::Tensor white_rgb) {
            black_rgb = toU8Hwc(std::move(black_rgb));
            white_rgb = toU8Hwc(std::move(white_rgb));

            if (black_rgb.shape() != white_rgb.shape() || black_rgb.shape()[2] < 3) {
                throw std::runtime_error("transparent viewport export expected matching RGB images");
            }

            const auto height = black_rgb.shape()[0];
            const auto width = black_rgb.shape()[1];
            auto rgba = core::Tensor::empty({height, width, std::size_t{4}},
                                            core::Device::CPU,
                                            core::DataType::UInt8);

            const auto* const black = black_rgb.ptr<uint8_t>();
            const auto* const white = white_rgb.ptr<uint8_t>();
            auto* const out = rgba.ptr<uint8_t>();
            const auto pixel_count = height * width;
            const auto src_channels = black_rgb.shape()[2];

            for (std::size_t i = 0; i < pixel_count; ++i) {
                const auto src = i * src_channels;
                const auto dst = i * std::size_t{4};
                const float delta =
                    (static_cast<float>(white[src]) - static_cast<float>(black[src]) +
                     static_cast<float>(white[src + 1]) - static_cast<float>(black[src + 1]) +
                     static_cast<float>(white[src + 2]) - static_cast<float>(black[src + 2])) /
                    (3.0f * 255.0f);
                const float alpha = std::clamp(1.0f - delta, 0.0f, 1.0f);
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const float recovered = alpha > 1.0e-6f
                                                ? static_cast<float>(black[src + channel]) / alpha
                                                : 0.0f;
                    out[dst + channel] = static_cast<uint8_t>(
                        std::lround(std::clamp(recovered, 0.0f, 255.0f)));
                }
                out[dst + 3] = static_cast<uint8_t>(
                    std::lround(std::clamp(alpha * 255.0f, 0.0f, 255.0f)));
            }

            return rgba;
        }
    } // namespace

    std::optional<PyCameraState> get_camera(const std::string& panel) {
        const auto info = viewInfoForPanelArg(panel);
        if (!info)
            return std::nullopt;

        const auto& r = info->rotation;
        glm::mat3 c2w;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                c2w[j][i] = r[i * 3 + j];

        return PyCameraState{
            .eye = {info->translation[0], info->translation[1], info->translation[2]},
            .target = {info->pivot[0], info->pivot[1], info->pivot[2]},
            .up = {c2w[1].x, c2w[1].y, c2w[1].z},
            .fov = info->fov,
        };
    }

    void set_camera(const std::tuple<float, float, float>& eye,
                    const std::tuple<float, float, float>& target,
                    const std::tuple<float, float, float>& up,
                    const std::string& panel) {
        const vis::SetViewParams params{
            .eye = {std::get<0>(eye), std::get<1>(eye), std::get<2>(eye)},
            .target = {std::get<0>(target), std::get<1>(target), std::get<2>(target)},
            .up = {std::get<0>(up), std::get<1>(up), std::get<2>(up)},
        };
        if (const auto panel_id = parsePanelArg(panel))
            vis::apply_set_view_for_panel(*panel_id, params);
        else
            vis::apply_set_view(params);
    }

    void set_camera_fov(float fov_degrees) {
        vis::apply_set_fov(fov_degrees);
    }

    std::optional<PyRenderSettings> get_render_settings() {
        auto settings = vis::get_render_settings();
        if (!settings)
            return std::nullopt;
        return PyRenderSettings(std::move(*settings));
    }

    nb::dict get_lod_stats() {
        nb::dict result;
        auto* rm = get_rendering_manager();
        if (!rm) {
            result["enabled"] = false;
            result["selected"] = 0;
            result["budget"] = 0;
            result["requested_budget"] = 0;
            result["levels"] = nb::list();
            return result;
        }
        const auto stats = rm->getLodStats();
        result["enabled"] = stats.enabled && stats.has_tree;
        result["active"] = stats.active;
        result["async_ready"] = stats.async_result_ready;
        result["selected"] = stats.selected_splats;
        result["budget"] = stats.max_splats;
        result["requested_budget"] = stats.requested_max_splats;
        result["budget_repair_limit"] = stats.budget_repair_limit;
        result["generation"] = stats.generation;
        result["selection_hash"] = stats.selection_hash;
        result["model_splats"] = stats.model_splats;
        result["full_quality_splats"] = stats.full_quality_splats;
        result["output_size"] = stats.output_size;
        result["frontier_size"] = stats.frontier_size;
        result["leaf_count"] = stats.leaf_count;
        result["budget_limited"] = stats.budget_limited;
        result["threshold_limited"] = stats.threshold_limited;
        result["output_limited"] = stats.output_limited;
        result["budget_fill_active"] = stats.budget_fill_active;
        result["budget_repair_active"] = stats.budget_repair_active;
        result["full_quality_reference"] = stats.full_quality_reference;
        result["transition_active"] = stats.transition_active;
        result["outside_view_nodes"] = stats.outside_view_nodes;
        result["behind_view_nodes"] = stats.behind_view_nodes;
        result["viewport_throttled_nodes"] = stats.viewport_throttled_nodes;
        result["touched_chunks"] = stats.touched_chunks;
        result["resident_chunks"] = stats.resident_chunks;
        result["pixel_scale_limit"] = stats.pixel_scale_limit;
        result["budget_fill_pixel_scale_limit"] = stats.budget_fill_pixel_scale_limit;
        result["min_pixel_scale"] = stats.min_pixel_scale;
        nb::list levels;
        for (const auto& [level, count] : stats.level_histogram) {
            nb::dict item;
            item["level"] = level;
            item["count"] = count;
            levels.append(item);
        }
        result["levels"] = levels;
        return result;
    }

} // namespace lfs::python

namespace {

    constexpr float DEFAULT_FOV = 60.0f;
    lfs::core::Tensor tensor_from_mat3_row_major(const glm::mat3& matrix) {
        auto tensor = lfs::core::Tensor::empty({3, 3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto* ptr = static_cast<float*>(tensor.data_ptr());
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                ptr[row * 3 + col] = matrix[col][row];
            }
        }
        return tensor;
    }

    lfs::core::Tensor tensor_from_vec3(const glm::vec3& value) {
        return lfs::core::Tensor::from_vector({value.x, value.y, value.z}, {3}, lfs::core::Device::CPU);
    }

    float vertical_fov_to_horizontal_fov(float vertical_fov_degrees, int width, int height) {
        if (width <= 0 || height <= 0) {
            return vertical_fov_degrees;
        }

        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const float half_vertical_fov_radians = vertical_fov_degrees * std::numbers::pi_v<float> / 360.0f;
        return std::atan(std::tan(half_vertical_fov_radians) * aspect) * 360.0f / std::numbers::pi_v<float>;
    }

    std::pair<lfs::core::Tensor, lfs::core::Tensor> compute_visualizer_pose(
        const std::tuple<float, float, float>& eye,
        const std::tuple<float, float, float>& target,
        const std::tuple<float, float, float>& up) {
        auto [ex, ey, ez] = eye;
        auto [tx, ty, tz] = target;
        auto [ux, uy, uz] = up;
        const glm::vec3 eye_vec{ex, ey, ez};
        return {
            tensor_from_mat3_row_major(
                lfs::rendering::makeVisualizerLookAtRotation(
                    eye_vec,
                    glm::vec3{tx, ty, tz},
                    glm::vec3{ux, uy, uz}))
                .cuda(),
            tensor_from_vec3(eye_vec).cuda()};
    }

} // namespace

namespace lfs::python {

    std::optional<PyTensor> render_view(const PyTensor& rotation, const PyTensor& translation, int width, int height,
                                        float fov_degrees, const PyTensor* bg_color) {
        (void)bg_color;

        const auto rotation_matrix = tensorToVisualizerRotation(rotation);
        const auto translation_vector = tensorToVisualizerTranslation(translation);
        if (!rotation_matrix || !translation_vector) {
            return std::nullopt;
        }

        auto image = renderViewThreadSafe(
            *rotation_matrix,
            *translation_vector,
            width,
            height,
            fov_degrees,
            PreviewReadback::FloatRgb,
            std::nullopt);
        if (!image || !image->is_valid() || image->ndim() != 3) {
            return std::nullopt;
        }
        if (image->size(0) <= 0 || image->size(1) <= 0 || image->size(2) != 3) {
            return std::nullopt;
        }

        auto output = *image;
        if (output.device() != core::Device::CPU) {
            output = output.cpu();
        }
        if (output.dtype() != core::DataType::Float32) {
            output = output.to(core::DataType::Float32);
        }
        output = output.contiguous();
        return PyTensor(std::move(output), true);
    }

    std::optional<PyTensor> render_view_u8(const PyTensor& rotation, const PyTensor& translation, int width, int height,
                                           float fov_degrees, const PyTensor* bg_color,
                                           std::optional<bool> orthographic,
                                           std::optional<float> ortho_scale) {
        (void)bg_color;

        const auto rotation_matrix = tensorToVisualizerRotation(rotation);
        const auto translation_vector = tensorToVisualizerTranslation(translation);
        if (!rotation_matrix || !translation_vector) {
            return std::nullopt;
        }

        auto image = renderViewThreadSafe(
            *rotation_matrix,
            *translation_vector,
            width,
            height,
            fov_degrees,
            PreviewReadback::UInt8Rgb,
            std::nullopt,
            orthographic,
            ortho_scale);
        if (!image || !image->is_valid() || image->ndim() != 3) {
            return std::nullopt;
        }
        if (image->size(0) <= 0 || image->size(1) <= 0 || image->size(2) != 3) {
            return std::nullopt;
        }

        auto output = *image;
        if (output.device() != core::Device::CPU) {
            output = output.cpu();
        }
        if (output.dtype() != core::DataType::UInt8) {
            output = output.to(core::DataType::UInt8);
        }
        output = output.contiguous();
        return PyTensor(std::move(output), true);
    }

    std::optional<PyTensor> compute_screen_positions(const PyTensor& rotation, const PyTensor& translation, int width,
                                                     int height, float fov_degrees) {
        (void)rotation;
        (void)translation;
        (void)width;
        (void)height;
        (void)fov_degrees;
        return std::nullopt;
    }

    std::optional<PyViewInfo> get_current_view(const std::string& panel) {
        const auto view_info = viewInfoForPanelArg(panel);
        if (!view_info)
            return std::nullopt;

        auto R = core::Tensor::empty({3, 3}, core::Device::CPU, core::DataType::Float32);
        auto T = core::Tensor::empty({3}, core::Device::CPU, core::DataType::Float32);

        std::memcpy(R.data_ptr(), view_info->rotation.data(), 9 * sizeof(float));
        std::memcpy(T.data_ptr(), view_info->translation.data(), 3 * sizeof(float));

        return PyViewInfo{
            .rotation = PyTensor(R.cuda(), true),
            .translation = PyTensor(T.cuda(), true),
            .width = view_info->width,
            .height = view_info->height,
            .fov_x = vertical_fov_to_horizontal_fov(view_info->fov, view_info->width, view_info->height),
            .fov_y = view_info->fov,
            .orthographic = view_info->orthographic,
            .ortho_scale = view_info->ortho_scale,
        };
    }

    std::optional<PyViewportRender> get_viewport_render() {
        return toPyViewportRender(vis::get_viewport_render(), false);
    }

    std::optional<PyViewportRender> capture_viewport() {
        return toPyViewportRender(captureViewportRenderThreadSafe(), true);
    }

    nb::dict export_viewport_image(const std::string& path,
                                   const std::string& format,
                                   int width,
                                   int height,
                                   const bool transparent,
                                   const int jpeg_quality) {
        auto output_path = core::utf8_to_path(path);
        const auto normalized_format = normalizeExportImageFormat(format, output_path);
        if (transparent && normalized_format != "png") {
            throw std::invalid_argument("transparent viewport export requires PNG format");
        }
        output_path = imageExportPathForFormat(std::move(output_path), normalized_format);

        auto view_info = viewInfoForPanelArg("main");
        if (!view_info) {
            throw std::runtime_error("no active viewport is available");
        }

        const int current_width = std::max(1, view_info->width);
        const int current_height = std::max(1, view_info->height);
        int target_width = width;
        int target_height = height;

        core::Tensor image;
        if (!transparent && target_width <= 0 && target_height <= 0) {
            auto captured = viewportRenderImageHwc(captureViewportRenderThreadSafe(), true);
            if (!captured || !captured->is_valid()) {
                throw std::runtime_error("viewport capture failed");
            }
            image = toU8Hwc(std::move(*captured));
            target_height = static_cast<int>(image.shape()[0]);
            target_width = static_cast<int>(image.shape()[1]);
        } else {
            if (target_width <= 0 && target_height <= 0) {
                target_width = current_width;
                target_height = current_height;
            } else if (target_width <= 0) {
                target_width = static_cast<int>(std::lround(
                    static_cast<double>(target_height) *
                    static_cast<double>(current_width) /
                    static_cast<double>(current_height)));
            } else if (target_height <= 0) {
                target_height = static_cast<int>(std::lround(
                    static_cast<double>(target_width) *
                    static_cast<double>(current_height) /
                    static_cast<double>(current_width)));
            }
            target_width = std::max(1, target_width);
            target_height = std::max(1, target_height);

            if (transparent) {
                try {
                    image = renderCurrentViewExport(
                        *view_info, target_width, target_height, vis::ExportPostProcessMode::Transparent);
                } catch (const std::exception& e) {
                    LOG_DEBUG("transparent viewport export direct RGBA render failed, falling back to BW2A: {}", e.what());
                    auto black = renderCurrentViewRgb8(
                        *view_info,
                        target_width,
                        target_height,
                        std::optional<glm::vec3>{glm::vec3{0.0f}});
                    auto white = renderCurrentViewRgb8(
                        *view_info,
                        target_width,
                        target_height,
                        std::optional<glm::vec3>{glm::vec3{1.0f}});
                    image = applyExportPostProcessThreadSafe(
                        recoverAlphaRgba(std::move(black), std::move(white)),
                        vis::ExportPostProcessMode::Transparent);
                }
            } else {
                const auto render_settings = vis::get_render_settings();
                const bool export_hdr_environment =
                    render_settings &&
                    render_settings->environment_mode ==
                        static_cast<int>(vis::EnvironmentBackgroundMode::Equirectangular) &&
                    !render_settings->environment_map_path.empty();
                image = renderCurrentViewExport(
                    *view_info,
                    target_width,
                    target_height,
                    export_hdr_environment ? vis::ExportPostProcessMode::EnvironmentComposite
                                           : vis::ExportPostProcessMode::Opaque);
            }
        }

        core::save_image_u8(output_path, image, jpeg_quality);

        nb::dict result;
        result["path"] = core::path_to_utf8(output_path);
        result["width"] = static_cast<int>(image.shape()[1]);
        result["height"] = static_cast<int>(image.shape()[0]);
        result["channels"] = static_cast<int>(image.shape()[2]);
        result["transparent"] = transparent;
        result["format"] = normalized_format;
        return result;
    }

    std::tuple<PyTensor, PyTensor> look_at(const std::tuple<float, float, float>& eye,
                                           const std::tuple<float, float, float>& target,
                                           const std::tuple<float, float, float>& up) {
        auto [R, T] = compute_visualizer_pose(eye, target, up);
        return {PyTensor(std::move(R), true), PyTensor(std::move(T), true)};
    }

    std::optional<PyTensor> render_at(const std::tuple<float, float, float>& eye,
                                      const std::tuple<float, float, float>& target, int width, int height,
                                      float fov_degrees, const std::tuple<float, float, float>& up,
                                      const PyTensor* bg_color) {
        auto [R, T] = compute_visualizer_pose(eye, target, up);
        PyTensor rotation(std::move(R), true);
        PyTensor translation(std::move(T), true);
        return render_view(rotation, translation, width, height, fov_degrees, bg_color);
    }

    std::optional<PyTensor> render_asset_preview(
        const std::string& path,
        const int width,
        const int height,
        const float focal_length_mm) {
        auto image = renderAssetPreviewThreadSafe(path, width, height, focal_length_mm);
        if (!image) {
            return std::nullopt;
        }
        return PyTensor(std::move(*image), true);
    }

    std::optional<PyTensor> render_asset_preview_from_camera(
        const std::string& path,
        const std::tuple<float, float, float>& eye,
        const std::tuple<float, float, float>& target,
        const int width,
        const int height,
        const float focal_length_mm,
        const std::tuple<float, float, float>& up) {
        const glm::vec3 eye_vec{
            std::get<0>(eye), std::get<1>(eye), std::get<2>(eye)};
        const glm::vec3 target_vec{
            std::get<0>(target), std::get<1>(target), std::get<2>(target)};
        const glm::vec3 up_vec{
            std::get<0>(up), std::get<1>(up), std::get<2>(up)};
        const glm::mat3 rotation = lfs::rendering::makeVisualizerLookAtRotation(
            eye_vec, target_vec, up_vec);
        auto image = renderAssetPreviewThreadSafe(
            path, width, height, focal_length_mm, &rotation, &eye_vec);
        if (!image) {
            return std::nullopt;
        }
        return PyTensor(std::move(*image), true);
    }

    void register_rendering(nb::module_& m) {
        nb::class_<PyViewInfo>(m, "ViewInfo")
            .def_ro("rotation", &PyViewInfo::rotation)
            .def_ro("translation", &PyViewInfo::translation)
            .def_ro("width", &PyViewInfo::width)
            .def_ro("height", &PyViewInfo::height)
            .def_ro("fov_x", &PyViewInfo::fov_x)
            .def_ro("fov_y", &PyViewInfo::fov_y)
            .def_ro("orthographic", &PyViewInfo::orthographic)
            .def_ro("ortho_scale", &PyViewInfo::ortho_scale)
            .def_prop_ro(
                "ortho_view_extent_world", [](const PyViewInfo& self) -> float {
                    if (!self.orthographic || self.ortho_scale <= 0.0f)
                        return 0.0f;
                    return static_cast<float>(self.height) / self.ortho_scale;
                },
                "Vertical view extent in world units (Blender-compatible orthographic scale). Larger when zoomed out, smaller when zoomed in.")
            .def_prop_ro(
                "position", [](const PyViewInfo& self) -> std::tuple<float, float, float> {
                    auto t = self.translation.tensor().cpu();
                    auto acc = t.accessor<float, 1>();
                    return {acc(0), acc(1), acc(2)}; }, "Camera position as (x, y, z) tuple");

        nb::class_<PyViewportRender>(m, "ViewportRender")
            .def_ro("image", &PyViewportRender::image)
            .def_ro("screen_positions", &PyViewportRender::screen_positions);

        m.def("get_viewport_render", &get_viewport_render,
              "Get the most recently captured CPU-visible viewport render if available (does not force GPU readback)");

        m.def("capture_viewport", &capture_viewport,
              "Capture viewport render explicitly (may read back from GPU; clones data, safe to use from background threads)");

        m.def("export_viewport_image",
              &export_viewport_image,
              nb::arg("path"),
              nb::arg("format") = std::string{},
              nb::arg("width") = 0,
              nb::arg("height") = 0,
              nb::arg("transparent") = false,
              nb::arg("jpeg_quality") = 95,
              R"doc(
Export the active viewport image to PNG or JPEG.

Args:
    path: Output path. The selected format extension is enforced when needed.
    format: 'png', 'jpg', or empty to infer from path.
    width: Target width in pixels. If zero with a positive height, preserves viewport aspect.
    height: Target height in pixels. If both dimensions are zero, captures the current viewport.
    transparent: For PNG only, export straight RGBA from the preview renderer.
    jpeg_quality: JPEG compression quality in [1, 100].

Returns:
    Dict with path, width, height, channels, format, and transparent.
)doc");

        m.def(
            "render_view",
            [](const PyTensor& rotation, const PyTensor& translation, int width, int height,
               float fov_degrees, const PyTensor* bg_color, bool with_depth,
               const std::string& depth_mode) -> nb::object {
                if (!with_depth) {
                    auto image = render_view(rotation, translation, width, height, fov_degrees, bg_color);
                    if (!image) {
                        return nb::none();
                    }
                    return nb::cast(std::move(*image));
                }
                const auto rotation_matrix = tensorToVisualizerRotation(rotation);
                const auto translation_vector = tensorToVisualizerTranslation(translation);
                if (!rotation_matrix || !translation_vector) {
                    return nb::none();
                }
                const bool expected_depth = depth_mode == "expected";
                auto rgbd = renderViewAndDepthThreadSafe(
                    *rotation_matrix, *translation_vector, width, height, fov_degrees, expected_depth);
                if (!rgbd) {
                    return nb::none();
                }
                return nb::make_tuple(
                    PyTensor(std::move(rgbd->first), true),
                    PyTensor(std::move(rgbd->second), true));
            },
            nb::arg("rotation"), nb::arg("translation"), nb::arg("width"), nb::arg("height"),
            nb::arg("fov") = DEFAULT_FOV, nb::arg("bg_color") = nb::none(), nb::arg("with_depth") = false,
            nb::arg("depth_mode") = std::string("median"),
            R"doc(
Render scene from arbitrary camera parameters.

Args:
    rotation: [3, 3] camera-to-world rotation in visualizer coordinates
    translation: [3] camera position in visualizer world coordinates
    width: Render width in pixels
    height: Render height in pixels
    fov: Vertical field of view in degrees (default: 60)
    bg_color: Accepted for compatibility; the Vulkan preview path uses current render settings
    with_depth: If True, also return the per-pixel linear depth from the same render
    depth_mode: "median" (default) = depth at 50% transmittance (sharp, undefined where
        coverage < 50%); "expected" = alpha-weighted depth (dense/hole-free, softer at edges)

Returns:
    with_depth=False: CPU Tensor [H, W, 3] RGB image
    with_depth=True: tuple (image [H, W, 3], depth [H, W]) of CPU float tensors
    or None if no active visualizer scene is available
)doc");

        m.def("render_view_u8", &render_view_u8, nb::arg("rotation"), nb::arg("translation"), nb::arg("width"), nb::arg("height"),
              nb::arg("fov") = DEFAULT_FOV, nb::arg("bg_color") = nb::none(),
              nb::arg("orthographic") = nb::none(), nb::arg("ortho_scale") = nb::none(),
              R"doc(
Render scene from arbitrary camera parameters as an 8-bit RGB image.

Args:
    rotation: [3, 3] camera-to-world rotation in visualizer coordinates
    translation: [3] camera position in visualizer world coordinates
    width: Render width in pixels
    height: Render height in pixels
    fov: Vertical field of view in degrees (default: 60)
    bg_color: Accepted for compatibility; the Vulkan preview path uses current render settings
    orthographic: Optional projection override. None uses current render settings.
    ortho_scale: Optional orthographic pixels-per-world-unit override.

Returns:
    CPU uint8 Tensor [H, W, 3] RGB image, or None if no active visualizer scene is available
)doc");

        m.def("compute_screen_positions", &compute_screen_positions, nb::arg("rotation"), nb::arg("translation"),
              nb::arg("width"), nb::arg("height"), nb::arg("fov") = DEFAULT_FOV,
              R"doc(
Compute screen positions of all Gaussians for a given camera view.

Args:
    rotation: [3, 3] camera-to-world rotation in visualizer coordinates
    translation: [3] camera position in visualizer world coordinates
    width: Viewport width in pixels
    height: Viewport height in pixels
    fov: Vertical field of view in degrees (default: 60)

Returns:
    Tensor [N, 2] with (x, y) pixel coordinates for each Gaussian
)doc");

        m.def("get_current_view", &get_current_view, nb::arg("panel") = std::string{"main"},
              R"doc(
Get current viewport camera pose (None if not available).

Args:
    panel: 'main' (default) returns the focused viewport, 'left'/'right' returns the
        per-panel camera. In independent split-view mode, the right panel has its own
        camera; otherwise both panels share the main camera.
)doc");

        nb::class_<PyCameraState>(m, "CameraState")
            .def_ro("eye", &PyCameraState::eye)
            .def_ro("target", &PyCameraState::target)
            .def_ro("up", &PyCameraState::up)
            .def_ro("fov", &PyCameraState::fov);

        m.def("get_camera", &get_camera, nb::arg("panel") = std::string{"main"},
              R"doc(
Get current viewport camera state (eye, target, up, fov) or None if unavailable.

Args:
    panel: 'main' (default), 'left', or 'right'. 'left'/'right' return the per-panel
        camera in independent split-view mode; otherwise both panels share the main camera.
)doc");

        m.def("set_camera", &set_camera,
              nb::arg("eye"), nb::arg("target"),
              nb::arg("up") = std::make_tuple(0.0f, 1.0f, 0.0f),
              nb::arg("panel") = std::string{"main"},
              R"doc(
Move the viewport camera to look from eye toward target.

Args:
    eye: camera position (x, y, z).
    target: look-at target (x, y, z).
    up: world up vector (default (0, 1, 0)).
    panel: 'main' (default), 'left', or 'right'. In independent split-view mode the right
        panel can be moved independently; otherwise this falls back to the main camera.
)doc");

        m.def("set_camera_fov", &set_camera_fov,
              nb::arg("fov"),
              "Set viewport field of view in degrees");

        m.def("look_at", &look_at, nb::arg("eye"), nb::arg("target"),
              nb::arg("up") = std::make_tuple(0.0f, 1.0f, 0.0f),
              "Compute a visualizer camera pose tuple (rotation, translation) for render_view from eye/target.");

        m.def("render_at", &render_at, nb::arg("eye"), nb::arg("target"), nb::arg("width"), nb::arg("height"),
              nb::arg("fov") = DEFAULT_FOV, nb::arg("up") = std::make_tuple(0.0f, 1.0f, 0.0f),
              nb::arg("bg_color") = nb::none(),
              "Render scene from eye looking at target. Returns [H,W,3] RGB tensor or None.");

        m.def("render_asset_preview", &render_asset_preview,
              nb::arg("path"), nb::arg("width") = 512, nb::arg("height") = 224,
              nb::arg("focal_length_mm") = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM,
              "Render an asset from the framed home camera into an offscreen thumbnail without mutating the live scene.");

        m.def("render_asset_preview_from_camera", &render_asset_preview_from_camera,
              nb::arg("path"), nb::arg("eye"), nb::arg("target"),
              nb::arg("width") = 512, nb::arg("height") = 224,
              nb::arg("focal_length_mm") = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM,
              nb::arg("up") = std::make_tuple(0.0f, 1.0f, 0.0f),
              "Render an asset from a custom camera pose into an offscreen thumbnail without mutating the live scene.");

        m.def(
            "get_render_scene", []() -> std::optional<PyScene> {
                auto* scene = get_render_scene();
                if (!scene)
                    return std::nullopt;
                return PyScene(scene);
            },
            "Get the current render scene (None if not available)");

        register_render_settings_properties();

        nb::class_<PyRenderSettings>(m, "RenderSettings")
            .def_prop_ro("__property_group__", &PyRenderSettings::property_group)
            .def("get", &PyRenderSettings::get, nb::arg("name"), "Get property value by name")
            .def("set", &PyRenderSettings::set, nb::arg("name"), nb::arg("value"), "Set property value by name")
            .def("prop_info", &PyRenderSettings::prop_info, nb::arg("name"))
            .def("get_all_properties", &PyRenderSettings::get_all_properties,
                 "Get all property descriptors as Python Property objects")
            .def(
                "__getattr__",
                [](PyRenderSettings& self, const std::string& name) -> nb::object {
                    if (!self.has_prop(name)) {
                        throw nb::attribute_error(("RenderSettings has no attribute '" + name + "'").c_str());
                    }
                    return self.prop_getattr(name);
                })
            .def(
                "__setattr__",
                [](PyRenderSettings& self, const std::string& name, nb::object value) {
                    if (!self.has_prop(name)) {
                        throw nb::attribute_error(("Cannot set attribute '" + name + "'").c_str());
                    }
                    self.prop_setattr(name, value);
                })
            .def("__dir__", &PyRenderSettings::python_dir);

        m.def("get_render_settings", &get_render_settings);
        m.def("get_lod_stats", &get_lod_stats,
              "Get LOD statistics: {enabled, selected, budget, levels:[{level, count}, ...]}");
    }

} // namespace lfs::python
