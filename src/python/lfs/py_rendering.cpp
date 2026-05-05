/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_rendering.hpp"
#include "core/camera.hpp"
#include "core/property_registry.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "py_scene.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/gs_rasterizer_tensor.hpp"
#include "rendering/image_layout.hpp"
#include "training/dataset.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/visualizer.hpp"

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <future>
#include <numbers>

#include <glm/glm.hpp>

namespace nb = nanobind;

namespace lfs::python {

    namespace {
        [[nodiscard]] std::optional<PyViewportRender> toPyViewportRender(
            const std::optional<vis::ViewportRender>& render,
            const bool clone_for_async) {
            if (!render || !render->image)
                return std::nullopt;

            auto image = clone_for_async ? render->image->clone() : *render->image;
            image = rendering::flipImageVertical(image, rendering::ImageLayout::CHW);
            image = image.permute({1, 2, 0});

            std::optional<PyTensor> screen_pos;
            if (render->screen_positions) {
                auto screen_positions = clone_for_async
                                            ? render->screen_positions->clone()
                                            : *render->screen_positions;
                screen_pos = PyTensor(std::move(screen_positions), true);
            }

            return PyViewportRender{
                .image = PyTensor(std::move(image), true),
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

            auto promise = std::make_shared<std::promise<std::optional<vis::ViewportRender>>>();
            auto future = promise->get_future();
            auto completed = std::make_shared<std::atomic_bool>(false);

            auto finish = [promise, completed](std::optional<vis::ViewportRender> result) mutable {
                if (!completed->exchange(true)) {
                    promise->set_value(std::move(result));
                }
            };

            const bool posted = viewer->postWork(vis::Visualizer::WorkItem{
                .run =
                    [invoke_capture, finish]() mutable {
                        finish(invoke_capture());
                    },
                .cancel =
                    [finish]() mutable {
                        finish(std::nullopt);
                    }});
            if (!posted) {
                return std::nullopt;
            }

            nb::gil_scoped_release release;
            return future.get();
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
                     {{"FastGS", "fast_gs", 0}, {"3DGUT", "3dgut", 1}, {"VkSplat", "vksplat", 2}}, 0);
        add_bool(&Proxy::gut, "gut", "GUT Mode", "Enable GUT rendering mode", false);
        add_bool(&Proxy::mip_filter, "mip_filter", "Mip Filter", "Enable mip-map filtering", false);
        add_float(&Proxy::render_scale, "render_scale", "Render Scale", "Render resolution scale", 1.0, 0.25, 1.0);
        add_int_enum(&Proxy::camera_metrics_mode, "camera_metrics_mode", "Camera Metrics",
                     "Compute metrics when jumping to a source camera",
                     {{"Off", "OFF", 0}, {"PSNR", "PSNR", 1}, {"PSNR + SSIM", "PSNR_SSIM", 2}}, 0);

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

} // namespace lfs::python

namespace {

    constexpr float DEFAULT_FOV = 60.0f;
    constexpr float DEFAULT_SCALE_THRESHOLD = 0.01f;

    struct SceneRenderInputs {
        std::unique_ptr<lfs::core::Tensor> model_transforms;
        std::unique_ptr<lfs::core::Tensor> transform_indices;
        std::vector<bool> node_visibility_mask;
    };

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

    lfs::core::Tensor tensor_from_mat4_row_major(const glm::mat4& matrix) {
        auto tensor = lfs::core::Tensor::empty({4, 4}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto* ptr = static_cast<float*>(tensor.data_ptr());
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                ptr[row * 4 + col] = matrix[col][row];
            }
        }
        return tensor;
    }

    glm::mat3 mat3_from_row_major_tensor(const lfs::core::Tensor& tensor) {
        const auto cpu = tensor.cpu().contiguous();
        return lfs::rendering::mat3FromRowMajor3x3(static_cast<const float*>(cpu.data_ptr()));
    }

    glm::vec3 vec3_from_tensor(const lfs::core::Tensor& tensor) {
        const auto cpu = tensor.cpu().contiguous();
        const auto* ptr = static_cast<const float*>(cpu.data_ptr());
        return {ptr[0], ptr[1], ptr[2]};
    }

    float fov_to_focal(float fov_degrees, int pixels) {
        return static_cast<float>(pixels) / (2.0f * std::tan(fov_degrees * std::numbers::pi_v<float> / 360.0f));
    }

    float vertical_fov_to_horizontal_fov(float vertical_fov_degrees, int width, int height) {
        if (width <= 0 || height <= 0) {
            return vertical_fov_degrees;
        }

        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const float half_vertical_fov_radians = vertical_fov_degrees * std::numbers::pi_v<float> / 360.0f;
        return std::atan(std::tan(half_vertical_fov_radians) * aspect) * 360.0f / std::numbers::pi_v<float>;
    }

    std::unique_ptr<lfs::core::Camera> create_camera_from_w2c(const glm::mat3& rotation,
                                                              const glm::vec3& translation,
                                                              int width,
                                                              int height,
                                                              float fov_degrees) {
        const float focal = fov_to_focal(fov_degrees, height);
        const float cx = static_cast<float>(width) / 2.0f;
        const float cy = static_cast<float>(height) / 2.0f;

        auto R = tensor_from_mat3_row_major(rotation);
        auto T = tensor_from_vec3(translation);
        auto radial = lfs::core::Tensor::zeros({6}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        auto tangential = lfs::core::Tensor::zeros({2}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        auto camera =
            std::make_unique<lfs::core::Camera>(std::move(R), std::move(T), focal, focal, cx, cy, std::move(radial),
                                                std::move(tangential), lfs::core::CameraModelType::PINHOLE,
                                                "virtual_camera", "", "", width, height, -1);
        camera->set_image_dimensions(width, height);
        return camera;
    }

    std::unique_ptr<lfs::core::Camera> create_camera_from_visualizer_pose(const glm::mat3& visualizer_rotation,
                                                                          const glm::vec3& visualizer_translation,
                                                                          int width,
                                                                          int height,
                                                                          float fov_degrees) {
        const glm::mat3 raster_camera_to_world =
            lfs::rendering::rasterCameraToWorldFromVisualizerRotation(visualizer_rotation);
        const glm::mat3 world_to_camera = glm::transpose(raster_camera_to_world);
        return create_camera_from_w2c(
            world_to_camera,
            -world_to_camera * visualizer_translation,
            width,
            height,
            fov_degrees);
    }

    std::unique_ptr<lfs::core::Camera> create_camera_from_visualizer_pose(const lfs::core::Tensor& rotation,
                                                                          const lfs::core::Tensor& translation,
                                                                          int width,
                                                                          int height,
                                                                          float fov_degrees) {
        return create_camera_from_visualizer_pose(
            mat3_from_row_major_tensor(rotation),
            vec3_from_tensor(translation),
            width,
            height,
            fov_degrees);
    }

    lfs::core::SplatData* get_model(lfs::core::Scene* scene) {
        return scene ? const_cast<lfs::core::SplatData*>(scene->getCombinedModel()) : nullptr;
    }

    SceneRenderInputs get_scene_render_inputs(lfs::core::Scene* scene, const size_t gaussian_count) {
        SceneRenderInputs inputs;
        if (!scene) {
            return inputs;
        }

        auto model_transforms = scene->getVisibleNodeTransforms();
        for (auto& transform : model_transforms) {
            transform = lfs::rendering::dataWorldTransformToVisualizerWorld(transform);
        }

        if (!model_transforms.empty()) {
            std::vector<float> transform_data(model_transforms.size() * 16);
            for (size_t i = 0; i < model_transforms.size(); ++i) {
                const auto transform_tensor = tensor_from_mat4_row_major(model_transforms[i]);
                std::memcpy(transform_data.data() + i * 16, transform_tensor.data_ptr(), 16 * sizeof(float));
            }
            inputs.model_transforms = std::make_unique<lfs::core::Tensor>(
                lfs::core::Tensor::from_vector(
                    transform_data,
                    {model_transforms.size(), 4, 4},
                    lfs::core::Device::CPU)
                    .cuda());
        }

        if (auto transform_indices = scene->getTransformIndices();
            transform_indices && transform_indices->is_valid() && transform_indices->numel() == gaussian_count) {
            inputs.transform_indices = std::make_unique<lfs::core::Tensor>(
                transform_indices->device() == lfs::core::Device::CUDA ? *transform_indices : transform_indices->cuda());
        }

        inputs.node_visibility_mask = scene->getNodeVisibilityMask();
        return inputs;
    }

    std::tuple<lfs::core::Tensor, lfs::core::Tensor> rasterize_scene(const lfs::core::Camera& camera,
                                                                     const lfs::core::SplatData& model,
                                                                     const lfs::core::Tensor& bg_color,
                                                                     const SceneRenderInputs& scene_inputs,
                                                                     lfs::core::Tensor* screen_positions_out = nullptr) {
        return lfs::rendering::rasterize_tensor(
            camera,
            model,
            bg_color,
            -1,
            false,
            DEFAULT_SCALE_THRESHOLD,
            scene_inputs.model_transforms.get(),
            scene_inputs.transform_indices.get(),
            nullptr,
            screen_positions_out,
            false,
            0.0f,
            0.0f,
            0.0f,
            true,
            nullptr,
            false,
            0.0f,
            false,
            nullptr,
            nullptr,
            nullptr,
            false,
            false,
            -1,
            nullptr,
            nullptr,
            false,
            false,
            -1,
            nullptr,
            nullptr,
            nullptr,
            false,
            nullptr,
            nullptr,
            -1,
            lfs::rendering::DEFAULT_FAR_PLANE,
            {},
            false,
            scene_inputs.node_visibility_mask);
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
        auto* scene = get_render_scene();
        auto* model = get_model(scene);
        if (!model)
            return std::nullopt;

        auto camera = create_camera_from_visualizer_pose(rotation.tensor(), translation.tensor(), width, height, fov_degrees);
        auto scene_inputs = get_scene_render_inputs(scene, model->size());

        const auto bg = bg_color ? bg_color->tensor().clone()
                                 : core::Tensor::zeros({3}, core::Device::CUDA, core::DataType::Float32);

        auto [image, alpha] = rasterize_scene(*camera, *model, bg, scene_inputs);
        image = rendering::flipImageVertical(image, rendering::ImageLayout::CHW);
        return PyTensor(image.permute({1, 2, 0}), true);
    }

    std::optional<PyTensor> compute_screen_positions(const PyTensor& rotation, const PyTensor& translation, int width,
                                                     int height, float fov_degrees) {
        auto* scene = get_render_scene();
        auto* model = get_model(scene);
        if (!model)
            return std::nullopt;

        auto camera = create_camera_from_visualizer_pose(rotation.tensor(), translation.tensor(), width, height, fov_degrees);
        auto scene_inputs = get_scene_render_inputs(scene, model->size());
        const auto bg = core::Tensor::zeros({3}, core::Device::CUDA, core::DataType::Float32);

        core::Tensor screen_positions;
        rasterize_scene(*camera, *model, bg, scene_inputs, &screen_positions);

        return PyTensor(std::move(screen_positions), true);
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
            .def_prop_ro("position", [](const PyViewInfo& self) -> std::tuple<float, float, float> {
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

        m.def("render_view", &render_view, nb::arg("rotation"), nb::arg("translation"), nb::arg("width"), nb::arg("height"),
              nb::arg("fov") = DEFAULT_FOV, nb::arg("bg_color") = nb::none(),
              R"doc(
Render scene from arbitrary camera parameters.

Args:
    rotation: [3, 3] camera-to-world rotation in visualizer coordinates
    translation: [3] camera position in visualizer world coordinates
    width: Render width in pixels
    height: Render height in pixels
    fov: Vertical field of view in degrees (default: 60)
    bg_color: Optional [3] RGB background color

Returns:
    Tensor [H, W, 3] RGB image on CUDA, or None if scene not available
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
    }

} // namespace lfs::python
