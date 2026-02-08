/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_rendering.hpp"
#include "core/camera.hpp"
#include "core/property_registry.hpp"
#include "core/tensor.hpp"
#include "py_scene.hpp"
#include "python/python_runtime.hpp"
#include "rendering/gs_rasterizer_tensor.hpp"
#include "training/dataset.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/scene/scene.hpp"

#include <cmath>
#include <cstring>

namespace nb = nanobind;

namespace lfs::python {

    void set_render_scene_context(vis::Scene* scene) {
        set_scene_for_python(scene);
    }

    vis::Scene* get_render_scene() {
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

        // Background
        add_color3(&Proxy::background_color, "background_color", "Color", "Viewport background color", {0.0, 0.0, 0.0});

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
                 "Show camera frustum wireframes", true);
        add_float(&Proxy::camera_frustum_scale, "camera_frustum_scale", "Scale", "Camera frustum display scale", 0.25,
                  0.01, 10.0);

        // Point Cloud Mode
        add_bool(&Proxy::point_cloud_mode, "point_cloud_mode", "Point Cloud Mode",
                 "Render as point cloud instead of splats", false);
        add_float(&Proxy::voxel_size, "voxel_size", "Point Size", "Point size in point cloud mode", 0.03, 0.001, 0.1);

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

        // View Settings
        add_float(&Proxy::focal_length_mm, "focal_length_mm", "Focal Length", "Focal length in mm", 35.0, 10.0, 200.0);
        add_int_enum(&Proxy::sh_degree, "sh_degree", "SH Degree", "Spherical harmonics degree",
                     {{"0", "0", 0}, {"1", "1", 1}, {"2", "2", 2}, {"3", "3", 3}}, 3);
        add_bool(&Proxy::equirectangular, "equirectangular", "Equirectangular", "Equirectangular projection mode",
                 false);
        add_bool(&Proxy::gut, "gut", "GUT Mode", "Enable GUT rendering mode", false);
        add_bool(&Proxy::mip_filter, "mip_filter", "Mip Filter", "Enable mip-map filtering", false);
        add_float(&Proxy::render_scale, "render_scale", "Render Scale", "Render resolution scale", 1.0, 0.25, 1.0);

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

        PropertyRegistry::instance().register_group(std::move(group));
    }

    PyRenderSettings::PyRenderSettings(vis::RenderSettingsProxy settings)
        : settings_(std::move(settings)),
          prop_(&settings_, "render_settings") {}

    void PyRenderSettings::set(const std::string& name, nb::object value) {
        prop_.setattr(name, value);
        vis::update_render_settings(settings_);
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

    float fov_to_focal(float fov_degrees, int pixels) {
        return static_cast<float>(pixels) / (2.0f * std::tan(fov_degrees * M_PI / 360.0f));
    }

    std::unique_ptr<lfs::core::Camera> create_camera(const lfs::core::Tensor& R, const lfs::core::Tensor& T, int width,
                                                     int height, float fov_degrees) {
        const float focal = fov_to_focal(fov_degrees, width);
        const float cx = static_cast<float>(width) / 2.0f;
        const float cy = static_cast<float>(height) / 2.0f;

        auto radial = lfs::core::Tensor::zeros({6}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        auto tangential = lfs::core::Tensor::zeros({2}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        auto camera =
            std::make_unique<lfs::core::Camera>(R.clone(), T.clone(), focal, focal, cx, cy, std::move(radial),
                                                std::move(tangential), lfs::core::CameraModelType::PINHOLE,
                                                "virtual_camera", "", "", width, height, -1);
        camera->set_image_dimensions(width, height);
        return camera;
    }

    lfs::core::SplatData* get_model(lfs::vis::Scene* scene) {
        return scene ? const_cast<lfs::core::SplatData*>(scene->getCombinedModel()) : nullptr;
    }

} // namespace

namespace lfs::python {

    std::optional<PyTensor> render_view(const PyTensor& rotation, const PyTensor& translation, int width, int height,
                                        float fov_degrees, const PyTensor* bg_color) {
        auto* scene = get_render_scene();
        auto* model = get_model(scene);
        if (!model)
            return std::nullopt;

        auto camera = create_camera(rotation.tensor(), translation.tensor(), width, height, fov_degrees);

        const auto bg = bg_color ? bg_color->tensor().clone()
                                 : core::Tensor::zeros({3}, core::Device::CUDA, core::DataType::Float32);

        auto [image, alpha] = rendering::rasterize_tensor(*camera, *model, bg);
        return PyTensor(image.permute({1, 2, 0}), true);
    }

    std::optional<PyTensor> compute_screen_positions(const PyTensor& rotation, const PyTensor& translation, int width,
                                                     int height, float fov_degrees) {
        auto* scene = get_render_scene();
        auto* model = get_model(scene);
        if (!model)
            return std::nullopt;

        auto camera = create_camera(rotation.tensor(), translation.tensor(), width, height, fov_degrees);
        const auto bg = core::Tensor::zeros({3}, core::Device::CUDA, core::DataType::Float32);

        core::Tensor screen_positions;
        rendering::rasterize_tensor(*camera, *model, bg, false, DEFAULT_SCALE_THRESHOLD, nullptr, nullptr, nullptr,
                                    &screen_positions);

        return PyTensor(std::move(screen_positions), true);
    }

    std::optional<PyViewInfo> get_current_view() {
        const auto view_info = vis::get_current_view_info();
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
            .fov_x = view_info->fov,
            .fov_y = view_info->fov,
        };
    }

    std::optional<PyViewportRender> get_viewport_render() {
        const auto render = vis::get_viewport_render();
        if (!render || !render->image)
            return std::nullopt;

        // Image is [3, H, W], permute to [H, W, 3] for Python
        auto image = render->image->permute({1, 2, 0});

        std::optional<PyTensor> screen_pos;
        if (render->screen_positions) {
            screen_pos = PyTensor(*render->screen_positions, true);
        }

        return PyViewportRender{
            .image = PyTensor(std::move(image), true),
            .screen_positions = std::move(screen_pos),
        };
    }

    std::optional<PyViewportRender> capture_viewport() {
        const auto render = vis::get_viewport_render();
        if (!render || !render->image)
            return std::nullopt;

        // Clone tensors for safe async use (independent of render loop)
        auto image = render->image->clone().permute({1, 2, 0});

        std::optional<PyTensor> screen_pos;
        if (render->screen_positions) {
            screen_pos = PyTensor(render->screen_positions->clone(), true);
        }

        return PyViewportRender{
            .image = PyTensor(std::move(image), true),
            .screen_positions = std::move(screen_pos),
        };
    }

    void register_rendering(nb::module_& m) {
        nb::class_<PyViewInfo>(m, "ViewInfo")
            .def_ro("rotation", &PyViewInfo::rotation)
            .def_ro("translation", &PyViewInfo::translation)
            .def_ro("width", &PyViewInfo::width)
            .def_ro("height", &PyViewInfo::height)
            .def_ro("fov_x", &PyViewInfo::fov_x)
            .def_ro("fov_y", &PyViewInfo::fov_y)
            .def_prop_ro(
                "position", [](const PyViewInfo& self) -> std::tuple<float, float, float> {
                    auto t = self.translation.tensor().cpu();
                    auto acc = t.accessor<float, 1>();
                    return {acc(0), acc(1), acc(2)};
                },
                "Camera position as (x, y, z) tuple");

        nb::class_<PyViewportRender>(m, "ViewportRender")
            .def_ro("image", &PyViewportRender::image)
            .def_ro("screen_positions", &PyViewportRender::screen_positions);

        m.def("get_viewport_render", &get_viewport_render,
              "Get the current viewport's rendered image and screen positions (None if not available)");

        m.def("capture_viewport", &capture_viewport,
              "Capture viewport render for async processing (clones data, safe to use from background threads)");

        m.def("render_view", &render_view, nb::arg("rotation"), nb::arg("translation"), nb::arg("width"), nb::arg("height"),
              nb::arg("fov") = DEFAULT_FOV, nb::arg("bg_color") = nb::none(),
              R"doc(
Render scene from arbitrary camera parameters.

Args:
    rotation: [3, 3] camera rotation matrix
    translation: [3] camera position
    width: Render width in pixels
    height: Render height in pixels
    fov: Field of view in degrees (default: 60)
    bg_color: Optional [3] RGB background color

Returns:
    Tensor [H, W, 3] RGB image on CUDA, or None if scene not available
)doc");

        m.def("compute_screen_positions", &compute_screen_positions, nb::arg("rotation"), nb::arg("translation"),
              nb::arg("width"), nb::arg("height"), nb::arg("fov") = DEFAULT_FOV,
              R"doc(
Compute screen positions of all Gaussians for a given camera view.

Args:
    rotation: [3, 3] camera rotation matrix
    translation: [3] camera position
    width: Viewport width in pixels
    height: Viewport height in pixels
    fov: Field of view in degrees (default: 60)

Returns:
    Tensor [N, 2] with (x, y) pixel coordinates for each Gaussian
)doc");

        m.def("get_current_view", &get_current_view, "Get current viewport camera info (None if not available)");

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
