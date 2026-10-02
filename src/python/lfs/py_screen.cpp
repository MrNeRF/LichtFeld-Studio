/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_screen.hpp"

#include "py_ui.hpp"
#include "py_viewer_dispatch.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/render_constants.hpp"
#include "screen/screen.hpp"
#include "screen/view3d_space.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/post_work_utils.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::python {

    namespace {

        vis::VisualizerImpl* visualizer_impl() {
            return dynamic_cast<vis::VisualizerImpl*>(get_visualizer());
        }

        void notify_screen_changed(vis::VisualizerImpl* impl) {
            if (auto* rendering = impl->getRenderingManager())
                rendering->markDirty(vis::DirtyFlag::ALL);
        }

        vis::screen::AreaId area_id(const int id) {
            return vis::screen::AreaId{id > 0 ? static_cast<std::uint32_t>(id) : 0};
        }

        nlohmann::json area_dict(vis::VisualizerImpl& impl, const vis::screen::AreaId id) {
            const auto& screen = impl.screens().screen();
            const auto* area = screen.area(id);
            nlohmann::json out = nlohmann::json::object();
            out["id"] = static_cast<int>(id.value);
            out["editor"] = area ? area->editor : std::string{};
            const auto rect = impl.areaRect(id);
            out["x"] = rect.x;
            out["y"] = rect.y;
            out["width"] = rect.w;
            out["height"] = rect.h;
            const bool is_view = area && area->editor == vis::screen::editors::kView3D;
            out["is_view"] = is_view;
            out["active_view"] = is_view && screen.activeView() == id;
            out["maximized"] = screen.maximized() == id;
            return out;
        }

        glm::vec3 vec3_from_seq(const nb::object& obj, const char* name) {
            if (!nb::isinstance<nb::sequence>(obj) && !nb::isinstance<nb::tuple>(obj) &&
                !nb::isinstance<nb::list>(obj))
                throw nb::type_error((std::string(name) + " must be a sequence of 3 numbers").c_str());
            if (nb::len(obj) != 3)
                throw nb::value_error((std::string(name) + " must have 3 components").c_str());
            glm::vec3 v;
            for (int i = 0; i < 3; ++i) {
                v[i] = nb::cast<float>(obj[i]);
                if (!std::isfinite(v[i]))
                    throw nb::value_error((std::string(name) + " must be finite").c_str());
            }
            return v;
        }

        nb::object json_to_python(const nlohmann::json& json) {
            if (json.is_null())
                return nb::none();
            if (json.is_boolean())
                return nb::cast(json.get<bool>());
            if (json.is_number_unsigned())
                return nb::cast(json.get<unsigned long long>());
            if (json.is_number_integer())
                return nb::cast(json.get<long long>());
            if (json.is_number_float())
                return nb::cast(json.get<double>());
            if (json.is_string())
                return nb::cast(json.get<std::string>());
            if (json.is_array()) {
                nb::list list;
                for (const auto& item : json)
                    list.append(json_to_python(item));
                return list;
            }
            if (json.is_object()) {
                nb::dict dict;
                for (const auto& [key, value] : json.items())
                    dict[key.c_str()] = json_to_python(value);
                return dict;
            }
            return nb::none();
        }

        std::vector<nb::dict> json_dicts_to_python(const nlohmann::json& array) {
            std::vector<nb::dict> out;
            out.reserve(array.size());
            for (const auto& item : array)
                out.push_back(nb::cast<nb::dict>(json_to_python(item)));
            return out;
        }

        nlohmann::json python_to_json(const nb::handle& obj) {
            if (obj.is_none())
                return nullptr;
            if (nb::isinstance<nb::bool_>(obj))
                return nb::cast<bool>(obj);
            if (nb::isinstance<nb::int_>(obj))
                return nb::cast<long long>(obj);
            if (nb::isinstance<nb::float_>(obj))
                return nb::cast<double>(obj);
            if (nb::isinstance<nb::str>(obj))
                return nb::cast<std::string>(obj);
            if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
                nlohmann::json array = nlohmann::json::array();
                for (const auto item : obj)
                    array.push_back(python_to_json(item));
                return array;
            }
            if (nb::isinstance<nb::dict>(obj)) {
                nlohmann::json object = nlohmann::json::object();
                for (const auto item : nb::cast<nb::dict>(obj))
                    object[nb::cast<std::string>(item.first)] = python_to_json(item.second);
                return object;
            }
            throw nb::type_error("Unsupported view settings value");
        }

    } // namespace

    void register_ui_screen(nb::module_& m) {
        auto screen = m.def_submodule("screen", "Editor screen areas and 3D views");

        screen.def(
            "areas",
            []() {
                return json_dicts_to_python(invoke_on_viewer(
                    []() -> nlohmann::json {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return nlohmann::json::array();
                        nlohmann::json out = nlohmann::json::array();
                        impl->screens().read([&](const vis::screen::Screen& s) {
                            for (const auto id : s.areas())
                                out.push_back(area_dict(*impl, id));
                        });
                        return out;
                    },
                    nlohmann::json::array()));
            },
            "List the screen's areas as dicts (id, editor, geometry, view flags)");

        screen.def(
            "editors",
            []() {
                return json_dicts_to_python(invoke_on_viewer(
                    []() -> nlohmann::json {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return nlohmann::json::array();
                        nlohmann::json out = nlohmann::json::array();
                        for (const auto& type : impl->screens().editorTypes().list()) {
                            nlohmann::json item;
                            item["id"] = type.id;
                            item["label"] = type.label;
                            item["multi_instance"] = type.multi_instance;
                            out.push_back(std::move(item));
                        }
                        return out;
                    },
                    nlohmann::json::array()));
            },
            "List registered editor types");

        screen.def(
            "split",
            [](const int area, const std::string& direction, const float factor) {
                if (direction != "vertical" && direction != "horizontal")
                    throw nb::value_error("direction must be 'vertical' or 'horizontal'");
                if (!(factor > 0.0f && factor < 1.0f) || !std::isfinite(factor))
                    throw nb::value_error("factor must be in (0, 1)");
                const auto axis = direction == "vertical" ? vis::screen::SplitAxis::Columns
                                                          : vis::screen::SplitAxis::Rows;
                return invoke_on_viewer(
                    [area, axis, factor]() {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return 0;
                        vis::screen::AreaId added;
                        impl->screens().edit([&](vis::screen::Screen& s) {
                            added = s.split(area_id(area), axis, factor);
                        });
                        if (added.valid())
                            notify_screen_changed(impl);
                        return static_cast<int>(added.value);
                    },
                    0);
            },
            nb::arg("area"), nb::arg("direction") = "vertical", nb::arg("factor") = 0.5f,
            "Split an area. direction is 'vertical' (side by side) or 'horizontal' (stacked).");

        screen.def(
            "join",
            [](const int keep, const int remove) {
                return invoke_on_viewer(
                    [keep, remove]() {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return false;
                        bool done = false;
                        impl->screens().edit([&](vis::screen::Screen& s) {
                            done = s.join(area_id(keep), area_id(remove));
                        });
                        if (done)
                            notify_screen_changed(impl);
                        return done;
                    },
                    false);
            },
            nb::arg("keep"), nb::arg("remove"), "Join two neighbouring areas, keeping `keep`");

        screen.def(
            "close",
            [](const int area) {
                return invoke_on_viewer(
                    [area]() {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return false;
                        bool done = false;
                        impl->screens().edit([&](vis::screen::Screen& s) { done = s.close(area_id(area)); });
                        if (done)
                            notify_screen_changed(impl);
                        return done;
                    },
                    false);
            },
            nb::arg("area"), "Close an area");

        screen.def(
            "swap",
            [](const int a, const int b) {
                return invoke_on_viewer(
                    [a, b]() {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return false;
                        bool done = false;
                        impl->screens().edit([&](vis::screen::Screen& s) { done = s.swap(area_id(a), area_id(b)); });
                        if (done)
                            notify_screen_changed(impl);
                        return done;
                    },
                    false);
            },
            nb::arg("a"), nb::arg("b"), "Swap the editors of two areas");

        screen.def(
            "set_editor",
            [](const int area, const std::string& editor) {
                return invoke_on_viewer(
                    [area, editor]() {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return false;
                        bool done = false;
                        impl->screens().edit([&](vis::screen::Screen& s) {
                            done = s.setEditor(area_id(area), editor);
                        });
                        if (done)
                            notify_screen_changed(impl);
                        return done;
                    },
                    false);
            },
            nb::arg("area"), nb::arg("editor"), "Show an editor in an area");

        screen.def(
            "open_editor",
            [](const std::string& editor) {
                return invoke_on_viewer(
                    [editor]() {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return 0;
                        vis::screen::AreaId id;
                        impl->screens().edit([&](vis::screen::Screen& s) { id = s.openEditor(editor); });
                        if (id.valid())
                            notify_screen_changed(impl);
                        return static_cast<int>(id.value);
                    },
                    0);
            },
            nb::arg("editor"), "Open an editor, splitting if needed. Returns the area id.");

        screen.def(
            "close_editor",
            [](const std::string& editor) {
                return invoke_on_viewer(
                    [editor]() {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return false;
                        bool done = false;
                        impl->screens().edit([&](vis::screen::Screen& s) { done = s.closeEditor(editor); });
                        if (done)
                            notify_screen_changed(impl);
                        return done;
                    },
                    false);
            },
            nb::arg("editor"), "Close every area showing this editor");

        screen.def(
            "toggle_maximized",
            [](const int area) {
                return invoke_on_viewer(
                    [area]() {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return false;
                        bool done = false;
                        impl->screens().edit([&](vis::screen::Screen& s) {
                            done = s.toggleMaximized(area_id(area));
                        });
                        if (done)
                            notify_screen_changed(impl);
                        return done;
                    },
                    false);
            },
            nb::arg("area"), "Maximize an area, or restore if it is already maximized");

        screen.def(
            "reset",
            []() {
                invoke_on_viewer([]() {
                    auto* impl = visualizer_impl();
                    if (!impl)
                        return;
                    impl->screens().resetToDefault();
                    notify_screen_changed(impl);
                });
            },
            "Reset the screen to the default layout");

        screen.def(
            "active_view",
            []() {
                return invoke_on_viewer(
                    []() {
                        auto* impl = visualizer_impl();
                        return impl ? static_cast<int>(impl->screens().screen().activeView().value) : 0;
                    },
                    0);
            },
            "Id of the active 3D view");

        screen.def(
            "set_active_view",
            [](const int view) {
                return invoke_on_viewer(
                    [view]() {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return false;
                        bool done = false;
                        impl->screens().edit([&](vis::screen::Screen& s) {
                            done = s.setActiveView(area_id(view));
                        });
                        if (done)
                            notify_screen_changed(impl);
                        return done;
                    },
                    false);
            },
            nb::arg("view"), "Set the active 3D view");

        screen.def(
            "view_command",
            [](const int view, const std::string& command) {
                return invoke_on_viewer(
                    [view, command]() {
                        auto* impl = visualizer_impl();
                        return impl && impl->runViewCommand(static_cast<vis::ViewId>(view), command);
                    },
                    false);
            },
            nb::arg("view"), nb::arg("command"), "Run a view command on a 3D view");

        screen.def(
            "view_camera",
            [](const int view) {
                auto out = nb::cast<nb::dict>(json_to_python(invoke_on_viewer(
                    [view]() -> nlohmann::json {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return nlohmann::json::object();
                        const auto* space = impl->screens().view3D(area_id(view));
                        if (!space)
                            return nlohmann::json::object();
                        const auto& cam = space->camera.camera;
                        nlohmann::json out = nlohmann::json::object();
                        nlohmann::json rotation = nlohmann::json::array();
                        for (int c = 0; c < 3; ++c)
                            for (int r = 0; r < 3; ++r)
                                rotation.push_back(cam.R[c][r]);
                        out["rotation"] = rotation;
                        out["translation"] = nlohmann::json::array({cam.t.x, cam.t.y, cam.t.z});
                        out["pivot"] = nlohmann::json::array({cam.pivot.x, cam.pivot.y, cam.pivot.z});
                        out["fov"] = lfs::rendering::focalLengthToVFov(space->settings.focal_length_mm);
                        out["orthographic"] = space->settings.orthographic;
                        out["ortho_scale"] = space->settings.ortho_scale;
                        out["width"] = space->camera.windowSize.x;
                        out["height"] = space->camera.windowSize.y;
                        return out;
                    },
                    nlohmann::json::object())));
                if (out.contains("translation")) {
                    out["translation"] = nb::tuple(nb::borrow<nb::object>(out["translation"]));
                    out["pivot"] = nb::tuple(nb::borrow<nb::object>(out["pivot"]));
                }
                return out;
            },
            nb::arg("view"), "Camera state of a 3D view");

        screen.def(
            "set_view_camera",
            [](const int view, const nb::object& eye, const nb::object& target, const nb::object& up) {
                const glm::vec3 eye_v = vec3_from_seq(eye, "eye");
                const glm::vec3 target_v = vec3_from_seq(target, "target");
                const glm::vec3 up_v = vec3_from_seq(up, "up");
                const auto rotation = lfs::rendering::tryMakeVisualizerLookAtRotation(eye_v, target_v, up_v);
                if (!rotation)
                    throw nb::value_error("eye, target and up must form a valid look-at");
                return invoke_on_viewer(
                    [view, eye_v, target_v, rotation = *rotation]() {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return false;
                        auto* space = impl->screens().view3D(area_id(view));
                        if (!space)
                            return false;
                        space->camera.setViewMatrix(rotation, eye_v);
                        space->camera.camera.setPivot(target_v);
                        notify_screen_changed(impl);
                        return true;
                    },
                    false);
            },
            nb::arg("view"), nb::arg("eye"), nb::arg("target"),
            nb::arg("up") = nb::make_tuple(0.0f, 1.0f, 0.0f),
            "Point a 3D view's camera at target from eye");

        screen.def(
            "view_settings",
            [](const int view) {
                return nb::cast<nb::dict>(json_to_python(invoke_on_viewer(
                    [view]() -> nlohmann::json {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return nlohmann::json::object();
                        const auto* space = impl->screens().view3D(area_id(view));
                        if (!space)
                            return nlohmann::json::object();
                        return vis::screen::viewSettingsToJson(space->settings);
                    },
                    nlohmann::json::object())));
            },
            nb::arg("view"), "ViewSettings of a 3D view as a dict");

        screen.def(
            "set_view_settings",
            [](const int view, const nb::kwargs& fields) {
                nlohmann::json patch = nlohmann::json::object();
                const auto known = vis::screen::viewSettingsToJson(vis::ViewSettings{});
                for (const auto item : fields) {
                    const auto key = nb::cast<std::string>(item.first);
                    if (!known.contains(key))
                        throw nb::value_error(("Unknown view setting: " + key).c_str());
                    patch[key] = python_to_json(item.second);
                }
                const auto result = invoke_on_viewer(
                    [view, patch]() -> std::string {
                        auto* impl = visualizer_impl();
                        if (!impl)
                            return "Visualizer is not available";
                        auto* space = impl->screens().view3D(area_id(view));
                        if (!space)
                            return "Not a 3D view";
                        auto restored = vis::screen::viewSettingsFromJson(patch, space->settings);
                        if (!restored)
                            return "Invalid view settings";
                        space->settings = *restored;
                        notify_screen_changed(impl);
                        return {};
                    },
                    std::string("Visualizer is not available"));
                if (!result.empty())
                    throw nb::value_error(result.c_str());
            },
            "Update ViewSettings fields on a 3D view. Unknown fields raise.");
    }

} // namespace lfs::python
