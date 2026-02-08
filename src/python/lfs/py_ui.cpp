/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_ui.hpp"
#include "control/command_api.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/property_registry.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "internal/resource_paths.hpp"
#include "py_gizmo.hpp"
#include "py_keymap.hpp"
#include "py_params.hpp"
#include "py_prop_registry.hpp"
#include "py_signals.hpp"
#include "py_uilist.hpp"
#include "py_viewport.hpp"
#include "python/python_runtime.hpp"
#include "python/ui_hooks.hpp"
#include "rendering/render_constants.hpp"
#include "visualizer/core/editor_context.hpp"
#include "visualizer/gui/panel_registry.hpp"
#include "visualizer/operator/operator_context.hpp"
#include "visualizer/operator/operator_registry.hpp"
#include "visualizer/operator/property_schema.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/scene/scene.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/theme/theme.hpp"
#include "visualizer/tools/unified_tool_registry.hpp"
#include "visualizer/training/training_manager.hpp"

#include "config.h"
#include <cuda_runtime.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <future>
#include <implot.h>
#include <stack>
#include <string_view>
#include <unordered_map>
#include <imgui.h>

#ifdef _WIN32
#include <shellapi.h>
#endif

namespace lfs::python {

    using lfs::training::CommandCenter;

    namespace {
        std::string get_class_id(nb::object cls) {
            auto mod = nb::cast<std::string>(cls.attr("__module__"));
            auto name = nb::cast<std::string>(cls.attr("__qualname__"));
            return mod + "." + name;
        }

        constexpr size_t INPUT_TEXT_BUFFER_SIZE = 1024;
        constexpr float GRID_AUTO_COLUMN_WIDTH = 100.0f;
        constexpr float ALERT_BG_ALPHA = 0.15f;
        constexpr int PROP_ENUM_COLOR_COUNT = 3;
        constexpr float BOX_BORDER_SIZE = 1.0f;

        ImVec4 tuple_to_imvec4(const std::tuple<float, float, float, float>& t) {
            return {std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t)};
        }

        // Icon cache for Python toolbar
        std::unordered_map<std::string, unsigned int> g_icon_cache;
        std::mutex g_icon_cache_mutex;

        // Plugin icon ownership tracking
        std::unordered_map<std::string, std::vector<std::string>> g_plugin_icons;
        std::mutex g_plugin_icons_mutex;

        // Operator cancel callback (Python callable)
        nb::callable g_cancel_operator_py_callback;

        // Modal event callback (Python callable)
        nb::callable g_modal_event_py_callback;

        // Python event callbacks
        nb::object g_popup_draw_callback;
        nb::object g_show_dataset_popup_callback;
        nb::object g_show_resume_popup_callback;
        nb::object g_request_exit_callback;

        // Redraw request flag for hot-reload notification
        std::atomic<bool> g_redraw_requested{false};

        // Speed overlay state for status bar
        struct SpeedOverlayState {
            float wasd_speed = 0.0f;
            float zoom_speed = 0.0f;
            std::chrono::steady_clock::time_point wasd_start;
            std::chrono::steady_clock::time_point zoom_start;
            bool wasd_visible = false;
            bool zoom_visible = false;
            static constexpr auto DURATION = std::chrono::milliseconds(3000);
            static constexpr float FADE_MS = 500.0f;

            void show_wasd(float speed) {
                wasd_speed = speed;
                wasd_visible = true;
                wasd_start = std::chrono::steady_clock::now();
            }

            void show_zoom(float speed) {
                zoom_speed = speed;
                zoom_visible = true;
                zoom_start = std::chrono::steady_clock::now();
            }

            std::tuple<float, float> get_wasd() const {
                if (!wasd_visible)
                    return {0.0f, 0.0f};
                const auto now = std::chrono::steady_clock::now();
                if (now - wasd_start >= DURATION)
                    return {0.0f, 0.0f};
                const auto remaining = DURATION - std::chrono::duration_cast<std::chrono::milliseconds>(now - wasd_start);
                const float alpha = (remaining.count() < FADE_MS) ? remaining.count() / FADE_MS : 1.0f;
                return {wasd_speed, alpha};
            }

            std::tuple<float, float> get_zoom() const {
                if (!zoom_visible)
                    return {0.0f, 0.0f};
                const auto now = std::chrono::steady_clock::now();
                if (now - zoom_start >= DURATION)
                    return {0.0f, 0.0f};
                const auto remaining = DURATION - std::chrono::duration_cast<std::chrono::milliseconds>(now - zoom_start);
                const float alpha = (remaining.count() < FADE_MS) ? remaining.count() / FADE_MS : 1.0f;
                return {zoom_speed, alpha};
            }
        };
        SpeedOverlayState g_speed_overlay_state;

        // Python operator instance registry - stores instances for property setting
        std::unordered_map<std::string, nb::object> g_python_operator_instances;
        std::mutex g_python_operator_mutex;

        nb::object get_python_operator_instance(const std::string& id) {
            std::lock_guard lock(g_python_operator_mutex);
            auto it = g_python_operator_instances.find(id);
            if (it != g_python_operator_instances.end()) {
                return it->second;
            }
            return nb::none();
        }

        void store_python_operator_instance(const std::string& id, nb::object instance) {
            std::lock_guard lock(g_python_operator_mutex);
            g_python_operator_instances[id] = std::move(instance);
        }

        void remove_python_operator_instance(const std::string& id) {
            std::lock_guard lock(g_python_operator_mutex);
            g_python_operator_instances.erase(id);
        }

        unsigned int load_icon_from_path(const std::filesystem::path& path, const std::string& cache_key) {
            const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);

            const auto result = lfs::python::create_gl_texture(data, width, height, channels);
            lfs::core::free_image(data);
            const auto texture_id = result.texture_id;

            {
                std::lock_guard lock(g_icon_cache_mutex);
                g_icon_cache[cache_key] = texture_id;
            }

            return texture_id;
        }

        constexpr const char* DEFAULT_ICON = "default.png";

        unsigned int load_default_icon() {
            {
                std::lock_guard lock(g_icon_cache_mutex);
                auto it = g_icon_cache.find(DEFAULT_ICON);
                if (it != g_icon_cache.end()) {
                    return it->second;
                }
            }
            try {
                return load_icon_from_path(lfs::vis::getAssetPath("icon/" + std::string(DEFAULT_ICON)), DEFAULT_ICON);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load default icon: {}", e.what());
                return 0;
            }
        }

        unsigned int load_icon_texture(const std::string& icon_name) {
            {
                std::lock_guard lock(g_icon_cache_mutex);
                auto it = g_icon_cache.find(icon_name);
                if (it != g_icon_cache.end()) {
                    return it->second;
                }
            }

            try {
                return load_icon_from_path(lfs::vis::getAssetPath("icon/" + icon_name), icon_name);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load icon {}: {}", icon_name, e.what());
                return load_default_icon();
            }
        }

        unsigned int load_scene_icon(const std::string& icon_name) {
            const std::string cache_key = "scene/" + icon_name + ".png";
            {
                std::lock_guard lock(g_icon_cache_mutex);
                auto it = g_icon_cache.find(cache_key);
                if (it != g_icon_cache.end()) {
                    return it->second;
                }
            }

            try {
                return load_icon_from_path(lfs::vis::getAssetPath("icon/scene/" + icon_name + ".png"), cache_key);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load scene icon {}: {}", icon_name, e.what());
                return load_default_icon();
            }
        }

        unsigned int load_plugin_icon(const std::string& icon_name,
                                      const std::string& plugin_path,
                                      const std::string& plugin_name) {
            const std::string cache_key = "plugin:" + plugin_name + ":" + icon_name;

            {
                std::lock_guard lock(g_icon_cache_mutex);
                auto it = g_icon_cache.find(cache_key);
                if (it != g_icon_cache.end()) {
                    return it->second;
                }
            }

            std::filesystem::path icon_path = std::filesystem::path(plugin_path) / "icons" / (icon_name + ".png");

            if (!std::filesystem::exists(icon_path)) {
                icon_path = lfs::vis::getAssetPath("icon/" + icon_name + ".png");
            }

            try {
                const auto tex = load_icon_from_path(icon_path, cache_key);

                {
                    std::lock_guard lock(g_plugin_icons_mutex);
                    g_plugin_icons[plugin_name].push_back(cache_key);
                }

                return tex;
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load plugin icon {}: {}", icon_name, e.what());
                return load_default_icon();
            }
        }

        void free_plugin_icons(const std::string& plugin_name) {
            std::vector<std::string> keys_to_free;
            {
                std::lock_guard lock(g_plugin_icons_mutex);
                auto it = g_plugin_icons.find(plugin_name);
                if (it != g_plugin_icons.end()) {
                    keys_to_free = std::move(it->second);
                    g_plugin_icons.erase(it);
                }
            }

            std::lock_guard lock(g_icon_cache_mutex);
            for (const auto& key : keys_to_free) {
                auto it = g_icon_cache.find(key);
                if (it != g_icon_cache.end()) {
                    lfs::python::delete_gl_texture(it->second);
                    g_icon_cache.erase(it);
                }
            }
        }

        // Thread-local layout stack for hierarchical layouts
        thread_local std::stack<LayoutContext> g_layout_stack;

        bool draw_prop_enum_button(nb::object data, const std::string& prop_id,
                                   const std::string& value, const std::string& text) {
            const std::string current = nb::cast<std::string>(data.attr(prop_id.c_str()));
            const bool selected = (current == value);
            const std::string& display = text.empty() ? value : text;

            if (selected) {
                const auto theme = get_current_theme();
                ImGui::PushStyleColor(ImGuiCol_Button, tuple_to_imvec4(theme.palette.primary));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, tuple_to_imvec4(theme.palette.primary));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, tuple_to_imvec4(theme.palette.primary_dim));
            }
            const bool clicked = ImGui::Button(display.c_str());
            if (selected)
                ImGui::PopStyleColor(PROP_ENUM_COLOR_COUNT);

            if (clicked && !selected) {
                data.attr(prop_id.c_str()) = nb::cast(value);
                return true;
            }
            return false;
        }

        // Window flags for Python bindings
        struct PyWindowFlags {
            static constexpr int None = 0;
            static constexpr int NoScrollbar = ImGuiWindowFlags_NoScrollbar;
            static constexpr int NoScrollWithMouse = ImGuiWindowFlags_NoScrollWithMouse;
            static constexpr int MenuBar = ImGuiWindowFlags_MenuBar;
            static constexpr int NoResize = ImGuiWindowFlags_NoResize;
            static constexpr int NoMove = ImGuiWindowFlags_NoMove;
            static constexpr int NoCollapse = ImGuiWindowFlags_NoCollapse;
            static constexpr int AlwaysAutoResize = ImGuiWindowFlags_AlwaysAutoResize;
            static constexpr int NoTitleBar = ImGuiWindowFlags_NoTitleBar;
            static constexpr int NoNavFocus = ImGuiWindowFlags_NoNavFocus;
            static constexpr int NoInputs = ImGuiWindowFlags_NoInputs;
            static constexpr int NoBackground = ImGuiWindowFlags_NoBackground;
            static constexpr int NoFocusOnAppearing = ImGuiWindowFlags_NoFocusOnAppearing;
            static constexpr int NoBringToFrontOnFocus = ImGuiWindowFlags_NoBringToFrontOnFocus;
        };

        // Image texture preload cache for Python image preview
        struct PreloadEntry {
            std::future<std::tuple<unsigned char*, int, int, int>> future;
            std::atomic<bool> ready{false};
        };
        std::unordered_map<std::string, std::unique_ptr<PreloadEntry>> g_preload_cache;
        std::mutex g_preload_mutex;

        int g_max_texture_size = 0;

        void ensure_max_texture_size() {
            if (g_max_texture_size == 0)
                g_max_texture_size = lfs::python::get_max_texture_size();
        }

        std::tuple<uint64_t, int, int> create_gl_texture_from_data(unsigned char* data, const int width, const int height, const int channels) {
            if (!data || width <= 0 || height <= 0)
                return {0, 0, 0};

            ensure_max_texture_size();

            const auto result = lfs::python::create_gl_texture(data, width, height, channels);
            return {static_cast<uint64_t>(result.texture_id), result.width, result.height};
        }

        vis::op::OperatorResult parse_operator_result(nb::object result, nb::object instance = nb::none()) {
            if (nb::isinstance<nb::set>(result)) {
                nb::set result_set = nb::cast<nb::set>(result);
                for (auto item : result_set) {
                    std::string s = nb::cast<std::string>(item);
                    if (s == "FINISHED")
                        return vis::op::OperatorResult::FINISHED;
                    if (s == "RUNNING_MODAL")
                        return vis::op::OperatorResult::RUNNING_MODAL;
                    if (s == "CANCELLED")
                        return vis::op::OperatorResult::CANCELLED;
                    if (s == "PASS_THROUGH")
                        return vis::op::OperatorResult::PASS_THROUGH;
                }
            } else if (nb::isinstance<nb::str>(result)) {
                std::string s = nb::cast<std::string>(result);
                if (s == "FINISHED")
                    return vis::op::OperatorResult::FINISHED;
                if (s == "RUNNING_MODAL")
                    return vis::op::OperatorResult::RUNNING_MODAL;
                if (s == "CANCELLED")
                    return vis::op::OperatorResult::CANCELLED;
                if (s == "PASS_THROUGH")
                    return vis::op::OperatorResult::PASS_THROUGH;
            } else if (nb::isinstance<nb::dict>(result)) {
                nb::dict result_dict = nb::cast<nb::dict>(result);
                vis::op::OperatorResult status = vis::op::OperatorResult::CANCELLED;
                nb::dict data;

                for (auto [key, value] : result_dict) {
                    std::string key_str = nb::cast<std::string>(key);
                    if (key_str == "status") {
                        std::string status_str = nb::cast<std::string>(value);
                        if (status_str == "FINISHED")
                            status = vis::op::OperatorResult::FINISHED;
                        else if (status_str == "RUNNING_MODAL")
                            status = vis::op::OperatorResult::RUNNING_MODAL;
                        else if (status_str == "CANCELLED")
                            status = vis::op::OperatorResult::CANCELLED;
                        else if (status_str == "PASS_THROUGH")
                            status = vis::op::OperatorResult::PASS_THROUGH;
                    } else {
                        data[key] = value;
                    }
                }

                if (instance.is_valid() && !instance.is_none()) {
                    nb::setattr(instance, "_return_data", data);
                }

                return status;
            }
            return vis::op::OperatorResult::CANCELLED;
        }

        std::vector<vis::op::PropertySchema> extract_property_schemas(nb::object cls) {
            std::vector<vis::op::PropertySchema> schemas;

            try {
                if (!nb::hasattr(cls, "__dict__")) {
                    return schemas;
                }
                nb::object dict_obj = cls.attr("__dict__");
                if (!nb::isinstance<nb::dict>(dict_obj)) {
                    return schemas;
                }
                nb::dict class_dict = nb::cast<nb::dict>(dict_obj);

                for (auto item : class_dict) {
                    nb::object name = nb::borrow(item.first);
                    nb::object value = nb::borrow(item.second);

                    if (!nb::hasattr(value, "_attr_name")) {
                        continue;
                    }

                    vis::op::PropertySchema schema;
                    schema.name = nb::cast<std::string>(name);

                    std::string type_name = nb::cast<std::string>(nb::borrow(value.type()).attr("__name__"));

                    if (type_name == "FloatProperty") {
                        schema.type = vis::op::PropertyType::FLOAT;
                        if (nb::hasattr(value, "min")) {
                            double min_val = nb::cast<double>(value.attr("min"));
                            if (min_val > -1e30) {
                                schema.min = min_val;
                            }
                        }
                        if (nb::hasattr(value, "max")) {
                            double max_val = nb::cast<double>(value.attr("max"));
                            if (max_val < 1e30) {
                                schema.max = max_val;
                            }
                        }
                        if (nb::hasattr(value, "precision")) {
                            schema.precision = nb::cast<int>(value.attr("precision"));
                        }
                    } else if (type_name == "IntProperty") {
                        schema.type = vis::op::PropertyType::INT;
                        if (nb::hasattr(value, "min")) {
                            schema.min = static_cast<double>(nb::cast<int64_t>(value.attr("min")));
                        }
                        if (nb::hasattr(value, "max")) {
                            schema.max = static_cast<double>(nb::cast<int64_t>(value.attr("max")));
                        }
                        if (nb::hasattr(value, "step")) {
                            schema.step = nb::cast<int>(value.attr("step"));
                        }
                    } else if (type_name == "BoolProperty") {
                        schema.type = vis::op::PropertyType::BOOL;
                    } else if (type_name == "StringProperty") {
                        schema.type = vis::op::PropertyType::STRING;
                        if (nb::hasattr(value, "maxlen")) {
                            schema.maxlen = nb::cast<int>(value.attr("maxlen"));
                        }
                    } else if (type_name == "EnumProperty") {
                        schema.type = vis::op::PropertyType::ENUM;
                        if (nb::hasattr(value, "items")) {
                            nb::object items_list = value.attr("items");
                            for (auto enum_item : items_list) {
                                auto tuple = nb::cast<nb::tuple>(enum_item);
                                if (nb::len(tuple) >= 3) {
                                    schema.enum_items.emplace_back(
                                        nb::cast<std::string>(tuple[0]),
                                        nb::cast<std::string>(tuple[1]),
                                        nb::cast<std::string>(tuple[2]));
                                }
                            }
                        }
                    } else if (type_name == "FloatVectorProperty") {
                        schema.type = vis::op::PropertyType::FLOAT_VECTOR;
                        if (nb::hasattr(value, "size")) {
                            schema.size = nb::cast<int>(value.attr("size"));
                        }
                        if (nb::hasattr(value, "min")) {
                            double min_val = nb::cast<double>(value.attr("min"));
                            if (min_val > -1e30) {
                                schema.min = min_val;
                            }
                        }
                        if (nb::hasattr(value, "max")) {
                            double max_val = nb::cast<double>(value.attr("max"));
                            if (max_val < 1e30) {
                                schema.max = max_val;
                            }
                        }
                    } else if (type_name == "IntVectorProperty") {
                        schema.type = vis::op::PropertyType::INT_VECTOR;
                        if (nb::hasattr(value, "size")) {
                            schema.size = nb::cast<int>(value.attr("size"));
                        }
                        if (nb::hasattr(value, "min")) {
                            schema.min = static_cast<double>(nb::cast<int64_t>(value.attr("min")));
                        }
                        if (nb::hasattr(value, "max")) {
                            schema.max = static_cast<double>(nb::cast<int64_t>(value.attr("max")));
                        }
                    } else if (type_name == "TensorProperty") {
                        schema.type = vis::op::PropertyType::TENSOR;
                    } else {
                        continue;
                    }

                    if (nb::hasattr(value, "subtype")) {
                        schema.subtype = nb::cast<std::string>(value.attr("subtype"));
                    }
                    if (nb::hasattr(value, "description")) {
                        schema.description = nb::cast<std::string>(value.attr("description"));
                    }

                    schemas.push_back(std::move(schema));
                }
            } catch (const std::exception& e) {
                LOG_DEBUG("Failed to extract property schemas: {}", e.what());
            }
            return schemas;
        }

        void register_python_operator_to_cpp(nb::object cls) {
            std::string id = get_class_id(cls);
            std::string label;
            std::string description;

            if (nb::hasattr(cls, "label")) {
                label = nb::cast<std::string>(cls.attr("label"));
            }
            if (nb::hasattr(cls, "description")) {
                description = nb::cast<std::string>(cls.attr("description"));
            }

            nb::object instance;
            try {
                instance = cls();
            } catch (const std::exception& e) {
                LOG_ERROR("register_python_operator: failed to create instance for '{}': {}", id, e.what());
                return;
            }

            bool has_poll = nb::hasattr(cls, "poll");
            bool has_invoke = nb::hasattr(instance, "invoke");
            bool has_execute = nb::hasattr(instance, "execute");
            bool has_modal = nb::hasattr(instance, "modal");
            bool has_cancel = nb::hasattr(instance, "cancel");
            bool has_undo = nb::hasattr(instance, "undo") && nb::hasattr(instance, "redo");

            vis::op::OperatorDescriptor desc;
            desc.python_class_id = id;
            desc.label = label.empty() ? id : label;
            desc.description = description;
            desc.source = vis::op::OperatorSource::PYTHON;
            desc.flags = vis::op::OperatorFlags::REGISTER;
            if (has_undo) {
                desc.flags = desc.flags | vis::op::OperatorFlags::UNDO;
            }
            if (has_modal) {
                desc.flags = desc.flags | vis::op::OperatorFlags::MODAL;
            }

            vis::op::CallbackOperator callbacks;

            if (has_poll) {
                callbacks.poll = [class_id = id]() -> bool {
                    nb::gil_scoped_acquire gil;
                    nb::object instance = get_python_operator_instance(class_id);
                    if (!instance.is_valid() || instance.is_none()) {
                        LOG_ERROR("Python operator '{}' not found during poll", class_id);
                        return false;
                    }
                    try {
                        nb::object cls = nb::borrow(instance.type());
                        return nb::cast<bool>(cls.attr("poll")(nb::none()));
                    } catch (const std::exception& e) {
                        LOG_ERROR("Operator poll error: {}", e.what());
                        return false;
                    }
                };
            }

            if (has_invoke || has_execute) {
                callbacks.invoke = [class_id = id, has_invoke](vis::op::OperatorProperties& /*props*/) -> vis::op::OperatorResult {
                    nb::gil_scoped_acquire gil;
                    nb::object instance = get_python_operator_instance(class_id);
                    if (!instance.is_valid() || instance.is_none()) {
                        LOG_ERROR("Python operator '{}' not found during invoke", class_id);
                        return vis::op::OperatorResult::CANCELLED;
                    }
                    try {
                        nb::object result;
                        if (has_invoke) {
                            PyEvent py_event;
                            py_event.type = "INVOKE";
                            py_event.value = "NOTHING";
                            result = instance.attr("invoke")(nb::none(), py_event);
                        } else {
                            result = instance.attr("execute")(nb::none());
                        }
                        return parse_operator_result(result, instance);
                    } catch (const std::exception& e) {
                        LOG_ERROR("Operator invoke error: {}", e.what());
                        return vis::op::OperatorResult::CANCELLED;
                    }
                };
            }

            if (has_modal) {
                callbacks.modal = [class_id = id](const vis::op::ModalEvent& event,
                                                  vis::op::OperatorProperties& /*props*/) -> vis::op::OperatorResult {
                    nb::gil_scoped_acquire gil;
                    nb::object instance = get_python_operator_instance(class_id);
                    if (!instance.is_valid() || instance.is_none()) {
                        LOG_ERROR("Python operator '{}' not found during modal", class_id);
                        return vis::op::OperatorResult::CANCELLED;
                    }
                    try {
                        PyEvent py_event = convert_modal_event(event);
                        nb::object result = instance.attr("modal")(nb::none(), py_event);
                        return parse_operator_result(result, instance);
                    } catch (const std::exception& e) {
                        LOG_ERROR("Operator modal error: {}", e.what());
                        return vis::op::OperatorResult::CANCELLED;
                    }
                };
            }

            if (has_cancel) {
                callbacks.cancel = [class_id = id]() {
                    nb::gil_scoped_acquire gil;
                    nb::object instance = get_python_operator_instance(class_id);
                    if (!instance.is_valid() || instance.is_none()) {
                        LOG_ERROR("Python operator '{}' not found during cancel", class_id);
                        return;
                    }
                    try {
                        instance.attr("cancel")(nb::none());
                    } catch (const std::exception& e) {
                        LOG_ERROR("Operator cancel error: {}", e.what());
                    }
                };
            }

            store_python_operator_instance(id, instance);
            vis::op::operators().registerCallbackOperator(std::move(desc), std::move(callbacks));

            auto schemas = extract_property_schemas(cls);
            if (!schemas.empty()) {
                vis::op::propertySchemas().registerSchema(id, std::move(schemas));
            }

            LOG_DEBUG("Operator '{}' registered", id);
        }
    } // namespace

    namespace {
        std::string glfw_key_to_type_string(int key) {
            if (key >= GLFW_KEY_A && key <= GLFW_KEY_Z) {
                return std::string("KEY_") + static_cast<char>('A' + (key - GLFW_KEY_A));
            }
            if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9) {
                return std::string("KEY_") + static_cast<char>('0' + (key - GLFW_KEY_0));
            }
            switch (key) {
            case GLFW_KEY_SPACE: return "SPACE";
            case GLFW_KEY_ESCAPE: return "ESC";
            case GLFW_KEY_ENTER: return "RET";
            case GLFW_KEY_TAB: return "TAB";
            case GLFW_KEY_BACKSPACE: return "BACK_SPACE";
            case GLFW_KEY_DELETE: return "DEL";
            case GLFW_KEY_LEFT: return "LEFT_ARROW";
            case GLFW_KEY_RIGHT: return "RIGHT_ARROW";
            case GLFW_KEY_UP: return "UP_ARROW";
            case GLFW_KEY_DOWN: return "DOWN_ARROW";
            case GLFW_KEY_HOME: return "HOME";
            case GLFW_KEY_END: return "END";
            case GLFW_KEY_PAGE_UP: return "PAGE_UP";
            case GLFW_KEY_PAGE_DOWN: return "PAGE_DOWN";
            case GLFW_KEY_LEFT_SHIFT:
            case GLFW_KEY_RIGHT_SHIFT: return "LEFT_SHIFT";
            case GLFW_KEY_LEFT_CONTROL:
            case GLFW_KEY_RIGHT_CONTROL: return "LEFT_CTRL";
            case GLFW_KEY_LEFT_ALT:
            case GLFW_KEY_RIGHT_ALT: return "LEFT_ALT";
            case GLFW_KEY_F1: return "F1";
            case GLFW_KEY_F2: return "F2";
            case GLFW_KEY_F3: return "F3";
            case GLFW_KEY_F4: return "F4";
            case GLFW_KEY_F5: return "F5";
            case GLFW_KEY_F6: return "F6";
            case GLFW_KEY_F7: return "F7";
            case GLFW_KEY_F8: return "F8";
            case GLFW_KEY_F9: return "F9";
            case GLFW_KEY_F10: return "F10";
            case GLFW_KEY_F11: return "F11";
            case GLFW_KEY_F12: return "F12";
            default: return "NONE";
            }
        }

        std::string glfw_button_to_type_string(int button) {
            switch (button) {
            case GLFW_MOUSE_BUTTON_LEFT: return "LEFTMOUSE";
            case GLFW_MOUSE_BUTTON_RIGHT: return "RIGHTMOUSE";
            case GLFW_MOUSE_BUTTON_MIDDLE: return "MIDDLEMOUSE";
            default: return "MOUSE";
            }
        }

        std::string action_to_value_string(int action) {
            switch (action) {
            case GLFW_PRESS: return "PRESS";
            case GLFW_RELEASE: return "RELEASE";
            default: return "NOTHING";
            }
        }
    } // namespace

    PyEvent convert_modal_event(const vis::op::ModalEvent& event) {
        PyEvent py_event;

        if (const auto* mouse_btn = event.as<vis::MouseButtonEvent>()) {
            py_event.type = glfw_button_to_type_string(mouse_btn->button);
            py_event.value = action_to_value_string(mouse_btn->action);
            py_event.mouse_x = mouse_btn->position.x;
            py_event.mouse_y = mouse_btn->position.y;
            py_event.mouse_region_x = mouse_btn->position.x;
            py_event.mouse_region_y = mouse_btn->position.y;
            py_event.shift = (mouse_btn->mods & GLFW_MOD_SHIFT) != 0;
            py_event.ctrl = (mouse_btn->mods & GLFW_MOD_CONTROL) != 0;
            py_event.alt = (mouse_btn->mods & GLFW_MOD_ALT) != 0;
        } else if (const auto* mouse_move = event.as<vis::MouseMoveEvent>()) {
            py_event.type = "MOUSEMOVE";
            py_event.value = "NOTHING";
            py_event.mouse_x = mouse_move->position.x;
            py_event.mouse_y = mouse_move->position.y;
            py_event.mouse_region_x = mouse_move->position.x;
            py_event.mouse_region_y = mouse_move->position.y;
            py_event.delta_x = mouse_move->delta.x;
            py_event.delta_y = mouse_move->delta.y;
        } else if (const auto* scroll = event.as<vis::MouseScrollEvent>()) {
            py_event.type = scroll->yoffset > 0 ? "WHEELUPMOUSE" : "WHEELDOWNMOUSE";
            py_event.value = "PRESS";
            py_event.scroll_x = scroll->xoffset;
            py_event.scroll_y = scroll->yoffset;
        } else if (const auto* key = event.as<vis::KeyEvent>()) {
            py_event.type = glfw_key_to_type_string(key->key);
            py_event.value = action_to_value_string(key->action);
            py_event.key_code = key->key;
            py_event.shift = (key->mods & GLFW_MOD_SHIFT) != 0;
            py_event.ctrl = (key->mods & GLFW_MOD_CONTROL) != 0;
            py_event.alt = (key->mods & GLFW_MOD_ALT) != 0;
        } else {
            py_event.type = "NONE";
            py_event.value = "NOTHING";
        }

        return py_event;
    }

    nb::object get_python_operator_instance(const std::string& id) {
        std::lock_guard lock(g_python_operator_mutex);
        auto it = g_python_operator_instances.find(id);
        if (it != g_python_operator_instances.end()) {
            return it->second;
        }
        return nb::none();
    }

    // --- PySubLayout ---

    PySubLayout::PySubLayout(PyUILayout* parent, LayoutType type, float split_factor,
                             int grid_columns)
        : parent_(parent),
          type_(type),
          split_factor_(split_factor),
          grid_columns_(grid_columns) {
        assert(parent_);
    }

    PySubLayout::PySubLayout(PySubLayout* parent_sub, LayoutType type, float split_factor,
                             int grid_columns)
        : parent_(parent_sub->parent_),
          type_(type),
          split_factor_(split_factor),
          grid_columns_(grid_columns) {
        assert(parent_);
        inherited_state_ = parent_sub->effective_state();
    }

    PySubLayout::~PySubLayout() {
        if (entered_)
            exit();
    }

    PySubLayout& PySubLayout::enter() {
        entered_ = true;

        LayoutContext ctx;
        ctx.type = type_;
        ctx.split_factor = split_factor_;
        ctx.child_index = 0;
        ctx.is_first_child = true;
        ctx.available_width = ImGui::GetContentRegionAvail().x;
        ctx.cursor_start_x = ImGui::GetCursorPosX();
        ctx.grid_columns = grid_columns_;

        switch (type_) {
        case LayoutType::Split:
            ImGui::BeginGroup();
            break;
        case LayoutType::Box: {
            const auto theme = get_current_theme();
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, theme.sizes.frame_rounding);
            ImGui::PushStyleColor(ImGuiCol_ChildBg, tuple_to_imvec4(theme.palette.surface));
            ImGui::PushStyleColor(ImGuiCol_Border, tuple_to_imvec4(theme.palette.border));
            ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, BOX_BORDER_SIZE);
            const std::string id = "##pybox_" + std::to_string(parent_->next_box_id());
            ImGui::BeginChild(id.c_str(), {0, 0},
                              ImGuiChildFlags_Border | ImGuiChildFlags_AutoResizeY);
            break;
        }
        case LayoutType::GridFlow: {
            int cols = grid_columns_;
            if (cols <= 0) {
                const float avail = ImGui::GetContentRegionAvail().x;
                cols = std::max(1, static_cast<int>(avail / GRID_AUTO_COLUMN_WIDTH));
            }
            ctx.grid_actual_columns = cols;
            const std::string id = "##pygrid_" + std::to_string(parent_->next_grid_id());
            ctx.table_active =
                ImGui::BeginTable(id.c_str(), cols, ImGuiTableFlags_SizingStretchSame);
            if (ctx.table_active)
                ImGui::TableNextRow();
            break;
        }
        default:
            break;
        }

        g_layout_stack.push(ctx);
        return *this;
    }

    void PySubLayout::exit() {
        if (!entered_ || g_layout_stack.empty())
            return;

        entered_ = false;
        const auto ctx = g_layout_stack.top();
        g_layout_stack.pop();

        switch (ctx.type) {
        case LayoutType::Split:
            if (ctx.child_index > 0) {
                ImGui::PopItemWidth();
                ImGui::EndGroup();
            }
            ImGui::EndGroup();
            break;
        case LayoutType::Row:
            break;
        case LayoutType::Box:
            ImGui::EndChild();
            ImGui::PopStyleColor(2);
            ImGui::PopStyleVar(2);
            break;
        case LayoutType::GridFlow:
            if (ctx.table_active)
                ImGui::EndTable();
            break;
        default:
            break;
        }
    }

    void PySubLayout::advance_child() {
        assert(!g_layout_stack.empty() && "advance_child() called outside layout context");
        if (g_layout_stack.empty())
            return;
        auto& ctx = g_layout_stack.top();
        switch (ctx.type) {
        case LayoutType::Row:
            if (!ctx.is_first_child)
                ImGui::SameLine();
            ctx.is_first_child = false;
            break;
        case LayoutType::Column:
            ctx.is_first_child = false;
            break;
        case LayoutType::Split:
            if (ctx.child_index == 0) {
                float w = ctx.available_width * ctx.split_factor;
                ImGui::BeginGroup();
                ImGui::PushItemWidth(w - ImGui::GetStyle().ItemSpacing.x);
            } else if (ctx.child_index == 1) {
                ImGui::PopItemWidth();
                ImGui::EndGroup();
                ImGui::SameLine();
                float w = ctx.available_width * (1.0f - ctx.split_factor);
                ImGui::BeginGroup();
                ImGui::PushItemWidth(w - ImGui::GetStyle().ItemSpacing.x);
            } else if (ctx.child_index == 2) {
                LOG_WARN("Split layout received more than 2 children");
            }
            ctx.child_index++;
            break;
        case LayoutType::GridFlow:
            if (ctx.table_active) {
                if (ctx.child_index > 0 && ctx.child_index % ctx.grid_actual_columns == 0)
                    ImGui::TableNextRow();
                ImGui::TableNextColumn();
            }
            ctx.child_index++;
            break;
        case LayoutType::Box:
        case LayoutType::Root:
            break;
        }
    }

    LayoutState PySubLayout::effective_state() const {
        LayoutState eff;
        eff.enabled = own_state_.enabled && inherited_state_.enabled;
        eff.active = own_state_.active && inherited_state_.active;
        eff.alert = own_state_.alert || inherited_state_.alert;
        eff.scale_x = own_state_.scale_x * inherited_state_.scale_x;
        eff.scale_y = own_state_.scale_y * inherited_state_.scale_y;
        return eff;
    }

    void PySubLayout::apply_state() {
        const auto eff = effective_state();
        if (!eff.enabled) {
            ImGui::BeginDisabled(true);
            disabled_pushed_ = true;
        }
        if (eff.alert) {
            const auto theme = get_current_theme();
            const auto err = tuple_to_imvec4(theme.palette.error);
            ImGui::PushStyleColor(ImGuiCol_Text, err);
            ImGui::PushStyleColor(ImGuiCol_FrameBg,
                                  ImVec4(err.x, err.y, err.z, ALERT_BG_ALPHA));
            color_push_count_ += 2;
        }
        const float scale = std::max(eff.scale_x, eff.scale_y);
        if (scale != 1.0f) {
            ImGui::SetWindowFontScale(scale);
            font_scale_pushed_ = true;
        }
    }

    void PySubLayout::pop_per_item_state() {
        if (font_scale_pushed_) {
            ImGui::SetWindowFontScale(1.0f);
            font_scale_pushed_ = false;
        }
        if (color_push_count_ > 0) {
            ImGui::PopStyleColor(color_push_count_);
            color_push_count_ = 0;
        }
        if (disabled_pushed_) {
            ImGui::EndDisabled();
            disabled_pushed_ = false;
        }
        if (own_state_.alert)
            own_state_.alert = false;
    }

    // Sub-layout creation
    PySubLayout PySubLayout::row() { return PySubLayout(this, LayoutType::Row); }
    PySubLayout PySubLayout::column() { return PySubLayout(this, LayoutType::Column); }
    PySubLayout PySubLayout::split(float factor) { return PySubLayout(this, LayoutType::Split, factor); }
    PySubLayout PySubLayout::box() { return PySubLayout(this, LayoutType::Box); }
    PySubLayout PySubLayout::grid_flow(int columns, bool, bool) {
        return PySubLayout(this, LayoutType::GridFlow, 0.5f, columns);
    }

    // Delegated widget methods
    void PySubLayout::label(const std::string& text) {
        advance_child();
        apply_state();
        parent_->label(text);
        pop_per_item_state();
    }
    bool PySubLayout::button(const std::string& l, std::tuple<float, float> size) {
        advance_child();
        apply_state();
        auto r = parent_->button(l, size);
        pop_per_item_state();
        return r;
    }
    bool PySubLayout::button_styled(const std::string& l, const std::string& style,
                                    std::tuple<float, float> size) {
        advance_child();
        apply_state();
        auto r = parent_->button_styled(l, style, size);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, nb::object> PySubLayout::prop(nb::object data, const std::string& prop_id,
                                                   std::optional<std::string> text) {
        advance_child();
        apply_state();
        auto r = parent_->prop(data, prop_id, text);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, bool> PySubLayout::checkbox(const std::string& l, bool v) {
        advance_child();
        apply_state();
        auto r = parent_->checkbox(l, v);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, float> PySubLayout::slider_float(const std::string& l, float v, float mn, float mx) {
        advance_child();
        apply_state();
        auto r = parent_->slider_float(l, v, mn, mx);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, int> PySubLayout::slider_int(const std::string& l, int v, int mn, int mx) {
        advance_child();
        apply_state();
        auto r = parent_->slider_int(l, v, mn, mx);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, float> PySubLayout::drag_float(const std::string& l, float v, float speed, float mn, float mx) {
        advance_child();
        apply_state();
        auto r = parent_->drag_float(l, v, speed, mn, mx);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, int> PySubLayout::drag_int(const std::string& l, int v, float speed, int mn, int mx) {
        advance_child();
        apply_state();
        auto r = parent_->drag_int(l, v, speed, mn, mx);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, std::string> PySubLayout::input_text(const std::string& l, const std::string& v) {
        advance_child();
        apply_state();
        auto r = parent_->input_text(l, v);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, int> PySubLayout::combo(const std::string& l, int idx,
                                             const std::vector<std::string>& items) {
        advance_child();
        apply_state();
        auto r = parent_->combo(l, idx, items);
        pop_per_item_state();
        return r;
    }
    void PySubLayout::separator() {
        advance_child();
        parent_->separator();
    }
    void PySubLayout::spacing() {
        advance_child();
        parent_->spacing();
    }
    void PySubLayout::heading(const std::string& text) {
        advance_child();
        apply_state();
        parent_->heading(text);
        pop_per_item_state();
    }
    bool PySubLayout::collapsing_header(const std::string& l, bool default_open) {
        advance_child();
        apply_state();
        auto r = parent_->collapsing_header(l, default_open);
        pop_per_item_state();
        return r;
    }
    bool PySubLayout::tree_node(const std::string& l) {
        advance_child();
        apply_state();
        auto r = parent_->tree_node(l);
        pop_per_item_state();
        return r;
    }
    void PySubLayout::tree_pop() {
        parent_->tree_pop();
    }
    void PySubLayout::progress_bar(float fraction, const std::string& overlay, float width,
                                   float height) {
        advance_child();
        apply_state();
        parent_->progress_bar(fraction, overlay, width, height);
        pop_per_item_state();
    }
    void PySubLayout::text_colored(const std::string& text, std::tuple<float, float, float, float> color) {
        advance_child();
        parent_->text_colored(text, color);
    }
    void PySubLayout::text_wrapped(const std::string& text) {
        advance_child();
        apply_state();
        parent_->text_wrapped(text);
        pop_per_item_state();
    }

    bool PySubLayout::prop_enum(nb::object data, const std::string& prop_id,
                                const std::string& value, const std::string& text) {
        advance_child();
        return draw_prop_enum_button(data, prop_id, value, text);
    }

    bool PySubLayout::begin_table(const std::string& id, int columns) {
        advance_child();
        apply_state();
        auto r = parent_->begin_table(id, columns);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, float> PySubLayout::input_float(const std::string& label, float value,
                                                     float step, float step_fast,
                                                     const std::string& format) {
        advance_child();
        apply_state();
        auto r = parent_->input_float(label, value, step, step_fast, format);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, int> PySubLayout::input_int(const std::string& label, int value,
                                                 int step, int step_fast) {
        advance_child();
        apply_state();
        auto r = parent_->input_int(label, value, step, step_fast);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, int> PySubLayout::input_int_formatted(const std::string& label, int value,
                                                           int step, int step_fast) {
        advance_child();
        apply_state();
        auto r = parent_->input_int_formatted(label, value, step, step_fast);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, float> PySubLayout::stepper_float(const std::string& label, float value,
                                                       const std::vector<float>& steps) {
        advance_child();
        apply_state();
        auto r = parent_->stepper_float(label, value, steps);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, int> PySubLayout::radio_button(const std::string& label, int current, int value) {
        advance_child();
        apply_state();
        auto r = parent_->radio_button(label, current, value);
        pop_per_item_state();
        return r;
    }
    bool PySubLayout::small_button(const std::string& label) {
        advance_child();
        apply_state();
        auto r = parent_->small_button(label);
        pop_per_item_state();
        return r;
    }
    bool PySubLayout::selectable(const std::string& label, bool selected, float height) {
        advance_child();
        apply_state();
        auto r = parent_->selectable(label, selected, height);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, std::tuple<float, float, float>> PySubLayout::color_edit3(
        const std::string& label, std::tuple<float, float, float> color) {
        advance_child();
        apply_state();
        auto r = parent_->color_edit3(label, color);
        pop_per_item_state();
        return r;
    }
    void PySubLayout::text_disabled(const std::string& text) {
        advance_child();
        apply_state();
        parent_->text_disabled(text);
        pop_per_item_state();
    }
    std::tuple<bool, int> PySubLayout::listbox(const std::string& label, int current_idx,
                                               const std::vector<std::string>& items, int height_items) {
        advance_child();
        apply_state();
        auto r = parent_->listbox(label, current_idx, items, height_items);
        pop_per_item_state();
        return r;
    }
    void PySubLayout::image(uint64_t texture_id, std::tuple<float, float> size,
                            std::tuple<float, float, float, float> tint) {
        advance_child();
        apply_state();
        parent_->image(texture_id, size, tint);
        pop_per_item_state();
    }
    bool PySubLayout::image_button(const std::string& id, uint64_t texture_id,
                                   std::tuple<float, float> size,
                                   std::tuple<float, float, float, float> tint) {
        advance_child();
        apply_state();
        auto r = parent_->image_button(id, texture_id, size, tint);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, std::string> PySubLayout::input_text_with_hint(const std::string& label,
                                                                    const std::string& hint,
                                                                    const std::string& value) {
        advance_child();
        apply_state();
        auto r = parent_->input_text_with_hint(label, hint, value);
        pop_per_item_state();
        return r;
    }
    std::tuple<bool, std::string> PySubLayout::input_text_enter(const std::string& label,
                                                                const std::string& value) {
        advance_child();
        apply_state();
        auto r = parent_->input_text_enter(label, value);
        pop_per_item_state();
        return r;
    }

    void PySubLayout::table_setup_column(const std::string& label, float width) const {
        parent_->table_setup_column(label, width);
    }
    void PySubLayout::table_next_row() const {
        parent_->table_next_row();
    }
    void PySubLayout::table_next_column() const {
        parent_->table_next_column();
    }
    void PySubLayout::table_headers_row() const {
        parent_->table_headers_row();
    }
    void PySubLayout::end_table() const {
        parent_->end_table();
    }

    void PySubLayout::push_item_width(float width) const {
        parent_->push_item_width(width);
    }
    void PySubLayout::pop_item_width() const {
        parent_->pop_item_width();
    }
    void PySubLayout::set_tooltip(const std::string& text) const {
        parent_->set_tooltip(text);
    }
    bool PySubLayout::is_item_hovered() const {
        return parent_->is_item_hovered();
    }
    bool PySubLayout::is_item_clicked(int button) const {
        return parent_->is_item_clicked(button);
    }
    void PySubLayout::begin_disabled(bool disabled) const {
        parent_->begin_disabled(disabled);
    }
    void PySubLayout::end_disabled() const {
        parent_->end_disabled();
    }
    void PySubLayout::same_line(float offset, float spacing) const {
        parent_->same_line(offset, spacing);
    }
    void PySubLayout::push_id(const std::string& id) const {
        parent_->push_id(id);
    }
    void PySubLayout::pop_id() const {
        parent_->pop_id();
    }
    bool PySubLayout::begin_child(const std::string& id, std::tuple<float, float> size, bool border) const {
        return parent_->begin_child(id, size, border);
    }
    void PySubLayout::end_child() const {
        parent_->end_child();
    }
    bool PySubLayout::begin_context_menu(const std::string& id) const {
        return parent_->begin_context_menu(id);
    }
    void PySubLayout::end_context_menu() const {
        parent_->end_context_menu();
    }
    bool PySubLayout::menu_item(const std::string& label, bool enabled) const {
        return parent_->menu_item(label, enabled);
    }
    bool PySubLayout::begin_menu(const std::string& label) const {
        return parent_->begin_menu(label);
    }
    void PySubLayout::end_menu() const {
        parent_->end_menu();
    }
    std::tuple<float, float> PySubLayout::get_content_region_avail() const {
        return parent_->get_content_region_avail();
    }

    // PyUILayout layout methods
    PySubLayout PyUILayout::row() { return PySubLayout(this, LayoutType::Row); }
    PySubLayout PyUILayout::column() { return PySubLayout(this, LayoutType::Column); }
    PySubLayout PyUILayout::split(float factor) { return PySubLayout(this, LayoutType::Split, factor); }
    PySubLayout PyUILayout::box() { return PySubLayout(this, LayoutType::Box); }
    PySubLayout PyUILayout::grid_flow(int columns, bool, bool) {
        return PySubLayout(this, LayoutType::GridFlow, 0.5f, columns);
    }

    bool PyUILayout::prop_enum(nb::object data, const std::string& prop_id,
                               const std::string& value, const std::string& text) {
        return draw_prop_enum_button(data, prop_id, value, text);
    }

    nb::object PyUILayout::operator_(const std::string& operator_id, const std::string& text,
                                     const std::string& icon) {
        const auto* desc = vis::op::operators().getDescriptor(operator_id);
        std::string label = text.empty() ? (desc ? desc->label : operator_id) : text;
        std::string btn_text = LOC(label.c_str());

        const bool can_execute = vis::op::operators().poll(operator_id);

        bool clicked = false;
        if (menu_depth_ > 0) {
            clicked = ImGui::MenuItem(btn_text.c_str(), nullptr, false, can_execute);
        } else {
            if (!can_execute) {
                ImGui::BeginDisabled();
            }

            if (!icon.empty()) {
                unsigned int tex_id = load_icon_texture(icon);
                if (tex_id) {
                    float size = ImGui::GetTextLineHeight();
                    clicked = ImGui::ImageButton(operator_id.c_str(), static_cast<ImTextureID>(tex_id),
                                                 ImVec2(size, size));
                    if (!btn_text.empty()) {
                        ImGui::SameLine();
                        ImGui::TextUnformatted(btn_text.c_str());
                    }
                } else {
                    clicked = ImGui::Button(btn_text.c_str());
                }
            } else {
                clicked = ImGui::Button(btn_text.c_str());
            }

            if (!can_execute) {
                ImGui::EndDisabled();
            }
        }

        auto props = PyOperatorProperties(operator_id);

        if (clicked) {
            vis::op::operators().invoke(operator_id);
        }

        return nb::cast(props);
    }

    std::tuple<bool, int> PyUILayout::prop_search(nb::object /*data*/, const std::string& prop_id,
                                                  nb::object search_data, const std::string& search_prop,
                                                  const std::string& text) {
        std::vector<std::string> items;
        int current_idx = 0;

        try {
            if (nb::hasattr(search_data, search_prop.c_str())) {
                nb::object collection = search_data.attr(search_prop.c_str());
                if (nb::hasattr(collection, "__iter__")) {
                    size_t idx = 0;
                    for (auto item : collection) {
                        if (nb::hasattr(item, "name")) {
                            items.push_back(nb::cast<std::string>(item.attr("name")));
                        } else {
                            items.push_back("Item " + std::to_string(idx));
                        }
                        ++idx;
                    }
                }
            }
        } catch (...) {
            LOG_WARN("prop_search: failed to enumerate items for '{}'", prop_id);
        }

        std::string label = text.empty() ? prop_id : text;
        auto [changed, new_idx] = combo(label, current_idx, items);
        return {changed, new_idx};
    }

    std::tuple<int, int> PyUILayout::template_list(const std::string& /*list_type_id*/,
                                                   const std::string& list_id,
                                                   nb::object data, const std::string& prop_id,
                                                   nb::object active_data, const std::string& active_prop,
                                                   int rows) {
        std::vector<std::string> items;
        int active_idx = 0;

        try {
            if (nb::hasattr(data, prop_id.c_str())) {
                nb::object collection = data.attr(prop_id.c_str());
                if (nb::hasattr(collection, "__iter__")) {
                    size_t idx = 0;
                    for (auto item : collection) {
                        if (nb::hasattr(item, "name")) {
                            items.push_back(nb::cast<std::string>(item.attr("name")));
                        } else {
                            items.push_back("Item " + std::to_string(idx));
                        }
                        ++idx;
                    }
                }
            }
            if (nb::hasattr(active_data, active_prop.c_str())) {
                active_idx = nb::cast<int>(active_data.attr(active_prop.c_str()));
            }
        } catch (...) {
            LOG_WARN("template_list: failed to get active index for '{}'", active_prop);
        }

        auto [changed, new_idx] = listbox(list_id, active_idx, items, rows);
        if (changed) {
            try {
                nb::setattr(active_data, active_prop.c_str(), nb::cast(new_idx));
            } catch (...) {
                LOG_WARN("template_list: failed to write selection for '{}'", active_prop);
            }
        }

        return {new_idx, static_cast<int>(items.size())};
    }

    void PyUILayout::menu(const std::string& menu_id, const std::string& text, const std::string& /*icon*/) {
        std::string label = text.empty() ? menu_id : text;
        if (ImGui::BeginMenu(label.c_str())) {
            // Menu content should be drawn by the registered Menu class
            // This just opens a submenu with the given label
            ImGui::TextDisabled("(Menu: %s)", menu_id.c_str());
            ImGui::EndMenu();
        }
    }

    void PyUILayout::popover(const std::string& panel_id, const std::string& text, const std::string& /*icon*/) {
        std::string btn_text = text.empty() ? panel_id : text;
        if (ImGui::Button(btn_text.c_str())) {
            ImGui::OpenPopup(panel_id.c_str());
        }

        if (ImGui::BeginPopup(panel_id.c_str())) {
            vis::gui::PanelDrawContext ctx;
            ctx.scene = get_scene_for_python();
            vis::gui::PanelRegistry::instance().draw_single_panel(panel_id, ctx);
            ImGui::EndPopup();
        }
    }

    // PyUILayout implementation - Drawing (for viewport overlays)
    namespace {
        ImU32 tuple_to_color32(std::tuple<float, float, float, float> color) {
            return IM_COL32(
                static_cast<int>(std::get<0>(color) * 255),
                static_cast<int>(std::get<1>(color) * 255),
                static_cast<int>(std::get<2>(color) * 255),
                static_cast<int>(std::get<3>(color) * 255));
        }
    } // namespace

    void PyUILayout::draw_circle(float x, float y, float radius,
                                 std::tuple<float, float, float, float> color,
                                 int segments, float thickness) {
        auto* const dl = ImGui::GetWindowDrawList();
        dl->AddCircle(ImVec2(x, y), radius, tuple_to_color32(color), segments, thickness);
    }

    void PyUILayout::draw_circle_filled(float x, float y, float radius,
                                        std::tuple<float, float, float, float> color,
                                        int segments) {
        auto* const dl = ImGui::GetWindowDrawList();
        dl->AddCircleFilled(ImVec2(x, y), radius, tuple_to_color32(color), segments);
    }

    void PyUILayout::draw_rect(float x0, float y0, float x1, float y1,
                               std::tuple<float, float, float, float> color,
                               float thickness) {
        auto* const dl = ImGui::GetWindowDrawList();
        dl->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), tuple_to_color32(color), 0.0f, 0, thickness);
    }

    void PyUILayout::draw_rect_filled(float x0, float y0, float x1, float y1,
                                      std::tuple<float, float, float, float> color, bool background) {
        auto* const dl = background ? ImGui::GetBackgroundDrawList() : ImGui::GetWindowDrawList();
        dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), tuple_to_color32(color));
    }

    void PyUILayout::draw_rect_rounded(float x0, float y0, float x1, float y1,
                                       std::tuple<float, float, float, float> color,
                                       float rounding, float thickness, bool background) {
        auto* const dl = background ? ImGui::GetBackgroundDrawList() : ImGui::GetWindowDrawList();
        dl->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), tuple_to_color32(color), rounding, 0, thickness);
    }

    void PyUILayout::draw_rect_rounded_filled(float x0, float y0, float x1, float y1,
                                              std::tuple<float, float, float, float> color,
                                              float rounding, bool background) {
        auto* const dl = background ? ImGui::GetBackgroundDrawList() : ImGui::GetWindowDrawList();
        dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), tuple_to_color32(color), rounding);
    }

    void PyUILayout::draw_triangle_filled(float x0, float y0, float x1, float y1, float x2, float y2,
                                          std::tuple<float, float, float, float> color, bool background) {
        auto* const dl = background ? ImGui::GetBackgroundDrawList() : ImGui::GetWindowDrawList();
        dl->AddTriangleFilled(ImVec2(x0, y0), ImVec2(x1, y1), ImVec2(x2, y2), tuple_to_color32(color));
    }

    void PyUILayout::draw_line(float x0, float y0, float x1, float y1,
                               std::tuple<float, float, float, float> color,
                               float thickness) {
        auto* dl = ImGui::GetForegroundDrawList();
        dl->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), tuple_to_color32(color), thickness);
    }

    void PyUILayout::draw_polyline(const std::vector<std::tuple<float, float>>& points,
                                   std::tuple<float, float, float, float> color,
                                   bool closed, float thickness) {
        if (points.empty())
            return;
        auto* dl = ImGui::GetForegroundDrawList();
        std::vector<ImVec2> im_points;
        im_points.reserve(points.size());
        for (const auto& [px, py] : points) {
            im_points.emplace_back(px, py);
        }
        dl->AddPolyline(im_points.data(), static_cast<int>(im_points.size()),
                        tuple_to_color32(color), closed ? ImDrawFlags_Closed : ImDrawFlags_None, thickness);
    }

    void PyUILayout::draw_poly_filled(const std::vector<std::tuple<float, float>>& points,
                                      std::tuple<float, float, float, float> color) {
        if (points.size() < 3)
            return;
        auto* dl = ImGui::GetForegroundDrawList();
        std::vector<ImVec2> im_points;
        im_points.reserve(points.size());
        for (const auto& [px, py] : points) {
            im_points.emplace_back(px, py);
        }
        dl->AddConvexPolyFilled(im_points.data(), static_cast<int>(im_points.size()), tuple_to_color32(color));
    }

    void PyUILayout::draw_text(float x, float y, const std::string& text,
                               std::tuple<float, float, float, float> color, bool background) {
        auto* const dl = background ? ImGui::GetBackgroundDrawList() : ImGui::GetForegroundDrawList();
        ImFont* const font = ImGui::GetFont();
        dl->AddText(font, font->FontSize, ImVec2(x, y), tuple_to_color32(color), text.c_str());
    }

    void PyUILayout::draw_window_rect_filled(float x0, float y0, float x1, float y1,
                                             std::tuple<float, float, float, float> color) {
        auto* const dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), tuple_to_color32(color));
    }

    void PyUILayout::draw_window_rect(float x0, float y0, float x1, float y1,
                                      std::tuple<float, float, float, float> color, float thickness) {
        auto* const dl = ImGui::GetWindowDrawList();
        dl->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), tuple_to_color32(color), 0.0f, 0, thickness);
    }

    void PyUILayout::draw_window_rect_rounded(float x0, float y0, float x1, float y1,
                                              std::tuple<float, float, float, float> color,
                                              float rounding, float thickness) {
        auto* const dl = ImGui::GetWindowDrawList();
        dl->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), tuple_to_color32(color), rounding, 0, thickness);
    }

    void PyUILayout::draw_window_rect_rounded_filled(float x0, float y0, float x1, float y1,
                                                     std::tuple<float, float, float, float> color,
                                                     float rounding) {
        auto* const dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), tuple_to_color32(color), rounding);
    }

    void PyUILayout::draw_window_line(float x0, float y0, float x1, float y1,
                                      std::tuple<float, float, float, float> color, float thickness) {
        auto* const dl = ImGui::GetWindowDrawList();
        dl->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), tuple_to_color32(color), thickness);
    }

    void PyUILayout::draw_window_text(float x, float y, const std::string& text,
                                      std::tuple<float, float, float, float> color) {
        auto* const dl = ImGui::GetWindowDrawList();
        ImFont* const font = ImGui::GetFont();
        dl->AddText(font, font->FontSize, ImVec2(x, y), tuple_to_color32(color), text.c_str());
    }

    void PyUILayout::draw_window_triangle_filled(float x0, float y0, float x1, float y1, float x2, float y2,
                                                 std::tuple<float, float, float, float> color) {
        auto* const dl = ImGui::GetWindowDrawList();
        dl->AddTriangleFilled(ImVec2(x0, y0), ImVec2(x1, y1), ImVec2(x2, y2), tuple_to_color32(color));
    }

    void PyUILayout::crf_curve_preview(const std::string& label, float gamma, float toe, float shoulder,
                                       float gamma_r, float gamma_g, float gamma_b) {
        vis::gui::widgets::CRFCurvePreview(label.c_str(), gamma, toe, shoulder, gamma_r, gamma_g, gamma_b);
    }

    std::tuple<bool, std::vector<float>> PyUILayout::chromaticity_diagram(
        const std::string& label,
        float red_x, float red_y, float green_x, float green_y,
        float blue_x, float blue_y, float neutral_x, float neutral_y,
        float range) {
        bool changed = vis::gui::widgets::ChromaticityDiagram(
            label.c_str(), &red_x, &red_y, &green_x, &green_y,
            &blue_x, &blue_y, &neutral_x, &neutral_y, range);
        return {changed, {red_x, red_y, green_x, green_y, blue_x, blue_y, neutral_x, neutral_y}};
    }

    // PyUILayout implementation - Text
    void PyUILayout::label(const std::string& text) {
        ImGui::TextUnformatted(text.c_str());
    }

    void PyUILayout::label_centered(const std::string& text) {
        const float text_width = ImGui::CalcTextSize(text.c_str()).x;
        const float avail_width = ImGui::GetContentRegionAvail().x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail_width - text_width) * 0.5f);
        ImGui::TextUnformatted(text.c_str());
    }

    void PyUILayout::heading(const std::string& text) {
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]); // Bold font
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopFont();
    }

    void PyUILayout::text_colored(const std::string& text, std::tuple<float, float, float, float> color) {
        ImGui::TextColored(tuple_to_imvec4(color), "%s", text.c_str());
    }

    void PyUILayout::text_colored_centered(const std::string& text, std::tuple<float, float, float, float> color) {
        const float text_width = ImGui::CalcTextSize(text.c_str()).x;
        const float avail_width = ImGui::GetContentRegionAvail().x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail_width - text_width) * 0.5f);
        ImGui::TextColored(tuple_to_imvec4(color), "%s", text.c_str());
    }

    void PyUILayout::text_selectable(const std::string& text, const float height) {
        constexpr ImGuiInputTextFlags FLAGS = ImGuiInputTextFlags_ReadOnly;
        const float h = height > 0 ? height : ImGui::GetTextLineHeight() * 3;
        ImGui::InputTextMultiline("##selectable", const_cast<char*>(text.c_str()), text.size() + 1, ImVec2(-1, h), FLAGS);
    }

    void PyUILayout::bullet_text(const std::string& text) {
        ImGui::BulletText("%s", text.c_str());
    }

    void PyUILayout::text_wrapped(const std::string& text) {
        ImGui::TextWrapped("%s", text.c_str());
    }

    void PyUILayout::text_disabled(const std::string& text) {
        ImGui::TextDisabled("%s", text.c_str());
    }

    bool PyUILayout::button(const std::string& label, std::tuple<float, float> size) {
        return ImGui::Button(label.c_str(), {std::get<0>(size), std::get<1>(size)});
    }

    bool PyUILayout::button_callback(const std::string& label, nb::object callback,
                                     std::tuple<float, float> size) {
        bool clicked = ImGui::Button(label.c_str(), {std::get<0>(size), std::get<1>(size)});
        if (clicked && !callback.is_none()) {
            try {
                callback();
            } catch (const nb::python_error& e) {
                LOG_ERROR("Button callback error: {}", e.what());
            }
        }
        return clicked;
    }

    bool PyUILayout::small_button(const std::string& label) {
        return ImGui::SmallButton(label.c_str());
    }

    std::tuple<bool, bool> PyUILayout::checkbox(const std::string& label, bool value) {
        bool v = value;
        bool changed = ImGui::Checkbox(label.c_str(), &v);
        return {changed, v};
    }

    std::tuple<bool, int> PyUILayout::radio_button(const std::string& label, int current, int value) {
        bool clicked = ImGui::RadioButton(label.c_str(), current == value);
        return {clicked, clicked ? value : current};
    }

    // Sliders
    std::tuple<bool, float> PyUILayout::slider_float(const std::string& label, float value, float min, float max) {
        float v = value;
        bool changed = ImGui::SliderFloat(label.c_str(), &v, min, max);
        return {changed, v};
    }

    std::tuple<bool, int> PyUILayout::slider_int(const std::string& label, int value, int min, int max) {
        int v = value;
        bool changed = ImGui::SliderInt(label.c_str(), &v, min, max);
        return {changed, v};
    }

    std::tuple<bool, std::tuple<float, float>> PyUILayout::slider_float2(
        const std::string& label, std::tuple<float, float> value, float min, float max) {
        float v[2] = {std::get<0>(value), std::get<1>(value)};
        bool changed = ImGui::SliderFloat2(label.c_str(), v, min, max);
        return {changed, {v[0], v[1]}};
    }

    std::tuple<bool, std::tuple<float, float, float>> PyUILayout::slider_float3(
        const std::string& label, std::tuple<float, float, float> value, float min, float max) {
        float v[3] = {std::get<0>(value), std::get<1>(value), std::get<2>(value)};
        bool changed = ImGui::SliderFloat3(label.c_str(), v, min, max);
        return {changed, {v[0], v[1], v[2]}};
    }

    // Drags
    std::tuple<bool, float> PyUILayout::drag_float(const std::string& label, float value,
                                                   float speed, float min, float max) {
        float v = value;
        bool changed = ImGui::DragFloat(label.c_str(), &v, speed, min, max);
        return {changed, v};
    }

    std::tuple<bool, int> PyUILayout::drag_int(const std::string& label, int value,
                                               float speed, int min, int max) {
        int v = value;
        bool changed = ImGui::DragInt(label.c_str(), &v, speed, min, max);
        return {changed, v};
    }

    // Input
    std::tuple<bool, std::string> PyUILayout::input_text(const std::string& label, const std::string& value) {
        char buffer[INPUT_TEXT_BUFFER_SIZE];
        std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\0';
        bool changed = ImGui::InputText(label.c_str(), buffer, sizeof(buffer));
        return {changed, std::string(buffer)};
    }

    std::tuple<bool, std::string> PyUILayout::input_text_with_hint(const std::string& label, const std::string& hint,
                                                                   const std::string& value) {
        char buffer[INPUT_TEXT_BUFFER_SIZE];
        std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\0';
        bool changed = ImGui::InputTextWithHint(label.c_str(), hint.c_str(), buffer, sizeof(buffer));
        return {changed, std::string(buffer)};
    }

    std::tuple<bool, float> PyUILayout::input_float(const std::string& label, float value, float step, float step_fast,
                                                    const std::string& format) {
        float v = value;
        const bool changed = ImGui::InputFloat(label.c_str(), &v, step, step_fast, format.c_str());
        return {changed, v};
    }

    std::tuple<bool, int> PyUILayout::input_int(const std::string& label, int value, int step, int step_fast) {
        int v = value;
        const bool changed = ImGui::InputInt(label.c_str(), &v, step, step_fast);
        return {changed, v};
    }

    std::tuple<bool, int> PyUILayout::input_int_formatted(const std::string& label, int value, int step, int step_fast) {
        int v = value;
        const bool changed = vis::gui::widgets::InputIntFormatted(label.c_str(), &v, step, step_fast);
        return {changed, v};
    }

    std::tuple<bool, float> PyUILayout::stepper_float(const std::string& label, float value,
                                                      const std::vector<float>& steps) {
        float v = value;
        bool changed = false;

        ImGui::PushID(label.c_str());

        const float avail = ImGui::GetContentRegionAvail().x;
        const float btn_height = ImGui::GetFrameHeight();
        const float spacing = ImGui::GetStyle().ItemSpacing.x;

        float buttons_total = 0.0f;
        for (size_t i = 0; i < steps.size(); ++i) {
            char minus_buf[32], plus_buf[32];
            std::snprintf(minus_buf, sizeof(minus_buf), "-%.6g", steps[i]);
            std::snprintf(plus_buf, sizeof(plus_buf), "+%.6g", steps[i]);
            buttons_total += ImGui::CalcTextSize(minus_buf).x + ImGui::GetStyle().FramePadding.x * 2;
            buttons_total += ImGui::CalcTextSize(plus_buf).x + ImGui::GetStyle().FramePadding.x * 2;
            buttons_total += spacing * 2;
        }

        const float input_width = std::max(avail - buttons_total - spacing, 60.0f);
        ImGui::SetNextItemWidth(input_width);
        if (ImGui::InputFloat("##val", &v, 0.0f, 0.0f, "%.3f")) {
            changed = true;
        }

        for (size_t i = 0; i < steps.size(); ++i) {
            const float step = steps[i];
            char buf[32];

            ImGui::SameLine();
            std::snprintf(buf, sizeof(buf), "-%.6g", step);
            ImGui::PushID(static_cast<int>(i * 2));
            if (ImGui::Button(buf, ImVec2(0, btn_height))) {
                v -= step;
                changed = true;
            }
            ImGui::PopID();

            ImGui::SameLine();
            std::snprintf(buf, sizeof(buf), "+%.6g", step);
            ImGui::PushID(static_cast<int>(i * 2 + 1));
            if (ImGui::Button(buf, ImVec2(0, btn_height))) {
                v += step;
                changed = true;
            }
            ImGui::PopID();
        }

        if (!label.empty() && label[0] != '#') {
            ImGui::SameLine();
            ImGui::TextUnformatted(label.c_str());
        }

        ImGui::PopID();
        return {changed, v};
    }

    std::tuple<bool, std::string> PyUILayout::path_input(const std::string& label, const std::string& value,
                                                         const bool folder_mode, const std::string& dialog_title) {
        char buffer[INPUT_TEXT_BUFFER_SIZE];
        std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\0';

        const float available = ImGui::GetContentRegionAvail().x;
        const float button_width = ImGui::CalcTextSize("...").x + ImGui::GetStyle().FramePadding.x * 2;
        const float input_width = available - button_width - ImGui::GetStyle().ItemSpacing.x;

        ImGui::SetNextItemWidth(input_width);
        bool changed = ImGui::InputText(label.c_str(), buffer, sizeof(buffer));

        ImGui::SameLine();
        const std::string btn_id = "...##" + label + "_browse";
        if (ImGui::Button(btn_id.c_str())) {
            const std::filesystem::path start_path = value.empty() ? std::filesystem::path{} : std::filesystem::path{value};
            std::filesystem::path result;
            if (folder_mode) {
                result = lfs::vis::gui::SelectFolderDialog(
                    dialog_title.empty() ? "Select Folder" : dialog_title, start_path);
            } else {
                result = lfs::vis::gui::OpenImageFileDialog(start_path);
            }
            if (!result.empty()) {
                std::strncpy(buffer, result.string().c_str(), sizeof(buffer) - 1);
                buffer[sizeof(buffer) - 1] = '\0';
                changed = true;
            }
        }

        return {changed, std::string(buffer)};
    }

    // Color
    std::tuple<bool, std::tuple<float, float, float>> PyUILayout::color_edit3(
        const std::string& label, std::tuple<float, float, float> color) {
        float c[3] = {std::get<0>(color), std::get<1>(color), std::get<2>(color)};
        bool changed = ImGui::ColorEdit3(label.c_str(), c);
        return {changed, {c[0], c[1], c[2]}};
    }

    std::tuple<bool, std::tuple<float, float, float, float>> PyUILayout::color_edit4(
        const std::string& label, std::tuple<float, float, float, float> color) {
        float c[4] = {std::get<0>(color), std::get<1>(color), std::get<2>(color), std::get<3>(color)};
        bool changed = ImGui::ColorEdit4(label.c_str(), c);
        return {changed, {c[0], c[1], c[2], c[3]}};
    }

    bool PyUILayout::color_button(const std::string& label, std::tuple<float, float, float, float> color,
                                  std::tuple<float, float> size) {
        return ImGui::ColorButton(label.c_str(), tuple_to_imvec4(color), 0,
                                  {std::get<0>(size), std::get<1>(size)});
    }

    // Selection
    std::tuple<bool, int> PyUILayout::combo(const std::string& label, int current_idx,
                                            const std::vector<std::string>& items) {
        int idx = current_idx;
        bool changed = false;
        if (ImGui::BeginCombo(label.c_str(), (idx >= 0 && idx < static_cast<int>(items.size()))
                                                 ? items[idx].c_str()
                                                 : "")) {
            for (int i = 0; i < static_cast<int>(items.size()); ++i) {
                bool selected = (i == idx);
                if (ImGui::Selectable(items[i].c_str(), selected)) {
                    idx = i;
                    changed = true;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        return {changed, idx};
    }

    std::tuple<bool, int> PyUILayout::listbox(const std::string& label, int current_idx,
                                              const std::vector<std::string>& items, int height_items) {
        int idx = current_idx;
        bool changed = false;
        if (ImGui::BeginListBox(label.c_str(), {0, height_items > 0 ? height_items * ImGui::GetTextLineHeightWithSpacing() : 0})) {
            for (int i = 0; i < static_cast<int>(items.size()); ++i) {
                bool selected = (i == idx);
                if (ImGui::Selectable(items[i].c_str(), selected)) {
                    idx = i;
                    changed = true;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndListBox();
        }
        return {changed, idx};
    }

    // Layout
    void PyUILayout::separator() {
        ImGui::Separator();
    }

    void PyUILayout::spacing() {
        ImGui::Spacing();
    }

    void PyUILayout::same_line(float offset, float spacing) {
        ImGui::SameLine(offset, spacing);
    }

    void PyUILayout::new_line() {
        ImGui::NewLine();
    }

    void PyUILayout::indent(float width) {
        ImGui::Indent(width);
    }

    void PyUILayout::unindent(float width) {
        ImGui::Unindent(width);
    }

    void PyUILayout::set_next_item_width(float width) {
        ImGui::SetNextItemWidth(width);
    }

    // Grouping
    void PyUILayout::begin_group() {
        ImGui::BeginGroup();
    }

    void PyUILayout::end_group() {
        ImGui::EndGroup();
    }

    bool PyUILayout::collapsing_header(const std::string& label, bool default_open) {
        return ImGui::CollapsingHeader(label.c_str(), default_open ? ImGuiTreeNodeFlags_DefaultOpen : 0);
    }

    bool PyUILayout::tree_node(const std::string& label) {
        return ImGui::TreeNode(label.c_str());
    }

    bool PyUILayout::tree_node_ex(const std::string& label, const std::string& flags_str) {
        ImGuiTreeNodeFlags flags = 0;
        if (flags_str.find("DefaultOpen") != std::string::npos)
            flags |= ImGuiTreeNodeFlags_DefaultOpen;
        if (flags_str.find("OpenOnArrow") != std::string::npos)
            flags |= ImGuiTreeNodeFlags_OpenOnArrow;
        if (flags_str.find("SpanAvailWidth") != std::string::npos)
            flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
        if (flags_str.find("Leaf") != std::string::npos)
            flags |= ImGuiTreeNodeFlags_Leaf;
        if (flags_str.find("NoTreePushOnOpen") != std::string::npos)
            flags |= ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (flags_str.find("Selected") != std::string::npos)
            flags |= ImGuiTreeNodeFlags_Selected;
        return ImGui::TreeNodeEx(label.c_str(), flags);
    }

    void PyUILayout::tree_pop() {
        ImGui::TreePop();
    }

    // Tables
    bool PyUILayout::begin_table(const std::string& id, int columns) {
        return ImGui::BeginTable(id.c_str(), columns);
    }

    void PyUILayout::end_table() {
        ImGui::EndTable();
    }

    void PyUILayout::table_next_row() {
        ImGui::TableNextRow();
    }

    void PyUILayout::table_next_column() {
        ImGui::TableNextColumn();
    }

    bool PyUILayout::table_set_column_index(int column) {
        return ImGui::TableSetColumnIndex(column);
    }

    void PyUILayout::table_headers_row() {
        ImGui::TableHeadersRow();
    }

    void PyUILayout::table_set_bg_color(int target, std::tuple<float, float, float, float> color) {
        ImU32 col = ImGui::ColorConvertFloat4ToU32(
            ImVec4(std::get<0>(color), std::get<1>(color), std::get<2>(color), std::get<3>(color)));
        ImGui::TableSetBgColor(static_cast<ImGuiTableBgTarget>(target), col);
    }

    void PyUILayout::table_setup_column(const std::string& label, const float width) {
        const ImGuiTableColumnFlags flags = (width > 0) ? ImGuiTableColumnFlags_WidthFixed : ImGuiTableColumnFlags_WidthStretch;
        ImGui::TableSetupColumn(label.c_str(), flags, width);
    }

    bool PyUILayout::button_styled(const std::string& label, const std::string& style,
                                   std::tuple<float, float> size) {
        using lfs::vis::gui::widgets::ButtonStyle;
        using lfs::vis::gui::widgets::ColoredButton;

        static const std::unordered_map<std::string_view, ButtonStyle> STYLE_MAP = {
            {"primary", ButtonStyle::Primary},
            {"success", ButtonStyle::Success},
            {"warning", ButtonStyle::Warning},
            {"error", ButtonStyle::Error},
            {"secondary", ButtonStyle::Secondary},
        };

        const auto it = STYLE_MAP.find(style);
        const ButtonStyle btn_style = (it != STYLE_MAP.end()) ? it->second : ButtonStyle::Secondary;
        return ColoredButton(label.c_str(), btn_style, {std::get<0>(size), std::get<1>(size)});
    }

    void PyUILayout::push_item_width(const float width) { ImGui::PushItemWidth(width); }
    void PyUILayout::pop_item_width() { ImGui::PopItemWidth(); }

    // Plots
    void PyUILayout::plot_lines(const std::string& label, const std::vector<float>& values,
                                float scale_min, float scale_max, std::tuple<float, float> size) {
        if (values.empty())
            return;
        ImGui::PlotLines(label.c_str(), values.data(), static_cast<int>(values.size()),
                         0, nullptr, scale_min, scale_max,
                         ImVec2(std::get<0>(size), std::get<1>(size)));
    }

    // Selectable
    bool PyUILayout::selectable(const std::string& label, bool selected, float height) {
        ImGuiSelectableFlags flags = ImGuiSelectableFlags_SpanAllColumns;
        ImVec2 size = {0, height};
        return ImGui::Selectable(label.c_str(), selected, flags, size);
    }

    bool PyUILayout::begin_context_menu(const std::string& id) {
        const auto& t = lfs::vis::theme();
        t.pushContextMenuStyle();
        const char* str_id = id.empty() ? nullptr : id.c_str();
        if (!ImGui::BeginPopupContextItem(str_id)) {
            lfs::vis::Theme::popContextMenuStyle();
            return false;
        }
        return true;
    }

    void PyUILayout::end_context_menu() {
        ImGui::EndPopup();
        lfs::vis::Theme::popContextMenuStyle();
    }

    bool PyUILayout::begin_popup(const std::string& id) {
        return ImGui::BeginPopup(id.c_str());
    }

    void PyUILayout::open_popup(const std::string& id) {
        ImGui::OpenPopup(id.c_str());
    }

    void PyUILayout::end_popup() {
        ImGui::EndPopup();
    }

    bool PyUILayout::menu_item(const std::string& label, bool enabled) {
        return ImGui::MenuItem(label.c_str(), nullptr, false, enabled);
    }

    bool PyUILayout::begin_menu(const std::string& label) {
        if (ImGui::BeginMenu(label.c_str())) {
            ++menu_depth_;
            return true;
        }
        return false;
    }

    void PyUILayout::end_menu() {
        assert(menu_depth_ > 0);
        --menu_depth_;
        ImGui::EndMenu();
    }

    std::tuple<bool, std::string> PyUILayout::input_text_enter(const std::string& label, const std::string& value) {
        char buffer[256];
        std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\0';
        const bool entered = ImGui::InputText(label.c_str(), buffer, sizeof(buffer),
                                              ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
        return {entered, std::string(buffer)};
    }

    void PyUILayout::set_keyboard_focus_here() {
        ImGui::SetKeyboardFocusHere();
    }

    bool PyUILayout::is_window_focused() const {
        return ImGui::IsWindowFocused();
    }

    bool PyUILayout::is_window_hovered() const {
        return ImGui::IsWindowHovered();
    }

    void PyUILayout::capture_keyboard_from_app(bool capture) {
        ImGui::GetIO().WantCaptureKeyboard = capture;
    }

    void PyUILayout::capture_mouse_from_app(bool capture) {
        ImGui::GetIO().WantCaptureMouse = capture;
    }

    void PyUILayout::set_scroll_here_y(float center_y_ratio) {
        ImGui::SetScrollHereY(center_y_ratio);
    }

    std::tuple<float, float> PyUILayout::get_cursor_screen_pos() const {
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        return {pos.x, pos.y};
    }

    std::tuple<float, float> PyUILayout::get_window_pos() const {
        const ImVec2 pos = ImGui::GetWindowPos();
        return {pos.x, pos.y};
    }

    float PyUILayout::get_window_width() const {
        return ImGui::GetWindowWidth();
    }

    float PyUILayout::get_text_line_height() const {
        return ImGui::GetTextLineHeight();
    }

    bool PyUILayout::begin_popup_modal(const std::string& title) {
        constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking |
                                           ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
        return ImGui::BeginPopupModal(title.c_str(), nullptr, flags);
    }

    void PyUILayout::end_popup_modal() {
        ImGui::EndPopup();
    }

    void PyUILayout::close_current_popup() {
        ImGui::CloseCurrentPopup();
    }

    void PyUILayout::set_next_window_pos_center() {
        const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, {0.5f, 0.5f});
    }

    void PyUILayout::set_next_window_pos_viewport_center(const bool always) {
        const auto [vp_x, vp_y] = get_viewport_pos();
        const auto [vp_w, vp_h] = get_viewport_size();
        const ImVec2 center{vp_x + vp_w * 0.5f, vp_y + vp_h * 0.5f};
        ImGui::SetNextWindowPos(center, always ? ImGuiCond_Always : ImGuiCond_Appearing, {0.5f, 0.5f});
    }

    void PyUILayout::set_next_window_focus() {
        ImGui::SetNextWindowFocus();
    }

    void PyUILayout::push_modal_style() {
        lfs::vis::theme().pushModalStyle();
    }

    void PyUILayout::pop_modal_style() {
        lfs::vis::theme().popModalStyle();
    }

    std::tuple<float, float> PyUILayout::get_content_region_avail() {
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        return {avail.x, avail.y};
    }

    std::tuple<float, float> PyUILayout::get_cursor_pos() {
        const ImVec2 pos = ImGui::GetCursorPos();
        return {pos.x, pos.y};
    }

    void PyUILayout::set_cursor_pos_x(float x) {
        ImGui::SetCursorPosX(x);
    }

    std::tuple<float, float> PyUILayout::calc_text_size(const std::string& text) {
        const ImVec2 size = ImGui::CalcTextSize(text.c_str());
        return {size.x, size.y};
    }

    void PyUILayout::begin_disabled(bool disabled) {
        ImGui::BeginDisabled(disabled);
    }

    void PyUILayout::end_disabled() {
        ImGui::EndDisabled();
    }

    // Images
    void PyUILayout::image(uint64_t texture_id, std::tuple<float, float> size,
                           std::tuple<float, float, float, float> tint) {
        ImGui::Image(static_cast<ImTextureID>(texture_id),
                     {std::get<0>(size), std::get<1>(size)},
                     {0, 0}, {1, 1},
                     tuple_to_imvec4(tint), {0, 0, 0, 0});
    }

    void PyUILayout::image_uv(uint64_t texture_id, std::tuple<float, float> size,
                              std::tuple<float, float> uv0, std::tuple<float, float> uv1,
                              std::tuple<float, float, float, float> tint) {
        ImGui::Image(static_cast<ImTextureID>(texture_id),
                     {std::get<0>(size), std::get<1>(size)},
                     {std::get<0>(uv0), std::get<1>(uv0)},
                     {std::get<0>(uv1), std::get<1>(uv1)},
                     tuple_to_imvec4(tint), {0, 0, 0, 0});
    }

    bool PyUILayout::image_button(const std::string& id, uint64_t texture_id, std::tuple<float, float> size,
                                  std::tuple<float, float, float, float> tint) {
        return ImGui::ImageButton(id.c_str(), static_cast<ImTextureID>(texture_id),
                                  {std::get<0>(size), std::get<1>(size)},
                                  {0, 0}, {1, 1},
                                  {0, 0, 0, 0}, tuple_to_imvec4(tint));
    }

    bool PyUILayout::toolbar_button(const std::string& id, uint64_t texture_id, std::tuple<float, float> size,
                                    bool selected, bool disabled, const std::string& tooltip) {
        const auto btn_size = ImVec2{std::get<0>(size), std::get<1>(size)};

        if (disabled) {
            ImGui::BeginDisabled();
        }

        const bool clicked = lfs::vis::gui::widgets::IconButton(
            id.c_str(), static_cast<unsigned int>(texture_id), btn_size, selected, id.c_str());

        if (disabled) {
            ImGui::EndDisabled();
        }

        if (!tooltip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
            lfs::vis::gui::widgets::SetThemedTooltip(tooltip.c_str());
        }

        return clicked;
    }

    // Drag-drop
    bool PyUILayout::begin_drag_drop_source() {
        return ImGui::BeginDragDropSource(ImGuiDragDropFlags_None);
    }

    void PyUILayout::set_drag_drop_payload(const std::string& type, const std::string& data) {
        ImGui::SetDragDropPayload(type.c_str(), data.data(), data.size());
    }

    void PyUILayout::end_drag_drop_source() {
        ImGui::EndDragDropSource();
    }

    bool PyUILayout::begin_drag_drop_target() {
        return ImGui::BeginDragDropTarget();
    }

    std::optional<std::string> PyUILayout::accept_drag_drop_payload(const std::string& type) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(type.c_str())) {
            return std::string(static_cast<const char*>(payload->Data), payload->DataSize);
        }
        return std::nullopt;
    }

    void PyUILayout::end_drag_drop_target() {
        ImGui::EndDragDropTarget();
    }

    // Misc
    void PyUILayout::progress_bar(float fraction, const std::string& overlay, float width,
                                  float height) {
        const float w = (width > 0) ? width : -FLT_MIN;
        const float h = (height > 0) ? height : 0;
        ImGui::ProgressBar(fraction, {w, h}, overlay.empty() ? nullptr : overlay.c_str());
    }

    void PyUILayout::set_tooltip(const std::string& text) {
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", text.c_str());
        }
    }

    bool PyUILayout::is_item_hovered() {
        return ImGui::IsItemHovered();
    }

    bool PyUILayout::is_item_clicked(int button) {
        return ImGui::IsItemClicked(button);
    }

    bool PyUILayout::is_mouse_double_clicked(int button) {
        return ImGui::IsMouseDoubleClicked(button);
    }

    bool PyUILayout::is_mouse_dragging(int button) {
        return ImGui::IsMouseDragging(button);
    }

    float PyUILayout::get_mouse_wheel() {
        return ImGui::GetIO().MouseWheel;
    }

    std::tuple<float, float> PyUILayout::get_mouse_delta() {
        const auto& delta = ImGui::GetIO().MouseDelta;
        return {delta.x, delta.y};
    }

    bool PyUILayout::invisible_button(const std::string& id, std::tuple<float, float> size) {
        return ImGui::InvisibleButton(id.c_str(), {std::get<0>(size), std::get<1>(size)});
    }

    bool PyUILayout::is_item_active() {
        return ImGui::IsItemActive();
    }

    void PyUILayout::set_cursor_pos(std::tuple<float, float> pos) {
        ImGui::SetCursorPos({std::get<0>(pos), std::get<1>(pos)});
    }

    bool PyUILayout::begin_child(const std::string& id, std::tuple<float, float> size, bool border) {
        return ImGui::BeginChild(id.c_str(), {std::get<0>(size), std::get<1>(size)}, border ? ImGuiChildFlags_Border : ImGuiChildFlags_None);
    }

    void PyUILayout::end_child() {
        ImGui::EndChild();
    }

    bool PyUILayout::begin_menu_bar() {
        return ImGui::BeginMenuBar();
    }

    void PyUILayout::end_menu_bar() {
        ImGui::EndMenuBar();
    }

    bool PyUILayout::menu_item_toggle(const std::string& label, const std::string& shortcut, bool selected) {
        return ImGui::MenuItem(label.c_str(), shortcut.c_str(), selected);
    }

    bool PyUILayout::menu_item_shortcut(const std::string& label, const std::string& shortcut, bool enabled) {
        return ImGui::MenuItem(label.c_str(), shortcut.c_str(), false, enabled);
    }

    void PyUILayout::push_id(const std::string& id) {
        ImGui::PushID(id.c_str());
    }

    void PyUILayout::push_id_int(int id) {
        ImGui::PushID(id);
    }

    void PyUILayout::pop_id() {
        ImGui::PopID();
    }

    // Window control
    bool PyUILayout::begin_window(const std::string& title, int flags) {
        return ImGui::Begin(title.c_str(), nullptr, flags);
    }

    std::tuple<bool, bool> PyUILayout::begin_window_closable(const std::string& title, int flags) {
        bool open = true;
        const bool visible = ImGui::Begin(title.c_str(), &open, flags | ImGuiWindowFlags_NoDocking);
        return {visible, open};
    }

    void PyUILayout::end_window() {
        ImGui::End();
    }

    void PyUILayout::push_window_style() {
        constexpr float WINDOW_ROUNDING = 8.0f;
        constexpr float WINDOW_PADDING = 16.0f;
        constexpr float WINDOW_BG_ALPHA = 0.98f;

        const auto& t = lfs::vis::theme();
        const float scale = python::get_shared_dpi_scale();
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, WINDOW_ROUNDING * scale);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(WINDOW_PADDING * scale, WINDOW_PADDING * scale));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(t.palette.surface.x, t.palette.surface.y,
                                                        t.palette.surface.z, WINDOW_BG_ALPHA));
    }

    void PyUILayout::pop_window_style() {
        ImGui::PopStyleColor(1);
        ImGui::PopStyleVar(2);
    }

    void PyUILayout::set_next_window_pos(std::tuple<float, float> pos, bool first_use) {
        ImGui::SetNextWindowPos({std::get<0>(pos), std::get<1>(pos)}, first_use ? ImGuiCond_FirstUseEver : ImGuiCond_Always);
    }

    void PyUILayout::set_next_window_size(std::tuple<float, float> size, bool first_use) {
        ImGui::SetNextWindowSize({std::get<0>(size), std::get<1>(size)}, first_use ? ImGuiCond_FirstUseEver : ImGuiCond_Always);
    }

    void PyUILayout::set_next_window_pos_centered(bool first_use) {
        const auto* viewport = ImGui::GetMainViewport();
        const ImVec2 center{viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                            viewport->WorkPos.y + viewport->WorkSize.y * 0.5f};
        ImGui::SetNextWindowPos(center, first_use ? ImGuiCond_FirstUseEver : ImGuiCond_Always, {0.5f, 0.5f});
    }

    std::tuple<float, float> PyUILayout::get_viewport_pos() {
        if (has_viewport_bounds()) {
            float x, y, w, h;
            get_viewport_bounds(x, y, w, h);
            return {x, y};
        }
        const auto* viewport = ImGui::GetMainViewport();
        return {viewport->WorkPos.x, viewport->WorkPos.y};
    }

    std::tuple<float, float> PyUILayout::get_viewport_size() {
        if (has_viewport_bounds()) {
            float x, y, w, h;
            get_viewport_bounds(x, y, w, h);
            return {w, h};
        }
        const auto* viewport = ImGui::GetMainViewport();
        return {viewport->WorkSize.x, viewport->WorkSize.y};
    }

    float PyUILayout::get_dpi_scale() {
        return python::get_shared_dpi_scale();
    }

    void PyUILayout::set_mouse_cursor_hand() {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }

    void PyUILayout::set_next_window_bg_alpha(float alpha) {
        ImGui::SetNextWindowBgAlpha(alpha);
    }

    namespace {
        ImGuiStyleVar string_to_style_var(const std::string& var) {
            static const std::unordered_map<std::string, ImGuiStyleVar> map = {
                {"WindowRounding", ImGuiStyleVar_WindowRounding},
                {"WindowPadding", ImGuiStyleVar_WindowPadding},
                {"FrameRounding", ImGuiStyleVar_FrameRounding},
                {"FramePadding", ImGuiStyleVar_FramePadding},
                {"ItemSpacing", ImGuiStyleVar_ItemSpacing},
                {"ItemInnerSpacing", ImGuiStyleVar_ItemInnerSpacing},
                {"IndentSpacing", ImGuiStyleVar_IndentSpacing},
                {"ScrollbarSize", ImGuiStyleVar_ScrollbarSize},
                {"ScrollbarRounding", ImGuiStyleVar_ScrollbarRounding},
                {"GrabMinSize", ImGuiStyleVar_GrabMinSize},
                {"GrabRounding", ImGuiStyleVar_GrabRounding},
                {"TabRounding", ImGuiStyleVar_TabRounding},
                {"ChildRounding", ImGuiStyleVar_ChildRounding},
                {"PopupRounding", ImGuiStyleVar_PopupRounding},
                {"WindowBorderSize", ImGuiStyleVar_WindowBorderSize},
                {"FrameBorderSize", ImGuiStyleVar_FrameBorderSize},
            };
            auto it = map.find(var);
            if (it != map.end())
                return it->second;
            LOG_WARN("Unknown ImGui style var: {}", var);
            return ImGuiStyleVar_WindowRounding;
        }

        ImGuiCol string_to_style_color(const std::string& col) {
            std::string lower = col;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

            static const std::unordered_map<std::string, ImGuiCol> map = {
                {"windowbg", ImGuiCol_WindowBg},
                {"childbg", ImGuiCol_ChildBg},
                {"popupbg", ImGuiCol_PopupBg},
                {"border", ImGuiCol_Border},
                {"framebg", ImGuiCol_FrameBg},
                {"framebghovered", ImGuiCol_FrameBgHovered},
                {"framebgactive", ImGuiCol_FrameBgActive},
                {"button", ImGuiCol_Button},
                {"buttonhovered", ImGuiCol_ButtonHovered},
                {"buttonactive", ImGuiCol_ButtonActive},
                {"header", ImGuiCol_Header},
                {"headerhovered", ImGuiCol_HeaderHovered},
                {"headeractive", ImGuiCol_HeaderActive},
                {"text", ImGuiCol_Text},
                {"textdisabled", ImGuiCol_TextDisabled},
                {"plothistogram", ImGuiCol_PlotHistogram},
                {"plothistogramhovered", ImGuiCol_PlotHistogramHovered},
            };
            auto it = map.find(lower);
            if (it != map.end())
                return it->second;
            LOG_WARN("Unknown ImGui style color: {}", col);
            return ImGuiCol_WindowBg;
        }
    } // namespace

    void PyUILayout::push_style_var_float(const std::string& var, float value) {
        ImGui::PushStyleVar(string_to_style_var(var), value);
    }

    void PyUILayout::push_style_var_vec2(const std::string& var, std::tuple<float, float> value) {
        ImGui::PushStyleVar(string_to_style_var(var), {std::get<0>(value), std::get<1>(value)});
    }

    void PyUILayout::pop_style_var(int count) {
        ImGui::PopStyleVar(count);
    }

    void PyUILayout::push_style_color(const std::string& col, std::tuple<float, float, float, float> color) {
        ImGui::PushStyleColor(string_to_style_color(col), tuple_to_imvec4(color));
    }

    void PyUILayout::pop_style_color(int count) {
        ImGui::PopStyleColor(count);
    }

    std::tuple<bool, nb::object> PyUILayout::prop(nb::object data,
                                                  const std::string& prop_id,
                                                  std::optional<std::string> text) {
        if (!nb::hasattr(data, "get_all_properties")) {
            throw std::runtime_error("prop() requires object with get_all_properties() method");
        }

        nb::object method = data.attr("get_all_properties");
        if (!PyCallable_Check(method.ptr())) {
            throw std::runtime_error("get_all_properties must be callable");
        }

        nb::dict all_props = nb::cast<nb::dict>(method());
        nb::str prop_key(prop_id.c_str());
        if (!all_props.contains(prop_key)) {
            throw std::runtime_error("Unknown property '" + prop_id + "'");
        }

        nb::object prop_desc = all_props[prop_key];
        nb::object current_value = data.attr(prop_id.c_str());
        const std::string prop_type = nb::cast<std::string>(
            nb::object(prop_desc.attr("__class__").attr("__name__")));

        std::string prop_name_str;
        if (nb::hasattr(prop_desc, "name")) {
            prop_name_str = nb::cast<std::string>(prop_desc.attr("name"));
        }
        std::string display_name = text.value_or(
            !prop_name_str.empty() ? prop_name_str : prop_id);

        std::string description;
        if (nb::hasattr(prop_desc, "description")) {
            description = nb::cast<std::string>(prop_desc.attr("description"));
        }

        bool changed = false;
        nb::object new_value = current_value;

        if (prop_type == "FloatProperty") {
            float v = nb::cast<float>(current_value);
            float min_v = nb::cast<float>(prop_desc.attr("min"));
            float max_v = nb::cast<float>(prop_desc.attr("max"));
            float step = nb::cast<float>(prop_desc.attr("step"));

            bool use_slider = (min_v != -std::numeric_limits<float>::infinity() &&
                               max_v != std::numeric_limits<float>::infinity());

            if (use_slider) {
                auto [c, nv] = slider_float(display_name, v, min_v, max_v);
                changed = c;
                new_value = nb::cast(nv);
            } else {
                auto [c, nv] = drag_float(display_name, v, step, min_v, max_v);
                changed = c;
                new_value = nb::cast(nv);
            }
        } else if (prop_type == "IntProperty") {
            int v = nb::cast<int>(current_value);
            int min_v = nb::cast<int>(prop_desc.attr("min"));
            int max_v = nb::cast<int>(prop_desc.attr("max"));

            bool use_slider = (min_v != -(1 << 30) && max_v != (1 << 30));

            if (use_slider) {
                auto [c, nv] = slider_int(display_name, v, min_v, max_v);
                changed = c;
                new_value = nb::cast(nv);
            } else {
                auto [c, nv] = drag_int(display_name, v, 1.0f, min_v, max_v);
                changed = c;
                new_value = nb::cast(nv);
            }
        } else if (prop_type == "BoolProperty") {
            bool v = nb::cast<bool>(current_value);
            auto [c, nv] = checkbox(display_name, v);
            changed = c;
            new_value = nb::cast(nv);
        } else if (prop_type == "StringProperty") {
            std::string v = nb::cast<std::string>(current_value);
            auto [c, nv] = input_text(display_name, v);
            changed = c;
            new_value = nb::cast(nv);
        } else if (prop_type == "EnumProperty") {
            nb::object items_obj = prop_desc.attr("items");
            std::vector<std::string> items;
            int current_idx = 0;
            std::string current_id = nb::cast<std::string>(current_value);

            size_t idx = 0;
            for (auto item : items_obj) {
                nb::tuple t = nb::cast<nb::tuple>(item);
                std::string identifier = nb::cast<std::string>(t[0]);
                std::string lbl = nb::cast<std::string>(t[1]);
                items.push_back(lbl);
                if (identifier == current_id) {
                    current_idx = static_cast<int>(idx);
                }
                idx++;
            }

            auto [c, new_idx] = combo(display_name, current_idx, items);
            changed = c;
            if (changed && new_idx >= 0 && new_idx < static_cast<int>(items.size())) {
                nb::tuple t = nb::cast<nb::tuple>(items_obj[new_idx]);
                new_value = t[0];
            } else {
                new_value = nb::cast(current_id);
            }
        } else if (prop_type == "FloatVectorProperty") {
            nb::tuple t = nb::cast<nb::tuple>(current_value);
            int size = nb::cast<int>(prop_desc.attr("size"));
            std::string subtype;
            if (nb::hasattr(prop_desc, "subtype")) {
                subtype = nb::cast<std::string>(prop_desc.attr("subtype"));
            }

            if (size == 3) {
                float v[3] = {nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2])};
                if (subtype == "COLOR" || subtype == "COLOR_GAMMA") {
                    changed = ImGui::ColorEdit3(display_name.c_str(), v);
                } else {
                    changed = ImGui::DragFloat3(display_name.c_str(), v, 0.01f);
                }
                new_value = nb::make_tuple(v[0], v[1], v[2]);
            } else if (size == 4) {
                float v[4] = {nb::cast<float>(t[0]), nb::cast<float>(t[1]),
                              nb::cast<float>(t[2]), nb::cast<float>(t[3])};
                if (subtype == "COLOR" || subtype == "COLOR_GAMMA") {
                    changed = ImGui::ColorEdit4(display_name.c_str(), v);
                } else {
                    changed = ImGui::DragFloat4(display_name.c_str(), v, 0.01f);
                }
                new_value = nb::make_tuple(v[0], v[1], v[2], v[3]);
            } else if (size == 2) {
                float v[2] = {nb::cast<float>(t[0]), nb::cast<float>(t[1])};
                changed = ImGui::DragFloat2(display_name.c_str(), v, 0.01f);
                new_value = nb::make_tuple(v[0], v[1]);
            }
        } else if (prop_type == "IntVectorProperty") {
            nb::tuple t = nb::cast<nb::tuple>(current_value);
            int size = nb::cast<int>(prop_desc.attr("size"));

            if (size == 3) {
                int v[3] = {nb::cast<int>(t[0]), nb::cast<int>(t[1]), nb::cast<int>(t[2])};
                changed = ImGui::DragInt3(display_name.c_str(), v);
                new_value = nb::make_tuple(v[0], v[1], v[2]);
            } else if (size == 4) {
                int v[4] = {nb::cast<int>(t[0]), nb::cast<int>(t[1]),
                            nb::cast<int>(t[2]), nb::cast<int>(t[3])};
                changed = ImGui::DragInt4(display_name.c_str(), v);
                new_value = nb::make_tuple(v[0], v[1], v[2], v[3]);
            } else if (size == 2) {
                int v[2] = {nb::cast<int>(t[0]), nb::cast<int>(t[1])};
                changed = ImGui::DragInt2(display_name.c_str(), v);
                new_value = nb::make_tuple(v[0], v[1]);
            }
        } else if (prop_type == "TensorProperty") {
            std::string info_str = "None";
            if (!current_value.is_none()) {
                try {
                    nb::object shape = current_value.attr("shape");
                    nb::object dtype = current_value.attr("dtype");
                    nb::object device = current_value.attr("device");

                    std::string shape_str = "[";
                    size_t idx = 0;
                    for (auto dim : shape) {
                        if (idx > 0)
                            shape_str += ", ";
                        shape_str += std::to_string(nb::cast<int64_t>(dim));
                        ++idx;
                    }
                    shape_str += "]";

                    size_t num_elements = 1;
                    for (auto dim : shape) {
                        num_elements *= nb::cast<size_t>(dim);
                    }
                    std::string dtype_str = nb::cast<std::string>(nb::str(dtype));
                    size_t bytes_per_elem = 4;
                    if (dtype_str.find("float32") != std::string::npos)
                        bytes_per_elem = 4;
                    else if (dtype_str.find("float16") != std::string::npos)
                        bytes_per_elem = 2;
                    else if (dtype_str.find("int32") != std::string::npos)
                        bytes_per_elem = 4;
                    else if (dtype_str.find("int64") != std::string::npos)
                        bytes_per_elem = 8;
                    else if (dtype_str.find("bool") != std::string::npos)
                        bytes_per_elem = 1;

                    double size_mb = static_cast<double>(num_elements * bytes_per_elem) / (1024.0 * 1024.0);
                    char size_buf[32];
                    if (size_mb >= 1.0) {
                        snprintf(size_buf, sizeof(size_buf), "%.1f MB", size_mb);
                    } else {
                        snprintf(size_buf, sizeof(size_buf), "%.1f KB", size_mb * 1024.0);
                    }

                    info_str = "Tensor(" + shape_str + ", " + dtype_str + ", " +
                               nb::cast<std::string>(nb::str(device)) + ") " + size_buf;
                } catch (...) {
                    info_str = "Tensor(...)";
                }
            }
            ImGui::TextDisabled("%s: %s", display_name.c_str(), info_str.c_str());
        }

        if (!description.empty() && is_item_hovered()) {
            set_tooltip(description);
        }

        if (changed) {
            nb::setattr(data, prop_id.c_str(), new_value);

            if (nb::hasattr(prop_desc, "update")) {
                nb::object update_cb = prop_desc.attr("update");
                if (!update_cb.is_none() && PyCallable_Check(update_cb.ptr())) {
                    try {
                        update_cb(data, nb::none());
                    } catch (const nb::python_error& e) {
                        LOG_ERROR("Property update callback error: {}", e.what());
                    }
                }
            }
        }

        return {changed, new_value};
    }

    // Register UI classes with nanobind module
    void register_ui(nb::module_& m) {
        // Call sub-registration functions
        register_ui_context(m);
        register_ui_theme(m);
        register_ui_panels(m);
        register_ui_hooks(m);
        register_ui_menus(m);
        register_ui_operators(m);
        register_ui_modals(m);

        // Hot-reload redraw request functions
        m.def(
            "request_redraw", []() { g_redraw_requested = true; }, "Request a UI redraw on next frame");
        m.def(
            "consume_redraw_request", []() {
                bool val = g_redraw_requested.exchange(false);
                return val;
            },
            "Consume and return pending redraw request flag");

        nb::class_<PyEvent>(m, "Event")
            .def(nb::init<>(), "Create a default Event")
            .def_rw("type", &PyEvent::type, "Event type ('MOUSEMOVE', 'LEFTMOUSE', 'KEY_A', etc.)")
            .def_rw("value", &PyEvent::value, "Event value ('PRESS', 'RELEASE', 'NOTHING')")
            .def_rw("mouse_x", &PyEvent::mouse_x, "Mouse X position")
            .def_rw("mouse_y", &PyEvent::mouse_y, "Mouse Y position")
            .def_rw("mouse_region_x", &PyEvent::mouse_region_x, "Mouse X position relative to region")
            .def_rw("mouse_region_y", &PyEvent::mouse_region_y, "Mouse Y position relative to region")
            .def_rw("delta_x", &PyEvent::delta_x, "Mouse delta X (for drag operations)")
            .def_rw("delta_y", &PyEvent::delta_y, "Mouse delta Y (for drag operations)")
            .def_rw("scroll_x", &PyEvent::scroll_x, "Scroll X offset")
            .def_rw("scroll_y", &PyEvent::scroll_y, "Scroll Y offset")
            .def_rw("shift", &PyEvent::shift, "Shift modifier is held")
            .def_rw("ctrl", &PyEvent::ctrl, "Ctrl modifier is held")
            .def_rw("alt", &PyEvent::alt, "Alt modifier is held")
            .def_rw("pressure", &PyEvent::pressure, "Tablet pressure (1.0 for mouse)")
            .def_rw("over_gui", &PyEvent::over_gui, "Mouse is over GUI element")
            .def_rw("key_code", &PyEvent::key_code, "Raw key code for KEY events")
            .def(
                "__repr__", [](const PyEvent& e) {
                    return "<Event type='" + e.type + "' value='" + e.value + "'>";
                },
                "Return string representation of the event");

        nb::class_<PySubLayout>(m, "SubLayout")
            .def("__enter__", &PySubLayout::enter, nb::rv_policy::reference)
            .def("__exit__", [](PySubLayout& s, nb::args) {
                s.exit();
                return false;
            })
            .def_prop_rw("enabled", &PySubLayout::get_enabled, &PySubLayout::set_enabled)
            .def_prop_rw("active", &PySubLayout::get_active, &PySubLayout::set_active)
            .def_prop_rw("alert", &PySubLayout::get_alert, &PySubLayout::set_alert)
            .def_prop_rw("scale_x", &PySubLayout::get_scale_x, &PySubLayout::set_scale_x)
            .def_prop_rw("scale_y", &PySubLayout::get_scale_y, &PySubLayout::set_scale_y)
            .def("row", &PySubLayout::row)
            .def("column", &PySubLayout::column)
            .def("split", &PySubLayout::split, nb::arg("factor") = 0.5f)
            .def("box", &PySubLayout::box)
            .def("grid_flow", &PySubLayout::grid_flow, nb::arg("columns") = 0, nb::arg("even_columns") = true, nb::arg("even_rows") = true)
            .def("prop_enum", &PySubLayout::prop_enum, nb::arg("data"), nb::arg("prop_id"), nb::arg("value"), nb::arg("text") = "")
            .def("label", &PySubLayout::label, nb::arg("text"))
            .def("button", &PySubLayout::button, nb::arg("label"), nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            .def("button_styled", &PySubLayout::button_styled, nb::arg("label"), nb::arg("style"), nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            .def("prop", &PySubLayout::prop, nb::arg("data"), nb::arg("prop_id"), nb::arg("text") = nb::none())
            .def("checkbox", &PySubLayout::checkbox, nb::arg("label"), nb::arg("value"))
            .def("slider_float", &PySubLayout::slider_float, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            .def("slider_int", &PySubLayout::slider_int, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            .def("drag_float", &PySubLayout::drag_float, nb::arg("label"), nb::arg("value"), nb::arg("speed") = 1.0f, nb::arg("min") = 0.0f, nb::arg("max") = 0.0f)
            .def("drag_int", &PySubLayout::drag_int, nb::arg("label"), nb::arg("value"), nb::arg("speed") = 1.0f, nb::arg("min") = 0, nb::arg("max") = 0)
            .def("input_text", &PySubLayout::input_text, nb::arg("label"), nb::arg("value"))
            .def("combo", &PySubLayout::combo, nb::arg("label"), nb::arg("current_idx"), nb::arg("items"))
            .def("separator", &PySubLayout::separator)
            .def("spacing", &PySubLayout::spacing)
            .def("heading", &PySubLayout::heading, nb::arg("text"))
            .def("collapsing_header", &PySubLayout::collapsing_header, nb::arg("label"), nb::arg("default_open") = false)
            .def("tree_node", &PySubLayout::tree_node, nb::arg("label"))
            .def("tree_pop", &PySubLayout::tree_pop)
            .def("progress_bar", &PySubLayout::progress_bar, nb::arg("fraction"), nb::arg("overlay") = "", nb::arg("width") = 0.0f, nb::arg("height") = 0.0f)
            .def("text_colored", &PySubLayout::text_colored, nb::arg("text"), nb::arg("color"))
            .def("text_wrapped", &PySubLayout::text_wrapped, nb::arg("text"))
            .def("begin_table", &PySubLayout::begin_table, nb::arg("id"), nb::arg("columns"))
            .def("input_float", &PySubLayout::input_float, nb::arg("label"), nb::arg("value"), nb::arg("step") = 0.0f, nb::arg("step_fast") = 0.0f, nb::arg("format") = "%.3f")
            .def("input_int", &PySubLayout::input_int, nb::arg("label"), nb::arg("value"), nb::arg("step") = 1, nb::arg("step_fast") = 100)
            .def("input_int_formatted", &PySubLayout::input_int_formatted, nb::arg("label"), nb::arg("value"), nb::arg("step") = 0, nb::arg("step_fast") = 0)
            .def("stepper_float", &PySubLayout::stepper_float, nb::arg("label"), nb::arg("value"), nb::arg("steps") = std::vector<float>{1.0f, 0.1f, 0.01f}, "Float input with increment/decrement buttons, returns (changed, value)")
            .def("radio_button", &PySubLayout::radio_button, nb::arg("label"), nb::arg("current"), nb::arg("value"))
            .def("small_button", &PySubLayout::small_button, nb::arg("label"))
            .def("selectable", &PySubLayout::selectable, nb::arg("label"), nb::arg("selected") = false, nb::arg("height") = 0.0f)
            .def("color_edit3", &PySubLayout::color_edit3, nb::arg("label"), nb::arg("color"))
            .def("text_disabled", &PySubLayout::text_disabled, nb::arg("text"))
            .def("listbox", &PySubLayout::listbox, nb::arg("label"), nb::arg("current_idx"), nb::arg("items"), nb::arg("height_items") = -1)
            .def("image", &PySubLayout::image, nb::arg("texture_id"), nb::arg("size"), nb::arg("tint") = std::make_tuple(1.0f, 1.0f, 1.0f, 1.0f))
            .def("image_button", &PySubLayout::image_button, nb::arg("id"), nb::arg("texture_id"), nb::arg("size"), nb::arg("tint") = std::make_tuple(1.0f, 1.0f, 1.0f, 1.0f))
            .def("input_text_with_hint", &PySubLayout::input_text_with_hint, nb::arg("label"), nb::arg("hint"), nb::arg("value"))
            .def("input_text_enter", &PySubLayout::input_text_enter, nb::arg("label"), nb::arg("value"))
            .def("table_setup_column", &PySubLayout::table_setup_column, nb::arg("label"), nb::arg("width") = 0.0f)
            .def("table_next_row", &PySubLayout::table_next_row)
            .def("table_next_column", &PySubLayout::table_next_column)
            .def("table_headers_row", &PySubLayout::table_headers_row)
            .def("end_table", &PySubLayout::end_table)
            .def("push_item_width", &PySubLayout::push_item_width, nb::arg("width"))
            .def("pop_item_width", &PySubLayout::pop_item_width)
            .def("set_tooltip", &PySubLayout::set_tooltip, nb::arg("text"))
            .def("is_item_hovered", &PySubLayout::is_item_hovered)
            .def("is_item_clicked", &PySubLayout::is_item_clicked, nb::arg("button") = 0)
            .def("begin_disabled", &PySubLayout::begin_disabled, nb::arg("disabled") = true)
            .def("end_disabled", &PySubLayout::end_disabled)
            .def("same_line", &PySubLayout::same_line, nb::arg("offset") = 0.0f, nb::arg("spacing") = -1.0f)
            .def("push_id", &PySubLayout::push_id, nb::arg("id"))
            .def("pop_id", &PySubLayout::pop_id)
            .def("begin_child", &PySubLayout::begin_child, nb::arg("id"), nb::arg("size"), nb::arg("border") = false)
            .def("end_child", &PySubLayout::end_child)
            .def("begin_context_menu", &PySubLayout::begin_context_menu, nb::arg("id") = "")
            .def("end_context_menu", &PySubLayout::end_context_menu)
            .def("menu_item", &PySubLayout::menu_item, nb::arg("label"), nb::arg("enabled") = true)
            .def("begin_menu", &PySubLayout::begin_menu, nb::arg("label"))
            .def("end_menu", &PySubLayout::end_menu)
            .def("get_content_region_avail", &PySubLayout::get_content_region_avail)
            .def("__getattr__", [](PySubLayout& self, const std::string& name) -> nb::object {
                nb::object parent_obj = nb::cast(self.parent(), nb::rv_policy::reference);
                if (!nb::hasattr(parent_obj, name.c_str()))
                    throw nb::attribute_error(name.c_str());
                nb::object method = parent_obj.attr(name.c_str());
                PySubLayout* self_ptr = &self;
                return nb::cpp_function([self_ptr, method](nb::args args, nb::kwargs kwargs) {
                    self_ptr->advance_child();
                    self_ptr->apply_state();
                    try {
                        nb::object result = method(*args, **kwargs);
                        self_ptr->pop_per_item_state();
                        return result;
                    } catch (...) {
                        self_ptr->pop_per_item_state();
                        throw;
                    }
                });
            });

        // PyUILayout - Window flags enum
        nb::class_<PyWindowFlags>(m, "WindowFlags")
            .def_ro_static("None", &PyWindowFlags::None, "No flags set")
            .def_ro_static("NoScrollbar", &PyWindowFlags::NoScrollbar, "Disable scrollbar")
            .def_ro_static("NoScrollWithMouse", &PyWindowFlags::NoScrollWithMouse, "Disable mouse wheel scrolling")
            .def_ro_static("MenuBar", &PyWindowFlags::MenuBar, "Enable menu bar")
            .def_ro_static("NoResize", &PyWindowFlags::NoResize, "Disable window resizing")
            .def_ro_static("NoMove", &PyWindowFlags::NoMove, "Disable window moving")
            .def_ro_static("NoCollapse", &PyWindowFlags::NoCollapse, "Disable window collapsing")
            .def_ro_static("AlwaysAutoResize", &PyWindowFlags::AlwaysAutoResize, "Auto-resize window to fit content")
            .def_ro_static("NoTitleBar", &PyWindowFlags::NoTitleBar, "Hide window title bar")
            .def_ro_static("NoNavFocus", &PyWindowFlags::NoNavFocus, "Disable navigation focus")
            .def_ro_static("NoInputs", &PyWindowFlags::NoInputs, "Disable all input capture")
            .def_ro_static("NoBackground", &PyWindowFlags::NoBackground, "Disable window background")
            .def_ro_static("NoFocusOnAppearing", &PyWindowFlags::NoFocusOnAppearing, "Disable focus when window appears")
            .def_ro_static("NoBringToFrontOnFocus", &PyWindowFlags::NoBringToFrontOnFocus, "Disable bringing window to front on focus");

        nb::class_<PyUILayout>(m, "UILayout")
            .def(nb::init<>(), "Create a UILayout for drawing UI elements")
            .def_prop_ro_static(
                "WindowFlags", [](nb::handle) { return PyWindowFlags{}; }, "Window flags constants")
            // Text
            .def("label", &PyUILayout::label, nb::arg("text"), "Draw a text label")
            .def("label_centered", &PyUILayout::label_centered, nb::arg("text"), "Draw a horizontally centered text label")
            .def("heading", &PyUILayout::heading, nb::arg("text"), "Draw a bold heading text")
            .def("text_colored", &PyUILayout::text_colored, nb::arg("text"), nb::arg("color"), "Draw text with RGBA color tuple")
            .def("text_colored_centered", &PyUILayout::text_colored_centered, nb::arg("text"), nb::arg("color"), "Draw centered text with RGBA color tuple")
            .def("text_selectable", &PyUILayout::text_selectable, nb::arg("text"), nb::arg("height") = 0.0f, "Draw selectable read-only text area")
            .def("text_wrapped", &PyUILayout::text_wrapped, nb::arg("text"), "Draw word-wrapped text")
            .def("text_disabled", &PyUILayout::text_disabled, nb::arg("text"), "Draw greyed-out disabled text")
            .def("bullet_text", &PyUILayout::bullet_text, nb::arg("text"), "Draw text with a bullet point prefix")
            // Buttons
            .def("button", &PyUILayout::button, nb::arg("label"), nb::arg("size") = std::make_tuple(0.0f, 0.0f), "Draw a button, returns True if clicked")
            .def("button_callback", &PyUILayout::button_callback, nb::arg("label"),
                 nb::arg("callback") = nb::none(), nb::arg("size") = std::make_tuple(0.0f, 0.0f), "Draw a button that invokes callback on click")
            .def("small_button", &PyUILayout::small_button, nb::arg("label"), "Draw a small inline button, returns True if clicked")
            .def("checkbox", &PyUILayout::checkbox, nb::arg("label"), nb::arg("value"), "Draw a checkbox, returns (changed, value)")
            .def("radio_button", &PyUILayout::radio_button, nb::arg("label"), nb::arg("current"), nb::arg("value"), "Draw a radio button, returns (clicked, selected_value)")
            // Sliders
            .def("slider_float", &PyUILayout::slider_float, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"), "Draw a float slider, returns (changed, value)")
            .def("slider_int", &PyUILayout::slider_int, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"), "Draw an int slider, returns (changed, value)")
            .def("slider_float2", &PyUILayout::slider_float2, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"), "Draw a 2-component float slider, returns (changed, value)")
            .def("slider_float3", &PyUILayout::slider_float3, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"), "Draw a 3-component float slider, returns (changed, value)")
            // Drags
            .def("drag_float", &PyUILayout::drag_float, nb::arg("label"), nb::arg("value"),
                 nb::arg("speed") = 1.0f, nb::arg("min") = 0.0f, nb::arg("max") = 0.0f, "Draw a draggable float input, returns (changed, value)")
            .def("drag_int", &PyUILayout::drag_int, nb::arg("label"), nb::arg("value"),
                 nb::arg("speed") = 1.0f, nb::arg("min") = 0, nb::arg("max") = 0, "Draw a draggable int input, returns (changed, value)")
            // Input
            .def("input_text", &PyUILayout::input_text, nb::arg("label"), nb::arg("value"), "Draw a text input field, returns (changed, value)")
            .def("input_text_with_hint", &PyUILayout::input_text_with_hint,
                 nb::arg("label"), nb::arg("hint"), nb::arg("value"), "Draw a text input with placeholder hint, returns (changed, value)")
            .def("input_float", &PyUILayout::input_float, nb::arg("label"), nb::arg("value"),
                 nb::arg("step") = 0.0f, nb::arg("step_fast") = 0.0f, nb::arg("format") = "%.3f", "Draw a float input field with step buttons, returns (changed, value)")
            .def("input_int", &PyUILayout::input_int, nb::arg("label"), nb::arg("value"),
                 nb::arg("step") = 1, nb::arg("step_fast") = 100, "Draw an int input field with step buttons, returns (changed, value)")
            .def("input_int_formatted", &PyUILayout::input_int_formatted, nb::arg("label"), nb::arg("value"),
                 nb::arg("step") = 0, nb::arg("step_fast") = 0, "Draw a formatted int input field, returns (changed, value)")
            .def("stepper_float", &PyUILayout::stepper_float, nb::arg("label"), nb::arg("value"),
                 nb::arg("steps") = std::vector<float>{1.0f, 0.1f, 0.01f},
                 "Draw a float input with increment/decrement buttons, returns (changed, value)")
            .def("path_input", &PyUILayout::path_input, nb::arg("label"), nb::arg("value"),
                 nb::arg("folder_mode") = true, nb::arg("dialog_title") = "", "Draw a path input with browse button, returns (changed, path)")
            // Color
            .def("color_edit3", &PyUILayout::color_edit3, nb::arg("label"), nb::arg("color"), "Draw an RGB color editor, returns (changed, color)")
            .def("color_edit4", &PyUILayout::color_edit4, nb::arg("label"), nb::arg("color"), "Draw an RGBA color editor, returns (changed, color)")
            .def("color_button", &PyUILayout::color_button, nb::arg("label"), nb::arg("color"),
                 nb::arg("size") = std::make_tuple(0.0f, 0.0f), "Draw a color swatch button, returns True if clicked")
            // Selection
            .def("combo", &PyUILayout::combo, nb::arg("label"), nb::arg("current_idx"), nb::arg("items"), "Draw a combo dropdown, returns (changed, index)")
            .def("listbox", &PyUILayout::listbox, nb::arg("label"), nb::arg("current_idx"),
                 nb::arg("items"), nb::arg("height_items") = -1, "Draw a listbox, returns (changed, index)")
            // Layout
            .def("separator", &PyUILayout::separator, "Draw a horizontal separator line")
            .def("spacing", &PyUILayout::spacing, "Add vertical spacing")
            .def("same_line", &PyUILayout::same_line, nb::arg("offset") = 0.0f, nb::arg("spacing") = -1.0f, "Place next element on the same line")
            .def("new_line", &PyUILayout::new_line, "Move cursor to a new line")
            .def("indent", &PyUILayout::indent, nb::arg("width") = 0.0f, "Increase indentation level")
            .def("unindent", &PyUILayout::unindent, nb::arg("width") = 0.0f, "Decrease indentation level")
            .def("set_next_item_width", &PyUILayout::set_next_item_width, nb::arg("width"), "Set width of the next UI element")
            // Grouping
            .def("begin_group", &PyUILayout::begin_group, "Begin a layout group")
            .def("end_group", &PyUILayout::end_group, "End a layout group")
            .def("collapsing_header", &PyUILayout::collapsing_header, nb::arg("label"), nb::arg("default_open") = false, "Draw a collapsible header, returns True if open")
            .def("tree_node", &PyUILayout::tree_node, nb::arg("label"), "Draw a tree node, returns True if open")
            .def("tree_node_ex", &PyUILayout::tree_node_ex, nb::arg("label"), nb::arg("flags") = "", "Draw a tree node with flags string, returns True if open")
            .def("tree_pop", &PyUILayout::tree_pop, "Pop a tree node level")
            // Tables
            .def("begin_table", &PyUILayout::begin_table, nb::arg("id"), nb::arg("columns"), "Begin a table with given column count, returns True if visible")
            .def("table_setup_column", &PyUILayout::table_setup_column, nb::arg("label"), nb::arg("width") = 0.0f, "Set up a table column with optional fixed width")
            .def("end_table", &PyUILayout::end_table, "End the current table")
            .def("table_next_row", &PyUILayout::table_next_row, "Advance to the next table row")
            .def("table_next_column", &PyUILayout::table_next_column, "Advance to the next table column")
            .def("table_set_column_index", &PyUILayout::table_set_column_index, nb::arg("column"), "Set active column by index, returns True if visible")
            .def("table_headers_row", &PyUILayout::table_headers_row, "Draw the table header row")
            .def("table_set_bg_color", &PyUILayout::table_set_bg_color, nb::arg("target"), nb::arg("color"), "Set table background color for target region")
            // Styled buttons
            .def("button_styled", &PyUILayout::button_styled, nb::arg("label"), nb::arg("style"),
                 nb::arg("size") = std::make_tuple(0.0f, 0.0f), "Draw a themed button (primary, success, warning, error, secondary)")
            // Item width
            .def("push_item_width", &PyUILayout::push_item_width, nb::arg("width"), "Push item width onto the stack")
            .def("pop_item_width", &PyUILayout::pop_item_width, "Pop item width from the stack")
            // Plots
            .def("plot_lines", &PyUILayout::plot_lines, nb::arg("label"), nb::arg("values"),
                 nb::arg("scale_min") = FLT_MAX, nb::arg("scale_max") = FLT_MAX,
                 nb::arg("size") = std::make_tuple(0.0f, 0.0f), "Draw a line plot from float values")
            // Selectable
            .def("selectable", &PyUILayout::selectable, nb::arg("label"),
                 nb::arg("selected") = false, nb::arg("height") = 0.0f, "Draw a selectable item, returns True if clicked")
            // Context menus
            .def("begin_context_menu", &PyUILayout::begin_context_menu, nb::arg("id") = "", "Begin a styled right-click context menu")
            .def("end_context_menu", &PyUILayout::end_context_menu, "End context menu")
            .def("begin_popup", &PyUILayout::begin_popup, nb::arg("id"), "Begin a popup by id, returns True if open")
            .def("open_popup", &PyUILayout::open_popup, nb::arg("id"), "Open a popup by id")
            .def("end_popup", &PyUILayout::end_popup, "End the current popup")
            .def("menu_item", &PyUILayout::menu_item, nb::arg("label"), nb::arg("enabled") = true, "Draw a menu item, returns True if clicked")
            .def("begin_menu", &PyUILayout::begin_menu, nb::arg("label"), "Begin a sub-menu, returns True if open")
            .def("end_menu", &PyUILayout::end_menu, "End the current sub-menu")
            // Rename input with auto-select and enter-to-confirm
            .def("input_text_enter", &PyUILayout::input_text_enter, nb::arg("label"), nb::arg("value"), "Draw a text input that confirms on Enter, returns (entered, value)")
            // Focus control
            .def("set_keyboard_focus_here", &PyUILayout::set_keyboard_focus_here, "Set keyboard focus to the next widget")
            .def("is_window_focused", &PyUILayout::is_window_focused, "Check if current window is focused")
            .def("is_window_hovered", &PyUILayout::is_window_hovered, "Check if current window is hovered")
            .def("capture_keyboard_from_app", &PyUILayout::capture_keyboard_from_app, nb::arg("capture") = true, "Set keyboard capture flag for the application")
            .def("capture_mouse_from_app", &PyUILayout::capture_mouse_from_app, nb::arg("capture") = true, "Set mouse capture flag for the application")
            // Scrolling
            .def("set_scroll_here_y", &PyUILayout::set_scroll_here_y, nb::arg("center_y_ratio") = 0.5f, "Scroll to current cursor Y position")
            // ImDrawList for custom row backgrounds
            .def("get_cursor_screen_pos", &PyUILayout::get_cursor_screen_pos, "Get cursor position in screen coordinates as (x, y)")
            .def("get_window_pos", &PyUILayout::get_window_pos, "Get window position in screen coordinates as (x, y)")
            .def("get_window_width", &PyUILayout::get_window_width, "Get current window width in pixels")
            .def("get_text_line_height", &PyUILayout::get_text_line_height, "Get height of a single text line in pixels")
            // Modal popups
            .def("begin_popup_modal", &PyUILayout::begin_popup_modal, nb::arg("title"), "Begin a modal popup, returns True if visible")
            .def("end_popup_modal", &PyUILayout::end_popup_modal, "End the current modal popup")
            .def("close_current_popup", &PyUILayout::close_current_popup, "Close the currently open popup")
            .def("set_next_window_pos_center", &PyUILayout::set_next_window_pos_center, "Center the next window on the main viewport")
            .def("set_next_window_pos_viewport_center", &PyUILayout::set_next_window_pos_viewport_center, nb::arg("always") = false, "Center the next window on the 3D viewport")
            .def("set_next_window_focus", &PyUILayout::set_next_window_focus, "Set focus to the next window")
            .def("push_modal_style", &PyUILayout::push_modal_style, "Push modal dialog style onto the style stack")
            .def("pop_modal_style", &PyUILayout::pop_modal_style, "Pop modal dialog style from the style stack")
            // Cursor and content region
            .def("get_content_region_avail", &PyUILayout::get_content_region_avail, "Get available content region as (width, height)")
            .def("get_cursor_pos", &PyUILayout::get_cursor_pos, "Get cursor position within the window as (x, y)")
            .def("set_cursor_pos_x", &PyUILayout::set_cursor_pos_x, nb::arg("x"), "Set horizontal cursor position within the window")
            .def("calc_text_size", &PyUILayout::calc_text_size, nb::arg("text"), "Calculate text dimensions as (width, height)")
            // Disabled state
            .def("begin_disabled", &PyUILayout::begin_disabled, nb::arg("disabled") = true, "Begin a disabled UI region")
            .def("end_disabled", &PyUILayout::end_disabled, "End a disabled UI region")
            // Images
            .def("image", &PyUILayout::image, nb::arg("texture_id"), nb::arg("size"),
                 nb::arg("tint") = std::make_tuple(1.0f, 1.0f, 1.0f, 1.0f), "Draw an image from a GL texture ID")
            .def("image_uv", &PyUILayout::image_uv, nb::arg("texture_id"), nb::arg("size"),
                 nb::arg("uv0"), nb::arg("uv1"),
                 nb::arg("tint") = std::make_tuple(1.0f, 1.0f, 1.0f, 1.0f), "Draw an image with custom UV coordinates")
            .def("image_button", &PyUILayout::image_button, nb::arg("id"), nb::arg("texture_id"),
                 nb::arg("size"), nb::arg("tint") = std::make_tuple(1.0f, 1.0f, 1.0f, 1.0f), "Draw an image button, returns True if clicked")
            .def("toolbar_button", &PyUILayout::toolbar_button, nb::arg("id"), nb::arg("texture_id"),
                 nb::arg("size"), nb::arg("selected") = false, nb::arg("disabled") = false,
                 nb::arg("tooltip") = "", "Draw a toolbar-style icon button with selection state")
            // Drag-drop
            .def("begin_drag_drop_source", &PyUILayout::begin_drag_drop_source, "Begin a drag-drop source on the last item, returns True if dragging")
            .def("set_drag_drop_payload", &PyUILayout::set_drag_drop_payload, nb::arg("type"), nb::arg("data"), "Set the drag-drop payload type and data string")
            .def("end_drag_drop_source", &PyUILayout::end_drag_drop_source, "End the drag-drop source")
            .def("begin_drag_drop_target", &PyUILayout::begin_drag_drop_target, "Begin a drag-drop target on the last item, returns True if active")
            .def("accept_drag_drop_payload", &PyUILayout::accept_drag_drop_payload, nb::arg("type"), "Accept a drag-drop payload by type, returns data string or None")
            .def("end_drag_drop_target", &PyUILayout::end_drag_drop_target, "End the drag-drop target")
            // Misc
            .def("progress_bar", &PyUILayout::progress_bar, nb::arg("fraction"), nb::arg("overlay") = "", nb::arg("width") = 0.0f, nb::arg("height") = 0.0f, "Draw a progress bar with fraction 0.0-1.0")
            .def("set_tooltip", &PyUILayout::set_tooltip, nb::arg("text"), "Show tooltip on hover of the previous item")
            .def("is_item_hovered", &PyUILayout::is_item_hovered, "Check if the previous item is hovered")
            .def("is_item_clicked", &PyUILayout::is_item_clicked, nb::arg("button") = 0, "Check if the previous item was clicked")
            .def("is_item_active", &PyUILayout::is_item_active, "Check if the previous item is active")
            .def("is_mouse_double_clicked", &PyUILayout::is_mouse_double_clicked, nb::arg("button") = 0, "Check if mouse button was double-clicked this frame")
            .def("is_mouse_dragging", &PyUILayout::is_mouse_dragging, nb::arg("button") = 0, "Check if mouse button is being dragged")
            .def("get_mouse_wheel", &PyUILayout::get_mouse_wheel, "Get mouse wheel delta for this frame")
            .def("get_mouse_delta", &PyUILayout::get_mouse_delta, "Get mouse movement delta as (dx, dy)")
            .def("invisible_button", &PyUILayout::invisible_button, nb::arg("id"), nb::arg("size"), "Draw an invisible button region, returns True if clicked")
            .def("set_cursor_pos", &PyUILayout::set_cursor_pos, nb::arg("pos"), "Set cursor position within the window as (x, y)")
            // Child windows
            .def("begin_child", &PyUILayout::begin_child, nb::arg("id"), nb::arg("size"), nb::arg("border") = false, "Begin a child window region, returns True if visible")
            .def("end_child", &PyUILayout::end_child, "End the child window region")
            // Menu bar
            .def("begin_menu_bar", &PyUILayout::begin_menu_bar, "Begin the window menu bar, returns True if visible")
            .def("end_menu_bar", &PyUILayout::end_menu_bar, "End the window menu bar")
            .def("menu_item_toggle", &PyUILayout::menu_item_toggle, nb::arg("label"), nb::arg("shortcut"), nb::arg("selected"), "Draw a toggleable menu item with shortcut text")
            .def("menu_item_shortcut", &PyUILayout::menu_item_shortcut, nb::arg("label"), nb::arg("shortcut"), nb::arg("enabled") = true, "Draw a menu item with shortcut text")
            .def("push_id", &PyUILayout::push_id, nb::arg("id"), "Push a string ID onto the ID stack")
            .def("push_id_int", &PyUILayout::push_id_int, nb::arg("id"), "Push an integer ID onto the ID stack")
            .def("pop_id", &PyUILayout::pop_id, "Pop the last ID from the ID stack")
            // Window
            .def("begin_window", &PyUILayout::begin_window, nb::arg("title"), nb::arg("flags") = 0, "Begin a window, returns True if not collapsed")
            .def("begin_window_closable", &PyUILayout::begin_window_closable, nb::arg("title"), nb::arg("flags") = 0, "Begin a closable window, returns (visible, open)")
            .def("end_window", &PyUILayout::end_window, "End the current window")
            .def("push_window_style", &PyUILayout::push_window_style, "Push themed window rounding and padding styles")
            .def("pop_window_style", &PyUILayout::pop_window_style, "Pop window styles pushed by push_window_style")
            // Window positioning
            .def("set_next_window_pos", &PyUILayout::set_next_window_pos, nb::arg("pos"), nb::arg("first_use") = false, "Set position of the next window as (x, y)")
            .def("set_next_window_size", &PyUILayout::set_next_window_size, nb::arg("size"), nb::arg("first_use") = false, "Set size of the next window as (width, height)")
            .def("set_next_window_pos_centered", &PyUILayout::set_next_window_pos_centered, nb::arg("first_use") = false, "Center the next window on the main viewport")
            .def("set_next_window_bg_alpha", &PyUILayout::set_next_window_bg_alpha, nb::arg("alpha"), "Set background alpha of the next window")
            .def("get_viewport_pos", &PyUILayout::get_viewport_pos, "Get 3D viewport position as (x, y)")
            .def("get_viewport_size", &PyUILayout::get_viewport_size, "Get 3D viewport size as (width, height)")
            .def("get_dpi_scale", &PyUILayout::get_dpi_scale, "Get current DPI scale factor")
            .def("set_mouse_cursor_hand", &PyUILayout::set_mouse_cursor_hand, "Set mouse cursor to hand pointer")
            // Style control
            .def("push_style_var", &PyUILayout::push_style_var_float, nb::arg("var"), nb::arg("value"), "Push a float style variable by name")
            .def("push_style_var_vec2", &PyUILayout::push_style_var_vec2, nb::arg("var"), nb::arg("value"), "Push a vec2 style variable by name")
            .def("pop_style_var", &PyUILayout::pop_style_var, nb::arg("count") = 1, "Pop style variables from the stack")
            .def("push_style_color", &PyUILayout::push_style_color, nb::arg("col"), nb::arg("color"), "Push a style color override by name")
            .def("pop_style_color", &PyUILayout::pop_style_color, nb::arg("count") = 1, "Pop style colors from the stack")
            // RNA-style property widget
            .def("prop", &PyUILayout::prop, nb::arg("data"), nb::arg("prop_id"),
                 nb::arg("text") = nb::none(),
                 "Draw a property widget based on metadata (auto-selects widget type)")
            .def("row", &PyUILayout::row, "Create a horizontal row sub-layout")
            .def("column", &PyUILayout::column, "Create a vertical column sub-layout")
            .def("split", &PyUILayout::split, nb::arg("factor") = 0.5f, "Create a split sub-layout with given factor")
            .def("box", &PyUILayout::box, "Create a bordered box sub-layout")
            .def("grid_flow", &PyUILayout::grid_flow, nb::arg("columns") = 0,
                 nb::arg("even_columns") = true, nb::arg("even_rows") = true,
                 "Create a responsive grid sub-layout")
            .def("prop_enum", &PyUILayout::prop_enum, nb::arg("data"), nb::arg("prop_id"),
                 nb::arg("value"), nb::arg("text") = "",
                 "Draw an enum toggle button for a property value")
            .def("operator_", &PyUILayout::operator_, nb::arg("operator_id"),
                 nb::arg("text") = "", nb::arg("icon") = "", "Draw a button that invokes a registered operator")
            .def("prop_search", &PyUILayout::prop_search, nb::arg("data"), nb::arg("prop_id"),
                 nb::arg("search_data"), nb::arg("search_prop"), nb::arg("text") = "",
                 "Searchable dropdown for selecting from a collection")
            .def("template_list", &PyUILayout::template_list, nb::arg("list_type_id"), nb::arg("list_id"),
                 nb::arg("data"), nb::arg("prop_id"), nb::arg("active_data"), nb::arg("active_prop"),
                 nb::arg("rows") = 5, "UIList template for drawing custom lists")
            .def("menu", &PyUILayout::menu, nb::arg("menu_id"), nb::arg("text") = "", nb::arg("icon") = "",
                 "Inline menu reference")
            .def("popover", &PyUILayout::popover, nb::arg("panel_id"), nb::arg("text") = "", nb::arg("icon") = "",
                 "Panel popover")
            // Drawing functions for viewport overlays
            .def("draw_circle", &PyUILayout::draw_circle,
                 nb::arg("x"), nb::arg("y"), nb::arg("radius"), nb::arg("color"),
                 nb::arg("segments") = 32, nb::arg("thickness") = 1.0f,
                 "Draw a circle outline at (x, y) with given radius and color")
            .def("draw_circle_filled", &PyUILayout::draw_circle_filled,
                 nb::arg("x"), nb::arg("y"), nb::arg("radius"), nb::arg("color"),
                 nb::arg("segments") = 32,
                 "Draw a filled circle at (x, y) with given radius and color")
            .def("draw_rect", &PyUILayout::draw_rect,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"),
                 nb::arg("thickness") = 1.0f,
                 "Draw a rectangle outline from (x0,y0) to (x1,y1)")
            .def("draw_rect_filled", &PyUILayout::draw_rect_filled,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"),
                 nb::arg("background") = false,
                 "Draw a filled rectangle from (x0,y0) to (x1,y1)")
            .def("draw_rect_rounded", &PyUILayout::draw_rect_rounded,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"),
                 nb::arg("rounding"), nb::arg("thickness") = 1.0f, nb::arg("background") = false,
                 "Draw a rounded rectangle outline")
            .def("draw_rect_rounded_filled", &PyUILayout::draw_rect_rounded_filled,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"),
                 nb::arg("rounding"), nb::arg("background") = false,
                 "Draw a filled rounded rectangle")
            .def("draw_triangle_filled", &PyUILayout::draw_triangle_filled,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("x2"), nb::arg("y2"),
                 nb::arg("color"), nb::arg("background") = false,
                 "Draw a filled triangle")
            .def("draw_line", &PyUILayout::draw_line,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"),
                 nb::arg("thickness") = 1.0f,
                 "Draw a line from (x0,y0) to (x1,y1)")
            .def("draw_polyline", &PyUILayout::draw_polyline,
                 nb::arg("points"), nb::arg("color"), nb::arg("closed") = false, nb::arg("thickness") = 1.0f,
                 "Draw a polyline through the given points")
            .def("draw_poly_filled", &PyUILayout::draw_poly_filled,
                 nb::arg("points"), nb::arg("color"),
                 "Draw a filled convex polygon")
            .def("draw_text", &PyUILayout::draw_text,
                 nb::arg("x"), nb::arg("y"), nb::arg("text"), nb::arg("color"), nb::arg("background") = false,
                 "Draw text at (x, y) with given color")
            // Window-scoped drawing (respects z-order)
            .def("draw_window_rect_filled", &PyUILayout::draw_window_rect_filled,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"),
                 "Draw a filled rectangle on the window draw list")
            .def("draw_window_rect", &PyUILayout::draw_window_rect,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"),
                 nb::arg("thickness") = 1.0f,
                 "Draw a rectangle outline on the window draw list")
            .def("draw_window_rect_rounded", &PyUILayout::draw_window_rect_rounded,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"),
                 nb::arg("rounding"), nb::arg("thickness") = 1.0f,
                 "Draw a rounded rectangle outline on the window draw list")
            .def("draw_window_rect_rounded_filled", &PyUILayout::draw_window_rect_rounded_filled,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"),
                 nb::arg("rounding"),
                 "Draw a filled rounded rectangle on the window draw list")
            .def("draw_window_line", &PyUILayout::draw_window_line,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"),
                 nb::arg("thickness") = 1.0f,
                 "Draw a line on the window draw list")
            .def("draw_window_text", &PyUILayout::draw_window_text,
                 nb::arg("x"), nb::arg("y"), nb::arg("text"), nb::arg("color"),
                 "Draw text on the window draw list")
            .def("draw_window_triangle_filled", &PyUILayout::draw_window_triangle_filled,
                 nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("x2"), nb::arg("y2"),
                 nb::arg("color"),
                 "Draw a filled triangle on the window draw list")
            .def("crf_curve_preview", &PyUILayout::crf_curve_preview,
                 nb::arg("label"), nb::arg("gamma"), nb::arg("toe"), nb::arg("shoulder"),
                 nb::arg("gamma_r") = 0.0f, nb::arg("gamma_g") = 0.0f, nb::arg("gamma_b") = 0.0f,
                 "Draw CRF tone curve preview widget")
            .def("chromaticity_diagram", &PyUILayout::chromaticity_diagram,
                 nb::arg("label"),
                 nb::arg("red_x"), nb::arg("red_y"),
                 nb::arg("green_x"), nb::arg("green_y"),
                 nb::arg("blue_x"), nb::arg("blue_y"),
                 nb::arg("neutral_x"), nb::arg("neutral_y"),
                 nb::arg("range") = 0.5f,
                 "Draw chromaticity diagram widget for color correction");

        // File dialogs
        m.def(
            "open_image_dialog",
            [](const std::string& start_dir) -> std::string {
                std::filesystem::path start_path;
                if (!start_dir.empty()) {
                    start_path = start_dir;
                }
                auto result = lfs::vis::gui::OpenImageFileDialog(start_path);
                return result.empty() ? "" : result.string();
            },
            nb::arg("start_dir") = "",
            "Open a file dialog to select an image file. Returns empty string if cancelled.");

        m.def(
            "open_folder_dialog",
            [](const std::string& title, const std::string& start_dir) -> std::string {
                std::filesystem::path start_path;
                if (!start_dir.empty()) {
                    start_path = start_dir;
                }
                auto result = lfs::vis::gui::SelectFolderDialog(title, start_path);
                return result.empty() ? "" : result.string();
            },
            nb::arg("title") = "Select Folder", nb::arg("start_dir") = "",
            "Open a folder selection dialog. Returns empty string if cancelled.");

        m.def(
            "open_ply_file_dialog",
            [](const std::string& start_dir) -> std::string {
                std::filesystem::path start_path;
                if (!start_dir.empty()) {
                    start_path = start_dir;
                }
                auto result = lfs::vis::gui::OpenPlyFileDialogNative(start_path);
                return result.empty() ? "" : result.string();
            },
            nb::arg("start_dir") = "",
            "Open a file dialog to select a PLY file. Returns empty string if cancelled.");

        m.def(
            "open_checkpoint_file_dialog",
            []() -> std::string {
                auto result = lfs::vis::gui::OpenCheckpointFileDialog();
                return result.empty() ? "" : result.string();
            },
            "Open a file dialog to select a checkpoint file. Returns empty string if cancelled.");

        m.def(
            "open_json_file_dialog",
            []() -> std::string {
                auto result = lfs::vis::gui::OpenJsonFileDialog();
                return result.empty() ? "" : result.string();
            },
            "Open a file dialog to select a JSON config file. Returns empty string if cancelled.");

        m.def(
            "save_json_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveJsonFileDialog(default_name);
                return result.empty() ? "" : result.string();
            },
            nb::arg("default_name") = "config.json",
            "Open a save file dialog for JSON files. Returns empty string if cancelled.");

        m.def(
            "save_ply_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SavePlyFileDialog(default_name);
                return result.empty() ? "" : result.string();
            },
            nb::arg("default_name") = "export.ply",
            "Open a save file dialog for PLY files. Returns empty string if cancelled.");

        m.def(
            "save_sog_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveSogFileDialog(default_name);
                return result.empty() ? "" : result.string();
            },
            nb::arg("default_name") = "export.sog",
            "Open a save file dialog for SOG files. Returns empty string if cancelled.");

        m.def(
            "save_spz_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveSpzFileDialog(default_name);
                return result.empty() ? "" : result.string();
            },
            nb::arg("default_name") = "export.spz",
            "Open a save file dialog for SPZ files. Returns empty string if cancelled.");

        m.def(
            "save_html_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveHtmlFileDialog(default_name);
                return result.empty() ? "" : result.string();
            },
            nb::arg("default_name") = "viewer.html",
            "Open a save file dialog for HTML viewer files. Returns empty string if cancelled.");

        m.def(
            "open_dataset_folder_dialog",
            []() -> std::string {
                auto result = lfs::vis::gui::OpenDatasetFolderDialogNative();
                return result.empty() ? "" : result.string();
            },
            "Open a folder dialog to select a dataset. Returns empty string if cancelled.");

        m.def(
            "open_video_file_dialog",
            []() -> std::string {
                auto result = lfs::vis::gui::OpenVideoFileDialog();
                return result.empty() ? "" : result.string();
            },
            "Open a file dialog to select a video file. Returns empty string if cancelled.");

        m.def(
            "open_url",
            [](const std::string& url) {
#ifdef _WIN32
                ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
                const std::string cmd = "xdg-open \"" + url + "\" &";
                std::system(cmd.c_str());
#endif
            },
            nb::arg("url"), "Open a URL in the default browser.");

        m.def(
            "set_tool",
            [](const std::string& tool_name) {
                using lfs::vis::ToolType;

                static const std::unordered_map<std::string, ToolType> TOOL_MAP = {
                    {"none", ToolType::None},
                    {"selection", ToolType::Selection},
                    {"translate", ToolType::Translate},
                    {"rotate", ToolType::Rotate},
                    {"scale", ToolType::Scale},
                    {"mirror", ToolType::Mirror},
                    {"brush", ToolType::Brush},
                    {"align", ToolType::Align},
                };

                std::string lower_name = tool_name;
                std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                               [](const unsigned char c) { return std::tolower(c); });

                const auto it = TOOL_MAP.find(lower_name);
                if (it == TOOL_MAP.end()) {
                    throw std::invalid_argument("Unknown tool: " + tool_name);
                }

                lfs::core::events::tools::SetToolbarTool{.tool_mode = static_cast<int>(it->second)}.emit();
            },
            nb::arg("tool"), "Switch to a toolbar tool (none, selection, translate, rotate, scale, mirror, brush, align, cropbox)");

        // Key enum (subset of commonly used keys)
        nb::enum_<ImGuiKey>(m, "Key")
            .value("ESCAPE", ImGuiKey_Escape)
            .value("ENTER", ImGuiKey_Enter)
            .value("TAB", ImGuiKey_Tab)
            .value("BACKSPACE", ImGuiKey_Backspace)
            .value("DELETE", ImGuiKey_Delete)
            .value("SPACE", ImGuiKey_Space)
            .value("LEFT", ImGuiKey_LeftArrow)
            .value("RIGHT", ImGuiKey_RightArrow)
            .value("UP", ImGuiKey_UpArrow)
            .value("DOWN", ImGuiKey_DownArrow)
            .value("HOME", ImGuiKey_Home)
            .value("END", ImGuiKey_End)
            .value("F", ImGuiKey_F)
            .value("I", ImGuiKey_I)
            .value("M", ImGuiKey_M)
            .value("R", ImGuiKey_R)
            .value("T", ImGuiKey_T)
            .value("_1", ImGuiKey_1)
            .value("MINUS", ImGuiKey_Minus)
            .value("EQUAL", ImGuiKey_Equal)
            .value("F2", ImGuiKey_F2);

        // Key input functions
        m.def(
            "is_key_pressed",
            [](ImGuiKey key, bool repeat) { return ImGui::IsKeyPressed(key, repeat); },
            nb::arg("key"), nb::arg("repeat") = true, "Check if a key was pressed this frame");

        m.def(
            "is_key_down",
            [](ImGuiKey key) { return ImGui::IsKeyDown(key); },
            nb::arg("key"), "Check if a key is currently held down");

        m.def(
            "is_ctrl_down",
            []() { return ImGui::GetIO().KeyCtrl; },
            "Check if Ctrl is currently held");

        m.def(
            "is_shift_down",
            []() { return ImGui::GetIO().KeyShift; },
            "Check if Shift is currently held");

        // Localization
        m.def(
            "tr",
            [](const std::string& key) { return lfs::event::LocalizationManager::getInstance().get(key); },
            nb::arg("key"), "Get localized string by key");

        m.def(
            "loc_set",
            [](const std::string& key, const std::string& value) {
                lfs::event::LocalizationManager::getInstance().setOverride(key, value);
            },
            nb::arg("key"), nb::arg("value"), "Override a localization string at runtime");

        m.def(
            "loc_clear",
            [](const std::string& key) {
                lfs::event::LocalizationManager::getInstance().clearOverride(key);
            },
            nb::arg("key"), "Clear a localization override");

        m.def(
            "loc_clear_all", []() {
                lfs::event::LocalizationManager::getInstance().clearAllOverrides();
            },
            "Clear all localization overrides");

        m.def(
            "register_popup_draw_callback",
            [](nb::object callback) {
                g_popup_draw_callback = callback;
                set_popup_draw_callback([]() {
                    if (g_popup_draw_callback && !g_popup_draw_callback.is_none()) {
                        try {
                            PyUILayout layout;
                            g_popup_draw_callback(layout);
                        } catch (const std::exception& e) {
                            LOG_ERROR("Popup callback error: {}", e.what());
                        }
                    }
                });
            },
            nb::arg("callback"), "Register a callback for drawing popup content");

        m.def(
            "unregister_popup_draw_callback",
            [](nb::object) {
                g_popup_draw_callback = nb::none();
                set_popup_draw_callback(nullptr);
            },
            nb::arg("callback"), "Unregister the popup draw callback");

        m.def(
            "on_show_dataset_load_popup",
            [](nb::object callback) {
                g_show_dataset_popup_callback = callback;
                lfs::core::events::cmd::ShowDatasetLoadPopup::when([](const auto& e) {
                    if (g_show_dataset_popup_callback && !g_show_dataset_popup_callback.is_none()) {
                        nb::gil_scoped_acquire guard;
                        try {
                            g_show_dataset_popup_callback(e.dataset_path.string());
                        } catch (const std::exception& ex) {
                            LOG_ERROR("ShowDatasetLoadPopup callback error: {}", ex.what());
                        }
                    }
                });
            },
            nb::arg("callback"),
            "Register callback for ShowDatasetLoadPopup event");

        m.def(
            "on_show_resume_checkpoint_popup",
            [](nb::object callback) {
                g_show_resume_popup_callback = callback;
                lfs::core::events::cmd::ShowResumeCheckpointPopup::when([](const auto& e) {
                    if (g_show_resume_popup_callback && !g_show_resume_popup_callback.is_none()) {
                        nb::gil_scoped_acquire guard;
                        try {
                            g_show_resume_popup_callback(e.checkpoint_path.string());
                        } catch (const std::exception& ex) {
                            LOG_ERROR("ShowResumeCheckpointPopup callback error: {}", ex.what());
                        }
                    }
                });
            },
            nb::arg("callback"),
            "Register callback for ShowResumeCheckpointPopup event");

        m.def(
            "on_request_exit",
            [](nb::object callback) {
                g_request_exit_callback = callback;
                lfs::core::events::cmd::RequestExit::when([](const auto&) {
                    if (g_request_exit_callback && !g_request_exit_callback.is_none()) {
                        nb::gil_scoped_acquire guard;
                        try {
                            g_request_exit_callback();
                        } catch (const std::exception& ex) {
                            LOG_ERROR("RequestExit callback error: {}", ex.what());
                        }
                    }
                });
            },
            nb::arg("callback"),
            "Register callback for RequestExit event");

        m.def(
            "set_exit_popup_open",
            [](bool open) { set_exit_popup_open(open); },
            nb::arg("open"),
            "Set exit popup open state (for window close callback)");

        m.def(
            "get_active_tool", []() -> std::string {
                auto* editor = get_editor_context();
                if (!editor)
                    return "";

                const auto& op_id = editor->getActiveOperator();
                if (!op_id.empty())
                    return op_id;

                switch (editor->getActiveTool()) {
                case vis::ToolType::Selection: return "builtin.select";
                case vis::ToolType::Translate: return "builtin.translate";
                case vis::ToolType::Rotate: return "builtin.rotate";
                case vis::ToolType::Scale: return "builtin.scale";
                case vis::ToolType::Mirror: return "builtin.mirror";
                case vis::ToolType::Brush: return "builtin.brush";
                case vis::ToolType::Align: return "builtin.align";
                default: return "";
                }
            },
            "Get the currently active tool id from C++ EditorContext");

        m.def(
            "set_active_tool", [](const std::string& id) {
                static const std::unordered_map<std::string, vis::ToolType> tool_map = {
                    {"builtin.select", vis::ToolType::Selection},
                    {"builtin.translate", vis::ToolType::Translate},
                    {"builtin.rotate", vis::ToolType::Rotate},
                    {"builtin.scale", vis::ToolType::Scale},
                    {"builtin.mirror", vis::ToolType::Mirror},
                    {"builtin.brush", vis::ToolType::Brush},
                    {"builtin.align", vis::ToolType::Align},
                };
                auto it = tool_map.find(id);
                if (it != tool_map.end()) {
                    lfs::core::events::tools::SetToolbarTool{.tool_mode = static_cast<int>(it->second)}.emit();
                }
            },
            nb::arg("id"), "Set the active tool via C++ event");

        m.def(
            "set_active_operator",
            [](const std::string& id, const std::string& gizmo_type) {
                if (auto* const editor = get_editor_context()) {
                    editor->setActiveOperator(id, gizmo_type);
                }
                vis::UnifiedToolRegistry::instance().setActiveTool(id);
            },
            nb::arg("id"), nb::arg("gizmo_type") = "", "Set active operator with optional gizmo type");

        m.def(
            "get_active_operator", []() -> std::string {
                auto* editor = get_editor_context();
                return editor ? editor->getActiveOperator() : "";
            },
            "Get the currently active operator id");

        m.def(
            "get_gizmo_type", []() -> std::string {
                auto* editor = get_editor_context();
                return editor ? editor->getGizmoType() : "";
            },
            "Get the gizmo type for the current operator");

        m.def(
            "clear_active_operator", []() {
                auto* editor = get_editor_context();
                if (editor) {
                    editor->clearActiveOperator();
                }
                vis::UnifiedToolRegistry::instance().clearActiveTool();
            },
            "Clear the active operator");

        m.def(
            "has_active_operator", []() -> bool {
                const auto* editor = get_editor_context();
                return editor && editor->hasActiveOperator();
            },
            "Check if an operator is currently active");

        m.def(
            "set_gizmo_type", [](const std::string& type) {
                if (auto* editor = get_editor_context())
                    editor->setGizmoType(type);
            },
            nb::arg("type"), "Set gizmo type without blocking camera");

        m.def(
            "clear_gizmo", []() {
                if (auto* editor = get_editor_context())
                    editor->clearGizmo();
            },
            "Clear gizmo type");

        m.def(
            "get_active_submode",
            []() { return vis::UnifiedToolRegistry::instance().getActiveSubmode(); },
            "Get active selection submode");

        m.def(
            "set_selection_mode",
            [](const std::string& mode) {
                vis::UnifiedToolRegistry::instance().setActiveSubmode(mode);

                static const std::unordered_map<std::string, int> MODE_MAP = {
                    {"centers", 0},
                    {"rectangle", 1},
                    {"polygon", 2},
                    {"lasso", 3},
                    {"rings", 4}};
                if (const auto it = MODE_MAP.find(mode); it != MODE_MAP.end()) {
                    lfs::core::events::tools::SetSelectionSubMode{.selection_mode = it->second}.emit();
                }
            },
            nb::arg("mode"), "Set selection mode");

        m.def(
            "execute_mirror",
            [](const std::string& axis) {
                static const std::unordered_map<std::string, int> AXIS_MAP = {{"x", 0}, {"y", 1}, {"z", 2}};
                if (const auto it = AXIS_MAP.find(axis); it != AXIS_MAP.end()) {
                    lfs::core::events::tools::ExecuteMirror{.axis = it->second}.emit();
                }
            },
            nb::arg("axis"), "Execute mirror on axis (x, y, z)");

        // Scene panel context menu actions
        m.def(
            "go_to_camera_view",
            [](int cam_uid) { lfs::core::events::cmd::GoToCamView{.cam_id = cam_uid}.emit(); },
            nb::arg("cam_uid"), "Go to camera view by UID");

        m.def(
            "apply_cropbox",
            []() { lfs::core::events::cmd::ApplyCropBox{}.emit(); },
            "Apply the selected cropbox");

        m.def(
            "fit_cropbox_to_scene",
            [](bool use_percentile) { lfs::core::events::cmd::FitCropBoxToScene{.use_percentile = use_percentile}.emit(); },
            nb::arg("use_percentile") = false, "Fit cropbox to scene bounds");

        m.def(
            "reset_cropbox",
            []() { lfs::core::events::cmd::ResetCropBox{}.emit(); },
            "Reset the selected cropbox");

        m.def(
            "add_cropbox",
            [](const std::string& node_name) { lfs::core::events::cmd::AddCropBox{.node_name = node_name}.emit(); },
            nb::arg("node_name"), "Add a cropbox to the specified node");

        m.def(
            "add_ellipsoid",
            [](const std::string& node_name) { lfs::core::events::cmd::AddCropEllipsoid{.node_name = node_name}.emit(); },
            nb::arg("node_name"), "Add an ellipsoid to the specified node");

        m.def(
            "apply_ellipsoid",
            []() { lfs::core::events::cmd::ApplyEllipsoid{}.emit(); },
            "Apply the selected ellipsoid");

        m.def(
            "reset_ellipsoid",
            []() { lfs::core::events::cmd::ResetEllipsoid{}.emit(); },
            "Reset the selected ellipsoid");

        m.def(
            "fit_ellipsoid_to_scene",
            [](bool use_percentile) { lfs::core::events::cmd::FitEllipsoidToScene{.use_percentile = use_percentile}.emit(); },
            nb::arg("use_percentile") = false, "Fit ellipsoid to scene bounds");

        m.def(
            "duplicate_node",
            [](const std::string& name) { lfs::core::events::cmd::DuplicateNode{.name = name}.emit(); },
            nb::arg("name"), "Duplicate a node and its children");

        m.def(
            "merge_group",
            [](const std::string& name) { lfs::core::events::cmd::MergeGroup{.name = name}.emit(); },
            nb::arg("name"), "Merge group children into a single PLY");

        // Image texture APIs for Python image preview
        m.def(
            "load_image_texture",
            [](const std::string& path) -> nb::tuple {
                try {
                    auto [data, w, h, channels] = lfs::core::load_image(std::filesystem::path(path), -1, -1);
                    if (!data)
                        return nb::make_tuple(0, 0, 0);

                    auto [tex_id, width, height] = create_gl_texture_from_data(data, w, h, channels);
                    lfs::core::free_image(data);
                    return nb::make_tuple(tex_id, width, height);
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to load image texture {}: {}", path, e.what());
                    return nb::make_tuple(0, 0, 0);
                }
            },
            nb::arg("path"), "Load image as GL texture, returns (texture_id, width, height)");

        m.def(
            "load_thumbnail",
            [](const std::string& path, int max_size) -> nb::tuple {
                try {
                    auto [data, w, h, channels] = lfs::core::load_image(std::filesystem::path(path), -1, max_size);
                    if (!data)
                        return nb::make_tuple(0, 0, 0);

                    auto [tex_id, width, height] = create_gl_texture_from_data(data, w, h, channels);
                    lfs::core::free_image(data);
                    return nb::make_tuple(tex_id, width, height);
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to load thumbnail {}: {}", path, e.what());
                    return nb::make_tuple(0, 0, 0);
                }
            },
            nb::arg("path"), nb::arg("max_size"), "Load downscaled image as GL texture, returns (texture_id, width, height)");

        m.def(
            "release_texture",
            [](const uint64_t tex_id) {
                if (tex_id > 0)
                    lfs::python::delete_gl_texture(static_cast<uint32_t>(tex_id));
            },
            nb::arg("texture_id"), "Release an OpenGL texture");

        m.def(
            "preload_image_async",
            [](const std::string& path) {
                std::lock_guard lock(g_preload_mutex);
                if (g_preload_cache.contains(path))
                    return;

                auto entry = std::make_unique<PreloadEntry>();
                entry->future = std::async(std::launch::async, [path_copy = path]() {
                    return lfs::core::load_image(std::filesystem::path(path_copy), -1, -1);
                });
                g_preload_cache[path] = std::move(entry);
            },
            nb::arg("path"), "Start async preload of image data");

        m.def(
            "is_preload_ready",
            [](const std::string& path) -> bool {
                std::lock_guard lock(g_preload_mutex);
                auto it = g_preload_cache.find(path);
                if (it == g_preload_cache.end())
                    return false;
                if (it->second->ready)
                    return true;
                if (it->second->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    it->second->ready = true;
                    return true;
                }
                return false;
            },
            nb::arg("path"), "Check if preloaded image is ready");

        m.def(
            "get_preloaded_texture",
            [](const std::string& path) -> nb::tuple {
                std::lock_guard lock(g_preload_mutex);
                auto it = g_preload_cache.find(path);
                if (it == g_preload_cache.end() || !it->second->ready)
                    return nb::make_tuple(0, 0, 0);

                auto [data, w, h, channels] = it->second->future.get();
                g_preload_cache.erase(it);

                if (!data)
                    return nb::make_tuple(0, 0, 0);

                auto [tex_id, width, height] = create_gl_texture_from_data(data, w, h, channels);
                lfs::core::free_image(data);
                return nb::make_tuple(tex_id, width, height);
            },
            nb::arg("path"), "Get preloaded image as GL texture, returns (texture_id, width, height)");

        m.def(
            "cancel_preload",
            [](const std::string& path) {
                std::lock_guard lock(g_preload_mutex);
                g_preload_cache.erase(path);
            },
            nb::arg("path"), "Cancel a pending preload");

        m.def(
            "clear_preload_cache",
            []() {
                std::lock_guard lock(g_preload_mutex);
                g_preload_cache.clear();
            },
            "Clear all pending preloads");

        m.def("is_sequencer_visible", &is_sequencer_visible, "Check if sequencer panel is visible");

        m.def(
            "section_header",
            [](const std::string& text) {
                const auto& t = lfs::vis::theme();
                ImGui::TextColored(t.palette.text_dim, "%s", text.c_str());
                ImGui::Separator();
            },
            nb::arg("text"), "Draw a section header with text and separator");

        m.def("set_sequencer_visible", &set_sequencer_visible, nb::arg("visible"),
              "Set sequencer panel visibility");

        // Overlay state functions for Python overlay panels
        m.def("is_drag_hovering", &is_drag_hovering,
              "Check if files are being dragged over the window");
        m.def("is_startup_visible", &is_startup_visible,
              "Check if startup overlay is visible");

        m.def(
            "is_scene_empty",
            []() {
                auto* editor = get_editor_context();
                return editor ? editor->isEmpty() : true;
            },
            "Check if no scene is loaded");

        m.def(
            "get_export_state",
            []() {
                nb::dict state;
                auto export_state = get_export_state();
                state["active"] = export_state.active;
                state["progress"] = export_state.progress;
                state["stage"] = export_state.stage;
                state["format"] = export_state.format;
                return state;
            },
            "Get current export progress state");

        m.def("cancel_export", &cancel_export,
              "Cancel an ongoing export operation");

        m.def(
            "get_import_state",
            []() {
                nb::dict state;
                auto import_state = get_import_state();
                state["active"] = import_state.active;
                state["show_completion"] = import_state.show_completion;
                state["progress"] = import_state.progress;
                state["stage"] = import_state.stage;
                state["dataset_type"] = import_state.dataset_type;
                state["path"] = import_state.path;
                state["success"] = import_state.success;
                state["error"] = import_state.error;
                state["num_images"] = import_state.num_images;
                state["num_points"] = import_state.num_points;
                state["seconds_since_completion"] = import_state.seconds_since_completion;
                return state;
            },
            "Get current import progress state");

        m.def("dismiss_import", &dismiss_import,
              "Dismiss the import completion overlay");

        m.def(
            "get_video_export_state",
            []() {
                nb::dict state;
                auto video_state = get_video_export_state();
                state["active"] = video_state.active;
                state["progress"] = video_state.progress;
                state["current_frame"] = video_state.current_frame;
                state["total_frames"] = video_state.total_frames;
                state["stage"] = video_state.stage;
                return state;
            },
            "Get current video export progress state");

        m.def("cancel_video_export", &cancel_video_export,
              "Cancel an ongoing video export operation");

        // Sequencer UI state for Python access (uses callback to avoid ImGuizmo dependency)
        nb::class_<SequencerUIStateData>(m, "SequencerUIState")
            .def_rw("show_camera_path", &SequencerUIStateData::show_camera_path, "Whether camera path is displayed in viewport")
            .def_rw("snap_to_grid", &SequencerUIStateData::snap_to_grid, "Whether keyframe snapping is enabled")
            .def_rw("snap_interval", &SequencerUIStateData::snap_interval, "Snap grid interval in frames")
            .def_rw("playback_speed", &SequencerUIStateData::playback_speed, "Playback speed multiplier")
            .def_rw("follow_playback", &SequencerUIStateData::follow_playback, "Whether viewport follows playback position")
            .def_rw("pip_preview_scale", &SequencerUIStateData::pip_preview_scale, "Picture-in-picture preview scale factor");

        m.def(
            "get_sequencer_state",
            []() -> SequencerUIStateData* { return get_sequencer_ui_state(); },
            nb::rv_policy::reference,
            "Get sequencer UI state for modification");

        m.def("has_keyframes", &has_keyframes,
              "Check if sequencer has any keyframes");

        m.def("save_camera_path", &save_camera_path,
              nb::arg("path"),
              "Save camera path to JSON file");

        m.def("load_camera_path", &load_camera_path,
              nb::arg("path"),
              "Load camera path from JSON file");

        m.def("clear_keyframes", &clear_keyframes,
              "Clear all keyframes");

        m.def("set_playback_speed", &set_playback_speed,
              nb::arg("speed"),
              "Set sequencer playback speed");

        m.def(
            "export_video",
            [](int width, int height, int framerate, int crf) {
                lfs::core::events::cmd::SequencerExportVideo{
                    .width = width,
                    .height = height,
                    .framerate = framerate,
                    .crf = crf}
                    .emit();
            },
            nb::arg("width"), nb::arg("height"), nb::arg("framerate"), nb::arg("crf"),
            "Export video with specified settings");

        m.def(
            "add_keyframe",
            []() { lfs::core::events::cmd::SequencerAddKeyframe{}.emit(); },
            "Add a keyframe at current camera position");

        m.def(
            "update_keyframe",
            []() { lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit(); },
            "Update selected keyframe to current camera position");

        m.def(
            "play_pause",
            []() { lfs::core::events::cmd::SequencerPlayPause{}.emit(); },
            "Toggle sequencer playback");

        // Section drawing wrappers - callable from Python panel draw()
        // These use callbacks to avoid ImGuizmo header dependency in py_ui.cpp
        m.def("draw_tools_section", &draw_tools_section,
              "Draw tools section (C++ implementation)");

        m.def("draw_console_button", &draw_console_button,
              "Draw system console button (C++ implementation)");

        m.def("get_pivot_mode", &get_pivot_mode, "Get pivot mode (0=Origin, 1=Bounds)");

        m.def("set_pivot_mode", &set_pivot_mode, nb::arg("mode"), "Set pivot mode (0=Origin, 1=Bounds)");

        m.def("get_transform_space", &get_transform_space, "Get transform space (0=Local, 1=World)");

        m.def("set_transform_space", &set_transform_space, nb::arg("space"), "Set transform space (0=Local, 1=World)");

        // Thumbnail system (for Getting Started window)
        m.def("request_thumbnail", &request_thumbnail, nb::arg("video_id"),
              "Request download of a YouTube thumbnail for the given video ID");
        m.def("process_thumbnails", &process_thumbnails,
              "Process pending thumbnail downloads (call every frame)");
        m.def("is_thumbnail_ready", &is_thumbnail_ready, nb::arg("video_id"),
              "Check if a thumbnail is ready to be displayed");
        m.def("get_thumbnail_texture", &get_thumbnail_texture, nb::arg("video_id"),
              "Get the OpenGL texture ID for a downloaded thumbnail (0 if not ready)");

        // Icon loading for data-driven toolbar
        m.def(
            "load_icon", [](const std::string& name) -> uint64_t {
                return static_cast<uint64_t>(load_icon_texture(name));
            },
            nb::arg("name"), "Load icon by name (e.g., 'selection.png'), returns texture ID");

        // Scene panel icons (e.g., 'visible', 'hidden', 'splat', etc.)
        m.def(
            "load_scene_icon", [](const std::string& name) -> uint64_t {
                return static_cast<uint64_t>(load_scene_icon(name));
            },
            nb::arg("name"), "Load scene icon by name (e.g., 'visible', 'splat'), returns texture ID");

        m.def(
            "load_plugin_icon",
            [](const std::string& icon_name, const std::string& plugin_path,
               const std::string& plugin_name) -> uint64_t {
                return static_cast<uint64_t>(load_plugin_icon(icon_name, plugin_path, plugin_name));
            },
            nb::arg("icon_name"), nb::arg("plugin_path"), nb::arg("plugin_name"),
            "Load icon from plugin folder with fallback to assets");

        m.def("free_plugin_icons", &free_plugin_icons, nb::arg("plugin_name"),
              "Free all icons associated with a plugin");

        // Scene panel state for Python scene panel
        m.def(
            "get_selected_camera_uid",
            []() -> int { return python::get_selected_camera_uid(); },
            "Get the UID of the currently selected camera, or -1 if no camera is selected");

        m.def(
            "get_invert_masks",
            []() -> bool { return python::get_invert_masks(); },
            "Get whether masks are inverted");

        // Theme control (for Python-driven View menu)
        m.def(
            "set_theme",
            [](const std::string& name) {
                if (name == "dark" || name == "Dark") {
                    vis::setTheme(vis::darkTheme());
                    vis::saveThemePreference(true);
                } else if (name == "light" || name == "Light") {
                    vis::setTheme(vis::lightTheme());
                    vis::saveThemePreference(false);
                }
            },
            nb::arg("name"), "Set theme ('dark' or 'light')");

        m.def(
            "get_theme",
            []() -> std::string { return vis::theme().name; },
            "Get current theme name ('Dark' or 'Light')");

        // Language control (for Python-driven Edit menu)
        m.def(
            "set_language",
            [](const std::string& lang_code) {
                lfs::event::LocalizationManager::getInstance().setLanguage(lang_code);
            },
            nb::arg("lang_code"), "Set language by code (e.g., 'en', 'de')");

        m.def(
            "get_current_language",
            []() -> std::string {
                return lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
            },
            "Get current language code");

        m.def(
            "get_languages",
            []() -> std::vector<std::tuple<std::string, std::string>> {
                auto& loc = lfs::event::LocalizationManager::getInstance();
                auto codes = loc.getAvailableLanguages();
                auto names = loc.getAvailableLanguageNames();
                std::vector<std::tuple<std::string, std::string>> result;
                for (size_t i = 0; i < codes.size() && i < names.size(); ++i) {
                    result.emplace_back(codes[i], names[i]);
                }
                return result;
            },
            "Get available languages as list of (code, name) tuples");

        m.def(
            "tr",
            [](const std::string& key) -> std::string {
                return lfs::event::LocalizationManager::getInstance().get(key.c_str());
            },
            nb::arg("key"), "Translate a string key");

        // Menu bar UI functions (for Python-driven menus)
        m.def("show_input_settings", &show_input_settings, "Show input settings window");
        m.def("show_python_console", &show_python_console, "Show Python console");
        m.def(
            "get_time", []() { return ImGui::GetTime(); }, "Get time in seconds since application start");

        // Register all Python callbacks via consolidated bridge
        PyBridge bridge;
        bridge.prepare_ui = []() {
            void* const ctx = get_imgui_context();
            if (!ctx)
                return;
            ImGui::SetCurrentContext(static_cast<ImGuiContext*>(ctx));
            void* alloc_fn{};
            void* free_fn{};
            void* user_data{};
            get_imgui_allocator_functions(&alloc_fn, &free_fn, &user_data);
            if (alloc_fn && free_fn) {
                ImGui::SetAllocatorFunctions(
                    reinterpret_cast<ImGuiMemAllocFunc>(alloc_fn),
                    reinterpret_cast<ImGuiMemFreeFunc>(free_fn),
                    user_data);
            }

            void* const plot_ctx = get_implot_context();
            if (plot_ctx)
                ImPlot::SetCurrentContext(static_cast<ImPlotContext*>(plot_ctx));
        };
        bridge.draw_menus = [](MenuLocation loc) { PyMenuRegistry::instance().draw_menu_items(loc); };
        bridge.has_menus = [](MenuLocation loc) { return PyMenuRegistry::instance().has_items(loc); };
        bridge.has_menu_bar_entries = []() { return PyMenuRegistry::instance().has_menu_bar_entries(); };
        bridge.get_menu_bar_entries = [](MenuBarEntryVisitor visitor, void* ctx) {
            auto entries = PyMenuRegistry::instance().get_menu_bar_entries();
            for (auto* entry : entries) {
                visitor(entry->idname.c_str(), entry->label.c_str(), entry->order, ctx);
            }
        };
        bridge.draw_menu_bar_entry = [](const char* idname) {
            if (idname)
                PyMenuRegistry::instance().draw_menu_bar_entry(idname);
        };
        bridge.draw_modals = []() { PyModalRegistry::instance().draw_modals(); };
        bridge.has_modals = []() { return PyModalRegistry::instance().has_open_modals(); };
        bridge.has_toolbar = []() { return true; }; // Always true - Python ToolRegistry has builtin tools
        bridge.cleanup = []() {
            PyPanelRegistry::instance().unregister_all();
            PyUIHookRegistry::instance().clear_all();
            PyMenuRegistry::instance().unregister_all();
            PyViewportDrawRegistry::instance().clear_all();
            PyGizmoRegistry::instance().unregister_all();
            PyUIListRegistry::instance().unregister_all();
            vis::op::operators().unregisterAllPython();
            shutdown_signal_bridge();
            g_cancel_operator_py_callback = nb::callable();
            g_modal_event_py_callback = nb::callable();
            g_popup_draw_callback = nb::object();
            g_show_dataset_popup_callback = nb::object();
            g_show_resume_popup_callback = nb::object();
            g_request_exit_callback = nb::object();
        };
        set_bridge(bridge);

        // Operator cancel callback binding (Python sets the callback, C++ calls it on ESC)
        // Stored in EditorContext to avoid static variable duplication across shared libraries
        m.def(
            "set_cancel_operator_callback", [](nb::callable cb) {
                g_cancel_operator_py_callback = std::move(cb);
                if (auto* editor = get_editor_context()) {
                    editor->setCancelOperatorCallback([]() {
                        if (g_cancel_operator_py_callback) {
                            try {
                                g_cancel_operator_py_callback();
                            } catch (const std::exception& e) {
                                LOG_ERROR("Error in operator cancel callback: {}", e.what());
                            }
                        }
                    });
                }
            },
            nb::arg("callback"), "Set callback for operator cancellation (called on ESC)");

        // Set up operator invoke callback to use C++ registry directly
        if (auto* editor = get_editor_context()) {
            editor->setInvokeOperatorCallback([](const char* operator_id) -> bool {
                auto result = vis::op::operators().invoke(operator_id);
                return result.status == vis::op::OperatorResult::FINISHED ||
                       result.status == vis::op::OperatorResult::RUNNING_MODAL;
            });
        }

        // Selection sub-mode access (for Python to read C++ toolbar state)
        m.def("get_selection_submode", &get_selection_submode, "Get current selection sub-mode (0=Brush, 1=Rectangle, 2=Polygon, 3=Lasso, 4=Rings)");

        // Keyboard capture for popup windows
        m.def("request_keyboard_capture", &request_keyboard_capture, nb::arg("owner_id"), "Request exclusive keyboard capture for a named owner");
        m.def("release_keyboard_capture", &release_keyboard_capture, nb::arg("owner_id"), "Release keyboard capture for a named owner");
        m.def("has_keyboard_capture_request", &has_keyboard_capture_request, "Check if any keyboard capture is currently active");

        // Modal event routing - allows Python operators to receive input events
        nb::enum_<ModalEvent::Type>(m, "ModalEventType")
            .value("MouseButton", ModalEvent::Type::MouseButton)
            .value("MouseMove", ModalEvent::Type::MouseMove)
            .value("Scroll", ModalEvent::Type::Scroll)
            .value("Key", ModalEvent::Type::Key);

        nb::class_<ModalEvent>(m, "ModalEvent")
            .def_ro("type", &ModalEvent::type, "Event type (MouseButton, MouseMove, Scroll, Key)")
            .def_ro("x", &ModalEvent::x, "Mouse X position")
            .def_ro("y", &ModalEvent::y, "Mouse Y position")
            .def_ro("delta_x", &ModalEvent::delta_x, "Mouse delta X")
            .def_ro("delta_y", &ModalEvent::delta_y, "Mouse delta Y")
            .def_ro("button", &ModalEvent::button, "Mouse button index")
            .def_ro("action", &ModalEvent::action, "Action code (press, release, repeat)")
            .def_ro("key", &ModalEvent::key, "Key code for keyboard events")
            .def_ro("mods", &ModalEvent::mods, "Modifier key bitmask (shift, ctrl, alt)")
            .def_ro("scroll_x", &ModalEvent::scroll_x, "Horizontal scroll offset")
            .def_ro("scroll_y", &ModalEvent::scroll_y, "Vertical scroll offset")
            .def_ro("over_gui", &ModalEvent::over_gui, "Whether mouse is over a GUI element");

        auto key = m.def_submodule("key", "Key codes");
        key.attr("SPACE") = GLFW_KEY_SPACE;
        key.attr("APOSTROPHE") = GLFW_KEY_APOSTROPHE;
        key.attr("COMMA") = GLFW_KEY_COMMA;
        key.attr("MINUS") = GLFW_KEY_MINUS;
        key.attr("PERIOD") = GLFW_KEY_PERIOD;
        key.attr("SLASH") = GLFW_KEY_SLASH;
        key.attr("NUM_0") = GLFW_KEY_0;
        key.attr("NUM_1") = GLFW_KEY_1;
        key.attr("NUM_2") = GLFW_KEY_2;
        key.attr("NUM_3") = GLFW_KEY_3;
        key.attr("NUM_4") = GLFW_KEY_4;
        key.attr("NUM_5") = GLFW_KEY_5;
        key.attr("NUM_6") = GLFW_KEY_6;
        key.attr("NUM_7") = GLFW_KEY_7;
        key.attr("NUM_8") = GLFW_KEY_8;
        key.attr("NUM_9") = GLFW_KEY_9;
        key.attr("A") = GLFW_KEY_A;
        key.attr("B") = GLFW_KEY_B;
        key.attr("C") = GLFW_KEY_C;
        key.attr("D") = GLFW_KEY_D;
        key.attr("E") = GLFW_KEY_E;
        key.attr("F") = GLFW_KEY_F;
        key.attr("G") = GLFW_KEY_G;
        key.attr("H") = GLFW_KEY_H;
        key.attr("I") = GLFW_KEY_I;
        key.attr("J") = GLFW_KEY_J;
        key.attr("K") = GLFW_KEY_K;
        key.attr("L") = GLFW_KEY_L;
        key.attr("M") = GLFW_KEY_M;
        key.attr("N") = GLFW_KEY_N;
        key.attr("O") = GLFW_KEY_O;
        key.attr("P") = GLFW_KEY_P;
        key.attr("Q") = GLFW_KEY_Q;
        key.attr("R") = GLFW_KEY_R;
        key.attr("S") = GLFW_KEY_S;
        key.attr("T") = GLFW_KEY_T;
        key.attr("U") = GLFW_KEY_U;
        key.attr("V") = GLFW_KEY_V;
        key.attr("W") = GLFW_KEY_W;
        key.attr("X") = GLFW_KEY_X;
        key.attr("Y") = GLFW_KEY_Y;
        key.attr("Z") = GLFW_KEY_Z;
        key.attr("ESCAPE") = GLFW_KEY_ESCAPE;
        key.attr("ENTER") = GLFW_KEY_ENTER;
        key.attr("TAB") = GLFW_KEY_TAB;
        key.attr("BACKSPACE") = GLFW_KEY_BACKSPACE;
        key.attr("INSERT") = GLFW_KEY_INSERT;
        key.attr("DELETE") = GLFW_KEY_DELETE;
        key.attr("RIGHT") = GLFW_KEY_RIGHT;
        key.attr("LEFT") = GLFW_KEY_LEFT;
        key.attr("DOWN") = GLFW_KEY_DOWN;
        key.attr("UP") = GLFW_KEY_UP;
        key.attr("F1") = GLFW_KEY_F1;
        key.attr("F2") = GLFW_KEY_F2;
        key.attr("F3") = GLFW_KEY_F3;
        key.attr("F4") = GLFW_KEY_F4;
        key.attr("F5") = GLFW_KEY_F5;
        key.attr("F6") = GLFW_KEY_F6;
        key.attr("F7") = GLFW_KEY_F7;
        key.attr("F8") = GLFW_KEY_F8;
        key.attr("F9") = GLFW_KEY_F9;
        key.attr("F10") = GLFW_KEY_F10;
        key.attr("F11") = GLFW_KEY_F11;
        key.attr("F12") = GLFW_KEY_F12;
        key.attr("KP_ENTER") = GLFW_KEY_KP_ENTER;

        auto mouse = m.def_submodule("mouse", "Mouse buttons");
        mouse.attr("LEFT") = GLFW_MOUSE_BUTTON_LEFT;
        mouse.attr("RIGHT") = GLFW_MOUSE_BUTTON_RIGHT;
        mouse.attr("MIDDLE") = GLFW_MOUSE_BUTTON_MIDDLE;

        auto mod = m.def_submodule("mod", "Modifier keys");
        mod.attr("SHIFT") = GLFW_MOD_SHIFT;
        mod.attr("CONTROL") = GLFW_MOD_CONTROL;
        mod.attr("ALT") = GLFW_MOD_ALT;
        mod.attr("SUPER") = GLFW_MOD_SUPER;

        auto action = m.def_submodule("action", "Action values");
        action.attr("PRESS") = GLFW_PRESS;
        action.attr("RELEASE") = GLFW_RELEASE;
        action.attr("REPEAT") = GLFW_REPEAT;

        m.def(
            "set_modal_event_callback", [](nb::callable cb) {
                g_modal_event_py_callback = std::move(cb);
                if (auto* editor = get_editor_context()) {
                    editor->setModalEventCallback([](const lfs::core::ModalEvent& evt) -> bool {
                        if (!g_modal_event_py_callback) {
                            return false;
                        }
                        try {
                            // Convert core::ModalEvent to python::ModalEvent for Python binding
                            ModalEvent py_evt{};
                            py_evt.type = static_cast<ModalEvent::Type>(evt.type);
                            py_evt.x = evt.x;
                            py_evt.y = evt.y;
                            py_evt.delta_x = evt.delta_x;
                            py_evt.delta_y = evt.delta_y;
                            py_evt.button = evt.button;
                            py_evt.action = evt.action;
                            py_evt.key = evt.key;
                            py_evt.mods = evt.mods;
                            py_evt.scroll_x = evt.scroll_x;
                            py_evt.scroll_y = evt.scroll_y;
                            py_evt.over_gui = evt.over_gui;
                            nb::object result = g_modal_event_py_callback(py_evt);
                            return nb::cast<bool>(result);
                        } catch (const std::exception& e) {
                            LOG_ERROR("Error in modal event callback: {}", e.what());
                            return false;
                        }
                    });
                }
            },
            nb::arg("callback"), "Set callback for modal events (input dispatch to active operator)");

        m.def(
            "is_point_cloud_forced", []() -> bool {
                auto* ec = get_editor_context();
                return ec && ec->forcePointCloudMode();
            },
            "Check if point cloud mode is forced (pre-training mode)");

        // Status bar bindings
        m.def(
            "get_fps", []() -> float {
                auto* rm = get_rendering_manager();
                if (!rm) {
                    return 0.0f;
                }
                return rm->getAverageFPS();
            },
            "Get current FPS");

        m.def(
            "get_gpu_memory", []() -> std::tuple<size_t, size_t> {
                size_t free_mem = 0, total_mem = 0;
                cudaMemGetInfo(&free_mem, &total_mem);
                return {total_mem - free_mem, total_mem};
            },
            "Get GPU memory (used, total) in bytes");

        m.def(
            "get_content_type", []() -> const char* {
                auto* sm = get_scene_manager();
                if (!sm)
                    return "empty";
                switch (sm->getContentType()) {
                case vis::SceneManager::ContentType::SplatFiles: return "splat_files";
                case vis::SceneManager::ContentType::Dataset: return "dataset";
                default: return "empty";
                }
            },
            "Get content type (empty, splat_files, dataset)");

        m.def(
            "get_git_commit", []() { return GIT_COMMIT_HASH_SHORT; },
            "Get git commit hash");

        m.def(
            "get_split_view_info", []() -> nb::dict {
                auto* rm = get_rendering_manager();
                nb::dict d;
                if (!rm) {
                    d["enabled"] = false;
                    return d;
                }
                auto info = rm->getSplitViewInfo();
                d["enabled"] = info.enabled;
                d["left_name"] = info.left_name;
                d["right_name"] = info.right_name;
                return d;
            },
            "Get split view info");

        m.def(
            "get_current_camera_id", []() -> int {
                auto* rm = get_rendering_manager();
                return rm ? rm->getCurrentCameraId() : -1;
            },
            "Get current camera ID for GT comparison");

        m.def(
            "get_split_view_mode", []() -> const char* {
                auto* rm = get_rendering_manager();
                if (!rm)
                    return "none";
                switch (rm->getSettings().split_view_mode) {
                case vis::SplitViewMode::GTComparison: return "gt_comparison";
                case vis::SplitViewMode::PLYComparison: return "ply_comparison";
                default: return "none";
                }
            },
            "Get split view mode (none, gt_comparison, ply_comparison)");

        m.def(
            "get_speed_overlay", []() -> std::tuple<float, float, float, float> {
                auto [wasd_speed, wasd_alpha] = g_speed_overlay_state.get_wasd();
                auto [zoom_speed, zoom_alpha] = g_speed_overlay_state.get_zoom();
                return {wasd_speed, wasd_alpha, zoom_speed, zoom_alpha};
            },
            "Get speed overlay state (wasd_speed, wasd_alpha, zoom_speed, zoom_alpha)");

        // Subscribe to speed events
        static bool speed_events_initialized = false;
        if (!speed_events_initialized) {
            lfs::core::events::ui::SpeedChanged::when([](const auto& e) {
                g_speed_overlay_state.show_wasd(e.current_speed);
            });
            lfs::core::events::ui::ZoomSpeedChanged::when([](const auto& e) {
                g_speed_overlay_state.show_zoom(e.zoom_speed);
            });
            speed_events_initialized = true;
        }

        m.def(
            "register_property_group", &register_python_property_group, nb::arg("group_id"),
            nb::arg("group_name"), nb::arg("property_group_class"),
            "Register a Python PropertyGroup class with the property registry");

        m.def(
            "unregister_property_group", &unregister_python_property_group, nb::arg("group_id"),
            "Unregister a Python PropertyGroup from the property registry");

        set_python_hook_invoker([](const char* panel, const char* section, bool prepend) {
            auto& registry = PyUIHookRegistry::instance();
            if (registry.has_hooks(panel, section)) {
                registry.invoke(panel, section, prepend ? PyHookPosition::Prepend : PyHookPosition::Append);
            }
        });

        set_python_hook_checker([](const char* panel, const char* section) -> bool {
            return PyUIHookRegistry::instance().has_hooks(panel, section);
        });
    }

    void register_class_api(nb::module_& m) {
        m.def(
            "register_class",
            [](nb::object cls) {
                nb::module_ builtins = nb::module_::import_("builtins");
                auto issubclass = builtins.attr("issubclass");

                nb::module_ types_module = nb::module_::import_("lfs_plugins.types");
                nb::object Panel_type = types_module.attr("Panel");
                nb::object Operator_type = types_module.attr("Operator");
                const nb::object Menu_type = types_module.attr("Menu");

                if (nb::cast<bool>(issubclass(cls, Panel_type))) {
                    PyPanelRegistry::instance().register_panel(cls);
                } else if (nb::cast<bool>(issubclass(cls, Operator_type))) {
                    register_python_operator_to_cpp(cls);
                    std::string idname = get_class_id(cls);
                    std::string label = idname;
                    if (nb::hasattr(cls, "label")) {
                        label = nb::cast<std::string>(cls.attr("label"));
                    }
                    register_python_property_group("operator." + idname, label, cls);
                } else if (nb::cast<bool>(issubclass(cls, Menu_type))) {
                    PyMenuRegistry::instance().register_menu(cls);
                } else {
                    throw std::runtime_error("register_class: must be subclass of Panel, Operator, or Menu");
                }
            },
            nb::arg("cls"), "Register a class (Panel, Operator, or Menu)");

        m.def(
            "unregister_class",
            [](nb::object cls) {
                nb::module_ builtins = nb::module_::import_("builtins");
                auto issubclass = builtins.attr("issubclass");

                nb::module_ types_module = nb::module_::import_("lfs_plugins.types");
                nb::object Panel_type = types_module.attr("Panel");
                nb::object Operator_type = types_module.attr("Operator");
                const nb::object Menu_type = types_module.attr("Menu");

                if (nb::cast<bool>(issubclass(cls, Panel_type))) {
                    PyPanelRegistry::instance().unregister_panel(cls);
                } else if (nb::cast<bool>(issubclass(cls, Operator_type))) {
                    std::string idname = get_class_id(cls);
                    vis::op::operators().unregisterOperator(idname);
                    vis::op::propertySchemas().unregisterSchema(idname);
                    remove_python_operator_instance(idname);
                } else if (nb::cast<bool>(issubclass(cls, Menu_type))) {
                    PyMenuRegistry::instance().unregister_menu(cls);
                }
            },
            nb::arg("cls"), "Unregister a class (Panel, Operator, or Menu)");
    }

} // namespace lfs::python
