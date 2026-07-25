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
#include "core/path_utils.hpp"
#include "core/property_registry.hpp"
#include "core/scene.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/rml_menu_bar.hpp"
#include "gui/utils/file_association.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "gui/vulkan_ui_texture.hpp"
#include "internal/resource_paths.hpp"
#include "io/exporter.hpp"
#include "py_command.hpp"
#include "py_gizmo.hpp"
#include "py_keymap.hpp"
#include "py_params.hpp"
#include "py_prop_registry.hpp"
#include "py_rml.hpp"
#include "py_signals.hpp"
#include "py_store.hpp"
#include "py_tensor.hpp"
#include "py_uilist.hpp"
#include "py_viewport.hpp"
#include "python/gil.hpp"
#include "python/python_runtime.hpp"
#include "python/ui_hooks.hpp"
#include "rendering/render_constants.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/core/editor_context.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/gui/panel_registry.hpp"
#include "visualizer/operation/undo_history.hpp"
#include "visualizer/operator/operator_context.hpp"
#include "visualizer/operator/operator_registry.hpp"
#include "visualizer/operator/property_schema.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/theme/theme.hpp"
#include "visualizer/tools/unified_tool_registry.hpp"
#include "visualizer/training/training_manager.hpp"

#include "config.h"

#include "visualizer/input/key_codes.hpp"

#include <SDL3/SDL_clipboard.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_mouse.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <stack>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#ifdef _WIN32
#include <shellapi.h>
#include <windows.h>
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

        // Icon cache for Python toolbar
        std::unordered_map<std::string, uint64_t> g_icon_cache;
        std::mutex g_icon_cache_mutex;

        // Plugin icon ownership tracking
        std::unordered_map<std::string, std::vector<std::string>> g_plugin_icons;
        std::mutex g_plugin_icons_mutex;

        // Dynamic texture tracking
        std::atomic<bool> g_texture_service_alive{true};
        std::mutex g_dynamic_textures_mutex;

        class PyDynamicTexture;
        std::unordered_set<PyDynamicTexture*> g_all_dynamic_textures;
        std::unordered_map<std::string, std::vector<PyDynamicTexture*>> g_plugin_textures;
        std::unordered_map<std::string, std::unique_ptr<PyDynamicTexture>> g_tensor_cache;

        class PyDynamicTexture {
        public:
            PyDynamicTexture() {
                std::lock_guard lock(g_dynamic_textures_mutex);
                g_all_dynamic_textures.insert(this);
            }

            explicit PyDynamicTexture(const std::string& plugin_name)
                : plugin_name_(plugin_name) {
                std::lock_guard lock(g_dynamic_textures_mutex);
                g_all_dynamic_textures.insert(this);
                if (!plugin_name_.empty())
                    g_plugin_textures[plugin_name_].push_back(this);
            }

            ~PyDynamicTexture() {
                destroy();
                std::lock_guard lock(g_dynamic_textures_mutex);
                g_all_dynamic_textures.erase(this);
                if (!plugin_name_.empty()) {
                    auto it = g_plugin_textures.find(plugin_name_);
                    if (it != g_plugin_textures.end()) {
                        auto& vec = it->second;
                        vec.erase(std::remove(vec.begin(), vec.end(), this), vec.end());
                        if (vec.empty())
                            g_plugin_textures.erase(it);
                    }
                }
            }

            PyDynamicTexture(const PyDynamicTexture&) = delete;
            PyDynamicTexture& operator=(const PyDynamicTexture&) = delete;
            PyDynamicTexture(PyDynamicTexture&&) = delete;
            PyDynamicTexture& operator=(PyDynamicTexture&&) = delete;

            void update(const PyTensor& py_tensor) {
                lfs::python::require_ui_texture_creation_thread();
                auto t = py_tensor.tensor();
                if (t.ndim() != 3)
                    throw std::invalid_argument("DynamicTexture requires 3D tensor [H, W, C]");
                if (t.size(2) != 3 && t.size(2) != 4)
                    throw std::invalid_argument("DynamicTexture channels must be 3 (RGB) or 4 (RGBA)");

                if (t.device() == core::Device::CPU)
                    t = t.cuda();
                const auto orig_dtype = t.dtype();
                if (orig_dtype != core::DataType::Float32)
                    t = t.to(core::DataType::Float32);
                if (orig_dtype == core::DataType::UInt8)
                    t = t / 255.0f;

                const int w = t.size(1);
                const int h = t.size(0);

                if (!texture_) {
                    texture_ = std::make_unique<lfs::vis::gui::VulkanUiTexture>();
                }

                if (!texture_->upload(t, w, h) || !texture_->valid())
                    throw std::runtime_error("Failed to update UI texture");
                width_ = w;
                height_ = h;
            }

            void destroy() {
                if (!texture_)
                    return;
                if (!g_texture_service_alive) {
                    texture_.release();
                } else if (lfs::python::on_graphics_thread()) {
                    texture_.reset();
                } else {
                    auto* raw = texture_.release();
                    lfs::python::schedule_graphics_callback([raw]() { delete raw; });
                }
                width_ = height_ = 0;
            }

            std::unique_ptr<lfs::vis::gui::VulkanUiTexture> release_texture() {
                width_ = height_ = 0;
                return std::move(texture_);
            }

            uint64_t texture_id() const {
                return texture_ ? static_cast<uint64_t>(texture_->textureId()) : 0;
            }

            std::string rml_src_url(const int width, const int height) const {
                if (!valid())
                    return {};
                const int resolved_width = width > 0 ? width : width_;
                const int resolved_height = height > 0 ? height : height_;
                return texture_->rmlSrcUrl(resolved_width, resolved_height);
            }

            int width() const { return width_; }
            int height() const { return height_; }
            bool valid() const { return texture_ != nullptr && texture_->valid() && width_ > 0 && height_ > 0; }

            std::tuple<float, float> uv1() const {
                if (!texture_)
                    return {1.0f, 1.0f};
                return {1.0f, 1.0f};
            }

        private:
            std::unique_ptr<lfs::vis::gui::VulkanUiTexture> texture_;
            std::string plugin_name_;
            int width_ = 0;
            int height_ = 0;
        };

        // Operator cancel callback (Python callable)
        nb::callable g_cancel_operator_py_callback;

        // Modal event callback (Python callable)
        nb::callable g_modal_event_py_callback;

        // Python event callbacks
        nb::object g_popup_draw_callback;
        nb::object g_show_dataset_popup_callback;
        nb::object g_show_resume_popup_callback;
        nb::object g_request_exit_callback;
        nb::object g_open_camera_preview_callback;
        nb::object g_save_asset_callback;

        constexpr std::string_view LEGACY_POPUP_PANEL = "__legacy_popup__";
        constexpr std::string_view LEGACY_POPUP_SECTION = "draw";
        const std::string LEGACY_POPUP_PANEL_STR{LEGACY_POPUP_PANEL};
        const std::string LEGACY_POPUP_SECTION_STR{LEGACY_POPUP_SECTION};

        void warnLegacyPopupDrawCallbackOnce() {
            static std::once_flag once;
            std::call_once(once, [] {
                LOG_WARN("Rml transition: 'register_popup_draw_callback' is a legacy immediate-mode "
                         "compatibility path. Keep existing plugins working, but prefer retained "
                         "Rml panels or UI hooks for new UI.");
            });
        }

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

    } // namespace

    std::string rml_src_for_dynamic_texture(const uint64_t texture_id, const int width, const int height) {
        if (texture_id == 0)
            return {};

        std::lock_guard lock(g_dynamic_textures_mutex);
        for (const auto* texture : g_all_dynamic_textures) {
            if (texture && texture->texture_id() == texture_id)
                return texture->rml_src_url(width, height);
        }
        return {};
    }

    namespace {

        void push_python_operator_undo_entry(const std::string& label, nb::object instance) {
            if (!instance.is_valid() || instance.is_none()) {
                return;
            }
            if (!nb::hasattr(instance, "undo") || !nb::hasattr(instance, "redo")) {
                return;
            }

            auto entry = std::make_unique<lfs::python::PyUndoEntry>(
                label,
                instance.attr("undo"),
                instance.attr("redo"),
                "python.operator",
                "python",
                "operator");
            vis::op::undoHistory().push(std::move(entry));
        }

        uint64_t load_icon_from_path(const std::filesystem::path& path, const std::string& cache_key) {
            const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);

            const auto result = lfs::python::create_ui_texture(data, width, height, channels);
            lfs::core::free_image(data);
            const auto texture_id = result.texture_id;

            {
                std::lock_guard lock(g_icon_cache_mutex);
                g_icon_cache[cache_key] = texture_id;
            }

            return texture_id;
        }

        constexpr const char* DEFAULT_ICON = "default.png";

        uint64_t load_default_icon() {
            {
                std::lock_guard lock(g_icon_cache_mutex);
                auto it = g_icon_cache.find(DEFAULT_ICON);
                if (it != g_icon_cache.end()) {
                    return it->second;
                }
            }
            lfs::python::require_ui_texture_creation_thread();
            try {
                return load_icon_from_path(lfs::vis::getAssetPath("icon/" + std::string(DEFAULT_ICON)), DEFAULT_ICON);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load default icon: {}", e.what());
                return 0;
            }
        }

        uint64_t load_icon_texture(const std::string& icon_name) {
            {
                std::lock_guard lock(g_icon_cache_mutex);
                auto it = g_icon_cache.find(icon_name);
                if (it != g_icon_cache.end()) {
                    return it->second;
                }
            }

            lfs::python::require_ui_texture_creation_thread();

            try {
                return load_icon_from_path(lfs::vis::getAssetPath("icon/" + icon_name), icon_name);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load icon {}: {}", icon_name, e.what());
                return load_default_icon();
            }
        }

        uint64_t load_scene_icon(const std::string& icon_name) {
            const std::string cache_key = "scene/" + icon_name + ".png";
            {
                std::lock_guard lock(g_icon_cache_mutex);
                auto it = g_icon_cache.find(cache_key);
                if (it != g_icon_cache.end()) {
                    return it->second;
                }
            }

            lfs::python::require_ui_texture_creation_thread();

            try {
                return load_icon_from_path(lfs::vis::getAssetPath("icon/scene/" + icon_name + ".png"), cache_key);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load scene icon {}: {}", icon_name, e.what());
                return load_default_icon();
            }
        }

        uint64_t load_plugin_icon(const std::string& icon_name,
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

            lfs::python::require_ui_texture_creation_thread();

            std::filesystem::path icon_path = lfs::core::utf8_to_path(plugin_path) / "icons" / (icon_name + ".png");

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

            std::vector<uint64_t> tex_ids;
            {
                std::lock_guard lock(g_icon_cache_mutex);
                for (const auto& key : keys_to_free) {
                    auto it = g_icon_cache.find(key);
                    if (it != g_icon_cache.end()) {
                        tex_ids.push_back(it->second);
                        g_icon_cache.erase(it);
                    }
                }
            }

            if (tex_ids.empty())
                return;

            if (lfs::python::on_graphics_thread()) {
                for (auto id : tex_ids)
                    lfs::python::delete_ui_texture(id);
            } else {
                lfs::python::schedule_graphics_callback([ids = std::move(tex_ids)]() {
                    for (auto id : ids)
                        lfs::python::delete_ui_texture(id);
                });
            }
        }

        void free_plugin_textures(const std::string& plugin_name) {
            const bool graphics_thread = lfs::python::on_graphics_thread();
            std::vector<lfs::vis::gui::VulkanUiTexture*> deferred;
            {
                std::lock_guard lock(g_dynamic_textures_mutex);
                auto it = g_plugin_textures.find(plugin_name);
                if (it == g_plugin_textures.end())
                    return;
                for (auto* tex : it->second) {
                    if (graphics_thread) {
                        tex->destroy();
                    } else {
                        auto texture = tex->release_texture();
                        if (texture)
                            deferred.push_back(texture.release());
                    }
                    g_all_dynamic_textures.erase(tex);
                }
                g_plugin_textures.erase(it);
            }
            if (!deferred.empty()) {
                lfs::python::schedule_graphics_callback([ptrs = std::move(deferred)]() {
                    for (auto* p : ptrs)
                        delete p;
                });
            }
        }

        // Thread-local layout stack for hierarchical layouts
        thread_local std::stack<LayoutContext> g_layout_stack;

        constexpr float kOverlayTextSize = 14.0f;
        constexpr float kPi = 3.14159265358979323846f;

        lfs::rendering::OverlayColor overlayColor(const nb::object& color) {
            const auto length = nb::len(color);
            assert(length == 3 || length == 4);
            if (length == 3) {
                const auto rgb =
                    nb::cast<std::tuple<float, float, float>>(color);
                return {
                    std::get<0>(rgb),
                    std::get<1>(rgb),
                    std::get<2>(rgb),
                    1.0f,
                };
            }
            if (length != 4)
                throw std::invalid_argument("overlay color must have 3 or 4 components");
            const auto rgba =
                nb::cast<std::tuple<float, float, float, float>>(color);
            return {
                std::get<0>(rgba),
                std::get<1>(rgba),
                std::get<2>(rgba),
                std::get<3>(rgba),
            };
        }

        lfs::rendering::ScreenOverlayRenderer* activeOverlayRenderer() {
            auto* const renderer = get_overlay_draw_context().renderer;
            return renderer && renderer->isFrameActive() ? renderer : nullptr;
        }

        const std::vector<glm::vec2>& roundedRectPoints(
            const float x0, const float y0, const float x1, const float y1,
            const float requested_radius) {
            static thread_local std::vector<glm::vec2> points;
            points.clear();

            const glm::vec2 min = glm::min(glm::vec2{x0, y0}, glm::vec2{x1, y1});
            const glm::vec2 max = glm::max(glm::vec2{x0, y0}, glm::vec2{x1, y1});
            const float radius =
                std::clamp(requested_radius, 0.0f,
                           0.5f * std::min(max.x - min.x, max.y - min.y));
            if (radius <= 0.0f) {
                points.emplace_back(min.x, min.y);
                points.emplace_back(max.x, min.y);
                points.emplace_back(max.x, max.y);
                points.emplace_back(min.x, max.y);
                return points;
            }

            const int segments = std::clamp(
                static_cast<int>(std::ceil(radius * 0.35f)), 3, 24);
            const std::array<glm::vec2, 4> centers = {
                glm::vec2{max.x - radius, min.y + radius},
                glm::vec2{max.x - radius, max.y - radius},
                glm::vec2{min.x + radius, max.y - radius},
                glm::vec2{min.x + radius, min.y + radius},
            };
            const std::array<float, 4> starts = {
                -0.5f * kPi,
                0.0f,
                0.5f * kPi,
                kPi,
            };

            points.reserve(static_cast<std::size_t>(segments + 1) * centers.size());
            for (std::size_t corner = 0; corner < centers.size(); ++corner) {
                for (int segment = 0; segment <= segments; ++segment) {
                    const float angle =
                        starts[corner] +
                        0.5f * kPi * static_cast<float>(segment) /
                            static_cast<float>(segments);
                    points.push_back(
                        centers[corner] +
                        radius * glm::vec2{std::cos(angle), std::sin(angle)});
                }
            }
            return points;
        }

        std::vector<glm::vec2> overlayPoints(
            const std::vector<std::tuple<float, float>>& points) {
            std::vector<glm::vec2> result;
            result.reserve(points.size());
            for (const auto& [x, y] : points)
                result.emplace_back(x, y);
            return result;
        }

        // Window flags for Python bindings
        struct PyWindowFlags {
            static constexpr int NONE = 0;
            static constexpr int NoScrollbar = 1 << 0;
            static constexpr int NoScrollWithMouse = 1 << 1;
            static constexpr int MenuBar = 1 << 2;
            static constexpr int NoResize = 1 << 3;
            static constexpr int NoMove = 1 << 4;
            static constexpr int NoCollapse = 1 << 5;
            static constexpr int AlwaysAutoResize = 1 << 6;
            static constexpr int NoTitleBar = 1 << 7;
            static constexpr int NoNavFocus = 1 << 8;
            static constexpr int NoInputs = 1 << 9;
            static constexpr int NoBackground = 1 << 10;
            static constexpr int NoFocusOnAppearing = 1 << 11;
            static constexpr int NoBringToFrontOnFocus = 1 << 12;
        };

        enum class PyKey {
            ESCAPE = 526,
            ENTER = 525,
            TAB = 512,
            BACKSPACE = 523,
            DELETE_KEY = 522,
            SPACE = 524,
            LEFT = 513,
            RIGHT = 514,
            UP = 515,
            DOWN = 516,
            HOME = 519,
            END = 520,
            F = 551,
            I = 554,
            M = 558,
            R = 563,
            T = 565,
            KEY_1 = 537,
            MINUS = 598,
            EQUAL = 602,
            F2 = 573,
        };

        struct KeyboardFrameState {
            std::array<bool, SDL_SCANCODE_COUNT> previous{};
            std::array<bool, SDL_SCANCODE_COUNT> current{};
            std::thread::id ui_thread;
            bool initialized = false;
        };

        KeyboardFrameState g_keyboard_frame;

        [[nodiscard]] SDL_Scancode to_sdl_scancode(const PyKey key) {
            switch (key) {
            case PyKey::ESCAPE: return SDL_SCANCODE_ESCAPE;
            case PyKey::ENTER: return SDL_SCANCODE_RETURN;
            case PyKey::TAB: return SDL_SCANCODE_TAB;
            case PyKey::BACKSPACE: return SDL_SCANCODE_BACKSPACE;
            case PyKey::DELETE_KEY: return SDL_SCANCODE_DELETE;
            case PyKey::SPACE: return SDL_SCANCODE_SPACE;
            case PyKey::LEFT: return SDL_SCANCODE_LEFT;
            case PyKey::RIGHT: return SDL_SCANCODE_RIGHT;
            case PyKey::UP: return SDL_SCANCODE_UP;
            case PyKey::DOWN: return SDL_SCANCODE_DOWN;
            case PyKey::HOME: return SDL_SCANCODE_HOME;
            case PyKey::END: return SDL_SCANCODE_END;
            case PyKey::F: return SDL_SCANCODE_F;
            case PyKey::I: return SDL_SCANCODE_I;
            case PyKey::M: return SDL_SCANCODE_M;
            case PyKey::R: return SDL_SCANCODE_R;
            case PyKey::T: return SDL_SCANCODE_T;
            case PyKey::KEY_1: return SDL_SCANCODE_1;
            case PyKey::MINUS: return SDL_SCANCODE_MINUS;
            case PyKey::EQUAL: return SDL_SCANCODE_EQUALS;
            case PyKey::F2: return SDL_SCANCODE_F2;
            }
            return SDL_SCANCODE_UNKNOWN;
        }

        void begin_keyboard_ui_frame() {
            g_keyboard_frame.previous = g_keyboard_frame.current;
            g_keyboard_frame.current.fill(false);

            int key_count = 0;
            const bool* const state = SDL_GetKeyboardState(&key_count);
            if (state) {
                const auto count = std::min(
                    static_cast<size_t>(std::max(key_count, 0)),
                    g_keyboard_frame.current.size());
                std::copy_n(state, count, g_keyboard_frame.current.begin());
            }
            g_keyboard_frame.ui_thread = std::this_thread::get_id();
            g_keyboard_frame.initialized = true;
        }

        [[nodiscard]] bool is_key_down(const PyKey key) {
            const auto scancode = to_sdl_scancode(key);
            if (scancode == SDL_SCANCODE_UNKNOWN)
                return false;

            int key_count = 0;
            const bool* const state = SDL_GetKeyboardState(&key_count);
            const auto index = static_cast<int>(scancode);
            return state && index >= 0 && index < key_count && state[index];
        }

        [[nodiscard]] bool is_key_pressed(const PyKey key) {
            if (!g_keyboard_frame.initialized)
                return false;
            assert(g_keyboard_frame.ui_thread == std::this_thread::get_id() &&
                   "is_key_pressed must be queried on the UI thread");

            const auto scancode = to_sdl_scancode(key);
            if (scancode == SDL_SCANCODE_UNKNOWN)
                return false;
            const auto index = static_cast<size_t>(scancode);
            return g_keyboard_frame.current[index] &&
                   !g_keyboard_frame.previous[index];
        }

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

        std::tuple<uint64_t, int, int> create_ui_texture_from_data(unsigned char* data, const int width, const int height, const int channels) {
            if (!data || width <= 0 || height <= 0)
                return {0, 0, 0};

            lfs::python::require_ui_texture_creation_thread();
            ensure_max_texture_size();

            const auto result = lfs::python::create_ui_texture(data, width, height, channels);
            return {static_cast<uint64_t>(result.texture_id), result.width, result.height};
        }

        // SDL returns clipboard payloads keyed by MIME type, chosen by the source app
        // (browsers use image/png, native copies may use image/bmp, etc.), so scan every
        // advertised image/* type rather than guessing a fixed set.
        bool clipboard_has_image() {
            size_t count = 0;
            char** mimes = SDL_GetClipboardMimeTypes(&count);
            if (!mimes)
                return false;
            bool found = false;
            for (size_t i = 0; i < count; ++i) {
                if (mimes[i] && std::string_view(mimes[i]).starts_with("image/")) {
                    found = true;
                    break;
                }
            }
            SDL_free(mimes);
            return found;
        }

        // Returns an owning RGB buffer (caller frees with lfs::core::free_image), or
        // {nullptr, 0, 0, 0} when the clipboard holds no decodable image.
        std::tuple<unsigned char*, int, int, int> decode_clipboard_image() {
            size_t count = 0;
            char** mimes = SDL_GetClipboardMimeTypes(&count);
            if (!mimes)
                return {nullptr, 0, 0, 0};

            std::tuple<unsigned char*, int, int, int> image{nullptr, 0, 0, 0};
            for (size_t i = 0; i < count; ++i) {
                if (!mimes[i] || !std::string_view(mimes[i]).starts_with("image/"))
                    continue;

                size_t size = 0;
                void* raw = SDL_GetClipboardData(mimes[i], &size);
                if (!raw)
                    continue;
                if (size == 0) {
                    SDL_free(raw);
                    continue;
                }

                try {
                    image = lfs::core::load_image_from_memory(
                        static_cast<const uint8_t*>(raw), size);
                } catch (const std::exception& e) {
                    LOG_WARN("Clipboard image decode failed ({}): {}", mimes[i], e.what());
                }
                SDL_free(raw);
                if (std::get<0>(image))
                    break;
            }
            SDL_free(mimes);
            return image;
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

        void apply_operator_props_to_python_instance(const std::string& operator_id,
                                                     const vis::op::OperatorProperties& props,
                                                     nb::object instance) {
            const auto* schemas = vis::op::propertySchemas().getSchema(operator_id);
            if (!schemas || !instance.is_valid() || instance.is_none()) {
                return;
            }

            for (const auto& schema : *schemas) {
                if (!props.has(schema.name)) {
                    continue;
                }

                try {
                    switch (schema.type) {
                    case vis::op::PropertyType::BOOL:
                        if (const auto value = props.get<bool>(schema.name)) {
                            nb::setattr(instance, schema.name.c_str(), nb::bool_(*value));
                        }
                        break;
                    case vis::op::PropertyType::INT:
                        if (const auto value = props.get<int>(schema.name)) {
                            nb::setattr(instance, schema.name.c_str(), nb::int_(*value));
                        }
                        break;
                    case vis::op::PropertyType::FLOAT:
                        if (const auto value = props.get<float>(schema.name)) {
                            nb::setattr(instance, schema.name.c_str(), nb::float_(*value));
                        }
                        break;
                    case vis::op::PropertyType::STRING:
                    case vis::op::PropertyType::ENUM:
                        if (const auto value = props.get<std::string>(schema.name)) {
                            nb::setattr(instance, schema.name.c_str(), nb::str(value->c_str()));
                        }
                        break;
                    case vis::op::PropertyType::FLOAT_VECTOR: {
                        nb::list values;
                        if (schema.size && *schema.size == 3) {
                            if (const auto value = props.get<glm::vec3>(schema.name)) {
                                values.append(nb::float_(value->x));
                                values.append(nb::float_(value->y));
                                values.append(nb::float_(value->z));
                                nb::setattr(instance, schema.name.c_str(), values);
                            }
                        } else if (const auto value = props.get<std::vector<float>>(schema.name)) {
                            for (float item : *value) {
                                values.append(nb::float_(item));
                            }
                            nb::setattr(instance, schema.name.c_str(), values);
                        }
                        break;
                    }
                    case vis::op::PropertyType::INT_VECTOR: {
                        if (const auto value = props.get<std::vector<int>>(schema.name)) {
                            nb::list values;
                            for (int item : *value) {
                                values.append(nb::int_(item));
                            }
                            nb::setattr(instance, schema.name.c_str(), values);
                        }
                        break;
                    }
                    case vis::op::PropertyType::TENSOR:
                        LOG_DEBUG("Skipping tensor property '{}' for Python operator '{}'",
                                  schema.name, operator_id);
                        break;
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("Failed to assign Python operator property '{}.{}': {}",
                              operator_id, schema.name, e.what());
                }
            }
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
                callbacks.invoke = [class_id = id, label = desc.label, has_invoke, has_undo](vis::op::OperatorProperties& props) -> vis::op::OperatorResult {
                    nb::gil_scoped_acquire gil;
                    nb::object instance = get_python_operator_instance(class_id);
                    if (!instance.is_valid() || instance.is_none()) {
                        LOG_ERROR("Python operator '{}' not found during invoke", class_id);
                        return vis::op::OperatorResult::CANCELLED;
                    }
                    try {
                        apply_operator_props_to_python_instance(class_id, props, instance);
                        nb::object result;
                        if (has_invoke) {
                            PyEvent py_event;
                            py_event.type = "INVOKE";
                            py_event.value = "NOTHING";
                            result = instance.attr("invoke")(nb::none(), py_event);
                        } else {
                            result = instance.attr("execute")(nb::none());
                        }
                        const auto status = parse_operator_result(result, instance);
                        if (has_undo && status == vis::op::OperatorResult::FINISHED) {
                            push_python_operator_undo_entry(label.empty() ? class_id : label, instance);
                        }
                        return status;
                    } catch (const std::exception& e) {
                        LOG_ERROR("Operator invoke error: {}", e.what());
                        return vis::op::OperatorResult::CANCELLED;
                    }
                };
            }

            if (has_modal) {
                callbacks.modal = [class_id = id, label = desc.label, has_undo](const vis::op::ModalEvent& event,
                                                                                vis::op::OperatorProperties& /*props*/) -> vis::op::OperatorResult {
                    nb::gil_scoped_acquire gil;
                    nb::object instance = get_python_operator_instance(class_id);
                    if (!instance.is_valid() || instance.is_none()) {
                        LOG_ERROR("Python operator '{}' not found during modal", class_id);
                        return vis::op::OperatorResult::CANCELLED;
                    }
                    try {
                        PyEvent py_event = convert_modal_event(event);
                        const auto redraw_generation_before = lfs::python::redraw_request_generation();
                        nb::object result = instance.attr("modal")(nb::none(), py_event);
                        if (lfs::python::redraw_request_generation() != redraw_generation_before)
                            lfs::python::request_pre_scene_panel_sync();
                        const auto status = parse_operator_result(result, instance);
                        if (has_undo && status == vis::op::OperatorResult::FINISHED) {
                            push_python_operator_undo_entry(label.empty() ? class_id : label, instance);
                        }
                        return status;
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
        std::string key_to_type_string(int key) {
            if (key >= lfs::vis::input::KEY_A && key <= lfs::vis::input::KEY_Z) {
                return std::string("KEY_") + static_cast<char>('A' + (key - lfs::vis::input::KEY_A));
            }
            if (key >= lfs::vis::input::KEY_0 && key <= lfs::vis::input::KEY_9) {
                return std::string("KEY_") + static_cast<char>('0' + (key - lfs::vis::input::KEY_0));
            }
            if (key >= lfs::vis::input::KEY_KP_0 && key <= lfs::vis::input::KEY_KP_9) {
                return std::string("KEY_") + static_cast<char>('0' + (key - lfs::vis::input::KEY_KP_0));
            }
            switch (key) {
            case lfs::vis::input::KEY_SPACE: return "SPACE";
            case lfs::vis::input::KEY_ESCAPE: return "ESC";
            case lfs::vis::input::KEY_ENTER: return "RET";
            case lfs::vis::input::KEY_KP_ENTER: return "RET";
            case lfs::vis::input::KEY_TAB: return "TAB";
            case lfs::vis::input::KEY_BACKSPACE: return "BACK_SPACE";
            case lfs::vis::input::KEY_DELETE: return "DEL";
            case lfs::vis::input::KEY_LEFT: return "LEFT_ARROW";
            case lfs::vis::input::KEY_RIGHT: return "RIGHT_ARROW";
            case lfs::vis::input::KEY_UP: return "UP_ARROW";
            case lfs::vis::input::KEY_DOWN: return "DOWN_ARROW";
            case lfs::vis::input::KEY_HOME: return "HOME";
            case lfs::vis::input::KEY_END: return "END";
            case lfs::vis::input::KEY_PAGE_UP: return "PAGE_UP";
            case lfs::vis::input::KEY_PAGE_DOWN: return "PAGE_DOWN";
            case lfs::vis::input::KEY_LEFT_SHIFT:
            case lfs::vis::input::KEY_RIGHT_SHIFT: return "LEFT_SHIFT";
            case lfs::vis::input::KEY_LEFT_CONTROL:
            case lfs::vis::input::KEY_RIGHT_CONTROL: return "LEFT_CTRL";
            case lfs::vis::input::KEY_LEFT_ALT:
            case lfs::vis::input::KEY_RIGHT_ALT: return "LEFT_ALT";
            case lfs::vis::input::KEY_F1: return "F1";
            case lfs::vis::input::KEY_F2: return "F2";
            case lfs::vis::input::KEY_F3: return "F3";
            case lfs::vis::input::KEY_F4: return "F4";
            case lfs::vis::input::KEY_F5: return "F5";
            case lfs::vis::input::KEY_F6: return "F6";
            case lfs::vis::input::KEY_F7: return "F7";
            case lfs::vis::input::KEY_F8: return "F8";
            case lfs::vis::input::KEY_F9: return "F9";
            case lfs::vis::input::KEY_F10: return "F10";
            case lfs::vis::input::KEY_F11: return "F11";
            case lfs::vis::input::KEY_F12: return "F12";
            default: return "NONE";
            }
        }

        std::string button_to_type_string(int button) {
            switch (button) {
            case static_cast<int>(lfs::vis::input::AppMouseButton::LEFT): return "LEFTMOUSE";
            case static_cast<int>(lfs::vis::input::AppMouseButton::RIGHT): return "RIGHTMOUSE";
            case static_cast<int>(lfs::vis::input::AppMouseButton::MIDDLE): return "MIDDLEMOUSE";
            default: return "MOUSE";
            }
        }

        std::string action_to_value_string(int action) {
            switch (action) {
            case lfs::vis::input::ACTION_PRESS: return "PRESS";
            case lfs::vis::input::ACTION_RELEASE: return "RELEASE";
            default: return "NOTHING";
            }
        }
    } // namespace

    PyEvent convert_modal_event(const vis::op::ModalEvent& event) {
        PyEvent py_event;

        if (const auto* mouse_btn = event.as<vis::MouseButtonEvent>()) {
            py_event.type = button_to_type_string(mouse_btn->button);
            py_event.value = action_to_value_string(mouse_btn->action);
            py_event.mouse_x = mouse_btn->position.x;
            py_event.mouse_y = mouse_btn->position.y;
            py_event.mouse_region_x = mouse_btn->position.x;
            py_event.mouse_region_y = mouse_btn->position.y;
            py_event.shift = (mouse_btn->mods & lfs::vis::input::KEYMOD_SHIFT) != 0;
            py_event.ctrl = (mouse_btn->mods & lfs::vis::input::KEYMOD_CTRL) != 0;
            py_event.alt = (mouse_btn->mods & lfs::vis::input::KEYMOD_ALT) != 0;
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
            py_event.type = key_to_type_string(key->key);
            py_event.value = action_to_value_string(key->action);
            py_event.key_code = key->key;
            py_event.shift = (key->mods & lfs::vis::input::KEYMOD_SHIFT) != 0;
            py_event.ctrl = (key->mods & lfs::vis::input::KEYMOD_CTRL) != 0;
            py_event.alt = (key->mods & lfs::vis::input::KEYMOD_ALT) != 0;
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
        ctx.available_width = std::get<0>(parent_->get_content_region_avail());
        ctx.cursor_start_x = 0.0f;
        ctx.grid_columns = grid_columns_;

        switch (type_) {
        case LayoutType::GridFlow: {
            int cols = grid_columns_;
            if (cols <= 0) {
                const float avail = ctx.available_width;
                cols = std::max(1, static_cast<int>(avail / GRID_AUTO_COLUMN_WIDTH));
            }
            ctx.grid_actual_columns = cols;
            ctx.table_active = cols > 0;
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
        case LayoutType::Row:
        case LayoutType::Box:
        case LayoutType::GridFlow:
        case LayoutType::Split:
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
            ctx.is_first_child = false;
            break;
        case LayoutType::Column:
            ctx.is_first_child = false;
            break;
        case LayoutType::Split:
            if (ctx.child_index == 2) {
                LOG_WARN("Split layout received more than 2 children");
            }
            ctx.child_index++;
            break;
        case LayoutType::GridFlow:
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
        disabled_pushed_ = !eff.enabled;
        color_push_count_ = eff.alert ? 1 : 0;
        font_scale_pushed_ = std::max(eff.scale_x, eff.scale_y) != 1.0f;
    }

    void PySubLayout::pop_per_item_state() {
        font_scale_pushed_ = false;
        color_push_count_ = 0;
        disabled_pushed_ = false;
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
    void PySubLayout::text_colored(const std::string& text, nb::object color) {
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
        apply_state();
        const bool changed = parent_->prop_enum(data, prop_id, value, text);
        pop_per_item_state();
        return changed;
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
                            nb::object tint) {
        advance_child();
        apply_state();
        parent_->image(texture_id, size, tint);
        pop_per_item_state();
    }
    bool PySubLayout::image_button(const std::string& id, uint64_t texture_id,
                                   std::tuple<float, float> size,
                                   nb::object tint) {
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
    bool PySubLayout::menu_item(const std::string& label, bool enabled, bool selected) const {
        return parent_->menu_item(label, enabled, selected);
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

    void PyUILayout::warnUnsupportedInDrawHook(const char* method) const {
        if (!isDrawHook())
            return;

        // All call sites pass static method-name literals, so pointer storage is stable.
        static std::array<std::atomic<const char*>, 256> warned_methods{};
        size_t hash = 1469598103934665603ULL;
        for (const unsigned char c : std::string_view(method)) {
            hash ^= c;
            hash *= 1099511628211ULL;
        }

        for (size_t probe = 0; probe < warned_methods.size(); ++probe) {
            auto& entry = warned_methods[(hash + probe) % warned_methods.size()];
            const char* existing = entry.load(std::memory_order_acquire);
            if (existing && std::strcmp(existing, method) == 0)
                return;
            if (!existing) {
                const char* expected = nullptr;
                if (entry.compare_exchange_strong(
                        expected, method, std::memory_order_acq_rel)) {
                    LOG_WARN("UILayout::{} unsupported in draw hooks; use a panel or document hook",
                             method);
                    return;
                }
                if (expected && std::strcmp(expected, method) == 0)
                    return;
            }
        }
        assert(false && "draw-hook warning registry exhausted");
    }

    namespace {
        void warnRetainedCustomElementOnce(const char* method, const char* element) {
            static std::mutex mutex;
            static std::unordered_set<std::string> warned_methods;
            std::lock_guard lock(mutex);
            if (warned_methods.emplace(method).second) {
                LOG_WARN("UILayout::{} is unavailable in layout APIs; use the retained RmlUi {} element",
                         method, element);
            }
        }
    } // namespace

    // PyUILayout layout methods
    PySubLayout PyUILayout::row() { return PySubLayout(this, LayoutType::Row); }
    PySubLayout PyUILayout::column() { return PySubLayout(this, LayoutType::Column); }
    PySubLayout PyUILayout::split(float factor) { return PySubLayout(this, LayoutType::Split, factor); }
    PySubLayout PyUILayout::box() { return PySubLayout(this, LayoutType::Box); }
    PySubLayout PyUILayout::grid_flow(int columns, bool, bool) {
        return PySubLayout(this, LayoutType::GridFlow, 0.5f, columns);
    }

    bool PyUILayout::prop_enum(nb::object, const std::string&,
                               const std::string&, const std::string&) {
        if (isDrawHook()) {
            warnUnsupportedInDrawHook("prop_enum");
            return false;
        }
        throw std::runtime_error("prop_enum requires an active RmlUILayout backend");
    }

    nb::object PyUILayout::operator_(const std::string& operator_id, const std::string& text,
                                     const std::string& /*icon*/) {
        warnUnsupportedInDrawHook("operator_");
        const auto* desc = vis::op::operators().getDescriptor(operator_id);
        const std::string label = text.empty() ? (desc ? desc->label : operator_id) : text;
        const std::string btn_text = LOC(label.c_str());
        const bool can_execute = vis::op::operators().poll(operator_id);

        if (collecting_ && collect_target_) {
            vis::gui::MenuItemDesc item;
            item.type = vis::gui::MenuItemDesc::Type::Operator;
            item.label = btn_text;
            item.operator_id = operator_id;
            item.enabled = can_execute;
            collect_target_->items.push_back(std::move(item));
        }

        return nb::cast(PyOperatorProperties(operator_id));
    }

    std::tuple<bool, int> PyUILayout::prop_search(nb::object /*data*/, const std::string& prop_id,
                                                  nb::object search_data, const std::string& search_prop,
                                                  const std::string& /*text*/) {
        warnUnsupportedInDrawHook("prop_search");
        int current_idx = 0;
        try {
            if (nb::hasattr(search_data, search_prop.c_str())) {
                nb::object collection = search_data.attr(search_prop.c_str());
                if (nb::hasattr(collection, "__iter__")) {
                    for (auto item : collection) {
                        if (nb::hasattr(item, "selected") && nb::cast<bool>(item.attr("selected"))) {
                            break;
                        }
                        ++current_idx;
                    }
                }
            }
        } catch (...) {
            LOG_WARN("prop_search: failed to enumerate items for '{}'", prop_id);
            current_idx = 0;
        }
        return {false, current_idx};
    }

    std::tuple<int, int> PyUILayout::template_list(const std::string& /*list_type_id*/,
                                                   const std::string& /*list_id*/,
                                                   nb::object, const std::string&,
                                                   nb::object, const std::string&,
                                                   int /*rows*/) {
        if (isDrawHook()) {
            warnUnsupportedInDrawHook("template_list");
            return {0, 0};
        }
        throw nb::type_error(
            "UILayout.template_list is unsupported; use the live "
            "RmlUILayout.template_list API from a Python panel draw callback");
    }

    void PyUILayout::menu(const std::string&, const std::string&, const std::string&) {
        warnUnsupportedInDrawHook("menu");
    }
    void PyUILayout::popover(const std::string&, const std::string&, const std::string&) {
        warnUnsupportedInDrawHook("popover");
    }

    void PyUILayout::draw_circle(const float x, const float y, const float radius,
                                 nb::object color, const int segments,
                                 const float thickness) {
        if (auto* const renderer = activeOverlayRenderer())
            renderer->addCircle({x, y}, radius, overlayColor(color), segments, thickness);
    }

    void PyUILayout::draw_circle_filled(const float x, const float y,
                                        const float radius, nb::object color,
                                        const int segments) {
        if (auto* const renderer = activeOverlayRenderer())
            renderer->addCircleFilled({x, y}, radius, overlayColor(color), segments);
    }

    void PyUILayout::draw_rect(const float x0, const float y0, const float x1,
                               const float y1, nb::object color,
                               const float thickness) {
        if (auto* const renderer = activeOverlayRenderer())
            renderer->addRect({x0, y0}, {x1, y1}, overlayColor(color), thickness);
    }

    void PyUILayout::draw_rect_filled(const float x0, const float y0,
                                      const float x1, const float y1,
                                      nb::object color, bool) {
        if (auto* const renderer = activeOverlayRenderer())
            renderer->addRectFilled({x0, y0}, {x1, y1}, overlayColor(color));
    }

    void PyUILayout::draw_rect_rounded(const float x0, const float y0,
                                       const float x1, const float y1,
                                       nb::object color, const float rounding,
                                       const float thickness, bool) {
        if (auto* const renderer = activeOverlayRenderer()) {
            const auto& points = roundedRectPoints(x0, y0, x1, y1, rounding);
            renderer->addPolyline(points, overlayColor(color), true, thickness);
        }
    }

    void PyUILayout::draw_rect_rounded_filled(
        const float x0, const float y0, const float x1, const float y1,
        nb::object color, const float rounding, bool) {
        if (auto* const renderer = activeOverlayRenderer()) {
            const auto& points = roundedRectPoints(x0, y0, x1, y1, rounding);
            renderer->addConvexPolyFilled(points, overlayColor(color));
        }
    }

    void PyUILayout::draw_triangle_filled(
        const float x0, const float y0, const float x1, const float y1,
        const float x2, const float y2, nb::object color, bool) {
        if (auto* const renderer = activeOverlayRenderer()) {
            renderer->addTriangleFilled(
                {x0, y0}, {x1, y1}, {x2, y2}, overlayColor(color));
        }
    }

    void PyUILayout::draw_line(const float x0, const float y0, const float x1,
                               const float y1, nb::object color,
                               const float thickness) {
        if (auto* const renderer = activeOverlayRenderer())
            renderer->addLine({x0, y0}, {x1, y1}, overlayColor(color), thickness);
    }

    void PyUILayout::draw_polyline(
        const std::vector<std::tuple<float, float>>& points, nb::object color,
        const bool closed, const float thickness) {
        if (auto* const renderer = activeOverlayRenderer()) {
            const auto converted = overlayPoints(points);
            renderer->addPolyline(converted, overlayColor(color), closed, thickness);
        }
    }

    void PyUILayout::draw_poly_filled(
        const std::vector<std::tuple<float, float>>& points, nb::object color) {
        if (auto* const renderer = activeOverlayRenderer()) {
            const auto converted = overlayPoints(points);
            renderer->addConvexPolyFilled(converted, overlayColor(color));
        }
    }

    void PyUILayout::draw_text(const float x, const float y,
                               const std::string& text, nb::object color, bool) {
        if (auto* const renderer = activeOverlayRenderer()) {
            renderer->addText({x, y}, text, overlayColor(color),
                              kOverlayTextSize * python::get_shared_dpi_scale());
        }
    }

    void PyUILayout::draw_window_rect_filled(
        const float x0, const float y0, const float x1, const float y1,
        nb::object color) {
        draw_rect_filled(x0, y0, x1, y1, std::move(color), false);
    }

    void PyUILayout::draw_window_rect(
        const float x0, const float y0, const float x1, const float y1,
        nb::object color, const float thickness) {
        draw_rect(x0, y0, x1, y1, std::move(color), thickness);
    }

    void PyUILayout::draw_window_rect_rounded(
        const float x0, const float y0, const float x1, const float y1,
        nb::object color, const float rounding, const float thickness) {
        draw_rect_rounded(x0, y0, x1, y1, std::move(color), rounding,
                          thickness, false);
    }

    void PyUILayout::draw_window_rect_rounded_filled(
        const float x0, const float y0, const float x1, const float y1,
        nb::object color, const float rounding) {
        draw_rect_rounded_filled(x0, y0, x1, y1, std::move(color), rounding,
                                 false);
    }

    void PyUILayout::draw_window_line(
        const float x0, const float y0, const float x1, const float y1,
        nb::object color, const float thickness) {
        draw_line(x0, y0, x1, y1, std::move(color), thickness);
    }

    void PyUILayout::draw_window_text(const float x, const float y,
                                      const std::string& text,
                                      nb::object color) {
        draw_text(x, y, text, std::move(color), false);
    }

    void PyUILayout::draw_window_triangle_filled(
        const float x0, const float y0, const float x1, const float y1,
        const float x2, const float y2, nb::object color) {
        draw_triangle_filled(x0, y0, x1, y1, x2, y2, std::move(color), false);
    }
    void PyUILayout::crf_curve_preview(const std::string&, float, float, float, float, float, float) {
        warnRetainedCustomElementOnce("crf_curve_preview", "<crf-curve>");
    }

    std::tuple<bool, std::vector<float>> PyUILayout::chromaticity_diagram(
        const std::string&,
        float red_x, float red_y, float green_x, float green_y,
        float blue_x, float blue_y, float neutral_x, float neutral_y,
        float) {
        warnRetainedCustomElementOnce(
            "chromaticity_diagram", "<chromaticity-diagram>");
        return {false, {red_x, red_y, green_x, green_y, blue_x, blue_y, neutral_x, neutral_y}};
    }

    void PyUILayout::label(const std::string&) { warnUnsupportedInDrawHook("label"); }
    void PyUILayout::label_centered(const std::string&) {
        warnUnsupportedInDrawHook("label_centered");
    }
    void PyUILayout::heading(const std::string&) { warnUnsupportedInDrawHook("heading"); }
    void PyUILayout::text_colored(const std::string&, nb::object) {
        warnUnsupportedInDrawHook("text_colored");
    }
    void PyUILayout::text_colored_centered(const std::string&, nb::object) {
        warnUnsupportedInDrawHook("text_colored_centered");
    }
    void PyUILayout::text_selectable(const std::string&, float) {
        warnUnsupportedInDrawHook("text_selectable");
    }
    void PyUILayout::bullet_text(const std::string&) {
        warnUnsupportedInDrawHook("bullet_text");
    }
    void PyUILayout::text_wrapped(const std::string&) {
        warnUnsupportedInDrawHook("text_wrapped");
    }
    void PyUILayout::text_disabled(const std::string&) {
        warnUnsupportedInDrawHook("text_disabled");
    }

    bool PyUILayout::button(const std::string&, std::tuple<float, float>) {
        warnUnsupportedInDrawHook("button");
        return false;
    }
    bool PyUILayout::button_callback(const std::string&, nb::object,
                                     std::tuple<float, float>) {
        warnUnsupportedInDrawHook("button_callback");
        return false;
    }
    bool PyUILayout::small_button(const std::string&) {
        warnUnsupportedInDrawHook("small_button");
        return false;
    }
    std::tuple<bool, bool> PyUILayout::checkbox(const std::string&, bool value) {
        warnUnsupportedInDrawHook("checkbox");
        return {false, value};
    }
    std::tuple<bool, int> PyUILayout::radio_button(const std::string&, int current, int) {
        warnUnsupportedInDrawHook("radio_button");
        return {false, current};
    }
    std::tuple<bool, float> PyUILayout::slider_float(const std::string&, float value,
                                                     float, float) {
        warnUnsupportedInDrawHook("slider_float");
        return {false, value};
    }
    std::tuple<bool, int> PyUILayout::slider_int(const std::string&, int value, int, int) {
        warnUnsupportedInDrawHook("slider_int");
        return {false, value};
    }
    std::tuple<bool, std::tuple<float, float>> PyUILayout::slider_float2(
        const std::string&, std::tuple<float, float> value, float, float) {
        warnUnsupportedInDrawHook("slider_float2");
        return {false, value};
    }
    std::tuple<bool, std::tuple<float, float, float>> PyUILayout::slider_float3(
        const std::string&, std::tuple<float, float, float> value, float, float) {
        warnUnsupportedInDrawHook("slider_float3");
        return {false, value};
    }
    std::tuple<bool, float> PyUILayout::drag_float(const std::string&, float value,
                                                   float, float, float) {
        warnUnsupportedInDrawHook("drag_float");
        return {false, value};
    }
    std::tuple<bool, int> PyUILayout::drag_int(const std::string&, int value,
                                               float, int, int) {
        warnUnsupportedInDrawHook("drag_int");
        return {false, value};
    }
    std::tuple<bool, std::string> PyUILayout::input_text(const std::string&,
                                                         const std::string& value) {
        warnUnsupportedInDrawHook("input_text");
        return {false, value};
    }
    std::tuple<bool, std::string> PyUILayout::input_text_with_hint(
        const std::string&, const std::string&, const std::string& value) {
        warnUnsupportedInDrawHook("input_text_with_hint");
        return {false, value};
    }
    std::tuple<bool, float> PyUILayout::input_float(
        const std::string&, float value, float, float, const std::string&) {
        warnUnsupportedInDrawHook("input_float");
        return {false, value};
    }
    std::tuple<bool, int> PyUILayout::input_int(const std::string&, int value, int, int) {
        warnUnsupportedInDrawHook("input_int");
        return {false, value};
    }
    std::tuple<bool, int> PyUILayout::input_int_formatted(const std::string&, int value,
                                                          int, int) {
        warnUnsupportedInDrawHook("input_int_formatted");
        return {false, value};
    }
    std::tuple<bool, float> PyUILayout::stepper_float(
        const std::string&, float value, const std::vector<float>&) {
        warnUnsupportedInDrawHook("stepper_float");
        return {false, value};
    }
    std::tuple<bool, std::string> PyUILayout::path_input(
        const std::string&, const std::string& value, bool, const std::string&) {
        warnUnsupportedInDrawHook("path_input");
        return {false, value};
    }
    std::tuple<bool, std::tuple<float, float, float>> PyUILayout::color_edit3(
        const std::string&, std::tuple<float, float, float> color) {
        warnUnsupportedInDrawHook("color_edit3");
        return {false, color};
    }
    std::tuple<bool, std::tuple<float, float, float, float>> PyUILayout::color_edit4(
        const std::string&, std::tuple<float, float, float, float> color) {
        warnUnsupportedInDrawHook("color_edit4");
        return {false, color};
    }
    std::tuple<bool, std::tuple<float, float, float>> PyUILayout::color_picker3(
        const std::string&, std::tuple<float, float, float> color) {
        warnUnsupportedInDrawHook("color_picker3");
        return {false, color};
    }
    bool PyUILayout::color_button(const std::string&, nb::object,
                                  std::tuple<float, float>) {
        warnUnsupportedInDrawHook("color_button");
        return false;
    }
    std::tuple<bool, int> PyUILayout::combo(const std::string&, int current_idx, const std::vector<std::string>&) {
        warnUnsupportedInDrawHook("combo");
        return {false, current_idx};
    }
    std::tuple<bool, int> PyUILayout::listbox(const std::string&, int current_idx, const std::vector<std::string>&, int) {
        warnUnsupportedInDrawHook("listbox");
        return {false, current_idx};
    }

    void PyUILayout::separator() {
        if (collecting_ && collect_target_) {
            vis::gui::MenuItemDesc item;
            item.type = vis::gui::MenuItemDesc::Type::Separator;
            collect_target_->items.push_back(std::move(item));
            return;
        }
        warnUnsupportedInDrawHook("separator");
    }
    void PyUILayout::spacing() { warnUnsupportedInDrawHook("spacing"); }
    void PyUILayout::same_line(float, float) { warnUnsupportedInDrawHook("same_line"); }
    void PyUILayout::new_line() { warnUnsupportedInDrawHook("new_line"); }
    void PyUILayout::indent(float) { warnUnsupportedInDrawHook("indent"); }
    void PyUILayout::unindent(float) { warnUnsupportedInDrawHook("unindent"); }
    void PyUILayout::set_next_item_width(float) {
        warnUnsupportedInDrawHook("set_next_item_width");
    }
    void PyUILayout::begin_group() { warnUnsupportedInDrawHook("begin_group"); }
    void PyUILayout::end_group() { warnUnsupportedInDrawHook("end_group"); }
    bool PyUILayout::collapsing_header(const std::string&, bool) {
        if (isDrawHook()) {
            warnUnsupportedInDrawHook("collapsing_header");
            return false;
        }
        return true;
    }
    bool PyUILayout::tree_node(const std::string&) {
        if (isDrawHook()) {
            warnUnsupportedInDrawHook("tree_node");
            return false;
        }
        return true;
    }
    bool PyUILayout::tree_node_ex(const std::string&, const std::string&) {
        if (isDrawHook()) {
            warnUnsupportedInDrawHook("tree_node_ex");
            return false;
        }
        return true;
    }
    void PyUILayout::set_next_item_open(bool) {
        warnUnsupportedInDrawHook("set_next_item_open");
    }
    void PyUILayout::tree_pop() { warnUnsupportedInDrawHook("tree_pop"); }
    bool PyUILayout::begin_table(const std::string&, int) {
        if (isDrawHook()) {
            warnUnsupportedInDrawHook("begin_table");
            return false;
        }
        return true;
    }
    void PyUILayout::end_table() { warnUnsupportedInDrawHook("end_table"); }
    void PyUILayout::table_next_row() { warnUnsupportedInDrawHook("table_next_row"); }
    void PyUILayout::table_next_column() {
        warnUnsupportedInDrawHook("table_next_column");
    }
    bool PyUILayout::table_set_column_index(int) {
        if (isDrawHook()) {
            warnUnsupportedInDrawHook("table_set_column_index");
            return false;
        }
        return true;
    }
    void PyUILayout::table_headers_row() {
        warnUnsupportedInDrawHook("table_headers_row");
    }
    void PyUILayout::table_set_bg_color(int, nb::object) {
        warnUnsupportedInDrawHook("table_set_bg_color");
    }
    void PyUILayout::table_setup_column(const std::string&, float) {
        warnUnsupportedInDrawHook("table_setup_column");
    }
    bool PyUILayout::button_styled(const std::string&, const std::string&,
                                   std::tuple<float, float>) {
        warnUnsupportedInDrawHook("button_styled");
        return false;
    }
    void PyUILayout::push_item_width(float) {
        warnUnsupportedInDrawHook("push_item_width");
    }
    void PyUILayout::pop_item_width() { warnUnsupportedInDrawHook("pop_item_width"); }
    void PyUILayout::plot_lines(const std::string&, const std::vector<float>&,
                                float, float, std::tuple<float, float>) {
        warnUnsupportedInDrawHook("plot_lines");
    }
    bool PyUILayout::selectable(const std::string&, bool, float) {
        warnUnsupportedInDrawHook("selectable");
        return false;
    }
    bool PyUILayout::begin_context_menu(const std::string&) {
        warnUnsupportedInDrawHook("begin_context_menu");
        return false;
    }
    void PyUILayout::end_context_menu() {
        warnUnsupportedInDrawHook("end_context_menu");
    }
    bool PyUILayout::begin_popup(const std::string&) {
        warnUnsupportedInDrawHook("begin_popup");
        return false;
    }
    void PyUILayout::open_popup(const std::string&) {
        warnUnsupportedInDrawHook("open_popup");
    }
    void PyUILayout::end_popup() { warnUnsupportedInDrawHook("end_popup"); }

    bool PyUILayout::menu_item(const std::string& label, bool enabled, bool selected) {
        if (collecting_ && collect_target_) {
            vis::gui::MenuItemDesc item;
            item.type = vis::gui::MenuItemDesc::Type::Item;
            item.label = label;
            item.enabled = enabled;
            item.selected = selected;
            const int idx = collect_callback_index_++;
            item.callback_index = idx;
            collect_target_->items.push_back(std::move(item));
            return execute_at_index_ == idx;
        }
        warnUnsupportedInDrawHook("menu_item");
        return false;
    }

    bool PyUILayout::begin_menu(const std::string& label) {
        if (collecting_ && collect_target_) {
            vis::gui::MenuItemDesc item;
            item.type = vis::gui::MenuItemDesc::Type::SubMenuBegin;
            item.label = label;
            collect_target_->items.push_back(std::move(item));
            ++menu_depth_;
            return true;
        }
        warnUnsupportedInDrawHook("begin_menu");
        return false;
    }

    void PyUILayout::end_menu() {
        if (menu_depth_ > 0) {
            --menu_depth_;
        }
        if (collecting_ && collect_target_) {
            vis::gui::MenuItemDesc item;
            item.type = vis::gui::MenuItemDesc::Type::SubMenuEnd;
            collect_target_->items.push_back(std::move(item));
            return;
        }
        warnUnsupportedInDrawHook("end_menu");
    }

    std::tuple<bool, std::string> PyUILayout::input_text_enter(
        const std::string&, const std::string& value) {
        warnUnsupportedInDrawHook("input_text_enter");
        return {false, value};
    }
    void PyUILayout::set_keyboard_focus_here() {
        warnUnsupportedInDrawHook("set_keyboard_focus_here");
    }
    bool PyUILayout::is_window_focused() const { return false; }
    bool PyUILayout::is_window_hovered() const { return false; }
    void PyUILayout::capture_keyboard_from_app(bool capture) { vis::gui::guiFocusState().want_capture_keyboard = capture; }
    void PyUILayout::capture_mouse_from_app(bool capture) { vis::gui::guiFocusState().want_capture_mouse = capture; }
    void PyUILayout::set_scroll_here_y(float) {
        warnUnsupportedInDrawHook("set_scroll_here_y");
    }
    std::tuple<float, float> PyUILayout::get_cursor_screen_pos() const { return {0.0f, 0.0f}; }
    std::tuple<float, float> PyUILayout::get_mouse_pos() const {
        float x = 0.0f;
        float y = 0.0f;
        SDL_GetMouseState(&x, &y);
        return {x, y};
    }
    std::tuple<float, float> PyUILayout::get_window_pos() const {
        return isDrawHook() ? window_pos_ : std::tuple<float, float>{0.0f, 0.0f};
    }
    float PyUILayout::get_window_width() const {
        return isDrawHook() ? std::get<0>(window_size_) : 0.0f;
    }
    float PyUILayout::get_text_line_height() const { return 16.0f * python::get_shared_dpi_scale(); }
    bool PyUILayout::begin_popup_modal(const std::string&) {
        warnUnsupportedInDrawHook("begin_popup_modal");
        return false;
    }
    void PyUILayout::end_popup_modal() {
        warnUnsupportedInDrawHook("end_popup_modal");
    }
    void PyUILayout::close_current_popup() {
        warnUnsupportedInDrawHook("close_current_popup");
    }
    void PyUILayout::set_next_window_pos_center() {
        warnUnsupportedInDrawHook("set_next_window_pos_center");
    }
    void PyUILayout::set_next_window_pos_viewport_center(bool) {
        warnUnsupportedInDrawHook("set_next_window_pos_viewport_center");
    }
    void PyUILayout::set_next_window_focus() {
        warnUnsupportedInDrawHook("set_next_window_focus");
    }
    void PyUILayout::push_modal_style() { lfs::vis::theme().pushModalStyle(); }
    void PyUILayout::pop_modal_style() { lfs::vis::theme().popModalStyle(); }
    std::tuple<float, float> PyUILayout::get_content_region_avail() { return get_viewport_size(); }
    std::tuple<float, float> PyUILayout::get_cursor_pos() { return {0.0f, 0.0f}; }
    void PyUILayout::set_cursor_pos_x(float) {
        warnUnsupportedInDrawHook("set_cursor_pos_x");
    }
    std::tuple<float, float> PyUILayout::calc_text_size(const std::string& text) {
        const float scale = python::get_shared_dpi_scale();
        if (auto* const renderer = activeOverlayRenderer()) {
            const glm::vec2 measured =
                renderer->measureText(text, kOverlayTextSize * scale);
            return {measured.x, measured.y};
        }
        return {static_cast<float>(text.size()) * 7.0f * scale, 16.0f * scale};
    }
    void PyUILayout::begin_disabled(bool) {
        warnUnsupportedInDrawHook("begin_disabled");
    }
    void PyUILayout::end_disabled() { warnUnsupportedInDrawHook("end_disabled"); }
    void PyUILayout::image(uint64_t, std::tuple<float, float>, nb::object) {
        warnUnsupportedInDrawHook("image");
    }
    void PyUILayout::image_uv(uint64_t, std::tuple<float, float>,
                              std::tuple<float, float>, std::tuple<float, float>,
                              nb::object) {
        warnUnsupportedInDrawHook("image_uv");
    }
    bool PyUILayout::image_button(const std::string&, uint64_t,
                                  std::tuple<float, float>, nb::object) {
        warnUnsupportedInDrawHook("image_button");
        return false;
    }
    bool PyUILayout::toolbar_button(const std::string&, uint64_t, std::tuple<float, float>, bool, bool, const std::string&) {
        warnUnsupportedInDrawHook("toolbar_button");
        return false;
    }
    bool PyUILayout::begin_drag_drop_source() {
        warnUnsupportedInDrawHook("begin_drag_drop_source");
        return false;
    }
    void PyUILayout::set_drag_drop_payload(const std::string&, const std::string&) {
        warnUnsupportedInDrawHook("set_drag_drop_payload");
    }
    void PyUILayout::end_drag_drop_source() {
        warnUnsupportedInDrawHook("end_drag_drop_source");
    }
    bool PyUILayout::begin_drag_drop_target() {
        warnUnsupportedInDrawHook("begin_drag_drop_target");
        return false;
    }
    std::optional<std::string> PyUILayout::accept_drag_drop_payload(const std::string&) {
        warnUnsupportedInDrawHook("accept_drag_drop_payload");
        return std::nullopt;
    }
    void PyUILayout::end_drag_drop_target() {
        warnUnsupportedInDrawHook("end_drag_drop_target");
    }
    void PyUILayout::progress_bar(float, const std::string&, float, float) {
        warnUnsupportedInDrawHook("progress_bar");
    }
    void PyUILayout::set_tooltip(const std::string&) {
        warnUnsupportedInDrawHook("set_tooltip");
    }
    bool PyUILayout::is_item_hovered() { return false; }
    bool PyUILayout::is_item_clicked(int) { return false; }
    bool PyUILayout::is_mouse_double_clicked(int) { return false; }
    bool PyUILayout::is_mouse_dragging(int) { return false; }
    float PyUILayout::get_mouse_wheel() { return 0.0f; }
    std::tuple<float, float> PyUILayout::get_mouse_delta() { return {0.0f, 0.0f}; }
    bool PyUILayout::invisible_button(const std::string&, std::tuple<float, float>) {
        warnUnsupportedInDrawHook("invisible_button");
        return false;
    }
    bool PyUILayout::is_item_active() { return false; }
    void PyUILayout::set_cursor_pos(std::tuple<float, float>) {
        warnUnsupportedInDrawHook("set_cursor_pos");
    }
    bool PyUILayout::begin_child(const std::string&, std::tuple<float, float>, bool) {
        if (isDrawHook()) {
            warnUnsupportedInDrawHook("begin_child");
            return false;
        }
        return true;
    }
    void PyUILayout::end_child() { warnUnsupportedInDrawHook("end_child"); }
    bool PyUILayout::begin_menu_bar() {
        if (isDrawHook()) {
            warnUnsupportedInDrawHook("begin_menu_bar");
            return false;
        }
        return true;
    }
    void PyUILayout::end_menu_bar() { warnUnsupportedInDrawHook("end_menu_bar"); }

    bool PyUILayout::menu_item_toggle(const std::string& label, const std::string& shortcut, bool selected) {
        if (collecting_ && collect_target_) {
            vis::gui::MenuItemDesc item;
            item.type = vis::gui::MenuItemDesc::Type::Toggle;
            item.label = label;
            item.shortcut = shortcut;
            item.selected = selected;
            const int idx = collect_callback_index_++;
            item.callback_index = idx;
            collect_target_->items.push_back(std::move(item));
            return execute_at_index_ == idx;
        }
        warnUnsupportedInDrawHook("menu_item_toggle");
        return false;
    }

    bool PyUILayout::menu_item_shortcut(const std::string& label, const std::string& shortcut, bool enabled) {
        if (collecting_ && collect_target_) {
            vis::gui::MenuItemDesc item;
            item.type = vis::gui::MenuItemDesc::Type::ShortcutItem;
            item.label = label;
            item.shortcut = shortcut;
            item.enabled = enabled;
            const int idx = collect_callback_index_++;
            item.callback_index = idx;
            collect_target_->items.push_back(std::move(item));
            return execute_at_index_ == idx;
        }
        warnUnsupportedInDrawHook("menu_item_shortcut");
        return false;
    }

    void PyUILayout::push_id(const std::string&) { warnUnsupportedInDrawHook("push_id"); }
    void PyUILayout::push_id_int(int) { warnUnsupportedInDrawHook("push_id_int"); }
    void PyUILayout::pop_id() { warnUnsupportedInDrawHook("pop_id"); }
    bool PyUILayout::begin_window(const std::string&, int) {
        if (isDrawHook()) {
            if (next_window_pos_set_)
                window_pos_ = next_window_pos_;
            if (next_window_size_set_)
                window_size_ = next_window_size_;
            next_window_pos_ = {0.0f, 0.0f};
            next_window_size_ = {0.0f, 0.0f};
            next_window_pos_set_ = false;
            next_window_size_set_ = false;
        }
        return true;
    }
    std::tuple<bool, bool> PyUILayout::begin_window_closable(const std::string&, int) {
        if (isDrawHook()) {
            warnUnsupportedInDrawHook("begin_window_closable");
            return {false, true};
        }
        return {true, true};
    }
    void PyUILayout::end_window() {}
    void PyUILayout::push_window_style() {
        warnUnsupportedInDrawHook("push_window_style");
    }
    void PyUILayout::pop_window_style() {
        warnUnsupportedInDrawHook("pop_window_style");
    }
    void PyUILayout::set_next_window_pos(std::tuple<float, float> pos, bool) {
        next_window_pos_ = pos;
        next_window_pos_set_ = true;
    }
    void PyUILayout::set_next_window_size(std::tuple<float, float> size, bool) {
        next_window_size_ = size;
        next_window_size_set_ = true;
    }
    void PyUILayout::set_next_window_pos_centered(bool) {
        const auto [viewport_x, viewport_y] = get_viewport_pos();
        const auto [viewport_w, viewport_h] = get_viewport_size();
        next_window_pos_ = {
            viewport_x + (viewport_w - std::get<0>(next_window_size_)) * 0.5f,
            viewport_y + (viewport_h - std::get<1>(next_window_size_)) * 0.5f,
        };
        next_window_pos_set_ = true;
    }
    void PyUILayout::set_next_window_bg_alpha(float) {
        warnUnsupportedInDrawHook("set_next_window_bg_alpha");
    }

    std::tuple<float, float> PyUILayout::get_viewport_pos() {
        if (has_viewport_bounds()) {
            float x, y, w, h;
            get_viewport_bounds(x, y, w, h);
            return {x, y};
        }
        return {0.0f, 0.0f};
    }

    std::tuple<float, float> PyUILayout::get_viewport_size() {
        if (has_viewport_bounds()) {
            float x, y, w, h;
            get_viewport_bounds(x, y, w, h);
            return {w, h};
        }
        return {0.0f, 0.0f};
    }

    float PyUILayout::get_dpi_scale() { return python::get_shared_dpi_scale(); }
    void PyUILayout::set_mouse_cursor_hand() {
        warnUnsupportedInDrawHook("set_mouse_cursor_hand");
    }
    void PyUILayout::push_style_var_float(const std::string&, float) {
        warnUnsupportedInDrawHook("push_style_var");
    }
    void PyUILayout::push_style_var_vec2(const std::string&, std::tuple<float, float>) {
        warnUnsupportedInDrawHook("push_style_var_vec2");
    }
    void PyUILayout::pop_style_var(int) {
        warnUnsupportedInDrawHook("pop_style_var");
    }
    void PyUILayout::push_style_color(const std::string&, nb::object) {
        warnUnsupportedInDrawHook("push_style_color");
    }
    void PyUILayout::pop_style_color(int) {
        warnUnsupportedInDrawHook("pop_style_color");
    }

    std::tuple<bool, nb::object> PyUILayout::prop(nb::object data,
                                                  const std::string& prop_id,
                                                  std::optional<std::string>) {
        warnUnsupportedInDrawHook("prop");
        if (isDrawHook()) {
            if (nb::hasattr(data, prop_id.c_str()))
                return {false, data.attr(prop_id.c_str())};
            return {false, nb::none()};
        }

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

        const bool has_get = nb::hasattr(data, "get") && PyCallable_Check(data.attr("get").ptr());
        nb::object current_value = has_get
                                       ? data.attr("get")(nb::cast(prop_id))
                                       : data.attr(prop_id.c_str());
        return {false, current_value};
    }

    void shutdown_dynamic_textures() {
        assert(lfs::python::on_graphics_thread());
        lfs::python::flush_graphics_callbacks();
        decltype(g_tensor_cache) cache_to_destroy;
        {
            std::lock_guard lock(g_dynamic_textures_mutex);
            for (auto* tex : g_all_dynamic_textures) {
                tex->destroy();
            }
            g_all_dynamic_textures.clear();
            g_plugin_textures.clear();
            g_texture_service_alive = false;
            cache_to_destroy = std::move(g_tensor_cache);
        }
        // ~PyDynamicTexture destructors run here without holding the mutex
    }

    void register_ui_context_menu(nb::module_& m) {
        const auto make_python_context_menu_callback =
            [](nb::object callback) -> lfs::vis::gui::GlobalContextMenu::ActionCallback {
            if (callback.is_none())
                return {};
            if (!PyCallable_Check(callback.ptr()))
                throw nb::type_error("show_context_menu on_action must be callable or None");

            PyObject* const callable = callback.ptr();
            Py_INCREF(callable);
            const auto callable_ref = std::shared_ptr<PyObject>(callable, [](PyObject* obj) {
                if (!obj || !lfs::python::can_acquire_gil())
                    return;
                const lfs::python::GilAcquire gil;
                Py_DECREF(obj);
            });

            return [callable_ref](const std::string_view action) {
                if (!lfs::python::can_acquire_gil()) {
                    LOG_ERROR("Unable to run Python context menu callback: Python GIL is unavailable");
                    return;
                }

                const lfs::python::GilAcquire gil;
                PyObject* const py_action = PyUnicode_FromStringAndSize(action.data(), action.size());
                if (!py_action) {
                    LOG_ERROR("Python context menu callback argument creation failed: {}",
                              lfs::python::extract_python_error());
                    return;
                }

                PyObject* const result = PyObject_CallFunctionObjArgs(callable_ref.get(), py_action, nullptr);
                Py_DECREF(py_action);
                if (result) {
                    Py_DECREF(result);
                } else {
                    LOG_ERROR("Python context menu callback failed: {}", lfs::python::extract_python_error());
                }
            };
        };

        m.def(
            "show_context_menu",
            [make_python_context_menu_callback](nb::list items, float sx, float sy, nb::object on_action) {
                auto* cm = get_global_context_menu();
                if (!cm)
                    return;

                std::vector<lfs::vis::gui::ContextMenuItem> vec;
                vec.reserve(nb::len(items));

                for (auto item_handle : items) {
                    auto d = nb::cast<nb::dict>(item_handle);
                    lfs::vis::gui::ContextMenuItem ci;
                    ci.label = nb::cast<std::string>(d["label"]);
                    ci.action = nb::cast<std::string>(d["action"]);
                    if (d.contains("separator_before"))
                        ci.separator_before = nb::cast<bool>(d["separator_before"]);
                    if (d.contains("is_label"))
                        ci.is_label = nb::cast<bool>(d["is_label"]);
                    if (d.contains("is_submenu_item"))
                        ci.is_submenu_item = nb::cast<bool>(d["is_submenu_item"]);
                    if (d.contains("is_active"))
                        ci.is_active = nb::cast<bool>(d["is_active"]);
                    vec.push_back(std::move(ci));
                }

                cm->request(std::move(vec), sx, sy, make_python_context_menu_callback(std::move(on_action)));
            },
            nb::arg("items"), nb::arg("screen_x"), nb::arg("screen_y"), nb::arg("on_action") = nb::none());

        m.def("poll_context_menu", []() -> std::string {
            auto* cm = get_global_context_menu();
            if (!cm)
                return "";
            return cm->pollResult();
        });

        m.def("get_mouse_screen_pos", []() -> nb::tuple {
            float x = 0.0f;
            float y = 0.0f;
            SDL_GetMouseState(&x, &y);
            return nb::make_tuple(x, y);
        });

        m.def(
            "get_display_size", []() -> nb::tuple {
                if (has_viewport_bounds()) {
                    float x, y, w, h;
                    get_viewport_bounds(x, y, w, h);
                    return nb::make_tuple(w, h);
                }
                return nb::make_tuple(0.0f, 0.0f);
            },
            "Get display work area size as (width, height)");
    }

    // Register UI classes with nanobind module
    void register_ui(nb::module_& m) {
        lfs::python::set_graphics_thread_id(std::this_thread::get_id());
        g_texture_service_alive = true;

        // Call sub-registration functions
        register_ui_context(m);
        register_ui_theme(m);
        register_ui_panels(m);
        register_rml_im_mode_layout(m);
        register_ui_hooks(m);
        register_ui_menus(m);
        register_ui_context_menu(m);
        register_ui_operators(m);
        register_ui_modals(m);
        register_rml_bindings(m);

        // Hot-reload redraw request functions
        m.def(
            "request_redraw", []() { lfs::python::request_redraw(); }, "Request a UI redraw on next frame");
        m.def(
            "consume_redraw_request", []() {
                return lfs::python::consume_redraw_request();
            },
            "Consume and return pending redraw request flag");
        const auto schedule_on_ui_thread = [](nb::callable callback) {
            if (!callback.is_valid())
                throw nb::type_error("schedule_on_ui_thread requires a callable");

            PyObject* const callable = callback.ptr();
            Py_INCREF(callable);
            const auto callable_ref = std::shared_ptr<PyObject>(callable, [](PyObject* obj) {
                if (!obj || !lfs::python::can_acquire_gil())
                    return;
                const lfs::python::GilAcquire gil;
                Py_DECREF(obj);
            });

            lfs::python::schedule_graphics_callback([callable_ref]() {
                if (!lfs::python::can_acquire_gil()) {
                    LOG_ERROR("Unable to run scheduled Python UI callback: Python GIL is unavailable");
                    return;
                }

                const lfs::python::GilAcquire gil;
                PyObject* const result = PyObject_CallNoArgs(callable_ref.get());
                if (result) {
                    Py_DECREF(result);
                } else {
                    LOG_ERROR("Scheduled Python UI callback failed: {}", lfs::python::extract_python_error());
                }
            });
            lfs::python::request_redraw();
        };
        m.def(
            "schedule_on_ui_thread",
            schedule_on_ui_thread,
            nb::arg("callback"),
            "Schedule a Python callable on the UI thread");
        m.def(
            "_run_on_ui_thread",
            schedule_on_ui_thread,
            nb::arg("callback"),
            "Schedule a Python callable on the UI thread");

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
            .def("image", &PySubLayout::image, nb::arg("texture_id"), nb::arg("size"), nb::arg("tint") = nb::none())
            .def("image_button", &PySubLayout::image_button, nb::arg("id"), nb::arg("texture_id"), nb::arg("size"), nb::arg("tint") = nb::none())
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
            .def("menu_item", &PySubLayout::menu_item, nb::arg("label"), nb::arg("enabled") = true, nb::arg("selected") = false)
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
            .def_ro_static("NONE", &PyWindowFlags::NONE, "No flags set")
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

        auto layout_class = nb::class_<PyUILayout>(m, "UILayout");
        layout_class.def(nb::init<>(), "Create a UILayout for drawing UI elements")
            .def_prop_ro_static(
                "WindowFlags", [](nb::handle) { return PyWindowFlags{}; }, "Window flags constants")
            // Text
            .def("label", &PyUILayout::label, nb::arg("text"), "Draw a text label")
            .def("label_centered", &PyUILayout::label_centered, nb::arg("text"), "Draw a horizontally centered text label")
            .def("heading", &PyUILayout::heading, nb::arg("text"), "Draw a bold heading text")
            .def("text_colored", &PyUILayout::text_colored, nb::arg("text"), nb::arg("color"), "Draw text with RGB or RGBA color tuple")
            .def("text_colored_centered", &PyUILayout::text_colored_centered, nb::arg("text"), nb::arg("color"), "Draw centered text with RGB or RGBA color tuple")
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
                 nb::arg("folder_mode") = true, nb::arg("dialog_title") = "",
                 "Unsupported on compatibility UILayout. In a draw hook, warns once "
                 "and returns (False, value); use RmlUILayout.path_input in a panel.")
            // Color
            .def("color_edit3", &PyUILayout::color_edit3, nb::arg("label"), nb::arg("color"), "Draw an RGB color editor, returns (changed, color)")
            .def("color_edit4", &PyUILayout::color_edit4, nb::arg("label"), nb::arg("color"), "Draw an RGBA color editor, returns (changed, color)")
            .def("color_picker3", &PyUILayout::color_picker3, nb::arg("label"), nb::arg("color"), "Draw a full RGB color picker widget, returns (changed, color)")
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
            .def("set_next_item_open", &PyUILayout::set_next_item_open, nb::arg("is_open"), "Force the next tree node or collapsing header open/closed")
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
            .def("menu_item", &PyUILayout::menu_item, nb::arg("label"), nb::arg("enabled") = true, nb::arg("selected") = false, "Draw a menu item, returns True if clicked")
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
            // Custom row backgrounds
            .def("get_cursor_screen_pos", &PyUILayout::get_cursor_screen_pos, "Get cursor position in screen coordinates as (x, y)")
            .def("get_mouse_pos", &PyUILayout::get_mouse_pos, "Get mouse position in screen coordinates as (x, y)")
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
                 nb::arg("tint") = nb::none(), "Draw an image from a UI texture ID")
            .def("image_uv", &PyUILayout::image_uv, nb::arg("texture_id"), nb::arg("size"),
                 nb::arg("uv0"), nb::arg("uv1"),
                 nb::arg("tint") = nb::none(), "Draw an image with custom UV coordinates")
            .def("image_button", &PyUILayout::image_button, nb::arg("id"), nb::arg("texture_id"),
                 nb::arg("size"), nb::arg("tint") = nb::none(), "Draw an image button, returns True if clicked")
            .def("toolbar_button", &PyUILayout::toolbar_button, nb::arg("id"), nb::arg("texture_id"),
                 nb::arg("size"), nb::arg("selected") = false, nb::arg("disabled") = false,
                 nb::arg("tooltip") = "", "Draw a toolbar-style icon button with selection state")
            .def(
                "image_texture", [](PyUILayout& self, PyDynamicTexture& tex, std::tuple<float, float> size, nb::object tint) {
                    if (!tex.valid())
                        return;
                    self.image_uv(tex.texture_id(), size, {0.0f, 0.0f}, tex.uv1(), std::move(tint));
                },
                nb::arg("texture"), nb::arg("size"), nb::arg("tint") = nb::none(), "Draw a DynamicTexture with automatic UV scaling")
            .def("image_tensor", [](PyUILayout& self, const std::string& label, PyTensor& tensor, std::tuple<float, float> size, nb::object tint) {
                    PyDynamicTexture* tex_ptr = nullptr;
                    {
                        std::lock_guard lock(g_dynamic_textures_mutex);
                        auto it = g_tensor_cache.find(label);
                        if (it != g_tensor_cache.end())
                            tex_ptr = it->second.get();
                    }
                    if (!tex_ptr) {
                        auto new_tex = std::make_unique<PyDynamicTexture>();
                        tex_ptr = new_tex.get();
                        std::lock_guard lock(g_dynamic_textures_mutex);
                        g_tensor_cache.try_emplace(label, std::move(new_tex));
                    }
                    tex_ptr->update(tensor);
                    auto& tex = *tex_ptr;
                    self.image_uv(tex.texture_id(), size, {0.0f, 0.0f}, tex.uv1(), std::move(tint)); }, nb::arg("label"), nb::arg("tensor"), nb::arg("size"), nb::arg("tint") = nb::none(), "Draw a tensor as an image, caching the UI texture by label")
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
            .def("prop", &PyUILayout::prop, nb::arg("data"), nb::arg("prop_id"), nb::arg("text") = nb::none(), "Draw a property widget based on metadata (auto-selects widget type)")
            .def("row", &PyUILayout::row, "Create a horizontal row sub-layout")
            .def("column", &PyUILayout::column, "Create a vertical column sub-layout")
            .def("split", &PyUILayout::split, nb::arg("factor") = 0.5f, "Create a split sub-layout with given factor")
            .def("box", &PyUILayout::box, "Create a bordered box sub-layout")
            .def("grid_flow", &PyUILayout::grid_flow, nb::arg("columns") = 0, nb::arg("even_columns") = true, nb::arg("even_rows") = true, "Create a responsive grid sub-layout")
            .def("prop_enum", &PyUILayout::prop_enum, nb::arg("data"), nb::arg("prop_id"), nb::arg("value"), nb::arg("text") = "", "Draw an enum toggle button for a property value")
            .def("operator_", &PyUILayout::operator_, nb::arg("operator_id"), nb::arg("text") = "", nb::arg("icon") = "", "Draw a button that invokes a registered operator")
            .def("prop_search", &PyUILayout::prop_search, nb::arg("data"), nb::arg("prop_id"), nb::arg("search_data"), nb::arg("search_prop"), nb::arg("text") = "", "Searchable dropdown for selecting from a collection")
            .def("template_list", &PyUILayout::template_list, nb::arg("list_type_id"), nb::arg("list_id"), nb::arg("data"), nb::arg("prop_id"), nb::arg("active_data"), nb::arg("active_prop"), nb::arg("rows") = 5, "Unsupported on UILayout; use RmlUILayout.template_list.")
            .def("menu", &PyUILayout::menu, nb::arg("menu_id"), nb::arg("text") = "", nb::arg("icon") = "", "Inline menu reference")
            .def("popover", &PyUILayout::popover, nb::arg("panel_id"), nb::arg("text") = "", nb::arg("icon") = "", "Panel popover")
            // Drawing functions for viewport overlays
            .def("draw_circle", &PyUILayout::draw_circle, nb::arg("x"), nb::arg("y"), nb::arg("radius"), nb::arg("color"), nb::arg("segments") = 32, nb::arg("thickness") = 1.0f, "Enqueue a circle in the active viewport ScreenOverlayRenderer using absolute screen coordinates. Emits nothing outside an overlay frame.")
            .def("draw_circle_filled", &PyUILayout::draw_circle_filled, nb::arg("x"), nb::arg("y"), nb::arg("radius"), nb::arg("color"), nb::arg("segments") = 32, "Draw a filled circle at (x, y) with given radius and color")
            .def("draw_rect", &PyUILayout::draw_rect, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("thickness") = 1.0f, "Draw a rectangle outline from (x0,y0) to (x1,y1)")
            .def("draw_rect_filled", &PyUILayout::draw_rect_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("background") = false, "Enqueue a filled overlay rectangle; background is compatibility-only.")
            .def("draw_rect_rounded", &PyUILayout::draw_rect_rounded, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("rounding"), nb::arg("thickness") = 1.0f, nb::arg("background") = false, "Enqueue a tessellated rounded outline; background is compatibility-only.")
            .def("draw_rect_rounded_filled", &PyUILayout::draw_rect_rounded_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("rounding"), nb::arg("background") = false, "Enqueue a tessellated rounded fill; background is compatibility-only.")
            .def("draw_triangle_filled", &PyUILayout::draw_triangle_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("x2"), nb::arg("y2"), nb::arg("color"), nb::arg("background") = false, "Enqueue a filled overlay triangle; background is compatibility-only.")
            .def("draw_line", &PyUILayout::draw_line, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("thickness") = 1.0f, "Draw a line from (x0,y0) to (x1,y1)")
            .def("draw_polyline", &PyUILayout::draw_polyline, nb::arg("points"), nb::arg("color"), nb::arg("closed") = false, nb::arg("thickness") = 1.0f, "Draw a polyline through the given points")
            .def("draw_poly_filled", &PyUILayout::draw_poly_filled, nb::arg("points"), nb::arg("color"), "Draw a filled convex polygon")
            .def("draw_text", &PyUILayout::draw_text, nb::arg("x"), nb::arg("y"), nb::arg("text"), nb::arg("color"), nb::arg("background") = false, "Enqueue overlay text at absolute screen coordinates; background is ignored.")
            // Window-prefixed compatibility aliases use the same absolute screen space.
            .def("draw_window_rect_filled", &PyUILayout::draw_window_rect_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), "Enqueue a filled rectangle in the viewport ScreenOverlayRenderer.")
            .def("draw_window_rect", &PyUILayout::draw_window_rect, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("thickness") = 1.0f, "Enqueue a rectangle outline in the viewport ScreenOverlayRenderer.")
            .def("draw_window_rect_rounded", &PyUILayout::draw_window_rect_rounded, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("rounding"), nb::arg("thickness") = 1.0f, "Enqueue a tessellated rounded outline in the viewport overlay.")
            .def("draw_window_rect_rounded_filled", &PyUILayout::draw_window_rect_rounded_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("rounding"), "Enqueue a tessellated rounded fill in the viewport overlay.")
            .def("draw_window_line", &PyUILayout::draw_window_line, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("thickness") = 1.0f, "Enqueue a line in the viewport ScreenOverlayRenderer.")
            .def("draw_window_text", &PyUILayout::draw_window_text, nb::arg("x"), nb::arg("y"), nb::arg("text"), nb::arg("color"), "Enqueue text in the viewport ScreenOverlayRenderer.")
            .def("draw_window_triangle_filled", &PyUILayout::draw_window_triangle_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("x2"), nb::arg("y2"), nb::arg("color"), "Enqueue a filled triangle in the viewport ScreenOverlayRenderer. Like every draw_window_* alias, coordinates are absolute screen space.")
            .def("crf_curve_preview", &PyUILayout::crf_curve_preview, nb::arg("label"), nb::arg("gamma"), nb::arg("toe"), nb::arg("shoulder"), nb::arg("gamma_r") = 0.0f, nb::arg("gamma_g") = 0.0f, nb::arg("gamma_b") = 0.0f, "Unsupported in layout APIs; use the retained RmlUi <crf-curve> custom element.")
            .def("chromaticity_diagram", &PyUILayout::chromaticity_diagram, nb::arg("label"), nb::arg("red_x"), nb::arg("red_y"), nb::arg("green_x"), nb::arg("green_y"), nb::arg("blue_x"), nb::arg("blue_y"), nb::arg("neutral_x"), nb::arg("neutral_y"), nb::arg("range") = 0.5f, "Unsupported in layout APIs; use the retained RmlUi <chromaticity-diagram> custom element.");
        add_template_methods_to_uilayout(layout_class);

        // File dialogs
        m.def(
            "open_image_dialog",
            [](const std::string& start_dir) -> std::string {
                std::filesystem::path start_path;
                if (!start_dir.empty()) {
                    start_path = lfs::core::utf8_to_path(start_dir);
                }
                auto result = lfs::vis::gui::OpenImageFileDialog(start_path);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("start_dir") = "",
            "Open a file dialog to select an image file. Returns empty string if cancelled.");

        m.def(
            "open_environment_map_dialog",
            [](const std::string& start_dir) -> std::string {
                std::filesystem::path start_path;
                if (!start_dir.empty()) {
                    start_path = lfs::core::utf8_to_path(start_dir);
                }
                auto result = lfs::vis::gui::OpenEnvironmentMapFileDialog(start_path);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("start_dir") = "",
            "Open a file dialog to select an environment map (.hdr, .exr). Returns empty string if cancelled.");

        m.def(
            "open_folder_dialog",
            [](const std::string& /*title*/, const std::string& start_dir) -> std::string {
                // `title` is accepted for Python API compatibility; native dialogs currently ignore it.
                std::filesystem::path start_path;
                if (!start_dir.empty()) {
                    start_path = lfs::core::utf8_to_path(start_dir);
                }
                auto result = lfs::vis::gui::PickFolderDialog(start_path);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("title") = "Select Folder", nb::arg("start_dir") = "",
            "Open a folder selection dialog. Returns empty string if cancelled. "
            "title is accepted for compatibility and currently ignored.");

        m.def(
            "open_ply_file_dialog",
            [](const std::string& start_dir) -> std::string {
                std::filesystem::path start_path;
                if (!start_dir.empty()) {
                    start_path = lfs::core::utf8_to_path(start_dir);
                }
                auto result = lfs::vis::gui::OpenPointCloudFileDialog(start_path);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("start_dir") = "",
            "Open a file dialog to select a splat file (.ply, .sog, .spz, .rad, .usd, .usda, .usdc, .usdz). Returns empty string if cancelled.");

        m.def(
            "open_mesh_file_dialog",
            [](const std::string& start_dir) -> std::string {
                std::filesystem::path start_path;
                if (!start_dir.empty()) {
                    start_path = lfs::core::utf8_to_path(start_dir);
                }
                auto result = lfs::vis::gui::OpenMeshFileDialog(start_path);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("start_dir") = "",
            "Open a file dialog to select a mesh file. Returns empty string if cancelled.");

        m.def(
            "open_checkpoint_file_dialog",
            []() -> std::string {
                auto result = lfs::vis::gui::OpenCheckpointFileDialog();
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            "Open a file dialog to select a checkpoint file. Returns empty string if cancelled.");

        m.def(
            "open_ppisp_file_dialog",
            [](const std::string& start_dir) -> std::string {
                std::filesystem::path start_path;
                if (!start_dir.empty()) {
                    start_path = lfs::core::utf8_to_path(start_dir);
                }
                auto result = lfs::vis::gui::OpenPPISPFileDialog(start_path);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("start_dir") = "",
            "Open a file dialog to select a PPISP sidecar file. Returns empty string if cancelled.");

        m.def(
            "open_json_file_dialog",
            []() -> std::string {
                auto result = lfs::vis::gui::OpenJsonFileDialog();
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            "Open a file dialog to select a JSON config file. Returns empty string if cancelled.");

        m.def(
            "open_csv_file_dialog",
            []() -> std::string {
                auto result = lfs::vis::gui::OpenCsvFileDialog();
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            "Open a file dialog to select a CSV file. Returns empty string if cancelled.");

        m.def(
            "open_xml_file_dialog",
            []() -> std::string {
                auto result = lfs::vis::gui::OpenXmlFileDialog();
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            "Open a file dialog to select a Metashape XML file. Returns empty string if cancelled.");

        m.def(
            "open_las_file_dialog",
            []() -> std::string {
                auto result = lfs::vis::gui::OpenLasFileDialog();
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            "Open a file dialog to select a LAS or LAZ point cloud file. Returns empty string if cancelled.");

        m.def(
            "save_las_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveLasFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "export",
            "Open a save file dialog for LAS files. Returns empty string if cancelled.");

        m.def(
            "save_laz_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveLazFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "export",
            "Open a save file dialog for LAZ compressed files. Returns empty string if cancelled.");

        m.def(
            "save_json_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveJsonFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "config.json",
            "Open a save file dialog for JSON files. Returns empty string if cancelled.");

        m.def(
            "save_png_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SavePngFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "export.png",
            "Open a save file dialog for PNG images. Returns empty string if cancelled.");

        m.def(
            "save_jpg_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveJpgFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "export.jpg",
            "Open a save file dialog for JPEG images. Returns empty string if cancelled.");

        m.def(
            "save_ply_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SavePlyFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "export",
            "Open a save file dialog for PLY files. Returns empty string if cancelled.");

        m.def(
            "save_sog_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveSogFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "export",
            "Open a save file dialog for SOG files. Returns empty string if cancelled.");

        m.def(
            "save_spz_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveSpzFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "export",
            "Open a save file dialog for SPZ files. Returns empty string if cancelled.");

        m.def(
            "save_usd_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveUsdFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "export",
            "Open a save file dialog for USD files. Returns empty string if cancelled.");

        m.def(
            "save_usdz_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveUsdzFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "export",
            "Open a save file dialog for USDZ files. Returns empty string if cancelled.");

        m.def(
            "save_html_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveHtmlFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "viewer",
            "Open a save file dialog for HTML viewer files. Returns empty string if cancelled.");

        m.def(
            "save_rad_file_dialog",
            [](const std::string& default_name) -> std::string {
                auto result = lfs::vis::gui::SaveRadFileDialog(default_name);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_name") = "export",
            "Open a save file dialog for RAD files. Returns empty string if cancelled.");

        m.def(
            "open_dataset_folder_dialog",
            [](const std::string& default_path) -> std::string {
                const auto default_fs_path = default_path.empty()
                                                 ? std::filesystem::path{}
                                                 : lfs::core::utf8_to_path(default_path);
                auto result = lfs::vis::gui::OpenDatasetFolderDialog(default_fs_path);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_path") = "",
            "Open a folder dialog to select a dataset. Returns empty string if cancelled.");

        m.def(
            "select_colmap_sparse_folder_dialog",
            [](const std::string& default_path) -> std::string {
                const auto default_fs_path = default_path.empty()
                                                 ? std::filesystem::path{}
                                                 : lfs::core::utf8_to_path(default_path);
                auto result = lfs::vis::gui::PickColmapSparseFolderDialog(default_fs_path);
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
            },
            nb::arg("default_path") = "",
            "Open a folder dialog to select the COLMAP sparse export folder. Returns empty string if cancelled.");

        m.def(
            "open_video_file_dialog",
            []() -> std::string {
                auto result = lfs::vis::gui::OpenVideoFileDialog();
                return result.empty() ? "" : lfs::core::path_to_utf8(result);
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
            nb::arg("tool"), "Switch to a toolbar tool (none, selection, translate, rotate, scale, mirror, align, cropbox)");

        // Key enum (subset of commonly used keys)
        nb::enum_<PyKey>(m, "Key")
            .value("ESCAPE", PyKey::ESCAPE)
            .value("ENTER", PyKey::ENTER)
            .value("TAB", PyKey::TAB)
            .value("BACKSPACE", PyKey::BACKSPACE)
            .value("DELETE", PyKey::DELETE_KEY)
            .value("SPACE", PyKey::SPACE)
            .value("LEFT", PyKey::LEFT)
            .value("RIGHT", PyKey::RIGHT)
            .value("UP", PyKey::UP)
            .value("DOWN", PyKey::DOWN)
            .value("HOME", PyKey::HOME)
            .value("END", PyKey::END)
            .value("F", PyKey::F)
            .value("I", PyKey::I)
            .value("M", PyKey::M)
            .value("R", PyKey::R)
            .value("T", PyKey::T)
            .value("_1", PyKey::KEY_1)
            .value("MINUS", PyKey::MINUS)
            .value("EQUAL", PyKey::EQUAL)
            .value("F2", PyKey::F2);

        // Key input functions
        m.def(
            "is_key_pressed",
            [](const PyKey key, bool repeat) {
                (void)repeat;
                return is_key_pressed(key);
            },
            nb::arg("key"), nb::arg("repeat") = false,
            "Return the SDL-backed rising edge for the current UI frame. Must be "
            "queried on the UI thread; returns False before frame initialization. "
            "The repeat argument is retained only for call compatibility; key "
            "repeat is not emitted.");

        m.def(
            "is_key_down",
            [](const PyKey key) { return is_key_down(key); },
            nb::arg("key"), "Return the current SDL keyboard level for the key.");

        m.def(
            "is_ctrl_down",
            []() { return (SDL_GetModState() & SDL_KMOD_CTRL) != 0; },
            "Check if Ctrl is currently held");

        m.def(
            "is_shift_down",
            []() { return (SDL_GetModState() & SDL_KMOD_SHIFT) != 0; },
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
                warnLegacyPopupDrawCallbackOnce();
                auto& hooks = PyUIHookRegistry::instance();
                if (g_popup_draw_callback && !g_popup_draw_callback.is_none()) {
                    hooks.remove_hook(LEGACY_POPUP_PANEL_STR,
                                      LEGACY_POPUP_SECTION_STR,
                                      g_popup_draw_callback);
                }
                g_popup_draw_callback = callback;
                if (g_popup_draw_callback && !g_popup_draw_callback.is_none()) {
                    hooks.add_hook(LEGACY_POPUP_PANEL_STR,
                                   LEGACY_POPUP_SECTION_STR,
                                   g_popup_draw_callback,
                                   PyHookPosition::Append);
                }
            },
            nb::arg("callback"), "Register a legacy immediate-mode callback for drawing popup content");

        m.def(
            "unregister_popup_draw_callback",
            [](nb::object callback) {
                PyUIHookRegistry::instance().remove_hook(LEGACY_POPUP_PANEL_STR,
                                                         LEGACY_POPUP_SECTION_STR,
                                                         callback);
                if (g_popup_draw_callback.is_valid() && g_popup_draw_callback.is(callback)) {
                    g_popup_draw_callback = nb::none();
                }
            },
            nb::arg("callback"), "Unregister a legacy popup draw callback");

        m.def(
            "on_show_dataset_load_popup",
            [](nb::object callback) {
                g_show_dataset_popup_callback = callback;
                lfs::core::events::cmd::ShowDatasetLoadPopup::when([](const auto& e) {
                    if (g_show_dataset_popup_callback && !g_show_dataset_popup_callback.is_none()) {
                        nb::gil_scoped_acquire guard;
                        try {
                            g_show_dataset_popup_callback(lfs::core::path_to_utf8(e.dataset_path));
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
                            g_show_resume_popup_callback(lfs::core::path_to_utf8(e.checkpoint_path));
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
            "on_open_camera_preview",
            [](nb::object callback) {
                g_open_camera_preview_callback = callback;
                lfs::core::events::cmd::OpenCameraPreview::when([](const auto& e) {
                    if (g_open_camera_preview_callback && !g_open_camera_preview_callback.is_none()) {
                        nb::gil_scoped_acquire guard;
                        try {
                            g_open_camera_preview_callback(e.cam_id);
                        } catch (const std::exception& ex) {
                            LOG_ERROR("OpenCameraPreview callback error: {}", ex.what());
                        }
                    }
                });
            },
            nb::arg("callback"),
            "Register callback for OpenCameraPreview event");

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
                case vis::ToolType::Align: return "builtin.align";
                default: return "";
                }
            },
            "Get the currently active tool id from C++ EditorContext");

        m.def(
            "is_tool_available", [](const std::string& id) -> bool {
                static const std::unordered_map<std::string, vis::ToolType> tool_map = {
                    {"builtin.select", vis::ToolType::Selection},
                    {"builtin.translate", vis::ToolType::Translate},
                    {"builtin.rotate", vis::ToolType::Rotate},
                    {"builtin.scale", vis::ToolType::Scale},
                    {"builtin.mirror", vis::ToolType::Mirror},
                    {"builtin.align", vis::ToolType::Align},
                };
                const auto it = tool_map.find(id);
                if (it == tool_map.end()) {
                    return false;
                }
                const auto* const editor = get_editor_context();
                return editor && editor->isToolAvailable(it->second);
            },
            nb::arg("id"), "Check whether a builtin tool is currently available");

        m.def(
            "set_active_tool", [](const std::string& id) {
                static const std::unordered_map<std::string, vis::ToolType> tool_map = {
                    {"builtin.select", vis::ToolType::Selection},
                    {"builtin.translate", vis::ToolType::Translate},
                    {"builtin.rotate", vis::ToolType::Rotate},
                    {"builtin.scale", vis::ToolType::Scale},
                    {"builtin.mirror", vis::ToolType::Mirror},
                    {"builtin.align", vis::ToolType::Align},
                };
                auto it = tool_map.find(id);
                if (it != tool_map.end()) {
                    if (auto* const editor = get_editor_context()) {
                        editor->setActiveTool(it->second);
                    }
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
            "can_edit_gaussian_selection", []() -> bool {
                const auto* editor = get_editor_context();
                return editor && editor->canSelectGaussians();
            },
            "Return true when Gaussian selection editing is available");

        m.def(
            "has_gaussian_selection", []() -> bool {
                const auto* sm = get_scene_manager();
                return sm && sm->getScene().hasSelection();
            },
            "Return true when any Gaussians are selected");

        m.def(
            "has_gaussian_clipboard", []() -> bool {
                const auto* sm = get_scene_manager();
                return sm && sm->hasGaussianClipboard();
            },
            "Return true when copied Gaussians are available for paste");

        m.def(
            "copy_gaussian_selection", []() {
                auto* editor = get_editor_context();
                auto* sm = get_scene_manager();
                if (!editor || !editor->canSelectGaussians() || !sm || !sm->getScene().hasSelection()) {
                    return;
                }
                sm->copySelectedGaussians();
            },
            "Copy selected Gaussians to the internal Gaussian clipboard");

        m.def(
            "cut_gaussian_selection", []() {
                auto* editor = get_editor_context();
                auto* sm = get_scene_manager();
                if (!editor || !editor->canSelectGaussians() || !sm || !sm->getScene().hasSelection()) {
                    return;
                }
                sm->cutSelectedGaussians();
            },
            "Cut selected Gaussians to the internal Gaussian clipboard");

        m.def(
            "paste_gaussian_selection", []() {
                auto* editor = get_editor_context();
                auto* sm = get_scene_manager();
                if (!editor || !editor->canSelectGaussians() || !sm || !sm->hasGaussianClipboard()) {
                    return;
                }
                sm->pasteSelectionFromClipboard();
            },
            "Paste copied Gaussians from the internal Gaussian clipboard");

        m.def(
            "invert_gaussian_selection", []() {
                const auto* editor = get_editor_context();
                if (!editor || !editor->canSelectGaussians()) {
                    return;
                }
                lfs::core::events::cmd::InvertSelection{}.emit();
            },
            "Invert the current Gaussian selection");

        m.def(
            "select_all_gaussians", []() {
                const auto* editor = get_editor_context();
                if (!editor || !editor->canSelectGaussians()) {
                    return;
                }
                lfs::core::events::cmd::SelectAll{}.emit();
            },
            "Select all editable Gaussians");

        m.def(
            "deselect_all_gaussians", []() {
                const auto* editor = get_editor_context();
                auto* sm = get_scene_manager();
                if (!editor || !editor->canSelectGaussians() || !sm || !sm->getScene().hasSelection()) {
                    return;
                }
                lfs::core::events::cmd::DeselectAll{}.emit();
            },
            "Deselect all selected Gaussians");

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
                    {"centers", static_cast<int>(lfs::vis::SelectionSubMode::Centers)},
                    {"rectangle", static_cast<int>(lfs::vis::SelectionSubMode::Rectangle)},
                    {"polygon", static_cast<int>(lfs::vis::SelectionSubMode::Polygon)},
                    {"lasso", static_cast<int>(lfs::vis::SelectionSubMode::Lasso)},
                    {"rings", static_cast<int>(lfs::vis::SelectionSubMode::Rings)},
                    {"color", static_cast<int>(lfs::vis::SelectionSubMode::Color)},
                    {"box", static_cast<int>(lfs::vis::SelectionSubMode::Box)},
                    {"sphere", static_cast<int>(lfs::vis::SelectionSubMode::Sphere)}};
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
            "open_camera_preview",
            [](int cam_uid) { lfs::core::events::cmd::OpenCameraPreview{.cam_id = cam_uid}.emit(); },
            nb::arg("cam_uid"), "Open the image preview panel for a camera UID");

        m.def(
            "toggle_gt_comparison",
            []() { lfs::core::events::cmd::ToggleGTComparison{}.emit(); },
            "Toggle ground-truth comparison split view");

        m.def(
            "is_gt_comparison_active",
            []() {
                auto* rm = lfs::python::get_rendering_manager();
                return rm && rm->isGTComparisonActive();
            },
            "Returns true if ground-truth comparison split view is currently enabled.");

        m.def(
            "get_gt_comparison_mode",
            []() -> const char* {
                auto* rm = lfs::python::get_rendering_manager();
                if (!rm)
                    return "rgb";
                switch (rm->getSettings().gt_comparison_mode) {
                case vis::GTComparisonMode::Normal: return "normal";
                case vis::GTComparisonMode::Depth: return "depth";
                case vis::GTComparisonMode::RGB:
                default: return "rgb";
                }
            },
            "Get ground-truth comparison mode: rgb, normal, or depth.");

        m.def(
            "set_gt_comparison_mode",
            [](const std::string& mode) {
                auto* rm = lfs::python::get_rendering_manager();
                if (!rm)
                    return;
                auto settings = rm->getSettings();
                if (mode == "rgb" || mode == "color" || mode == "image") {
                    settings.gt_comparison_mode = vis::GTComparisonMode::RGB;
                } else if (mode == "normal" || mode == "normals") {
                    settings.gt_comparison_mode = vis::GTComparisonMode::Normal;
                } else if (mode == "depth") {
                    settings.gt_comparison_mode = vis::GTComparisonMode::Depth;
                } else {
                    throw nb::value_error("GT comparison mode must be 'rgb', 'normal', or 'depth'");
                }
                rm->updateSettings(settings, vis::DirtyFlag::ALL);
            },
            nb::arg("mode"), "Set ground-truth comparison mode.");

        m.def(
            "cycle_gt_comparison_mode",
            []() -> const char* {
                auto* rm = lfs::python::get_rendering_manager();
                if (!rm)
                    return "rgb";
                auto settings = rm->getSettings();
                switch (settings.gt_comparison_mode) {
                case vis::GTComparisonMode::RGB:
                    settings.gt_comparison_mode = vis::GTComparisonMode::Normal;
                    break;
                case vis::GTComparisonMode::Normal:
                    settings.gt_comparison_mode = vis::GTComparisonMode::Depth;
                    break;
                case vis::GTComparisonMode::Depth:
                default:
                    settings.gt_comparison_mode = vis::GTComparisonMode::RGB;
                    break;
                }
                rm->updateSettings(settings, vis::DirtyFlag::ALL);
                switch (settings.gt_comparison_mode) {
                case vis::GTComparisonMode::Normal: return "normal";
                case vis::GTComparisonMode::Depth: return "depth";
                case vis::GTComparisonMode::RGB:
                default: return "rgb";
                }
            },
            "Cycle ground-truth comparison mode: rgb -> normal -> depth -> rgb.");

        m.def(
            "reveal_in_file_manager",
            [](const std::string& utf8_path) {
                return lfs::core::reveal_in_file_manager(lfs::core::utf8_to_path(utf8_path));
            },
            nb::arg("path"),
            "Reveal a file or directory in the OS file manager. Returns true on success.");

        m.def(
            "apply_cropbox",
            []() { lfs::core::events::cmd::ApplyCropBox{}.emit(); },
            "Apply the selected cropbox");

        m.def(
            "set_crop_tool_shape",
            [](const std::string& shape) {
                if (auto* const gui = lfs::python::get_gui_manager()) {
                    gui->gizmo().setCropToolShape(shape);
                }
            },
            nb::arg("shape"),
            "Set the active crop tool shape: box or ellipsoid");

        m.def(
            "get_crop_tool_shape",
            []() -> std::string {
                if (auto* const gui = lfs::python::get_gui_manager()) {
                    return gui->gizmo().cropToolShape();
                }
                return "box";
            },
            "Get the active crop tool shape");

        m.def(
            "set_crop_tool_operation",
            [](const std::string& operation) {
                if (auto* const gui = lfs::python::get_gui_manager()) {
                    gui->gizmo().setCropToolOperation(operation);
                }
            },
            nb::arg("operation"),
            "Set the active crop or selection-volume gizmo operation: translate, rotate, or scale");

        m.def(
            "get_crop_tool_operation",
            []() -> std::string {
                if (auto* const gui = lfs::python::get_gui_manager()) {
                    return gui->gizmo().cropToolOperation();
                }
                return "translate";
            },
            "Get the active crop or selection-volume gizmo operation");

        m.def(
            "apply_crop_tool",
            []() {
                if (auto* const gui = lfs::python::get_gui_manager()) {
                    if (gui->gizmo().cropToolShape() == "ellipsoid") {
                        lfs::core::events::cmd::ApplyEllipsoid{}.emit();
                    } else {
                        lfs::core::events::cmd::ApplyCropBox{}.emit();
                    }
                }
            },
            "Apply the active crop tool primitive through the node-backed crop command path");

        m.def(
            "fit_crop_tool",
            [](bool use_percentile) {
                if (auto* const gui = lfs::python::get_gui_manager()) {
                    if (gui->gizmo().cropToolShape() == "ellipsoid") {
                        lfs::core::events::cmd::FitEllipsoidToScene{.use_percentile = use_percentile}.emit();
                    } else {
                        lfs::core::events::cmd::FitCropBoxToScene{.use_percentile = use_percentile}.emit();
                    }
                }
            },
            nb::arg("use_percentile") = false,
            "Fit the active crop tool primitive through the node-backed crop command path");

        m.def(
            "reset_crop_tool",
            []() {
                if (auto* const gui = lfs::python::get_gui_manager()) {
                    if (gui->gizmo().cropToolShape() == "ellipsoid") {
                        lfs::core::events::cmd::ResetEllipsoid{}.emit();
                    } else {
                        lfs::core::events::cmd::ResetCropBox{}.emit();
                    }
                }
            },
            "Reset the active crop tool primitive through the node-backed crop command path");

        m.def(
            "delete_crop_tool_volume",
            []() {
                if (auto* const gui = lfs::python::get_gui_manager()) {
                    gui->gizmo().deleteActiveCropToolVolume();
                }
            },
            "Delete the active crop tool primitive and leave the crop tool");

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

        m.def(
            "save_node_to_disk",
            [](const std::string& node_name) {
                lfs::core::Scene* scene = get_application_scene();
                if (!scene)
                    scene = get_scene_for_python();
                if (!scene) {
                    LOG_WARN("save_node_to_disk: no scene available");
                    return;
                }

                const auto* node = scene->getNode(node_name);
                if (!node) {
                    LOG_WARN("save_node_to_disk: node not found: '{}'", node_name);
                    return;
                }

                std::string default_name = node_name;
                if (default_name.empty())
                    default_name = "scene_node";

                const auto path = lfs::vis::gui::SavePlyFileDialog(default_name);
                if (path.empty())
                    return;

                const lfs::io::PlySaveOptions options{
                    .output_path = path,
                    .binary = true,
                    .async = false};

                lfs::io::Result<void> result = std::unexpected(
                    lfs::io::Error{lfs::io::ErrorCode::INTERNAL_ERROR, "uninitialized"});
                switch (node->type) {
                case lfs::core::NodeType::POINTCLOUD: {
                    if (!node->point_cloud || node->point_cloud->size() <= 0) {
                        LOG_WARN("save_node_to_disk: point cloud '{}' has no data", node_name);
                        return;
                    }
                    result = lfs::io::save_ply(*node->point_cloud, options);
                    break;
                }
                case lfs::core::NodeType::SPLAT: {
                    if (!node->model || node->model->size() <= 0) {
                        LOG_WARN("save_node_to_disk: splat '{}' has no data", node_name);
                        return;
                    }
                    result = lfs::io::save_ply(*node->model, options);
                    break;
                }
                default:
                    LOG_WARN("save_node_to_disk: unsupported node type for '{}': {}", node_name, static_cast<int>(node->type));
                    return;
                }

                if (!result) {
                    LOG_ERROR("Failed to save '{}' to {}: {}",
                              node_name,
                              lfs::core::path_to_utf8(path),
                              result.error().message);
                } else {
                    LOG_INFO("Saved '{}' to {}", node_name, lfs::core::path_to_utf8(path));
                }
            },
            nb::arg("node_name"),
            "Save a SPLAT or POINTCLOUD node to disk as a PLY file. Opens a file dialog; does nothing if cancelled.");

        // Image texture APIs for Python image preview
        m.def(
            "load_image_texture",
            [](const std::string& path) -> nb::tuple {
                try {
                    auto [data, w, h, channels] = lfs::core::load_image(lfs::core::utf8_to_path(path), -1, -1);
                    if (!data)
                        return nb::make_tuple(0, 0, 0);

                    auto [tex_id, width, height] = create_ui_texture_from_data(data, w, h, channels);
                    lfs::core::free_image(data);
                    return nb::make_tuple(tex_id, width, height);
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to load image texture {}: {}", path, e.what());
                    return nb::make_tuple(0, 0, 0);
                }
            },
            nb::arg("path"), "Load image as UI texture, returns (texture_id, width, height)");

        m.def(
            "load_thumbnail",
            [](const std::string& path, int max_size) -> nb::tuple {
                try {
                    auto [data, w, h, channels] = lfs::core::load_image(lfs::core::utf8_to_path(path), -1, max_size);
                    if (!data)
                        return nb::make_tuple(0, 0, 0);

                    auto [tex_id, width, height] = create_ui_texture_from_data(data, w, h, channels);
                    lfs::core::free_image(data);
                    return nb::make_tuple(tex_id, width, height);
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to load thumbnail {}: {}", path, e.what());
                    return nb::make_tuple(0, 0, 0);
                }
            },
            nb::arg("path"), nb::arg("max_size"), "Load downscaled image as UI texture, returns (texture_id, width, height)");

        m.def(
            "release_texture",
            [](const uint64_t tex_id) {
                if (tex_id > 0)
                    lfs::python::delete_ui_texture(tex_id);
            },
            nb::arg("texture_id"), "Release a UI texture");

        m.def(
            "get_image_info",
            [](const std::string& path) -> nb::tuple {
                auto [w, h, c] = lfs::core::get_image_info(lfs::core::utf8_to_path(path));
                return nb::make_tuple(w, h, c);
            },
            nb::arg("path"), "Get image dimensions without loading pixel data, returns (width, height, channels)");

        m.def(
            "sample_image_color",
            [](const std::string& path, int x, int y, int radius) -> nb::tuple {
                try {
                    auto [data, w, h, channels] = lfs::core::load_image(lfs::core::utf8_to_path(path), -1, -1);
                    if (!data)
                        return nb::make_tuple(0.0f, 0.0f, 0.0f);

                    const int x0 = std::max(0, x - radius);
                    const int y0 = std::max(0, y - radius);
                    const int x1 = std::min(w - 1, x + radius);
                    const int y1 = std::min(h - 1, y + radius);

                    double r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;
                    int count = 0;
                    const int ch = std::min(channels, 3);
                    for (int py = y0; py <= y1; ++py) {
                        for (int px = x0; px <= x1; ++px) {
                            const unsigned char* pixel = data + (static_cast<size_t>(py) * w + px) * channels;
                            r_sum += pixel[0];
                            g_sum += (ch > 1) ? pixel[1] : pixel[0];
                            b_sum += (ch > 2) ? pixel[2] : pixel[0];
                            ++count;
                        }
                    }

                    lfs::core::free_image(data);

                    if (count == 0)
                        return nb::make_tuple(0.0f, 0.0f, 0.0f);

                    return nb::make_tuple(
                        static_cast<float>(r_sum / (count * 255.0)),
                        static_cast<float>(g_sum / (count * 255.0)),
                        static_cast<float>(b_sum / (count * 255.0)));
                } catch (const std::exception& e) {
                    LOG_WARN("sample_image_color failed for {}: {}", path, e.what());
                    return nb::make_tuple(0.0f, 0.0f, 0.0f);
                }
            },
            nb::arg("path"), nb::arg("x"), nb::arg("y"), nb::arg("radius") = 10,
            "Sample average color around pixel (x, y) within given radius, returns (r, g, b) in 0..1");

        m.def(
            "preload_image_async",
            [](const std::string& path) {
                std::lock_guard lock(g_preload_mutex);
                if (g_preload_cache.contains(path))
                    return;

                auto entry = std::make_unique<PreloadEntry>();
                entry->future = std::async(std::launch::async, [path_copy = path]() {
                    return lfs::core::load_image(lfs::core::utf8_to_path(path_copy), -1, -1);
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

                auto [tex_id, width, height] = create_ui_texture_from_data(data, w, h, channels);
                lfs::core::free_image(data);
                return nb::make_tuple(tex_id, width, height);
            },
            nb::arg("path"), "Get preloaded image as UI texture, returns (texture_id, width, height)");

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
            [](const std::string&) {},
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

        // Sequencer UI state for Python access
        nb::class_<SequencerUIStateData>(m, "SequencerUIState")
            .def_rw("show_camera_path", &SequencerUIStateData::show_camera_path, "Whether camera path is displayed in viewport")
            .def_rw("snap_to_grid", &SequencerUIStateData::snap_to_grid, "Whether keyframe snapping is enabled")
            .def_rw("snap_interval", &SequencerUIStateData::snap_interval, "Snap grid interval in frames")
            .def_rw("playback_speed", &SequencerUIStateData::playback_speed, "Playback speed multiplier")
            .def_rw("follow_playback", &SequencerUIStateData::follow_playback, "Whether viewport follows playback position")
            .def_rw("show_pip_preview", &SequencerUIStateData::show_pip_preview, "Whether PiP preview window is shown")
            .def_rw("pip_preview_scale", &SequencerUIStateData::pip_preview_scale, "Picture-in-picture preview scale factor")
            .def_rw("show_film_strip", &SequencerUIStateData::show_film_strip, "Whether film strip thumbnails are shown above sequencer")
            .def_rw("sequence_fps", &SequencerUIStateData::sequence_fps, "Playback FPS for loaded PLY sequences")
            .def_ro("selected_keyframe", &SequencerUIStateData::selected_keyframe);

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

        m.def(
            "go_to_keyframe",
            [](size_t index) { lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = index}.emit(); },
            nb::arg("index"),
            "Navigate viewport to keyframe camera pose");

        m.def(
            "select_keyframe",
            [](size_t index) { lfs::core::events::cmd::SequencerSelectKeyframe{.keyframe_index = index}.emit(); },
            nb::arg("index"),
            "Select keyframe in timeline");

        m.def(
            "delete_keyframe",
            [](size_t index) { lfs::core::events::cmd::SequencerDeleteKeyframe{.keyframe_index = index}.emit(); },
            nb::arg("index"),
            "Delete keyframe by index");

        m.def(
            "set_keyframe_easing",
            [](size_t index, int easing) {
                lfs::core::events::cmd::SequencerSetKeyframeEasing{.keyframe_index = index, .easing_type = easing}.emit();
            },
            nb::arg("index"), nb::arg("easing"),
            "Set easing type for keyframe (0=Linear, 1=EaseIn, 2=EaseOut, 3=EaseInOut)");

        // Section drawing wrappers - callable from Python panel draw()
        // These use callbacks to keep py_ui.cpp independent of visualizer internals
        m.def("draw_tools_section", &draw_tools_section,
              "Draw tools section (C++ implementation)");

        m.def("draw_console_button", &draw_console_button,
              "Draw system console button (C++ implementation)");

        m.def("toggle_system_console", &toggle_system_console,
              "Toggle system console visibility");

        m.def(
            "is_windows_platform", []() -> bool {
#ifdef WIN32
                return true;
#else
                return false;
#endif
            },
            "Returns true on Windows");

        m.def(
            "register_file_associations", []() -> bool {
                return lfs::vis::gui::registerFileAssociations();
            },
            "Register LichtFeld Studio as a supported handler for .ply, .sog, .spz, .rad, .usd, .usda, .usdc, .usdz files (Windows only)");

        m.def(
            "open_file_association_settings", []() -> bool {
                return lfs::vis::gui::openFileAssociationSettings();
            },
            "Open the Windows Default Apps UI for LichtFeld Studio file associations (Windows only)");

        m.def(
            "unregister_file_associations", []() -> bool {
                return lfs::vis::gui::unregisterFileAssociations();
            },
            "Remove LichtFeld Studio file associations for .ply, .sog, .spz, .rad, .usd, .usda, .usdc, .usdz (Windows only)");

        m.def(
            "are_file_associations_registered", []() -> bool {
                return lfs::vis::gui::areFileAssociationsRegistered();
            },
            "Check if LichtFeld Studio is the default handler for .ply, .sog, .spz, .rad, .usd, .usda, .usdc, .usdz (Windows only)");

        m.def("get_pivot_mode", &get_pivot_mode, "Get pivot mode (0=Origin, 1=Bounds)");

        m.def("set_pivot_mode", &set_pivot_mode, nb::arg("mode"), "Set pivot mode (0=Origin, 1=Bounds)");

        m.def("get_transform_space", &get_transform_space, "Get transform space (0=Local, 1=World)");

        m.def("set_transform_space", &set_transform_space, nb::arg("space"), "Set transform space (0=Local, 1=World)");

        m.def("get_multi_transform_mode", &get_multi_transform_mode, "Get multi-transform mode (0=Group, 1=Individual)");

        m.def("set_multi_transform_mode", &set_multi_transform_mode, nb::arg("mode"), "Set multi-transform mode (0=Group, 1=Individual)");

        // Thumbnail system (for Getting Started window)
        m.def("request_thumbnail", &request_thumbnail, nb::arg("video_id"),
              "Request download of a YouTube thumbnail for the given video ID");
        m.def("process_thumbnails", &process_thumbnails,
              "Process pending thumbnail downloads (call every frame)");
        m.def("is_thumbnail_ready", &is_thumbnail_ready, nb::arg("video_id"),
              "Check if a thumbnail is ready to be displayed");
        m.def("get_thumbnail_texture", &get_thumbnail_texture, nb::arg("video_id"),
              "Get the UI texture ID for a downloaded thumbnail (0 if not ready)");

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

        m.def("free_plugin_textures", &free_plugin_textures, nb::arg("plugin_name"),
              "Free all dynamic textures associated with a plugin");

        // Asset Manager save callback
        m.def(
            "set_save_asset_callback", [](nb::callable save_cb) {
                  g_save_asset_callback = std::move(save_cb);
                  set_save_asset_callback(
                      [](const char* node_name) {
                          if (g_save_asset_callback) {
                              try {
                                  nb::gil_scoped_acquire gil;
                                  g_save_asset_callback(node_name);
                              } catch (const std::exception& e) {
                                  LOG_ERROR("Save asset callback failed: {}", e.what());
                              }
                          }
                      }); }, nb::arg("save_cb"), "Set callback for Save Asset operation from scene graph");

        nb::class_<PyDynamicTexture>(m, "DynamicTexture")
            .def(nb::init<>())
            .def(
                "__init__", [](PyDynamicTexture* self, PyTensor& tensor, const std::string& plugin_name) {
                    new (self) PyDynamicTexture(plugin_name);
                    self->update(tensor);
                },
                nb::arg("tensor"), nb::arg("plugin_name") = "")
            .def("update", &PyDynamicTexture::update, nb::arg("tensor"))
            .def("destroy", &PyDynamicTexture::destroy)
            .def_prop_ro("id", &PyDynamicTexture::texture_id)
            .def_prop_ro("width", &PyDynamicTexture::width)
            .def_prop_ro("height", &PyDynamicTexture::height)
            .def_prop_ro("valid", &PyDynamicTexture::valid)
            .def_prop_ro("uv1", &PyDynamicTexture::uv1);

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
                if (vis::setThemeByName(name)) {
                    vis::saveThemePreferenceName(name);
                }
            },
            nb::arg("name"), "Set theme by stable theme id");

        m.def(
            "get_theme",
            []() -> std::string { return vis::currentThemeId(); },
            "Get current stable theme id");

        m.def(
            "themes",
            []() {
                nb::list themes;
                vis::visitThemePresetInfos([&themes](const vis::ThemePresetInfo& info) {
                    nb::dict item;
                    item["id"] = info.id;
                    item["name"] = info.name;
                    item["label_key"] = info.label_key;
                    item["mode"] = info.mode;
                    item["order"] = info.order;
                    themes.append(item);
                });
                return themes;
            },
            "Get available theme presets with stable ids and UI metadata");

        m.def(
            "set_ui_scale",
            [](float scale) {
                vis::saveUiScalePreference(scale);
                core::events::internal::UiScaleChangeRequested{scale}.emit();
            },
            nb::arg("scale"), "Set UI scale (0.0 = auto from OS, or 1.0-4.0)");

        m.def(
            "get_ui_scale",
            []() -> float { return python::get_shared_dpi_scale(); },
            "Get current UI scale factor");

        m.def(
            "get_ui_scale_preference",
            []() -> float { return vis::loadUiScalePreference(); },
            "Get saved UI scale preference (0.0 = auto)");

        m.def(
            "set_clipboard_text",
            [](const std::string& text) { SDL_SetClipboardText(text.c_str()); },
            nb::arg("text"), "Copy text to the system clipboard");

        m.def(
            "has_clipboard_image",
            []() { return clipboard_has_image(); },
            "Return True if the system clipboard holds an image");

        m.def(
            "get_clipboard_image_texture",
            []() -> nb::tuple {
                auto [data, w, h, channels] = decode_clipboard_image();
                if (!data)
                    return nb::make_tuple(0, 0, 0);
                const auto [tex_id, width, height] = create_ui_texture_from_data(data, w, h, channels);
                lfs::core::free_image(data);
                return nb::make_tuple(tex_id, width, height);
            },
            "Read an image from the clipboard as a UI texture, returns (texture_id, width, height)");

        m.def(
            "save_clipboard_image",
            [](const std::string& path) {
                const auto image = decode_clipboard_image();
                if (!std::get<0>(image))
                    return false;
                bool saved = false;
                try {
                    saved = lfs::core::save_img_data(lfs::core::utf8_to_path(path), image);
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to save clipboard image to {}: {}", path, e.what());
                }
                lfs::core::free_image(std::get<0>(image));
                return saved;
            },
            nb::arg("path"), "Decode the clipboard image and write it to path; returns success");

        m.def(
            "set_mouse_cursor_hand",
            []() {},
            "Set mouse cursor to hand pointer for this frame");

        // Language control (for Python-driven Edit menu)
        m.def(
            "set_language",
            [](const std::string& lang_code) {
                if (lfs::event::LocalizationManager::getInstance().setLanguage(lang_code)) {
                    lfs::vis::publish_language_generation();
                }
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
            "get_time",
            []() {
                static const auto start = std::chrono::steady_clock::now();
                const auto now = std::chrono::steady_clock::now();
                return std::chrono::duration<double>(now - start).count();
            },
            "Get time in seconds since application start");

        // Register all Python callbacks via consolidated bridge
        PyBridge bridge;
        bridge.begin_ui_frame = []() { begin_keyboard_ui_frame(); };
        bridge.prepare_ui = []() {};
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
        bridge.collect_menu_content = [](const char* idname, MenuItemVisitor visitor, void* ctx) {
            if (!idname)
                return;
            auto content = PyMenuRegistry::instance().collect_menu_content(idname);
            for (const auto& item : content.items) {
                MenuItemInfo info;
                info.type = static_cast<int>(item.type);
                info.label = item.label.c_str();
                info.operator_id = item.operator_id.c_str();
                info.shortcut = item.shortcut.c_str();
                info.enabled = item.enabled;
                info.selected = item.selected;
                info.callback_index = item.callback_index;
                visitor(&info, ctx);
            }
        };
        bridge.execute_menu_callback = [](const char* idname, int idx) {
            if (idname)
                PyMenuRegistry::instance().execute_menu_callback(idname, idx);
        };
        bridge.draw_modals = []() { PyModalRegistry::instance().draw_modals(); };
        bridge.has_modals = []() { return PyModalRegistry::instance().has_open_modals(); };

        if (const auto& enqueue_cb = get_modal_enqueue_callback())
            PyModalRegistry::instance().set_enqueue_callback(enqueue_cb);
        bridge.has_toolbar = []() { return true; }; // Always true - Python ToolRegistry has builtin tools
        bridge.shutdown_ui_resources = []() { shutdown_dynamic_textures(); };
        bridge.cleanup = []() {
            PyPanelRegistry::instance().unregister_all();
            PyUIHookRegistry::instance().clear_all();
            PyMenuRegistry::instance().unregister_all();
            PyViewportDrawRegistry::instance().clear_all();
            PyGizmoRegistry::instance().unregister_all();
            PyUIListRegistry::instance().unregister_all();
            vis::op::operators().unregisterAllPython();
            shutdown_store_bridge();
            shutdown_signal_bridge();
            g_cancel_operator_py_callback = nb::callable();
            g_modal_event_py_callback = nb::callable();
            g_popup_draw_callback = nb::object();
            g_show_dataset_popup_callback = nb::object();
            g_show_resume_popup_callback = nb::object();
            g_request_exit_callback = nb::object();
            g_open_camera_preview_callback = nb::object();
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
        m.def("get_selection_submode", &get_selection_submode,
              "Get current selection sub-mode (0=Centers, 1=Rectangle, 2=Polygon, 3=Lasso, 4=Rings, 5=Color, 6=Box, 7=Sphere)");

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
        key.attr("SPACE") = lfs::vis::input::KEY_SPACE;
        key.attr("APOSTROPHE") = lfs::vis::input::KEY_APOSTROPHE;
        key.attr("COMMA") = lfs::vis::input::KEY_COMMA;
        key.attr("MINUS") = lfs::vis::input::KEY_MINUS;
        key.attr("PERIOD") = lfs::vis::input::KEY_PERIOD;
        key.attr("SLASH") = lfs::vis::input::KEY_SLASH;
        key.attr("NUM_0") = lfs::vis::input::KEY_0;
        key.attr("NUM_1") = lfs::vis::input::KEY_1;
        key.attr("NUM_2") = lfs::vis::input::KEY_2;
        key.attr("NUM_3") = lfs::vis::input::KEY_3;
        key.attr("NUM_4") = lfs::vis::input::KEY_4;
        key.attr("NUM_5") = lfs::vis::input::KEY_5;
        key.attr("NUM_6") = lfs::vis::input::KEY_6;
        key.attr("NUM_7") = lfs::vis::input::KEY_7;
        key.attr("NUM_8") = lfs::vis::input::KEY_8;
        key.attr("NUM_9") = lfs::vis::input::KEY_9;
        key.attr("A") = lfs::vis::input::KEY_A;
        key.attr("B") = lfs::vis::input::KEY_B;
        key.attr("C") = lfs::vis::input::KEY_C;
        key.attr("D") = lfs::vis::input::KEY_D;
        key.attr("E") = lfs::vis::input::KEY_E;
        key.attr("F") = lfs::vis::input::KEY_F;
        key.attr("G") = lfs::vis::input::KEY_G;
        key.attr("H") = lfs::vis::input::KEY_H;
        key.attr("I") = lfs::vis::input::KEY_I;
        key.attr("J") = lfs::vis::input::KEY_J;
        key.attr("K") = lfs::vis::input::KEY_K;
        key.attr("L") = lfs::vis::input::KEY_L;
        key.attr("M") = lfs::vis::input::KEY_M;
        key.attr("N") = lfs::vis::input::KEY_N;
        key.attr("O") = lfs::vis::input::KEY_O;
        key.attr("P") = lfs::vis::input::KEY_P;
        key.attr("Q") = lfs::vis::input::KEY_Q;
        key.attr("R") = lfs::vis::input::KEY_R;
        key.attr("S") = lfs::vis::input::KEY_S;
        key.attr("T") = lfs::vis::input::KEY_T;
        key.attr("U") = lfs::vis::input::KEY_U;
        key.attr("V") = lfs::vis::input::KEY_V;
        key.attr("W") = lfs::vis::input::KEY_W;
        key.attr("X") = lfs::vis::input::KEY_X;
        key.attr("Y") = lfs::vis::input::KEY_Y;
        key.attr("Z") = lfs::vis::input::KEY_Z;
        key.attr("ESCAPE") = lfs::vis::input::KEY_ESCAPE;
        key.attr("ENTER") = lfs::vis::input::KEY_ENTER;
        key.attr("TAB") = lfs::vis::input::KEY_TAB;
        key.attr("BACKSPACE") = lfs::vis::input::KEY_BACKSPACE;
        key.attr("INSERT") = lfs::vis::input::KEY_INSERT;
        key.attr("DELETE") = lfs::vis::input::KEY_DELETE;
        key.attr("RIGHT") = lfs::vis::input::KEY_RIGHT;
        key.attr("LEFT") = lfs::vis::input::KEY_LEFT;
        key.attr("DOWN") = lfs::vis::input::KEY_DOWN;
        key.attr("UP") = lfs::vis::input::KEY_UP;
        key.attr("F1") = lfs::vis::input::KEY_F1;
        key.attr("F2") = lfs::vis::input::KEY_F2;
        key.attr("F3") = lfs::vis::input::KEY_F3;
        key.attr("F4") = lfs::vis::input::KEY_F4;
        key.attr("F5") = lfs::vis::input::KEY_F5;
        key.attr("F6") = lfs::vis::input::KEY_F6;
        key.attr("F7") = lfs::vis::input::KEY_F7;
        key.attr("F8") = lfs::vis::input::KEY_F8;
        key.attr("F9") = lfs::vis::input::KEY_F9;
        key.attr("F10") = lfs::vis::input::KEY_F10;
        key.attr("F11") = lfs::vis::input::KEY_F11;
        key.attr("F12") = lfs::vis::input::KEY_F12;
        key.attr("KP_0") = lfs::vis::input::KEY_KP_0;
        key.attr("KP_1") = lfs::vis::input::KEY_KP_1;
        key.attr("KP_2") = lfs::vis::input::KEY_KP_2;
        key.attr("KP_3") = lfs::vis::input::KEY_KP_3;
        key.attr("KP_4") = lfs::vis::input::KEY_KP_4;
        key.attr("KP_5") = lfs::vis::input::KEY_KP_5;
        key.attr("KP_6") = lfs::vis::input::KEY_KP_6;
        key.attr("KP_7") = lfs::vis::input::KEY_KP_7;
        key.attr("KP_8") = lfs::vis::input::KEY_KP_8;
        key.attr("KP_9") = lfs::vis::input::KEY_KP_9;
        key.attr("KP_DECIMAL") = lfs::vis::input::KEY_KP_DECIMAL;
        key.attr("KP_DIVIDE") = lfs::vis::input::KEY_KP_DIVIDE;
        key.attr("KP_MULTIPLY") = lfs::vis::input::KEY_KP_MULTIPLY;
        key.attr("KP_SUBTRACT") = lfs::vis::input::KEY_KP_SUBTRACT;
        key.attr("KP_ADD") = lfs::vis::input::KEY_KP_ADD;
        key.attr("KP_ENTER") = lfs::vis::input::KEY_KP_ENTER;
        key.attr("KP_EQUAL") = lfs::vis::input::KEY_KP_EQUAL;

        auto mouse = m.def_submodule("mouse", "Mouse buttons");
        mouse.attr("LEFT") = static_cast<int>(lfs::vis::input::AppMouseButton::LEFT);
        mouse.attr("RIGHT") = static_cast<int>(lfs::vis::input::AppMouseButton::RIGHT);
        mouse.attr("MIDDLE") = static_cast<int>(lfs::vis::input::AppMouseButton::MIDDLE);

        auto mod = m.def_submodule("mod", "Modifier keys");
        mod.attr("SHIFT") = lfs::vis::input::KEYMOD_SHIFT;
        mod.attr("CONTROL") = lfs::vis::input::KEYMOD_CTRL;
        mod.attr("ALT") = lfs::vis::input::KEYMOD_ALT;
        mod.attr("SUPER") = lfs::vis::input::KEYMOD_SUPER;

        auto action = m.def_submodule("action", "Action values");
        action.attr("PRESS") = lfs::vis::input::ACTION_PRESS;
        action.attr("RELEASE") = lfs::vis::input::ACTION_RELEASE;
        action.attr("REPEAT") = lfs::vis::input::ACTION_REPEAT;

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
                case vis::SplitViewMode::IndependentDual: return "independent_dual";
                default: return "none";
                }
            },
            "Get split view mode (none, gt_comparison, ply_comparison, independent_dual)");

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

        set_python_document_hook_invoker([](const char* panel, const char* section,
                                            void* document, bool prepend) -> bool {
            auto& registry = PyUIHookRegistry::instance();
            const auto position = prepend ? PyHookPosition::Prepend : PyHookPosition::Append;
            if (registry.has_hooks(panel, section, position)) {
                return registry.invoke_document(
                    panel, section, static_cast<Rml::ElementDocument*>(document), position);
            }
            return false;
        });

        set_python_hook_checker([](const char* panel, const char* section, const bool prepend) -> bool {
            return PyUIHookRegistry::instance().has_hooks(
                panel, section, prepend ? PyHookPosition::Prepend : PyHookPosition::Append);
        });

        set_popup_draw_callback([]() {
            auto& registry = PyUIHookRegistry::instance();
            if (registry.has_hooks(LEGACY_POPUP_PANEL_STR, LEGACY_POPUP_SECTION_STR)) {
                registry.invoke(LEGACY_POPUP_PANEL_STR,
                                LEGACY_POPUP_SECTION_STR,
                                PyHookPosition::Append);
            }
        });
        set_popup_has_callback([]() {
            return PyUIHookRegistry::instance().has_hooks(
                LEGACY_POPUP_PANEL_STR,
                LEGACY_POPUP_SECTION_STR,
                PyHookPosition::Append);
        });
    }

    void register_class_api(nb::module_& m) {
        m.def(
            "register_class",
            [](nb::object cls) {
                nb::module_ builtins = nb::module_::import_("builtins");
                auto issubclass = builtins.attr("issubclass");

                nb::object Panel_type = nb::module_::import_("lichtfeld").attr("ui").attr("Panel");
                nb::module_ types_module = nb::module_::import_("lfs_plugins.types");
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
                    throw nb::type_error(
                        "register_class: cls must be a subclass of Panel, Operator, or Menu");
                }
            },
            nb::arg("cls"), "Register a class (Panel, Operator, or Menu)");

        m.def(
            "unregister_class",
            [](nb::object cls) {
                nb::module_ builtins = nb::module_::import_("builtins");
                auto issubclass = builtins.attr("issubclass");

                nb::object Panel_type = nb::module_::import_("lichtfeld").attr("ui").attr("Panel");
                nb::module_ types_module = nb::module_::import_("lfs_plugins.types");
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
