/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_runtime.hpp"
#include "core/modal_event.hpp"
#include "core/operator_callbacks.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <mutex>
#include <unordered_set>

#include <Python.h>

namespace lfs::python {

    namespace {
        PyBridge g_bridge{};

        ExportCallback g_export_callback = nullptr;
        DrawPopupsCallback g_popup_draw_callback = nullptr;
        EnsureInitializedCallback g_ensure_initialized_callback = nullptr;

        // Exit popup state for window close callback (thread-safe)
        std::atomic<bool> g_exit_popup_open{false};

        // Sequencer callbacks
        IsSequencerVisibleCallback g_is_sequencer_visible_cb = nullptr;
        SetSequencerVisibleCallback g_set_sequencer_visible_cb = nullptr;

        // Overlay state callbacks
        IsDragHoveringCallback g_is_drag_hovering_cb = nullptr;
        IsStartupVisibleCallback g_is_startup_visible_cb = nullptr;
        GetExportStateCallback g_get_export_state_cb = nullptr;
        CancelExportCallback g_cancel_export_cb = nullptr;
        GetImportStateCallback g_get_import_state_cb = nullptr;
        DismissImportCallback g_dismiss_import_cb = nullptr;
        GetVideoExportStateCallback g_get_video_export_state_cb = nullptr;
        CancelVideoExportCallback g_cancel_video_export_cb = nullptr;

        // Sequencer timeline callbacks
        HasKeyframesCallback g_has_keyframes_cb = nullptr;
        SaveCameraPathCallback g_save_camera_path_cb = nullptr;
        LoadCameraPathCallback g_load_camera_path_cb = nullptr;
        ClearKeyframesCallback g_clear_keyframes_cb = nullptr;
        SetPlaybackSpeedCallback g_set_playback_speed_cb = nullptr;

        // Menu bar UI callbacks
        ShowInputSettingsCallback g_show_input_settings_cb = nullptr;
        ShowPythonConsoleCallback g_show_python_console_cb = nullptr;

        // Section drawing callbacks
        SectionDrawCallbacks g_section_draw_callbacks;
        GetSequencerUIStateCallback g_get_sequencer_ui_state_cb = nullptr;

        // Pivot mode callbacks
        GetPivotModeCallback g_get_pivot_mode_cb = nullptr;
        SetPivotModeCallback g_set_pivot_mode_cb = nullptr;

        // Transform space callbacks
        GetTransformSpaceCallback g_get_transform_space_cb = nullptr;
        SetTransformSpaceCallback g_set_transform_space_cb = nullptr;

        // Thumbnail callbacks
        RequestThumbnailCallback g_request_thumbnail_cb = nullptr;
        ProcessThumbnailsCallback g_process_thumbnails_cb = nullptr;
        IsThumbnailReadyCallback g_is_thumbnail_ready_cb = nullptr;
        GetThumbnailTextureCallback g_get_thumbnail_texture_cb = nullptr;

        // Selection sub-mode (shared between C++ toolbar and Python operator)
        std::atomic<int> g_selection_submode{0};

        // Note: Operator callbacks (cancel, invoke, modal event) are now stored in EditorContext
        // to avoid static variable duplication across shared libraries (RTLD_LOCAL issue)

        // Operation context (short-lived, per-call)
        // Thread-safety: Set/cleared only via SceneContextGuard RAII, accessed from single thread
        vis::Scene* g_scene_for_python = nullptr;

        ApplicationSceneContext g_app_scene_context;
        std::atomic<vis::TrainerManager*> g_trainer_manager{nullptr};
        std::atomic<vis::ParameterManager*> g_parameter_manager{nullptr};
        std::atomic<vis::RenderingManager*> g_rendering_manager{nullptr};
        // EditorContext pointer for public API (type-erased to avoid CUDA deps in this file)
        std::atomic<void*> g_editor_context{nullptr};
        // Same pointer cast to IOperatorCallbacks* for callback dispatch (EditorContext implements this interface)
        std::atomic<lfs::core::IOperatorCallbacks*> g_operator_callbacks{nullptr};
        std::atomic<vis::gui::GuiManager*> g_gui_manager{nullptr};
        std::atomic<vis::SceneManager*> g_scene_manager{nullptr};
        std::atomic<vis::SelectionService*> g_selection_service{nullptr};
        std::atomic<vis::input::InputBindings*> g_keymap_bindings{nullptr};
        std::atomic<bool> g_gil_state_ready{false};

        // Main thread GIL state - stored here in the shared library
        // to ensure single copy across all modules
        std::atomic<PyThreadState*> g_main_thread_state{nullptr};

        // Initialization guards - must be in shared library
        std::once_flag g_py_init_once;
        std::once_flag g_redirect_once;
        std::atomic<bool> g_plugins_loaded{false};

        // ImGui state shared across DLL boundaries (set once during init, read from render thread)
        void* g_imgui_context{nullptr};
        void* g_imgui_alloc_fn{nullptr};
        void* g_imgui_free_fn{nullptr};
        void* g_imgui_alloc_user_data{nullptr};

        constexpr float DEFAULT_DPI_SCALE{1.0f};

        CreateTextureFn g_create_texture{nullptr};
        DeleteTextureFn g_delete_texture{nullptr};
        MaxTextureSizeFn g_max_texture_size_fn{nullptr};

        void* g_implot_context{nullptr};

        void* g_view_context_state{nullptr};
        float g_shared_dpi_scale{DEFAULT_DPI_SCALE};

        // Viewport bounds (set by gui_manager each frame)
        // Protected by mutex for multi-field atomicity
        struct ViewportBounds {
            std::mutex mutex;
            float x{0}, y{0}, w{0}, h{0};
            bool is_set{false};
        };
        ViewportBounds g_viewport;

        // Thread-local frame context for unified access
        thread_local PyContext g_frame_context;

        // Cached state updated via signal bridge
        std::atomic<bool> g_has_selection{false};
        std::atomic<bool> g_is_training{false};

        // Redraw request flag
        std::atomic<bool> g_redraw_requested{false};
    } // namespace

    // Bridge API
    void set_bridge(const PyBridge& b) { g_bridge = b; }
    const PyBridge& bridge() { return g_bridge; }

    // Keymap bindings
    void set_keymap_bindings(vis::input::InputBindings* bindings) { g_keymap_bindings.store(bindings); }
    vis::input::InputBindings* get_keymap_bindings() { return g_keymap_bindings.load(); }

    void set_context(const PyContext& ctx) {
        g_frame_context = ctx;

        // Derived state from signal bridge atomics
        g_frame_context.cached_has_selection = g_has_selection.load(std::memory_order_relaxed);
        g_frame_context.cached_is_training = g_is_training.load(std::memory_order_relaxed);

        // Override scene_generation from atomic state
        g_frame_context.scene_generation = g_app_scene_context.generation();

        // UI state from callbacks (fallback to stored values if no callback)
        if (g_get_pivot_mode_cb) {
            g_frame_context.pivot_mode = g_get_pivot_mode_cb();
        }
        if (g_get_transform_space_cb) {
            g_frame_context.transform_space = g_get_transform_space_cb();
        }
        g_frame_context.selection_submode = g_selection_submode.load();
    }

    const PyContext& context() { return g_frame_context; }

    // Redraw request mechanism
    void request_redraw() { g_redraw_requested.store(true, std::memory_order_release); }

    bool consume_redraw_request() {
        return g_redraw_requested.exchange(false, std::memory_order_acq_rel);
    }

    // Operation context (short-lived)
    void set_scene_for_python(vis::Scene* scene) { g_scene_for_python = scene; }
    vis::Scene* get_scene_for_python() { return g_scene_for_python; }

    void set_trainer_manager(vis::TrainerManager* tm) { g_trainer_manager.store(tm); }
    vis::TrainerManager* get_trainer_manager() { return g_trainer_manager.load(); }

    void set_parameter_manager(vis::ParameterManager* pm) { g_parameter_manager.store(pm); }
    vis::ParameterManager* get_parameter_manager() { return g_parameter_manager.load(); }

    void set_rendering_manager(vis::RenderingManager* rm) { g_rendering_manager.store(rm); }
    vis::RenderingManager* get_rendering_manager() { return g_rendering_manager.load(); }

    void set_editor_context(vis::EditorContext* ec) {
        g_editor_context.store(ec);
    }
    vis::EditorContext* get_editor_context() { return static_cast<vis::EditorContext*>(g_editor_context.load()); }

    void set_operator_callbacks(core::IOperatorCallbacks* callbacks) {
        g_operator_callbacks.store(callbacks);
    }

    void set_gui_manager(vis::gui::GuiManager* gm) { g_gui_manager.store(gm); }
    vis::gui::GuiManager* get_gui_manager() { return g_gui_manager.load(); }

    namespace {
        GetSelectedCameraUidCallback g_get_selected_camera_cb = nullptr;
        GetInvertMasksCallback g_get_invert_masks_cb = nullptr;
    } // namespace

    void set_selected_camera_callback(GetSelectedCameraUidCallback cb) {
        g_get_selected_camera_cb = cb;
    }

    void set_invert_masks_callback(GetInvertMasksCallback cb) {
        g_get_invert_masks_cb = cb;
    }

    int get_selected_camera_uid() {
        return g_get_selected_camera_cb ? g_get_selected_camera_cb() : -1;
    }

    bool get_invert_masks() {
        return g_get_invert_masks_cb ? g_get_invert_masks_cb() : false;
    }

    void set_sequencer_callbacks(IsSequencerVisibleCallback get_cb, SetSequencerVisibleCallback set_cb) {
        g_is_sequencer_visible_cb = get_cb;
        g_set_sequencer_visible_cb = set_cb;
    }

    bool is_sequencer_visible() {
        return g_is_sequencer_visible_cb ? g_is_sequencer_visible_cb() : false;
    }

    void set_sequencer_visible(bool visible) {
        if (g_set_sequencer_visible_cb)
            g_set_sequencer_visible_cb(visible);
    }

    void set_overlay_callbacks(IsDragHoveringCallback drag_cb,
                               IsStartupVisibleCallback startup_cb,
                               GetExportStateCallback export_cb,
                               CancelExportCallback cancel_cb,
                               GetImportStateCallback import_cb,
                               DismissImportCallback dismiss_import_cb,
                               GetVideoExportStateCallback video_cb,
                               CancelVideoExportCallback cancel_video_cb) {
        g_is_drag_hovering_cb = drag_cb;
        g_is_startup_visible_cb = startup_cb;
        g_get_export_state_cb = export_cb;
        g_cancel_export_cb = cancel_cb;
        g_get_import_state_cb = import_cb;
        g_dismiss_import_cb = dismiss_import_cb;
        g_get_video_export_state_cb = video_cb;
        g_cancel_video_export_cb = cancel_video_cb;
    }

    bool is_drag_hovering() {
        return g_is_drag_hovering_cb ? g_is_drag_hovering_cb() : false;
    }

    bool is_startup_visible() {
        return g_is_startup_visible_cb ? g_is_startup_visible_cb() : false;
    }

    OverlayExportState get_export_state() {
        return g_get_export_state_cb ? g_get_export_state_cb() : OverlayExportState{};
    }

    void cancel_export() {
        if (g_cancel_export_cb)
            g_cancel_export_cb();
    }

    OverlayImportState get_import_state() {
        return g_get_import_state_cb ? g_get_import_state_cb() : OverlayImportState{};
    }

    void dismiss_import() {
        if (g_dismiss_import_cb)
            g_dismiss_import_cb();
    }

    OverlayVideoExportState get_video_export_state() {
        return g_get_video_export_state_cb ? g_get_video_export_state_cb() : OverlayVideoExportState{};
    }

    void cancel_video_export() {
        if (g_cancel_video_export_cb)
            g_cancel_video_export_cb();
    }

    void set_sequencer_timeline_callbacks(
        HasKeyframesCallback has_keyframes_cb,
        SaveCameraPathCallback save_path_cb,
        LoadCameraPathCallback load_path_cb,
        ClearKeyframesCallback clear_cb,
        SetPlaybackSpeedCallback set_speed_cb) {
        g_has_keyframes_cb = has_keyframes_cb;
        g_save_camera_path_cb = save_path_cb;
        g_load_camera_path_cb = load_path_cb;
        g_clear_keyframes_cb = clear_cb;
        g_set_playback_speed_cb = set_speed_cb;
    }

    bool has_keyframes() {
        return g_has_keyframes_cb ? g_has_keyframes_cb() : false;
    }

    bool save_camera_path(const std::string& path) {
        return g_save_camera_path_cb ? g_save_camera_path_cb(path) : false;
    }

    bool load_camera_path(const std::string& path) {
        return g_load_camera_path_cb ? g_load_camera_path_cb(path) : false;
    }

    void clear_keyframes() {
        if (g_clear_keyframes_cb)
            g_clear_keyframes_cb();
    }

    void set_playback_speed(float speed) {
        if (g_set_playback_speed_cb)
            g_set_playback_speed_cb(speed);
    }

    void set_show_input_settings_callback(ShowInputSettingsCallback cb) {
        g_show_input_settings_cb = cb;
    }

    void set_show_python_console_callback(ShowPythonConsoleCallback cb) {
        g_show_python_console_cb = cb;
    }

    void show_input_settings() {
        if (g_show_input_settings_cb)
            g_show_input_settings_cb();
    }

    void show_python_console() {
        if (g_show_python_console_cb)
            g_show_python_console_cb();
    }

    void set_section_draw_callbacks(const SectionDrawCallbacks& callbacks) {
        g_section_draw_callbacks = callbacks;
    }

    void draw_tools_section() {
        if (g_section_draw_callbacks.draw_tools_section)
            g_section_draw_callbacks.draw_tools_section();
    }

    void draw_console_button() {
        if (g_section_draw_callbacks.draw_console_button)
            g_section_draw_callbacks.draw_console_button();
    }

    void set_sequencer_ui_state_callback(GetSequencerUIStateCallback cb) {
        g_get_sequencer_ui_state_cb = cb;
    }

    SequencerUIStateData* get_sequencer_ui_state() {
        return g_get_sequencer_ui_state_cb ? g_get_sequencer_ui_state_cb() : nullptr;
    }

    void set_pivot_mode_callbacks(GetPivotModeCallback get_cb, SetPivotModeCallback set_cb) {
        g_get_pivot_mode_cb = get_cb;
        g_set_pivot_mode_cb = set_cb;
    }

    int get_pivot_mode() {
        return g_get_pivot_mode_cb ? g_get_pivot_mode_cb() : 0;
    }

    void set_pivot_mode(int mode) {
        if (g_set_pivot_mode_cb)
            g_set_pivot_mode_cb(mode);
    }

    void set_transform_space_callbacks(GetTransformSpaceCallback get_cb, SetTransformSpaceCallback set_cb) {
        g_get_transform_space_cb = get_cb;
        g_set_transform_space_cb = set_cb;
    }

    int get_transform_space() {
        return g_get_transform_space_cb ? g_get_transform_space_cb() : 0;
    }

    void set_transform_space(int space) {
        if (g_set_transform_space_cb)
            g_set_transform_space_cb(space);
    }

    void set_scene_manager(vis::SceneManager* sm) { g_scene_manager.store(sm); }
    vis::SceneManager* get_scene_manager() { return g_scene_manager.load(); }

    void set_selection_service(vis::SelectionService* ss) { g_selection_service.store(ss); }
    vis::SelectionService* get_selection_service() { return g_selection_service.load(); }

    // Application context (long-lived)
    void ApplicationSceneContext::set(vis::Scene* scene) {
        scene_.store(scene);
        generation_.fetch_add(1);
    }

    vis::Scene* ApplicationSceneContext::get() const { return scene_.load(); }

    uint64_t ApplicationSceneContext::generation() const { return generation_.load(); }

    void set_application_scene(vis::Scene* scene) { g_app_scene_context.set(scene); }

    vis::Scene* get_application_scene() { return g_app_scene_context.get(); }

    uint64_t get_scene_generation() { return g_app_scene_context.generation(); }

    void set_gil_state_ready(const bool ready) { g_gil_state_ready.store(ready, std::memory_order_release); }
    bool is_gil_state_ready() { return g_gil_state_ready.load(std::memory_order_acquire); }

    void set_main_thread_state(void* state) {
        g_main_thread_state.store(static_cast<PyThreadState*>(state), std::memory_order_release);
    }

    void* get_main_thread_state() {
        return g_main_thread_state.load(std::memory_order_acquire);
    }

    void acquire_gil_main_thread() {
        PyThreadState* state = g_main_thread_state.exchange(nullptr, std::memory_order_acq_rel);
        if (state) {
            PyEval_RestoreThread(state);
        }
    }

    void release_gil_main_thread() {
        PyThreadState* state = PyEval_SaveThread();
        g_main_thread_state.store(state, std::memory_order_release);
    }

    // Initialization guards
    void call_once_py_init(std::function<void()> fn) {
        std::call_once(g_py_init_once, std::move(fn));
    }

    void call_once_redirect(std::function<void()> fn) {
        std::call_once(g_redirect_once, std::move(fn));
    }

    void mark_plugins_loaded() { g_plugins_loaded.store(true, std::memory_order_release); }
    bool are_plugins_loaded() { return g_plugins_loaded.load(std::memory_order_acquire); }

    void set_imgui_context(void* ctx) {
        g_imgui_context = ctx;
    }

    void* get_imgui_context() {
        return g_imgui_context;
    }

    void set_imgui_allocator_functions(void* alloc_func, void* free_func, void* user_data) {
        g_imgui_alloc_fn = alloc_func;
        g_imgui_free_fn = free_func;
        g_imgui_alloc_user_data = user_data;
    }

    void get_imgui_allocator_functions(void** alloc_func, void** free_func, void** user_data) {
        *alloc_func = g_imgui_alloc_fn;
        *free_func = g_imgui_free_fn;
        *user_data = g_imgui_alloc_user_data;
    }

    void set_gl_texture_service(const CreateTextureFn create, const DeleteTextureFn del, const MaxTextureSizeFn max_size) {
        assert(create && del && max_size);
        g_create_texture = create;
        g_delete_texture = del;
        g_max_texture_size_fn = max_size;
    }

    TextureResult create_gl_texture(const unsigned char* data, const int w, const int h, const int channels) {
        if (!g_create_texture)
            return {0, w, h};
        return g_create_texture(data, w, h, channels);
    }

    void delete_gl_texture(const uint32_t texture_id) {
        if (!g_delete_texture || texture_id == 0)
            return;
        g_delete_texture(texture_id);
    }

    int get_max_texture_size() {
        assert(g_max_texture_size_fn);
        return g_max_texture_size_fn();
    }

    void set_implot_context(void* ctx) { g_implot_context = ctx; }
    void* get_implot_context() { return g_implot_context; }

    void set_view_context_state(void* state) {
        g_view_context_state = state;
    }

    void* get_view_context_state() {
        return g_view_context_state;
    }

    void set_shared_dpi_scale(float scale) {
        g_shared_dpi_scale = scale;
    }

    float get_shared_dpi_scale() {
        return g_shared_dpi_scale;
    }

    void set_ensure_initialized_callback(EnsureInitializedCallback cb) {
        g_ensure_initialized_callback = cb;
    }

    std::string extract_python_error() {
        PyObject *type, *value, *tb;
        PyErr_Fetch(&type, &value, &tb);
        std::string msg = "(unknown error)";

        if (value) {
            PyObject* str = PyObject_Str(value);
            if (str) {
                const char* c_msg = PyUnicode_AsUTF8(str);
                if (c_msg) {
                    msg = c_msg;
                }
                Py_DECREF(str);
            }
        }

        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(tb);
        return msg;
    }

    void invoke_python_cleanup() {
        if (g_bridge.cleanup) {
            g_bridge.cleanup();
        }
    }

    void draw_python_panels(const PanelSpace space, lfs::vis::Scene* scene) {
        if (!g_bridge.draw_panels)
            return;

        assert(Py_IsInitialized());
        if (!Py_IsInitialized() || !is_gil_state_ready())
            return;

        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();

        set_scene_for_python(scene);
        const PyGILState_STATE gil = PyGILState_Ensure();
        g_bridge.draw_panels(space);
        PyGILState_Release(gil);
        set_scene_for_python(nullptr);
    }

    bool has_python_panels(PanelSpace space) {
        if (g_ensure_initialized_callback)
            g_ensure_initialized_callback();

        if (!g_bridge.has_panels)
            return false;

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return false;
        const PyGILState_STATE gil = PyGILState_Ensure();
        const bool result = g_bridge.has_panels(space);
        PyGILState_Release(gil);
        return result;
    }

    std::vector<std::string> get_python_panel_names(PanelSpace space) {
        if (g_ensure_initialized_callback)
            g_ensure_initialized_callback();

        if (!g_bridge.get_panel_names)
            return {};

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return {};
        const PyGILState_STATE gil = PyGILState_Ensure();

        std::vector<std::string> result;
        g_bridge.get_panel_names(
            space,
            [](const char* name, void* ctx) {
                static_cast<std::vector<std::string>*>(ctx)->emplace_back(name);
            },
            &result);

        PyGILState_Release(gil);
        return result;
    }

    void draw_python_panel(const std::string& name, lfs::vis::Scene* scene) {
        if (!g_bridge.draw_single_panel)
            return;

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return;

        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();

        set_scene_for_python(scene);
        const PyGILState_STATE gil = PyGILState_Ensure();
        g_bridge.draw_single_panel(name.c_str());
        PyGILState_Release(gil);
        set_scene_for_python(nullptr);
    }

    bool has_main_panel_tabs() {
        if (g_ensure_initialized_callback)
            g_ensure_initialized_callback();

        if (!g_bridge.has_main_panel_tabs)
            return false;

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return false;
        const PyGILState_STATE gil = PyGILState_Ensure();
        const bool result = g_bridge.has_main_panel_tabs();
        PyGILState_Release(gil);
        return result;
    }

    std::vector<MainPanelTabInfo> get_main_panel_tabs() {
        if (g_ensure_initialized_callback)
            g_ensure_initialized_callback();

        if (!g_bridge.get_main_panel_tabs)
            return {};

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return {};
        const PyGILState_STATE gil = PyGILState_Ensure();

        std::vector<MainPanelTabInfo> result;
        g_bridge.get_main_panel_tabs(
            [](const char* idname, const char* label, int order, bool enabled, void* ctx) {
                auto* vec = static_cast<std::vector<MainPanelTabInfo>*>(ctx);
                vec->push_back({idname, label, order, enabled});
            },
            &result);

        PyGILState_Release(gil);
        return result;
    }

    void draw_main_panel_tab(const std::string& idname, lfs::vis::Scene* scene) {
        if (!g_bridge.draw_main_panel_tab)
            return;

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return;

        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();

        set_scene_for_python(scene);
        const PyGILState_STATE gil = PyGILState_Ensure();
        g_bridge.draw_main_panel_tab(idname.c_str());
        PyGILState_Release(gil);
        set_scene_for_python(nullptr);
    }

    void draw_python_menu_items(MenuLocation location) {
        if (!g_bridge.draw_menus)
            return;
#ifndef NDEBUG
        assert(Py_IsInitialized() && "Python not initialized before draw_python_menu_items");
#endif
        if (!Py_IsInitialized() || !is_gil_state_ready())
            return;

        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();

        const PyGILState_STATE gil = PyGILState_Ensure();
        g_bridge.draw_menus(location);
        PyGILState_Release(gil);
    }

    bool has_python_menu_items(MenuLocation location) {
        if (g_ensure_initialized_callback)
            g_ensure_initialized_callback();

        if (!g_bridge.has_menus)
            return false;

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return false;
        const PyGILState_STATE gil = PyGILState_Ensure();
        const bool result = g_bridge.has_menus(location);
        PyGILState_Release(gil);
        return result;
    }

    bool has_menu_bar_entries() {
        if (g_ensure_initialized_callback)
            g_ensure_initialized_callback();

        if (!g_bridge.has_menu_bar_entries)
            return false;

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return false;
        const PyGILState_STATE gil = PyGILState_Ensure();
        const bool result = g_bridge.has_menu_bar_entries();
        PyGILState_Release(gil);
        return result;
    }

    std::vector<MenuBarEntry> get_menu_bar_entries() {
        if (g_ensure_initialized_callback)
            g_ensure_initialized_callback();

        if (!g_bridge.get_menu_bar_entries)
            return {};

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return {};
        const PyGILState_STATE gil = PyGILState_Ensure();

        std::vector<MenuBarEntry> result;
        g_bridge.get_menu_bar_entries(
            [](const char* idname, const char* label, int order, void* ctx) {
                auto* vec = static_cast<std::vector<MenuBarEntry>*>(ctx);
                vec->push_back({idname, label, order});
            },
            &result);

        PyGILState_Release(gil);
        return result;
    }

    void draw_menu_bar_entry(const std::string& idname) {
        if (!g_bridge.draw_menu_bar_entry)
            return;

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return;

        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();

        const PyGILState_STATE gil = PyGILState_Ensure();
        g_bridge.draw_menu_bar_entry(idname.c_str());
        PyGILState_Release(gil);
    }

    void draw_python_modals(lfs::vis::Scene* scene) {
        if (!g_bridge.draw_modals)
            return;
#ifndef NDEBUG
        assert(Py_IsInitialized() && "Python not initialized before draw_python_modals");
#endif
        if (!Py_IsInitialized() || !is_gil_state_ready())
            return;

        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();

        set_scene_for_python(scene);
        const PyGILState_STATE gil = PyGILState_Ensure();
        g_bridge.draw_modals();
        PyGILState_Release(gil);
        set_scene_for_python(nullptr);
    }

    bool has_python_modals() {
        if (!g_bridge.has_modals)
            return false;

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return false;
        const PyGILState_STATE gil = PyGILState_Ensure();
        const bool result = g_bridge.has_modals();
        PyGILState_Release(gil);
        return result;
    }

    void set_popup_draw_callback(DrawPopupsCallback cb) { g_popup_draw_callback = cb; }

    void draw_python_popups(lfs::vis::Scene* scene) {
        if (!g_popup_draw_callback)
            return;

        assert(Py_IsInitialized());
        if (!Py_IsInitialized() || !is_gil_state_ready())
            return;

        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();

        set_scene_for_python(scene);
        const PyGILState_STATE gil = PyGILState_Ensure();
        g_popup_draw_callback();
        PyGILState_Release(gil);
        set_scene_for_python(nullptr);
    }

    void set_export_callback(ExportCallback cb) { g_export_callback = cb; }

    void invoke_export(int format, const std::string& path,
                       const std::vector<std::string>& node_names, int sh_degree) {
        if (!g_export_callback)
            return;

        std::vector<const char*> names_ptrs;
        names_ptrs.reserve(node_names.size());
        for (const auto& name : node_names) {
            names_ptrs.push_back(name.c_str());
        }
        g_export_callback(format, path.c_str(), names_ptrs.data(),
                          static_cast<int>(names_ptrs.size()), sh_degree);
    }

    bool has_python_toolbar() {
        if (!g_bridge.has_toolbar)
            return false;

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return false;
        const PyGILState_STATE gil = PyGILState_Ensure();
        const bool result = g_bridge.has_toolbar();
        PyGILState_Release(gil);
        return result;
    }

    void cancel_active_operator() {
        static std::atomic<bool> in_cancel{false};

        if (in_cancel.load())
            return;

        auto* callbacks = g_operator_callbacks.load();
        if (!callbacks || !callbacks->hasCancelOperatorCallback())
            return;

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return;

        in_cancel.store(true);
        const PyGILState_STATE gil = PyGILState_Ensure();
        callbacks->cancelOperator();
        PyGILState_Release(gil);
        in_cancel.store(false);
    }

    bool invoke_operator(const std::string& operator_id) {
        static std::atomic<bool> in_invoke{false};

        if (in_invoke.load())
            return false;

        auto* callbacks = g_operator_callbacks.load();
        if (!callbacks || !callbacks->hasInvokeOperatorCallback())
            return false;

        if (!Py_IsInitialized() || !is_gil_state_ready())
            return false;

        in_invoke.store(true);
        const PyGILState_STATE gil = PyGILState_Ensure();
        const bool result = callbacks->invokeOperator(operator_id.c_str());
        PyGILState_Release(gil);
        in_invoke.store(false);
        return result;
    }

    // Selection sub-mode
    void set_selection_submode(int mode) { g_selection_submode.store(mode); }
    int get_selection_submode() { return g_selection_submode.load(); }

    bool dispatch_modal_event(const ModalEvent& event) {
        auto* callbacks = g_operator_callbacks.load();
        if (!callbacks || !callbacks->hasModalEventCallback()) {
            return false;
        }

        // Convert python::ModalEvent to core::ModalEvent
        lfs::core::ModalEvent core_evt{};
        core_evt.type = static_cast<lfs::core::ModalEvent::Type>(event.type);
        core_evt.x = event.x;
        core_evt.y = event.y;
        core_evt.delta_x = event.delta_x;
        core_evt.delta_y = event.delta_y;
        core_evt.button = event.button;
        core_evt.action = event.action;
        core_evt.key = event.key;
        core_evt.mods = event.mods;
        core_evt.scroll_x = event.scroll_x;
        core_evt.scroll_y = event.scroll_y;

        if (!Py_IsInitialized() || !is_gil_state_ready()) {
            return false;
        }
        const PyGILState_STATE gil = PyGILState_Ensure();
        const bool consumed = callbacks->dispatchModalEvent(core_evt);
        PyGILState_Release(gil);
        return consumed;
    }

    void set_viewport_bounds(float x, float y, float w, float h) {
        std::lock_guard lock(g_viewport.mutex);
        g_viewport.x = x;
        g_viewport.y = y;
        g_viewport.w = w;
        g_viewport.h = h;
        g_viewport.is_set = true;
    }

    void get_viewport_bounds(float& x, float& y, float& w, float& h) {
        std::lock_guard lock(g_viewport.mutex);
        x = g_viewport.x;
        y = g_viewport.y;
        w = g_viewport.w;
        h = g_viewport.h;
    }

    bool has_viewport_bounds() {
        std::lock_guard lock(g_viewport.mutex);
        return g_viewport.is_set;
    }

    bool is_exit_popup_open() { return g_exit_popup_open.load(); }
    void set_exit_popup_open(bool open) { g_exit_popup_open.store(open); }

    void set_thumbnail_callbacks(RequestThumbnailCallback request_cb,
                                 ProcessThumbnailsCallback process_cb,
                                 IsThumbnailReadyCallback ready_cb,
                                 GetThumbnailTextureCallback texture_cb) {
        g_request_thumbnail_cb = request_cb;
        g_process_thumbnails_cb = process_cb;
        g_is_thumbnail_ready_cb = ready_cb;
        g_get_thumbnail_texture_cb = texture_cb;
    }

    void request_thumbnail(const std::string& video_id) {
        if (g_request_thumbnail_cb) {
            g_request_thumbnail_cb(video_id.c_str());
        }
    }

    void process_thumbnails() {
        if (g_process_thumbnails_cb) {
            g_process_thumbnails_cb();
        }
    }

    bool is_thumbnail_ready(const std::string& video_id) {
        if (g_is_thumbnail_ready_cb) {
            return g_is_thumbnail_ready_cb(video_id.c_str());
        }
        return false;
    }

    uint64_t get_thumbnail_texture(const std::string& video_id) {
        if (g_get_thumbnail_texture_cb) {
            return g_get_thumbnail_texture_cb(video_id.c_str());
        }
        return 0;
    }

    namespace {
        std::mutex g_keyboard_capture_mutex;
        std::unordered_set<std::string> g_keyboard_capture_owners;
    } // namespace

    void request_keyboard_capture(const std::string& owner_id) {
        std::lock_guard lock(g_keyboard_capture_mutex);
        g_keyboard_capture_owners.insert(owner_id);
    }

    void release_keyboard_capture(const std::string& owner_id) {
        std::lock_guard lock(g_keyboard_capture_mutex);
        g_keyboard_capture_owners.erase(owner_id);
    }

    bool has_keyboard_capture_request() {
        std::lock_guard lock(g_keyboard_capture_mutex);
        return !g_keyboard_capture_owners.empty();
    }

    namespace {
        InvalidatePollCacheCallback g_invalidate_poll_cache_cb = nullptr;
    } // namespace

    void set_invalidate_poll_cache_callback(InvalidatePollCacheCallback cb) {
        g_invalidate_poll_cache_cb = cb;
    }

    void invalidate_poll_caches(uint8_t dependency) {
        if (g_invalidate_poll_cache_cb) {
            g_invalidate_poll_cache_cb(dependency);
        }
    }

    namespace {
        SignalBridgeCallbacks g_signal_bridge_callbacks;
    } // namespace

    void set_signal_bridge_callbacks(const SignalBridgeCallbacks& callbacks) {
        g_signal_bridge_callbacks = callbacks;
    }

    void update_training_progress(int iteration, float loss, std::size_t num_gaussians) {
        if (g_signal_bridge_callbacks.training_progress) {
            g_signal_bridge_callbacks.training_progress(iteration, loss, num_gaussians);
        }
    }

    void update_training_state(bool is_training, const char* state) {
        g_is_training.store(is_training, std::memory_order_relaxed);
        if (g_signal_bridge_callbacks.training_state) {
            g_signal_bridge_callbacks.training_state(is_training, state);
        }
    }

    void update_trainer_loaded(bool has_trainer, int max_iterations) {
        if (g_signal_bridge_callbacks.trainer_loaded) {
            g_signal_bridge_callbacks.trainer_loaded(has_trainer, max_iterations);
        }
    }

    void update_psnr(float psnr) {
        if (g_signal_bridge_callbacks.psnr) {
            g_signal_bridge_callbacks.psnr(psnr);
        }
    }

    void update_scene(bool has_scene, const char* path) {
        if (g_signal_bridge_callbacks.scene) {
            g_signal_bridge_callbacks.scene(has_scene, path);
        }
    }

    void update_selection(bool has_selection, int count) {
        g_has_selection.store(has_selection, std::memory_order_relaxed);
        if (g_signal_bridge_callbacks.selection) {
            g_signal_bridge_callbacks.selection(has_selection, count);
        }
    }

    void flush_signals() {
        if (g_signal_bridge_callbacks.flush) {
            g_signal_bridge_callbacks.flush();
        }
    }

} // namespace lfs::python
