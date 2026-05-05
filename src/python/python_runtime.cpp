/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_runtime.hpp"
#include "core/logger.hpp"
#include "core/modal_event.hpp"
#include "core/operator_callbacks.hpp"
#include "gil.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <mutex>
#include <unordered_set>

#include "python_compat.hpp"

namespace lfs::python {

    namespace {
        class AtomicBoolResetGuard {
            std::atomic<bool>& flag_;

        public:
            explicit AtomicBoolResetGuard(std::atomic<bool>& flag) noexcept
                : flag_(flag) {}

            ~AtomicBoolResetGuard() {
                flag_.store(false, std::memory_order_release);
            }

            AtomicBoolResetGuard(const AtomicBoolResetGuard&) = delete;
            AtomicBoolResetGuard& operator=(const AtomicBoolResetGuard&) = delete;
            AtomicBoolResetGuard(AtomicBoolResetGuard&&) = delete;
            AtomicBoolResetGuard& operator=(AtomicBoolResetGuard&&) = delete;
        };

        PyBridge g_bridge{};

        ExportCallback g_export_callback = nullptr;
        DrawPopupsCallback g_popup_draw_callback = nullptr;
        EnsureInitializedCallback g_ensure_initialized_callback = nullptr;

        // Exit popup state for window close callback (thread-safe)
        std::atomic<bool> g_exit_popup_open{false};

        // Graphics-thread callback queue (set once during module init, before any reader threads)
        std::thread::id g_graphics_thread_id{};
        std::mutex g_graphics_callbacks_mutex;
        std::vector<std::function<void()>> g_graphics_callbacks;

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
        GetSequencerUIStateCallback g_get_sequencer_ui_state_cb;

        // Pivot mode callbacks
        GetPivotModeCallback g_get_pivot_mode_cb = nullptr;
        SetPivotModeCallback g_set_pivot_mode_cb = nullptr;

        // Transform space callbacks
        GetTransformSpaceCallback g_get_transform_space_cb = nullptr;
        SetTransformSpaceCallback g_set_transform_space_cb = nullptr;

        // Asset Manager save callback
        SaveAssetCallback g_save_asset_cb = nullptr;

        // Thumbnail callbacks
        RequestThumbnailCallback g_request_thumbnail_cb = nullptr;
        ProcessThumbnailsCallback g_process_thumbnails_cb = nullptr;
        IsThumbnailReadyCallback g_is_thumbnail_ready_cb = nullptr;
        GetThumbnailTextureCallback g_get_thumbnail_texture_cb = nullptr;

        // Viewport overlay callbacks
        HasViewportDrawHandlersCallback g_has_viewport_draw_handlers_cb = nullptr;
        InvokeViewportOverlayCallback g_invoke_viewport_overlay_cb = nullptr;

        // Selection sub-mode (shared between C++ toolbar and Python operator)
        std::atomic<int> g_selection_submode{0};

        // Note: Operator callbacks (cancel, invoke, modal event) are now stored in EditorContext
        // to avoid static variable duplication across shared libraries (RTLD_LOCAL issue)

        // Operation context (short-lived, per-call)
        // Thread-safety: Set/cleared only via SceneContextGuard RAII, accessed from single thread
        core::Scene* g_scene_for_python = nullptr;

        ApplicationSceneContext g_app_scene_context;
        std::atomic<vis::Visualizer*> g_visualizer{nullptr};
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
        void* g_rml_manager{nullptr};
        RmlContextDestroyFn g_rml_context_destroy_handler{nullptr};
        RmlPanelHostOps g_rml_panel_host_ops{};
        RmlDocRegisterCallback g_rml_doc_register_cb{nullptr};
        RmlDocUnregisterCallback g_rml_doc_unregister_cb{nullptr};

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
        MainLoopWakeCallback g_main_loop_wake_callback = nullptr;
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
    void request_redraw() {
        const bool was_requested = g_redraw_requested.exchange(true, std::memory_order_acq_rel);
        if (!was_requested && g_main_loop_wake_callback)
            g_main_loop_wake_callback();
    }

    bool consume_redraw_request() {
        return g_redraw_requested.exchange(false, std::memory_order_acq_rel);
    }

    void set_main_loop_wake_callback(MainLoopWakeCallback cb) {
        g_main_loop_wake_callback = cb;
    }

    // Operation context (short-lived)
    void set_scene_for_python(core::Scene* scene) { g_scene_for_python = scene; }
    core::Scene* get_scene_for_python() { return g_scene_for_python; }

    void set_trainer_manager(vis::TrainerManager* tm) { g_trainer_manager.store(tm); }
    vis::TrainerManager* get_trainer_manager() { return g_trainer_manager.load(); }

    void set_visualizer(vis::Visualizer* viewer) { g_visualizer.store(viewer); }
    vis::Visualizer* get_visualizer() { return g_visualizer.load(); }

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
        std::atomic<vis::gui::GlobalContextMenu*> g_global_context_menu{nullptr};
    }

    void set_global_context_menu(vis::gui::GlobalContextMenu* cm) { g_global_context_menu.store(cm); }
    vis::gui::GlobalContextMenu* get_global_context_menu() { return g_global_context_menu.load(); }

    namespace {
        Mesh2SplatStartFn g_m2s_start;
        std::function<bool()> g_m2s_active;
        std::function<float()> g_m2s_progress;
        std::function<std::string()> g_m2s_stage;
        std::function<std::string()> g_m2s_error;
    } // namespace

    void set_mesh2splat_callbacks(Mesh2SplatStartFn start,
                                  std::function<bool()> is_active,
                                  std::function<float()> get_progress,
                                  std::function<std::string()> get_stage,
                                  std::function<std::string()> get_error) {
        g_m2s_start = std::move(start);
        g_m2s_active = std::move(is_active);
        g_m2s_progress = std::move(get_progress);
        g_m2s_stage = std::move(get_stage);
        g_m2s_error = std::move(get_error);
    }

    void invoke_mesh2splat_start(std::shared_ptr<core::MeshData> mesh, const std::string& name,
                                 const core::Mesh2SplatOptions& options) {
        if (g_m2s_start)
            g_m2s_start(std::move(mesh), name, options);
    }

    bool invoke_mesh2splat_active() { return g_m2s_active ? g_m2s_active() : false; }
    float invoke_mesh2splat_progress() { return g_m2s_progress ? g_m2s_progress() : 0.0f; }
    std::string invoke_mesh2splat_stage() { return g_m2s_stage ? g_m2s_stage() : std::string{}; }
    std::string invoke_mesh2splat_error() { return g_m2s_error ? g_m2s_error() : std::string{}; }

    namespace {
        SplatSimplifyStartFn g_splat_simplify_start;
        std::function<void()> g_splat_simplify_cancel;
        std::function<bool()> g_splat_simplify_active;
        std::function<float()> g_splat_simplify_progress;
        std::function<std::string()> g_splat_simplify_stage;
        std::function<std::string()> g_splat_simplify_error;
    } // namespace

    void set_splat_simplify_callbacks(SplatSimplifyStartFn start,
                                      std::function<void()> cancel,
                                      std::function<bool()> is_active,
                                      std::function<float()> get_progress,
                                      std::function<std::string()> get_stage,
                                      std::function<std::string()> get_error) {
        g_splat_simplify_start = std::move(start);
        g_splat_simplify_cancel = std::move(cancel);
        g_splat_simplify_active = std::move(is_active);
        g_splat_simplify_progress = std::move(get_progress);
        g_splat_simplify_stage = std::move(get_stage);
        g_splat_simplify_error = std::move(get_error);
    }

    void invoke_splat_simplify_start(const std::string& name, const core::SplatSimplifyOptions& options) {
        if (g_splat_simplify_start)
            g_splat_simplify_start(name, options);
    }

    void invoke_splat_simplify_cancel() {
        if (g_splat_simplify_cancel)
            g_splat_simplify_cancel();
    }

    bool invoke_splat_simplify_active() { return g_splat_simplify_active ? g_splat_simplify_active() : false; }
    float invoke_splat_simplify_progress() { return g_splat_simplify_progress ? g_splat_simplify_progress() : 0.0f; }
    std::string invoke_splat_simplify_stage() { return g_splat_simplify_stage ? g_splat_simplify_stage() : std::string{}; }
    std::string invoke_splat_simplify_error() { return g_splat_simplify_error ? g_splat_simplify_error() : std::string{}; }

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

    void toggle_system_console() {
        if (g_section_draw_callbacks.toggle_system_console)
            g_section_draw_callbacks.toggle_system_console();
    }

    void set_sequencer_ui_state_callback(GetSequencerUIStateCallback cb) {
        g_get_sequencer_ui_state_cb = std::move(cb);
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

    void set_save_asset_callback(SaveAssetCallback save_cb) {
        g_save_asset_cb = save_cb;
    }

    void invoke_save_asset(const std::string& node_name) {
        if (g_save_asset_cb)
            g_save_asset_cb(node_name.c_str());
    }

    void set_scene_manager(vis::SceneManager* sm) { g_scene_manager.store(sm); }
    vis::SceneManager* get_scene_manager() { return g_scene_manager.load(); }

    void set_selection_service(vis::SelectionService* ss) { g_selection_service.store(ss); }
    vis::SelectionService* get_selection_service() { return g_selection_service.load(); }

    // Application context (long-lived)
    void ApplicationSceneContext::set(core::Scene* scene) {
        scene_.store(scene);
        mutation_flags_.store(0, std::memory_order_release);
        generation_.fetch_add(1);
    }

    core::Scene* ApplicationSceneContext::get() const { return scene_.load(); }

    uint64_t ApplicationSceneContext::generation() const { return generation_.load(); }

    uint32_t ApplicationSceneContext::mutation_flags() const {
        return mutation_flags_.load(std::memory_order_acquire);
    }

    uint32_t ApplicationSceneContext::consume_mutation_flags() {
        return mutation_flags_.exchange(0, std::memory_order_acq_rel);
    }

    void ApplicationSceneContext::bump() { generation_.fetch_add(1); }

    void ApplicationSceneContext::set_mutation_flags(const uint32_t flags) {
        mutation_flags_.fetch_or(flags, std::memory_order_acq_rel);
    }

    void set_application_scene(core::Scene* scene) { g_app_scene_context.set(scene); }

    core::Scene* get_application_scene() { return g_app_scene_context.get(); }

    uint64_t get_scene_generation() { return g_app_scene_context.generation(); }

    uint32_t get_scene_mutation_flags() { return g_app_scene_context.mutation_flags(); }

    uint32_t consume_scene_mutation_flags() { return g_app_scene_context.consume_mutation_flags(); }

    void bump_scene_generation() { g_app_scene_context.bump(); }

    void set_scene_mutation_flags(const uint32_t flags) {
        g_app_scene_context.set_mutation_flags(flags);
    }

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

    void set_ui_texture_service(const CreateTextureFn create, const DeleteTextureFn del, const MaxTextureSizeFn max_size) {
        assert(create && del && max_size);
        g_create_texture = create;
        g_delete_texture = del;
        g_max_texture_size_fn = max_size;
    }

    TextureResult create_ui_texture(const unsigned char* data, const int w, const int h, const int channels) {
        if (!g_create_texture)
            return {0, w, h};
        return g_create_texture(data, w, h, channels);
    }

    void delete_ui_texture(const uint64_t texture_id) {
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

    void set_rml_manager(void* manager) {
        g_rml_manager = manager;
    }

    void* get_rml_manager() {
        return g_rml_manager;
    }

    void set_rml_context_destroy_handler(RmlContextDestroyFn fn) {
        g_rml_context_destroy_handler = fn;
    }

    RmlContextDestroyFn get_rml_context_destroy_handler() {
        return g_rml_context_destroy_handler;
    }

    void set_rml_panel_host_ops(const RmlPanelHostOps& ops) {
        g_rml_panel_host_ops = ops;
    }

    const RmlPanelHostOps& get_rml_panel_host_ops() {
        return g_rml_panel_host_ops;
    }

    void set_rml_doc_registry_callbacks(RmlDocRegisterCallback reg_cb,
                                        RmlDocUnregisterCallback unreg_cb) {
        g_rml_doc_register_cb = reg_cb;
        g_rml_doc_unregister_cb = unreg_cb;
    }

    void register_rml_document(const char* name, void* doc) {
        if (g_rml_doc_register_cb)
            g_rml_doc_register_cb(name, doc);
    }

    void unregister_rml_document(const char* name) {
        if (g_rml_doc_unregister_cb)
            g_rml_doc_unregister_cb(name);
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

    void shutdown_python_ui_resources() {
        if (g_bridge.shutdown_ui_resources) {
            g_bridge.shutdown_ui_resources();
        }
    }

    void draw_python_menu_items(MenuLocation location) {
        if (!g_bridge.draw_menus)
            return;
#ifndef NDEBUG
        assert(Py_IsInitialized() && "Python not initialized before draw_python_menu_items");
#endif
        if (!can_acquire_gil())
            return;

        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();

        const GilAcquire gil;
        g_bridge.draw_menus(location);
    }

    bool has_python_menu_items(MenuLocation location) {
        if (g_ensure_initialized_callback)
            g_ensure_initialized_callback();

        if (!g_bridge.has_menus)
            return false;

        if (!can_acquire_gil())
            return false;
        const GilAcquire gil;
        return g_bridge.has_menus(location);
    }

    bool has_menu_bar_entries() {
        if (g_ensure_initialized_callback)
            g_ensure_initialized_callback();

        if (!g_bridge.has_menu_bar_entries)
            return false;

        if (!can_acquire_gil())
            return false;
        const GilAcquire gil;
        return g_bridge.has_menu_bar_entries();
    }

    std::vector<MenuBarEntry> get_menu_bar_entries() {
        if (g_ensure_initialized_callback)
            g_ensure_initialized_callback();

        if (!g_bridge.get_menu_bar_entries)
            return {};

        if (!can_acquire_gil())
            return {};

        const GilAcquire gil;
        std::vector<MenuBarEntry> result;
        g_bridge.get_menu_bar_entries(
            [](const char* idname, const char* label, int order, void* ctx) {
                auto* vec = static_cast<std::vector<MenuBarEntry>*>(ctx);
                vec->push_back({idname, label, order});
            },
            &result);

        return result;
    }

    void draw_menu_bar_entry(const std::string& idname) {
        if (!g_bridge.draw_menu_bar_entry)
            return;

        if (!can_acquire_gil())
            return;

        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();

        const GilAcquire gil;
        g_bridge.draw_menu_bar_entry(idname.c_str());
    }

    void collect_menu_content(const std::string& idname, MenuItemVisitor visitor, void* user_data) {
        if (!g_bridge.collect_menu_content)
            return;
        if (!can_acquire_gil())
            return;
        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();
        const GilAcquire gil;
        g_bridge.collect_menu_content(idname.c_str(), visitor, user_data);
    }

    void execute_menu_callback(const std::string& idname, int callback_index) {
        if (!g_bridge.execute_menu_callback)
            return;
        if (!can_acquire_gil())
            return;
        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();
        const GilAcquire gil;
        g_bridge.execute_menu_callback(idname.c_str(), callback_index);
    }

    void draw_python_modals(lfs::core::Scene* scene) {
        if (!g_bridge.draw_modals)
            return;
#ifndef NDEBUG
        assert(Py_IsInitialized() && "Python not initialized before draw_python_modals");
#endif
        if (!can_acquire_gil())
            return;

        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();

        const SceneContextGuard scene_guard(scene);
        {
            const GilAcquire gil;
            g_bridge.draw_modals();
        }
    }

    bool has_python_modals() {
        if (!g_bridge.has_modals)
            return false;

        if (!can_acquire_gil())
            return false;
        const GilAcquire gil;
        return g_bridge.has_modals();
    }

    namespace {
        ModalEnqueueCallback g_modal_enqueue_callback;
    }

    void set_modal_enqueue_callback(ModalEnqueueCallback cb) { g_modal_enqueue_callback = std::move(cb); }
    const ModalEnqueueCallback& get_modal_enqueue_callback() { return g_modal_enqueue_callback; }

    void set_popup_draw_callback(DrawPopupsCallback cb) { g_popup_draw_callback = cb; }

    void draw_python_popups(lfs::core::Scene* scene) {
        if (!g_popup_draw_callback)
            return;

        assert(Py_IsInitialized());
        if (!can_acquire_gil())
            return;

        if (g_bridge.prepare_ui)
            g_bridge.prepare_ui();

        const SceneContextGuard scene_guard(scene);
        {
            const GilAcquire gil;
            g_popup_draw_callback();
        }
    }

    void set_export_callback(ExportCallback cb) { g_export_callback = cb; }

    void invoke_export(int format, const std::string& path,
                       const std::vector<std::string>& node_names, int sh_degree,
                       const std::vector<float>& rad_lod_ratios,
                       bool rad_flip_y) {
        if (!g_export_callback)
            return;

        std::vector<const char*> names_ptrs;
        names_ptrs.reserve(node_names.size());
        for (const auto& name : node_names) {
            names_ptrs.push_back(name.c_str());
        }
        g_export_callback(format, path.c_str(), names_ptrs.data(),
                          static_cast<int>(names_ptrs.size()), sh_degree,
                          rad_lod_ratios.data(), static_cast<int>(rad_lod_ratios.size()),
                          rad_flip_y);
    }

    bool has_python_toolbar() {
        if (!g_bridge.has_toolbar)
            return false;

        if (!can_acquire_gil())
            return false;
        const GilAcquire gil;
        return g_bridge.has_toolbar();
    }

    void cancel_active_operator() {
        static std::atomic<bool> in_cancel{false};

        bool expected = false;
        if (!in_cancel.compare_exchange_strong(expected, true))
            return;
        const AtomicBoolResetGuard reset_guard(in_cancel);

        auto* callbacks = g_operator_callbacks.load();
        if (!callbacks || !callbacks->hasCancelOperatorCallback())
            return;

        if (!can_acquire_gil())
            return;

        try {
            const GilAcquire gil;
            callbacks->cancelOperator();
        } catch (const std::exception& e) {
            LOG_ERROR("cancel_active_operator callback failed: {}", e.what());
        } catch (...) {
            LOG_ERROR("cancel_active_operator callback failed with unknown exception");
        }
    }

    bool invoke_operator(const std::string& operator_id) {
        static std::atomic<bool> in_invoke{false};

        bool expected = false;
        if (!in_invoke.compare_exchange_strong(expected, true))
            return false;
        const AtomicBoolResetGuard reset_guard(in_invoke);

        auto* callbacks = g_operator_callbacks.load();
        if (!callbacks || !callbacks->hasInvokeOperatorCallback())
            return false;

        if (!can_acquire_gil())
            return false;

        try {
            const GilAcquire gil;
            return callbacks->invokeOperator(operator_id.c_str());
        } catch (const std::exception& e) {
            LOG_ERROR("invoke_operator '{}' callback failed: {}", operator_id, e.what());
        } catch (...) {
            LOG_ERROR("invoke_operator '{}' callback failed with unknown exception", operator_id);
        }
        return false;
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
        core_evt.over_gui = event.over_gui;

        if (!can_acquire_gil())
            return false;
        const GilAcquire gil;
        return callbacks->dispatchModalEvent(core_evt);
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

    void set_graphics_thread_id(std::thread::id id) { g_graphics_thread_id = id; }

    bool on_graphics_thread() {
        return g_graphics_thread_id != std::thread::id{} &&
               std::this_thread::get_id() == g_graphics_thread_id;
    }

    void schedule_graphics_callback(std::function<void()> fn) {
        std::lock_guard lock(g_graphics_callbacks_mutex);
        g_graphics_callbacks.push_back(std::move(fn));
    }

    void flush_graphics_callbacks() {
        std::vector<std::function<void()>> pending;
        {
            std::lock_guard lock(g_graphics_callbacks_mutex);
            pending.swap(g_graphics_callbacks);
        }
        for (auto& fn : pending)
            fn();
    }

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

    void set_viewport_overlay_callbacks(HasViewportDrawHandlersCallback has_cb,
                                        InvokeViewportOverlayCallback invoke_cb) {
        g_has_viewport_draw_handlers_cb = has_cb;
        g_invoke_viewport_overlay_cb = invoke_cb;
    }

    bool has_viewport_draw_handlers() {
        return g_has_viewport_draw_handlers_cb && g_has_viewport_draw_handlers_cb();
    }

    void invoke_viewport_overlay(const float* view_matrix, const float* proj_matrix,
                                 const float* vp_pos, const float* vp_size,
                                 const float* cam_pos, const float* cam_fwd,
                                 void* overlay_renderer,
                                 void* draw_list) {
        if (g_invoke_viewport_overlay_cb) {
            g_invoke_viewport_overlay_cb(view_matrix, proj_matrix, vp_pos, vp_size,
                                         cam_pos, cam_fwd, overlay_renderer, draw_list);
        }
    }

} // namespace lfs::python
