/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
#ifdef LFS_PYTHON_RUNTIME_EXPORTS
#define LFS_PYTHON_RUNTIME_API __declspec(dllexport)
#else
#define LFS_PYTHON_RUNTIME_API __declspec(dllimport)
#endif
#else
#define LFS_PYTHON_RUNTIME_API
#endif

namespace lfs::core {
    class IOperatorCallbacks;
} // namespace lfs::core

namespace lfs::vis {
    class Scene;
    class SceneManager;
    class TrainerManager;
    class ParameterManager;
    class RenderingManager;
    class EditorContext;
    class SelectionService;
    namespace gui {
        class GuiManager;
    }
    namespace input {
        class InputBindings;
    }
} // namespace lfs::vis

namespace lfs::python {

    // Panel space types
    enum class PanelSpace {
        SidePanel,
        Floating,
        ViewportOverlay,
        Dockable,
        MainPanelTab, // Main right panel tabs (Rendering, Training, etc.)
        SceneHeader,  // Scene panel above the tabs
        StatusBar     // Bottom status bar (22px height)
    };

    struct PyContext {
        // Managers (set directly)
        vis::Scene* scene = nullptr;
        vis::SceneManager* scene_manager = nullptr;
        vis::TrainerManager* trainer = nullptr;
        vis::RenderingManager* rendering = nullptr;
        vis::SelectionService* selection = nullptr;

        // Derived state (owned by set_context(), do not set externally)
        uint64_t scene_generation = 0;
        bool cached_has_selection = false;
        bool cached_is_training = false;

        // UI state (consolidated from callbacks)
        int selection_submode = 0;
        int pivot_mode = 0;
        int transform_space = 0;

        // Viewport (set before UI drawing)
        float vp_x = 0, vp_y = 0, vp_w = 0, vp_h = 0;

        // Helpers - use cached values for fast access
        bool has_scene() const { return scene != nullptr; }
        bool has_selection() const { return cached_has_selection; }
        bool is_training() const { return cached_is_training; }
    };

    // The one source of truth
    LFS_PYTHON_RUNTIME_API void set_context(const PyContext& ctx);
    LFS_PYTHON_RUNTIME_API const PyContext& context();

    // UI redraw request mechanism
    LFS_PYTHON_RUNTIME_API void request_redraw();
    LFS_PYTHON_RUNTIME_API bool consume_redraw_request();

    // Panel callbacks (C-style function pointers for DLL boundary safety)
    using DrawPanelsCallback = void (*)(PanelSpace);
    using DrawSinglePanelCallback = void (*)(const char*);
    using HasPanelsCallback = bool (*)(PanelSpace);
    using CleanupCallback = void (*)();
    using EnsureInitializedCallback = void (*)();
    using PanelNameVisitor = void (*)(const char* name, void* user_data);
    using GetPanelNamesCallback = void (*)(PanelSpace, PanelNameVisitor, void* user_data);

    // Bootstrap callback - set at startup, triggers Python initialization
    LFS_PYTHON_RUNTIME_API void set_ensure_initialized_callback(EnsureInitializedCallback cb);

    // Menu locations (mirrors py_ui.hpp MenuLocation)
    enum class MenuLocation {
        File,
        Edit,
        View,
        Window,
        Help,
        MenuBar // Top-level menu bar entries (Python-driven menus)
    };

    // Menu bar entry info for C++ UI code
    struct MenuBarEntry {
        std::string idname;
        std::string label;
        int order;
    };

    // Menu callbacks (C-style function pointers)
    using DrawMenuItemsCallback = void (*)(MenuLocation);
    using HasMenuItemsCallback = bool (*)(MenuLocation);

    // UI context preparation callback type
    using PrepareUIContextCallback = void (*)();

    // Safe Python error extraction - avoids nanobind::python_error::what() crash on Windows
    LFS_PYTHON_RUNTIME_API std::string extract_python_error();

    // Main panel tab info for C++ UI code
    struct MainPanelTabInfo {
        std::string idname;
        std::string label;
        int order;
        bool enabled;
    };

    // C++ interface for the visualizer
    LFS_PYTHON_RUNTIME_API void draw_python_panels(PanelSpace space, lfs::vis::Scene* scene = nullptr);
    LFS_PYTHON_RUNTIME_API void draw_python_panel(const std::string& name, lfs::vis::Scene* scene = nullptr);
    LFS_PYTHON_RUNTIME_API bool has_python_panels(PanelSpace space);
    LFS_PYTHON_RUNTIME_API std::vector<std::string> get_python_panel_names(PanelSpace space);
    LFS_PYTHON_RUNTIME_API void invoke_python_cleanup();

    // Main panel tab interface for C++ UI code
    LFS_PYTHON_RUNTIME_API bool has_main_panel_tabs();
    LFS_PYTHON_RUNTIME_API std::vector<MainPanelTabInfo> get_main_panel_tabs();
    LFS_PYTHON_RUNTIME_API void draw_main_panel_tab(const std::string& idname, lfs::vis::Scene* scene = nullptr);

    // Menu interface for C++ code
    LFS_PYTHON_RUNTIME_API void draw_python_menu_items(MenuLocation location);
    LFS_PYTHON_RUNTIME_API bool has_python_menu_items(MenuLocation location);

    // Menu bar entry interface for C++ UI code (Python-driven menu bar)
    LFS_PYTHON_RUNTIME_API bool has_menu_bar_entries();
    LFS_PYTHON_RUNTIME_API std::vector<MenuBarEntry> get_menu_bar_entries();
    LFS_PYTHON_RUNTIME_API void draw_menu_bar_entry(const std::string& idname);

    // Modal dialog callbacks
    using DrawModalsCallback = void (*)();
    using HasModalsCallback = bool (*)();

    LFS_PYTHON_RUNTIME_API void draw_python_modals(lfs::vis::Scene* scene = nullptr);
    LFS_PYTHON_RUNTIME_API bool has_python_modals();

    using DrawPopupsCallback = void (*)();
    LFS_PYTHON_RUNTIME_API void set_popup_draw_callback(DrawPopupsCallback cb);
    LFS_PYTHON_RUNTIME_API void draw_python_popups(lfs::vis::Scene* scene = nullptr);

    using ExportCallback = void (*)(int format, const char* path, const char** node_names, int node_count, int sh_degree);
    LFS_PYTHON_RUNTIME_API void set_export_callback(ExportCallback cb);
    LFS_PYTHON_RUNTIME_API void invoke_export(int format, const std::string& path,
                                              const std::vector<std::string>& node_names, int sh_degree);

    using HasToolbarCallback = bool (*)();

    LFS_PYTHON_RUNTIME_API bool has_python_toolbar();

    // Operator callbacks
    using CancelActiveOperatorCallback = void (*)();
    using InvokeOperatorCallback = bool (*)(const char*);

    LFS_PYTHON_RUNTIME_API void cancel_active_operator();
    LFS_PYTHON_RUNTIME_API bool invoke_operator(const std::string& operator_id);

    // Selection sub-mode (mirrors panels::SelectionSubMode for Python access)
    LFS_PYTHON_RUNTIME_API void set_selection_submode(int mode);
    LFS_PYTHON_RUNTIME_API int get_selection_submode();

    // Thumbnail system callbacks (for Getting Started window)
    using RequestThumbnailCallback = void (*)(const char* video_id);
    using ProcessThumbnailsCallback = void (*)();
    using IsThumbnailReadyCallback = bool (*)(const char* video_id);
    using GetThumbnailTextureCallback = uint64_t (*)(const char* video_id);

    LFS_PYTHON_RUNTIME_API void set_thumbnail_callbacks(
        RequestThumbnailCallback request_cb,
        ProcessThumbnailsCallback process_cb,
        IsThumbnailReadyCallback ready_cb,
        GetThumbnailTextureCallback texture_cb);

    LFS_PYTHON_RUNTIME_API void request_thumbnail(const std::string& video_id);
    LFS_PYTHON_RUNTIME_API void process_thumbnails();
    LFS_PYTHON_RUNTIME_API bool is_thumbnail_ready(const std::string& video_id);
    LFS_PYTHON_RUNTIME_API uint64_t get_thumbnail_texture(const std::string& video_id);

    // Modal event routing - InputController dispatches events to Python operators
    struct ModalEvent {
        enum class Type { MouseButton,
                          MouseMove,
                          Scroll,
                          Key };
        Type type;

        // Mouse state
        double x, y;
        double delta_x, delta_y;
        int button; // GLFW_MOUSE_BUTTON_*
        int action; // GLFW_PRESS, GLFW_RELEASE

        // Key state
        int key;
        int mods; // GLFW_MOD_SHIFT, etc.

        // Scroll
        double scroll_x, scroll_y;

        // GUI state - true if mouse is over ImGui window
        bool over_gui = false;
    };

    // Callback type: returns true if event was consumed by the operator
    using ModalEventCallback = bool (*)(const ModalEvent&);

    LFS_PYTHON_RUNTIME_API bool dispatch_modal_event(const ModalEvent& event);

    // Visitor callback for main panel tabs
    using MainPanelTabVisitor = void (*)(const char* idname, const char* label, int order, bool enabled, void* user_data);

    // Menu bar entry visitor callback
    using MenuBarEntryVisitor = void (*)(const char* idname, const char* label, int order, void* user_data);

    // Menu bar entry callbacks
    using HasMenuBarEntriesCallback = bool (*)();
    using GetMenuBarEntriesCallback = void (*)(MenuBarEntryVisitor, void* user_data);
    using DrawMenuBarEntryCallback = void (*)(const char* idname);

    struct PyBridge {
        // Panels
        void (*draw_panels)(PanelSpace) = nullptr;
        void (*draw_single_panel)(const char*) = nullptr;
        bool (*has_panels)(PanelSpace) = nullptr;
        void (*get_panel_names)(PanelSpace, PanelNameVisitor, void*) = nullptr;

        // Main panel tabs
        bool (*has_main_panel_tabs)() = nullptr;
        void (*get_main_panel_tabs)(MainPanelTabVisitor, void*) = nullptr;
        void (*draw_main_panel_tab)(const char*) = nullptr;

        // Menus
        void (*draw_menus)(MenuLocation) = nullptr;
        bool (*has_menus)(MenuLocation) = nullptr;

        // Menu bar entries (Python-driven top-level menus)
        bool (*has_menu_bar_entries)() = nullptr;
        void (*get_menu_bar_entries)(MenuBarEntryVisitor, void*) = nullptr;
        void (*draw_menu_bar_entry)(const char*) = nullptr;

        // Modals
        void (*draw_modals)() = nullptr;
        bool (*has_modals)() = nullptr;

        // Toolbar
        bool (*has_toolbar)() = nullptr;

        // Lifecycle
        void (*cleanup)() = nullptr;
        void (*prepare_ui)() = nullptr;
    };

    LFS_PYTHON_RUNTIME_API void set_bridge(const PyBridge& b);
    LFS_PYTHON_RUNTIME_API const PyBridge& bridge();

    // Keymap bindings for Python access
    LFS_PYTHON_RUNTIME_API void set_keymap_bindings(vis::input::InputBindings* bindings);
    LFS_PYTHON_RUNTIME_API vis::input::InputBindings* get_keymap_bindings();

    // Operation context for Python code (short-lived, per-call)
    LFS_PYTHON_RUNTIME_API void set_scene_for_python(vis::Scene* scene);
    LFS_PYTHON_RUNTIME_API vis::Scene* get_scene_for_python();

    LFS_PYTHON_RUNTIME_API void set_trainer_manager(vis::TrainerManager* tm);
    LFS_PYTHON_RUNTIME_API vis::TrainerManager* get_trainer_manager();

    LFS_PYTHON_RUNTIME_API void set_parameter_manager(vis::ParameterManager* pm);
    LFS_PYTHON_RUNTIME_API vis::ParameterManager* get_parameter_manager();

    LFS_PYTHON_RUNTIME_API void set_rendering_manager(vis::RenderingManager* rm);
    LFS_PYTHON_RUNTIME_API vis::RenderingManager* get_rendering_manager();

    LFS_PYTHON_RUNTIME_API void set_editor_context(vis::EditorContext* ec);
    LFS_PYTHON_RUNTIME_API vis::EditorContext* get_editor_context();

    // Separate setter for operator callbacks interface (EditorContext implements IOperatorCallbacks)
    // This allows python_runtime.cpp to dispatch callbacks without needing full EditorContext definition
    LFS_PYTHON_RUNTIME_API void set_operator_callbacks(core::IOperatorCallbacks* callbacks);

    LFS_PYTHON_RUNTIME_API void set_gui_manager(vis::gui::GuiManager* gm);
    LFS_PYTHON_RUNTIME_API vis::gui::GuiManager* get_gui_manager();

    // Scene panel state callbacks
    using GetSelectedCameraUidCallback = int (*)();
    using GetInvertMasksCallback = bool (*)();
    LFS_PYTHON_RUNTIME_API void set_selected_camera_callback(GetSelectedCameraUidCallback cb);
    LFS_PYTHON_RUNTIME_API void set_invert_masks_callback(GetInvertMasksCallback cb);
    LFS_PYTHON_RUNTIME_API int get_selected_camera_uid();
    LFS_PYTHON_RUNTIME_API bool get_invert_masks();

    // Sequencer visibility
    using IsSequencerVisibleCallback = bool (*)();
    using SetSequencerVisibleCallback = void (*)(bool);
    LFS_PYTHON_RUNTIME_API void set_sequencer_callbacks(IsSequencerVisibleCallback get_cb,
                                                        SetSequencerVisibleCallback set_cb);
    LFS_PYTHON_RUNTIME_API bool is_sequencer_visible();
    LFS_PYTHON_RUNTIME_API void set_sequencer_visible(bool visible);

    // Overlay state callbacks for Python overlay panels
    struct OverlayExportState {
        bool active = false;
        float progress = 0.0f;
        std::string stage;
        std::string format;
    };

    struct OverlayImportState {
        bool active = false;
        bool show_completion = false;
        float progress = 0.0f;
        std::string stage;
        std::string dataset_type;
        std::string path;
        bool success = false;
        std::string error;
        size_t num_images = 0;
        size_t num_points = 0;
        float seconds_since_completion = 0.0f;
    };

    struct OverlayVideoExportState {
        bool active = false;
        float progress = 0.0f;
        int current_frame = 0;
        int total_frames = 0;
        std::string stage;
    };

    using IsDragHoveringCallback = bool (*)();
    using IsStartupVisibleCallback = bool (*)();
    using GetExportStateCallback = OverlayExportState (*)();
    using CancelExportCallback = void (*)();
    using GetImportStateCallback = OverlayImportState (*)();
    using DismissImportCallback = void (*)();
    using GetVideoExportStateCallback = OverlayVideoExportState (*)();
    using CancelVideoExportCallback = void (*)();

    LFS_PYTHON_RUNTIME_API void set_overlay_callbacks(IsDragHoveringCallback drag_cb,
                                                      IsStartupVisibleCallback startup_cb,
                                                      GetExportStateCallback export_cb,
                                                      CancelExportCallback cancel_cb,
                                                      GetImportStateCallback import_cb,
                                                      DismissImportCallback dismiss_import_cb,
                                                      GetVideoExportStateCallback video_cb,
                                                      CancelVideoExportCallback cancel_video_cb);
    LFS_PYTHON_RUNTIME_API bool is_drag_hovering();
    LFS_PYTHON_RUNTIME_API bool is_startup_visible();
    LFS_PYTHON_RUNTIME_API OverlayExportState get_export_state();
    LFS_PYTHON_RUNTIME_API void cancel_export();
    LFS_PYTHON_RUNTIME_API OverlayImportState get_import_state();
    LFS_PYTHON_RUNTIME_API void dismiss_import();
    LFS_PYTHON_RUNTIME_API OverlayVideoExportState get_video_export_state();
    LFS_PYTHON_RUNTIME_API void cancel_video_export();

    // Sequencer timeline access callbacks
    using HasKeyframesCallback = bool (*)();
    using SaveCameraPathCallback = bool (*)(const std::string&);
    using LoadCameraPathCallback = bool (*)(const std::string&);
    using ClearKeyframesCallback = void (*)();
    using SetPlaybackSpeedCallback = void (*)(float);

    LFS_PYTHON_RUNTIME_API void set_sequencer_timeline_callbacks(
        HasKeyframesCallback has_keyframes_cb,
        SaveCameraPathCallback save_path_cb,
        LoadCameraPathCallback load_path_cb,
        ClearKeyframesCallback clear_cb,
        SetPlaybackSpeedCallback set_speed_cb);

    LFS_PYTHON_RUNTIME_API bool has_keyframes();
    LFS_PYTHON_RUNTIME_API bool save_camera_path(const std::string& path);
    LFS_PYTHON_RUNTIME_API bool load_camera_path(const std::string& path);
    LFS_PYTHON_RUNTIME_API void clear_keyframes();
    LFS_PYTHON_RUNTIME_API void set_playback_speed(float speed);

    // Menu bar UI callbacks (for Python-driven menus)
    using ShowInputSettingsCallback = void (*)();
    using ShowPythonConsoleCallback = void (*)();
    LFS_PYTHON_RUNTIME_API void set_show_input_settings_callback(ShowInputSettingsCallback cb);
    LFS_PYTHON_RUNTIME_API void set_show_python_console_callback(ShowPythonConsoleCallback cb);
    LFS_PYTHON_RUNTIME_API void show_input_settings();
    LFS_PYTHON_RUNTIME_API void show_python_console();

    // Section drawing callbacks (for Python-first UI)
    using DrawSectionCallback = void (*)();

    struct SectionDrawCallbacks {
        DrawSectionCallback draw_tools_section = nullptr;
        DrawSectionCallback draw_console_button = nullptr;
    };

    LFS_PYTHON_RUNTIME_API void set_section_draw_callbacks(const SectionDrawCallbacks& callbacks);
    LFS_PYTHON_RUNTIME_API void draw_tools_section();
    LFS_PYTHON_RUNTIME_API void draw_console_button();

    // Sequencer UI state access (for Python modification)
    // Must match layout of vis::gui::panels::SequencerUIState
    struct SequencerUIStateData {
        bool show_camera_path = true;
        bool snap_to_grid = false;
        float snap_interval = 0.5f;
        float playback_speed = 1.0f;
        bool follow_playback = false;
        float pip_preview_scale = 1.0f;
        int preset = 0;
        int custom_width = 1920;
        int custom_height = 1080;
        int framerate = 30;
        int quality = 18;
    };

    using GetSequencerUIStateCallback = SequencerUIStateData* (*)();
    LFS_PYTHON_RUNTIME_API void set_sequencer_ui_state_callback(GetSequencerUIStateCallback cb);
    LFS_PYTHON_RUNTIME_API SequencerUIStateData* get_sequencer_ui_state();

    // Pivot mode (for transform gizmos)
    using GetPivotModeCallback = int (*)();
    using SetPivotModeCallback = void (*)(int);
    LFS_PYTHON_RUNTIME_API void set_pivot_mode_callbacks(GetPivotModeCallback get_cb,
                                                         SetPivotModeCallback set_cb);
    LFS_PYTHON_RUNTIME_API int get_pivot_mode();
    LFS_PYTHON_RUNTIME_API void set_pivot_mode(int mode);

    // Transform space (local/world for transform gizmos)
    using GetTransformSpaceCallback = int (*)();
    using SetTransformSpaceCallback = void (*)(int);
    LFS_PYTHON_RUNTIME_API void set_transform_space_callbacks(GetTransformSpaceCallback get_cb,
                                                              SetTransformSpaceCallback set_cb);
    LFS_PYTHON_RUNTIME_API int get_transform_space();
    LFS_PYTHON_RUNTIME_API void set_transform_space(int space);

    LFS_PYTHON_RUNTIME_API void set_scene_manager(vis::SceneManager* sm);
    LFS_PYTHON_RUNTIME_API vis::SceneManager* get_scene_manager();

    LFS_PYTHON_RUNTIME_API void set_selection_service(vis::SelectionService* ss);
    LFS_PYTHON_RUNTIME_API vis::SelectionService* get_selection_service();

    // Viewport bounds for Python UI (set by gui_manager before drawing panels)
    LFS_PYTHON_RUNTIME_API void set_viewport_bounds(float x, float y, float w, float h);
    LFS_PYTHON_RUNTIME_API void get_viewport_bounds(float& x, float& y, float& w, float& h);
    LFS_PYTHON_RUNTIME_API bool has_viewport_bounds();

    // RAII guard for operation context (used for capability invocations)
    class SceneContextGuard {
    public:
        explicit SceneContextGuard(vis::Scene* scene) { set_scene_for_python(scene); }
        ~SceneContextGuard() { set_scene_for_python(nullptr); }

        SceneContextGuard(const SceneContextGuard&) = delete;
        SceneContextGuard& operator=(const SceneContextGuard&) = delete;
        SceneContextGuard(SceneContextGuard&&) = delete;
        SceneContextGuard& operator=(SceneContextGuard&&) = delete;
    };

    // Application scene context (long-lived, persists while model is loaded)
    class LFS_PYTHON_RUNTIME_API ApplicationSceneContext {
    public:
        void set(vis::Scene* scene);
        vis::Scene* get() const;
        uint64_t generation() const;

    private:
        std::atomic<vis::Scene*> scene_{nullptr};
        std::atomic<uint64_t> generation_{0};
    };

    LFS_PYTHON_RUNTIME_API void set_application_scene(vis::Scene* scene);
    LFS_PYTHON_RUNTIME_API vis::Scene* get_application_scene();
    LFS_PYTHON_RUNTIME_API uint64_t get_scene_generation();

    LFS_PYTHON_RUNTIME_API void set_gil_state_ready(bool ready);
    LFS_PYTHON_RUNTIME_API bool is_gil_state_ready();

    // Main thread GIL management - these live in the shared library
    // to ensure a single copy across all modules (exe and pyd)
    // Note: void* used to avoid Python.h dependency in header
    LFS_PYTHON_RUNTIME_API void set_main_thread_state(void* state);
    LFS_PYTHON_RUNTIME_API void* get_main_thread_state();
    LFS_PYTHON_RUNTIME_API void acquire_gil_main_thread();
    LFS_PYTHON_RUNTIME_API void release_gil_main_thread();

    // Initialization guards - must be in shared library to avoid duplication
    // These prevent double-init when static lib is linked into both exe and pyd
    LFS_PYTHON_RUNTIME_API void call_once_py_init(std::function<void()> fn);
    LFS_PYTHON_RUNTIME_API void call_once_redirect(std::function<void()> fn);
    LFS_PYTHON_RUNTIME_API void mark_plugins_loaded();
    LFS_PYTHON_RUNTIME_API bool are_plugins_loaded();

    // GL texture service — all GL calls happen in the EXE, Python DLL calls through these
    struct TextureResult {
        uint32_t texture_id;
        int width;
        int height;
    };

    using CreateTextureFn = TextureResult (*)(const unsigned char* data, int w, int h, int channels);
    using DeleteTextureFn = void (*)(uint32_t texture_id);
    using MaxTextureSizeFn = int (*)();

    LFS_PYTHON_RUNTIME_API void set_gl_texture_service(CreateTextureFn create, DeleteTextureFn del,
                                                       MaxTextureSizeFn max_size);
    LFS_PYTHON_RUNTIME_API TextureResult create_gl_texture(const unsigned char* data, int w, int h, int channels);
    LFS_PYTHON_RUNTIME_API void delete_gl_texture(uint32_t texture_id);
    LFS_PYTHON_RUNTIME_API int get_max_texture_size();

    // ImGui state sharing across DLL boundaries (void* to avoid imgui.h dependency)
    LFS_PYTHON_RUNTIME_API void set_imgui_context(void* ctx);
    LFS_PYTHON_RUNTIME_API void* get_imgui_context();
    LFS_PYTHON_RUNTIME_API void set_imgui_allocator_functions(void* alloc_func, void* free_func, void* user_data);
    LFS_PYTHON_RUNTIME_API void get_imgui_allocator_functions(void** alloc_func, void** free_func, void** user_data);

    LFS_PYTHON_RUNTIME_API void set_implot_context(void* ctx);
    LFS_PYTHON_RUNTIME_API void* get_implot_context();

    LFS_PYTHON_RUNTIME_API void set_view_context_state(void* state);
    LFS_PYTHON_RUNTIME_API void* get_view_context_state();
    LFS_PYTHON_RUNTIME_API void set_shared_dpi_scale(float scale);
    LFS_PYTHON_RUNTIME_API float get_shared_dpi_scale();

    // Exit popup state - thread-safe flag for window close callback
    LFS_PYTHON_RUNTIME_API bool is_exit_popup_open();
    LFS_PYTHON_RUNTIME_API void set_exit_popup_open(bool open);

    // Keyboard capture for Python popup windows
    LFS_PYTHON_RUNTIME_API void request_keyboard_capture(const std::string& owner_id);
    LFS_PYTHON_RUNTIME_API void release_keyboard_capture(const std::string& owner_id);
    LFS_PYTHON_RUNTIME_API bool has_keyboard_capture_request();

    // Poll cache invalidation - called when selection/training state changes
    // PollDependency values: NONE=0, SELECTION=1, TRAINING=2, SCENE=4, ALL=7
    using InvalidatePollCacheCallback = void (*)(uint8_t);
    LFS_PYTHON_RUNTIME_API void set_invalidate_poll_cache_callback(InvalidatePollCacheCallback cb);
    LFS_PYTHON_RUNTIME_API void invalidate_poll_caches(uint8_t dependency = 7);

    // Signal bridge callbacks - registered by Python module, called by visualizer
    using SignalFlushCallback = void (*)();
    using TrainingProgressCallback = void (*)(int iteration, float loss, std::size_t num_gaussians);
    using TrainingStateCallback = void (*)(bool is_training, const char* state);
    using TrainerLoadedCallback = void (*)(bool has_trainer, int max_iterations);
    using PsnrCallback = void (*)(float psnr);
    using SceneCallback = void (*)(bool has_scene, const char* path);
    using SelectionCallback = void (*)(bool has_selection, int count);

    struct SignalBridgeCallbacks {
        SignalFlushCallback flush = nullptr;
        TrainingProgressCallback training_progress = nullptr;
        TrainingStateCallback training_state = nullptr;
        TrainerLoadedCallback trainer_loaded = nullptr;
        PsnrCallback psnr = nullptr;
        SceneCallback scene = nullptr;
        SelectionCallback selection = nullptr;
    };

    LFS_PYTHON_RUNTIME_API void set_signal_bridge_callbacks(const SignalBridgeCallbacks& callbacks);

    // Signal update functions - called by visualizer, dispatch to Python via callbacks
    LFS_PYTHON_RUNTIME_API void update_training_progress(int iteration, float loss, std::size_t num_gaussians);
    LFS_PYTHON_RUNTIME_API void update_training_state(bool is_training, const char* state);
    LFS_PYTHON_RUNTIME_API void update_trainer_loaded(bool has_trainer, int max_iterations);
    LFS_PYTHON_RUNTIME_API void update_psnr(float psnr);
    LFS_PYTHON_RUNTIME_API void update_scene(bool has_scene, const char* path);
    LFS_PYTHON_RUNTIME_API void update_selection(bool has_selection, int count);
    LFS_PYTHON_RUNTIME_API void flush_signals();

} // namespace lfs::python
