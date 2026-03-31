/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/mesh2splat.hpp"
#include "core/modal_request.hpp"
#include "core/splat_simplify.hpp"
#include "visualizer/gui/panel_height_mode.hpp"
#include "visualizer/gui/panel_space.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
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
    struct MeshData;
    class Scene;
} // namespace lfs::core

namespace lfs::vis {
    class SceneManager;
    class TrainerManager;
    class ParameterManager;
    class RenderingManager;
    class EditorContext;
    class SelectionService;
    namespace gui {
        class GuiManager;
        class GlobalContextMenu;
    } // namespace gui
    namespace input {
        class InputBindings;
    }
} // namespace lfs::vis

namespace lfs::python {

    using PanelSpace = vis::gui::PanelSpace;
    using PanelHeightMode = vis::gui::PanelHeightMode;

    struct PyContext {
        // Managers (set directly)
        core::Scene* scene = nullptr;
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
    using MainLoopWakeCallback = void (*)();
    LFS_PYTHON_RUNTIME_API void set_main_loop_wake_callback(MainLoopWakeCallback cb);

    using CleanupCallback = void (*)();
    using EnsureInitializedCallback = void (*)();

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

    LFS_PYTHON_RUNTIME_API void invoke_python_cleanup();
    LFS_PYTHON_RUNTIME_API void shutdown_python_gl_resources();

    // Menu interface for C++ code
    LFS_PYTHON_RUNTIME_API void draw_python_menu_items(MenuLocation location);
    LFS_PYTHON_RUNTIME_API bool has_python_menu_items(MenuLocation location);

    // Menu bar entry interface for C++ UI code (Python-driven menu bar)
    LFS_PYTHON_RUNTIME_API bool has_menu_bar_entries();
    LFS_PYTHON_RUNTIME_API std::vector<MenuBarEntry> get_menu_bar_entries();
    LFS_PYTHON_RUNTIME_API void draw_menu_bar_entry(const std::string& idname);

    struct MenuItemInfo {
        int type;
        const char* label;
        const char* operator_id;
        const char* shortcut;
        bool enabled;
        bool selected;
        int callback_index;
    };
    using MenuItemVisitor = void (*)(const MenuItemInfo* item, void* user_data);

    LFS_PYTHON_RUNTIME_API void collect_menu_content(const std::string& idname,
                                                     MenuItemVisitor visitor, void* user_data);
    LFS_PYTHON_RUNTIME_API void execute_menu_callback(const std::string& idname, int callback_index);

    // Modal dialog callbacks
    using DrawModalsCallback = void (*)();
    using HasModalsCallback = bool (*)();

    LFS_PYTHON_RUNTIME_API void draw_python_modals(lfs::core::Scene* scene = nullptr);
    LFS_PYTHON_RUNTIME_API bool has_python_modals();

    // Modal enqueue callback - routes ModalRequests from PyModalRegistry to the overlay
    using ModalEnqueueCallback = std::function<void(lfs::core::ModalRequest)>;
    LFS_PYTHON_RUNTIME_API void set_modal_enqueue_callback(ModalEnqueueCallback cb);
    LFS_PYTHON_RUNTIME_API const ModalEnqueueCallback& get_modal_enqueue_callback();

    using DrawPopupsCallback = void (*)();
    LFS_PYTHON_RUNTIME_API void set_popup_draw_callback(DrawPopupsCallback cb);
    LFS_PYTHON_RUNTIME_API void draw_python_popups(lfs::core::Scene* scene = nullptr);

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
        int button;
        int action;

        // Key state
        int key;
        int mods;

        // Scroll
        double scroll_x, scroll_y;

        // GUI state - true if mouse is over ImGui window
        bool over_gui = false;
    };

    // Callback type: returns true if event was consumed by the operator
    using ModalEventCallback = bool (*)(const ModalEvent&);

    LFS_PYTHON_RUNTIME_API bool dispatch_modal_event(const ModalEvent& event);

    // Menu bar entry visitor callback
    using MenuBarEntryVisitor = void (*)(const char* idname, const char* label, int order, void* user_data);

    // Menu bar entry callbacks
    using HasMenuBarEntriesCallback = bool (*)();
    using GetMenuBarEntriesCallback = void (*)(MenuBarEntryVisitor, void* user_data);
    using DrawMenuBarEntryCallback = void (*)(const char* idname);

    struct PyBridge {
        // Menus
        void (*draw_menus)(MenuLocation) = nullptr;
        bool (*has_menus)(MenuLocation) = nullptr;

        // Menu bar entries (Python-driven top-level menus)
        bool (*has_menu_bar_entries)() = nullptr;
        void (*get_menu_bar_entries)(MenuBarEntryVisitor, void*) = nullptr;
        void (*draw_menu_bar_entry)(const char*) = nullptr;
        void (*collect_menu_content)(const char*, MenuItemVisitor, void*) = nullptr;
        void (*execute_menu_callback)(const char*, int) = nullptr;

        // Modals
        void (*draw_modals)() = nullptr;
        bool (*has_modals)() = nullptr;

        // Toolbar
        bool (*has_toolbar)() = nullptr;

        // Lifecycle
        void (*cleanup)() = nullptr;
        void (*prepare_ui)() = nullptr;
        void (*shutdown_gl_resources)() = nullptr;
    };

    LFS_PYTHON_RUNTIME_API void set_bridge(const PyBridge& b);
    LFS_PYTHON_RUNTIME_API const PyBridge& bridge();

    // Keymap bindings for Python access
    LFS_PYTHON_RUNTIME_API void set_keymap_bindings(vis::input::InputBindings* bindings);
    LFS_PYTHON_RUNTIME_API vis::input::InputBindings* get_keymap_bindings();

    // Operation context for Python code (short-lived, per-call)
    LFS_PYTHON_RUNTIME_API void set_scene_for_python(core::Scene* scene);
    LFS_PYTHON_RUNTIME_API core::Scene* get_scene_for_python();

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

    LFS_PYTHON_RUNTIME_API void set_global_context_menu(vis::gui::GlobalContextMenu* cm);
    LFS_PYTHON_RUNTIME_API vis::gui::GlobalContextMenu* get_global_context_menu();

    using Mesh2SplatStartFn = std::function<void(std::shared_ptr<core::MeshData>, std::string,
                                                 core::Mesh2SplatOptions)>;
    LFS_PYTHON_RUNTIME_API void set_mesh2splat_callbacks(
        Mesh2SplatStartFn start,
        std::function<bool()> is_active,
        std::function<float()> get_progress,
        std::function<std::string()> get_stage,
        std::function<std::string()> get_error);
    LFS_PYTHON_RUNTIME_API void invoke_mesh2splat_start(std::shared_ptr<core::MeshData> mesh, const std::string& name,
                                                        const core::Mesh2SplatOptions& options);
    LFS_PYTHON_RUNTIME_API bool invoke_mesh2splat_active();
    LFS_PYTHON_RUNTIME_API float invoke_mesh2splat_progress();
    LFS_PYTHON_RUNTIME_API std::string invoke_mesh2splat_stage();
    LFS_PYTHON_RUNTIME_API std::string invoke_mesh2splat_error();

    using SplatSimplifyStartFn = std::function<void(std::string, core::SplatSimplifyOptions)>;
    LFS_PYTHON_RUNTIME_API void set_splat_simplify_callbacks(
        SplatSimplifyStartFn start,
        std::function<void()> cancel,
        std::function<bool()> is_active,
        std::function<float()> get_progress,
        std::function<std::string()> get_stage,
        std::function<std::string()> get_error);
    LFS_PYTHON_RUNTIME_API void invoke_splat_simplify_start(const std::string& name,
                                                            const core::SplatSimplifyOptions& options);
    LFS_PYTHON_RUNTIME_API void invoke_splat_simplify_cancel();
    LFS_PYTHON_RUNTIME_API bool invoke_splat_simplify_active();
    LFS_PYTHON_RUNTIME_API float invoke_splat_simplify_progress();
    LFS_PYTHON_RUNTIME_API std::string invoke_splat_simplify_stage();
    LFS_PYTHON_RUNTIME_API std::string invoke_splat_simplify_error();

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
        DrawSectionCallback toggle_system_console = nullptr;
    };

    LFS_PYTHON_RUNTIME_API void set_section_draw_callbacks(const SectionDrawCallbacks& callbacks);
    LFS_PYTHON_RUNTIME_API void draw_tools_section();
    LFS_PYTHON_RUNTIME_API void draw_console_button();
    LFS_PYTHON_RUNTIME_API void toggle_system_console();

    // Sequencer UI state access (for Python modification)
    // Must match layout of vis::gui::panels::SequencerUIState
    struct SequencerUIStateData {
        bool show_camera_path = true;
        bool snap_to_grid = false;
        float snap_interval = 0.5f;
        float playback_speed = 1.0f;
        bool follow_playback = false;
        bool show_pip_preview = true;
        float pip_preview_scale = 1.0f;
        bool show_film_strip = true;
        bool equirectangular = false;
        int preset = 0;
        int custom_width = 1920;
        int custom_height = 1080;
        int framerate = 30;
        int quality = 18;
        int selected_keyframe = -1;
    };

    using GetSequencerUIStateCallback = std::function<SequencerUIStateData*()>;
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
        explicit SceneContextGuard(core::Scene* scene) { set_scene_for_python(scene); }
        ~SceneContextGuard() { set_scene_for_python(nullptr); }

        SceneContextGuard(const SceneContextGuard&) = delete;
        SceneContextGuard& operator=(const SceneContextGuard&) = delete;
        SceneContextGuard(SceneContextGuard&&) = delete;
        SceneContextGuard& operator=(SceneContextGuard&&) = delete;
    };

    // Application scene context (long-lived, persists while model is loaded)
    class LFS_PYTHON_RUNTIME_API ApplicationSceneContext {
    public:
        void set(core::Scene* scene);
        core::Scene* get() const;
        uint64_t generation() const;
        uint32_t mutation_flags() const;
        uint32_t consume_mutation_flags();
        void bump();
        void set_mutation_flags(uint32_t flags);

    private:
        std::atomic<core::Scene*> scene_{nullptr};
        std::atomic<uint64_t> generation_{0};
        std::atomic<uint32_t> mutation_flags_{0};
    };

    LFS_PYTHON_RUNTIME_API void set_application_scene(core::Scene* scene);
    LFS_PYTHON_RUNTIME_API core::Scene* get_application_scene();
    LFS_PYTHON_RUNTIME_API uint64_t get_scene_generation();
    LFS_PYTHON_RUNTIME_API uint32_t get_scene_mutation_flags();
    LFS_PYTHON_RUNTIME_API uint32_t consume_scene_mutation_flags();
    LFS_PYTHON_RUNTIME_API void bump_scene_generation();
    LFS_PYTHON_RUNTIME_API void set_scene_mutation_flags(uint32_t flags);

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

    LFS_PYTHON_RUNTIME_API void set_rml_manager(void* manager);
    LFS_PYTHON_RUNTIME_API void* get_rml_manager();

    using RmlContextDestroyFn = void (*)(void* context);
    LFS_PYTHON_RUNTIME_API void set_rml_context_destroy_handler(RmlContextDestroyFn fn);
    LFS_PYTHON_RUNTIME_API RmlContextDestroyFn get_rml_context_destroy_handler();

    // RmlPanelHost opaque operations — function pointers set by the exe
    // to avoid linking RmlUI code into the Python module (.so)
    struct PanelDrawContext;

    struct RmlPanelHostOps {
        void* (*create)(void* manager, const char* context_name, const char* rml_path,
                        const char* inline_rcss);
        void (*destroy)(void* host);
        void (*draw)(void* host, const void* draw_ctx);
        void (*draw_direct)(void* host, float x, float y, float w, float h);
        void (*prepare_direct)(void* host, float w, float h);
        void (*prepare_layout)(void* host, float w, float h);
        void* (*get_document)(void* host);
        bool (*is_loaded)(void* host);
        void (*set_height_mode)(void* host, int mode);
        float (*get_content_height)(void* host);
        bool (*ensure_context)(void* host);
        bool (*ensure_document)(void* host);
        bool (*reload_document)(void* host);
        void* (*get_context)(void* host);
        void (*set_foreground)(void* host, bool fg);
        void (*mark_content_dirty)(void* host);
        void (*set_input_clip_y)(void* host, float y_min, float y_max);
        void (*set_input)(void* host, const void* input);
        void (*set_forced_height)(void* host, float h);
        bool (*needs_animation)(void* host);
    };

    LFS_PYTHON_RUNTIME_API void set_rml_panel_host_ops(const RmlPanelHostOps& ops);
    LFS_PYTHON_RUNTIME_API const RmlPanelHostOps& get_rml_panel_host_ops();

    using RmlDocRegisterCallback = void (*)(const char* name, void* doc);
    using RmlDocUnregisterCallback = void (*)(const char* name);

    LFS_PYTHON_RUNTIME_API void set_rml_doc_registry_callbacks(RmlDocRegisterCallback reg_cb,
                                                               RmlDocUnregisterCallback unreg_cb);
    LFS_PYTHON_RUNTIME_API void register_rml_document(const char* name, void* doc);
    LFS_PYTHON_RUNTIME_API void unregister_rml_document(const char* name);

    // GL-thread callback queue — schedule work that requires the GL context
    LFS_PYTHON_RUNTIME_API void set_gl_thread_id(std::thread::id id);
    LFS_PYTHON_RUNTIME_API bool on_gl_thread();
    LFS_PYTHON_RUNTIME_API void schedule_gl_callback(std::function<void()> fn);
    LFS_PYTHON_RUNTIME_API void flush_gl_callbacks();

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

    // Viewport draw overlay - bridge from visualizer to Python draw handlers
    // view_matrix/proj_matrix: column-major 4x4, others: float arrays
    // draw_list: opaque pointer to ImDrawList (cast by implementation)
    using HasViewportDrawHandlersCallback = bool (*)();
    using InvokeViewportOverlayCallback = void (*)(const float* view_matrix, const float* proj_matrix,
                                                   const float* vp_pos, const float* vp_size,
                                                   const float* cam_pos, const float* cam_fwd,
                                                   void* draw_list);

    LFS_PYTHON_RUNTIME_API void set_viewport_overlay_callbacks(HasViewportDrawHandlersCallback has_cb,
                                                               InvokeViewportOverlayCallback invoke_cb);
    LFS_PYTHON_RUNTIME_API bool has_viewport_draw_handlers();
    LFS_PYTHON_RUNTIME_API void invoke_viewport_overlay(const float* view_matrix, const float* proj_matrix,
                                                        const float* vp_pos, const float* vp_size,
                                                        const float* cam_pos, const float* cam_fwd,
                                                        void* draw_list);

} // namespace lfs::python
