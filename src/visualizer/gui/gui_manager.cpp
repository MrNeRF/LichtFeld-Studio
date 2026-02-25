/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// glad must be included before OpenGL headers
// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/gui_manager.hpp"
#include "control/command_api.hpp"
#include "core/cuda_version.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/native_panels.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/mesh2splat_panel.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "gui/windows/file_browser.hpp"
#include "io/video_frame_extractor.hpp"
#include <implot.h>

#include "input/input_controller.hpp"
#include "internal/resource_paths.hpp"
#include "tools/align_tool.hpp"

#include "core/events.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "python/package_manager.hpp"
#include "python/python_runtime.hpp"
#include "python/ui_hooks.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "tools/brush_tool.hpp"
#include "tools/selection_tool.hpp"
#include "visualizer_impl.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl3.h>
#include <imgui_internal.h>
#include <ImGuizmo.h>

namespace lfs::vis::gui {

    GuiManager::GuiManager(VisualizerImpl* viewer)
        : viewer_(viewer),
          sequencer_ui_(viewer, sequencer_ui_state_),
          gizmo_manager_(viewer),
          async_tasks_(viewer) {

        panel_layout_.loadState();

        // Create components
        file_browser_ = std::make_unique<FileBrowser>();
        menu_bar_ = std::make_unique<MenuBar>();
        disk_space_error_dialog_ = std::make_unique<DiskSpaceErrorDialog>();
        video_extractor_dialog_ = std::make_unique<lfs::gui::VideoExtractorDialog>();

        // Wire up video extractor dialog callback
        video_extractor_dialog_->setOnStartExtraction([this](const lfs::gui::VideoExtractionParams& params) {
            if (video_extraction_thread_ && video_extraction_thread_->joinable())
                video_extraction_thread_->join();

            auto* dialog = video_extractor_dialog_.get();
            video_extraction_thread_.emplace([params, dialog]() {
                lfs::io::VideoFrameExtractor extractor;

                lfs::io::VideoFrameExtractor::Params extract_params;
                extract_params.video_path = params.video_path;
                extract_params.output_dir = params.output_dir;
                extract_params.mode = params.mode;
                extract_params.fps = params.fps;
                extract_params.frame_interval = params.frame_interval;
                extract_params.format = params.format;
                extract_params.jpg_quality = params.jpg_quality;
                extract_params.start_time = params.start_time;
                extract_params.end_time = params.end_time;
                extract_params.resolution_mode = params.resolution_mode;
                extract_params.scale = params.scale;
                extract_params.custom_width = params.custom_width;
                extract_params.custom_height = params.custom_height;
                extract_params.filename_pattern = params.filename_pattern;

                extract_params.progress_callback = [dialog](int current, int total) {
                    dialog->updateProgress(current, total);
                };

                std::string error;
                if (!extractor.extract(extract_params, error)) {
                    LOG_ERROR("Video frame extraction failed: {}", error);
                    dialog->setExtractionError(error);
                } else {
                    LOG_INFO("Video frame extraction completed successfully");
                    dialog->setExtractionComplete();
                }
            });
        });

        // Initialize window states
        window_states_["file_browser"] = false;
        window_states_["scene_panel"] = true;
        window_states_["system_console"] = false;
        window_states_["training_tab"] = false;
        window_states_["export_dialog"] = false;
        window_states_["python_console"] = false;

        setupEventHandlers();
        async_tasks_.setupEvents();
        sequencer_ui_.setupEvents();
        gizmo_manager_.setupEvents();
        checkCudaVersionAndNotify();
    }

    void GuiManager::checkCudaVersionAndNotify() {
        using namespace lfs::core;
        const auto info = check_cuda_version();
        if (!info.query_failed && !info.supported) {
            constexpr int MIN_MAJOR = MIN_CUDA_VERSION / 1000;
            constexpr int MIN_MINOR = (MIN_CUDA_VERSION % 1000) / 10;
            events::state::CudaVersionUnsupported{
                .major = info.major,
                .minor = info.minor,
                .min_major = MIN_MAJOR,
                .min_minor = MIN_MINOR}
                .emit();
        }
    }

    GuiManager::~GuiManager() = default;

    void GuiManager::initMenuBar() {
        menu_bar_->setOnShowPythonConsole([this]() {
            window_states_["python_console"] = !window_states_["python_console"];
        });
    }

    FontSet GuiManager::buildFontSet() const {
        FontSet fs{font_regular_, font_bold_, font_heading_, font_small_, font_section_, font_monospace_};
        for (int i = 0; i < FontSet::MONO_SIZE_COUNT; ++i) {
            fs.monospace_sized[i] = mono_fonts_[i];
            fs.monospace_sizes[i] = mono_font_scales_[i];
        }
        return fs;
    }

    void GuiManager::init() {
        // ImGui initialization
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();

        // Share ImGui state with Python module across DLL boundaries
        ImGuiContext* const ctx = ImGui::GetCurrentContext();
        lfs::python::set_imgui_context(ctx);

        ImGuiMemAllocFunc alloc_fn{};
        ImGuiMemFreeFunc free_fn{};
        void* alloc_user_data{};
        ImGui::GetAllocatorFunctions(&alloc_fn, &free_fn, &alloc_user_data);
        lfs::python::set_imgui_allocator_functions(
            reinterpret_cast<void*>(alloc_fn),
            reinterpret_cast<void*>(free_fn),
            alloc_user_data);
        lfs::python::set_implot_context(ImPlot::GetCurrentContext());

        lfs::python::set_gl_texture_service(
            [](const unsigned char* data, const int w, const int h, const int channels) -> lfs::python::TextureResult {
                if (!data || w <= 0 || h <= 0)
                    return {0, 0, 0};

                GLuint tex = 0;
                glGenTextures(1, &tex);
                if (tex == 0)
                    return {0, 0, 0};

                glBindTexture(GL_TEXTURE_2D, tex);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

                GLenum format = GL_RGB;
                GLenum internal_format = GL_RGB8;
                if (channels == 1) {
                    format = GL_RED;
                    internal_format = GL_R8;
                } else if (channels == 4) {
                    format = GL_RGBA;
                    internal_format = GL_RGBA8;
                }

                glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, format, GL_UNSIGNED_BYTE, data);

                if (channels == 1) {
                    GLint swizzle[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
                    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle);
                }

                glBindTexture(GL_TEXTURE_2D, 0);
                return {tex, w, h};
            },
            [](const uint32_t tex) {
                if (tex > 0) {
                    const auto gl_tex = static_cast<GLuint>(tex);
                    glDeleteTextures(1, &gl_tex);
                }
            },
            []() -> int {
                constexpr int FALLBACK_MAX_TEXTURE_SIZE = 4096;
                GLint sz = 0;
                glGetIntegerv(GL_MAX_TEXTURE_SIZE, &sz);
                return sz > 0 ? sz : FALLBACK_MAX_TEXTURE_SIZE;
            });

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
        io.ConfigWindowsMoveFromTitleBarOnly = true;

        // Platform/Renderer initialization
        ImGui_ImplSDL3_InitForOpenGL(viewer_->getWindow(), SDL_GL_GetCurrentContext());
        ImGui_ImplOpenGL3_Init("#version 430");

        // Initialize localization system
        auto& loc = lfs::event::LocalizationManager::getInstance();
        const std::string locale_path = lfs::core::path_to_utf8(lfs::core::getLocalesDir());
        if (!loc.initialize(locale_path)) {
            LOG_WARN("Failed to initialize localization system, using default strings");
        } else {
            LOG_INFO("Localization initialized with language: {}", loc.getCurrentLanguageName());
        }

        float xscale = SDL_GetWindowDisplayScale(viewer_->getWindow());

        // Clamping / safety net for weird DPI values
        // Support up to 4.0x scale for high-DPI displays (e.g., 6K monitors)
        xscale = std::clamp(xscale, 1.0f, 4.0f);

        // Store DPI scale for use by UI components
        lfs::python::set_shared_dpi_scale(xscale);
        lfs::vis::setThemeDpiScale(xscale);

        // Set application icon - use the resource path helper
        try {
            const auto icon_path = lfs::vis::getAssetPath("lichtfeld-icon.png");
            const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(icon_path);

            SDL_Surface* icon_surface = SDL_CreateSurfaceFrom(width, height, SDL_PIXELFORMAT_RGBA32, data, width * 4);
            if (icon_surface) {
                SDL_SetWindowIcon(viewer_->getWindow(), icon_surface);
                SDL_DestroySurface(icon_surface);
            }
            lfs::core::free_image(data);
        } catch (const std::exception& e) {
            LOG_WARN("Could not load application icon: {}", e.what());
        }

        // Apply theme first to get font settings
        applyDefaultStyle();

        // Load fonts
        const auto& t = theme();
        try {
            const auto regular_path = lfs::vis::getAssetPath("fonts/" + t.fonts.regular_path);
            const auto bold_path = lfs::vis::getAssetPath("fonts/" + t.fonts.bold_path);
            const auto japanese_path = lfs::vis::getAssetPath("fonts/NotoSansJP-Regular.ttf");
            const auto korean_path = lfs::vis::getAssetPath("fonts/NotoSansKR-Regular.ttf");

            // Helper to check if font file is valid
            const auto is_font_valid = [](const std::filesystem::path& path) -> bool {
                constexpr size_t MIN_FONT_FILE_SIZE = 100;
                return std::filesystem::exists(path) && std::filesystem::file_size(path) >= MIN_FONT_FILE_SIZE;
            };

            // Latin-only font loader for bold/heading/small/section
            const auto load_font_latin_only =
                [&](const std::filesystem::path& path, const float size) -> ImFont* {
                if (!is_font_valid(path)) {
                    LOG_WARN("Font file invalid: {}", lfs::core::path_to_utf8(path));
                    return nullptr;
                }
                const std::string path_utf8 = lfs::core::path_to_utf8(path);
                return io.Fonts->AddFontFromFileTTF(path_utf8.c_str(), size);
            };

            const auto merge_cjk = [&](const float size) {
                if (is_font_valid(japanese_path)) {
                    ImFontConfig config;
                    config.MergeMode = true;
                    config.OversampleH = 1;
                    const std::string japanese_path_utf8 = lfs::core::path_to_utf8(japanese_path);
                    io.Fonts->AddFontFromFileTTF(japanese_path_utf8.c_str(), size, &config,
                                                 io.Fonts->GetGlyphRangesJapanese());
                    io.Fonts->AddFontFromFileTTF(japanese_path_utf8.c_str(), size, &config,
                                                 io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
                }

                if (is_font_valid(korean_path)) {
                    ImFontConfig config;
                    config.MergeMode = true;
                    config.OversampleH = 1;
                    const std::string korean_path_utf8 = lfs::core::path_to_utf8(korean_path);
                    io.Fonts->AddFontFromFileTTF(korean_path_utf8.c_str(), size, &config,
                                                 io.Fonts->GetGlyphRangesKorean());
                }
            };

            const auto load_font_with_cjk =
                [&](const std::filesystem::path& path, const float size) -> ImFont* {
                ImFont* font = load_font_latin_only(path, size);
                if (!font)
                    return nullptr;
                merge_cjk(size);
                return font;
            };

            font_regular_ = load_font_with_cjk(regular_path, t.fonts.base_size * xscale);
            font_bold_ = load_font_with_cjk(bold_path, t.fonts.base_size * xscale);
            font_heading_ = load_font_with_cjk(bold_path, t.fonts.heading_size * xscale);
            font_small_ = load_font_with_cjk(regular_path, t.fonts.small_size * xscale);
            font_section_ = load_font_with_cjk(bold_path, t.fonts.section_size * xscale);

            // Monospace font at multiple sizes for crisp scaling
            const auto monospace_path = lfs::vis::getAssetPath("fonts/JetBrainsMono-Regular.ttf");
            if (is_font_valid(monospace_path)) {
                const std::string mono_path_utf8 = lfs::core::path_to_utf8(monospace_path);

                static constexpr ImWchar GLYPH_RANGES[] = {
                    0x0020,
                    0x00FF, // Basic Latin + Latin Supplement
                    0x2190,
                    0x21FF, // Arrows
                    0x2500,
                    0x257F, // Box Drawing
                    0x2580,
                    0x259F, // Block Elements
                    0x25A0,
                    0x25FF, // Geometric Shapes
                    0,
                };

                static constexpr float MONO_SCALES[] = {0.7f, 1.0f, 1.3f, 1.7f, 2.2f};
                static_assert(std::size(MONO_SCALES) == FontSet::MONO_SIZE_COUNT);

                for (int i = 0; i < FontSet::MONO_SIZE_COUNT; ++i) {
                    ImFontConfig config;
                    config.GlyphRanges = GLYPH_RANGES;
                    const float size = t.fonts.base_size * xscale * MONO_SCALES[i];
                    mono_fonts_[i] = io.Fonts->AddFontFromFileTTF(mono_path_utf8.c_str(), size, &config);
                    mono_font_scales_[i] = MONO_SCALES[i];
                }
                font_monospace_ = mono_fonts_[1];
                if (font_monospace_) {
                    LOG_INFO("Loaded monospace font: JetBrainsMono-Regular.ttf ({} sizes)", FontSet::MONO_SIZE_COUNT);
                }
            }
            if (!font_monospace_) {
                font_monospace_ = font_regular_;
                LOG_WARN("Monospace font not found, using regular font for code editor");
            }

            const bool all_loaded = font_regular_ && font_bold_ && font_heading_ && font_small_ && font_section_;
            if (!all_loaded) {
                LOG_WARN("Some fonts failed to load, using fallback");
                ImFont* const fallback = font_regular_ ? font_regular_ : io.Fonts->AddFontDefault();
                if (!font_regular_)
                    font_regular_ = fallback;
                if (!font_bold_)
                    font_bold_ = fallback;
                if (!font_heading_)
                    font_heading_ = fallback;
                if (!font_small_)
                    font_small_ = fallback;
                if (!font_section_)
                    font_section_ = fallback;
            } else {
                LOG_INFO("Loaded fonts: {} and {}", t.fonts.regular_path, t.fonts.bold_path);
                if (is_font_valid(japanese_path)) {
                    LOG_INFO("Japanese + Chinese font support enabled");
                }
                if (is_font_valid(korean_path)) {
                    LOG_INFO("Korean font support enabled");
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Font loading failed: {}", e.what());
            ImFont* const fallback = io.Fonts->AddFontDefault();
            font_regular_ = font_bold_ = font_heading_ = font_small_ = font_section_ = fallback;
        }

        io.Fonts->TexDesiredWidth = 2048;
        if (!io.Fonts->Build()) {
            LOG_ERROR("Font atlas build failed — CJK glyphs may be missing");
        }
        ImGui_ImplOpenGL3_CreateFontsTexture();

        setFileSelectedCallback([this](const std::filesystem::path& path, const bool is_dataset) {
            window_states_["file_browser"] = false;
            if (is_dataset) {
                lfs::core::events::cmd::ShowDatasetLoadPopup{.dataset_path = path}.emit();
            } else {
                lfs::core::events::cmd::LoadFile{.path = path, .is_dataset = false}.emit();
            }
        });

        initMenuBar();
        menu_bar_->setFonts(buildFontSet());

        startup_overlay_.loadTextures();

        if (!drag_drop_.init(viewer_->getWindow())) {
            LOG_WARN("Native drag-drop initialization failed, drag-drop will use SDL events only");
        }
        drag_drop_.setFileDropCallback([this](const std::vector<std::string>& paths) {
            LOG_INFO("Files dropped via native drag-drop: {} file(s)", paths.size());
            if (auto* const ic = viewer_->getInputController()) {
                ic->handleFileDrop(paths);
            } else {
                LOG_ERROR("InputController not available for file drop handling");
            }
        });

        registerNativePanels();
    }

    void GuiManager::shutdown() {
        lfs::python::shutdown_python_gl_resources();
        panel_layout_.saveState();

        if (video_extraction_thread_ && video_extraction_thread_->joinable())
            video_extraction_thread_->join();
        video_extraction_thread_.reset();

        async_tasks_.shutdown();

        sequencer_ui_.destroyGLResources();
        startup_overlay_.destroyTextures();
        drag_drop_.shutdown();

        if (ImGui::GetCurrentContext()) {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplSDL3_Shutdown();
            ImPlot::DestroyContext();
            ImGui::DestroyContext();
        }
    }

    void GuiManager::registerNativePanels() {
        using namespace native_panels;
        auto& reg = PanelRegistry::instance();

        auto make_panel = [this](auto panel) -> std::shared_ptr<IPanel> {
            auto ptr = std::make_shared<decltype(panel)>(std::move(panel));
            native_panel_storage_.push_back(ptr);
            return ptr;
        };

        auto reg_panel = [&](const std::string& idname, const std::string& label,
                             std::shared_ptr<IPanel> panel, PanelSpace space, int order,
                             uint32_t options = 0, float initial_width = 0, float initial_height = 0) {
            PanelInfo info;
            info.panel = std::move(panel);
            info.label = label;
            info.idname = idname;
            info.space = space;
            info.order = order;
            info.options = options;
            info.is_native = true;
            info.initial_width = initial_width;
            info.initial_height = initial_height;
            reg.register_panel(std::move(info));
        };

        constexpr uint32_t SELF = static_cast<uint32_t>(PanelOption::SELF_MANAGED);

        // Floating panels (self-managed windows)
        reg_panel("native.file_browser", "File Browser",
                  make_panel(FileBrowserPanel(file_browser_.get(), &window_states_["file_browser"])),
                  PanelSpace::Floating, 10, SELF);

        reg_panel("native.video_extractor", "Video Extractor",
                  make_panel(VideoExtractorPanel(video_extractor_dialog_.get())),
                  PanelSpace::Floating, 11, 0, 750.0f);
        reg.set_panel_enabled("native.video_extractor", false);

        reg_panel("native.disk_space_error", "Disk Space Error",
                  make_panel(DiskSpaceErrorPanel(disk_space_error_dialog_.get())),
                  PanelSpace::Floating, 900, SELF);

        reg_panel("native.mesh2splat", "Mesh to Splat",
                  make_panel(panels::Mesh2SplatPanel(viewer_)),
                  PanelSpace::Floating, 12, 0, 400.0f);
        reg.set_panel_enabled("native.mesh2splat", false);

        // Viewport overlays (ordered by draw priority)
        reg_panel("native.selection_overlay", "Selection Overlay",
                  make_panel(SelectionOverlayPanel(this)),
                  PanelSpace::ViewportOverlay, 200);

        reg_panel("native.node_transform_gizmo", "Node Transform",
                  make_panel(NodeTransformGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 300);

        reg_panel("native.cropbox_gizmo", "Crop Box",
                  make_panel(CropBoxGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 301);

        reg_panel("native.ellipsoid_gizmo", "Ellipsoid",
                  make_panel(EllipsoidGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 302);

        reg_panel("native.sequencer", "Sequencer",
                  make_panel(SequencerPanel(&sequencer_ui_, &panel_layout_)),
                  PanelSpace::ViewportOverlay, 500);

        reg_panel("native.python_overlay", "Python Overlay",
                  make_panel(PythonOverlayPanel(this)),
                  PanelSpace::ViewportOverlay, 500);

        reg_panel("native.viewport_decorations", "Viewport Decorations",
                  make_panel(ViewportDecorationsPanel(this)),
                  PanelSpace::ViewportOverlay, 800);

        reg_panel("native.viewport_gizmo", "Viewport Gizmo",
                  make_panel(ViewportGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 900);

        reg_panel("native.pie_menu", "Pie Menu",
                  make_panel(PieMenuPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 950);

        reg_panel("native.startup_overlay", "Startup Overlay",
                  make_panel(StartupOverlayPanel(&startup_overlay_, font_small_, &drag_drop_hovering_)),
                  PanelSpace::ViewportOverlay, 1000);
    }

    void GuiManager::render() {
        drag_drop_.pollEvents();
        drag_drop_hovering_ = drag_drop_.isDragHovering();

        // Start frame
        ImGui_ImplOpenGL3_NewFrame();

        ImGui_ImplSDL3_NewFrame();

        // Check mouse state before ImGui::NewFrame() updates WantCaptureMouse
        ImVec2 mouse_pos = ImGui::GetMousePos();
        bool mouse_in_viewport = isPositionInViewport(mouse_pos.x, mouse_pos.y);

        ImGui::NewFrame();

        if (ImGui::IsKeyPressed(ImGuiKey_Escape) && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) {
            ImGui::ClearActiveID();
            if (auto* editor = panels::PythonConsoleState::getInstance().getEditor()) {
                editor->unfocus();
            }
        }

        // Check for async import completion (must happen on main thread)
        async_tasks_.pollImportCompletion();
        async_tasks_.pollMesh2SplatCompletion();

        // Poll UV package manager for async operations
        python::PackageManager::instance().poll();

        // Hot-reload themes (check once per second)
        {
            static auto last_check = std::chrono::steady_clock::now();
            const auto now = std::chrono::steady_clock::now();
            if (now - last_check > std::chrono::seconds(1)) {
                checkThemeFileChanges();
                last_check = now;
            }
        }

        // Initialize ImGuizmo for this frame
        ImGuizmo::BeginFrame();

        if (menu_bar_ && !ui_hidden_) {
            menu_bar_->render();
        }

        updateInputOverrides(mouse_in_viewport);

        // Create main dockspace
        const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(main_viewport->WorkPos);
        ImGui::SetNextWindowSize(main_viewport->WorkSize);
        ImGui::SetNextWindowViewport(main_viewport->ID);

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking |
                                        ImGuiWindowFlags_NoTitleBar |
                                        ImGuiWindowFlags_NoCollapse |
                                        ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove |
                                        ImGuiWindowFlags_NoBringToFrontOnFocus |
                                        ImGuiWindowFlags_NoNavFocus |
                                        ImGuiWindowFlags_NoBackground;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        ImGui::Begin("DockSpace", nullptr, window_flags);
        ImGui::PopStyleVar(3);

        // DockSpace ID
        ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");

        // Create dockspace
        ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

        ImGui::End();

        // Update editor context state for this frame
        auto& editor_ctx = viewer_->getEditorContext();
        editor_ctx.update(viewer_->getSceneManager(), viewer_->getTrainerManager());

        // Create context for this frame
        UIContext ctx{
            .viewer = viewer_,
            .file_browser = file_browser_.get(),
            .window_states = &window_states_,
            .editor = &editor_ctx,
            .sequencer_controller = &sequencer_ui_.controller(),
            .fonts = buildFontSet()};

        // Build draw context for panel registry
        lfs::core::Scene* scene = nullptr;
        if (auto* sm = ctx.viewer->getSceneManager()) {
            scene = &sm->getScene();
        }
        PanelDrawContext draw_ctx;
        draw_ctx.ui = &ctx;
        draw_ctx.viewport = &viewport_layout_;
        draw_ctx.scene = scene;
        draw_ctx.ui_hidden = ui_hidden_;
        draw_ctx.scene_generation = python::get_scene_generation();
        if (auto* sm = ctx.viewer->getSceneManager())
            draw_ctx.has_selection = sm->hasSelectedNode();
        if (auto* cc = lfs::event::command_center())
            draw_ctx.is_training = cc->snapshot().is_running;

        auto& reg = PanelRegistry::instance();

        panel_layout_.renderRightPanel(ctx, draw_ctx, show_main_panel_, ui_hidden_, window_states_, focus_panel_name_);

        python::set_viewport_bounds(viewport_layout_.pos.x, viewport_layout_.pos.y,
                                    viewport_layout_.size.x, viewport_layout_.size.y);

        reg.draw_panels(PanelSpace::Floating, draw_ctx);
        reg.draw_panels(PanelSpace::Dockable, draw_ctx);

        gizmo_manager_.updateToolState(ctx, ui_hidden_);
        gizmo_manager_.updateCropFlash();

        reg.draw_panels(PanelSpace::ViewportOverlay, draw_ctx);

        // Recompute viewport layout
        viewport_layout_ = panel_layout_.computeViewportLayout(
            show_main_panel_, ui_hidden_, window_states_["python_console"]);

        if (!ui_hidden_) {
            reg.draw_panels(PanelSpace::StatusBar, draw_ctx);
        }

        python::draw_python_modals(scene);
        python::draw_python_popups(scene);

        // Notification popups are rendered via PyModalRegistry (draw_modals in Python bridge)

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Clean up GL state after ImGui rendering (ImGui can leave VAO/shader bindings corrupted)
        glBindVertexArray(0);
        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        // Clear any errors ImGui might have generated
        while (glGetError() != GL_NO_ERROR) {}

        // Update and Render additional Platform Windows (for multi-viewport)
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            SDL_Window* backup_window = SDL_GL_GetCurrentWindow();
            SDL_GLContext backup_context = SDL_GL_GetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            SDL_GL_MakeCurrent(backup_window, backup_context);

            // Clean up GL state after multi-viewport rendering too
            glBindVertexArray(0);
            glUseProgram(0);
            glBindTexture(GL_TEXTURE_2D, 0);
            while (glGetError() != GL_NO_ERROR) {}
        }
    }

    void GuiManager::renderSelectionOverlays(const UIContext& ctx) {
        if (auto* const tool = ctx.viewer->getBrushTool(); tool && tool->isEnabled() && !ui_hidden_) {
            tool->renderUI(ctx, nullptr);
        }
        if (auto* const tool = ctx.viewer->getSelectionTool(); tool && tool->isEnabled() && !ui_hidden_) {
            tool->renderUI(ctx, nullptr);
        }

        const bool mouse_over_ui = ImGui::GetIO().WantCaptureMouse;
        if (!ui_hidden_ && !mouse_over_ui && viewport_layout_.size.x > 0 && viewport_layout_.size.y > 0) {
            auto* rm = ctx.viewer->getRenderingManager();
            auto* draw_list = ImGui::GetForegroundDrawList();

            if (rm && rm->isBrushActive()) {
                const auto& t = theme();
                float bx, by, br;
                bool add_mode;
                rm->getBrushState(bx, by, br, add_mode);

                const float render_scale = rm->getSettings().render_scale;
                const ImVec2 screen_pos(viewport_layout_.pos.x + bx / render_scale,
                                        viewport_layout_.pos.y + by / render_scale);
                const float screen_radius = br / render_scale;

                const ImU32 brush_color = add_mode
                                              ? toU32WithAlpha(t.palette.success, 0.8f)
                                              : toU32WithAlpha(t.palette.error, 0.8f);
                draw_list->AddCircle(screen_pos, screen_radius, brush_color, 32, 2.0f);
                draw_list->AddCircleFilled(screen_pos, 3.0f, brush_color);
            }

            if (rm && rm->isRectPreviewActive()) {
                const auto& t = theme();
                float rx0, ry0, rx1, ry1;
                bool add_mode;
                rm->getRectPreview(rx0, ry0, rx1, ry1, add_mode);

                const float render_scale = rm->getSettings().render_scale;
                const ImVec2 p0(viewport_layout_.pos.x + rx0 / render_scale, viewport_layout_.pos.y + ry0 / render_scale);
                const ImVec2 p1(viewport_layout_.pos.x + rx1 / render_scale, viewport_layout_.pos.y + ry1 / render_scale);

                const ImU32 fill_color = add_mode
                                             ? toU32WithAlpha(t.palette.success, 0.15f)
                                             : toU32WithAlpha(t.palette.error, 0.15f);
                const ImU32 border_color = add_mode
                                               ? toU32WithAlpha(t.palette.success, 0.8f)
                                               : toU32WithAlpha(t.palette.error, 0.8f);

                draw_list->AddRectFilled(p0, p1, fill_color);
                draw_list->AddRect(p0, p1, border_color, 0.0f, 0, 2.0f);
            }

            if (rm && rm->isPolygonPreviewActive()) {
                const auto& t = theme();
                const auto& points = rm->getPolygonPoints();
                const bool closed = rm->isPolygonClosed();
                const bool add_mode = rm->isPolygonAddMode();

                if (!points.empty()) {
                    const float render_scale = rm->getSettings().render_scale;
                    const ImU32 line_color = add_mode
                                                 ? toU32WithAlpha(t.palette.success, 0.8f)
                                                 : toU32WithAlpha(t.palette.error, 0.8f);
                    const ImU32 fill_color = add_mode
                                                 ? toU32WithAlpha(t.palette.success, 0.15f)
                                                 : toU32WithAlpha(t.palette.error, 0.15f);
                    const ImU32 vertex_color = add_mode
                                                   ? toU32WithAlpha(t.palette.success, 1.0f)
                                                   : toU32WithAlpha(t.palette.error, 1.0f);
                    const ImU32 line_to_mouse_color = add_mode
                                                          ? toU32WithAlpha(t.palette.success, 0.5f)
                                                          : toU32WithAlpha(t.palette.error, 0.5f);

                    std::vector<ImVec2> screen_points;
                    screen_points.reserve(points.size());
                    for (const auto& [px, py] : points) {
                        screen_points.emplace_back(viewport_layout_.pos.x + px / render_scale,
                                                   viewport_layout_.pos.y + py / render_scale);
                    }

                    if (closed && screen_points.size() >= 3) {
                        draw_list->AddConvexPolyFilled(screen_points.data(), static_cast<int>(screen_points.size()), fill_color);
                    }

                    for (size_t i = 0; i + 1 < screen_points.size(); ++i) {
                        draw_list->AddLine(screen_points[i], screen_points[i + 1], line_color, 2.0f);
                    }
                    if (closed && screen_points.size() >= 3) {
                        draw_list->AddLine(screen_points.back(), screen_points.front(), line_color, 2.0f);
                    }

                    if (!closed) {
                        const ImVec2 mouse_pos = ImGui::GetMousePos();
                        draw_list->AddLine(screen_points.back(), mouse_pos, line_to_mouse_color, 1.0f);

                        constexpr float CLOSE_THRESHOLD = 12.0f;
                        if (screen_points.size() >= 3) {
                            const float dx = mouse_pos.x - screen_points.front().x;
                            const float dy = mouse_pos.y - screen_points.front().y;
                            if (dx * dx + dy * dy < CLOSE_THRESHOLD * CLOSE_THRESHOLD) {
                                draw_list->AddCircle(screen_points.front(), 9.0f, vertex_color, 16, 2.0f);
                            }
                        }
                    }

                    for (const auto& sp : screen_points) {
                        draw_list->AddCircleFilled(sp, 5.0f, vertex_color);
                    }
                }
            }

            if (rm && rm->isLassoPreviewActive()) {
                const auto& t = theme();
                const auto& points = rm->getLassoPoints();
                const bool add_mode = rm->isLassoAddMode();

                if (points.size() >= 2) {
                    const float render_scale = rm->getSettings().render_scale;
                    const ImU32 line_color = add_mode
                                                 ? toU32WithAlpha(t.palette.success, 0.8f)
                                                 : toU32WithAlpha(t.palette.error, 0.8f);

                    ImVec2 prev(viewport_layout_.pos.x + points[0].first / render_scale,
                                viewport_layout_.pos.y + points[0].second / render_scale);
                    for (size_t i = 1; i < points.size(); ++i) {
                        ImVec2 curr(viewport_layout_.pos.x + points[i].first / render_scale,
                                    viewport_layout_.pos.y + points[i].second / render_scale);
                        draw_list->AddLine(prev, curr, line_color, 2.0f);
                        prev = curr;
                    }
                }
            }
        }

        auto* align_tool = ctx.viewer->getAlignTool();
        if (align_tool && align_tool->isEnabled() && !ui_hidden_) {
            align_tool->renderUI(ctx, nullptr);
        }

        if (auto* const ic = ctx.viewer->getInputController();
            !ui_hidden_ && ic && ic->isNodeRectDragging()) {
            const auto start = ic->getNodeRectStart();
            const auto end = ic->getNodeRectEnd();
            const auto& t = theme();
            auto* const draw_list = ImGui::GetForegroundDrawList();
            draw_list->AddRectFilled({start.x, start.y}, {end.x, end.y}, toU32WithAlpha(t.palette.warning, 0.15f));
            draw_list->AddRect({start.x, start.y}, {end.x, end.y}, toU32WithAlpha(t.palette.warning, 0.85f), 0.0f, 0, 2.0f);
        }
    }

    void GuiManager::renderViewportDecorations() {
        if (viewport_layout_.size.x > 0 && viewport_layout_.size.y > 0) {
            widgets::DrawViewportVignette(viewport_layout_.pos, viewport_layout_.size);
        }

        if (!ui_hidden_ && viewport_layout_.size.x > 0 && viewport_layout_.size.y > 0) {
            const auto& t = theme();
            const float r = t.viewport.corner_radius;
            if (r > 0.0f) {
                auto* const dl = ImGui::GetBackgroundDrawList();
                const ImU32 bg = toU32(t.palette.background);
                const float x1 = viewport_layout_.pos.x, y1 = viewport_layout_.pos.y;
                const float x2 = x1 + viewport_layout_.size.x, y2 = y1 + viewport_layout_.size.y;

                constexpr int CORNER_ARC_SEGMENTS = 12;
                const auto maskCorner = [&](const ImVec2 corner, const ImVec2 edge,
                                            const ImVec2 center, const float a0, const float a1) {
                    dl->PathLineTo(corner);
                    dl->PathLineTo(edge);
                    dl->PathArcTo(center, r, a0, a1, CORNER_ARC_SEGMENTS);
                    dl->PathLineTo(corner);
                    dl->PathFillConvex(bg);
                };
                maskCorner({x1, y1}, {x1, y1 + r}, {x1 + r, y1 + r}, IM_PI, IM_PI * 1.5f);
                maskCorner({x2, y1}, {x2 - r, y1}, {x2 - r, y1 + r}, IM_PI * 1.5f, IM_PI * 2.0f);
                maskCorner({x1, y2}, {x1 + r, y2}, {x1 + r, y2 - r}, IM_PI * 0.5f, IM_PI);
                maskCorner({x2, y2}, {x2, y2 - r}, {x2 - r, y2 - r}, 0.0f, IM_PI * 0.5f);

                if (t.viewport.border_size > 0.0f) {
                    dl->AddRect({x1, y1}, {x2, y2}, t.viewport_border_u32(), r,
                                ImDrawFlags_RoundCornersAll, t.viewport.border_size);
                }
            }
        }
    }

    void GuiManager::updateInputOverrides(bool mouse_in_viewport) {
        const bool any_popup_or_modal_open = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        const bool imgui_wants_input = ImGui::GetIO().WantTextInput || ImGui::GetIO().WantCaptureKeyboard;

        if ((ImGuizmo::IsOver() || ImGuizmo::IsUsing()) && !any_popup_or_modal_open) {
            ImGui::GetIO().WantCaptureMouse = false;
            ImGui::GetIO().WantCaptureKeyboard = false;
        }

        if (mouse_in_viewport && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) &&
            !any_popup_or_modal_open && !imgui_wants_input) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
                ImGui::GetIO().WantCaptureMouse = false;
            }
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                ImGui::ClearActiveID();
                ImGui::GetIO().WantCaptureKeyboard = false;
                if (auto* editor = panels::PythonConsoleState::getInstance().getEditor()) {
                    editor->unfocus();
                }
            }
        }

        auto* rendering_manager = viewer_->getRenderingManager();
        if (rendering_manager) {
            const auto& settings = rendering_manager->getSettings();
            if (settings.point_cloud_mode && mouse_in_viewport &&
                !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) &&
                !any_popup_or_modal_open && !imgui_wants_input) {
                ImGui::GetIO().WantCaptureMouse = false;
                ImGui::GetIO().WantCaptureKeyboard = false;
            }
        }
    }

    ImVec2 GuiManager::getViewportPos() const {
        return viewport_layout_.pos;
    }

    ImVec2 GuiManager::getViewportSize() const {
        return viewport_layout_.size;
    }

    bool GuiManager::isMouseInViewport() const {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        return mouse_pos.x >= viewport_layout_.pos.x &&
               mouse_pos.y >= viewport_layout_.pos.y &&
               mouse_pos.x < viewport_layout_.pos.x + viewport_layout_.size.x &&
               mouse_pos.y < viewport_layout_.pos.y + viewport_layout_.size.y;
    }

    bool GuiManager::isViewportFocused() const {
        return viewport_layout_.has_focus;
    }

    bool GuiManager::isPositionInViewport(double x, double y) const {
        const ImGuiViewport* main_viewport = ImGui::GetMainViewport();

        // Convert to window-relative coordinates
        float rel_x = static_cast<float>(x) - main_viewport->WorkPos.x;
        float rel_y = static_cast<float>(y) - main_viewport->WorkPos.y;

        // Check if within viewport bounds
        return (rel_x >= viewport_layout_.pos.x &&
                rel_x < viewport_layout_.pos.x + viewport_layout_.size.x &&
                rel_y >= viewport_layout_.pos.y &&
                rel_y < viewport_layout_.pos.y + viewport_layout_.size.y);
    }

    void GuiManager::setupEventHandlers() {
        using namespace lfs::core::events;

        ui::FileDropReceived::when([this](const auto&) {
            startup_overlay_.dismiss();
            drag_drop_.resetHovering();
        });

        cmd::ShowWindow::when([this](const auto& e) {
            showWindow(e.window_name, e.show);
        });

        cmd::GoToCamView::when([this](const auto& e) {
            if (auto* sm = viewer_->getSceneManager()) {
                const auto& scene = sm->getScene();
                for (const auto* node : scene.getNodes()) {
                    if (node->type == core::NodeType::CAMERA && node->camera_uid == e.cam_id) {
                        ui::NodeSelected{.path = node->name, .type = "Camera", .metadata = {}}.emit();
                        break;
                    }
                }
            }
        });

        ui::FocusTrainingPanel::when([this](const auto&) {
            focus_panel_name_ = "Training";
        });

        ui::ToggleUI::when([this](const auto&) {
            ui_hidden_ = !ui_hidden_;
        });

        ui::ToggleFullscreen::when([this](const auto&) {
            if (auto* wm = viewer_->getWindowManager()) {
                wm->toggleFullscreen();
            }
        });

        internal::DisplayScaleChanged::when([this](const auto& e) {
            const float scale = std::clamp(e.scale, 1.0f, 4.0f);
            lfs::python::set_shared_dpi_scale(scale);
            lfs::vis::setThemeDpiScale(scale);
            LOG_INFO("Display scale changed to {:.2f}", scale);
        });

        state::DiskSpaceSaveFailed::when([this](const auto& e) {
            // Non-disk-space errors are handled by notification_bridge.cpp
            if (!e.is_disk_space_error)
                return;

            if (!disk_space_error_dialog_)
                return;

            const DiskSpaceErrorDialog::ErrorInfo info{
                .path = e.path,
                .error_message = e.error,
                .required_bytes = e.required_bytes,
                .available_bytes = e.available_bytes,
                .iteration = e.iteration,
                .is_checkpoint = e.is_checkpoint};

            if (e.is_checkpoint) {
                auto on_retry = [this, iteration = e.iteration]() {
                    if (auto* tm = viewer_->getTrainerManager()) {
                        if (tm->isFinished() || !tm->isTrainingActive()) {
                            if (auto* trainer = tm->getTrainer()) {
                                LOG_INFO("Retrying save at iteration {}", iteration);
                                trainer->save_final_ply_and_checkpoint(iteration);
                            }
                        } else {
                            tm->requestSaveCheckpoint();
                        }
                    }
                };

                auto on_change_location = [this, iteration = e.iteration](const std::filesystem::path& new_path) {
                    if (auto* tm = viewer_->getTrainerManager()) {
                        if (auto* trainer = tm->getTrainer()) {
                            auto params = trainer->getParams();
                            params.dataset.output_path = new_path;
                            trainer->setParams(params);
                            LOG_INFO("Output path changed to: {}", lfs::core::path_to_utf8(new_path));

                            if (tm->isFinished() || !tm->isTrainingActive()) {
                                trainer->save_final_ply_and_checkpoint(iteration);
                            } else {
                                tm->requestSaveCheckpoint();
                            }
                        }
                    }
                };

                auto on_cancel = []() {
                    LOG_WARN("Checkpoint save cancelled by user");
                };

                disk_space_error_dialog_->show(info, on_retry, on_change_location, on_cancel);
            } else {
                auto on_retry = []() {};

                auto on_change_location = [](const std::filesystem::path& new_path) {
                    LOG_INFO("Re-export manually using File > Export to: {}", lfs::core::path_to_utf8(new_path));
                };

                auto on_cancel = []() {
                    LOG_INFO("Export cancelled by user");
                };

                disk_space_error_dialog_->show(info, on_retry, on_change_location, on_cancel);
            }
        });

        state::DatasetLoadCompleted::when([this](const auto& e) {
            if (e.success) {
                focus_panel_name_ = "Training";
            }
        });

        internal::TrainerReady::when([this](const auto&) {
            focus_panel_name_ = "Training";
        });
    }

    bool GuiManager::isCapturingInput() const {
        if (auto* input_controller = viewer_->getInputController()) {
            return input_controller->getBindings().isCapturing();
        }
        return false;
    }

    bool GuiManager::isModalWindowOpen() const {
        // Check any ImGui popup/modal (covers Python popups and floating panels)
        return ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
    }

    void GuiManager::captureKey(int key, int mods) {
        if (auto* input_controller = viewer_->getInputController()) {
            input_controller->getBindings().captureKey(key, mods);
        }
    }

    void GuiManager::captureMouseButton(int button, int mods) {
        if (auto* input_controller = viewer_->getInputController()) {
            input_controller->getBindings().captureMouseButton(button, mods);
        }
    }

    void GuiManager::requestThumbnail(const std::string& video_id) {
        if (menu_bar_) {
            menu_bar_->requestThumbnail(video_id);
        }
    }

    void GuiManager::processThumbnails() {
        if (menu_bar_) {
            menu_bar_->processThumbnails();
        }
    }

    bool GuiManager::isThumbnailReady(const std::string& video_id) const {
        return menu_bar_ ? menu_bar_->isThumbnailReady(video_id) : false;
    }

    uint64_t GuiManager::getThumbnailTexture(const std::string& video_id) const {
        return menu_bar_ ? menu_bar_->getThumbnailTexture(video_id) : 0;
    }

    int GuiManager::getHighlightedCameraUid() const {
        if (auto* sm = viewer_->getSceneManager()) {
            return sm->getSelectedCameraUid();
        }
        return -1;
    }

    void GuiManager::applyDefaultStyle() {
        // Initialize theme system using saved preference
        const std::string preferred_theme = loadThemePreferenceName();
        if (!setThemeByName(preferred_theme)) {
            setTheme(darkTheme());
        }
    }

    void GuiManager::showWindow(const std::string& name, bool show) {
        window_states_[name] = show;
    }

    bool GuiManager::needsAnimationFrame() const {
        if (startup_overlay_.needsAnimationFrame())
            return true;
        if (video_extractor_dialog_ && video_extractor_dialog_->isVideoPlaying())
            return true;
        return false;
    }

    void GuiManager::dismissStartupOverlay() {
        startup_overlay_.dismiss();
    }

    void GuiManager::setFileSelectedCallback(std::function<void(const std::filesystem::path&, bool)> callback) {
        if (file_browser_) {
            file_browser_->setOnFileSelected(callback);
        }
    }

    void GuiManager::requestExitConfirmation() {
        startup_overlay_.dismiss();
        lfs::core::events::cmd::RequestExit{}.emit();
    }

    bool GuiManager::isExitConfirmationPending() const {
        return lfs::python::is_exit_popup_open();
    }

} // namespace lfs::vis::gui
