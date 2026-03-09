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
#include "gui/layout_state.hpp"
#include "gui/native_panels.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/file_association.hpp"
#include "gui/utils/windows_utils.hpp"
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
#include <fstream>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl3.h>
#include <imgui_internal.h>
#include <iterator>
#include <ImGuizmo.h>

namespace lfs::vis::gui {

    namespace {
        int imguiKeyToSDLScancode(ImGuiKey key) {
            // clang-format off
            switch (key) {
            case ImGuiKey_Space:      return SDL_SCANCODE_SPACE;
            case ImGuiKey_Backspace:  return SDL_SCANCODE_BACKSPACE;
            case ImGuiKey_Tab:        return SDL_SCANCODE_TAB;
            case ImGuiKey_Enter:      return SDL_SCANCODE_RETURN;
            case ImGuiKey_Escape:     return SDL_SCANCODE_ESCAPE;
            case ImGuiKey_Delete:     return SDL_SCANCODE_DELETE;
            case ImGuiKey_Insert:     return SDL_SCANCODE_INSERT;
            case ImGuiKey_Home:       return SDL_SCANCODE_HOME;
            case ImGuiKey_End:        return SDL_SCANCODE_END;
            case ImGuiKey_PageUp:     return SDL_SCANCODE_PAGEUP;
            case ImGuiKey_PageDown:   return SDL_SCANCODE_PAGEDOWN;
            case ImGuiKey_LeftArrow:  return SDL_SCANCODE_LEFT;
            case ImGuiKey_UpArrow:    return SDL_SCANCODE_UP;
            case ImGuiKey_RightArrow: return SDL_SCANCODE_RIGHT;
            case ImGuiKey_DownArrow:  return SDL_SCANCODE_DOWN;
            case ImGuiKey_F1:  return SDL_SCANCODE_F1;
            case ImGuiKey_F2:  return SDL_SCANCODE_F2;
            case ImGuiKey_F3:  return SDL_SCANCODE_F3;
            case ImGuiKey_F4:  return SDL_SCANCODE_F4;
            case ImGuiKey_F5:  return SDL_SCANCODE_F5;
            case ImGuiKey_F6:  return SDL_SCANCODE_F6;
            case ImGuiKey_F7:  return SDL_SCANCODE_F7;
            case ImGuiKey_F8:  return SDL_SCANCODE_F8;
            case ImGuiKey_F9:  return SDL_SCANCODE_F9;
            case ImGuiKey_F10: return SDL_SCANCODE_F10;
            case ImGuiKey_F11: return SDL_SCANCODE_F11;
            case ImGuiKey_F12: return SDL_SCANCODE_F12;
            case ImGuiKey_Equal:          return SDL_SCANCODE_EQUALS;
            case ImGuiKey_Minus:          return SDL_SCANCODE_MINUS;
            case ImGuiKey_KeypadAdd:      return SDL_SCANCODE_KP_PLUS;
            case ImGuiKey_KeypadSubtract: return SDL_SCANCODE_KP_MINUS;
            default: break;
            }
            // clang-format on

            if (key >= ImGuiKey_A && key <= ImGuiKey_Z)
                return SDL_SCANCODE_A + (key - ImGuiKey_A);
            if (key == ImGuiKey_0)
                return SDL_SCANCODE_0;
            if (key >= ImGuiKey_1 && key <= ImGuiKey_9)
                return SDL_SCANCODE_1 + (key - ImGuiKey_1);
            return -1;
        }

        void applyFrameKeyboardCapture() {
            if (!RmlPanelHost::consumeFrameWantsKeyboard())
                return;

            ImGui::GetIO().WantCaptureKeyboard = true;
            ImGui::GetIO().WantTextInput = true;
        }

        void drawFrameTooltip(const std::string& tip) {
            if (tip.empty())
                return;

            const auto& p = lfs::vis::theme().palette;
            auto* vp = ImGui::GetMainViewport();
            auto* fg = ImGui::GetForegroundDrawList(vp);
            const ImVec2 mouse = ImGui::GetMousePos();
            const ImVec2 pad(8, 6);
            const ImVec2 text_size = ImGui::CalcTextSize(tip.c_str());
            const float box_w = text_size.x + pad.x * 2;
            const float box_h = text_size.y + pad.y * 2;

            ImVec2 box_min(mouse.x + 14, mouse.y + 18);
            if (box_min.x + box_w > vp->Pos.x + vp->Size.x)
                box_min.x = mouse.x - 14 - box_w;
            if (box_min.y + box_h > vp->Pos.y + vp->Size.y)
                box_min.y = mouse.y - 18 - box_h;

            const ImVec2 box_max(box_min.x + box_w, box_min.y + box_h);
            fg->AddRectFilled(box_min, box_max,
                              ImGui::ColorConvertFloat4ToU32(p.surface_bright), 4.0f);
            fg->AddRect(box_min, box_max, ImGui::ColorConvertFloat4ToU32(p.border), 4.0f);
            fg->AddText(ImVec2(box_min.x + pad.x, box_min.y + pad.y),
                        ImGui::ColorConvertFloat4ToU32(p.text), tip.c_str());
        }
    } // namespace

    GuiManager::GuiManager(VisualizerImpl* viewer)
        : viewer_(viewer),
          sequencer_ui_(viewer, sequencer_ui_state_, &rmlui_manager_),
          gizmo_manager_(viewer),
          async_tasks_(viewer) {

        panel_layout_.loadState();

        // Create components
        menu_bar_ = std::make_unique<MenuBar>();
        rml_modal_overlay_ = std::make_unique<RmlModalOverlay>(&rmlui_manager_);
        global_context_menu_ = std::make_unique<GlobalContextMenu>(&rmlui_manager_);
        lfs::python::set_global_context_menu(global_context_menu_.get());
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
        window_states_["scene_panel"] = true;
        window_states_["system_console"] = false;
        window_states_["training_tab"] = false;
        window_states_["export_dialog"] = false;
        window_states_["python_console"] = false;

        lfs::python::set_modal_enqueue_callback(
            [this](lfs::core::ModalRequest req) { rml_modal_overlay_->enqueue(std::move(req)); });

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
            pending_cuda_warning_ = info;
        }
    }

    void GuiManager::promptFileAssociation() {
#ifdef _WIN32
        if (file_association_checked_)
            return;
        file_association_checked_ = true;

        LayoutState state;
        state.load();

        if (!state.file_association.empty())
            return;
        if (areFileAssociationsRegistered())
            return;

        using namespace lichtfeld::Strings;
        lfs::core::ModalRequest req;
        req.title = LOC(FileAssociation::TITLE);
        req.body_rml = "<p>" + std::string(LOC(FileAssociation::MESSAGE)) + "</p>";
        req.style = lfs::core::ModalStyle::Info;
        req.buttons = {
            {LOC(FileAssociation::YES), "primary"},
            {LOC(FileAssociation::NOT_NOW), "secondary"},
            {LOC(FileAssociation::DONT_ASK), "secondary"},
        };
        req.on_result = [](const lfs::core::ModalResult& result) {
            LayoutState ls;
            ls.load();

            if (result.button_label == LOC(FileAssociation::YES)) {
                registerFileAssociations();
                ls.file_association = "registered";
            } else if (result.button_label == LOC(FileAssociation::DONT_ASK)) {
                ls.file_association = "declined";
            } else {
                return;
            }
            ls.save();
        };

        rml_modal_overlay_->enqueue(std::move(req));
#endif
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

    void GuiManager::rebuildFonts(float scale) {
        ImGuiIO& io = ImGui::GetIO();

        ImGui_ImplOpenGL3_DestroyFontsTexture();
        io.Fonts->Clear();

        const auto& t = theme();
        try {
            const auto regular_path = lfs::vis::getAssetPath("fonts/" + t.fonts.regular_path);
            const auto bold_path = lfs::vis::getAssetPath("fonts/" + t.fonts.bold_path);
            const auto japanese_path = lfs::vis::getAssetPath("fonts/NotoSansJP-Regular.ttf");
            const auto korean_path = lfs::vis::getAssetPath("fonts/NotoSansKR-Regular.ttf");

            const auto is_font_valid = [](const std::filesystem::path& path) -> bool {
                constexpr size_t MIN_FONT_FILE_SIZE = 100;
                return std::filesystem::exists(path) && std::filesystem::file_size(path) >= MIN_FONT_FILE_SIZE;
            };

            const auto load_font_latin_only =
                [&](const std::filesystem::path& path, const float size) -> ImFont* {
                if (!is_font_valid(path))
                    return nullptr;
                const std::string path_utf8 = lfs::core::path_to_utf8(path);
                ImFontConfig config;
                config.PixelSnapH = true;
                return io.Fonts->AddFontFromFileTTF(path_utf8.c_str(), size, &config);
            };

            const auto merge_cjk = [&](const float size) {
                if (is_font_valid(japanese_path)) {
                    ImFontConfig config;
                    config.MergeMode = true;
                    config.OversampleH = 1;
                    config.PixelSnapH = true;
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
                    config.PixelSnapH = true;
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

            font_regular_ = load_font_with_cjk(regular_path, t.fonts.base_size * scale);
            font_bold_ = load_font_with_cjk(bold_path, t.fonts.base_size * scale);
            font_heading_ = load_font_with_cjk(bold_path, t.fonts.heading_size * scale);
            font_small_ = load_font_with_cjk(regular_path, t.fonts.small_size * scale);
            font_section_ = load_font_with_cjk(bold_path, t.fonts.section_size * scale);

            const auto monospace_path = lfs::vis::getAssetPath("fonts/JetBrainsMono-Regular.ttf");
            if (is_font_valid(monospace_path)) {
                const std::string mono_path_utf8 = lfs::core::path_to_utf8(monospace_path);

                static constexpr ImWchar GLYPH_RANGES[] = {
                    0x0020,
                    0x00FF,
                    0x2190,
                    0x21FF,
                    0x2500,
                    0x257F,
                    0x2580,
                    0x259F,
                    0x25A0,
                    0x25FF,
                    0,
                };

                static constexpr float MONO_SCALES[] = {0.7f, 1.0f, 1.3f, 1.7f, 2.2f};
                static_assert(std::size(MONO_SCALES) == FontSet::MONO_SIZE_COUNT);

                for (int i = 0; i < FontSet::MONO_SIZE_COUNT; ++i) {
                    ImFontConfig config;
                    config.GlyphRanges = GLYPH_RANGES;
                    config.PixelSnapH = true;
                    const float size = t.fonts.base_size * scale * MONO_SCALES[i];
                    mono_fonts_[i] = io.Fonts->AddFontFromFileTTF(mono_path_utf8.c_str(), size, &config);
                    mono_font_scales_[i] = MONO_SCALES[i];
                }
                font_monospace_ = mono_fonts_[1];
            }
            if (!font_monospace_)
                font_monospace_ = font_regular_;

            const bool all_loaded = font_regular_ && font_bold_ && font_heading_ && font_small_ && font_section_;
            if (!all_loaded) {
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
    }

    void GuiManager::applyUiScale(float scale) {
        scale = std::clamp(scale, 1.0f, 4.0f);

        rmlui_manager_.setDpRatio(scale);
        lfs::vis::setThemeDpiScale(scale);
        lfs::python::set_shared_dpi_scale(scale);
        applyDefaultStyle();
        rebuildFonts(scale);
        current_ui_scale_ = scale;

        LOG_INFO("UI scale applied: {:.2f}", scale);
    }

    void GuiManager::loadImGuiSettings() {
        if (imgui_ini_path_.empty())
            return;

        try {
            if (!std::filesystem::exists(imgui_ini_path_))
                return;

            std::ifstream file;
            if (!lfs::core::open_file_for_read(imgui_ini_path_, std::ios::binary, file)) {
                LOG_WARN("Failed to open ImGui settings file: {}", lfs::core::path_to_utf8(imgui_ini_path_));
                return;
            }

            const std::string ini_data((std::istreambuf_iterator<char>(file)),
                                       std::istreambuf_iterator<char>());
            ImGui::LoadIniSettingsFromMemory(ini_data.c_str(), ini_data.size());
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load ImGui settings: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to load ImGui settings: unknown error");
        }
    }

    void GuiManager::saveImGuiSettings() const {
        if (imgui_ini_path_.empty() || !ImGui::GetCurrentContext())
            return;

        try {
            std::filesystem::create_directories(imgui_ini_path_.parent_path());

            size_t ini_size = 0;
            const char* ini_data = ImGui::SaveIniSettingsToMemory(&ini_size);

            std::ofstream file;
            if (!lfs::core::open_file_for_write(imgui_ini_path_,
                                                std::ios::binary | std::ios::trunc,
                                                file)) {
                LOG_WARN("Failed to open ImGui settings for writing: {}",
                         lfs::core::path_to_utf8(imgui_ini_path_));
                return;
            }

            file.write(ini_data, static_cast<std::streamsize>(ini_size));
            if (!file) {
                LOG_WARN("Failed to write ImGui settings: {}",
                         lfs::core::path_to_utf8(imgui_ini_path_));
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save ImGui settings: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to save ImGui settings: unknown error");
        }
    }

    void GuiManager::persistImGuiSettingsIfNeeded() {
        ImGuiIO& io = ImGui::GetIO();
        if (!io.WantSaveIniSettings)
            return;

        saveImGuiSettings();
        io.WantSaveIniSettings = false;
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
        imgui_ini_path_ = LayoutState::getConfigDir() / "imgui.ini";
        io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
        io.ConfigWindowsMoveFromTitleBarOnly = true;
        loadImGuiSettings();

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

        float saved_scale = lfs::vis::loadUiScalePreference();
        if (saved_scale <= 0.0f)
            saved_scale = SDL_GetWindowDisplayScale(viewer_->getWindow());
        current_ui_scale_ = std::clamp(saved_scale, 1.0f, 4.0f);

        lfs::python::set_shared_dpi_scale(current_ui_scale_);
        lfs::vis::setThemeDpiScale(current_ui_scale_);

        // Set application icon
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

        applyDefaultStyle();
        rebuildFonts(current_ui_scale_);

        initMenuBar();

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

        rmlui_manager_.init(viewer_->getWindow(), current_ui_scale_);
        lfs::vis::setThemeChangeCallback([this](const std::string& theme_id) {
            rmlui_manager_.activateTheme(theme_id);
        });
        lfs::python::set_rml_manager(&rmlui_manager_);

        startup_overlay_.init(&rmlui_manager_);
        rml_shell_frame_.init(&rmlui_manager_);
        rml_right_panel_.init(&rmlui_manager_);
        rml_right_panel_.on_tab_changed = [this](const std::string& idname) {
            panel_layout_.setActiveTab(idname);
        };
        rml_right_panel_.on_splitter_delta = [this](float delta_y) {
            viewer_->getRenderingManager()->setViewportResizeActive(true);
            const auto* mvp = ImGui::GetMainViewport();
            ScreenState ss;
            ss.work_pos = {mvp->WorkPos.x, mvp->WorkPos.y};
            ss.work_size = {mvp->WorkSize.x, mvp->WorkSize.y};
            panel_layout_.adjustScenePanelRatio(delta_y, ss);
        };
        rml_right_panel_.on_splitter_end = [this]() {
            viewer_->getRenderingManager()->setViewportResizeActive(false);
        };
        rml_right_panel_.on_resize_delta = [this](float dx) {
            viewer_->getRenderingManager()->setViewportResizeActive(true);
            const auto* mvp = ImGui::GetMainViewport();
            ScreenState ss;
            ss.work_pos = {mvp->WorkPos.x, mvp->WorkPos.y};
            ss.work_size = {mvp->WorkSize.x, mvp->WorkSize.y};
            panel_layout_.applyResizeDelta(dx, ss);
        };
        rml_right_panel_.on_resize_end = [this]() {
            viewer_->getRenderingManager()->setViewportResizeActive(false);
        };
        rml_viewport_overlay_.init(&rmlui_manager_);
        rml_menu_bar_.init(&rmlui_manager_);
        rml_status_bar_.init(&rmlui_manager_);

        lfs::python::RmlPanelHostOps ops{};
        ops.create = [](void* mgr, const char* name, const char* rml) -> void* {
            return new RmlPanelHost(static_cast<RmlUIManager*>(mgr),
                                    std::string(name), std::string(rml));
        };
        ops.destroy = [](void* host) {
            delete static_cast<RmlPanelHost*>(host);
        };
        ops.draw = [](void* host, const void* ctx) {
            auto* h = static_cast<RmlPanelHost*>(host);
            float aw = ImGui::GetContentRegionAvail().x;
            float ah = ImGui::GetContentRegionAvail().y;
            ImVec2 pos = ImGui::GetCursorScreenPos();

            PanelInputState fallback;
            if (!h->hasInput()) {
                auto* mvp = ImGui::GetMainViewport();
                const auto& fio = ImGui::GetIO();
                fallback.mouse_x = fio.MousePos.x;
                fallback.mouse_y = fio.MousePos.y;
                fallback.mouse_down[0] = ImGui::IsMouseDown(ImGuiMouseButton_Left);
                fallback.mouse_clicked[0] = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
                fallback.mouse_released[0] = ImGui::IsMouseReleased(ImGuiMouseButton_Left);
                fallback.mouse_clicked[1] = ImGui::IsMouseClicked(ImGuiMouseButton_Right);
                fallback.mouse_released[1] = ImGui::IsMouseReleased(ImGuiMouseButton_Right);
                fallback.mouse_wheel = fio.MouseWheel;
                fallback.key_ctrl = fio.KeyCtrl;
                fallback.key_shift = fio.KeyShift;
                fallback.key_alt = fio.KeyAlt;
                fallback.key_super = fio.KeySuper;
                fallback.screen_w = static_cast<int>(mvp->Size.x);
                fallback.screen_h = static_cast<int>(mvp->Size.y);
                h->setInput(&fallback);
            }
            h->draw(*static_cast<const PanelDrawContext*>(ctx),
                    aw, ah, pos.x, pos.y);
            h->setInput(nullptr);
        };
        ops.draw_direct = [](void* host, float x, float y, float w, float h) {
            auto* hp = static_cast<RmlPanelHost*>(host);
            PanelInputState fallback;
            if (!hp->hasInput()) {
                auto* mvp = ImGui::GetMainViewport();
                const auto& fio = ImGui::GetIO();
                fallback.mouse_x = fio.MousePos.x;
                fallback.mouse_y = fio.MousePos.y;
                fallback.mouse_down[0] = ImGui::IsMouseDown(ImGuiMouseButton_Left);
                fallback.mouse_down[1] = ImGui::IsMouseDown(ImGuiMouseButton_Right);
                fallback.mouse_clicked[0] = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
                fallback.mouse_clicked[1] = ImGui::IsMouseClicked(ImGuiMouseButton_Right);
                fallback.mouse_released[0] = ImGui::IsMouseReleased(ImGuiMouseButton_Left);
                fallback.mouse_released[1] = ImGui::IsMouseReleased(ImGuiMouseButton_Right);
                fallback.mouse_wheel = fio.MouseWheel;
                fallback.key_ctrl = fio.KeyCtrl;
                fallback.key_shift = fio.KeyShift;
                fallback.key_alt = fio.KeyAlt;
                fallback.key_super = fio.KeySuper;
                fallback.screen_w = static_cast<int>(mvp->Size.x);
                fallback.screen_h = static_cast<int>(mvp->Size.y);
                fallback.bg_draw_list = ImGui::GetForegroundDrawList(mvp);
                fallback.fg_draw_list = ImGui::GetForegroundDrawList(mvp);
                hp->setInput(&fallback);
            }
            hp->drawDirect(x, y, w, h);
            hp->setInput(nullptr);
        };
        ops.prepare_direct = [](void* host, float w, float h) {
            auto* hp = static_cast<RmlPanelHost*>(host);
            PanelInputState fallback;
            if (!hp->hasInput()) {
                auto* mvp = ImGui::GetMainViewport();
                const auto& fio = ImGui::GetIO();
                fallback.mouse_x = fio.MousePos.x;
                fallback.mouse_y = fio.MousePos.y;
                fallback.mouse_down[0] = ImGui::IsMouseDown(ImGuiMouseButton_Left);
                fallback.mouse_down[1] = ImGui::IsMouseDown(ImGuiMouseButton_Right);
                fallback.mouse_clicked[0] = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
                fallback.mouse_clicked[1] = ImGui::IsMouseClicked(ImGuiMouseButton_Right);
                fallback.mouse_released[0] = ImGui::IsMouseReleased(ImGuiMouseButton_Left);
                fallback.mouse_released[1] = ImGui::IsMouseReleased(ImGuiMouseButton_Right);
                fallback.mouse_wheel = fio.MouseWheel;
                fallback.key_ctrl = fio.KeyCtrl;
                fallback.key_shift = fio.KeyShift;
                fallback.key_alt = fio.KeyAlt;
                fallback.key_super = fio.KeySuper;
                fallback.screen_w = static_cast<int>(mvp->Size.x);
                fallback.screen_h = static_cast<int>(mvp->Size.y);
                hp->setInput(&fallback);
            }
            hp->prepareDirect(w, h);
            hp->setInput(nullptr);
        };
        ops.prepare_layout = [](void* host, float w, float h) {
            static_cast<RmlPanelHost*>(host)->syncDirectLayout(w, h);
        };
        ops.get_document = [](void* host) -> void* {
            return static_cast<RmlPanelHost*>(host)->getDocument();
        };
        ops.is_loaded = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->isDocumentLoaded();
        };
        ops.set_height_mode = [](void* host, int mode) {
            static_cast<RmlPanelHost*>(host)->setHeightMode(
                static_cast<HeightMode>(mode));
        };
        ops.get_content_height = [](void* host) -> float {
            return static_cast<RmlPanelHost*>(host)->getContentHeight();
        };
        ops.ensure_context = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->ensureContext();
        };
        ops.ensure_document = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->ensureDocumentLoaded();
        };
        ops.get_context = [](void* host) -> void* {
            return static_cast<RmlPanelHost*>(host)->getContext();
        };
        ops.set_foreground = [](void* host, bool fg) {
            static_cast<RmlPanelHost*>(host)->setForeground(fg);
        };
        ops.mark_content_dirty = [](void* host) {
            static_cast<RmlPanelHost*>(host)->markContentDirty();
        };
        ops.set_input_clip_y = [](void* host, float y_min, float y_max) {
            static_cast<RmlPanelHost*>(host)->setInputClipY(y_min, y_max);
        };
        ops.set_input = [](void* host, const void* input) {
            static_cast<RmlPanelHost*>(host)->setInput(
                static_cast<const PanelInputState*>(input));
        };
        ops.set_forced_height = [](void* host, float h) {
            static_cast<RmlPanelHost*>(host)->setForcedHeight(h);
        };
        ops.needs_animation = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->needsAnimationFrame();
        };
        lfs::python::set_rml_panel_host_ops(ops);

        registerNativePanels();
    }

    void GuiManager::shutdown() {
        panel_layout_.saveState();

        if (video_extraction_thread_ && video_extraction_thread_->joinable())
            video_extraction_thread_->join();
        video_extraction_thread_.reset();

        async_tasks_.shutdown();

        const bool need_gil = lfs::python::get_main_thread_state() != nullptr;
        if (need_gil)
            lfs::python::acquire_gil_main_thread();

        lfs::python::shutdown_python_gl_resources();

        global_context_menu_->destroyGLResources();
        rml_status_bar_.shutdown();
        rml_menu_bar_.shutdown();
        rml_viewport_overlay_.shutdown();
        rml_right_panel_.shutdown();
        rml_shell_frame_.shutdown();
        startup_overlay_.shutdown();
        rmlui_manager_.shutdown();

        if (need_gil)
            lfs::python::release_gil_main_thread();

        sequencer_ui_.destroyGLResources();
        drag_drop_.shutdown();

        if (ImGui::GetCurrentContext()) {
            saveImGuiSettings();
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

        // Floating panels (self-managed windows)
        reg_panel("native.video_extractor", "Video Extractor",
                  make_panel(VideoExtractorPanel(video_extractor_dialog_.get())),
                  PanelSpace::Floating, 11, 0, 750.0f);
        reg.set_panel_enabled("native.video_extractor", false);

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
                  make_panel(StartupOverlayPanel(&startup_overlay_, &drag_drop_hovering_)),
                  PanelSpace::ViewportOverlay, 1000);

        reg_panel("native.status_bar", "##StatusBar",
                  make_panel(RmlStatusBarPanel(&rml_status_bar_)),
                  PanelSpace::StatusBar, 0);
    }

    void GuiManager::render() {
        if (pending_cuda_warning_) {
            constexpr int MIN_MAJOR = lfs::core::MIN_CUDA_VERSION / 1000;
            constexpr int MIN_MINOR = (lfs::core::MIN_CUDA_VERSION % 1000) / 10;
            lfs::core::events::state::CudaVersionUnsupported{
                .major = pending_cuda_warning_->major,
                .minor = pending_cuda_warning_->minor,
                .min_major = MIN_MAJOR,
                .min_minor = MIN_MINOR}
                .emit();
            pending_cuda_warning_.reset();
        }

        promptFileAssociation();

        if (pending_ui_scale_ > 0.0f) {
            applyUiScale(pending_ui_scale_);
            pending_ui_scale_ = 0.0f;
        }

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
                if (checkThemeFileChanges()) {
                    rml_theme::invalidateThemeMediaCache();
                }
                last_check = now;
            }
        }

        // Initialize ImGuizmo for this frame
        ImGuizmo::BeginFrame();

        if (menu_bar_ && !ui_hidden_) {
            menu_bar_->render();

            if (menu_bar_->hasMenuEntries()) {
                auto entries = menu_bar_->getMenuEntries();
                std::vector<std::string> labels;
                std::vector<std::string> idnames;
                labels.reserve(entries.size());
                idnames.reserve(entries.size());
                for (const auto& entry : entries) {
                    labels.emplace_back(LOC(entry.label.c_str()));
                    idnames.emplace_back(entry.idname);
                }
                rml_menu_bar_.updateLabels(labels, idnames);
            } else {
                rml_menu_bar_.updateLabels({}, {});
            }

            // Reserve work area for the RML menu bar via ImGui's internal inset mechanism
            {
                auto* vp = static_cast<ImGuiViewportP*>(ImGui::GetMainViewport());
                float bar_h = rml_menu_bar_.barHeight();
                vp->BuildWorkInsetMin.y = ImMax(vp->BuildWorkInsetMin.y, bar_h);
                vp->WorkInsetMin.y = ImMax(vp->WorkInsetMin.y, bar_h);
                vp->UpdateWorkRect();
            }

            const auto& io = ImGui::GetIO();
            PanelInputState menu_input;
            menu_input.mouse_x = io.MousePos.x;
            menu_input.mouse_y = io.MousePos.y;
            menu_input.screen_x = ImGui::GetMainViewport()->Pos.x;
            menu_input.screen_y = ImGui::GetMainViewport()->Pos.y;
            menu_input.mouse_down[0] = ImGui::IsMouseDown(ImGuiMouseButton_Left);
            menu_input.mouse_clicked[0] = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
            menu_input.screen_w = static_cast<int>(ImGui::GetMainViewport()->Size.x);
            menu_input.screen_h = static_cast<int>(ImGui::GetMainViewport()->Size.y);

            rml_menu_bar_.processInput(menu_input);

            if (rml_menu_bar_.wantsInput())
                ImGui::GetIO().WantCaptureMouse = true;

            rml_menu_bar_.draw(menu_input.screen_w, menu_input.screen_h);
        } else {
            rml_menu_bar_.suspend();
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

        if (!ui_hidden_) {
            const auto* mvp = ImGui::GetMainViewport();
            const float status_bar_h = PanelLayoutManager::STATUS_BAR_HEIGHT * current_ui_scale_;
            const float panel_h = mvp->WorkSize.y - status_bar_h;

            ShellRegions shell_regions;
            shell_regions.screen = {mvp->Pos.x, mvp->Pos.y, mvp->Size.x, mvp->Size.y};
            shell_regions.menu = {mvp->Pos.x, mvp->Pos.y,
                                  mvp->Size.x, mvp->WorkPos.y - mvp->Pos.y};

            if (show_main_panel_) {
                const float rpw = panel_layout_.getRightPanelWidth();
                shell_regions.right_panel = {
                    mvp->WorkPos.x + mvp->WorkSize.x - rpw,
                    mvp->WorkPos.y,
                    rpw,
                    panel_h,
                };
            }

            shell_regions.status = {
                mvp->WorkPos.x,
                mvp->WorkPos.y + mvp->WorkSize.y - status_bar_h,
                mvp->WorkSize.x,
                status_bar_h,
            };

            rml_shell_frame_.render(shell_regions);
        }

        // Update editor context state for this frame
        auto& editor_ctx = viewer_->getEditorContext();
        editor_ctx.update(viewer_->getSceneManager(), viewer_->getTrainerManager());

        // Create context for this frame
        UIContext ctx{
            .viewer = viewer_,
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
        draw_ctx.frame_serial = ++panel_frame_serial_;
        draw_ctx.scene_generation = python::get_scene_generation();
        if (auto* sm = ctx.viewer->getSceneManager())
            draw_ctx.has_selection = sm->hasSelectedNode();
        if (auto* cc = lfs::event::command_center())
            draw_ctx.is_training = cc->snapshot().is_running;

        auto& reg = PanelRegistry::instance();

        reg.preload_panels(PanelSpace::SceneHeader, draw_ctx);
        reg.preload_panels(PanelSpace::SidePanel, draw_ctx);

        const auto* mvp_input = ImGui::GetMainViewport();
        const auto& io = ImGui::GetIO();
        PanelInputState panel_input;
        panel_input.mouse_x = io.MousePos.x;
        panel_input.mouse_y = io.MousePos.y;
        panel_input.screen_x = mvp_input->Pos.x;
        panel_input.screen_y = mvp_input->Pos.y;
        panel_input.mouse_down[0] = ImGui::IsMouseDown(ImGuiMouseButton_Left);
        panel_input.mouse_down[1] = ImGui::IsMouseDown(ImGuiMouseButton_Right);
        panel_input.mouse_down[2] = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
        panel_input.mouse_clicked[0] = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
        panel_input.mouse_clicked[1] = ImGui::IsMouseClicked(ImGuiMouseButton_Right);
        panel_input.mouse_clicked[2] = ImGui::IsMouseClicked(ImGuiMouseButton_Middle);
        panel_input.mouse_released[0] = ImGui::IsMouseReleased(ImGuiMouseButton_Left);
        panel_input.mouse_released[1] = ImGui::IsMouseReleased(ImGuiMouseButton_Right);
        panel_input.mouse_released[2] = ImGui::IsMouseReleased(ImGuiMouseButton_Middle);
        panel_input.screen_w = static_cast<int>(mvp_input->Size.x);
        panel_input.screen_h = static_cast<int>(mvp_input->Size.y);
        panel_input.bg_draw_list = ImGui::GetBackgroundDrawList(ImGui::GetMainViewport());
        panel_input.fg_draw_list = ImGui::GetForegroundDrawList(ImGui::GetMainViewport());
        RmlPanelHost::clearQueuedForegroundComposites();
        panel_input.mouse_wheel = io.MouseWheel;
        panel_input.key_ctrl = io.KeyCtrl;
        panel_input.key_shift = io.KeyShift;
        panel_input.key_alt = io.KeyAlt;
        panel_input.key_super = io.KeySuper;

        for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END; ++k) {
            auto imgui_key = static_cast<ImGuiKey>(k);
            int sc = imguiKeyToSDLScancode(imgui_key);
            if (sc < 0)
                continue;
            if (ImGui::IsKeyPressed(imgui_key, false))
                panel_input.keys_pressed.push_back(sc);
            if (ImGui::IsKeyReleased(imgui_key))
                panel_input.keys_released.push_back(sc);
        }

        global_context_menu_->processInput(panel_input);
        if (global_context_menu_->isOpen())
            panel_input.mouse_wheel = 0;

        ScreenState screen;
        screen.work_pos = {mvp_input->WorkPos.x, mvp_input->WorkPos.y};
        screen.work_size = {mvp_input->WorkSize.x, mvp_input->WorkSize.y};
        screen.any_item_active = ImGui::IsAnyItemActive();

        constexpr uint8_t kUiLayoutSettleFrames = 3;
        const bool python_console_visible = window_states_["python_console"];
        const bool ui_layout_changed =
            std::abs(screen.work_pos.x - last_ui_layout_work_pos_.x) > 0.5f ||
            std::abs(screen.work_pos.y - last_ui_layout_work_pos_.y) > 0.5f ||
            std::abs(screen.work_size.x - last_ui_layout_work_size_.x) > 0.5f ||
            std::abs(screen.work_size.y - last_ui_layout_work_size_.y) > 0.5f ||
            std::abs(panel_layout_.getRightPanelWidth() - last_ui_layout_right_panel_w_) > 0.5f ||
            std::abs(panel_layout_.getScenePanelRatio() - last_ui_layout_scene_ratio_) > 0.0001f ||
            std::abs(panel_layout_.getPythonConsoleWidth() - last_ui_layout_python_console_w_) > 0.5f ||
            show_main_panel_ != last_ui_layout_show_main_panel_ ||
            ui_hidden_ != last_ui_layout_ui_hidden_ ||
            python_console_visible != last_ui_layout_python_console_visible_ ||
            panel_layout_.getActiveTab() != last_ui_layout_active_tab_;

        if (ui_layout_changed) {
            ui_layout_settle_frames_ = kUiLayoutSettleFrames;
            last_ui_layout_work_pos_ = screen.work_pos;
            last_ui_layout_work_size_ = screen.work_size;
            last_ui_layout_right_panel_w_ = panel_layout_.getRightPanelWidth();
            last_ui_layout_scene_ratio_ = panel_layout_.getScenePanelRatio();
            last_ui_layout_python_console_w_ = panel_layout_.getPythonConsoleWidth();
            last_ui_layout_show_main_panel_ = show_main_panel_;
            last_ui_layout_ui_hidden_ = ui_hidden_;
            last_ui_layout_python_console_visible_ = python_console_visible;
            last_ui_layout_active_tab_ = panel_layout_.getActiveTab();
        }

        if (show_main_panel_ && !ui_hidden_) {
            const float sbh = PanelLayoutManager::STATUS_BAR_HEIGHT * current_ui_scale_;
            const float rpw = panel_layout_.getRightPanelWidth();
            const float ph = screen.work_size.y - sbh;
            const float splitter_h = PanelLayoutManager::SPLITTER_H * current_ui_scale_;
            const float avail_h = ph - 16.0f;
            const float scene_h = std::max(80.0f * current_ui_scale_,
                                           avail_h * panel_layout_.getScenePanelRatio() - splitter_h * 0.5f);

            RightPanelLayout rp_layout;
            rp_layout.pos = glm::vec2(screen.work_pos.x + screen.work_size.x - rpw, screen.work_pos.y);
            rp_layout.size = glm::vec2(rpw, ph);
            rp_layout.scene_h = scene_h + 8.0f;
            rp_layout.splitter_h = splitter_h;

            rml_right_panel_.processInput(rp_layout, panel_input);

            if (rml_right_panel_.wantsInput())
                ImGui::GetIO().WantCaptureMouse = true;

            const auto main_tabs = reg.get_panels_for_space(PanelSpace::MainPanelTab);
            std::vector<TabSnapshot> tab_snaps;
            tab_snaps.reserve(main_tabs.size());
            for (const auto& t : main_tabs)
                tab_snaps.push_back({t.idname, t.label});

            rml_right_panel_.render(rp_layout, tab_snaps, panel_layout_.getActiveTab(),
                                    panel_input.screen_x, panel_input.screen_y,
                                    panel_input.screen_w, panel_input.screen_h);
        }

        panel_layout_.renderRightPanel(ctx, draw_ctx, show_main_panel_, ui_hidden_,
                                       window_states_, focus_panel_name_, panel_input, screen);

        applyFrameKeyboardCapture();

        // Apply cursor requests from right panel and panel layout
        auto apply_cursor = [](CursorRequest req) {
            switch (req) {
            case CursorRequest::ResizeEW: ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW); break;
            case CursorRequest::ResizeNS: ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS); break;
            default: break;
            }
        };
        apply_cursor(rml_right_panel_.getCursorRequest());
        apply_cursor(panel_layout_.getCursorRequest());

        python::set_viewport_bounds(viewport_layout_.pos.x, viewport_layout_.pos.y,
                                    viewport_layout_.size.x, viewport_layout_.size.y);

        PanelInputState floating_input = panel_input;
        floating_input.bg_draw_list = ImGui::GetForegroundDrawList(ImGui::GetMainViewport());
        reg.draw_panels(PanelSpace::Floating, draw_ctx, &floating_input);
        reg.draw_panels(PanelSpace::Dockable, draw_ctx);

        applyFrameKeyboardCapture();

        gizmo_manager_.updateToolState(ctx, ui_hidden_);
        gizmo_manager_.updateCropFlash();

        rml_viewport_overlay_.setViewportBounds(
            viewport_layout_.pos, viewport_layout_.size,
            {panel_input.screen_x, panel_input.screen_y});
        rml_viewport_overlay_.processInput();
        if (lfs::python::has_python_hooks("viewport_overlay", "draw")) {
            lfs::python::invoke_python_hooks("viewport_overlay", "draw", true);
            lfs::python::invoke_python_hooks("viewport_overlay", "draw", false);
        }
        reg.draw_panels(PanelSpace::ViewportOverlay, draw_ctx);

        rml_viewport_overlay_.render();

        applyFrameKeyboardCapture();
        drawFrameTooltip(RmlPanelHost::consumeFrameTooltip());

        // Recompute viewport layout
        viewport_layout_ = panel_layout_.computeViewportLayout(
            show_main_panel_, ui_hidden_, window_states_["python_console"], screen);

        if (!ui_hidden_) {
            reg.draw_panels(PanelSpace::StatusBar, draw_ctx);
        }

        python::draw_python_modals(scene);
        python::draw_python_popups(scene);

        rml_modal_overlay_->processInput(panel_input);
        rml_viewport_overlay_.compositeToScreen(panel_input.screen_w, panel_input.screen_h);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Clean up GL state after ImGui rendering (ImGui can leave VAO/shader bindings corrupted)
        glBindVertexArray(0);
        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        // Clear any errors ImGui might have generated
        while (glGetError() != GL_NO_ERROR) {}

        RmlPanelHost::flushQueuedForegroundComposites(panel_input.screen_w, panel_input.screen_h);
        sequencer_ui_.compositeOverlays(panel_input.screen_w, panel_input.screen_h);

        if (menu_bar_ && !ui_hidden_ && rml_menu_bar_.fbo().valid()) {
            const float menu_height = rml_menu_bar_.isOpen()
                                          ? static_cast<float>(panel_input.screen_h)
                                          : rml_menu_bar_.barHeight();
            rml_menu_bar_.fbo().blitToScreen(
                0.0f, 0.0f,
                static_cast<float>(panel_input.screen_w),
                menu_height,
                panel_input.screen_w, panel_input.screen_h);
        }

        global_context_menu_->render(panel_input.screen_w, panel_input.screen_h,
                                     panel_input.screen_x, panel_input.screen_y);

        {
            const auto* mvp_modal = ImGui::GetMainViewport();
            rml_modal_overlay_->render(static_cast<int>(mvp_modal->Size.x),
                                       static_cast<int>(mvp_modal->Size.y),
                                       mvp_modal->Pos.x, mvp_modal->Pos.y,
                                       viewport_layout_.pos.x, viewport_layout_.pos.y,
                                       viewport_layout_.size.x, viewport_layout_.size.y);
        }

        glBindVertexArray(0);
        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D, 0);
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

        if (!ui_layout_changed && ui_layout_settle_frames_ > 0)
            --ui_layout_settle_frames_;

        persistImGuiSettingsIfNeeded();
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
                const auto& world_points = rm->getPolygonWorldPoints();
                const bool closed = rm->isPolygonClosed();
                const bool add_mode = rm->isPolygonAddMode();

                if (!world_points.empty()) {
                    const auto& viewport = ctx.viewer->getViewport();
                    const glm::mat4 view = viewport.getViewMatrix();
                    const glm::mat4 proj = viewport.getProjectionMatrix(rm->getFocalLengthMm());
                    const glm::mat4 vp_mat = proj * view;

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
                    screen_points.reserve(world_points.size());
                    bool all_visible = true;
                    for (const auto& wp : world_points) {
                        const glm::vec4 clip = vp_mat * glm::vec4(wp, 1.0f);
                        if (clip.w <= 0.0f) {
                            all_visible = false;
                            break;
                        }
                        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                        screen_points.emplace_back(
                            viewport_layout_.pos.x + (ndc.x * 0.5f + 0.5f) * viewport_layout_.size.x,
                            viewport_layout_.pos.y + (1.0f - (ndc.y * 0.5f + 0.5f)) * viewport_layout_.size.y);
                    }
                    if (!all_visible) {
                        screen_points.clear();
                    }

                    const ImVec2 clip_min(viewport_layout_.pos.x, viewport_layout_.pos.y);
                    float clip_bottom = viewport_layout_.pos.y + viewport_layout_.size.y;
                    if (panel_layout_.isShowSequencer()) {
                        const float seq_top = sequencer_ui_.panelTopY();
                        if (seq_top > 0.0f) {
                            clip_bottom = std::min(clip_bottom, seq_top);
                        }
                    }
                    const ImVec2 clip_max(viewport_layout_.pos.x + viewport_layout_.size.x, clip_bottom);
                    draw_list->PushClipRect(clip_min, clip_max, true);

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

                    if (closed && screen_points.size() >= 3) {
                        float cx = 0.0f, cy = 0.0f;
                        for (const auto& sp : screen_points) {
                            cx += sp.x;
                            cy += sp.y;
                        }
                        cx /= static_cast<float>(screen_points.size());
                        cy /= static_cast<float>(screen_points.size());

                        const char* hint = "Enter to confirm";
                        const ImVec2 text_size = ImGui::CalcTextSize(hint);
                        draw_list->AddText(
                            ImVec2(cx - text_size.x * 0.5f, cy - text_size.y * 0.5f),
                            toU32WithAlpha(t.palette.text, 0.9f), hint);
                    }

                    draw_list->PopClipRect();
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
            const ImVec2 vp_pos(viewport_layout_.pos.x, viewport_layout_.pos.y);
            const ImVec2 vp_size(viewport_layout_.size.x, viewport_layout_.size.y);
            widgets::DrawViewportVignette(vp_pos, vp_size);
        }

        if (!ui_hidden_ && viewport_layout_.size.x > 0 && viewport_layout_.size.y > 0) {
            const auto& t = theme();
            const float r = t.viewport.corner_radius;
            if (r > 0.0f) {
                auto* const dl = ImGui::GetBackgroundDrawList();
                const ImU32 bg = toU32(t.menu_background());
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

                if (show_main_panel_) {
                    const float rpw = panel_layout_.getRightPanelWidth();
                    auto* mvp = ImGui::GetMainViewport();
                    const float px = mvp->WorkPos.x + mvp->WorkSize.x - rpw;
                    const float py1 = mvp->WorkPos.y;
                    const float py2 = py1 + mvp->WorkSize.y - PanelLayoutManager::STATUS_BAR_HEIGHT * current_ui_scale_;
                    maskCorner({px, py1}, {px, py1 + r}, {px + r, py1 + r}, IM_PI, IM_PI * 1.5f);
                    maskCorner({px, py2}, {px + r, py2}, {px + r, py2 - r}, IM_PI * 0.5f, IM_PI);
                }

                if (t.viewport.border_size > 0.0f) {
                    dl->AddRect({x1, y1}, {x2, y2}, t.viewport_border_u32(), r,
                                ImDrawFlags_RoundCornersAll, t.viewport.border_size);
                }
            }
        }
    }

    void GuiManager::updateInputOverrides(bool mouse_in_viewport) {
        if (rml_menu_bar_.wantsInput())
            return;

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

    glm::vec2 GuiManager::getViewportPos() const {
        return viewport_layout_.pos;
    }

    glm::vec2 GuiManager::getViewportSize() const {
        return viewport_layout_.size;
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
            if (lfs::vis::loadUiScalePreference() <= 0.0f) {
                pending_ui_scale_ = std::clamp(e.scale, 1.0f, 4.0f);
            }
        });

        internal::UiScaleChangeRequested::when([this](const auto& e) {
            if (e.scale <= 0.0f) {
                pending_ui_scale_ = std::clamp(SDL_GetWindowDisplayScale(viewer_->getWindow()), 1.0f, 4.0f);
            } else {
                pending_ui_scale_ = std::clamp(e.scale, 1.0f, 4.0f);
            }
        });

        state::DiskSpaceSaveFailed::when([this](const auto& e) {
            using namespace lichtfeld::Strings;
            if (!e.is_disk_space_error)
                return;

            auto formatBytes = [](size_t bytes) -> std::string {
                constexpr double KB = 1024.0;
                constexpr double MB = KB * 1024.0;
                constexpr double GB = MB * 1024.0;
                if (bytes >= static_cast<size_t>(GB))
                    return std::format("{:.2f} GB", static_cast<double>(bytes) / GB);
                if (bytes >= static_cast<size_t>(MB))
                    return std::format("{:.2f} MB", static_cast<double>(bytes) / MB);
                if (bytes >= static_cast<size_t>(KB))
                    return std::format("{:.2f} KB", static_cast<double>(bytes) / KB);
                return std::format("{} bytes", bytes);
            };

            const std::string subtitle = e.is_checkpoint
                                             ? std::format("{} {})", LOC(DiskSpaceDialog::CHECKPOINT_SAVE_FAILED), e.iteration)
                                             : std::string(LOC(DiskSpaceDialog::EXPORT_FAILED));

            std::string body;
            body += std::format("<div>{}</div>", LOC(DiskSpaceDialog::INSUFFICIENT_SPACE_PREFIX));
            body += std::format("<div class=\"content-row\"><span class=\"dim-text\">{} </span>{}</div>",
                                LOC(DiskSpaceDialog::LOCATION_LABEL), lfs::core::path_to_utf8(e.path.parent_path()));
            body += std::format("<div class=\"content-row\"><span class=\"dim-text\">{} </span>{}</div>",
                                LOC(DiskSpaceDialog::REQUIRED_LABEL), formatBytes(e.required_bytes));
            if (e.available_bytes > 0) {
                body += std::format("<div class=\"content-row\"><span class=\"dim-text\">{} </span>"
                                    "<span class=\"error-text\">{}</span></div>",
                                    LOC(DiskSpaceDialog::AVAILABLE_LABEL), formatBytes(e.available_bytes));
            }
            body += std::format("<div class=\"warning-text\">{}</div>", LOC(DiskSpaceDialog::INSTRUCTION));

            lfs::core::ModalRequest req;
            req.title = std::format("{} | {}", LOC(DiskSpaceDialog::ERROR_LABEL), subtitle);
            req.body_rml = body;
            req.style = lfs::core::ModalStyle::Error;
            req.width_dp = 480;
            req.buttons = {
                {LOC(DiskSpaceDialog::CANCEL), "secondary"},
                {LOC(DiskSpaceDialog::CHANGE_LOCATION), "warning"},
                {LOC(DiskSpaceDialog::RETRY), "primary"}};

            auto path = e.path;
            auto iteration = e.iteration;
            auto is_checkpoint = e.is_checkpoint;

            req.on_result = [this, path, iteration, is_checkpoint](const lfs::core::ModalResult& result) {
                if (result.button_label == LOC(DiskSpaceDialog::RETRY)) {
                    if (is_checkpoint) {
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
                    }
                } else if (result.button_label == LOC(DiskSpaceDialog::CHANGE_LOCATION)) {
                    std::filesystem::path new_location = SelectFolderDialog(
                        LOC(DiskSpaceDialog::SELECT_OUTPUT_LOCATION), path.parent_path());
                    if (!new_location.empty() && is_checkpoint) {
                        if (auto* tm = viewer_->getTrainerManager()) {
                            if (auto* trainer = tm->getTrainer()) {
                                auto params = trainer->getParams();
                                params.dataset.output_path = new_location;
                                trainer->setParams(params);
                                LOG_INFO("Output path changed to: {}", lfs::core::path_to_utf8(new_location));
                                if (tm->isFinished() || !tm->isTrainingActive())
                                    trainer->save_final_ply_and_checkpoint(iteration);
                                else
                                    tm->requestSaveCheckpoint();
                            }
                        }
                    } else if (!new_location.empty()) {
                        LOG_INFO("Re-export manually using File > Export to: {}",
                                 lfs::core::path_to_utf8(new_location));
                    }
                } else {
                    if (is_checkpoint)
                        LOG_WARN("Checkpoint save cancelled by user");
                    else
                        LOG_INFO("Export cancelled by user");
                }
            };
            req.on_cancel = [is_checkpoint]() {
                if (is_checkpoint)
                    LOG_WARN("Checkpoint save cancelled by user");
                else
                    LOG_INFO("Export cancelled by user");
            };

            rml_modal_overlay_->enqueue(std::move(req));
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
        return ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) ||
               rml_modal_overlay_->isOpen();
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
        const std::string preferred_theme = loadThemePreferenceName();
        if (!setThemeByName(preferred_theme)) {
            setTheme(darkTheme());
        }
        rmlui_manager_.activateTheme(currentThemeId());
    }

    void GuiManager::showWindow(const std::string& name, bool show) {
        window_states_[name] = show;
    }

    bool GuiManager::needsAnimationFrame() const {
        if (startup_overlay_.needsAnimationFrame())
            return true;
        if (video_extractor_dialog_ && video_extractor_dialog_->isVideoPlaying())
            return true;
        if (ui_layout_settle_frames_ > 0)
            return true;
        if (rml_right_panel_.needsAnimationFrame())
            return true;
        if (PanelRegistry::instance().needsAnimationFrame())
            return true;
        return false;
    }

    void GuiManager::dismissStartupOverlay() {
        startup_overlay_.dismiss();
    }

    void GuiManager::requestExitConfirmation() {
        startup_overlay_.dismiss();
        lfs::core::events::cmd::RequestExit{}.emit();
    }

    bool GuiManager::isExitConfirmationPending() const {
        return lfs::python::is_exit_popup_open();
    }

} // namespace lfs::vis::gui
