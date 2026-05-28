/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rmlui_manager.hpp"
#include "config.h"
#include "core/logger.hpp"
#include "gui/rmlui/elements/chromaticity_element.hpp"
#include "gui/rmlui/elements/color_picker_element.hpp"
#include "gui/rmlui/elements/crf_curve_element.hpp"
#include "gui/rmlui/elements/loss_graph_element.hpp"
#include "gui/rmlui/elements/python_editor_element.hpp"
#include "gui/rmlui/elements/scene_graph_element.hpp"
#include "gui/rmlui/elements/terminal_element.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_text_input_handler.hpp"
#include "gui/rmlui/rmlui_system_interface.hpp"
#include "internal/resource_paths.hpp"
#include "python/python_runtime.hpp"

#include "gui/rmlui/rmlui_vk_backend.hpp"
#include "window/vulkan_context.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/Matrix4.h>
#include <RmlUi/Debugger.h>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <string_view>
#include <vector>

namespace lfs::vis::gui {

    namespace {
        bool envFlagEnabled(const char* name) {
            const char* value = std::getenv(name);
            if (!value || !*value)
                return false;
            return std::string_view(value) != "0";
        }

        std::string timerSafeContextName(const std::string_view name) {
            if (name.empty())
                return "unknown";

            std::string safe;
            safe.reserve(name.size());
            for (const unsigned char ch : name) {
                if (std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.') {
                    safe.push_back(static_cast<char>(ch));
                } else {
                    safe.push_back('_');
                }
            }
            return safe;
        }
    } // namespace

    RmlUIManager::RmlUIManager() = default;

    RmlUIManager::~RmlUIManager() {
        if (initialized_)
            shutdown();
    }

    bool RmlUIManager::initVulkan(SDL_Window* window, lfs::vis::VulkanContext& vulkan_context, float dp_ratio) {
        auto render_interface = std::make_unique<RenderInterface_VK>();
        RenderInterface_VK::ExternalContext context{};
        context.instance = vulkan_context.instance();
        context.physical_device = vulkan_context.physicalDevice();
        context.device = vulkan_context.device();
        context.pipeline_cache = vulkan_context.pipelineCache();
        context.graphics_queue = vulkan_context.graphicsQueue();
        context.graphics_queue_family = vulkan_context.graphicsQueueFamily();
        context.color_format = vulkan_context.swapchainFormat();
        context.depth_stencil_format = vulkan_context.depthStencilFormat();
        context.extent = vulkan_context.swapchainExtent();

        auto* vulkan_render_interface = render_interface.get();
        if (!vulkan_render_interface->InitializeExternal(context)) {
            LOG_ERROR("Failed to initialize RmlUI Vulkan render interface");
            return false;
        }

        return initWithRenderInterface(window, dp_ratio, std::move(render_interface), vulkan_render_interface);
    }

    bool RmlUIManager::initWithRenderInterface(SDL_Window* window,
                                               float dp_ratio,
                                               std::unique_ptr<Rml::RenderInterface> render_interface,
                                               RenderInterface_VK* vulkan_render_interface) {
        assert(!initialized_);
        assert(window);
        assert(dp_ratio >= 1.0f);
        assert(render_interface);

        dp_ratio_ = dp_ratio;
        window_ = window;
        debugger_enabled_ = envFlagEnabled("LFS_RML_DEBUGGER");

        system_interface_ = std::make_unique<RmlSystemInterface>(window);
        owned_render_interface_ = std::move(render_interface);
        vulkan_render_interface_ = vulkan_render_interface;
        text_input_handler_ = std::make_unique<RmlTextInputHandler>();

        Rml::SetSystemInterface(system_interface_.get());
        Rml::SetRenderInterface(owned_render_interface_.get());
        Rml::SetTextInputHandler(text_input_handler_.get());

        if (!Rml::Initialise()) {
            LOG_ERROR("Failed to initialize RmlUI");
            if (vulkan_render_interface_)
                vulkan_render_interface_->ShutdownExternal();
            owned_render_interface_.reset();
            vulkan_render_interface_ = nullptr;
            text_input_handler_.reset();
            system_interface_.reset();
            return false;
        }

        static Rml::ElementInstancerGeneric<ChromaticityElement> chromaticity_instancer;
        static Rml::ElementInstancerGeneric<ColorPickerElement> color_picker_instancer;
        static Rml::ElementInstancerGeneric<CRFCurveElement> crf_curve_instancer;
        static Rml::ElementInstancerGeneric<LossGraphElement> loss_graph_instancer;
        static Rml::ElementInstancerGeneric<PythonEditorElement> python_editor_instancer;
        static Rml::ElementInstancerGeneric<SceneGraphElement> scene_graph_instancer;
        static Rml::ElementInstancerGeneric<TerminalElement> terminal_instancer;
        Rml::Factory::RegisterElementInstancer("chromaticity-diagram", &chromaticity_instancer);
        Rml::Factory::RegisterElementInstancer("color-picker", &color_picker_instancer);
        Rml::Factory::RegisterElementInstancer("crf-curve", &crf_curve_instancer);
        Rml::Factory::RegisterElementInstancer("loss-graph", &loss_graph_instancer);
        Rml::Factory::RegisterElementInstancer("python-editor-view", &python_editor_instancer);
        Rml::Factory::RegisterElementInstancer("scene-graph", &scene_graph_instancer);
        Rml::Factory::RegisterElementInstancer("terminal-view", &terminal_instancer);

        try {
            struct FontSpec {
                const char* asset;
                const char* family;
                Rml::Style::FontStyle style;
                Rml::Style::FontWeight weight;
                bool fallback;
            };
            const FontSpec specs[] = {
                {"fonts/Inter-Regular.ttf", "Inter", Rml::Style::FontStyle::Normal, Rml::Style::FontWeight::Normal, true},
                {"fonts/Inter-SemiBold.ttf", "Inter", Rml::Style::FontStyle::Normal, Rml::Style::FontWeight(600), false},
                {"fonts/JetBrainsMono-Regular.ttf", "JetBrains Mono", Rml::Style::FontStyle::Normal, Rml::Style::FontWeight::Normal, false},
            };
            constexpr std::size_t kFontCount = sizeof(specs) / sizeof(specs[0]);

            struct LoadedFont {
                std::filesystem::path path;
                std::vector<std::byte> bytes;
            };
            std::array<std::future<LoadedFont>, kFontCount> futures;
            for (std::size_t i = 0; i < kFontCount; ++i) {
                const char* asset = specs[i].asset;
                futures[i] = std::async(std::launch::async, [asset]() {
                    LoadedFont out;
                    out.path = lfs::vis::getAssetPath(asset);
                    std::ifstream f(out.path, std::ios::binary | std::ios::ate);
                    if (!f)
                        return out;
                    const auto sz = f.tellg();
                    if (sz <= 0)
                        return out;
                    f.seekg(0, std::ios::beg);
                    out.bytes.resize(static_cast<std::size_t>(sz));
                    f.read(reinterpret_cast<char*>(out.bytes.data()), sz);
                    return out;
                });
            }

            font_blobs_.reserve(kFontCount);
            for (std::size_t i = 0; i < kFontCount; ++i) {
                LoadedFont loaded = futures[i].get();
                if (loaded.bytes.empty()) {
                    LOG_WARN("RmlUI: failed to read {}", specs[i].asset);
                    continue;
                }
                font_blobs_.push_back(std::move(loaded.bytes));
                const auto& blob = font_blobs_.back();
                Rml::Span<const Rml::byte> data{
                    reinterpret_cast<const Rml::byte*>(blob.data()), blob.size()};
                if (Rml::LoadFontFace(data, specs[i].family, specs[i].style, specs[i].weight, specs[i].fallback)) {
                    LOG_INFO("RmlUI: loaded font {}", loaded.path.string());
                } else {
                    LOG_WARN("RmlUI: failed to register {}", loaded.path.string());
                }
            }
        } catch (const std::exception& e) {
            LOG_WARN("RmlUI: font load error: {}", e.what());
        }

        initialized_ = true;
        LOG_INFO("RmlUI initialized");
        return true;
    }

    void RmlUIManager::ensureCjkFontsLoaded() {
        if (cjk_fonts_loaded_ || !initialized_)
            return;

        struct CjkSpec {
            const char* asset;
            const char* family;
        };
        const CjkSpec specs[] = {
            {"fonts/NotoSansJP-Regular.ttf", "Noto Sans JP"},
            {"fonts/NotoSansKR-Regular.ttf", "Noto Sans KR"},
        };

        struct LoadedFont {
            std::filesystem::path path;
            std::vector<std::byte> bytes;
        };
        std::array<std::future<LoadedFont>, 2> futures;
        for (std::size_t i = 0; i < 2; ++i) {
            const char* asset = specs[i].asset;
            futures[i] = std::async(std::launch::async, [asset]() {
                LoadedFont out;
                try {
                    out.path = lfs::vis::getAssetPath(asset);
                    std::ifstream f(out.path, std::ios::binary | std::ios::ate);
                    if (!f)
                        return out;
                    const auto sz = f.tellg();
                    if (sz <= 0)
                        return out;
                    f.seekg(0, std::ios::beg);
                    out.bytes.resize(static_cast<std::size_t>(sz));
                    f.read(reinterpret_cast<char*>(out.bytes.data()), sz);
                } catch (...) {
                }
                return out;
            });
        }

        bool all_loaded = true;
        for (std::size_t i = 0; i < 2; ++i) {
            LoadedFont loaded = futures[i].get();
            if (loaded.bytes.empty()) {
                LOG_WARN("RmlUI: failed to read {}", specs[i].asset);
                all_loaded = false;
                continue;
            }
            font_blobs_.push_back(std::move(loaded.bytes));
            const auto& blob = font_blobs_.back();
            Rml::Span<const Rml::byte> data{
                reinterpret_cast<const Rml::byte*>(blob.data()), blob.size()};
            if (Rml::LoadFontFace(data, specs[i].family, Rml::Style::FontStyle::Normal,
                                  Rml::Style::FontWeight::Normal, true)) {
                LOG_INFO("RmlUI: loaded CJK font {}", loaded.path.string());
            } else {
                LOG_WARN("RmlUI: failed to register {}", loaded.path.string());
                all_loaded = false;
            }
        }
        cjk_fonts_loaded_ = all_loaded;
    }

    void RmlUIManager::shutdown() {
        if (!initialized_)
            return;

        if (debugger_initialized_) {
            Rml::Debugger::Shutdown();
            debugger_initialized_ = false;
        }

        while (!contexts_.empty())
            destroyContext(contexts_.begin()->first);

        if (Rml::GetTextInputHandler() == text_input_handler_.get())
            Rml::SetTextInputHandler(nullptr);
        Rml::Shutdown();
        if (vulkan_render_interface_)
            vulkan_render_interface_->ShutdownExternal();
        owned_render_interface_.reset();
        vulkan_render_interface_ = nullptr;
        vulkan_queue_.clear();
        vulkan_foreground_queue_.clear();
        context_names_.clear();
        text_input_handler_.reset();
        system_interface_.reset();
        resize_deferring_ = false;
        vulkan_frame_active_ = false;
        initialized_ = false;

        LOG_INFO("RmlUI shut down");
    }

    void RmlUIManager::setDpRatio(float ratio) {
        assert(ratio >= 1.0f);
        if (!initialized_)
            return;
        dp_ratio_ = ratio;
        for (auto& [name, ctx] : contexts_) {
            ctx->SetDensityIndependentPixelRatio(ratio);
        }
    }

    Rml::Context* RmlUIManager::createContext(const std::string& name, int width, int height) {
        assert(initialized_);

        auto it = contexts_.find(name);
        if (it != contexts_.end()) {
            return it->second;
        }

        Rml::Context* ctx = Rml::CreateContext(name, Rml::Vector2i(width, height));
        if (!ctx) {
            LOG_ERROR("RmlUI: failed to create context '{}'", name);
            return nullptr;
        }

        ctx->SetDensityIndependentPixelRatio(dp_ratio_);
        ctx->SetDefaultScrollBehavior(Rml::ScrollBehavior::Instant, 1.0f);
        if (!active_theme_id_.empty())
            ctx->ActivateTheme(active_theme_id_, true);

        if (debugger_enabled_ && !debugger_initialized_) {
            debugger_initialized_ = Rml::Debugger::Initialise(ctx);
            if (debugger_initialized_) {
                Rml::Debugger::SetVisible(true);
                LOG_INFO("RmlUI debugger enabled on context '{}'", name);
            } else {
                LOG_WARN("RmlUI debugger requested but failed to initialize on context '{}'", name);
            }
        }

        contexts_[name] = ctx;
        context_names_[ctx] = timerSafeContextName(name);
        return ctx;
    }

    Rml::Context* RmlUIManager::getContext(const std::string& name) {
        auto it = contexts_.find(name);
        return it != contexts_.end() ? it->second : nullptr;
    }

    void RmlUIManager::destroyContext(const std::string& name) {
        if (!initialized_)
            return;

        auto it = contexts_.find(name);
        if (it != contexts_.end()) {
            Rml::Context* const context = it->second;
            if (system_interface_)
                system_interface_->releaseContext(context);
            if (auto fn = lfs::python::get_rml_context_destroy_handler())
                fn(context);
            context_names_.erase(context);
            tracked_context_frames_.erase(context);
            Rml::RemoveContext(name);
            contexts_.erase(it);
        }
    }

    void RmlUIManager::activateTheme(const std::string& theme_id) {
        if (theme_id == active_theme_id_)
            return;
        for (auto& [name, ctx] : contexts_) {
            if (!active_theme_id_.empty())
                ctx->ActivateTheme(active_theme_id_, false);
            ctx->ActivateTheme(theme_id, true);
        }
        active_theme_id_ = theme_id;
    }

    void RmlUIManager::beginFrameCursorTracking() {
        if (system_interface_)
            system_interface_->beginFrame();
        tracked_context_frames_.clear();
        tracked_context_order_ = 0;
    }

    void RmlUIManager::trackContextFrame(const Rml::Context* const context,
                                         const int window_x,
                                         const int window_y) {
        if (system_interface_)
            system_interface_->trackContext(context, window_x, window_y);
        if (!context)
            return;

        const auto dimensions = context->GetDimensions();
        auto& frame = tracked_context_frames_[context];
        const bool needs_passive_frames = frame.needs_passive_mouse_move_frames;
        frame = TrackedContextFrame{
            .context = const_cast<Rml::Context*>(context),
            .window_x = window_x,
            .window_y = window_y,
            .width = dimensions.x,
            .height = dimensions.y,
            .order = ++tracked_context_order_,
            .needs_passive_mouse_move_frames = needs_passive_frames,
        };
    }

    void RmlUIManager::setContextNeedsPassiveMouseMoveFrames(
        const Rml::Context* const context,
        const bool needs_frames) {
        if (!context)
            return;
        if (auto it = tracked_context_frames_.find(context); it != tracked_context_frames_.end())
            it->second.needs_passive_mouse_move_frames = needs_frames;
    }

    RmlCursorRequest RmlUIManager::consumeCursorRequest() {
        return system_interface_ ? system_interface_->consumeCursorRequest()
                                 : RmlCursorRequest::None;
    }

    bool RmlUIManager::passiveMouseMoveNeedsRender(const float window_x,
                                                   const float window_y) const {
        if (tracked_context_frames_.empty())
            return true;

        const TrackedContextFrame* top_context = nullptr;
        bool any_active_context = false;
        for (const auto& [_, frame] : tracked_context_frames_) {
            auto* const context = frame.context;
            if (!context)
                continue;

            auto* const hover = context->GetHoverElement();
            if (frame.needs_passive_mouse_move_frames ||
                (hover && hover->GetTagName() != "body")) {
                any_active_context = true;
            }

            if (frame.width <= 0 || frame.height <= 0)
                continue;

            const float local_x = window_x - static_cast<float>(frame.window_x);
            const float local_y = window_y - static_cast<float>(frame.window_y);
            if (local_x < 0.0f || local_y < 0.0f ||
                local_x >= static_cast<float>(frame.width) ||
                local_y >= static_cast<float>(frame.height)) {
                continue;
            }

            if (!top_context || frame.order > top_context->order)
                top_context = &frame;
        }

        if (!top_context)
            return any_active_context;

        auto* const context = top_context->context;
        if (!context)
            return true;
        if (top_context->needs_passive_mouse_move_frames)
            return true;

        auto* const current_hover = context->GetHoverElement();
        auto* const next_hover =
            context->GetElementAtPoint(Rml::Vector2f{
                window_x - static_cast<float>(top_context->window_x),
                window_y - static_cast<float>(top_context->window_y),
            });
        return next_hover != current_hover;
    }

    bool RmlUIManager::wantsCaptureMouse() const {
        for (const auto& [_, context] : contexts_) {
            if (!context)
                continue;
            auto* const hover = context->GetHoverElement();
            if (hover && hover->GetTagName() != "body")
                return true;
        }
        return false;
    }

    bool RmlUIManager::wantsCaptureKeyboard() const {
        for (const auto& [_, context] : contexts_) {
            if (!context)
                continue;
            if (rml_input::hasFocusedKeyboardTarget(context->GetFocusElement()))
                return true;
        }
        return false;
    }

    bool RmlUIManager::wantsTextInput() const {
        for (const auto& [_, context] : contexts_) {
            if (!context)
                continue;
            if (rml_input::wantsTextInput(context->GetFocusElement()))
                return true;
        }
        return false;
    }

    bool RmlUIManager::anyItemActive() const {
        for (const auto& [_, context] : contexts_) {
            if (!context)
                continue;
            if (rml_input::hasFocusedKeyboardTarget(context->GetFocusElement()))
                return true;
        }
        return false;
    }

    void RmlUIManager::queueVulkanContext(Rml::Context* const context,
                                          const float offset_x,
                                          const float offset_y,
                                          const bool foreground,
                                          const bool clip_enabled,
                                          const float clip_x1,
                                          const float clip_y1,
                                          const float clip_x2,
                                          const float clip_y2) {
        if (!context || !vulkan_render_interface_)
            return;
        auto& queue = foreground ? vulkan_foreground_queue_ : vulkan_queue_;
        std::string context_name = "unknown";
        if (const auto it = context_names_.find(context); it != context_names_.end())
            context_name = it->second;
        queue.push_back({
            .context = context,
            .context_name = std::move(context_name),
            .offset_x = offset_x,
            .offset_y = offset_y,
            .clip_enabled = clip_enabled,
            .clip_x1 = clip_x1,
            .clip_y1 = clip_y1,
            .clip_x2 = clip_x2,
            .clip_y2 = clip_y2,
        });
    }

    void RmlUIManager::queueCachedVulkanContext(const CachedVulkanContextDraw& draw) {
        if (!draw.context || !draw.cache || !vulkan_render_interface_ ||
            draw.cache_width <= 0 || draw.cache_height <= 0 ||
            draw.draw_width <= 0.0f || draw.draw_height <= 0.0f)
            return;

        auto& queue = draw.foreground ? vulkan_foreground_queue_ : vulkan_queue_;
        std::string context_name = "unknown";
        if (const auto it = context_names_.find(draw.context); it != context_names_.end())
            context_name = it->second;
        queue.push_back({
            .context = draw.context,
            .context_name = std::move(context_name),
            .offset_x = draw.offset_x,
            .offset_y = draw.offset_y,
            .clip_enabled = draw.clip_enabled,
            .clip_x1 = draw.clip.x1,
            .clip_y1 = draw.clip.y1,
            .clip_x2 = draw.clip.x2,
            .clip_y2 = draw.clip.y2,
            .cache = draw.cache,
            .cache_width = draw.cache_width,
            .cache_height = draw.cache_height,
            .draw_width = draw.draw_width,
            .draw_height = draw.draw_height,
            .refresh_cache = draw.refresh,
        });
    }

    void RmlUIManager::releaseCachedVulkanContext(CachedVulkanContextRender& cache) {
        if (vulkan_render_interface_ && cache.texture != 0)
            vulkan_render_interface_->ReleaseTexture(cache.texture);
        cache.texture = {};
        cache.width = 0;
        cache.height = 0;
    }

    void RmlUIManager::clearVulkanQueue() {
        vulkan_queue_.clear();
        vulkan_foreground_queue_.clear();
    }

    bool RmlUIManager::beginVulkanFrame(const VkCommandBuffer command_buffer,
                                        const VkExtent2D extent,
                                        const VkImage swapchain_image,
                                        const VkImageView swapchain_image_view,
                                        const VkImageView depth_stencil_image_view,
                                        const std::size_t frame_slot) {
        if (!vulkan_render_interface_ || command_buffer == VK_NULL_HANDLE || swapchain_image_view == VK_NULL_HANDLE ||
            depth_stencil_image_view == VK_NULL_HANDLE)
            return false;
        vulkan_render_interface_->BeginExternalFrame(command_buffer,
                                                     extent,
                                                     swapchain_image,
                                                     swapchain_image_view,
                                                     depth_stencil_image_view,
                                                     frame_slot);
        vulkan_frame_active_ = true;
        return true;
    }

    void RmlUIManager::renderQueuedVulkanContexts(const bool foreground) {
        auto& queue = foreground ? vulkan_foreground_queue_ : vulkan_queue_;
        if (!vulkan_render_interface_ || !vulkan_frame_active_) {
            queue.clear();
            return;
        }

        for (const auto& command : queue) {
            if (!command.context)
                continue;

            const std::string timer_name = std::string("gui_render.rmlui_record.") +
                                           (foreground ? "foreground.context." : "background.context.") +
                                           command.context_name;
            if (command.cache) {
                const bool refresh_cache =
                    command.refresh_cache ||
                    command.cache->texture == 0 ||
                    command.cache->width != command.cache_width ||
                    command.cache->height != command.cache_height;
                if (refresh_cache) {
                    lfs::core::ScopedTimer timer(timer_name + ".cache_refresh", 0.25);
                    if (command.cache->texture != 0)
                        releaseCachedVulkanContext(*command.cache);

                    vulkan_render_interface_->ResetContextRenderState();
                    const Rml::LayerHandle layer = vulkan_render_interface_->PushLayer();
                    if (layer != 0) {
                        vulkan_render_interface_->SetContextOffset(0.0f, 0.0f);
                        vulkan_render_interface_->SetContextClipRect(0.0f,
                                                                     0.0f,
                                                                     static_cast<float>(command.cache_width),
                                                                     static_cast<float>(command.cache_height));
                        command.context->Render();
                        command.cache->texture = vulkan_render_interface_->SaveLayerAsTexture();
                        vulkan_render_interface_->PopLayer();
                        const bool saved = command.cache->texture != 0;
                        command.cache->width = saved ? command.cache_width : 0;
                        command.cache->height = saved ? command.cache_height : 0;
                    }
                }

                if (command.cache->texture != 0) {
                    const std::string blit_timer_name =
                        std::string("gui_render.rmlui_record.") +
                        (foreground ? "foreground.cached_context." : "background.cached_context.") +
                        command.context_name;
                    lfs::core::ScopedTimer timer(blit_timer_name, 0.25);
                    vulkan_render_interface_->ResetContextRenderState();
                    if (command.clip_enabled) {
                        vulkan_render_interface_->SetContextClipRect(command.clip_x1,
                                                                     command.clip_y1,
                                                                     command.clip_x2,
                                                                     command.clip_y2);
                    }
                    vulkan_render_interface_->RenderTextureQuad(command.cache->texture,
                                                                command.offset_x,
                                                                command.offset_y,
                                                                command.draw_width,
                                                                command.draw_height);
                } else {
                    lfs::core::ScopedTimer timer(timer_name);
                    vulkan_render_interface_->ResetContextRenderState();
                    if (command.clip_enabled) {
                        vulkan_render_interface_->SetContextClipRect(command.clip_x1,
                                                                     command.clip_y1,
                                                                     command.clip_x2,
                                                                     command.clip_y2);
                    }
                    vulkan_render_interface_->SetContextOffset(command.offset_x, command.offset_y);
                    command.context->Render();
                }
            } else {
                lfs::core::ScopedTimer timer(timer_name);
                vulkan_render_interface_->ResetContextRenderState();
                if (command.clip_enabled) {
                    vulkan_render_interface_->SetContextClipRect(command.clip_x1,
                                                                 command.clip_y1,
                                                                 command.clip_x2,
                                                                 command.clip_y2);
                }
                vulkan_render_interface_->SetContextOffset(command.offset_x, command.offset_y);
                command.context->Render();
            }
            vulkan_render_interface_->ResetContextRenderState();
        }

        queue.clear();
    }

    void RmlUIManager::endVulkanFrame() {
        if (!vulkan_render_interface_ || !vulkan_frame_active_)
            return;
        vulkan_render_interface_->EndExternalFrame();
        vulkan_frame_active_ = false;
    }

} // namespace lfs::vis::gui
