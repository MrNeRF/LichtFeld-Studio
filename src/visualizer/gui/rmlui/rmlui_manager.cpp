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

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include "gui/rmlui/rmlui_vk_backend.hpp"
#include "window/vulkan_context.hpp"
#endif

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/Matrix4.h>
#include <RmlUi/Debugger.h>
#include <cassert>
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
    } // namespace

    RmlUIManager::RmlUIManager() = default;

    RmlUIManager::~RmlUIManager() {
        if (initialized_)
            shutdown();
    }

    bool RmlUIManager::initVulkan(SDL_Window* window, lfs::vis::VulkanContext& vulkan_context, float dp_ratio) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
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
#else
        (void)window;
        (void)vulkan_context;
        (void)dp_ratio;
        LOG_ERROR("RmlUI Vulkan initialization requested, but Vulkan viewer dependencies are disabled");
        return false;
#endif
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
#ifdef LFS_VULKAN_VIEWER_ENABLED
            if (vulkan_render_interface_)
                vulkan_render_interface_->ShutdownExternal();
#endif
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
        cjk_fonts_loaded_ = true;

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

        for (std::size_t i = 0; i < 2; ++i) {
            LoadedFont loaded = futures[i].get();
            if (loaded.bytes.empty()) {
                LOG_WARN("RmlUI: failed to read {}", specs[i].asset);
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
            }
        }
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
#ifdef LFS_VULKAN_VIEWER_ENABLED
        if (vulkan_render_interface_)
            vulkan_render_interface_->ShutdownExternal();
#endif
        owned_render_interface_.reset();
        vulkan_render_interface_ = nullptr;
        vulkan_queue_.clear();
        vulkan_foreground_queue_.clear();
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
            if (system_interface_)
                system_interface_->releaseContext(it->second);
            if (auto fn = lfs::python::get_rml_context_destroy_handler())
                fn(it->second);
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
    }

    void RmlUIManager::trackContextFrame(const Rml::Context* const context,
                                         const int window_x,
                                         const int window_y) {
        if (system_interface_)
            system_interface_->trackContext(context, window_x, window_y);
    }

    RmlCursorRequest RmlUIManager::consumeCursorRequest() {
        return system_interface_ ? system_interface_->consumeCursorRequest()
                                 : RmlCursorRequest::None;
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
        queue.push_back({
            .context = context,
            .offset_x = offset_x,
            .offset_y = offset_y,
            .clip_enabled = clip_enabled,
            .clip_x1 = clip_x1,
            .clip_y1 = clip_y1,
            .clip_x2 = clip_x2,
            .clip_y2 = clip_y2,
        });
    }

    void RmlUIManager::clearVulkanQueue() {
        vulkan_queue_.clear();
        vulkan_foreground_queue_.clear();
    }

#ifdef LFS_VULKAN_VIEWER_ENABLED
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

            vulkan_render_interface_->ResetContextRenderState();
            if (command.clip_enabled) {
                vulkan_render_interface_->SetContextClipRect(command.clip_x1,
                                                             command.clip_y1,
                                                             command.clip_x2,
                                                             command.clip_y2);
            }
            vulkan_render_interface_->SetContextOffset(command.offset_x, command.offset_y);
            command.context->Render();
            vulkan_render_interface_->ResetContextRenderState();
        }

        vulkan_render_interface_->ResetContextRenderState();
        queue.clear();
    }

    void RmlUIManager::endVulkanFrame() {
        if (!vulkan_render_interface_ || !vulkan_frame_active_)
            return;
        vulkan_render_interface_->EndExternalFrame();
        vulkan_frame_active_ = false;
    }
#endif

} // namespace lfs::vis::gui
