/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rmlui_manager.hpp"
#include "config.h"
#include "core/environment.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/rmlui/elements/chromaticity_element.hpp"
#include "gui/rmlui/elements/color_picker_element.hpp"
#include "gui/rmlui/elements/colour_offset_element.hpp"
#include "gui/rmlui/elements/crf_curve_element.hpp"
#include "gui/rmlui/elements/loss_graph_element.hpp"
#include "gui/rmlui/elements/node_canvas_element.hpp"
#include "gui/rmlui/elements/node_curve_element.hpp"
#include "gui/rmlui/elements/python_editor_element.hpp"
#include "gui/rmlui/elements/scene_graph_element.hpp"
#include "gui/rmlui/elements/terminal_element.hpp"
#include "gui/rmlui/elements/vram_timeline_element.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_text_input_handler.hpp"
#include "gui/rmlui/rmlui_system_interface.hpp"
#include "internal/resource_paths.hpp"
#include "python/python_runtime.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/nodes/modifier_manager.hpp"

#include "gui/rmlui/rmlui_vk_backend.hpp"
#include "window/vulkan_context.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/Elements/ElementFormControlTextArea.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/Matrix4.h>
#include <RmlUi/Debugger.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <string_view>
#include <vector>

namespace lfs::vis::gui {

    namespace {
        struct InputCallbackScope {
            bool& active;
            const bool previous;
            explicit InputCallbackScope(bool& flag) : active(flag), previous(std::exchange(flag, true)) {}
            ~InputCallbackScope() { active = previous; }
        };

        constexpr const char* INPUT_LIFECYCLE_EVENTS[] = {"focus", "mousedown", "dragstart", "dragend", "hide", "unload"};

        // Text field values never pass through TranslateString, so typed, pasted
        // and bound values report emoji here for the fallback font.
        template <typename FormControl>
        class TextFieldNotingEmoji final : public FormControl {
        public:
            using FormControl::FormControl;

        protected:
            void OnAttributeChange(const Rml::ElementAttributes& changed_attributes) override {
                FormControl::OnAttributeChange(changed_attributes);
                const auto value = changed_attributes.find("value");
                if (value == changed_attributes.end())
                    return;
                if (auto* const system_interface = dynamic_cast<RmlSystemInterface*>(Rml::GetSystemInterface()))
                    system_interface->noteShownText(value->second.Get<Rml::String>());
            }
        };

        bool pointInRect(const RmlRect& rect, const float x, const float y) {
            return x >= rect.x1 && y >= rect.y1 && x < rect.x2 && y < rect.y2;
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

    RmlUIManager::RmlUIManager(SDL_Window* window) : window_(window) {}

    RmlUIManager::~RmlUIManager() {
        if (initialized_ || !input_handlers_.empty())
            shutdown();
    }

    std::uint64_t RmlUIManager::beginDragPayload(std::string type,
                                                 std::string data,
                                                 std::string label) {
        if (type.empty() || data.empty())
            return 0;
        if (current_drag_context_id_ &&
            (!contextById(current_drag_context_id_) ||
             std::ranges::any_of(pending_pointer_cancellations_, [this](const auto& pending) {
                 return pending.context_id == current_drag_context_id_;
             })))
            return 0;
        std::scoped_lock lock(drag_payload_mutex_);
        const std::uint64_t token = next_drag_payload_token_++;
        if (next_drag_payload_token_ == 0)
            next_drag_payload_token_ = 1;
        drag_payload_context_id_ = current_drag_context_id_;
        drag_payload_ = RmlDragPayload{
            .token = token,
            .type = std::move(type),
            .data = std::move(data),
            .label = std::move(label),
        };
        return token;
    }

    bool RmlUIManager::endDragPayload(const std::uint64_t token) {
        std::scoped_lock lock(drag_payload_mutex_);
        if (!drag_payload_ || drag_payload_->token != token)
            return false;
        drag_payload_->released = true;
        return true;
    }

    bool RmlUIManager::cancelDragPayload(const std::uint64_t token) {
        std::scoped_lock lock(drag_payload_mutex_);
        if (!drag_payload_ || drag_payload_->token != token)
            return false;
        drag_payload_.reset();
        return true;
    }

    void RmlUIManager::cancelDragPayload() {
        if (active_scene_graph_element_)
            active_scene_graph_element_->cancelDrag();
        std::scoped_lock lock(drag_payload_mutex_);
        drag_payload_.reset();
    }

    std::optional<RmlDragPayload> RmlUIManager::dragPayload() const {
        std::scoped_lock lock(drag_payload_mutex_);
        return drag_payload_;
    }

    std::optional<RmlDragPayload> RmlUIManager::takeReleasedDragPayload() {
        std::scoped_lock lock(drag_payload_mutex_);
        if (!drag_payload_ || !drag_payload_->released)
            return std::nullopt;
        auto result = std::move(drag_payload_);
        drag_payload_.reset();
        return result;
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
        context.extent = vulkan_context.framebufferExtent();
        context.host_image_copy = vulkan_context.hasHostImageCopy();

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
        debugger_enabled_ = lfs::core::environment::flag("LFS_RML_DEBUGGER");

        system_interface_ = std::make_unique<RmlSystemInterface>(window);
        owned_render_interface_ = std::move(render_interface);
        vulkan_render_interface_ = vulkan_render_interface;
        text_input_handler_ = std::make_unique<RmlTextInputHandler>([this] { return accepts_text_activation_; });

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

        static Rml::ElementInstancerGeneric<TextFieldNotingEmoji<Rml::ElementFormControlInput>> input_instancer;
        static Rml::ElementInstancerGeneric<TextFieldNotingEmoji<Rml::ElementFormControlTextArea>> textarea_instancer;
        Rml::Factory::RegisterElementInstancer("input", &input_instancer);
        Rml::Factory::RegisterElementInstancer("textarea", &textarea_instancer);
        static Rml::ElementInstancerGeneric<ChromaticityElement> chromaticity_instancer;
        static Rml::ElementInstancerGeneric<ColorPickerElement> color_picker_instancer;
        static Rml::ElementInstancerGeneric<ColourOffsetElement> colour_offset_instancer;
        static Rml::ElementInstancerGeneric<NodeCurveElement> node_curve_instancer;
        static Rml::ElementInstancerGeneric<CRFCurveElement> crf_curve_instancer;
        static Rml::ElementInstancerGeneric<LossGraphElement> loss_graph_instancer;
        static Rml::ElementInstancerGeneric<NodeCanvasElement> node_canvas_instancer;
        static Rml::ElementInstancerGeneric<VramTimelineElement> vram_timeline_instancer;
        static Rml::ElementInstancerGeneric<PythonEditorElement> python_editor_instancer;
        static Rml::ElementInstancerGeneric<SceneGraphElement> scene_graph_instancer;
        static Rml::ElementInstancerGeneric<TerminalElement> terminal_instancer;
        Rml::Factory::RegisterElementInstancer("chromaticity-diagram", &chromaticity_instancer);
        Rml::Factory::RegisterElementInstancer("color-picker", &color_picker_instancer);
        Rml::Factory::RegisterElementInstancer("colour-offset", &colour_offset_instancer);
        Rml::Factory::RegisterElementInstancer("node-curve", &node_curve_instancer);
        Rml::Factory::RegisterElementInstancer("crf-curve", &crf_curve_instancer);
        Rml::Factory::RegisterElementInstancer("loss-graph", &loss_graph_instancer);
        Rml::Factory::RegisterElementInstancer("node-canvas", &node_canvas_instancer);
        Rml::Factory::RegisterElementInstancer("vram-timeline", &vram_timeline_instancer);
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
        if (cjk_fonts_loaded_ || cjk_fonts_load_attempted_ || !initialized_)
            return;
        cjk_fonts_load_attempted_ = true;

        struct CjkSpec {
            const char* asset;
            const char* family;
        };
        constexpr std::array<CjkSpec, 2> specs = {{
            {"fonts/NotoSansJP-Regular.ttf", "Noto Sans JP"},
            {"fonts/NotoSansKR-Regular.ttf", "Noto Sans KR"},
        }};

        struct LoadedFont {
            std::filesystem::path path;
            std::vector<std::byte> bytes;
        };
        std::array<std::future<LoadedFont>, specs.size()> futures;
        for (std::size_t i = 0; i < specs.size(); ++i) {
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

        bool any_loaded = false;
        font_blobs_.reserve(font_blobs_.size() + specs.size());
        for (std::size_t i = 0; i < specs.size(); ++i) {
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
                any_loaded = true;
            } else {
                LOG_WARN("RmlUI: failed to register {}", loaded.path.string());
                font_blobs_.pop_back();
            }
        }
        cjk_fonts_loaded_ = any_loaded;
        if (any_loaded)
            Rml::ReleaseFontResources();
    }

    namespace {
        std::string systemEmojiFontPath() {
#ifdef _WIN32
            const char* const windir = std::getenv("WINDIR");
            const std::filesystem::path candidate =
                std::filesystem::path(windir ? windir : "C:\\Windows") / "Fonts" / "seguiemj.ttf";
            std::error_code ec;
            return std::filesystem::is_regular_file(candidate, ec) ? lfs::core::path_to_utf8(candidate) : std::string{};
#elif defined(__APPLE__)
            // Apple Color Emoji is a collection of roughly 180 MB; holding it in memory is not worth a fallback.
            return {};
#else
            constexpr std::array<const char*, 7> candidates = {
                "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
                "/usr/share/fonts/noto/NotoColorEmoji.ttf",
                "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
                "/usr/share/fonts/google-noto-color-emoji-fonts/NotoColorEmoji.ttf",
                "/usr/share/fonts/noto-emoji/NotoColorEmoji.ttf",
                "/usr/share/fonts/TTF/NotoColorEmoji.ttf",
                "/usr/local/share/fonts/NotoColorEmoji.ttf",
            };
            for (const char* const candidate : candidates) {
                std::error_code ec;
                if (std::filesystem::is_regular_file(candidate, ec))
                    return candidate;
            }
            return {};
#endif
        }

        std::vector<std::byte> readFontFile(const std::string& path) {
            std::vector<std::byte> bytes;
            std::ifstream f(lfs::core::utf8_to_path(path), std::ios::binary | std::ios::ate);
            if (!f)
                return bytes;
            const auto size = f.tellg();
            if (size <= 0)
                return bytes;
            f.seekg(0, std::ios::beg);
            bytes.resize(static_cast<std::size_t>(size));
            if (!f.read(reinterpret_cast<char*>(bytes.data()), size))
                bytes.clear();
            return bytes;
        }
    } // namespace

    void RmlUIManager::serviceEmojiFont() {
        if (emoji_font_settled_ || !initialized_)
            return;
        if (!emoji_font_read_.valid()) {
            if (!system_interface_ || !system_interface_->sawAstralText())
                return;
            emoji_font_path_ = systemEmojiFontPath();
            if (emoji_font_path_.empty()) {
                LOG_INFO("RmlUI: no system color emoji font found; emoji show as missing glyphs");
                emoji_font_settled_ = true;
                return;
            }
            emoji_font_read_ = std::async(std::launch::async, readFontFile, emoji_font_path_);
            return;
        }
        if (emoji_font_read_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return;
        emoji_font_settled_ = true;
        auto bytes = emoji_font_read_.get();
        if (bytes.empty()) {
            LOG_WARN("RmlUI: failed to read emoji font {}", emoji_font_path_);
            return;
        }
        font_blobs_.push_back(std::move(bytes));
        const auto& blob = font_blobs_.back();
        const Rml::Span<const Rml::byte> data{reinterpret_cast<const Rml::byte*>(blob.data()), blob.size()};
        if (!Rml::LoadFontFace(data, "Emoji", Rml::Style::FontStyle::Normal, Rml::Style::FontWeight::Normal, true)) {
            LOG_WARN("RmlUI: failed to register emoji font {}", emoji_font_path_);
            font_blobs_.pop_back();
            return;
        }
        LOG_INFO("RmlUI: loaded emoji font {}", emoji_font_path_);
        Rml::ReleaseFontResources();
    }

    void RmlUIManager::shutdown() {
        cancelDragPayload();
        // Input routing can also borrow externally owned contexts without init.
        for (const auto& [context, _] : input_handlers_)
            for (const auto* type : INPUT_LIFECYCLE_EVENTS)
                context->GetRootElement()->RemoveEventListener(type, this, true);
        if (!initialized_) {
            input_handlers_.clear();
            context_ids_.clear();
            return;
        }
        if (debugger_initialized_) {
            Rml::Debugger::Shutdown();
            debugger_initialized_ = false;
        }
        while (!contexts_.empty()) {
            destroyContext(contexts_.begin()->first);
            flushInputLifecycle();
        }
        input_handlers_.clear();
        key_owners_.clear();
        keyboard_context_ = nullptr;

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
        font_blobs_.clear();
        cjk_fonts_loaded_ = false;
        cjk_fonts_load_attempted_ = false;
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
            return getContext(name);
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
        context_ids_[ctx] = next_context_id_++;
        context_names_[ctx] = timerSafeContextName(name);
        return ctx;
    }

    Rml::Context* RmlUIManager::getContext(const std::string& name) {
        auto it = contexts_.find(name);
        if (it == contexts_.end() || std::ranges::contains(pending_context_destructions_, context_ids_.at(it->second)))
            return nullptr;
        return it->second;
    }

    void RmlUIManager::destroyContext(const std::string& name) {
        const auto it = contexts_.find(name);
        if (it == contexts_.end())
            return;
        auto* context = it->second;
        const auto id = context_ids_.at(context);
        if (std::ranges::contains(pending_context_destructions_, id))
            return;
        pending_context_destructions_.push_back(id);
        cancelPointerInput(context);
        if (auto handler = input_handlers_.find(context); handler != input_handlers_.end())
            handler->second.enabled = false;
        if (keyboard_context_ == context)
            keyboard_context_ = nullptr;
        // Preserve synchronous teardown for owners outside input dispatch. A
        // callback must keep RmlUi alive until the enclosing call has returned.
        if (!input_dispatch_active_ && !dispatching_input_ && !flushing_input_lifecycle_) {
            std::erase(pending_context_destructions_, id);
            destroyContextNow(id);
        }
    }

    Rml::Context* RmlUIManager::contextById(const uint64_t id, const bool include_retired) const {
        if (!include_retired && std::ranges::contains(pending_context_destructions_, id))
            return nullptr;
        for (const auto& [context, candidate] : context_ids_)
            if (candidate == id)
                return context;
        return nullptr;
    }

    void RmlUIManager::destroyContextNow(const uint64_t id) {
        auto* context = contextById(id, true);
        if (!context)
            return;
        const std::string name = context->GetName();
        // Retire the identity before callbacks can request destruction again.
        context_ids_.erase(context);
        if (current_drag_context_id_ == id)
            current_drag_context_id_ = 0;
        contexts_.erase(name);
        auto erase_context_commands = [context](std::vector<VulkanContextCommand>& queue) {
            std::erase_if(queue, [context](const VulkanContextCommand& command) {
                return command.context == context;
            });
        };
        erase_context_commands(vulkan_queue_);
        erase_context_commands(vulkan_foreground_queue_);
        if (system_interface_)
            system_interface_->releaseContext(context);
        for (const auto* type : INPUT_LIFECYCLE_EVENTS)
            context->GetRootElement()->RemoveEventListener(type, this, true);
        input_handlers_.erase(context);
        if (keyboard_context_ == context)
            keyboard_context_ = nullptr;
        for (auto& [_, owner] : key_owners_)
            if (owner.context == context)
                owner.context = nullptr;
        context_names_.erase(context);
        tracked_context_frames_.erase(context);
        previous_context_frames_.erase(context);
        tooltip_reveal_deadlines_.erase(context);
        if (auto fn = lfs::python::get_rml_context_destroy_handler())
            fn(context);
        Rml::RemoveContext(name);
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
        flushInputLifecycle();
        ++input_frame_;
        if (system_interface_)
            system_interface_->beginFrame();
        previous_context_frames_ = tracked_context_frames_;
        tracked_context_frames_.clear();
        tracked_context_order_ = 0;
    }

    void RmlUIManager::trackContextFrame(const Rml::Context* const context,
                                         const int window_x,
                                         const int window_y,
                                         std::optional<RmlRect> active_overlay) {
        if (system_interface_)
            system_interface_->trackContext(context, window_x, window_y);
        if (!context)
            return;

        if (auto it = input_handlers_.find(const_cast<Rml::Context*>(context)); it != input_handlers_.end())
            it->second.frame = input_frame_;
        const auto dimensions = context->GetDimensions();
        auto& frame = tracked_context_frames_[context];
        const bool needs_passive_frames = frame.needs_passive_mouse_move_frames;
        if (active_overlay &&
            (active_overlay->x2 <= active_overlay->x1 ||
             active_overlay->y2 <= active_overlay->y1)) {
            active_overlay.reset();
        }
        frame = TrackedContextFrame{
            .context = const_cast<Rml::Context*>(context),
            .window_x = window_x,
            .window_y = window_y,
            .width = dimensions.x,
            .height = dimensions.y,
            .order = ++tracked_context_order_,
            .needs_passive_mouse_move_frames = needs_passive_frames,
            .active_overlay = active_overlay,
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

    void RmlUIManager::setContextTooltipRevealDeadline(
        const Rml::Context* const context,
        const std::optional<std::chrono::steady_clock::time_point> deadline) {
        if (!context)
            return;
        if (deadline)
            tooltip_reveal_deadlines_[context] = *deadline;
        else
            tooltip_reveal_deadlines_.erase(context);
    }

    std::optional<double> RmlUIManager::secondsUntilTooltipReveal() const {
        const auto now = std::chrono::steady_clock::now();
        std::optional<std::chrono::steady_clock::duration> earliest;
        for (const auto& [_, deadline] : tooltip_reveal_deadlines_) {
            if (deadline <= now)
                continue; // Past-due reveals are painted by a render, not the wait cap.
            const auto remaining = deadline - now;
            if (!earliest || remaining < *earliest)
                earliest = remaining;
        }
        if (!earliest)
            return std::nullopt;
        return std::chrono::duration<double>(*earliest).count();
    }

    RmlCursorRequest RmlUIManager::consumeCursorRequest() {
        return system_interface_ ? system_interface_->consumeCursorRequest()
                                 : RmlCursorRequest::None;
    }

    bool RmlUIManager::passiveMouseMoveNeedsRender(const float window_x,
                                                   const float window_y) const {
        if (tracked_context_frames_.empty())
            return true;

        const TrackedContextFrame* top_overlay_context = nullptr;
        const TrackedContextFrame* top_context = nullptr;
        bool any_active_context = false;
        bool any_tooltip_context = false;
        for (const auto& [_, frame] : tracked_context_frames_) {
            auto* const context = frame.context;
            if (!context)
                continue;

            auto* const hover = context->GetHoverElement();
            any_tooltip_context |= frame.needs_passive_mouse_move_frames;
            if (frame.needs_passive_mouse_move_frames ||
                (hover && hover->GetTagName() != "body")) {
                any_active_context = true;
            }

            const float local_x = window_x - static_cast<float>(frame.window_x);
            const float local_y = window_y - static_cast<float>(frame.window_y);
            if (frame.active_overlay && pointInRect(*frame.active_overlay, local_x, local_y)) {
                any_active_context = true;
                if (!top_overlay_context || frame.order > top_overlay_context->order)
                    top_overlay_context = &frame;
            }

            if (frame.width <= 0 || frame.height <= 0)
                continue;

            if (local_x < 0.0f || local_y < 0.0f ||
                local_x >= static_cast<float>(frame.width) ||
                local_y >= static_cast<float>(frame.height)) {
                continue;
            }

            if (!top_context || frame.order > top_context->order)
                top_context = &frame;
        }

        if (top_overlay_context)
            top_context = top_overlay_context;

        // A tooltip anywhere must see the pointer leave so it can hide, even
        // when another context is under the pointer.
        if (any_tooltip_context)
            return true;
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

    bool RmlUIManager::activeOverlayContainsPoint(const float window_x,
                                                  const float window_y) const {
        for (const auto& [_, frame] : tracked_context_frames_) {
            if (!frame.context || !frame.active_overlay)
                continue;

            const float local_x = window_x - static_cast<float>(frame.window_x);
            const float local_y = window_y - static_cast<float>(frame.window_y);
            if (pointInRect(*frame.active_overlay, local_x, local_y))
                return true;
        }
        return false;
    }

    bool RmlUIManager::activeOverlayOccludesContext(const Rml::Context* const context,
                                                    const float window_x,
                                                    const float window_y) const {
        const TrackedContextFrame* owner = nullptr;
        for (const auto& [_, frame] : previous_context_frames_) {
            if (!frame.context || !frame.active_overlay)
                continue;

            const float local_x = window_x - static_cast<float>(frame.window_x);
            const float local_y = window_y - static_cast<float>(frame.window_y);
            if (!pointInRect(*frame.active_overlay, local_x, local_y))
                continue;

            if (!owner || frame.order > owner->order)
                owner = &frame;
        }

        return owner && owner->context != context;
    }

    bool RmlUIManager::focusContext(Rml::Context* context, const bool activate) {
        // A focus callback may outlive its element's document during a panel
        // rebuild. Detached elements have no context and cannot own keyboard input.
        const auto identity = context_ids_.find(context);
        if (identity == context_ids_.end() || !contextById(identity->second))
            return false;
        const auto current = input_handlers_.find(keyboard_context_);
        if (!activate && context != keyboard_context_ && current != input_handlers_.end() &&
            current->second.enabled && current->second.exclusive &&
            (current->second.persistent || current->second.frame == input_frame_))
            return false;
        InputCallbackScope callback_scope(dispatching_input_);
        const auto id = identity->second;
        keyboard_context_ = context;
        std::vector<uint64_t> others;
        for (const auto& [other, _] : input_handlers_)
            if (other != context)
                others.push_back(context_ids_.at(other));
        for (const auto other_id : others)
            if (auto* other = contextById(other_id))
                if (auto* focused = other->GetFocusElement())
                    focused->Blur();
        return contextById(id) != nullptr;
    }

    void RmlUIManager::cancelPointerInput(Rml::Context* context, const bool unloading) {
        const auto it = input_handlers_.find(context);
        const auto identity = context_ids_.find(context);
        if (it == input_handlers_.end() || identity == context_ids_.end())
            return;
        const auto id = identity->second;
        auto& registered = it->second;
        const bool accepted = std::ranges::contains(registered.pointer_presses, PointerPressState::Accepted);
        for (int button = 0; button < 3; ++button) {
            if (registered.pointer_presses[button] == PointerPressState::Accepted)
                registered.pointer_presses[button] = PointerPressState::Blocked;
            registered.pointer_documents[button] = nullptr;
            registered.input.mouse_down[button] = false;
            registered.input.mouse_clicked[button] = false;
            registered.input.mouse_released[button] = false;
        }
        // Invalidate the payload before drag-end callbacks can release it to a
        // viewport, including callers that do not inspect the cancelled flag.
        {
            std::scoped_lock lock(drag_payload_mutex_);
            if (drag_payload_context_id_ == id)
                drag_payload_.reset();
        }
        if (accepted)
            pending_pointer_cancellations_.push_back({id, unloading ? registered.drag_element : nullptr});
        else if (unloading)
            for (auto& pending : pending_pointer_cancellations_)
                if (pending.context_id == id)
                    pending.unloaded_drag = registered.drag_element;
    }

    void RmlUIManager::flushInputLifecycle() {
        if (flushing_input_lifecycle_ || input_dispatch_active_ || dispatching_input_)
            return;
        flushing_input_lifecycle_ = true;
        while (!pending_pointer_cancellations_.empty() || !pending_context_destructions_.empty()) {
            while (!pending_pointer_cancellations_.empty()) {
                auto cancellation = std::move(pending_pointer_cancellations_.back());
                pending_pointer_cancellations_.pop_back();
                if (std::ranges::contains(pending_context_destructions_, cancellation.context_id))
                    continue;
                if (auto* context = contextById(cancellation.context_id))
                    context->ProcessMouseButtonCancel(0, 0);
                // UnloadDocument clears RmlUi's drag pointer before returning.
                // Its detached element may still need the cancelled drag-end.
                if (contextById(cancellation.context_id) && cancellation.unloaded_drag &&
                    !std::ranges::contains(pending_context_destructions_, cancellation.context_id)) {
                    Rml::Dictionary parameters;
                    parameters["cancelled"] = true;
                    cancellation.unloaded_drag->DispatchEvent("dragend", parameters);
                }
            }
            if (!pending_context_destructions_.empty()) {
                const auto id = pending_context_destructions_.back();
                pending_context_destructions_.pop_back();
                destroyContextNow(id);
            }
        }
        flushing_input_lifecycle_ = false;
    }

    void RmlUIManager::ProcessEvent(Rml::Event& event) {
        auto* element = event.GetTargetElement();
        auto* context = element->GetContext();
        const auto it = input_handlers_.find(context);
        if (it != input_handlers_.end()) {
            if (event.GetType() == "mousedown") {
                const int button = event.GetParameter("button", -1);
                if (button >= 0 && button < 3 && it->second.pointer_presses[button] == PointerPressState::Accepted)
                    if (auto* document = element->GetOwnerDocument())
                        it->second.pointer_documents[button] = document->GetObserverPtr();
            } else if (event.GetType() == "dragstart") {
                it->second.drag_element = element->GetObserverPtr();
                current_drag_context_id_ = context_ids_.at(context);
            } else if (event.GetType() == "dragend") {
                it->second.drag_element = nullptr;
                if (current_drag_context_id_ == context_ids_.at(context))
                    current_drag_context_id_ = 0;
            } else if (event.GetType() == "hide" || event.GetType() == "unload") {
                const bool owns_drag = it->second.drag_element && it->second.drag_element->GetOwnerDocument() == element;
                if (std::ranges::any_of(it->second.pointer_documents, [element](const auto& document) { return document.get() == element; }) || owns_drag)
                    // Only unloading the drag source removes RmlUi's drag pointer.
                    // Unloading another pressed document leaves that drag intact.
                    cancelPointerInput(context, event.GetType() == "unload" && owns_drag);
            }
        }
        if (event.GetType() == "focus" && rml_input::hasFocusedKeyboardTarget(element)) {
            accepts_text_activation_ = focusContext(element->GetContext());
            if (!accepts_text_activation_)
                rejected_focus_.push_back(element->GetObserverPtr());
        }
    }

    bool RmlUIManager::registerInput(Rml::Context* context, const PanelInputState& input,
                                     std::function<void(const PanelInputState&)> handler, const bool exclusive,
                                     std::function<bool(float, float)> pointer_blocker) {
        if (dispatching_input_)
            return false;
        if (!context_ids_.contains(context))
            context_ids_[context] = next_context_id_++;
        if (!input_handlers_.contains(context))
            for (const auto* type : INPUT_LIFECYCLE_EVENTS)
                context->GetRootElement()->AddEventListener(type, this, true);
        auto& registered = input_handlers_[context];
        const auto context_id = context_ids_.at(context);
        if (!contextById(context_id))
            return true;
        registered.pointer_blocker = std::move(pointer_blocker);
        auto& passive = registered.input;
        const bool pointer_owned = std::ranges::contains(registered.pointer_presses, PointerPressState::Accepted);
        if (!pointer_owned) {
            passive.mouse_x = input.mouse_x;
            passive.mouse_y = input.mouse_y;
            for (int button = 0; button < 3; ++button)
                passive.mouse_down[button] = input.mouse_down[button] && registered.pointer_presses[button] != PointerPressState::Blocked;
        }
        passive.screen_x = input.screen_x;
        passive.screen_y = input.screen_y;
        passive.screen_w = input.screen_w;
        passive.screen_h = input.screen_h;
        passive.bg_draw_list = input.bg_draw_list;
        passive.fg_draw_list = input.fg_draw_list;
        passive.key_ctrl = input.key_ctrl;
        passive.key_shift = input.key_shift;
        passive.key_alt = input.key_alt;
        passive.key_super = input.key_super;
        registered.callback = std::make_shared<std::function<void(const PanelInputState&)>>(handler);
        registered.frame = input_frame_;
        registered.exclusive = exclusive;
        registered.enabled = true;
        dispatching_input_ = true;
        handler(passive);
        dispatching_input_ = false;
        context = contextById(context_id);
        if (!context)
            return true;
        if (exclusive || (!keyboard_context_ && rml_input::hasFocusedKeyboardTarget(context->GetFocusElement())))
            focusContext(context);
        return true;
    }

    void RmlUIManager::activateInput(Rml::Context* context, std::function<void(const PanelInputState&)> handler,
                                     const bool exclusive, std::vector<SDL_Scancode> shortcuts) {
        if (!context_ids_.contains(context))
            context_ids_[context] = next_context_id_++;
        if (!input_handlers_.contains(context))
            for (const auto* type : INPUT_LIFECYCLE_EVENTS)
                context->GetRootElement()->AddEventListener(type, this, true);
        if (!contextById(context_ids_.at(context)))
            return;
        auto& registered = input_handlers_[context];
        registered.callback = std::make_shared<std::function<void(const PanelInputState&)>>(std::move(handler));
        registered.frame = input_frame_;
        registered.exclusive = exclusive;
        registered.shortcuts = std::move(shortcuts);
        registered.persistent = true;
        registered.enabled = true;
        if (exclusive)
            focusContext(context, true);
        syncTextInput();
    }

    void RmlUIManager::deactivateInput(Rml::Context* context, const bool keep_pointer_input) {
        const auto identity = context_ids_.find(context);
        if (identity == context_ids_.end())
            return;
        const auto id = identity->second;
        InputCallbackScope callback_scope(dispatching_input_);
        cancelPointerInput(context);
        if (auto it = input_handlers_.find(context); it != input_handlers_.end()) {
            it->second.enabled = keep_pointer_input;
            it->second.persistent = false;
            it->second.exclusive = false;
            it->second.shortcuts.clear();
        }
        if (auto* live = contextById(id))
            if (auto* focused = live->GetFocusElement())
                focused->Blur();
        context = contextById(id);
        if (context && keyboard_context_ == context)
            keyboard_context_ = nullptr;
        syncTextInput();
    }

    void RmlUIManager::syncTextInput() {
        InputCallbackScope callback_scope(dispatching_input_);
        // Focus notifications precede RmlUi's assignment of its focus element.
        // Finish rejected transitions after that assignment, preserving the
        // exclusive owner's text-input handler and composition throughout.
        while (!rejected_focus_.empty()) {
            auto element = rejected_focus_.back();
            rejected_focus_.pop_back();
            if (element && element->GetContext() != keyboard_context_)
                element->Blur();
        }
        const auto it = input_handlers_.find(keyboard_context_);
        const bool active = it != input_handlers_.end() && it->second.enabled && (it->second.persistent || it->second.frame == input_frame_);
        auto* focused = active ? keyboard_context_->GetFocusElement() : nullptr;
        auto& focus = guiFocusState();
        focus.want_text_input = rml_input::wantsTextInput(focused);
        focus.any_item_active = rml_input::hasFocusedKeyboardTarget(focused);
        focus.want_capture_keyboard = (active && it->second.exclusive) || rml_input::hasFocusedKeyboardTarget(focused);
        const bool collect_text = focus.want_text_input;
        if (window_ && SDL_TextInputActive(window_) != collect_text) {
            if (std::getenv("LFS_TRACE_INPUT"))
                LOG_INFO("INPUT SDL text state -> {} owner={}", collect_text,
                         active ? keyboard_context_->GetName() : "none");
            if (collect_text)
                SDL_StartTextInput(window_);
            else
                SDL_StopTextInput(window_);
        }
    }

    RmlUIManager::InputDispatchResult RmlUIManager::dispatchInputEvent(const SDL_Event& event) {
        flushInputLifecycle();
        input_dispatch_active_ = true;
        struct FinishDispatch {
            RmlUIManager& manager;
            ~FinishDispatch() {
                manager.input_dispatch_active_ = false;
                manager.flushInputLifecycle();
            }
        } finish{*this};
        InputDispatchResult result;
        const bool key = event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP;
        const bool text = event.type == SDL_EVENT_TEXT_INPUT || event.type == SDL_EVENT_TEXT_EDITING;
        const bool pointer = event.type == SDL_EVENT_MOUSE_BUTTON_DOWN || event.type == SDL_EVENT_MOUSE_BUTTON_UP ||
                             event.type == SDL_EVENT_MOUSE_MOTION || event.type == SDL_EVENT_MOUSE_WHEEL ||
                             event.type == SDL_EVENT_PINCH_UPDATE;
        if (!key && !text && !pointer) {
            if (event.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
                key_owners_.clear();
                std::fill(std::begin(input_mouse_down_), std::end(input_mouse_down_), false);
                std::vector<uint64_t> contexts;
                for (const auto& [context, _] : input_handlers_)
                    contexts.push_back(context_ids_.at(context));
                for (const auto id : contexts)
                    if (auto* context = contextById(id))
                        cancelPointerInput(context);
                cancelDragPayload();
            }
            syncTextInput();
            return result;
        }
        dispatch_frame_.beginFrame();
        dispatch_frame_.processEvent(event);
        auto& single = dispatch_input_;
        buildPanelInputFromSDL(dispatch_frame_, single);
        InputEventDispatch dispatch;
        for (auto& input : single.input_events)
            input.dispatch = &dispatch;
        if (pointer) {
            if (event.type == SDL_EVENT_MOUSE_MOTION) {
                single.mouse_x = event.motion.x;
                single.mouse_y = event.motion.y;
            } else if (event.type == SDL_EVENT_PINCH_UPDATE) {
                const auto pointer = input::wheelPointerInPixels(window_);
                single.mouse_x = pointer.x;
                single.mouse_y = pointer.y;
            } else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
                single.mouse_x = event.wheel.mouse_x;
                single.mouse_y = event.wheel.mouse_y;
            } else {
                single.mouse_x = event.button.x;
                single.mouse_y = event.button.y;
                for (auto& button : single.mouse_button_events) {
                    input_mouse_down_[button.button] = button.down;
                }
            }
        }
        if (pointer && window_ && event.type != SDL_EVENT_PINCH_UPDATE) {
            const auto scale = input::windowPixelScale(window_);
            single.mouse_x *= scale.x;
            single.mouse_y *= scale.y;
            for (auto& button : single.mouse_button_events) {
                button.x *= scale.x;
                button.y *= scale.y;
            }
        }
        auto invoke = [&](const uint64_t id) {
            auto* context = contextById(id);
            const auto it = input_handlers_.find(context);
            if (it == input_handlers_.end() || !it->second.enabled)
                return;
            auto& input = it->second.input;
            input.input_events = single.input_events;
            input.keys_pressed = single.keys_pressed;
            input.mouse_button_events = single.mouse_button_events;
            if (pointer) {
                input.mouse_x = single.mouse_x;
                input.mouse_y = single.mouse_y;
                std::copy(std::begin(single.mouse_clicked), std::end(single.mouse_clicked), input.mouse_clicked);
                std::copy(std::begin(single.mouse_released), std::end(single.mouse_released), input.mouse_released);
                std::copy(std::begin(input_mouse_down_), std::end(input_mouse_down_), input.mouse_down);
                input.mouse_wheel = single.mouse_wheel;
                input.mouse_wheel_x = single.mouse_wheel_x;
                input.pinch_scale = single.pinch_scale;
                // Occlusion decides ownership at the press. Accepted gestures keep
                // their motion and release; blocked gestures never gain a release.
                auto& registered = it->second;
                const bool blocked = registered.pointer_blocker &&
                                     registered.pointer_blocker(single.mouse_x, single.mouse_y);
                for (const auto& button : single.mouse_button_events)
                    if (button.down)
                        registered.pointer_presses[button.button] = blocked ? PointerPressState::Blocked : PointerPressState::Accepted;
                const bool pointer_owned = std::ranges::contains(registered.pointer_presses, PointerPressState::Accepted);
                const auto suppress_button = [&](int button) {
                    const auto state = registered.pointer_presses[button];
                    return state == PointerPressState::Blocked || (blocked && state != PointerPressState::Accepted);
                };
                std::erase_if(input.mouse_button_events, [&](const auto& button) {
                    return suppress_button(button.button);
                });
                std::erase_if(input.input_events, [&](const auto& event) {
                    return event.kind == FrameInputEventKind::MouseButton && input.mouse_button_events.empty();
                });
                for (int button = 0; button < 3; ++button) {
                    if (suppress_button(button))
                        input.mouse_clicked[button] = input.mouse_released[button] = input.mouse_down[button] = false;
                }
                if (blocked) {
                    if (!pointer_owned)
                        input.mouse_x = input.mouse_y = -1e9f;
                    input.mouse_wheel = input.mouse_wheel_x = 0;
                    input.pinch_scale = 1.0f;
                }
            }
            const auto mods = key ? event.key.mod : SDL_GetModState();
            input.key_ctrl = mods & SDL_KMOD_CTRL;
            input.key_shift = mods & SDL_KMOD_SHIFT;
            input.key_alt = mods & SDL_KMOD_ALT;
            input.key_super = mods & SDL_KMOD_GUI;
            auto callback = it->second.callback;
            dispatching_input_ = true;
            (*callback)(input);
            dispatching_input_ = false;
            context = contextById(id);
            if (!context)
                return;
            const auto current = input_handlers_.find(context);
            if (current == input_handlers_.end())
                return;
            // Keep release ownership through the callback: its initial mouse move
            // can start a drag and hide the source before the button-up is handled.
            if (pointer)
                for (int button = 0; button < 3; ++button)
                    if (single.mouse_released[button]) {
                        current->second.pointer_presses[button] = PointerPressState::None;
                        current->second.pointer_documents[button] = nullptr;
                    }
            auto& remaining = current->second.input;
            remaining.input_events.clear();
            remaining.keys_pressed.clear();
            remaining.mouse_button_events.clear();
            std::fill(std::begin(remaining.mouse_clicked), std::end(remaining.mouse_clicked), false);
            std::fill(std::begin(remaining.mouse_released), std::end(remaining.mouse_released), false);
            remaining.mouse_wheel = remaining.mouse_wheel_x = 0;
            remaining.pinch_scale = 1.0f;
        };
        if (pointer) {
            auto& contexts = pointer_contexts_;
            contexts.clear();
            // Render-time pointer masking must not suppress a later native press.
            // Each context hit-tests the current event, including outside-click blur.
            for (const auto& [context, handler] : input_handlers_)
                if (handler.enabled && (handler.persistent || handler.frame == input_frame_))
                    contexts.push_back(context_ids_.at(context));
            // Underlays see an outside click before the target context focuses.
            std::ranges::sort(contexts, [&](const auto a, const auto b) {
                return tracked_context_frames_[contextById(a)].order < tracked_context_frames_[contextById(b)].order;
            });
            const auto owner = input_handlers_.find(keyboard_context_);
            if (owner != input_handlers_.end() && owner->second.enabled && owner->second.exclusive &&
                (owner->second.persistent || owner->second.frame == input_frame_)) {
                const auto exclusive_context = context_ids_.at(keyboard_context_);
                for (const auto id : contexts) {
                    if (id == exclusive_context)
                        continue;
                    const auto it = input_handlers_.find(contextById(id));
                    if (it == input_handlers_.end())
                        continue;
                    const auto& presses = it->second.pointer_presses;
                    const bool owned_motion = event.type == SDL_EVENT_MOUSE_MOTION &&
                                              std::ranges::contains(presses, PointerPressState::Accepted);
                    const bool owned_release = std::ranges::any_of(single.mouse_button_events, [&](const auto& button) {
                        return !button.down && presses[button.button] == PointerPressState::Accepted;
                    });
                    if (owned_motion || owned_release)
                        invoke(id);
                }
                invoke(exclusive_context);
            } else {
                for (const auto id : contexts) {
                    invoke(id);
                    const auto current = input_handlers_.find(keyboard_context_);
                    if (current != input_handlers_.end() && current->second.enabled && current->second.exclusive &&
                        (current->second.persistent || current->second.frame == input_frame_))
                        break;
                }
            }
        } else {
            auto* context = keyboard_context_;
            auto it = input_handlers_.find(context);
            if (it == input_handlers_.end() || !it->second.enabled || (!it->second.persistent && it->second.frame != input_frame_) ||
                (!it->second.exclusive && !rml_input::hasFocusedKeyboardTarget(context->GetFocusElement())))
                context = nullptr;
            const auto shortcutContext = [&]() {
                Rml::Context* target = nullptr;
                if (event.type == SDL_EVENT_KEY_DOWN) {
                    for (const auto& [candidate, handler] : input_handlers_) {
                        if (handler.enabled && (handler.persistent || handler.frame == input_frame_) &&
                            std::ranges::find(handler.shortcuts, event.key.scancode) != handler.shortcuts.end() &&
                            (!target || tracked_context_frames_[candidate].order > tracked_context_frames_[target].order))
                            target = candidate;
                    }
                }
                return target;
            };
            if (!context) {
                context = shortcutContext();
                it = input_handlers_.find(context);
            }
            auto* focused = context ? context->GetFocusElement() : nullptr;
            auto pressed_element = focused ? focused->GetObserverPtr() : ElementObserver();
            const bool text_owner = rml_input::wantsTextInput(focused);
            const bool exclusive = context && it->second.exclusive;
            bool gui_release = false;
            if (event.type == SDL_EVENT_KEY_UP) {
                const auto owner = key_owners_.find(event.key.scancode);
                if (owner != key_owners_.end()) {
                    const auto released_owner = owner->second;
                    key_owners_.erase(owner);
                    context = released_owner.context;
                    const auto id = context ? context_ids_.at(context) : 0;
                    gui_release = released_owner.gui;
                    result.owned_release = !gui_release;
                    result.consumed = released_owner.gui && (!context || !released_owner.element);
                    if (context && released_owner.element && released_owner.element.get() != context->GetFocusElement()) {
                        Rml::Dictionary parameters;
                        parameters["key_identifier"] = sdlScancodeToRml(event.key.scancode);
                        parameters["ctrl_key"] = bool(event.key.mod & SDL_KMOD_CTRL);
                        parameters["shift_key"] = bool(event.key.mod & SDL_KMOD_SHIFT);
                        parameters["alt_key"] = bool(event.key.mod & SDL_KMOD_ALT);
                        parameters["meta_key"] = bool(event.key.mod & SDL_KMOD_GUI);
                        released_owner.element->DispatchEvent("keyup", parameters);
                        context = contextById(id);
                        result.consumed = true;
                    }
                }
            }
            if (context && !result.consumed) {
                if (std::getenv("LFS_TRACE_INPUT") && text)
                    LOG_INFO("INPUT dispatch owner={} text={}", context->GetName(), single.input_events.front().text);
                const auto id = context_ids_.at(context);
                invoke(id);
                context = contextById(id);
                result.consumed = gui_release || text_owner || exclusive || dispatch.consumed ||
                                  !input_handlers_.contains(context) || !input_handlers_.at(context).enabled ||
                                  context->GetFocusElement() != pressed_element.get();
            }
            // Blurring a control can leave its inert parent focused. Give
            // unhandled keys to viewport shortcuts without taking text ownership.
            if (!result.consumed) {
                if (auto* shortcut = shortcutContext(); shortcut && shortcut != context) {
                    context = shortcut;
                    focused = context->GetFocusElement();
                    pressed_element = focused ? focused->GetObserverPtr() : ElementObserver();
                    const auto id = context_ids_.at(context);
                    invoke(id);
                    context = contextById(id);
                    result.consumed = dispatch.consumed || !input_handlers_.contains(context) ||
                                      !input_handlers_.at(context).enabled || context->GetFocusElement() != pressed_element.get();
                }
            }
            if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat)
                key_owners_[event.key.scancode] = {result.consumed && input_handlers_.contains(context) ? context : nullptr,
                                                   pressed_element, result.consumed};
        }
        syncTextInput();
        return result;
    }

    bool RmlUIManager::wantsCaptureKeyboard() const {
        const auto it = input_handlers_.find(keyboard_context_);
        return it != input_handlers_.end() && it->second.enabled && (it->second.persistent || it->second.frame == input_frame_) &&
               (it->second.exclusive || rml_input::hasFocusedKeyboardTarget(keyboard_context_->GetFocusElement()));
    }

    bool RmlUIManager::wantsTextInput() const {
        const auto it = input_handlers_.find(keyboard_context_);
        return it != input_handlers_.end() && it->second.enabled && (it->second.persistent || it->second.frame == input_frame_) &&
               rml_input::wantsTextInput(keyboard_context_->GetFocusElement());
    }

    bool RmlUIManager::anyItemActive() const {
        return wantsCaptureKeyboard();
    }

    bool RmlUIManager::refreshLocalizedDocuments() {
        bool changed = false;
        for (const auto& [_, context] : contexts_) {
            if (!context)
                continue;
            for (int i = 0; i < context->GetNumDocuments(); ++i)
                changed |= rml_documents::refreshLocalizedContent(context->GetDocument(i));
        }
        return changed;
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
            .cache_visible_region = draw.cache_visible_region,
        });
    }

    void RmlUIManager::releaseCachedVulkanContext(CachedVulkanContextRender& cache) {
        if (vulkan_render_interface_ && cache.texture != 0)
            vulkan_render_interface_->ReleaseTexture(cache.texture);
        cache.texture = {};
        cache.width = 0;
        cache.height = 0;
        cache.depends_on_preview_textures = false;
        cache.preview_texture_generation = 0;
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
        vulkan_frame_extent_ = extent;
        return true;
    }

    void RmlUIManager::renderQueuedVulkanContexts(const bool foreground) {
        auto& queue = foreground ? vulkan_foreground_queue_ : vulkan_queue_;
        if (!vulkan_render_interface_ || !vulkan_frame_active_) {
            queue.clear();
            return;
        }

        const auto previewDependencyChanged = [this](const CachedVulkanContextRender& cache) {
            return cache.depends_on_preview_textures &&
                   cache.preview_texture_generation !=
                       vulkan_render_interface_->previewTextureGeneration();
        };
        const auto recordPreviewDependency = [this](CachedVulkanContextRender& cache,
                                                    const bool saved) {
            cache.depends_on_preview_textures =
                saved && vulkan_render_interface_->currentContextUsedPreviewTexture();
            cache.preview_texture_generation = cache.depends_on_preview_textures
                                                   ? vulkan_render_interface_->previewTextureGeneration()
                                                   : 0;
        };

        for (const auto& command : queue) {
            if (!command.context)
                continue;

            const auto canvas_start = command.context_name == "node_editor"
                                          ? std::optional{std::chrono::steady_clock::now()}
                                          : std::nullopt;

            const std::string timer_name = std::string("gui_render.rmlui_record.") +
                                           (foreground ? "foreground.context." : "background.context.") +
                                           command.context_name;
            // Fixed overlays cache the full context at (0,0) and blit it
            // (optionally scaled) to the draw rect. That would break scrollable
            // panels whose content is taller than the framebuffer: SaveLayerAsTexture
            // clamps the capture to the framebuffer and the blit would then stretch
            // it back to full size (magnified). Such panels set cache_visible_region
            // so we cache only the on-screen clipped window, which always fits the
            // framebuffer and blits 1:1.
            if (command.cache && command.cache_visible_region) {
                const int fb_w = static_cast<int>(vulkan_frame_extent_.width);
                const int fb_h = static_cast<int>(vulkan_frame_extent_.height);
                const int left = std::clamp(static_cast<int>(std::floor(command.clip_x1)), 0, fb_w);
                const int top = std::clamp(static_cast<int>(std::floor(command.clip_y1)), 0, fb_h);
                const int right = std::clamp(static_cast<int>(std::ceil(command.clip_x2)), 0, fb_w);
                const int bottom = std::clamp(static_cast<int>(std::ceil(command.clip_y2)), 0, fb_h);
                const int vis_w = right - left;
                const int vis_h = bottom - top;
                const float fleft = static_cast<float>(left);
                const float ftop = static_cast<float>(top);
                const float fright = static_cast<float>(right);
                const float fbottom = static_cast<float>(bottom);

                if (vis_w <= 0 || vis_h <= 0) {
                    if (command.cache->texture != 0)
                        releaseCachedVulkanContext(*command.cache);
                } else {
                    const VkRect2D capture_region{
                        {left, top},
                        {static_cast<uint32_t>(vis_w), static_cast<uint32_t>(vis_h)}};
                    const bool region_changed =
                        command.cache->width != vis_w || command.cache->height != vis_h ||
                        std::abs(command.cache->offset_x - command.offset_x) > 0.5f ||
                        std::abs(command.cache->offset_y - command.offset_y) > 0.5f ||
                        std::abs(command.cache->clip_x1 - fleft) > 0.5f ||
                        std::abs(command.cache->clip_y1 - ftop) > 0.5f;
                    const bool refresh_cache =
                        command.refresh_cache || command.cache->texture == 0 || region_changed ||
                        previewDependencyChanged(*command.cache);

                    if (refresh_cache) {
                        lfs::core::ScopedTimer timer(
                            timer_name + ".cache_refresh", 0.25,
                            lfs::core::LogLevel::Performance, LFS_SOURCE_SITE_CURRENT());
                        // Same capture extent → reuse the existing image (copy into it).
                        // Extent change → deferred-delete the old image and allocate fresh.
                        const Rml::TextureHandle reuse_texture =
                            (command.cache->texture != 0 && command.cache->width == vis_w &&
                             command.cache->height == vis_h)
                                ? command.cache->texture
                                : Rml::TextureHandle{};
                        if (command.cache->texture != 0 && reuse_texture == 0)
                            releaseCachedVulkanContext(*command.cache);

                        vulkan_render_interface_->ResetContextRenderState();
                        vulkan_render_interface_->BeginCacheCapture(left, top, vis_w, vis_h);
                        vulkan_render_interface_->SetContextOffset(command.offset_x, command.offset_y);
                        vulkan_render_interface_->SetContextClipRect(fleft, ftop, fright, fbottom);
                        const Rml::LayerHandle layer = vulkan_render_interface_->PushLayer();
                        if (layer != 0) {
                            command.context->Render();
                            const Rml::TextureHandle saved_texture =
                                vulkan_render_interface_->SaveLayerRegionAsTexture(capture_region, reuse_texture);
                            if (reuse_texture != 0 && saved_texture != 0 && saved_texture != reuse_texture)
                                vulkan_render_interface_->ReleaseTexture(reuse_texture);
                            // On save failure keep a still-valid reuse handle (avoid leaking it).
                            command.cache->texture =
                                saved_texture != 0 ? saved_texture : reuse_texture;
                            vulkan_render_interface_->SetTextureDebugName(command.cache->texture,
                                                                          command.context_name);
                            vulkan_render_interface_->PopLayer();
                            const bool saved = command.cache->texture != 0;
                            command.cache->width = saved ? vis_w : 0;
                            command.cache->height = saved ? vis_h : 0;
                            command.cache->offset_x = command.offset_x;
                            command.cache->offset_y = command.offset_y;
                            command.cache->clip_x1 = fleft;
                            command.cache->clip_y1 = ftop;
                            command.cache->clip_x2 = fright;
                            command.cache->clip_y2 = fbottom;
                            recordPreviewDependency(*command.cache, saved);
                        }
                        vulkan_render_interface_->EndCacheCapture();
                    }

                    if (command.cache->texture != 0) {
                        const std::string blit_timer_name =
                            std::string("gui_render.rmlui_record.") +
                            (foreground ? "foreground.cached_context." : "background.cached_context.") +
                            command.context_name;
                        lfs::core::ScopedTimer timer(
                            blit_timer_name, 0.25, lfs::core::LogLevel::Performance,
                            LFS_SOURCE_SITE_CURRENT());
                        vulkan_render_interface_->ResetContextRenderState();
                        vulkan_render_interface_->SetContextClipRect(fleft, ftop, fright, fbottom);
                        vulkan_render_interface_->RenderTextureQuad(command.cache->texture,
                                                                    fleft,
                                                                    ftop,
                                                                    static_cast<float>(vis_w),
                                                                    static_cast<float>(vis_h));
                    } else {
                        lfs::core::ScopedTimer timer(
                            timer_name, lfs::core::LogLevel::Performance,
                            LFS_SOURCE_SITE_CURRENT());
                        vulkan_render_interface_->ResetContextRenderState();
                        vulkan_render_interface_->SetContextClipRect(command.clip_x1,
                                                                     command.clip_y1,
                                                                     command.clip_x2,
                                                                     command.clip_y2);
                        vulkan_render_interface_->SetContextOffset(command.offset_x, command.offset_y);
                        command.context->Render();
                    }
                }
            } else if (command.cache) {
                const VkRect2D capture_region{
                    {0, 0},
                    {static_cast<uint32_t>(command.cache_width),
                     static_cast<uint32_t>(command.cache_height)}};
                const bool refresh_cache =
                    command.refresh_cache ||
                    command.cache->texture == 0 ||
                    command.cache->width != command.cache_width ||
                    command.cache->height != command.cache_height ||
                    previewDependencyChanged(*command.cache);
                if (refresh_cache) {
                    lfs::core::ScopedTimer timer(
                        timer_name + ".cache_refresh", 0.25,
                        lfs::core::LogLevel::Performance, LFS_SOURCE_SITE_CURRENT());
                    const Rml::TextureHandle reuse_texture =
                        (command.cache->texture != 0 && command.cache->width == command.cache_width &&
                         command.cache->height == command.cache_height)
                            ? command.cache->texture
                            : Rml::TextureHandle{};
                    if (command.cache->texture != 0 && reuse_texture == 0)
                        releaseCachedVulkanContext(*command.cache);

                    vulkan_render_interface_->ResetContextRenderState();
                    vulkan_render_interface_->BeginCacheCapture(0, 0, command.cache_width, command.cache_height);
                    vulkan_render_interface_->SetContextOffset(0.0f, 0.0f);
                    vulkan_render_interface_->SetContextClipRect(0.0f,
                                                                 0.0f,
                                                                 static_cast<float>(command.cache_width),
                                                                 static_cast<float>(command.cache_height));
                    const Rml::LayerHandle layer = vulkan_render_interface_->PushLayer();
                    if (layer != 0) {
                        command.context->Render();
                        const Rml::TextureHandle saved_texture =
                            vulkan_render_interface_->SaveLayerRegionAsTexture(capture_region, reuse_texture);
                        if (reuse_texture != 0 && saved_texture != 0 && saved_texture != reuse_texture)
                            vulkan_render_interface_->ReleaseTexture(reuse_texture);
                        // On save failure keep a still-valid reuse handle (avoid leaking it).
                        command.cache->texture =
                            saved_texture != 0 ? saved_texture : reuse_texture;
                        vulkan_render_interface_->SetTextureDebugName(command.cache->texture,
                                                                      command.context_name);
                        vulkan_render_interface_->PopLayer();
                        const bool saved = command.cache->texture != 0;
                        command.cache->width = saved ? command.cache_width : 0;
                        command.cache->height = saved ? command.cache_height : 0;
                        recordPreviewDependency(*command.cache, saved);
                    }
                    vulkan_render_interface_->EndCacheCapture();
                }

                if (command.cache->texture != 0) {
                    const std::string blit_timer_name =
                        std::string("gui_render.rmlui_record.") +
                        (foreground ? "foreground.cached_context." : "background.cached_context.") +
                        command.context_name;
                    lfs::core::ScopedTimer timer(
                        blit_timer_name, 0.25, lfs::core::LogLevel::Performance,
                        LFS_SOURCE_SITE_CURRENT());
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
                    lfs::core::ScopedTimer timer(
                        timer_name, lfs::core::LogLevel::Performance,
                        LFS_SOURCE_SITE_CURRENT());
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
                lfs::core::ScopedTimer timer(
                    timer_name, lfs::core::LogLevel::Performance,
                    LFS_SOURCE_SITE_CURRENT());
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
            if (canvas_start)
                if (auto* scene = services().sceneOrNull())
                    scene->modifierManager().recordCanvasFrame(
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - *canvas_start).count());
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
