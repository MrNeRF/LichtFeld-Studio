/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"
#include "core/export.hpp"

#include "gui/panel_layout.hpp"
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/ObserverPtr.h>
#include <RmlUi/Core/Types.h>
#include <functional>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

struct SDL_Window;
class RenderInterface_VK;

namespace Rml {
    class Context;
    class RenderInterface;
} // namespace Rml

namespace lfs::vis {
    class WindowInputDispatchTest;
    class VulkanContext;
} // namespace lfs::vis

namespace lfs::vis::gui {

    class RmlSystemInterface;
    class RmlTextInputHandler;
    class SceneGraphElement;
    enum class RmlCursorRequest : uint8_t;

    struct CachedVulkanContextRender {
        Rml::TextureHandle texture = {};
        int width = 0;
        int height = 0;
        // For visible-region caches: the panel offset and clipped window the
        // texture was captured at. The cache is stale if any of these change.
        float offset_x = 0.0f;
        float offset_y = 0.0f;
        float clip_x1 = 0.0f;
        float clip_y1 = 0.0f;
        float clip_x2 = 0.0f;
        float clip_y2 = 0.0f;
        bool depends_on_preview_textures = false;
        uint64_t preview_texture_generation = 0;
    };

    struct RmlRect {
        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;
    };

    struct CachedVulkanContextDraw {
        Rml::Context* context = nullptr;
        CachedVulkanContextRender* cache = nullptr;
        int cache_width = 0;
        int cache_height = 0;
        float offset_x = 0.0f;
        float offset_y = 0.0f;
        float draw_width = 0.0f;
        float draw_height = 0.0f;
        bool refresh = false;
        bool foreground = false;
        bool clip_enabled = false;
        // Cache only the clipped on-screen window instead of the full context.
        // Used by scrollable panels whose content can exceed the framebuffer.
        bool cache_visible_region = false;
        RmlRect clip;
    };

    struct RmlDragPayload {
        std::uint64_t token = 0;
        std::string type;
        std::string data;
        std::string label;
        bool released = false;
    };

    class RmlUIManager : public Rml::EventListener {
    public:
        LFS_VIS_API explicit RmlUIManager(SDL_Window* window = nullptr);
        LFS_VIS_API ~RmlUIManager();

        bool initVulkan(SDL_Window* window, lfs::vis::VulkanContext& vulkan_context, float dp_ratio = 1.0f);
        LFS_VIS_API void shutdown();
        [[nodiscard]] bool isInitialized() const { return initialized_; }

        float getDpRatio() const { return dp_ratio_; }
        void setDpRatio(float ratio);

        LFS_VIS_API Rml::Context* createContext(const std::string& name, int width, int height);
        LFS_VIS_API Rml::Context* getContext(const std::string& name);
        LFS_VIS_API void destroyContext(const std::string& name);

        void ensureCjkFontsLoaded();
        // Registers the system color emoji font as a fallback face once text
        // above U+FFFF has been shown; the file is read off the UI thread.
        void serviceEmojiFont();

        void setResizeDeferring(bool defer) { resize_deferring_ = defer; }
        [[nodiscard]] bool isResizeDeferring() const { return resize_deferring_; }

        void activateTheme(const std::string& theme_id);

        RenderInterface_VK* getVulkanRenderInterface() const { return vulkan_render_interface_; }
        RmlTextInputHandler* getTextInputHandler() const { return text_input_handler_.get(); }
        SDL_Window* getWindow() const { return window_; }

        // Render passes register geometry and handlers. SDL polling dispatches each
        // event once, before polling the next event or invoking scene operators.
        template <typename Handler>
        bool routeInput(Rml::Context* context, const PanelInputState& input, Handler&& handler, bool exclusive = false,
                        std::function<bool(float, float)> pointer_blocker = {}) {
            if (dispatching_input_)
                return false;
            return registerInput(context, input, std::forward<Handler>(handler), exclusive, std::move(pointer_blocker));
        }
        LFS_VIS_API void activateInput(Rml::Context* context, std::function<void(const PanelInputState&)> handler,
                                       bool exclusive = true, std::vector<SDL_Scancode> shortcuts = {});
        LFS_VIS_API void deactivateInput(Rml::Context* context, bool keep_pointer_input = false);
        struct InputDispatchResult {
            bool consumed = false;
            bool owned_release = false;
        };
        LFS_VIS_API InputDispatchResult dispatchInputEvent(const SDL_Event& event);
        LFS_VIS_API void syncTextInput();
        LFS_VIS_API void ProcessEvent(Rml::Event& event) override;

        void queueVulkanContext(Rml::Context* context,
                                float offset_x = 0.0f,
                                float offset_y = 0.0f,
                                bool foreground = false,
                                bool clip_enabled = false,
                                float clip_x1 = 0.0f,
                                float clip_y1 = 0.0f,
                                float clip_x2 = 0.0f,
                                float clip_y2 = 0.0f);
        void queueCachedVulkanContext(const CachedVulkanContextDraw& draw);
        void releaseCachedVulkanContext(CachedVulkanContextRender& cache);
        void clearVulkanQueue();
        [[nodiscard]] bool beginVulkanFrame(VkCommandBuffer command_buffer,
                                            VkExtent2D extent,
                                            VkImage swapchain_image,
                                            VkImageView swapchain_image_view,
                                            VkImageView depth_stencil_image_view,
                                            std::size_t frame_slot);
        void renderQueuedVulkanContexts(bool foreground);
        void endVulkanFrame();

        LFS_VIS_API void beginFrameCursorTracking();
        LFS_VIS_API void trackContextFrame(const Rml::Context* context, int window_x, int window_y,
                                           std::optional<RmlRect> active_overlay = std::nullopt);
        LFS_VIS_API void setContextNeedsPassiveMouseMoveFrames(const Rml::Context* context, bool needs_frames);
        // Registers (or clears, on nullopt) the time a context's pending tooltip
        // is due to appear, so the idle loop can wake exactly at that moment.
        void setContextTooltipRevealDeadline(
            const Rml::Context* context,
            std::optional<std::chrono::steady_clock::time_point> deadline);
        // Seconds until the earliest pending tooltip is due across all contexts,
        // or empty when none is counting down.
        [[nodiscard]] std::optional<double> secondsUntilTooltipReveal() const;
        RmlCursorRequest consumeCursorRequest();
        [[nodiscard]] LFS_VIS_API bool passiveMouseMoveNeedsRender(float window_x, float window_y) const;
        [[nodiscard]] LFS_VIS_API bool activeOverlayContainsPoint(float window_x,
                                                                  float window_y) const;
        [[nodiscard]] bool activeOverlayOccludesContext(const Rml::Context* context,
                                                        float window_x,
                                                        float window_y) const;

        // Focus-state aggregators across all live RmlUi contexts so viewport input
        // suppression reflects the actual GUI surface the user is interacting with.
        [[nodiscard]] LFS_VIS_API bool wantsCaptureKeyboard() const;
        [[nodiscard]] LFS_VIS_API bool wantsTextInput() const;
        [[nodiscard]] bool anyItemActive() const;
        bool refreshLocalizedDocuments();

        LFS_VIS_API std::uint64_t beginDragPayload(std::string type,
                                                   std::string data,
                                                   std::string label = {});
        LFS_VIS_API bool endDragPayload(std::uint64_t token);
        LFS_VIS_API bool cancelDragPayload(std::uint64_t token);
        LFS_VIS_API void cancelDragPayload();
        void setActiveSceneGraphElement(SceneGraphElement* element) {
            active_scene_graph_element_ = element;
        }
        [[nodiscard]] LFS_VIS_API std::optional<RmlDragPayload> dragPayload() const;
        LFS_VIS_API std::optional<RmlDragPayload> takeReleasedDragPayload();

    private:
        friend class lfs::vis::WindowInputDispatchTest;
        struct VulkanContextCommand {
            Rml::Context* context = nullptr;
            std::string context_name;
            float offset_x = 0.0f;
            float offset_y = 0.0f;
            bool clip_enabled = false;
            float clip_x1 = 0.0f;
            float clip_y1 = 0.0f;
            float clip_x2 = 0.0f;
            float clip_y2 = 0.0f;
            CachedVulkanContextRender* cache = nullptr;
            int cache_width = 0;
            int cache_height = 0;
            float draw_width = 0.0f;
            float draw_height = 0.0f;
            bool refresh_cache = false;
            bool cache_visible_region = false;
        };

        struct TrackedContextFrame {
            Rml::Context* context = nullptr;
            int window_x = 0;
            int window_y = 0;
            int width = 0;
            int height = 0;
            std::uint64_t order = 0;
            bool needs_passive_mouse_move_frames = false;
            std::optional<RmlRect> active_overlay;
        };

        LFS_VIS_API bool initWithRenderInterface(SDL_Window* window,
                                                 float dp_ratio,
                                                 std::unique_ptr<Rml::RenderInterface> render_interface,
                                                 RenderInterface_VK* vulkan_render_interface);

        LFS_VIS_API bool registerInput(Rml::Context* context, const PanelInputState& input,
                                       std::function<void(const PanelInputState&)> handler, bool exclusive,
                                       std::function<bool(float, float)> pointer_blocker);
        enum class PointerPressState { None,
                                       Accepted,
                                       Blocked };
        struct InputHandler {
            std::shared_ptr<std::function<void(const PanelInputState&)>> callback;
            PanelInputState input;
            std::function<bool(float, float)> pointer_blocker;
            PointerPressState pointer_presses[3] = {};
            Rml::ObserverPtr<Rml::Element> pointer_documents[3];
            Rml::ObserverPtr<Rml::Element> drag_element;
            std::vector<SDL_Scancode> shortcuts;
            uint64_t frame = 0;
            bool exclusive = false;
            bool enabled = true;
            bool persistent = false;
        };
        struct KeyOwner {
            Rml::Context* context = nullptr;
            Rml::ObserverPtr<Rml::Element> element;
            bool gui = false;
        };
        std::unordered_map<Rml::Context*, InputHandler> input_handlers_;
        std::unordered_map<SDL_Scancode, KeyOwner> key_owners_;
        Rml::Context* keyboard_context_ = nullptr;
        uint64_t input_frame_ = 0;
        bool dispatching_input_ = false;
        bool input_mouse_down_[3] = {};
        FrameInputBuffer dispatch_frame_;
        PanelInputState dispatch_input_;
        std::vector<uint64_t> pointer_contexts_;
        bool focusContext(Rml::Context* context, bool activate = false);
        void cancelPointerInput(Rml::Context* context, bool unloading = false);
        void flushInputLifecycle();
        void destroyContextNow(uint64_t id);
        Rml::Context* contextById(uint64_t id, bool include_retired = false) const;
        struct PointerCancellation {
            uint64_t context_id;
            Rml::ObserverPtr<Rml::Element> unloaded_drag;
        };
        std::vector<PointerCancellation> pending_pointer_cancellations_;
        std::vector<uint64_t> pending_context_destructions_;
        std::unordered_map<Rml::Context*, uint64_t> context_ids_;
        uint64_t next_context_id_ = 1;
        uint64_t current_drag_context_id_ = 0;
        bool flushing_input_lifecycle_ = false;
        bool input_dispatch_active_ = false;
        bool accepts_text_activation_ = true;
        std::vector<Rml::ObserverPtr<Rml::Element>> rejected_focus_;

        std::unique_ptr<RmlSystemInterface> system_interface_;
        std::unique_ptr<Rml::RenderInterface> owned_render_interface_;
        RenderInterface_VK* vulkan_render_interface_ = nullptr;
        std::unique_ptr<RmlTextInputHandler> text_input_handler_;
        std::vector<std::vector<std::byte>> font_blobs_;
        bool cjk_fonts_loaded_ = false;
        bool cjk_fonts_load_attempted_ = false;
        std::future<std::vector<std::byte>> emoji_font_read_;
        std::string emoji_font_path_;
        bool emoji_font_settled_ = false;
        std::unordered_map<std::string, Rml::Context*> contexts_;
        std::unordered_map<const Rml::Context*, std::string> context_names_;
        std::unordered_map<const Rml::Context*, TrackedContextFrame> tracked_context_frames_;
        std::unordered_map<const Rml::Context*, TrackedContextFrame> previous_context_frames_;
        std::unordered_map<const Rml::Context*, std::chrono::steady_clock::time_point>
            tooltip_reveal_deadlines_;
        std::vector<VulkanContextCommand> vulkan_queue_;
        std::vector<VulkanContextCommand> vulkan_foreground_queue_;
        SDL_Window* window_ = nullptr;
        float dp_ratio_ = 1.0f;
        std::string active_theme_id_;
        bool resize_deferring_ = false;
        bool debugger_enabled_ = false;
        bool debugger_initialized_ = false;
        bool vulkan_frame_active_ = false;
        VkExtent2D vulkan_frame_extent_{};
        bool initialized_ = false;
        std::uint64_t tracked_context_order_ = 0;
        mutable std::mutex drag_payload_mutex_;
        std::optional<RmlDragPayload> drag_payload_;
        uint64_t drag_payload_context_id_ = 0;
        std::uint64_t next_drag_payload_token_ = 1;
        SceneGraphElement* active_scene_graph_element_ = nullptr;
    };

} // namespace lfs::vis::gui
