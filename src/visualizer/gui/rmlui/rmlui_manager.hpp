/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"

#include <RmlUi/Core/Types.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

struct SDL_Window;
class RenderInterface_VK;

namespace Rml {
    class Context;
    class RenderInterface;
} // namespace Rml

namespace lfs::vis {
    class VulkanContext;
}

namespace lfs::vis::gui {

    class RmlSystemInterface;
    class RmlTextInputHandler;
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

    class RmlUIManager {
    public:
        RmlUIManager();
        ~RmlUIManager();

        bool initVulkan(SDL_Window* window, lfs::vis::VulkanContext& vulkan_context, float dp_ratio = 1.0f);
        void shutdown();
        [[nodiscard]] bool isInitialized() const { return initialized_; }

        float getDpRatio() const { return dp_ratio_; }
        void setDpRatio(float ratio);

        Rml::Context* createContext(const std::string& name, int width, int height);
        Rml::Context* getContext(const std::string& name);
        void destroyContext(const std::string& name);

        void ensureCjkFontsLoaded();

        void setResizeDeferring(bool defer) { resize_deferring_ = defer; }
        [[nodiscard]] bool isResizeDeferring() const { return resize_deferring_; }

        void activateTheme(const std::string& theme_id);

        RenderInterface_VK* getVulkanRenderInterface() const { return vulkan_render_interface_; }
        RmlTextInputHandler* getTextInputHandler() const { return text_input_handler_.get(); }
        SDL_Window* getWindow() const { return window_; }

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

        void beginFrameCursorTracking();
        void trackContextFrame(const Rml::Context* context, int window_x, int window_y,
                               std::optional<RmlRect> active_overlay = std::nullopt);
        void setContextNeedsPassiveMouseMoveFrames(const Rml::Context* context, bool needs_frames);
        // Registers (or clears, on nullopt) the time a context's pending tooltip
        // is due to appear, so the idle loop can wake exactly at that moment.
        void setContextTooltipRevealDeadline(
            const Rml::Context* context,
            std::optional<std::chrono::steady_clock::time_point> deadline);
        // Seconds until the earliest pending tooltip is due across all contexts,
        // or empty when none is counting down.
        [[nodiscard]] std::optional<double> secondsUntilTooltipReveal() const;
        RmlCursorRequest consumeCursorRequest();
        [[nodiscard]] bool passiveMouseMoveNeedsRender(float window_x, float window_y) const;
        [[nodiscard]] bool activeOverlayContainsPoint(float window_x, float window_y) const;
        [[nodiscard]] bool activeOverlayOccludesContext(const Rml::Context* context,
                                                        float window_x,
                                                        float window_y) const;

        // Focus-state aggregators across all live RmlUi contexts so viewport input
        // suppression reflects the actual GUI surface the user is interacting with.
        [[nodiscard]] bool wantsCaptureKeyboard() const;
        [[nodiscard]] bool wantsTextInput() const;
        [[nodiscard]] bool anyItemActive() const;

    private:
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

        bool initWithRenderInterface(SDL_Window* window,
                                     float dp_ratio,
                                     std::unique_ptr<Rml::RenderInterface> render_interface,
                                     RenderInterface_VK* vulkan_render_interface);

        std::unique_ptr<RmlSystemInterface> system_interface_;
        std::unique_ptr<Rml::RenderInterface> owned_render_interface_;
        RenderInterface_VK* vulkan_render_interface_ = nullptr;
        std::unique_ptr<RmlTextInputHandler> text_input_handler_;
        std::vector<std::vector<std::byte>> font_blobs_;
        bool cjk_fonts_loaded_ = false;
        bool cjk_fonts_load_attempted_ = false;
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
    };

} // namespace lfs::vis::gui
