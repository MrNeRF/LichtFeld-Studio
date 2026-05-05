/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <vulkan/vulkan.h>
#endif

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
        const std::string& activeThemeId() const { return active_theme_id_; }

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
        void clearVulkanQueue();
#ifdef LFS_VULKAN_VIEWER_ENABLED
        [[nodiscard]] bool beginVulkanFrame(VkCommandBuffer command_buffer,
                                            VkExtent2D extent,
                                            VkImage swapchain_image,
                                            VkImageView swapchain_image_view,
                                            VkImageView depth_stencil_image_view,
                                            std::size_t frame_slot);
        void renderQueuedVulkanContexts(bool foreground);
        void endVulkanFrame();
#endif

        void beginFrameCursorTracking();
        void trackContextFrame(const Rml::Context* context, int window_x, int window_y);
        RmlCursorRequest consumeCursorRequest();

        // Focus-state aggregators across all live RmlUi contexts. These replace prior
        // ImGui::GetIO().WantCapture* / ImGui::IsAnyItemActive() reads so viewport input
        // suppression reflects the actual GUI surface the user is interacting with.
        [[nodiscard]] bool wantsCaptureMouse() const;
        [[nodiscard]] bool wantsCaptureKeyboard() const;
        [[nodiscard]] bool wantsTextInput() const;
        [[nodiscard]] bool anyItemActive() const;

    private:
        struct VulkanContextCommand {
            Rml::Context* context = nullptr;
            float offset_x = 0.0f;
            float offset_y = 0.0f;
            bool clip_enabled = false;
            float clip_x1 = 0.0f;
            float clip_y1 = 0.0f;
            float clip_x2 = 0.0f;
            float clip_y2 = 0.0f;
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
        std::unordered_map<std::string, Rml::Context*> contexts_;
        std::vector<VulkanContextCommand> vulkan_queue_;
        std::vector<VulkanContextCommand> vulkan_foreground_queue_;
        SDL_Window* window_ = nullptr;
        float dp_ratio_ = 1.0f;
        std::string active_theme_id_;
        bool resize_deferring_ = false;
        bool debugger_enabled_ = false;
        bool debugger_initialized_ = false;
        bool vulkan_frame_active_ = false;
        bool initialized_ = false;
    };

} // namespace lfs::vis::gui
