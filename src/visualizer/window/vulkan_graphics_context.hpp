/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "graphics_context.hpp"
#include "vulkan_context.hpp"

namespace lfs::vis {

    class LFS_VIS_API VulkanGraphicsContext final : public GraphicsContext {
    public:
        VulkanGraphicsContext();
        ~VulkanGraphicsContext() override;

        bool initialize(SDL_Window* window, int framebuffer_width,
                        int framebuffer_height) override;
        bool initializeHeadless() override;
        void shutdown() override;
        void notifyFramebufferResized(int width, int height,
                                      GraphicsResizeIntent intent) override;
        [[nodiscard]] bool hasPendingResize() const override;
        [[nodiscard]] bool pendingResizeReady() const override;
        [[nodiscard]] double secondsUntilPendingResizeReady() const override;
        [[nodiscard]] lfs::Result<std::optional<GraphicsFrame>>
        beginFrame(const std::array<float, 4>& clear_color) override;
        [[nodiscard]] lfs::Status endFrame() override;
        [[nodiscard]] bool presentBootstrapFrame(float r, float g, float b, float a) override;
        [[nodiscard]] bool hasActiveFrame() const noexcept override;
        [[nodiscard]] lfs::Result<WindowCapture> captureFinalFrameRgba() override;
        [[nodiscard]] bool waitForNextFrameSlot() override;
        [[nodiscard]] bool waitForSubmittedFrames() override;
        [[nodiscard]] bool waitIdle() override;
        [[nodiscard]] RendererTerminalState terminalState() const noexcept override;
        [[nodiscard]] const std::string& lastError() const noexcept override;
        [[nodiscard]] GraphicsCapabilities capabilities() const noexcept override;
        void flushPipelineCache() override;
        [[nodiscard]] lfs::core::SplatTensorAllocator
        splatTensorAllocator(bool preserve_float_shN = false) override;
        void connectTensorBackend() override;
        void disconnectTensorBackend() override;

        [[nodiscard]] VulkanContext& vulkanContext() noexcept { return context_; }
        [[nodiscard]] const VulkanContext& vulkanContext() const noexcept { return context_; }
        [[nodiscard]] const VulkanContext::Frame* referenceFrame(const GraphicsFrame& frame) const noexcept;

    private:
        VulkanContext context_;
        VulkanContext::Frame frame_{};
        std::uint64_t frame_generation_ = 0;
        bool tensor_backend_adopted_ = false;
        bool headless_ = false;
    };

    [[nodiscard]] LFS_VIS_API VulkanContext* vulkanContextOrNull(GraphicsContext* context) noexcept;
    [[nodiscard]] LFS_VIS_API const VulkanContext::Frame* vulkanFrameOrNull(
        const GraphicsContext* context, const GraphicsFrame& frame) noexcept;

} // namespace lfs::vis
