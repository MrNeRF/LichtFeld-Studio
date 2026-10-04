/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "graphics_context.hpp"

#include "core/tensor.hpp"

#include <atomic>
#include <memory>
#include <mutex>

namespace lfs::vis {

    class LFS_VIS_API MetalGraphicsContext final : public GraphicsContext {
    public:
        MetalGraphicsContext();
        ~MetalGraphicsContext() override;

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
        [[nodiscard]] lfs::core::Tensor* finalImageTensor(const GraphicsFrame& frame) override;
        [[nodiscard]] const lfs::core::Tensor* finalImageTensor(const GraphicsFrame& frame) const override;
        [[nodiscard]] bool presentBootstrapFrame(float r, float g, float b, float a) override;
        [[nodiscard]] bool hasActiveFrame() const noexcept override;
        [[nodiscard]] lfs::Result<WindowCapture> captureFinalFrameRgba() override;
        [[nodiscard]] std::future<lfs::Result<WindowCapture>> captureFinalFrameRgbaAsync() override;
        [[nodiscard]] bool waitForNextFrameSlot() override;
        [[nodiscard]] bool waitForSubmittedFrames() override;
        [[nodiscard]] bool waitIdle() override;
        [[nodiscard]] std::uint64_t lastSubmitSerial() const override;
        [[nodiscard]] std::uint64_t lastSuccessfulSubmitSerial() const override;
        [[nodiscard]] std::uint64_t completedSubmitSerial() const override;
        [[nodiscard]] bool waitForCompletedSubmitSerial(std::uint64_t serial) override;
        [[nodiscard]] RendererTerminalState terminalState() const noexcept override;
        [[nodiscard]] const std::string& lastError() const noexcept override;
        void noteFailure(const std::exception& exception) override;
        [[nodiscard]] GraphicsCapabilities capabilities() const noexcept override;
        void flushPipelineCache() override;
        [[nodiscard]] lfs::core::SplatTensorAllocator
        splatTensorAllocator(bool preserve_float_shN = false) override;
        void connectTensorBackend() override;
        void disconnectTensorBackend() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
