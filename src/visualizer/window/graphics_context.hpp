/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_fwd.hpp"
#include "renderer_terminal_state.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct SDL_Window;

namespace lfs::vis {

    enum class GraphicsResizeIntent : std::uint8_t {
        Interactive,
        Exact,
    };

    struct GraphicsCapabilities {
        bool native_metal = false;
        bool mesh_rendering = false;
        bool wireframe = false;
        bool wide_lines = false;
        bool external_memory_interop = false;
        bool external_semaphore_interop = false;
        bool environment_map = true;
        bool split_view = true;
        bool temporal_upscaling = true;
        bool mesh2splat = true;
        bool hdr_libplacebo = true;
    };

    struct WindowCapture {
        int width = 0;
        int height = 0;
        std::vector<std::uint8_t> rgba;
    };

    // Scheduling token only. Native presentation resources never leave the
    // window implementation; renderable application images are tensors.
    struct GraphicsFrame {
        std::size_t frame_slot = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint64_t generation = 0;
    };

    class LFS_VIS_API GraphicsContext {
    public:
        virtual ~GraphicsContext() = default;

        virtual bool initialize(SDL_Window* window, int framebuffer_width,
                                int framebuffer_height) = 0;
        virtual bool initializeHeadless() = 0;
        virtual void shutdown() = 0;

        virtual void notifyFramebufferResized(int width, int height,
                                              GraphicsResizeIntent intent) = 0;
        [[nodiscard]] virtual bool hasPendingResize() const = 0;
        [[nodiscard]] virtual bool pendingResizeReady() const = 0;
        [[nodiscard]] virtual double secondsUntilPendingResizeReady() const = 0;

        [[nodiscard]] virtual lfs::Result<std::optional<GraphicsFrame>>
        beginFrame(const std::array<float, 4>& clear_color) = 0;
        [[nodiscard]] virtual lfs::Status endFrame() = 0;
        // The Metal/tensor path renders the complete application frame into
        // this image. Vulkan's reference path renders into its native
        // swapchain and therefore returns nullptr until it is migrated.
        [[nodiscard]] virtual lfs::core::Tensor* finalImageTensor(const GraphicsFrame&) {
            return nullptr;
        }
        [[nodiscard]] virtual const lfs::core::Tensor* finalImageTensor(const GraphicsFrame&) const {
            return nullptr;
        }
        [[nodiscard]] virtual bool presentBootstrapFrame(float r, float g, float b, float a) = 0;
        [[nodiscard]] virtual bool hasActiveFrame() const noexcept = 0;
        [[nodiscard]] virtual lfs::Result<WindowCapture> captureFinalFrameRgba() = 0;

        [[nodiscard]] virtual bool waitForNextFrameSlot() = 0;
        [[nodiscard]] virtual bool waitForSubmittedFrames() = 0;
        [[nodiscard]] virtual bool waitIdle() = 0;

        [[nodiscard]] virtual RendererTerminalState terminalState() const noexcept = 0;
        [[nodiscard]] virtual const std::string& lastError() const noexcept = 0;
        [[nodiscard]] virtual GraphicsCapabilities capabilities() const noexcept = 0;
        virtual void flushPipelineCache() = 0;
        [[nodiscard]] virtual lfs::core::SplatTensorAllocator
        splatTensorAllocator(bool preserve_float_shN = false) = 0;

        // Backend-owned setup used while the window context is live.  This
        // keeps native tensor adoption out of WindowManager and its callers.
        virtual void connectTensorBackend() = 0;
        virtual void disconnectTensorBackend() = 0;
    };

    [[nodiscard]] LFS_VIS_API std::unique_ptr<GraphicsContext> createGraphicsContext();
} // namespace lfs::vis
