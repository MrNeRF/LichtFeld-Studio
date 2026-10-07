/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#ifdef __APPLE__
#include "core/error.hpp"
#include "scene_renderer.hpp"
#include <array>

namespace lfs::vis {
#ifdef LFS_GRAPHICS_VULKAN
    // Outputs are Vulkan-owned images exported to Metal for the Vulkan compositor.
    class VulkanContext;
    using MetalViewportPresentation = VulkanContext;
#else
    // Outputs are native Metal textures; copyOutputs moves them into tensors.
    struct MetalViewportPresentation {};
#endif
    // Explicit adapter at the pre-existing desktop string-error boundary.
    // Native callers retain the structured Result/Status instead.
    template <class T>
    auto legacyMetalResult(lfs::Result<T> result) {
        return std::move(result).into_expected().transform_error(
            [](const lfs::Error& error) { return lfs::format_for_developer(error); });
    }

    // Native rasterization into textures the compositor consumes: exported to
    // Vulkan, or copied into image tensors for the tensor compositor. This boundary contains no Objective-C types, keeping Apple headers out of the
    // cross-platform viewport and Python bindings.
    class LFS_VIS_API MetalViewportRenderer {
    public:
        MetalViewportRenderer();
        ~MetalViewportRenderer();
        static bool supports(const core::SplatData&, const rendering::ViewportRenderRequest&);
        // Diagnostics are opt-in and separate from ordinary rendering and wall latency.
        struct FrameDiagnostics {
            uint64_t required_instances = 0;
            uint32_t reserved_instances = 0, input_splats = 0;
            uint32_t blend_threads = 0; // Gaussian dispatch size from completed GPU status
            uint32_t maximum_tile_instances = 0;
            double gpu_command_ms = 0;
            bool counter_timestamps_available = false;
            std::array<double, 5> gpu_stage_ms{}; // projection, instances, sort, blend, present
        };
        void setProfilingEnabled(bool);
        // Wake only when a completed frame needs a larger reservation.
        void setRetryCallback(std::function<void()>);
        lfs::Result<FrameDiagnostics> frameDiagnostics(RenderTargetId) const;
        void setLodSettings(size_t pool_splats, float vram_fraction, uint32_t fade_frames);
        static bool supportsSelection(const core::SplatData&, const SceneRenderer::SelectionMaskRequest&);
        lfs::Result<core::Tensor> buildSelectionMask(MetalViewportPresentation&, const core::SplatData&, const SceneRenderer::SelectionMaskRequest&);
        static bool supportsPoints(const PointSceneRenderer::RenderRequest&);
        lfs::Result<PointSceneRenderer::RenderResult> renderPoints(
            MetalViewportPresentation&, const PointSceneRenderer::RenderRequest&, RenderTargetId);
        lfs::Result<SceneRenderer::RenderResult> render(
            MetalViewportPresentation&, const core::SplatData&, const rendering::ViewportRenderRequest&,
            RenderTargetId, bool expected_depth = false, bool wait_for_pages = false);
        glm::ivec2 size(RenderTargetId) const;
#ifndef LFS_GRAPHICS_VULKAN
        // Copies the latest output into contiguous [H,W,4] UInt8 color and
        // [H,W] Float32 depth tensors, ordered after its render on the GPU.
        lfs::Status copyOutputs(RenderTargetId, core::Tensor& color, core::Tensor* depth) const;
#endif
        // Explicit validation/readback boundary: waits for the native command and
        // distinguishes a complete image from capacity-overflow fallback output.
        lfs::Result<bool> outputComplete(RenderTargetId) const;
        // Deferred diagnostic readback: never waits for the live GPU cut.
        SceneRenderer::GpuLodSelectionStatus gpuLodSelectionStatus(RenderTargetId) const;
        lfs::Status readColor(RenderTargetId,
                              core::Tensor&, int x, int y) const;
        lfs::Result<float> readDepth(const SceneRenderer::DepthSampleRequest&) const;
        // Ticket storage retains GPU staging, never a host destination after abandon.
        static bool nativeTicket(uint64_t ticket) { return (ticket >> 63) != 0; }
        lfs::Result<uint64_t> submitReadback(RenderTargetId,
                                             core::Tensor&, int x, int y, bool depth) const;
        lfs::Result<SceneRenderer::ReadbackTicketStatus> pollReadback(uint64_t, bool wait) const;
        void abandonReadback(uint64_t) const;
        size_t outstandingReadbacks() const;
        std::shared_ptr<void> retainOutputImage(SceneImageViewHandle);
        lfs::Status release(RenderTargetId);
        lfs::Status releaseAll();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
#endif
