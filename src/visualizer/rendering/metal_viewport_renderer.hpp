/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#ifdef __APPLE__
#include "core/error.hpp"
#include <array>
#include "point_cloud_vulkan_renderer.hpp"
#include "vksplat_viewport_renderer.hpp"

namespace lfs::vis {
    // Explicit adapter at the pre-existing desktop string-error boundary.
    // Native callers retain the structured Result/Status instead.
    template <class T>
    auto legacyMetalResult(lfs::Result<T> result) {
        return std::move(result).into_expected().transform_error(
            [](const lfs::Error& error) { return lfs::format_for_developer(error); });
    }

    // Native rasterization with imported textures for the existing desktop compositor.
    // This boundary contains no Objective-C types, keeping Apple headers out of the
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
            double gpu_command_ms = 0;
            bool counter_timestamps_available = false;
            std::array<double, 5> gpu_stage_ms{}; // projection, instances, sort, blend, present
        };
        void setProfilingEnabled(bool);
        lfs::Result<FrameDiagnostics> frameDiagnostics(VksplatViewportRenderer::OutputSlot) const;
        void setLodSettings(size_t pool_splats, float vram_fraction, uint32_t fade_frames);
        static bool supportsSelection(const core::SplatData&, const VksplatViewportRenderer::SelectionMaskRequest&);
        lfs::Result<core::Tensor> buildSelectionMask(VulkanContext&, const core::SplatData&, const VksplatViewportRenderer::SelectionMaskRequest&);
        static bool supportsPoints(const PointCloudVulkanRenderer::RenderRequest&);
        lfs::Result<PointCloudVulkanRenderer::RenderResult> renderPoints(
            VulkanContext&, const PointCloudVulkanRenderer::RenderRequest&, PointCloudVulkanRenderer::OutputSlot);
        lfs::Result<VksplatViewportRenderer::RenderResult> render(
            VulkanContext&, const core::SplatData&, const rendering::ViewportRenderRequest&,
            VksplatViewportRenderer::OutputSlot, bool expected_depth = false, bool wait_for_pages = false);
        glm::ivec2 size(VksplatViewportRenderer::OutputSlot) const;
        // Explicit validation/readback boundary: waits for the native command and
        // distinguishes a complete image from capacity-overflow fallback output.
        lfs::Result<bool> outputComplete(VksplatViewportRenderer::OutputSlot) const;
        // Deferred diagnostic readback: never waits for the live GPU cut.
        VksplatViewportRenderer::GpuLodSelectionStatus gpuLodSelectionStatus(VksplatViewportRenderer::OutputSlot) const;
        lfs::Status readColor(VksplatViewportRenderer::OutputSlot,
                              core::Tensor&, int x, int y) const;
        lfs::Result<float> readDepth(const VksplatViewportRenderer::DepthSampleRequest&) const;
        // Ticket storage retains GPU staging, never a host destination after abandon.
        static bool nativeTicket(uint64_t ticket) { return (ticket >> 63) != 0; }
        lfs::Result<uint64_t> submitReadback(VksplatViewportRenderer::OutputSlot,
                                             core::Tensor&, int x, int y, bool depth) const;
        lfs::Result<VksplatViewportRenderer::ReadbackTicketStatus> pollReadback(uint64_t, bool wait) const;
        void abandonReadback(uint64_t) const;
        size_t outstandingReadbacks() const;
        lfs::Status release(VksplatViewportRenderer::OutputSlot);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
#endif
