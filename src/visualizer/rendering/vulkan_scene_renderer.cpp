/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "point_cloud_vulkan_renderer.hpp"
#include "scene_training_interop.hpp"
#include "vksplat_viewport_renderer.hpp"
#include "vulkan_scene_renderer_factory.hpp"
namespace lfs::vis {
    namespace {
        class VulkanTrainingInterop final : public SceneTrainingInterop {
            VulkanContext& context_;
            VksplatViewportRenderer& renderer_;

        public:
            VulkanTrainingInterop(VulkanContext& context, VksplatViewportRenderer& renderer) : context_(context), renderer_(renderer) {}
            bool hasLiveTrainerReleaseFence() const override { return renderer_.hasLiveTrainerReleaseFence(); }
            void* renderCompleteTimeline() const override { return renderer_.renderCompleteTimeline(); }
            uint64_t renderCompleteValue() const override { return renderer_.renderCompleteValue(); }
            std::expected<void, std::string> ensureHandshakeReady() override { return renderer_.ensureHandshakeReady(context_); }
            std::expected<void, std::string> ensureTrainingSharedScratchReady(size_t count, glm::ivec2 size) override { return renderer_.ensureTrainingSharedScratchReady(context_, count, size); }
            void releaseScratchOnIdle(bool shared, bool reclaim) override { renderer_.releaseScratchOnIdle(shared, reclaim); }
            void requestArenaHandoff() override { renderer_.requestArenaHandoff(); }
            void cancelArenaHandoff() override { renderer_.cancelArenaHandoff(); }
            bool pollArenaHandoff() override { return renderer_.pollArenaHandoff(); }
            bool waitForArenaHandoff(std::chrono::milliseconds timeout) override { return renderer_.waitForArenaHandoff(timeout); }
            void setLiveSubmitCallback(std::function<void(uint64_t)> callback) override { renderer_.setLiveSubmitCallback(std::move(callback)); }
        };
        class VulkanSceneRenderer final : public SceneRenderer {
            VulkanContext& context_;
            VksplatViewportRenderer renderer_;
            VulkanTrainingInterop training_;

        public:
            explicit VulkanSceneRenderer(VulkanContext& context) : context_(context), training_(context, renderer_) {}
            SceneTrainingInterop* trainingInterop() override { return &training_; }
            std::expected<void, std::string> prepareDevice() override { return renderer_.prepareDevice(context_); }
            std::expected<RenderResult, std::string> render(const core::SplatData& model, const rendering::ViewportRenderRequest& request,
                                                            bool upload, RenderTargetId target, bool synchronize, bool deterministic) override {
                return renderer_.render(context_, model, request, upload, target, synchronize, deterministic);
            }
            std::expected<RenderResult, std::string> rerenderSelectionOverlay(const core::SplatData& model,
                                                                              const rendering::ViewportRenderRequest& request, RenderTargetId target, bool synchronize) override {
                return renderer_.rerenderSelectionOverlay(context_, model, request, target, synchronize);
            }
            bool nextOutputImagesNeedResize(glm::ivec2 size, RenderTargetId target) const override { return renderer_.nextOutputImagesNeedResize(size, target); }
            std::expected<std::shared_ptr<core::Tensor>, std::string> readColorImage(RenderTargetId target, OutputImageFormat format) const override {
                switch (format) {
                case OutputImageFormat::RgbFloat: return renderer_.readOutputImage(context_, target);
                case OutputImageFormat::RgbaFloat: return renderer_.readOutputImageRgba(context_, target);
                case OutputImageFormat::Rgb8: return renderer_.readOutputImageRgb8(context_, target);
                case OutputImageFormat::Rgba8: return renderer_.readOutputImageRgba8(context_, target);
                }
                return std::unexpected("Unsupported color output format");
            }
            std::expected<std::shared_ptr<core::Tensor>, std::string> readPreviewDepth(RenderTargetId target) const override { return renderer_.readPreviewDepth(context_, target); }
            void setDepthCaptureMode(bool enabled, bool expected) override { renderer_.setDepthCaptureMode(enabled, expected); }
            std::expected<void, std::string> readOutputImageIntoCpuHwc(RenderTargetId target, core::Tensor& destination, int x, int y) const override { return renderer_.readOutputImageIntoCpuHwc(context_, target, destination, x, y); }
            std::expected<float, std::string> sampleDepthAtPixel(const DepthSampleRequest& request) const override { return renderer_.sampleDepthAtPixel(context_, request); }
            std::expected<uint64_t, std::string> submitReadbackTicket(const ReadbackRequest& request) const override {
                if (request.kind == ReadbackRequest::Kind::Depth)
                    return renderer_.submitReadOutputDepthImageTicket(context_, request.target, request.destination);
                return renderer_.submitReadOutputImageIntoCpuHwcTicket(context_, request.target, request.destination, request.offset.x, request.offset.y);
            }
            std::expected<ReadbackTicketStatus, std::string> pollReadbackTicket(uint64_t ticket) const override { return renderer_.pollReadbackTicket(ticket); }
            std::expected<void, std::string> waitReadbackTicket(uint64_t ticket) const override { return renderer_.waitReadbackTicket(ticket); }
            void abandonReadbackTicket(uint64_t ticket) const override { renderer_.abandonReadbackTicket(ticket); }
            ReadbackStats readbackStats() const override { return renderer_.readbackStats(); }
            std::expected<core::Tensor, std::string> buildSelectionMask(const core::SplatData& model, const SelectionMaskRequest& request, bool synchronize) override { return renderer_.buildSelectionMask(context_, model, request, synchronize); }
            bool hasRenderTarget(RenderTargetId target) const override { return renderer_.hasRenderTarget(target); }
            bool releaseRenderTarget(RenderTargetId target) override { return renderer_.releaseRenderTarget(target); }
            void releaseSceneResources() override { renderer_.releaseSceneResources(); }
            void reset() override { renderer_.reset(); }
            void configureLod(const LodSettings& settings) override {
                renderer_.setLodPagePoolBudget(settings.page_pool_splats);
                renderer_.setLodPoolVramFraction(settings.pool_vram_fraction);
                renderer_.setLodFadeFrames(settings.fade_frames);
            }
            GpuLodSelectionStatus gpuLodSelectionStatus(RenderTargetId target) const override { return renderer_.gpuLodSelectionStatus(target); }
            std::optional<LodPageCache::Snapshot> ensureLodPageCacheSnapshot(const core::SplatData& model) override { return renderer_.ensureLodPageCacheSnapshot(model); }
            void setCameraNavigating(bool navigating) override { renderer_.setCameraNavigating(navigating); }
        };
        class VulkanPointSceneRenderer final : public PointSceneRenderer {
            VulkanContext& context_;
            PointCloudVulkanRenderer renderer_;

        public:
            explicit VulkanPointSceneRenderer(VulkanContext& context) : context_(context) {}
            std::expected<RenderResult, std::string> render(const RenderRequest& request, RenderTargetId target) override { return renderer_.render(context_, request, target); }
            std::expected<std::shared_ptr<core::Tensor>, std::string> readOutputImage(RenderTargetId target) override { return renderer_.readOutputImage(context_, target); }
            bool hasRenderTarget(RenderTargetId target) const override { return renderer_.hasRenderTarget(target); }
            bool releaseRenderTarget(RenderTargetId target) override { return renderer_.releaseRenderTarget(target); }
            void reset() override { renderer_.reset(); }
        };
    } // namespace
    std::unique_ptr<SceneRenderer> createVulkanSceneRenderer(VulkanContext& context) { return std::make_unique<VulkanSceneRenderer>(context); }
    std::unique_ptr<PointSceneRenderer> createVulkanPointSceneRenderer(VulkanContext& context) { return std::make_unique<VulkanPointSceneRenderer>(context); }
} // namespace lfs::vis
