/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "scene_renderer.hpp"
#ifdef __APPLE__
#include "core/logger.hpp"
#include "metal_viewport_renderer.hpp"
#include "python/python_runtime.hpp"
#include <atomic>
#include <format>
#include <unordered_map>
#include <unordered_set>

namespace lfs::vis {
    namespace {
        class MetalSceneRenderer final : public SceneRenderer {
            mutable std::unique_ptr<MetalViewportRenderer> native_;
            std::shared_ptr<std::atomic_bool> retry_ = std::make_shared<std::atomic_bool>(false);
            std::unordered_set<RenderTargetId, RenderTargetIdHash> outputs_, released_;
            size_t pool_ = 0;
            float fraction_ = .6f;
            uint32_t fade_ = 8;
            bool capture_ = false, expected_ = false;
            MetalViewportRenderer& native() const {
                if (!native_) {
                    native_ = std::make_unique<MetalViewportRenderer>();
                    const auto retry = retry_;
                    native_->setRetryCallback([retry] {
                        retry->store(true, std::memory_order_release);
                        lfs::python::request_redraw();
                    });
                }
                return *native_;
            }
            std::expected<std::shared_ptr<core::Tensor>, std::string> readImage(RenderTargetId target, size_t channels, core::DataType dtype) const {
                if (!hasRenderTarget(target))
                    return std::unexpected(std::format("Metal color output is unavailable (target={})", target.value));
                const auto size = native().size(target);
                auto tensor = core::Tensor::empty({size_t(size.y), size_t(size.x), channels}, core::Device::CPU, dtype);
                auto status = legacyMetalResult(native().readColor(target, tensor, 0, 0));
                if (!status)
                    return std::unexpected(status.error());
                return std::make_shared<core::Tensor>(std::move(tensor));
            }

        public:
            std::expected<RenderResult, std::string> render(VulkanContext& c, const core::SplatData& m,
                                                            const rendering::ViewportRenderRequest& r, bool, RenderTargetId t, bool = false, bool deterministic = false) override {
                if (!t.valid() || released_.contains(t))
                    return std::unexpected(std::format("Invalid or released Metal render target (target={})", t.value));
                try {
                    auto& n = native();
                    n.setLodSettings(pool_, fraction_, fade_);
                    auto result = legacyMetalResult(n.render(c, m, r, t, expected_, deterministic || capture_));
                    if (!result)
                        return result;
                    if (deterministic || capture_) {
                        for (int attempt = 0;; ++attempt) {
                            auto complete = legacyMetalResult(n.outputComplete(t));
                            if (!complete)
                                return std::unexpected(complete.error());
                            if (*complete)
                                break;
                            if (attempt == 4)
                                return std::unexpected(std::format("Metal export reservation did not converge (target={}, attempts={})", t.value, attempt + 1));
                            result = legacyMetalResult(n.render(c, m, r, t, expected_, true));
                            if (!result)
                                return result;
                        }
                    }
                    outputs_.insert(t);
                    return result;
                } catch (const std::exception& e) { return std::unexpected(e.what()); }
            }
            bool takeRefinementRequest() override { return retry_->exchange(false, std::memory_order_acq_rel); }
            bool nextOutputImagesNeedResize(glm::ivec2 size, RenderTargetId t) const override {
                return hasRenderTarget(t) && native().size(t) != size;
            }
            auto readOutputImage(VulkanContext&, RenderTargetId t) const -> std::expected<std::shared_ptr<core::Tensor>, std::string> override { return readImage(t, 3, core::DataType::Float32); }
            auto readOutputImageRgba(VulkanContext&, RenderTargetId t) const -> std::expected<std::shared_ptr<core::Tensor>, std::string> override { return readImage(t, 4, core::DataType::Float32); }
            auto readOutputImageRgb8(VulkanContext&, RenderTargetId t) const -> std::expected<std::shared_ptr<core::Tensor>, std::string> override { return readImage(t, 3, core::DataType::UInt8); }
            auto readOutputImageRgba8(VulkanContext&, RenderTargetId t) const -> std::expected<std::shared_ptr<core::Tensor>, std::string> override { return readImage(t, 4, core::DataType::UInt8); }
            auto readPreviewDepth(VulkanContext& c, RenderTargetId t) const -> std::expected<std::shared_ptr<core::Tensor>, std::string> override {
                if (!hasRenderTarget(t))
                    return std::unexpected(std::format("Metal depth output is unavailable (target={})", t.value));
                const auto size = native().size(t);
                auto tensor = core::Tensor::empty({size_t(size.y), size_t(size.x)}, core::Device::CPU, core::DataType::Float32);
                auto ticket = submitReadOutputDepthImageTicket(c, t, tensor);
                if (!ticket)
                    return std::unexpected(ticket.error());
                auto ready = waitReadbackTicket(*ticket);
                if (!ready)
                    return std::unexpected(ready.error());
                return std::make_shared<core::Tensor>(std::move(tensor));
            }
            void setDepthCaptureMode(bool on, bool expected) override {
                capture_ = on;
                expected_ = on && expected;
            }
            auto readOutputImageIntoCpuHwc(VulkanContext&, RenderTargetId t, core::Tensor& dst, int x, int y) const -> std::expected<void, std::string> override { return legacyMetalResult(native().readColor(t, dst, x, y)); }
            auto sampleDepthAtPixel(VulkanContext&, const DepthSampleRequest& r) const -> std::expected<float, std::string> override { return legacyMetalResult(native().readDepth(r)); }
            auto submitReadOutputImageIntoCpuHwcTicket(VulkanContext&, RenderTargetId t, core::Tensor& dst, int x, int y) const -> std::expected<uint64_t, std::string> override { return legacyMetalResult(native().submitReadback(t, dst, x, y, false)); }
            auto submitReadOutputDepthImageTicket(VulkanContext&, RenderTargetId t, core::Tensor& dst) const -> std::expected<uint64_t, std::string> override { return legacyMetalResult(native().submitReadback(t, dst, 0, 0, true)); }
            auto pollReadbackTicket(uint64_t t) const -> std::expected<ReadbackTicketStatus, std::string> override { return legacyMetalResult(native().pollReadback(t, false)); }
            auto waitReadbackTicket(uint64_t t) const -> std::expected<void, std::string> override {
                const auto ready = legacyMetalResult(native().pollReadback(t, true));
                if (!ready)
                    return std::unexpected(ready.error());
                if (*ready != ReadbackTicketStatus::Ready)
                    return std::unexpected(std::format("Metal readback did not complete (ticket={}, status={})", t, int(*ready)));
                return {};
            }
            void abandonReadbackTicket(uint64_t t) const override {
                if (native_)
                    native_->abandonReadback(t);
            }
            size_t outstandingReadbackTickets() const override { return native_ ? native_->outstandingReadbacks() : 0; }
            auto buildSelectionMask(VulkanContext& c, const core::SplatData& m, const SelectionMaskRequest& r, bool) -> std::expected<core::Tensor, std::string> override {
                try {
                    return legacyMetalResult(native().buildSelectionMask(c, m, r));
                } catch (const std::exception& e) { return std::unexpected(e.what()); }
            }
            bool hasRenderTarget(RenderTargetId t) const override { return outputs_.contains(t); }
            bool releaseRenderTarget(RenderTargetId t) override {
                if (native_) {
                    auto status = native_->release(t);
                    if (!status) {
                        LOG_ERROR("{}", lfs::format_for_developer(status.error()));
                        return false;
                    }
                }
                outputs_.erase(t);
                released_.insert(t);
                return true;
            }
            void releaseSceneResources() override {
                if (native_) {
                    auto status = native_->releaseAll();
                    if (!status)
                        throw lfs::Exception(status.error());
                }
                outputs_.clear();
            }
            void reset() override { releaseSceneResources(); }
            void setLodPagePoolBudget(size_t v) override { pool_ = v; }
            void setLodPoolVramFraction(float v) override { fraction_ = v; }
            void setLodFadeFrames(uint32_t v) override { fade_ = v; }
            GpuLodSelectionStatus gpuLodSelectionStatus(RenderTargetId t) const override { return native_ ? native_->gpuLodSelectionStatus(t) : GpuLodSelectionStatus{}; }
        };
        class MetalPointSceneRenderer final : public PointSceneRenderer {
            std::unique_ptr<MetalViewportRenderer> native_;
            std::shared_ptr<std::atomic_bool> retry_ = std::make_shared<std::atomic_bool>(false);
            struct PointUpload {
                core::Tensor source_positions, source_colors, positions, colors;
                uint64_t positions_revision = 0, colors_revision = 0;
            };
            std::unordered_map<RenderTargetId, PointUpload, RenderTargetIdHash> uploads_;
            std::unordered_set<RenderTargetId, RenderTargetIdHash> outputs_, released_;

        public:
            auto render(VulkanContext& c, const RenderRequest& r, RenderTargetId t) -> std::expected<RenderResult, std::string> override {
                if (!t.valid() || released_.contains(t))
                    return std::unexpected(std::format("Invalid or released Metal point target (target={})", t.value));
                try {
                    if (!native_) {
                        native_ = std::make_unique<MetalViewportRenderer>();
                        const auto retry = retry_;
                        native_->setRetryCallback([retry] {
                            retry->store(true, std::memory_order_release);
                            lfs::python::request_redraw();
                        });
                    }
                    auto request = r;
                    std::vector<core::Tensor> staged;
                    staged.reserve(6);
                    const auto stage = [&](const core::Tensor*& tensor) {
                        if (tensor && tensor->is_valid() && tensor->device() == core::Device::CPU) {
                            staged.push_back(tensor->to(core::Device::GPU));
                            tensor = &staged.back();
                        }
                    };
                    auto& upload = uploads_[t];
                    const auto cache = [](const core::Tensor*& input, core::Tensor& source, core::Tensor& resident,
                                          uint64_t revision, uint64_t& cached_revision, bool colors) {
                        if (!input || !input->is_valid() ||
                            (input->device() != core::Device::CPU && (!colors || input->dtype() == core::DataType::Float32)))
                            return;
                        if (!source.is_valid() || source.data_ptr() != input->data_ptr() ||
                            source.shape() != input->shape() || source.dtype() != input->dtype() || cached_revision != revision) {
                            // Retain the original owner as well as its identity: a freed
                            // source address must not alias another scene's cached data.
                            source = *input;
                            auto converted = colors ? input->to(core::DataType::Float32) : *input;
                            if (colors && input->dtype() == core::DataType::UInt8)
                                converted = converted / 255.f;
                            resident = converted.to(core::Device::GPU).contiguous();
                            cached_revision = revision;
                        }
                        input = &resident;
                    };
                    cache(request.positions, upload.source_positions, upload.positions,
                          request.positions_revision, upload.positions_revision, false);
                    cache(request.colors, upload.source_colors, upload.colors,
                          request.colors_revision, upload.colors_revision, true);
                    stage(request.transform_indices);
                    stage(request.deleted_mask);
                    stage(request.selection_mask);
                    stage(request.preview_selection_mask);
                    auto result = legacyMetalResult(native_->renderPoints(c, request, t));
                    if (result && r.synchronize_output) {
                        const auto complete = legacyMetalResult(native_->outputComplete(t));
                        if (!complete)
                            return std::unexpected(complete.error());
                        if (!*complete)
                            return std::unexpected("Metal point import output did not complete");
                    }
                    if (result)
                        outputs_.insert(t);
                    return result;
                } catch (const std::exception& e) { return std::unexpected(e.what()); }
            }
            bool takeRefinementRequest() override { return retry_->exchange(false, std::memory_order_acq_rel); }
            auto readOutputImage(VulkanContext&, RenderTargetId t) -> std::expected<std::shared_ptr<core::Tensor>, std::string> override {
                if (!hasRenderTarget(t))
                    return std::unexpected(std::format("Metal point output is unavailable (target={})", t.value));
                const auto size = native_->size(t);
                auto tensor = core::Tensor::empty({size_t(size.y), size_t(size.x), size_t(3)}, core::Device::CPU, core::DataType::Float32);
                auto ready = legacyMetalResult(native_->readColor(t, tensor, 0, 0));
                if (!ready)
                    return std::unexpected(ready.error());
                return std::make_shared<core::Tensor>(std::move(tensor));
            }
            bool hasRenderTarget(RenderTargetId t) const override { return outputs_.contains(t); }
            bool releaseRenderTarget(RenderTargetId t) override {
                if (native_) {
                    auto status = native_->release(t);
                    if (!status)
                        return false;
                }
                uploads_.erase(t);
                outputs_.erase(t);
                released_.insert(t);
                return true;
            }
            void reset() override {
                if (native_) {
                    auto status = native_->releaseAll();
                    if (!status)
                        throw lfs::Exception(status.error());
                }
                uploads_.clear();
                outputs_.clear();
            }
        };
    } // namespace
    std::unique_ptr<SceneRenderer> createSceneRenderer() { return std::make_unique<MetalSceneRenderer>(); }
    std::unique_ptr<PointSceneRenderer> createPointSceneRenderer() { return std::make_unique<MetalPointSceneRenderer>(); }
    void preloadSceneRenderer() {}
} // namespace lfs::vis
#else
#include "point_cloud_vulkan_renderer.hpp"
#include "vksplat_viewport_renderer.hpp"
namespace lfs::vis {
    std::unique_ptr<SceneRenderer> createSceneRenderer() { return std::make_unique<VksplatViewportRenderer>(); }
    std::unique_ptr<PointSceneRenderer> createPointSceneRenderer() { return std::make_unique<PointCloudVulkanRenderer>(); }
    void preloadSceneRenderer() { preloadVkSplatSpirvFiles(); }
} // namespace lfs::vis
#endif
