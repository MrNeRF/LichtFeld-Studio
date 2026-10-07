/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "scene_renderer_factory.hpp"
#ifdef LFS_GRAPHICS_VULKAN
#include "window/vulkan_graphics_context.hpp"
#endif
#include <format>
#include <stdexcept>

namespace lfs::vis {
    namespace {
        lfs::Error noTensorOutput(std::string_view renderer) {
            return lfs::make_error({.code = lfs::ErrorCode::Unsupported,
                                    .domain = lfs::ErrorDomain::Rendering,
                                    .detail = std::format("This {} renderer has no device-resident tensor output", renderer),
                                    .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    } // namespace
    lfs::Result<SceneRenderer::OutputTensors> SceneRenderer::readOutputTensors(RenderTargetId) const {
        return lfs::Result<SceneRenderer::OutputTensors>(noTensorOutput("scene"));
    }
    lfs::Result<SceneRenderer::OutputTensors> PointSceneRenderer::readOutputTensors(RenderTargetId) const {
        return lfs::Result<SceneRenderer::OutputTensors>(noTensorOutput("point"));
    }
} // namespace lfs::vis

#ifdef LFS_TENSOR_METAL
#include "core/gpu_backend_fwd.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "metal_viewport_renderer.hpp"
#include "python/python_runtime.hpp"
#include <atomic>
#include <format>
#include <unordered_map>
#include <unordered_set>

namespace lfs::vis {
    namespace {
        lfs::Error outputError(std::string detail) {
            return lfs::make_error({.code = lfs::ErrorCode::Unavailable,
                                    .domain = lfs::ErrorDomain::Rendering,
                                    .detail = std::move(detail),
                                    .detection = LFS_SOURCE_SITE_CURRENT()});
        }
#ifndef LFS_GRAPHICS_VULKAN
        lfs::Result<SceneRenderer::OutputTensors> copyOutputTensors(
            const MetalViewportRenderer& native, RenderTargetId target) {
            const auto size = native.size(target);
            if (size.x <= 0 || size.y <= 0)
                return lfs::Result<SceneRenderer::OutputTensors>(outputError(std::format("Metal output is unavailable (target={})", target.value)));
            const core::GpuBackendScope scope(core::GpuBackend::Metal);
            auto color = std::make_shared<core::Tensor>(core::Tensor::empty({size_t(size.y), size_t(size.x), 4}, core::Device::GPU, core::DataType::UInt8));
            auto depth = std::make_shared<core::Tensor>(core::Tensor::empty({size_t(size.y), size_t(size.x)}, core::Device::GPU, core::DataType::Float32));
            if (auto status = native.copyOutputs(target, *color, depth.get()); !status)
                return lfs::Result<SceneRenderer::OutputTensors>(std::move(status).error());
            return SceneRenderer::OutputTensors{std::move(color), std::move(depth)};
        }
#endif
        class MetalSceneRenderer final : public SceneRenderer {
            MetalViewportPresentation& context_;
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
            explicit MetalSceneRenderer(MetalViewportPresentation& context) : context_(context) {}
            std::expected<RenderResult, std::string> render(const core::SplatData& m,
                                                            const rendering::ViewportRenderRequest& r, bool, RenderTargetId t, bool = false, bool deterministic = false) override {
                if (!t.valid() || released_.contains(t))
                    return std::unexpected(std::format("Invalid or released Metal render target (target={})", t.value));
                try {
                    auto& n = native();
                    n.setLodSettings(pool_, fraction_, fade_);
                    auto result = legacyMetalResult(n.render(context_, m, r, t, expected_, deterministic || capture_));
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
                            result = legacyMetalResult(n.render(context_, m, r, t, expected_, true));
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
            auto readColorImage(RenderTargetId t, OutputImageFormat format) const -> std::expected<std::shared_ptr<core::Tensor>, std::string> override {
                const bool rgba = format == OutputImageFormat::RgbaFloat || format == OutputImageFormat::Rgba8;
                const bool bytes = format == OutputImageFormat::Rgb8 || format == OutputImageFormat::Rgba8;
                return readImage(t, rgba ? 4 : 3, bytes ? core::DataType::UInt8 : core::DataType::Float32);
            }
            auto readPreviewDepth(RenderTargetId t) const -> std::expected<std::shared_ptr<core::Tensor>, std::string> override {
                if (!hasRenderTarget(t))
                    return std::unexpected(std::format("Metal depth output is unavailable (target={})", t.value));
                const auto size = native().size(t);
                auto tensor = core::Tensor::empty({size_t(size.y), size_t(size.x)}, core::Device::CPU, core::DataType::Float32);
                auto ticket = submitReadOutputDepthImageTicket(t, tensor);
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
            auto readOutputImageIntoCpuHwc(RenderTargetId t, core::Tensor& dst, int x, int y) const -> std::expected<void, std::string> override { return legacyMetalResult(native().readColor(t, dst, x, y)); }
            auto sampleDepthAtPixel(const DepthSampleRequest& r) const -> std::expected<float, std::string> override { return legacyMetalResult(native().readDepth(r)); }
            auto submitReadbackTicket(const ReadbackRequest& request) const -> std::expected<uint64_t, std::string> override {
                return legacyMetalResult(native().submitReadback(request.target, request.destination, request.offset.x, request.offset.y,
                                                                 request.kind == ReadbackRequest::Kind::Depth));
            }
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
            ReadbackStats readbackStats() const override { return {native_ ? native_->outstandingReadbacks() : 0, 0, 0}; }
            auto buildSelectionMask(const core::SplatData& m, const SelectionMaskRequest& r, bool) -> std::expected<core::Tensor, std::string> override {
                try {
                    return legacyMetalResult(native().buildSelectionMask(context_, m, r));
                } catch (const std::exception& e) { return std::unexpected(e.what()); }
            }
#ifndef LFS_GRAPHICS_VULKAN
            auto readOutputTensors(RenderTargetId t) const -> lfs::Result<OutputTensors> override {
                if (!hasRenderTarget(t))
                    return lfs::Result<SceneRenderer::OutputTensors>(outputError(std::format("Metal scene output is unavailable (target={})", t.value)));
                return copyOutputTensors(native(), t);
            }
#endif
            std::shared_ptr<void> retainOutputImage(SceneImageViewHandle image) override { return native_ ? native_->retainOutputImage(image) : nullptr; }
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
            void configureLod(const LodSettings& settings) override {
                pool_ = settings.page_pool_splats;
                fraction_ = settings.pool_vram_fraction;
                fade_ = settings.fade_frames;
            }
            GpuLodSelectionStatus gpuLodSelectionStatus(RenderTargetId t) const override { return native_ ? native_->gpuLodSelectionStatus(t) : GpuLodSelectionStatus{}; }
        };
        class MetalPointSceneRenderer final : public PointSceneRenderer {
            MetalViewportPresentation& context_;
            std::unique_ptr<MetalViewportRenderer> native_;
            std::shared_ptr<std::atomic_bool> retry_ = std::make_shared<std::atomic_bool>(false);
            struct PointUpload {
                core::Tensor source_positions, source_colors, positions, colors;
                uint64_t positions_revision = 0, colors_revision = 0;
            };
            std::unordered_map<RenderTargetId, PointUpload, RenderTargetIdHash> uploads_;
            struct DepthReadback {
                glm::ivec2 pixel, source_size, extent;
                core::Tensor depth;
                uint64_t ticket = 0;
            };
            std::unordered_map<RenderTargetId, DepthReadback, RenderTargetIdHash> depth_readbacks_;
            void abandonDepthReadback(RenderTargetId target) {
                const auto it = depth_readbacks_.find(target);
                if (it == depth_readbacks_.end())
                    return;
                if (native_ && it->second.ticket)
                    native_->abandonReadback(it->second.ticket);
                depth_readbacks_.erase(it);
            }
            std::unordered_set<RenderTargetId, RenderTargetIdHash> outputs_, released_;

        public:
            explicit MetalPointSceneRenderer(MetalViewportPresentation& context) : context_(context) {}
            ~MetalPointSceneRenderer() override {
                while (!depth_readbacks_.empty())
                    abandonDepthReadback(depth_readbacks_.begin()->first);
            }
            auto render(const RenderRequest& r, RenderTargetId t) -> std::expected<RenderResult, std::string> override {
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
                    auto result = legacyMetalResult(native_->renderPoints(context_, request, t));
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
            lfs::Result<float> sampleDepthAtPixel(const DepthSampleRequest& request) override {
                if (!native_ || !hasRenderTarget(request.target))
                    return outputError("Metal point depth output is unavailable");
                if (!request.nonblocking) {
                    abandonDepthReadback(request.target);
                    return native_->readDepth({.pixel = request.pixel, .source_size = request.source_size, .target = request.target});
                }
                const auto extent = native_->size(request.target);
                if (extent.x <= 0 || extent.y <= 0)
                    return -1.0f;
                auto pixel = request.pixel;
                if (request.source_size.x > 0 && request.source_size.y > 0)
                    pixel = glm::clamp(glm::ivec2(glm::round((glm::vec2(pixel) + 0.5f) * glm::vec2(extent) /
                                                                 glm::vec2(request.source_size) -
                                                             0.5f)),
                                       glm::ivec2(0), extent - 1);
                if (pixel.x < 0 || pixel.y < 0 || pixel.x >= extent.x || pixel.y >= extent.y)
                    return -1.0f;
                auto it = depth_readbacks_.find(request.target);
                if (it != depth_readbacks_.end() &&
                    (it->second.pixel != request.pixel || it->second.source_size != request.source_size ||
                     it->second.extent != extent)) {
                    abandonDepthReadback(request.target);
                    it = depth_readbacks_.end();
                }
                if (it == depth_readbacks_.end()) {
                    DepthReadback pending{.pixel = request.pixel, .source_size = request.source_size, .extent = extent, .depth = core::Tensor::empty({size_t(extent.y), size_t(extent.x)}, core::Device::CPU, core::DataType::Float32)};
                    const auto submitted = native_->submitReadback(request.target, pending.depth, 0, 0, true);
                    if (!submitted)
                        return submitted.error();
                    pending.ticket = *submitted;
                    it = depth_readbacks_.emplace(request.target, std::move(pending)).first;
                }
                const auto ready = native_->pollReadback(it->second.ticket, false);
                if (!ready) {
                    abandonDepthReadback(request.target);
                    return ready.error();
                }
                if (*ready == SceneRenderer::ReadbackTicketStatus::NotReady) {
                    retry_->store(true, std::memory_order_release);
                    lfs::python::request_redraw_after(1.0 / 60.0);
                    return kDepthSamplePending;
                }
                if (*ready != SceneRenderer::ReadbackTicketStatus::Ready) {
                    abandonDepthReadback(request.target);
                    return outputError("Metal point depth readback failed");
                }
                const float depth = it->second.depth.ptr<float>()[size_t(pixel.y) * extent.x + pixel.x];
                depth_readbacks_.erase(it);
                return depth > 0.0f && depth < 1.0e9f ? depth : -1.0f;
            }
            bool takeRefinementRequest() override { return retry_->exchange(false, std::memory_order_acq_rel); }
            auto readOutputImage(RenderTargetId t) -> std::expected<std::shared_ptr<core::Tensor>, std::string> override {
                if (!hasRenderTarget(t))
                    return std::unexpected(std::format("Metal point output is unavailable (target={})", t.value));
                const auto size = native_->size(t);
                auto tensor = core::Tensor::empty({size_t(size.y), size_t(size.x), size_t(3)}, core::Device::CPU, core::DataType::Float32);
                auto ready = legacyMetalResult(native_->readColor(t, tensor, 0, 0));
                if (!ready)
                    return std::unexpected(ready.error());
                return std::make_shared<core::Tensor>(std::move(tensor));
            }
#ifndef LFS_GRAPHICS_VULKAN
            auto readOutputTensors(RenderTargetId t) const -> lfs::Result<SceneRenderer::OutputTensors> override {
                if (!hasRenderTarget(t) || !native_)
                    return lfs::Result<SceneRenderer::OutputTensors>(outputError(std::format("Metal point output is unavailable (target={})", t.value)));
                return copyOutputTensors(*native_, t);
            }
#endif
            bool hasRenderTarget(RenderTargetId t) const override { return outputs_.contains(t); }
            bool releaseRenderTarget(RenderTargetId t) override {
                abandonDepthReadback(t);
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
                while (!depth_readbacks_.empty())
                    abandonDepthReadback(depth_readbacks_.begin()->first);
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
#ifdef LFS_GRAPHICS_VULKAN
    std::unique_ptr<SceneRenderer> createSceneRenderer(GraphicsContext& graphics) {
        auto* context = vulkanContextOrNull(&graphics);
        if (!context)
            throw std::runtime_error("The Metal scene renderer requires Vulkan compositor resources");
        return std::make_unique<MetalSceneRenderer>(*context);
    }
    std::unique_ptr<PointSceneRenderer> createPointSceneRenderer(GraphicsContext& graphics) {
        auto* context = vulkanContextOrNull(&graphics);
        if (!context)
            throw std::runtime_error("The Metal point renderer requires Vulkan compositor resources");
        return std::make_unique<MetalPointSceneRenderer>(*context);
    }
#else
    namespace {
        // Stateless: tensor-compositor outputs need no graphics resources.
        MetalViewportPresentation tensor_presentation;
    } // namespace
    std::unique_ptr<SceneRenderer> createSceneRenderer(GraphicsContext&) {
        return std::make_unique<MetalSceneRenderer>(tensor_presentation);
    }
    std::unique_ptr<PointSceneRenderer> createPointSceneRenderer(GraphicsContext&) {
        return std::make_unique<MetalPointSceneRenderer>(tensor_presentation);
    }
#endif
    void preloadSceneRenderer() {}
} // namespace lfs::vis
#else
#include "vksplat_viewport_renderer.hpp"
#include "vulkan_scene_renderer_factory.hpp"
namespace lfs::vis {
    std::unique_ptr<SceneRenderer> createSceneRenderer(GraphicsContext& graphics) {
        auto* context = vulkanContextOrNull(&graphics);
        if (!context)
            throw std::runtime_error("Vulkan scene rendering requires Vulkan graphics resources");
        return createVulkanSceneRenderer(*context);
    }
    std::unique_ptr<PointSceneRenderer> createPointSceneRenderer(GraphicsContext& graphics) {
        auto* context = vulkanContextOrNull(&graphics);
        if (!context)
            throw std::runtime_error("Vulkan point rendering requires Vulkan graphics resources");
        return createVulkanPointSceneRenderer(*context);
    }
    void preloadSceneRenderer() { preloadVkSplatSpirvFiles(); }
} // namespace lfs::vis
#endif
