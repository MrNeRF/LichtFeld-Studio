/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/error.hpp"
#include "core/export.hpp"
#include "core/splat_data.hpp"
#include "lod_page_cache.hpp"
#include "render_target_id.hpp"
#include "rendering/rendering.hpp"
#include "scene_output.hpp"
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::vis {
    class SceneTrainingInterop;
    // Scene APIs are selected once per platform. Opaque output handles describe
    // compositor transport, independent of the API that rasterizes the scene.
    class LFS_VIS_API SceneRenderer {
    public:
        struct RenderResult {
            SceneImageHandle image;
            SceneImageViewHandle image_view;
            SceneImageLayout image_layout;
            std::uint64_t generation = 0;
            SceneImageHandle depth_image;
            SceneImageViewHandle depth_image_view;
            SceneImageLayout depth_image_layout;
            std::uint64_t depth_generation = 0;
            glm::ivec2 size{0, 0};       // valid/logical extent (compose/readback)
            glm::ivec2 alloc_size{0, 0}; // allocated image extent (may exceed size)
            bool flip_y = false;
            SceneTimelineHandle completion_semaphore;
            std::uint64_t completion_value = 0;
            std::uint64_t lod_page_generation = 0;
            // True while page decodes/uploads are still in flight.
            bool lod_streaming_active = false;
            rendering::ViewerBackend viewer_backend = rendering::ViewerBackend::Vulkan;
        };
        enum class SelectionMaskShape : std::uint32_t {
            Brush = 0,
            Rectangle = 1,
            Polygon = 2,
            Ring = 3,
        };
        struct SelectionMaskRequest {
            lfs::rendering::FrameView frame_view;
            lfs::rendering::GaussianSceneState scene;
            SelectionMaskShape shape = SelectionMaskShape::Brush;
            std::vector<glm::vec4> primitives;
            std::vector<glm::vec2> polygon_vertices;
            bool gut = false;
            bool equirectangular = false;
            bool mip_filter = false;
            float ring_width = 0.01f;
            std::uint32_t* picked_ring_id_out = nullptr;
        };
        struct DepthSampleRequest {
            glm::ivec2 pixel{0, 0};
            // Coordinate space of `pixel`. When positive, the renderer maps the
            // sample into the actual output image size for the selected slot.
            glm::ivec2 source_size{0, 0};
            RenderTargetId target{};
        };
        enum class ReadbackTicketStatus : std::uint8_t {
            NotReady = 0,
            Ready = 1,
            Failed = 2,
        };
        struct GpuLodSelectionStatus {
            bool active = false;
            std::size_t selected = 0;
            std::size_t capacity = 0;
            std::size_t overflow = 0;
            float pixel_scale_feedback = 1.0f;
            std::size_t resident_chunks = 0;
            std::size_t chunk_count = 0;
            std::size_t touched_chunks = 0;
            std::size_t miss_chunks = 0;
            std::size_t deferred_requests = 0;
            bool admission_frozen = false;
            std::size_t pool_pages = 0;
            std::size_t streaming_jobs = 0;
        };
        enum class OutputImageFormat { RgbFloat,
                                       RgbaFloat,
                                       Rgb8,
                                       Rgba8 };
        struct ReadbackRequest {
            enum class Kind { Color,
                              Depth };
            RenderTargetId target;
            core::Tensor& destination;
            glm::ivec2 offset{0, 0};
            Kind kind = Kind::Color;
        };
        struct ReadbackStats {
            size_t outstanding = 0;
            uint64_t ring_full_waits = 0;
            uint64_t cell_pin_waits = 0;
        };
        struct LodSettings {
            size_t page_pool_splats = 0;
            float pool_vram_fraction = .6f;
            uint32_t fade_frames = 8;
        };
        virtual ~SceneRenderer() = default;
        virtual std::expected<void, std::string> prepareDevice() { return {}; }
        virtual std::expected<RenderResult, std::string> render(const core::SplatData&,
                                                                const rendering::ViewportRenderRequest&, bool force_input_upload, RenderTargetId,
                                                                bool synchronize_input_upload = false, bool deterministic_export = false) = 0;
        virtual std::expected<RenderResult, std::string> rerenderSelectionOverlay(
            const core::SplatData& model, const rendering::ViewportRenderRequest& request,
            RenderTargetId target, bool synchronize_input_read = false) {
            return render(model, request, false, target, synchronize_input_read);
        }
        virtual bool nextOutputImagesNeedResize(glm::ivec2, RenderTargetId) const = 0;
        virtual std::expected<std::shared_ptr<core::Tensor>, std::string> readColorImage(RenderTargetId, OutputImageFormat) const = 0;
        std::expected<std::shared_ptr<core::Tensor>, std::string> readOutputImage(RenderTargetId target) const { return readColorImage(target, OutputImageFormat::RgbFloat); }
        std::expected<std::shared_ptr<core::Tensor>, std::string> readOutputImageRgba(RenderTargetId target) const { return readColorImage(target, OutputImageFormat::RgbaFloat); }
        std::expected<std::shared_ptr<core::Tensor>, std::string> readOutputImageRgb8(RenderTargetId target) const { return readColorImage(target, OutputImageFormat::Rgb8); }
        std::expected<std::shared_ptr<core::Tensor>, std::string> readOutputImageRgba8(RenderTargetId target) const { return readColorImage(target, OutputImageFormat::Rgba8); }
        virtual std::expected<std::shared_ptr<core::Tensor>, std::string> readPreviewDepth(RenderTargetId) const = 0;
        virtual void setDepthCaptureMode(bool, bool expected = false) = 0;
        virtual std::expected<void, std::string> readOutputImageIntoCpuHwc(RenderTargetId, core::Tensor&, int, int) const = 0;
        virtual std::expected<float, std::string> sampleDepthAtPixel(const DepthSampleRequest&) const = 0;
        virtual std::expected<uint64_t, std::string> submitReadbackTicket(const ReadbackRequest&) const = 0;
        std::expected<uint64_t, std::string> submitReadOutputImageIntoCpuHwcTicket(RenderTargetId target, core::Tensor& destination, int x, int y) const {
            return submitReadbackTicket({target, destination, {x, y}, ReadbackRequest::Kind::Color});
        }
        std::expected<uint64_t, std::string> submitReadOutputDepthImageTicket(RenderTargetId target, core::Tensor& destination) const {
            return submitReadbackTicket({target, destination, {}, ReadbackRequest::Kind::Depth});
        }
        virtual std::expected<ReadbackTicketStatus, std::string> pollReadbackTicket(uint64_t) const = 0;
        virtual std::expected<void, std::string> waitReadbackTicket(uint64_t) const = 0;
        virtual void abandonReadbackTicket(uint64_t) const = 0;
        virtual ReadbackStats readbackStats() const = 0;
        virtual std::expected<core::Tensor, std::string> buildSelectionMask(const core::SplatData&, const SelectionMaskRequest&, bool) = 0;
        // Device-resident output for the tensor compositor: UInt8 [H,W,4] color
        // and Float32 [H,W] depth, ordered after the render on the GPU.
        struct OutputTensors {
            std::shared_ptr<core::Tensor> color;
            std::shared_ptr<core::Tensor> depth;
        };
        virtual lfs::Result<OutputTensors> readOutputTensors(RenderTargetId) const;
        virtual bool hasRenderTarget(RenderTargetId) const = 0;
        virtual bool releaseRenderTarget(RenderTargetId) = 0;
        virtual void releaseSceneResources() = 0;
        virtual void reset() = 0;
        virtual void configureLod(const LodSettings&) = 0;
        virtual GpuLodSelectionStatus gpuLodSelectionStatus(RenderTargetId) const = 0;
        virtual std::optional<LodPageCache::Snapshot> ensureLodPageCacheSnapshot(const core::SplatData&) { return std::nullopt; }
        virtual void setCameraNavigating(bool) {}
        virtual bool takeRefinementRequest() { return false; }
        // Optional capability; its trainer protocol is defined separately.
        virtual SceneTrainingInterop* trainingInterop() { return nullptr; }
    };
    class LFS_VIS_API PointSceneRenderer {
    public:
        struct RenderResult {
            SceneImageHandle image;
            SceneImageViewHandle image_view;
            SceneImageLayout image_layout;
            std::uint64_t generation = 0;
            SceneImageHandle depth_image;
            SceneImageViewHandle depth_image_view;
            SceneImageLayout depth_image_layout;
            std::uint64_t depth_generation = 0;
            glm::ivec2 size{0, 0};
            bool flip_y = false;
            SceneTimelineHandle completion_semaphore;
            std::uint64_t completion_value = 0;
            rendering::ViewerBackend viewer_backend = rendering::ViewerBackend::Vulkan;
        };

        struct CropBox {
            glm::mat4 to_local{1.0f};
            glm::vec3 min{0.0f};
            glm::vec3 max{0.0f};
            bool inverse = false;
            bool desaturate = false;
        };

        struct CropEllipsoid {
            glm::mat4 to_local{1.0f};
            glm::vec3 radii{1.0f};
            bool inverse = false;
            bool desaturate = false;
        };

        struct RenderRequest {
            // Positions/colors are float [N, 3] tensors in resident GPU or CPU
            // storage. Uploads cache ownership and caller-provided content
            // revisions for in-place tensor mutations.
            const lfs::core::Tensor* positions = nullptr;
            const lfs::core::Tensor* colors = nullptr;
            std::uint64_t positions_revision = 0;
            std::uint64_t colors_revision = 0;

            // Optional model_transforms[K, 16] + per-point transform_indices[N];
            // empty/null disables the transform path.
            const std::vector<glm::mat4>* model_transforms = nullptr;
            const lfs::core::Tensor* transform_indices = nullptr;

            // Optional per-transform visibility (size matches model_transforms).
            const std::vector<bool>* node_visibility_mask = nullptr;

            // Optional per-point soft-delete mask. Nonzero entries are hidden.
            const lfs::core::Tensor* deleted_mask = nullptr;
            std::uint64_t deleted_mask_revision = 0;

            // Optional selection overlays. Masks are per-point UInt8/Bool tensors
            // where 0 means unselected and nonzero means selection group/preview.
            const lfs::core::Tensor* selection_mask = nullptr;
            const lfs::core::Tensor* preview_selection_mask = nullptr;
            const std::array<glm::vec4, lfs::rendering::kSelectionColorTableCount>* selection_colors = nullptr;
            bool preview_selection_additive = true;
            std::uint64_t selection_revision = 0;
            std::uint64_t preview_selection_revision = 0;

            // Optional crop. When set, points outside the active crop volume are
            // dropped (default) or rendered desaturated.
            std::optional<CropBox> crop;
            std::optional<CropEllipsoid> crop_ellipsoid;

            glm::mat4 view{1.0f};
            glm::mat4 view_projection{1.0f};
            glm::ivec2 size{0, 0};
            glm::vec3 background_color{0.0f};
            bool transparent_background = false;
            // Import readiness requires a completed output; interactive frames
            // expose their dependency and leave the wait on the GPU.
            bool synchronize_output = false;
            bool orthographic = false;
            float ortho_scale = 1.0f;
            float focal_y = 1.0f;
            float voxel_size = 0.01f;
            float scaling_modifier = 1.0f;
            bool depth_view = false;
            float depth_view_min = lfs::rendering::DEFAULT_DEPTH_VIEW_MIN;
            float depth_view_max = lfs::rendering::DEFAULT_DEPTH_VIEW_MAX;
            lfs::rendering::DepthVisualizationMode depth_visualization_mode =
                lfs::rendering::DepthVisualizationMode::Palette;
        };

        virtual ~PointSceneRenderer() = default;
        virtual bool takeRefinementRequest() { return false; }
        virtual std::expected<RenderResult, std::string> render(const RenderRequest&, RenderTargetId) = 0;
        virtual std::expected<std::shared_ptr<core::Tensor>, std::string> readOutputImage(RenderTargetId) = 0;
        virtual lfs::Result<SceneRenderer::OutputTensors> readOutputTensors(RenderTargetId) const;
        virtual std::expected<float, std::string> sampleDepthAtPixel(const SceneRenderer::DepthSampleRequest&) {
            return std::unexpected("Point renderer does not support depth sampling");
        }
        virtual bool hasRenderTarget(RenderTargetId) const = 0;
        virtual bool releaseRenderTarget(RenderTargetId) = 0;
        virtual void reset() = 0;
    };
} // namespace lfs::vis
