/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/splat_data.hpp"
#include "lod_page_cache.hpp"
#include "render_target_id.hpp"
#include "rendering/rendering.hpp"
#include "scene_output.hpp"
#include <chrono>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::vis {
    class VulkanContext;
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
        virtual ~SceneRenderer() = default;
        virtual std::expected<void, std::string> prepareDevice(VulkanContext&) { return {}; }
        virtual std::expected<RenderResult, std::string> render(VulkanContext&, const core::SplatData&,
                                                                const rendering::ViewportRenderRequest&, bool force_input_upload, RenderTargetId,
                                                                bool synchronize_input_upload = false, bool deterministic_export = false) = 0;
        virtual std::expected<RenderResult, std::string> rerenderSelectionOverlay(VulkanContext& c,
                                                                                  const core::SplatData& m, const rendering::ViewportRenderRequest& r, RenderTargetId t,
                                                                                  bool synchronize_input_read = false) { return render(c, m, r, false, t, synchronize_input_read); }
        virtual bool nextOutputImagesNeedResize(glm::ivec2, RenderTargetId) const = 0;
        virtual std::expected<std::shared_ptr<core::Tensor>, std::string> readOutputImage(VulkanContext&, RenderTargetId) const = 0;
        virtual std::expected<std::shared_ptr<core::Tensor>, std::string> readOutputImageRgba(VulkanContext&, RenderTargetId) const = 0;
        virtual std::expected<std::shared_ptr<core::Tensor>, std::string> readOutputImageRgb8(VulkanContext&, RenderTargetId) const = 0;
        virtual std::expected<std::shared_ptr<core::Tensor>, std::string> readOutputImageRgba8(VulkanContext&, RenderTargetId) const = 0;
        virtual std::expected<std::shared_ptr<core::Tensor>, std::string> readPreviewDepth(VulkanContext&, RenderTargetId) const = 0;
        virtual void setDepthCaptureMode(bool, bool expected = false) = 0;
        virtual std::expected<void, std::string> readOutputImageIntoCpuHwc(VulkanContext&, RenderTargetId, core::Tensor&, int, int) const = 0;
        virtual std::expected<float, std::string> sampleDepthAtPixel(VulkanContext&, const DepthSampleRequest&) const = 0;
        virtual std::expected<uint64_t, std::string> submitReadOutputImageIntoCpuHwcTicket(VulkanContext&, RenderTargetId, core::Tensor&, int, int) const = 0;
        virtual std::expected<uint64_t, std::string> submitReadOutputDepthImageTicket(VulkanContext&, RenderTargetId, core::Tensor&) const = 0;
        virtual std::expected<ReadbackTicketStatus, std::string> pollReadbackTicket(uint64_t) const = 0;
        virtual std::expected<void, std::string> waitReadbackTicket(uint64_t) const = 0;
        virtual void abandonReadbackTicket(uint64_t) const = 0;
        virtual size_t outstandingReadbackTickets() const = 0;
        virtual uint64_t readbackRingFullWaitCount() const { return 0; }
        virtual uint64_t readbackCellPinWaitCount() const { return 0; }
        virtual std::expected<core::Tensor, std::string> buildSelectionMask(VulkanContext&, const core::SplatData&, const SelectionMaskRequest&, bool) = 0;
        virtual bool hasRenderTarget(RenderTargetId) const = 0;
        virtual bool releaseRenderTarget(RenderTargetId) = 0;
        virtual void releaseSceneResources() = 0;
        virtual void reset() = 0;
        virtual void setLodPagePoolBudget(size_t) = 0;
        virtual void setLodPoolVramFraction(float) = 0;
        virtual void setLodFadeFrames(uint32_t) = 0;
        virtual GpuLodSelectionStatus gpuLodSelectionStatus(RenderTargetId) const = 0;
        virtual std::optional<LodPageCache::Snapshot> ensureLodPageCacheSnapshot(const core::SplatData&) { return std::nullopt; }
        // CUDA/Vulkan shared scratch is optional; native readers order their own
        // tensor access and do not participate in that arena's handoff protocol.
        virtual bool hasLiveTrainerReleaseFence() const { return false; }
        virtual void* renderCompleteTimeline() const { return nullptr; }
        virtual uint64_t renderCompleteValue() const { return 0; }
        virtual std::expected<void, std::string> ensureHandshakeReady(VulkanContext&) { return {}; }
        virtual std::expected<void, std::string> ensureTrainingSharedScratchReady(VulkanContext&, size_t, glm::ivec2) { return {}; }
        virtual void releaseScratchOnIdle(bool, bool = false) {}
        virtual void requestArenaHandoff() {}
        virtual void cancelArenaHandoff() {}
        virtual bool pollArenaHandoff() { return true; }
        virtual bool waitForArenaHandoff(std::chrono::milliseconds) { return true; }
        virtual void setCameraNavigating(bool) {}
        virtual bool takeRefinementRequest() { return false; }
        virtual void setLiveSubmitCallback(std::function<void(uint64_t)>) {}
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
        virtual std::expected<RenderResult, std::string> render(VulkanContext&, const RenderRequest&, RenderTargetId) = 0;
        virtual std::expected<std::shared_ptr<core::Tensor>, std::string> readOutputImage(VulkanContext&, RenderTargetId) = 0;
        virtual bool hasRenderTarget(RenderTargetId) const = 0;
        virtual bool releaseRenderTarget(RenderTargetId) = 0;
        virtual void reset() = 0;
    };
    LFS_VIS_API std::unique_ptr<SceneRenderer> createSceneRenderer();
    LFS_VIS_API std::unique_ptr<PointSceneRenderer> createPointSceneRenderer();
    LFS_VIS_API void preloadSceneRenderer();
} // namespace lfs::vis
