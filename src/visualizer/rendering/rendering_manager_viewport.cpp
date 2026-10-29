/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/services.hpp"
#include "core/splat_data_transform.hpp"
#include "mesh_offscreen_renderer.hpp"
#include "model_renderability.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "scene/scene_render_state.hpp"
#include "scene_renderer_factory.hpp"
#include "split_view_service.hpp"
#if LFS_BUILD_TRAINER
#include "training/trainer.hpp"
#endif
#include "core/training_manager.hpp"
#include "scene_custom_effects.hpp"
#include "scene_renderer.hpp"
#include "scene_training_interop.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "window/graphics_context.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <new>
#include <shared_mutex>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis {

    namespace {
        Error previewImageFailure(ErrorCode code, std::string detail) {
            return make_error({.code = code, .domain = ErrorDomain::Rendering, .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        std::shared_ptr<core::Tensor> legacyPreviewImage(Result<std::shared_ptr<core::Tensor>> result) {
            return result ? std::move(*result) : nullptr;
        }
        constexpr std::size_t kPreviewPixelStateBytesPerPixel = 4u * sizeof(float);
        // Large exports render in bands of at most this much per-pixel state. Rendering is cheap
        // next to encoding, so small bands keep export VRAM flat at any output size.
        constexpr std::size_t kMaxNativePreviewPixelStateBytes = std::size_t{256} << 20;
        constexpr float kMaxValidDepth = 1e9f;
        constexpr int kMinPreviewSubdivisionHeight = 512;
        // Capture strips preserve the shared 32-pixel macro-tile boundaries.
        constexpr int kPreviewTileHeightAlignment = 32;
        static_assert(kMinPreviewSubdivisionHeight % kPreviewTileHeightAlignment == 0);

        [[nodiscard]] bool isTileInstanceOverflow(const std::string_view error) {
            return error.find("tile-instance prefix sum overflowed signed 32-bit capacity") !=
                   std::string_view::npos;
        }
        [[nodiscard]] std::optional<std::shared_lock<std::shared_mutex>> acquireLiveModelRenderLock(
            const SceneManager* const scene_manager) {
            std::optional<std::shared_lock<std::shared_mutex>> lock;
#if LFS_BUILD_TRAINER
            if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
                if (const auto* trainer = tm->getTrainer()) {
                    lock.emplace(trainer->getRenderMutex());
                }
            }
#endif
            return lock;
        }

        [[nodiscard]] bool previewRenderNeedsTiling(const int width, const int height) {
            if (width <= 0 || height <= 0) {
                return false;
            }
            const auto pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
            return pixel_count > kMaxNativePreviewPixelStateBytes / kPreviewPixelStateBytesPerPixel;
        }

        [[nodiscard]] int previewTileHeightForWidth(const int width) {
            if (width <= 0) {
                return 1;
            }
            const std::size_t max_pixels = kMaxNativePreviewPixelStateBytes / kPreviewPixelStateBytesPerPixel;
            const int height = static_cast<int>(max_pixels / static_cast<std::size_t>(width));
            // Preserve the full image's macro/legacy tile boundaries. Changing
            // the tile grouping changes blend rounding and median depth.
            return std::max(kPreviewTileHeightAlignment,
                            height / kPreviewTileHeightAlignment * kPreviewTileHeightAlignment);
        }

        [[nodiscard]] std::optional<float> sampleDepthTensorAt(
            const lfs::core::Tensor& depth,
            const glm::ivec2& pixel) {
            if (!depth.is_valid() ||
                depth.device() != lfs::core::Device::CPU ||
                depth.dtype() != lfs::core::DataType::Float32 ||
                depth.ndim() != 2 ||
                !depth.is_contiguous()) {
                return std::nullopt;
            }
            const int width = static_cast<int>(depth.size(1));
            const int height = static_cast<int>(depth.size(0));
            if (width <= 0 || height <= 0 ||
                pixel.x < 0 || pixel.y < 0 ||
                pixel.x >= width || pixel.y >= height) {
                return std::nullopt;
            }
            const float* const values = depth.ptr<float>();
            if (!values) {
                return std::nullopt;
            }
            const float value = values[static_cast<std::size_t>(pixel.y) * static_cast<std::size_t>(width) +
                                       static_cast<std::size_t>(pixel.x)];
            if (!std::isfinite(value) || value <= 0.0f || value >= kMaxValidDepth) {
                return std::nullopt;
            }
            return value;
        }

        [[nodiscard]] lfs::rendering::CameraIntrinsics previewTileIntrinsics(
            const int full_width,
            const int full_height,
            const float focal_length_mm) {
            const auto [fx, fy] = lfs::rendering::computePixelFocalLengths(
                {full_width, full_height},
                focal_length_mm);
            return lfs::rendering::CameraIntrinsics{
                .focal_x = fx,
                .focal_y = fy,
                .center_x = static_cast<float>(full_width) * 0.5f,
                .center_y = static_cast<float>(full_height) * 0.5f,
            };
        }

        // Mesh layers carry float RGBA and depth per pixel; small bands keep them cheap.
        constexpr std::size_t kMaxExportMeshBandPixels = std::size_t{1} << 22;

        struct PixelRect {
            glm::ivec2 origin{0};
            glm::ivec2 size{0};
        };

        // Projected vertices bound every triangle unless one crosses the camera plane.
        [[nodiscard]] PixelRect meshScreenRect(const std::vector<ViewportMeshDrawItem>& items,
                                               const glm::mat4& view_projection,
                                               const glm::ivec2 image_size,
                                               const int margin) {
            glm::vec2 ndc_min(std::numeric_limits<float>::max());
            glm::vec2 ndc_max(std::numeric_limits<float>::lowest());
            for (const auto& item : items) {
                const auto vertices = item.mesh->vertices.cpu().contiguous();
                assert(vertices.ndim() == 2 && vertices.size(1) == 3);
                const float* const xyz = vertices.ptr<float>();
                const glm::mat4 mvp = view_projection * item.model;
                for (std::size_t i = 0; i < vertices.size(0); ++i) {
                    const glm::vec4 clip = mvp * glm::vec4(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], 1.0f);
                    if (clip.w <= 1e-6f) {
                        return {{0, 0}, image_size};
                    }
                    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
                    ndc_min = glm::min(ndc_min, ndc);
                    ndc_max = glm::max(ndc_max, ndc);
                }
            }
            if (ndc_min.x > ndc_max.x) {
                return {};
            }
            const auto pixel = [margin](const double position, const int extent, const bool upper) {
                const double edge = upper ? std::ceil(position * extent) + margin : std::floor(position * extent) - margin;
                return static_cast<int>(std::clamp(edge, 0.0, static_cast<double>(extent)));
            };
            const glm::ivec2 begin{pixel((ndc_min.x + 1.0) * 0.5, image_size.x, false),
                                   pixel((1.0 - ndc_max.y) * 0.5, image_size.y, false)};
            const glm::ivec2 end{pixel((ndc_max.x + 1.0) * 0.5, image_size.x, true),
                                 pixel((1.0 - ndc_min.y) * 0.5, image_size.y, true)};
            if (end.x <= begin.x || end.y <= begin.y) {
                return {};
            }
            return {begin, end - begin};
        }

        // The meshes a viewport export composites, and the pixels they cover; none when off screen.
        struct ExportMeshPass {
            ViewportMeshPassDesc params;
            glm::mat4 projection{1.0f};
            glm::mat4 view_matrix{1.0f};
            PixelRect rect;
        };
        [[nodiscard]] std::optional<ExportMeshPass> exportMeshPass(const SceneRenderState& state,
                                                                   const RenderSettings& settings,
                                                                   const RenderingManager::ExportImageRequest& request,
                                                                   const float rasterization_scale,
                                                                   const std::optional<float> ortho_scale) {
            const lfs::rendering::ViewportData view{
                .rotation = request.rotation,
                .translation = request.translation,
                .size = {request.width, request.height},
                .focal_length_mm = std::clamp(request.focal_length_mm,
                                              lfs::rendering::MIN_FOCAL_LENGTH_MM,
                                              lfs::rendering::MAX_FOCAL_LENGTH_MM),
                .orthographic = request.orthographic_override.value_or(settings.orthographic),
                .ortho_scale = ortho_scale && std::isfinite(*ortho_scale) && *ortho_scale > 0.0f ? *ortho_scale
                                                                                                 : settings.ortho_scale,
            };
            ExportMeshPass pass{
                .params = {
                    .view_projection = {},
                    .camera_position = view.translation,
                    .camera_backward = view.rotation[2],
                    .items = buildViewportMeshDrawItems(state, settings, view.rotation[2]),
                    .panels = {},
                },
            };
            if (pass.params.items.empty()) {
                return std::nullopt;
            }
            for (auto& item : pass.params.items) {
                item.wireframe_width *= rasterization_scale;
            }
            pass.projection = view.getProjectionMatrix();
            pass.view_matrix = view.getViewMatrix();
            const int line_margin = static_cast<int>(std::ceil(settings.mesh_wireframe_width * rasterization_scale)) + 1;
            pass.rect = meshScreenRect(pass.params.items, pass.projection * pass.view_matrix,
                                       {request.width, request.height}, line_margin);
            if (pass.rect.size.x <= 0 || pass.rect.size.y <= 0) {
                return std::nullopt;
            }
            return pass;
        }

        [[nodiscard]] lfs::Error meshExportError(std::string detail) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Rendering,
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        // The image an export starts from when no splat is visible: background color, or zero alpha
        // for transparent and environment exports.
        [[nodiscard]] lfs::core::Tensor emptyExportImage(const int width,
                                                         const int height,
                                                         const ExportPostProcessMode mode,
                                                         const glm::vec3& background_color) {
            if (mode != ExportPostProcessMode::Opaque) {
                return lfs::core::Tensor::zeros(
                    {static_cast<size_t>(height), static_cast<size_t>(width), size_t{4}},
                    lfs::core::Device::CPU,
                    lfs::core::DataType::UInt8);
            }
            auto image = lfs::core::Tensor::empty(
                {static_cast<size_t>(height), static_cast<size_t>(width), size_t{3}},
                lfs::core::Device::CPU,
                lfs::core::DataType::UInt8);
            if (!image.is_valid()) {
                return image;
            }
            std::vector<std::uint8_t> row(static_cast<std::size_t>(width) * 3);
            for (std::size_t x = 0; x < row.size(); ++x) {
                row[x] = static_cast<std::uint8_t>(std::clamp(background_color[x % 3], 0.0f, 1.0f) * 255.0f + 0.5f);
            }
            auto* const pixels = image.ptr<std::uint8_t>();
            for (std::size_t y = 0; y < static_cast<std::size_t>(height); ++y) {
                std::memcpy(pixels + y * row.size(), row.data(), row.size());
            }
            return image;
        }

    } // namespace

    RenderingManager::ContentBounds RenderingManager::getContentBounds(ViewId view, const glm::ivec2& viewport_size) const {
        const int viewport_width = std::max(viewport_size.x, 0);
        const int viewport_height = std::max(viewport_size.y, 0);
        ContentBounds bounds{
            0.0f,
            0.0f,
            static_cast<float>(viewport_width),
            static_cast<float>(viewport_height)};

        if (viewState(view).gt_comparison_published_actual_frame_) {
            const auto& published = *viewState(view).gt_comparison_published_actual_frame_;
            const glm::ivec4 physical_rect =
                detail::centeredGTComparisonContentRect(
                    published.framebuffer_extent,
                    published.crop.extent);
            const glm::vec4 logical_rect =
                detail::physicalToLogicalContentRect(
                    physical_rect,
                    published.framebuffer_extent,
                    {viewport_width, viewport_height});
            if (logical_rect.z > 0.0f && logical_rect.w > 0.0f) {
                bounds.x = logical_rect.x;
                bounds.y = logical_rect.y;
                bounds.width = logical_rect.z;
                bounds.height = logical_rect.w;
            }
            return bounds;
        }

        if (viewState(view).split_view_service_.isGTComparisonActive(settingsForView(viewState(view).id))) {
            glm::ivec2 content_dims{0, 0};
            if (const auto service_dims = viewState(view).split_view_service_.gtContentDimensions()) {
                content_dims = *service_dims;
            } else {
                content_dims = viewState(view).vulkan_gt_comparison_content_size_;
            }
            if (content_dims.x <= 0 || content_dims.y <= 0 ||
                viewport_width <= 0 || viewport_height <= 0) {
                return bounds;
            }

            const float content_aspect = static_cast<float>(content_dims.x) / content_dims.y;
            const float viewport_aspect = static_cast<float>(viewport_width) / viewport_height;

            if (content_aspect > viewport_aspect) {
                const int content_height =
                    std::clamp(static_cast<int>(std::lround(static_cast<float>(viewport_width) / content_aspect)),
                               1,
                               viewport_height);
                bounds.width = static_cast<float>(viewport_width);
                bounds.height = static_cast<float>(content_height);
                bounds.x = 0.0f;
                bounds.y = static_cast<float>(std::max((viewport_height - content_height) / 2, 0));
            } else {
                const int content_width =
                    std::clamp(static_cast<int>(std::lround(static_cast<float>(viewport_height) * content_aspect)),
                               1,
                               viewport_width);
                bounds.height = static_cast<float>(viewport_height);
                bounds.width = static_cast<float>(content_width);
                bounds.x = static_cast<float>(std::max((viewport_width - content_width) / 2, 0));
                bounds.y = 0.0f;
            }
        }
        return bounds;
    }

    std::optional<RenderingManager::GTSelectionContext> RenderingManager::gtComparisonSelectionContext(ViewId id) const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (id == kNoView)
            id = activeViewId();
        const auto settings = view_source_.viewSettings(id);
        if (!settings)
            return std::nullopt;
        const auto& state = viewState(id);
        if (!state.split_view_service_.isGTComparisonActive(RenderSettings(settings_, *settings))) {
            return std::nullopt;
        }
        if (!state.vulkan_gt_comparison_selection_view_.has_value()) {
            return std::nullopt;
        }
        const auto& view = *state.vulkan_gt_comparison_selection_view_;
        if (view.size.x <= 0 || view.size.y <= 0) {
            return std::nullopt;
        }
        if (view.size != state.vulkan_gt_comparison_content_size_) {
            return std::nullopt;
        }
        // Fallback contract: when any condition fails the accessor returns nullopt and every
        // downstream consumer behaves exactly as it does today.
        return GTSelectionContext{.camera = view.camera, .size = view.size};
    }

    std::optional<RenderingManager::MutableViewerPanelInfo> RenderingManager::resolveViewerPanel(
        ViewId view, Viewport& viewport,
        const glm::vec2& viewport_pos,
        const glm::vec2& viewport_size,
        const std::optional<glm::vec2> screen_point,
        const std::optional<SplitViewPanelId> panel_override) {
        const glm::ivec2 rendered_size = viewState(view).viewport_artifact_service_.renderedSize();
        const int full_render_width =
            rendered_size.x > 0 ? rendered_size.x : std::max(static_cast<int>(viewport_size.x), 1);
        const int full_render_height =
            rendered_size.y > 0 ? rendered_size.y : std::max(static_cast<int>(viewport_size.y), 1);

        MutableViewerPanelInfo info{
            .panel = SplitViewPanelId::Left,
            .viewport = &viewport,
            .x = viewport_pos.x,
            .y = viewport_pos.y,
            .width = viewport_size.x,
            .height = viewport_size.y,
            .render_width = full_render_width,
            .render_height = full_render_height,
        };

        const auto settings = settingsForView(view);
        if (splitViewUsesComparisonPanels(settings.split_view_mode)) {
            const auto bounds = getContentBounds(view, glm::ivec2(viewport_size));
            const float divider = viewport_pos.x + bounds.x + bounds.width * settings.split_position;
            info.panel = panel_override.value_or(screen_point && screen_point->x >= divider
                                                     ? SplitViewPanelId::Right
                                                     : SplitViewPanelId::Left);
        }

        return info.valid() ? std::optional<MutableViewerPanelInfo>(info) : std::nullopt;
    }

    std::optional<RenderingManager::ViewerPanelInfo> RenderingManager::resolveViewerPanel(
        ViewId view, const Viewport& viewport,
        const glm::vec2& viewport_pos,
        const glm::vec2& viewport_size,
        const std::optional<glm::vec2> screen_point,
        const std::optional<SplitViewPanelId> panel_override) const {
        const glm::ivec2 rendered_size = viewState(view).viewport_artifact_service_.renderedSize();
        const int full_render_width =
            rendered_size.x > 0 ? rendered_size.x : std::max(static_cast<int>(viewport_size.x), 1);
        const int full_render_height =
            rendered_size.y > 0 ? rendered_size.y : std::max(static_cast<int>(viewport_size.y), 1);

        ViewerPanelInfo info{
            .panel = SplitViewPanelId::Left,
            .viewport = &viewport,
            .x = viewport_pos.x,
            .y = viewport_pos.y,
            .width = viewport_size.x,
            .height = viewport_size.y,
            .render_width = full_render_width,
            .render_height = full_render_height,
        };

        const auto settings = settingsForView(view);
        if (splitViewUsesComparisonPanels(settings.split_view_mode)) {
            const auto bounds = getContentBounds(view, glm::ivec2(viewport_size));
            const float divider = viewport_pos.x + bounds.x + bounds.width * settings.split_position;
            info.panel = panel_override.value_or(screen_point && screen_point->x >= divider
                                                     ? SplitViewPanelId::Right
                                                     : SplitViewPanelId::Left);
        }

        return info.valid() ? std::optional<ViewerPanelInfo>(info) : std::nullopt;
    }

    lfs::rendering::RenderingEngine* RenderingManager::getRenderingEngine() {
        // The Vulkan path sets initialized_ without creating engine_ — it only
        // builds the auxiliary CPU engine lazily for point-cloud renders. Other
        // callers (camera frustum picking, masked depth queries) still need it,
        // so create it on demand here too.
        if (!engine_) {
            initialize();
        }
        return engine_.get();
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::getViewportImageIfAvailable() const {
        return this->state().viewport_artifact_service_.getCapturedImageIfCurrent();
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::captureViewportImage() {
        if (this->state().viewport_artifact_service_.hasLazyCapture()) {
            return this->state().viewport_artifact_service_.resolveLazyCapture();
        }

        return getViewportImageIfAvailable();
    }

    int RenderingManager::pickCameraFrustum(ViewId view, const glm::vec2& mouse_pos) {
        const int previous_hovered_camera = camera_interaction_service_.hoveredCameraId();
        bool hover_changed = false;
        auto* const engine = getRenderingEngine();
        const int hovered_camera = camera_interaction_service_.pickCameraFrustum(
            engine, viewState(view).viewport_interaction_context_.scene_manager,
            viewState(view).viewport_interaction_context_, settingsForView(view),
            mouse_pos,
            hover_changed);

        if (hover_changed) {
            LOG_DEBUG("Camera hover changed: {} -> {}", previous_hovered_camera, hovered_camera);
            markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
        }

        return hovered_camera;
    }

    RenderingManager::PreviewImageReadbackConfig RenderingManager::previewImageReadbackConfig(
        const PreviewImageReadback readback,
        const bool has_background_color_override) {
        PreviewImageReadbackConfig config{};
        switch (readback) {
        case PreviewImageReadback::LinearRgba:
        case PreviewImageReadback::LinearOpaqueRgba:
            config.dtype = core::DataType::Float32;
            config.channels = 4;
            config.capture_float_color = true;
            config.transparent_background_override = readback == PreviewImageReadback::LinearRgba;
            break;
        case PreviewImageReadback::FloatRgb:
            config.dtype = lfs::core::DataType::Float32;
            config.channels = 3;
            break;
        case PreviewImageReadback::UInt8Rgb:
            config.dtype = lfs::core::DataType::UInt8;
            config.channels = 3;
            break;
        case PreviewImageReadback::UInt8Rgba:
            config.dtype = lfs::core::DataType::UInt8;
            config.channels = 4;
            config.transparent_background_override = true;
            break;
        }
        if (has_background_color_override && !config.transparent_background_override.has_value()) {
            config.transparent_background_override = false;
        }
        return config;
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImage(SceneManager* const scene_manager,
                                                                            const glm::mat3& rotation,
                                                                            const glm::vec3& position,
                                                                            const float focal_length_mm,
                                                                            const int width,
                                                                            const int height,
                                                                            std::optional<glm::vec3> background_color_override,
                                                                            std::optional<bool> orthographic_override,
                                                                            std::optional<float> ortho_scale_override,
                                                                            const bool equirectangular,
                                                                            const bool deterministic_export,
                                                                            const std::optional<float> scene_time) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto render_state = scene_manager ? scene_manager->buildRenderState({.current_geometry = true}) : SceneRenderState{};
        const auto* const model = render_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return {};
        }
        attachCustomEffect(render_state, *scene_manager, scene_time.value_or(scene_time_));

        if (previewRenderNeedsTiling(width, height)) {
            return legacyPreviewImage(renderPreviewImageTiledWithState(
                scene_manager,
                *model,
                std::move(render_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                render_lock.has_value(),
                background_color_override,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::FloatRgb, 1.0f, equirectangular, deterministic_export));
        }

        return legacyPreviewImage(renderPreviewImageWithState(
            scene_manager,
            *model,
            std::move(render_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock.has_value(),
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            PreviewImageReadback::FloatRgb, 1.0f, equirectangular, nullptr, true, deterministic_export));
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderDatasetCameraImage(SceneManager* const scene_manager,
                                                                                  const lfs::core::Camera& camera) {
        if (!scene_manager || camera.camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR) {
            return {};
        }
        const glm::ivec2 size{std::max(camera.image_width(), camera.camera_width()),
                              std::max(camera.image_height(), camera.camera_height())};
        const auto render_camera = detail::buildGTRenderCamera(
            camera, size, detail::currentSceneTransform(scene_manager, camera.uid()));
        if (!render_camera || !render_camera->intrinsics) {
            return {};
        }

        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto render_state = scene_manager->buildRenderState();
        const auto* const model = render_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return {};
        }
        return legacyPreviewImage(renderPreviewImageWithState(
            scene_manager,
            *model,
            std::move(render_state),
            render_camera->rotation,
            render_camera->translation,
            lfs::rendering::vFovToFocalLength(glm::degrees(camera.FoVy())),
            size.x,
            size.y,
            render_lock.has_value(),
            render_camera->intrinsics,
            false,
            std::nullopt,
            std::nullopt,
            PreviewImageReadback::FloatRgb));
    }

    std::expected<void, std::string> RenderingManager::renderDepthCaptureToPreviewSlotWithState(
        const RenderSettings& settings,
        SceneManager* const scene_manager,
        const lfs::core::SplatData& source_model,
        SceneRenderState scene_state,
        const glm::mat3& rotation,
        const glm::vec3& position,
        const float focal_length_mm,
        const int width,
        const int height,
        const bool render_lock_held,
        const bool expected_depth,
        std::optional<glm::vec3> background_color_override,
        std::optional<bool> orthographic_override,
        std::optional<float> ortho_scale_override) {
        if (width <= 0 || height <= 0) {
            return std::unexpected("invalid preview depth render dimensions");
        }
        if (!last_graphics_context_) {
            return std::unexpected("no Vulkan context is available");
        }
        const auto leaf_view = lodLeafRenderView(source_model);
        const lfs::core::SplatData& model = leaf_view ? *leaf_view : source_model;
        if (scene_state.combined_model == &source_model) {
            scene_state.combined_model = &model;
        }

        if (!hasRenderableGaussians(&model)) {
            return std::unexpected("no renderable Gaussian model is available");
        }
        if (previewRenderNeedsTiling(width, height)) {
            // Depth comes from the single per-frame pixel_depth scratch; a tiled
            // render would need per-tile depth assembly, which isn't wired yet.
            LOG_WARN("render_view depth unavailable for tiled preview size {}x{}", width, height);
            return std::unexpected("preview depth render would require tiled depth assembly");
        }

        // The macro-tile (HiGS) chain only yields per-macro-tile median depth;
        // force the legacy per-pixel chain for the depth-capture render so the
        // readback matches the image resolution.
        if (!scene_renderer_) {
            scene_renderer_ = createSceneRenderer(*last_graphics_context_);
        }
        scene_renderer_->setDepthCaptureMode(true, expected_depth);
        struct DepthCaptureModeGuard {
            SceneRenderer* renderer;
            ~DepthCaptureModeGuard() { renderer->setDepthCaptureMode(false); }
        } depth_capture_guard{scene_renderer_.get()};

        auto rendered = renderPreviewImageToPreviewSlotWithState(
            settings,
            scene_manager,
            model,
            scene_state,
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock_held,
            std::nullopt,
            {},
            {},
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            std::nullopt);
        return rendered;
    }

    RenderingManager::PreviewRgbd RenderingManager::renderPreviewImageAndDepth(
        SceneManager* const scene_manager,
        const glm::mat3& rotation,
        const glm::vec3& position,
        const float focal_length_mm,
        const int width,
        const int height,
        const bool expected_depth,
        std::optional<glm::vec3> background_color_override) {
        PreviewRgbd result{};
        if (width <= 0 || height <= 0) {
            return result;
        }
        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto render_state = scene_manager ? scene_manager->buildRenderState({.current_geometry = true}) : SceneRenderState{};
        const auto* const model = render_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return result;
        }
        attachCustomEffect(render_state, *scene_manager, scene_time_);

        auto rendered = renderDepthCaptureToPreviewSlotWithState(
            getSettings(),
            scene_manager,
            *model,
            std::move(render_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock.has_value(),
            expected_depth,
            background_color_override,
            std::nullopt,
            std::nullopt);
        if (!rendered) {
            LOG_ERROR("Gaussian preview rgbd render failed: {}", rendered.error());
            return result;
        }

        // image and depth are read from the same render: the Preview output slot
        // and the pixel_depth scratch it just wrote (still resident — the Preview
        // path uses private scratch, which render() does not release).
        auto image = scene_renderer_->readOutputImage(preview_render_target_);
        if (!image) {
            LOG_ERROR("Gaussian preview rgbd image readback failed: {}", image.error());
            return result;
        }
        auto depth = scene_renderer_->readPreviewDepth(preview_render_target_);
        if (!depth) {
            LOG_ERROR("Gaussian preview depth readback failed: {}", depth.error());
            return result;
        }
        result.image = std::move(*image);
        result.depth = std::move(*depth);
        return result;
    }

    float RenderingManager::exportRasterizationScale(const int target_height, const int reference_height) const {
        if (reference_height <= 0) {
            return 1.0f;
        }
        // Use the actual viewport render resolution, including render scale.
        // The caller's reference height covers exports before a frame is ready.
        const int source_height = this->state().vulkan_viewport_image_size_.y > 0
                                      ? this->state().vulkan_viewport_image_size_.y
                                      : reference_height;
        return static_cast<float>(target_height) / source_height;
    }

    std::optional<float> RenderingManager::exportOrthoScale(const std::optional<float> scale,
                                                            const int target_height,
                                                            const int reference_height) const {
        if (!scale || reference_height <= 0) {
            return scale;
        }
        // Orthographic intrinsics are pixels per world unit at the actual
        // viewport render resolution. Scale once to avoid double rounding.
        const int source_height = this->state().vulkan_viewport_image_size_.y > 0
                                      ? this->state().vulkan_viewport_image_size_.y
                                      : reference_height;
        return static_cast<float>(static_cast<double>(*scale) * target_height / source_height);
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImageRgb8(SceneManager* const scene_manager,
                                                                                const glm::mat3& rotation,
                                                                                const glm::vec3& position,
                                                                                const float focal_length_mm,
                                                                                const int width,
                                                                                const int height,
                                                                                std::optional<glm::vec3> background_color_override,
                                                                                std::optional<bool> orthographic_override,
                                                                                std::optional<float> ortho_scale_override,
                                                                                const int reference_height,
                                                                                const bool equirectangular) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        const float rasterization_scale = exportRasterizationScale(height, reference_height);
        ortho_scale_override = exportOrthoScale(ortho_scale_override, height, reference_height);
        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto render_state = scene_manager ? scene_manager->buildRenderState({.current_geometry = true}) : SceneRenderState{};
        const auto* const model = render_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return {};
        }
        attachCustomEffect(render_state, *scene_manager, scene_time_);

        if (previewRenderNeedsTiling(width, height)) {
            return legacyPreviewImage(renderPreviewImageTiledWithState(
                scene_manager,
                *model,
                std::move(render_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                render_lock.has_value(),
                background_color_override,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::UInt8Rgb,
                rasterization_scale, equirectangular));
        }

        return legacyPreviewImage(renderPreviewImageWithState(
            scene_manager,
            *model,
            std::move(render_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock.has_value(),
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            PreviewImageReadback::UInt8Rgb,
            rasterization_scale, equirectangular));
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderLightFieldViewRgb8(
        SceneManager* scene_manager, const glm::mat3& rotation, const glm::vec3& position,
        const int width, const int height, const float fov_degrees, const float principal_shift_x,
        const float near_distance, const float far_distance) {
        if (width <= 0 || height <= 0 || width > 2048 || height > 2048 ||
            !std::isfinite(fov_degrees) || fov_degrees < 1.0f || fov_degrees > 120.0f ||
            !std::isfinite(principal_shift_x) || !std::isfinite(near_distance) ||
            !std::isfinite(far_distance) || near_distance < 0.0f || far_distance <= near_distance) {
            return {};
        }
        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto state = scene_manager ? scene_manager->buildRenderState({.current_geometry = true}) : SceneRenderState{};
        const auto* model = state.combined_model;
        if (!hasRenderableGaussians(model))
            return {};
        attachCustomEffect(state, *scene_manager, scene_time_);

        auto settings = getSettings();
        settings.scaling_modifier = 1.0f;
        settings.depth_clip_enabled = false;
        settings.orthographic = false;
        settings.depth_view = false;
        settings.point_cloud_mode = false;
        settings.show_rings = false;
        settings.show_center_markers = false;
        settings.selection_outline = false;
        settings.desaturate_unselected = false;
        settings.depth_filter_enabled = true;
        settings.depth_filter_viz_mode = 2; // Hide splats outside the near/far slab.
        settings.depth_filter_transform = lfs::geometry::EuclideanTransform(glm::quat_cast(rotation), position);
        settings.depth_filter_min = glm::vec3(-1.0e9f, -1.0e9f, -far_distance);
        settings.depth_filter_max = glm::vec3(1.0e9f, 1.0e9f, -near_distance);
        const float focal = height / (2.0f * std::tan(glm::radians(fov_degrees) * 0.5f));
        const lfs::rendering::CameraIntrinsics intrinsics{
            .focal_x = focal,
            .focal_y = focal,
            .center_x = width * 0.5f + principal_shift_x,
            .center_y = height * 0.5f};
        auto image = renderPreviewImageWithState(
            scene_manager, *model, std::move(state), rotation, position,
            lfs::rendering::vFovToFocalLength(fov_degrees), width, height,
            render_lock.has_value(), intrinsics, false, std::nullopt, std::nullopt,
            PreviewImageReadback::UInt8Rgb, 1.0f, false, &settings, false);
        releasePreviewImageResources();
        return legacyPreviewImage(std::move(image));
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImageRgba8(SceneManager* const scene_manager,
                                                                                 const glm::mat3& rotation,
                                                                                 const glm::vec3& position,
                                                                                 const float focal_length_mm,
                                                                                 const int width,
                                                                                 const int height,
                                                                                 std::optional<bool> orthographic_override,
                                                                                 std::optional<float> ortho_scale_override,
                                                                                 const int reference_height,
                                                                                 const bool equirectangular) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        const float rasterization_scale = exportRasterizationScale(height, reference_height);
        ortho_scale_override = exportOrthoScale(ortho_scale_override, height, reference_height);
        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto render_state = scene_manager ? scene_manager->buildRenderState({.current_geometry = true}) : SceneRenderState{};
        const auto* const model = render_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return {};
        }
        attachCustomEffect(render_state, *scene_manager, scene_time_);

        if (previewRenderNeedsTiling(width, height)) {
            return legacyPreviewImage(renderPreviewImageTiledWithState(
                scene_manager,
                *model,
                std::move(render_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                render_lock.has_value(),
                std::nullopt,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::UInt8Rgba,
                rasterization_scale, equirectangular));
        }

        return legacyPreviewImage(renderPreviewImageWithState(
            scene_manager,
            *model,
            std::move(render_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock.has_value(),
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            std::nullopt,
            PreviewImageReadback::UInt8Rgba,
            rasterization_scale, equirectangular));
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImage(const lfs::core::SplatData& model,
                                                                            SceneRenderState scene_state,
                                                                            const glm::mat3& rotation,
                                                                            const glm::vec3& position,
                                                                            const float focal_length_mm,
                                                                            const int width,
                                                                            const int height,
                                                                            std::optional<glm::vec3> background_color_override,
                                                                            std::optional<bool> orthographic_override,
                                                                            std::optional<float> ortho_scale_override,
                                                                            const bool equirectangular,
                                                                            const bool deterministic_export) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        if (previewRenderNeedsTiling(width, height)) {
            return legacyPreviewImage(renderPreviewImageTiledWithState(
                nullptr,
                model,
                std::move(scene_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                false,
                background_color_override,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::FloatRgb, 1.0f, equirectangular, deterministic_export));
        }

        return legacyPreviewImage(renderPreviewImageWithState(
            nullptr,
            model,
            std::move(scene_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            false,
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            PreviewImageReadback::FloatRgb, 1.0f, equirectangular, nullptr, true, deterministic_export));
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImageRgb8(const lfs::core::SplatData& model,
                                                                                SceneRenderState scene_state,
                                                                                const glm::mat3& rotation,
                                                                                const glm::vec3& position,
                                                                                const float focal_length_mm,
                                                                                const int width,
                                                                                const int height,
                                                                                std::optional<glm::vec3> background_color_override,
                                                                                std::optional<bool> orthographic_override,
                                                                                std::optional<float> ortho_scale_override,
                                                                                const bool equirectangular) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        if (previewRenderNeedsTiling(width, height)) {
            return legacyPreviewImage(renderPreviewImageTiledWithState(
                nullptr,
                model,
                std::move(scene_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                false,
                background_color_override,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::UInt8Rgb, 1.0f, equirectangular));
        }

        return legacyPreviewImage(renderPreviewImageWithState(
            nullptr,
            model,
            std::move(scene_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            false,
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            PreviewImageReadback::UInt8Rgb, 1.0f, equirectangular));
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImageRgba8(const lfs::core::SplatData& model,
                                                                                 SceneRenderState scene_state,
                                                                                 const glm::mat3& rotation,
                                                                                 const glm::vec3& position,
                                                                                 const float focal_length_mm,
                                                                                 const int width,
                                                                                 const int height,
                                                                                 std::optional<bool> orthographic_override,
                                                                                 std::optional<float> ortho_scale_override,
                                                                                 const bool equirectangular) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        if (previewRenderNeedsTiling(width, height)) {
            return legacyPreviewImage(renderPreviewImageTiledWithState(
                nullptr,
                model,
                std::move(scene_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                false,
                std::nullopt,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::UInt8Rgba, 1.0f, equirectangular));
        }

        return legacyPreviewImage(renderPreviewImageWithState(
            nullptr,
            model,
            std::move(scene_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            false,
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            std::nullopt,
            PreviewImageReadback::UInt8Rgba, 1.0f, equirectangular));
    }

    void RenderingManager::releasePreviewImageResources() {
        if (scene_renderer_) {
            if (scene_renderer_->hasRenderTarget(preview_render_target_) &&
                scene_renderer_->releaseRenderTarget(preview_render_target_)) {
                render_targets_.release(preview_render_target_);
                preview_render_target_ = render_targets_.allocate();
            }
        }
        const std::lock_guard lock(lod_leaf_view_mutex_);
        lod_leaf_view_.reset();
        lod_leaf_view_source_ = nullptr;
        lod_leaf_view_tree_ = nullptr;
    }

    void RenderingManager::releaseLodLeafRenderViewUnlessFor(const lfs::core::SplatData* const model) {
        const std::lock_guard lock(lod_leaf_view_mutex_);
        if (lod_leaf_view_ && lod_leaf_view_source_ != model) {
            lod_leaf_view_.reset();
            lod_leaf_view_source_ = nullptr;
            lod_leaf_view_tree_ = nullptr;
        }
    }

    Result<bool> RenderingManager::prepareShaderExport(const ExportLease& lease, const core::SplatData* model, const RenderSettings* settings) {
        if (lease.error)
            return *lease.error;
        if (lease.ended())
            return make_error(ErrorInit{.code = ErrorCode::Cancelled, .domain = ErrorDomain::Rendering, .user_message = "The export was cancelled.", .detection = LFS_SOURCE_SITE_CURRENT()});
        if (!lease.request)
            return true;
        if (lease.per_frame && !settings) {
            auto* scene = services().sceneOrNull();
            if (!scene)
                return shadersNotReady();
            auto lock = acquireLiveModelRenderLock(scene);
            const auto state = scene->buildRenderState({.current_geometry = true});
            const auto current = getSettings();
            return prepareShaderExport(lease, state.combined_model, &current);
        }
        const auto failure = [](std::string message) { return make_error(ErrorInit{.code = ErrorCode::Unavailable, .domain = ErrorDomain::Rendering, .user_message = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()}); };
        if (!last_graphics_context_)
            return failure("No graphics context is available for shader export.");
        if (!scene_renderer_)
            scene_renderer_ = createSceneRenderer(*last_graphics_context_);
        auto* const effects = customEffects();
        if (!effects)
            return failure("User shaders need the Vulkan viewport renderer.");
        if (auto initialized = effects->initializePrograms(); !initialized)
            return std::move(initialized).error();
        std::vector<std::string> hosts;
        if (settings && (settings->point_cloud_mode || !model || model->size() == 0))
            return true;
        const auto& frame = lease.frame;
        const bool framed = frame.width > 0 && frame.height > 0;
        if (model && settings && framed) {
            const auto leaf_view = lodLeafRenderView(*model);
            if (leaf_view)
                model = leaf_view.get();
            rendering::ViewportRenderRequest request;
            request.gut = rendering::isGutBackend(rendering::normalizeViewerRasterBackend(settings->raster_backend, settings->gut));
            request.sh_degree = settings->sh_degree;
            request.frame_view.size = {1, 1};
            request.deterministic_export = true; // exports and scripted previews keep stable ties
            if (auto prepared = effects->plan(*model, request, preview_render_target_); !prepared)
                return std::move(prepared).error();
            const auto plan = [&](const bool banded, const bool capture) {
                request.frame_view.subregion_full_size = banded ? glm::ivec2(1) : glm::ivec2(0);
                const auto host = effects->planHost(request, preview_render_target_, false, capture, false);
                if (!host.empty() && !std::ranges::contains(hosts, host))
                    hosts.emplace_back(host);
            };
            // The size alone decides banding; a depth capture is never tiled.
            plan(!frame.depth && previewRenderNeedsTiling(frame.width, frame.height), frame.depth);
            if (frame.bands)
                plan(true, false);
        }
        if (!lease.missing.empty()) {
            for (auto& host : std::exchange(lease.missing, {}))
                if (!std::ranges::contains(hosts, host))
                    hosts.push_back(std::move(host));
            lease.snapshot.reset(); // the draw found it incomplete
            lease.discovered = true;
        }
        if (!framed && !lease.discovered)
            return true; // nothing planned: the draw links what it finds missing
        if (!lease.snapshot || !hosts.empty()) {
            auto snapshot = shader_service_->exportSnapshot(lease, hosts);
            if (!snapshot)
                return snapshot.error();
            if (!*snapshot)
                return false;
            lease.snapshot = *snapshot;
        }
        return effects->preparePrograms(lease.snapshot->programs, lease.programs, [changes = lease.changes] { changes->notify(); });
    }

    std::shared_ptr<const lfs::core::SplatData> RenderingManager::lodLeafRenderView(
        const lfs::core::SplatData& model) {
        if (!model.lod_tree || !model.lod_tree->has_tree()) {
            return nullptr;
        }
        const std::lock_guard lock(lod_leaf_view_mutex_);
        const auto rows = static_cast<std::size_t>(model.size());
        const auto deleted_version = model.deleted_mask_version();
        if (!lod_leaf_view_ || lod_leaf_view_source_ != &model || lod_leaf_view_tree_ != model.lod_tree.get() ||
            lod_leaf_view_rows_ != rows || lod_leaf_view_deleted_version_ != deleted_version) {
            auto view = lfs::core::make_lod_leaf_view(model);
            // Keep decoded opacity in storage owned by the active graphics backend.
            if (view->opacity_raw().data_ptr() != model.opacity_raw().data_ptr()) {
                if (const auto allocator = makeSplatTensorAllocator()) {
                    auto opacity = allocator(view->opacity_raw().shape(), rows,
                                             lfs::core::DataType::Float32, "SplatData.opacity");
                    opacity.copy_from(view->opacity_raw());
                    view->opacity_raw() = std::move(opacity);
                }
            }
            lod_leaf_view_ = std::move(view);
            lod_leaf_view_source_ = &model;
            lod_leaf_view_tree_ = model.lod_tree.get();
            lod_leaf_view_rows_ = rows;
            lod_leaf_view_deleted_version_ = deleted_version;
        }
        return lod_leaf_view_;
    }

    Result<core::Tensor> RenderingManager::renderLinearImage(SceneManager* scene_manager, const ExportImageRequest& request) {
        return renderLinearImageImpl(scene_manager, nullptr, {}, request);
    }

    Result<core::Tensor> RenderingManager::renderLinearImage(const core::SplatData& model, SceneRenderState state, const ExportImageRequest& request) {
        return renderLinearImageImpl(nullptr, &model, std::move(state), request);
    }

    Result<core::Tensor> RenderingManager::renderLinearImageImpl(SceneManager* scene_manager, const core::SplatData* snapshot,
                                                                 SceneRenderState snapshot_state, const ExportImageRequest& request) {
        if (request.width <= 0 || request.height <= 0)
            return previewImageFailure(ErrorCode::InvalidArgument, "Invalid float render dimensions");
        if (request.mode != ExportPostProcessMode::Transparent && request.mode != ExportPostProcessMode::Opaque)
            return previewImageFailure(ErrorCode::Unsupported, "Raw linear render capture precedes environment export post-processing");
        struct PreviewResourceGuard {
            RenderingManager& manager;
            ~PreviewResourceGuard() { manager.releasePreviewImageResources(); }
        } guard{*this};
        try {
            auto render_lock = acquireLiveModelRenderLock(scene_manager);
            auto state = snapshot ? std::move(snapshot_state) : scene_manager ? scene_manager->buildRenderState({.current_geometry = true})
                                                                              : SceneRenderState{};
            const auto* model = snapshot ? snapshot : state.combined_model;
            if (!hasRenderableGaussians(model))
                return previewImageFailure(ErrorCode::FailedPrecondition, "No renderable Gaussian model is available");
            const auto readback = request.mode == ExportPostProcessMode::Transparent ? PreviewImageReadback::LinearRgba : PreviewImageReadback::LinearOpaqueRgba;
            const auto ortho_scale = exportOrthoScale(request.ortho_scale_override, request.height, request.reference_height);
            const float scale = exportRasterizationScale(request.height, request.reference_height);
            auto image = previewRenderNeedsTiling(request.width, request.height)
                             ? renderPreviewImageTiledWithState(scene_manager, *model, std::move(state), request.rotation, request.translation,
                                                                request.focal_length_mm, request.width, request.height, render_lock.has_value(), std::nullopt,
                                                                request.orthographic_override, ortho_scale, readback, scale, request.equirectangular)
                             : renderPreviewImageWithState(scene_manager, *model, std::move(state), request.rotation, request.translation,
                                                           request.focal_length_mm, request.width, request.height, render_lock.has_value(), std::nullopt,
                                                           request.orthographic_override, ortho_scale, std::nullopt, readback, scale, request.equirectangular);
            // The captured CPU tensor owns its samples; renderer resources can retire.
            if (!image)
                return std::move(image).error();
            return std::move(**image);
        } catch (const Exception& error) {
            return error.error();
        } catch (const std::bad_alloc&) {
            return previewImageFailure(ErrorCode::ResourceExhausted, "Float render capture allocation failed");
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): Preserve the exception detail in the typed rendering error.
            return previewImageFailure(ErrorCode::Unavailable, error.what());
        }
    }

    std::expected<lfs::core::Tensor, std::string> RenderingManager::renderExportImage(
        SceneManager* const scene_manager, const ExportImageRequest& request) {
        if (request.width <= 0 || request.height <= 0) {
            return std::unexpected("invalid export image dimensions");
        }

        const bool needs_alpha = request.mode != ExportPostProcessMode::Opaque;
        const auto rendered =
            needs_alpha
                ? renderPreviewImageRgba8(scene_manager,
                                          request.rotation,
                                          request.translation,
                                          request.focal_length_mm,
                                          request.width,
                                          request.height,
                                          request.orthographic_override,
                                          request.ortho_scale_override,
                                          request.reference_height)
                : renderPreviewImageRgb8(scene_manager,
                                         request.rotation,
                                         request.translation,
                                         request.focal_length_mm,
                                         request.width,
                                         request.height,
                                         std::nullopt,
                                         request.orthographic_override,
                                         request.ortho_scale_override,
                                         request.reference_height);
        if (last_graphics_context_ &&
            last_graphics_context_->terminalState() != RendererTerminalState::Running) {
            return std::unexpected("renderer is unavailable after a GPU failure; restart LichtFeld Studio");
        }
        releasePreviewImageResources();

        const auto settings = getSettings();
        lfs::core::Tensor image;
        if (rendered && rendered->is_valid()) {
            image = std::move(*rendered);
        } else {
            // Scene content only. Grid and camera frustums stay viewport overlays.
            // An empty result means no point cloud; error is a render failure.
            struct PointCloudExport {
                lfs::core::Tensor image;
                bool present = false;
                std::string error;
            };
            auto point_cloud = [&]() -> PointCloudExport {
                if (!scene_manager) {
                    return {};
                }
                auto render_lock = acquireLiveModelRenderLock(scene_manager);
                auto render_state = scene_manager->buildRenderState({.current_geometry = true});
                const auto* const cloud = render_state.point_cloud;
                if (!cloud || cloud->size() <= 0 || !cloud->means.is_valid() || !cloud->colors.is_valid()) {
                    return {};
                }
                if (!last_graphics_context_) {
                    return {.image = {}, .present = false, .error = "no Vulkan context is available"};
                }
                if (last_graphics_context_->terminalState() != RendererTerminalState::Running) {
                    return {.image = {},
                            .present = false,
                            .error = "renderer is unavailable after a GPU failure; restart LichtFeld Studio"};
                }

                const int width = request.width;
                const int height = request.height;
                const bool transparent = request.mode != ExportPostProcessMode::Opaque;
                RenderSettings preview_settings = settings;
                preview_settings.focal_length_mm = std::clamp(
                    request.focal_length_mm, lfs::rendering::MIN_FOCAL_LENGTH_MM, lfs::rendering::MAX_FOCAL_LENGTH_MM);
                preview_settings.split_view_mode = SplitViewMode::Disabled;
                preview_settings.equirectangular = false;
                if (request.orthographic_override) {
                    preview_settings.orthographic = *request.orthographic_override;
                }
                if (const auto ortho_scale =
                        exportOrthoScale(request.ortho_scale_override, height, request.reference_height);
                    ortho_scale && std::isfinite(*ortho_scale) && *ortho_scale > 0.0f) {
                    preview_settings.ortho_scale = *ortho_scale;
                }

                Viewport viewport(static_cast<std::size_t>(width), static_cast<std::size_t>(height));
                viewport.setViewMatrix(request.rotation, request.translation);
                std::vector<glm::mat4> transforms{render_state.point_cloud_transform};
                FrameContext frame_ctx{
                    .viewport = viewport,
                    .scene_manager = scene_manager,
                    .scene_state = std::move(render_state),
                    .settings = preview_settings,
                    .render_size = {width, height},
                    .viewport_pos = {0, 0},
                    .gaussian_selection_visible = gaussian_selection_visible_,
                    .cursor_preview = {},
                    .gizmo = {},
                    .view_panels = {},
                };
                auto pc_request = buildPointCloudRenderRequest(frame_ctx, {width, height}, transforms);
                auto point_request = buildPointSceneRequest(pc_request, preview_settings);
                point_request.positions = &cloud->means;
                point_request.colors = &cloud->colors;
                point_request.transparent_background = transparent;
                point_request.equirectangular = false;
                // Export reads the image immediately. The live viewport leaves this wait on the GPU.
                point_request.synchronize_output = true;
                if (transparent) {
                    point_request.background_color = glm::vec3(0.0f);
                }

                // One-shot renderer and target, so export does not resize the live viewport's images.
                // A missing graphics backend throws; the caller already rejected a terminal context.
                std::unique_ptr<PointSceneRenderer> renderer = createPointSceneRenderer(*last_graphics_context_);
                const RenderTargetId export_target = render_targets_.allocate();
                struct ExportTargetGuard {
                    PointSceneRenderer& renderer;
                    RenderTargetRegistry& targets;
                    RenderTargetId id;
                    ~ExportTargetGuard() {
                        renderer.releaseRenderTarget(id);
                        targets.release(id);
                    }
                } export_target_guard{*renderer, render_targets_, export_target};
                if (auto rendered_points = renderer->render(point_request, export_target); !rendered_points) {
                    return {.image = {}, .present = false, .error = rendered_points.error()};
                }
                auto readback = renderer->readOutputImage(export_target, true);
                if (!readback || !*readback || !(*readback)->is_valid()) {
                    return {.image = {},
                            .present = false,
                            .error = readback ? "export render produced no image" : readback.error()};
                }
                auto exported = transparent ? std::move(**readback) : (*readback)->slice(2, 0, 3).contiguous();
                if (!exported.is_valid() || !exported.is_contiguous()) {
                    return {.image = {}, .present = false, .error = "export render produced no image"};
                }
                return {.image = std::move(exported), .present = true, .error = {}};
            }();
            if (!point_cloud.error.empty()) {
                return std::unexpected(std::move(point_cloud.error));
            }
            if (point_cloud.present) {
                image = std::move(point_cloud.image);
            } else if (request.mode == ExportPostProcessMode::EnvironmentComposite ||
                       (scene_manager && !scene_manager->getScene().getVisibleMeshes().empty())) {
                // No Gaussians or point cloud: environment (zero alpha) or background, then meshes.
                image = emptyExportImage(request.width, request.height, request.mode, settings.background_color);
                if (!image.is_valid()) {
                    return std::unexpected("export failed to allocate the background image");
                }
            } else {
                return std::unexpected("export render produced no image");
            }
        }

        const ExportPostProcessView view{
            .rotation = request.rotation,
            .focal_length_mm = request.focal_length_mm,
            .equirectangular_view = settings.equirectangular,
            .controller_predict_size = this->state().frame_lifecycle_service_.lastViewportSize(),
        };
        auto processed = applyExportPostProcess(
            std::move(image), scene_manager, settings, getCurrentCameraId(), request.mode, view);
        if (!processed) {
            return processed;
        }
        // Meshes go on last: the viewport draws them after appearance correction and environment.
        auto meshes = compositeExportMeshes(scene_manager, request, *processed);
        releasePreviewImageResources();
        if (!meshes) {
            return std::unexpected(lfs::format_for_developer(meshes.error()));
        }
        return processed;
    }

    Result<core::Tensor> RenderingManager::renderTransparentExportFallback(
        SceneManager* const scene_manager, const ExportImageRequest& request) {
        const auto pass = [&](const float background) {
            return renderPreviewImageRgb8(scene_manager, request.rotation, request.translation,
                                          request.focal_length_mm, request.width, request.height,
                                          glm::vec3{background}, request.orthographic_override,
                                          request.ortho_scale_override, request.reference_height);
        };
        const auto black_rgb = pass(0.0f);
        const auto white_rgb = pass(1.0f);
        if (!black_rgb || !white_rgb || !black_rgb->is_valid() || !white_rgb->is_valid() ||
            black_rgb->shape() != white_rgb->shape() || black_rgb->ndim() != 3 || black_rgb->shape()[2] != 3) {
            return make_error(ErrorInit{
                .code = ErrorCode::FailedPrecondition,
                .domain = ErrorDomain::Rendering,
                .user_message = "transparent viewport export expected matching RGB images",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        auto rgba = core::Tensor::empty({black_rgb->shape()[0], black_rgb->shape()[1], std::size_t{4}},
                                        core::Device::CPU, core::DataType::UInt8);
        const auto* const black = black_rgb->ptr<uint8_t>();
        const auto* const white = white_rgb->ptr<uint8_t>();
        auto* const out = rgba.ptr<uint8_t>();
        for (std::size_t i = 0; i < black_rgb->numel() / 3; ++i) {
            const auto src = i * 3;
            const auto dst = i * 4;
            const float delta = (static_cast<float>(white[src]) - static_cast<float>(black[src]) +
                                 static_cast<float>(white[src + 1]) - static_cast<float>(black[src + 1]) +
                                 static_cast<float>(white[src + 2]) - static_cast<float>(black[src + 2])) /
                                (3.0f * 255.0f);
            const float alpha = std::clamp(1.0f - delta, 0.0f, 1.0f);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const float recovered = alpha > 1.0e-6f ? static_cast<float>(black[src + channel]) / alpha : 0.0f;
                out[dst + channel] = static_cast<uint8_t>(std::lround(std::clamp(recovered, 0.0f, 255.0f)));
            }
            out[dst + 3] = static_cast<uint8_t>(std::lround(alpha * 255.0f));
        }
        const auto settings = getSettings();
        const ExportPostProcessView view{
            .rotation = request.rotation,
            .focal_length_mm = request.focal_length_mm,
            .equirectangular_view = settings.equirectangular,
            .controller_predict_size = this->state().frame_lifecycle_service_.lastViewportSize(),
        };
        return from_legacy_expected(
            applyExportPostProcess(std::move(rgba), scene_manager, settings, getCurrentCameraId(),
                                   ExportPostProcessMode::Transparent, view),
            LegacyErrorContext{
                .code = ErrorCode::FailedPrecondition,
                .domain = ErrorDomain::Rendering,
                .operation = "Post-process transparent export",
                .source = LFS_SOURCE_SITE_CURRENT(),
            });
    }

    RenderingManager::ExportFrame RenderingManager::exportImageFrame(SceneManager* const scene_manager,
                                                                     const ExportImageRequest& request) {
        ExportFrame frame{.width = request.width, .height = request.height};
        if (!scene_manager) {
            return frame;
        }
        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        const auto render_state = scene_manager->buildRenderState({.current_geometry = true});
        const int height = request.height;
        frame.bands = hasRenderableGaussians(render_state.combined_model) &&
                      exportMeshPass(render_state, getSettings(), request,
                                     exportRasterizationScale(height, request.reference_height),
                                     exportOrthoScale(request.ortho_scale_override, height, request.reference_height));
        return frame;
    }

    lfs::Status RenderingManager::compositeExportMeshes(
        SceneManager* const scene_manager,
        const ExportImageRequest& request,
        lfs::core::Tensor& image) {
        if (!scene_manager || !last_graphics_context_) {
            return {};
        }
        const int width = request.width;
        const int height = request.height;
        assert(image.is_valid() && image.device() == lfs::core::Device::CPU &&
               image.dtype() == lfs::core::DataType::UInt8 && image.ndim() == 3 && image.is_contiguous());
        assert(static_cast<int>(image.size(0)) == height && static_cast<int>(image.size(1)) == width);
        assert(image.size(2) == 3 || image.size(2) == 4);

        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto render_state = scene_manager->buildRenderState({.current_geometry = true});
        const auto settings = getSettings();
        const float rasterization_scale = exportRasterizationScale(height, request.reference_height);
        const auto ortho_scale = exportOrthoScale(request.ortho_scale_override, height, request.reference_height);
        auto pass = exportMeshPass(render_state, settings, request, rasterization_scale, ortho_scale);
        if (!pass) {
            return {};
        }
        auto& mesh_params = pass->params;
        const glm::mat4& projection = pass->projection;
        const glm::mat4& view_matrix = pass->view_matrix;
        const auto mesh_rect = pass->rect;
        const int rows_begin = mesh_rect.origin.y;
        const int rows_end = mesh_rect.origin.y + mesh_rect.size.y;

        const auto* const model = render_state.combined_model;
        const bool has_splats = hasRenderableGaussians(model);
        const auto intrinsics = previewTileIntrinsics(width, height, request.focal_length_mm);
        const int mesh_rows_limit =
            std::max(1, static_cast<int>(kMaxExportMeshBandPixels / static_cast<std::size_t>(mesh_rect.size.x)));
        MeshOffscreenRenderer mesh_renderer;
        lfs::core::Tensor splat_depth;
        int band_height_limit = previewTileHeightForWidth(width);
        for (int band_y = rows_begin / kPreviewTileHeightAlignment * kPreviewTileHeightAlignment;
             band_y < rows_end;) {
            const int aligned_rows_left = (rows_end - band_y + kPreviewTileHeightAlignment - 1) /
                                          kPreviewTileHeightAlignment * kPreviewTileHeightAlignment;
            int band_height = std::min({band_height_limit, height - band_y, aligned_rows_left});
            if (has_splats) {
                while (true) {
                    auto rendered = renderPreviewImageToPreviewSlotWithState(
                        settings,
                        scene_manager,
                        *model,
                        render_state,
                        request.rotation,
                        request.translation,
                        request.focal_length_mm,
                        width,
                        band_height,
                        render_lock.has_value(),
                        intrinsics,
                        {0, band_y},
                        {width, height},
                        request.orthographic_override,
                        ortho_scale,
                        std::nullopt,
                        std::nullopt,
                        rasterization_scale,
                        true);
                    if (rendered) {
                        break;
                    }
                    if (!isTileInstanceOverflow(rendered.error()) || band_height <= kMinPreviewSubdivisionHeight) {
                        return lfs::Status::failure(meshExportError(std::format("mesh export depth render failed: {}", rendered.error())));
                    }
                    band_height = std::max(kMinPreviewSubdivisionHeight,
                                           (band_height / 2) / kPreviewTileHeightAlignment * kPreviewTileHeightAlignment);
                    band_height_limit = band_height;
                }
                splat_depth = lfs::core::Tensor::empty(
                    {static_cast<std::size_t>(band_height), static_cast<std::size_t>(width)},
                    lfs::core::Device::CPU,
                    lfs::core::DataType::Float32);
                auto ticket = scene_renderer_->submitReadOutputDepthImageTicket(
                    preview_render_target_, splat_depth);
                if (!ticket) {
                    return lfs::Status::failure(meshExportError(std::format("mesh export depth readback failed: {}", ticket.error())));
                }
                if (auto waited = scene_renderer_->waitReadbackTicket(*ticket); !waited) {
                    return lfs::Status::failure(meshExportError(std::format("mesh export depth readback failed: {}", waited.error())));
                }
            }
            const int mesh_end = std::min(band_y + band_height, rows_end);
            for (int mesh_y = std::max(band_y, rows_begin); mesh_y < mesh_end;) {
                const int mesh_rows = std::min(mesh_rows_limit, mesh_end - mesh_y);
                const glm::ivec2 origin{mesh_rect.origin.x, mesh_y};
                const glm::ivec2 size{mesh_rect.size.x, mesh_rows};
                mesh_params.view_projection =
                    cropProjectionToRect(projection, {width, height}, origin, size) * view_matrix;
                auto layer = mesh_renderer.render(*last_graphics_context_, mesh_params, projection, size.x, size.y);
                if (!layer) {
                    return lfs::Status::failure(meshExportError(std::format("mesh export render failed: {}", lfs::format_for_developer(layer.error()))));
                }
                compositeMeshLayer(
                    *layer,
                    has_splats ? splat_depth.ptr<float>() + static_cast<std::size_t>(mesh_y - band_y) * width + origin.x
                               : nullptr,
                    static_cast<std::size_t>(width),
                    image,
                    origin);
                mesh_y += mesh_rows;
            }
            band_y += band_height;
        }
        return {};
    }

    void RenderingManager::releaseVrResources() {
        for (auto& target : vr_render_targets_) {
            if (!target.valid())
                continue;
            if (scene_renderer_ && scene_renderer_->hasRenderTarget(target) && !scene_renderer_->releaseRenderTarget(target))
                continue;
            render_targets_.release(target);
            target = {};
        }
    }

    RenderSettings RenderingManager::vrRenderSettings(RenderSettings settings) {
        settings.split_view_mode = SplitViewMode::Disabled;
        settings.orthographic = settings.equirectangular = settings.depth_view = false;
        settings.point_cloud_mode = false;
        settings.show_grid = settings.show_coord_axes = settings.show_pivot = false;
        settings.show_camera_frustums = settings.show_rings = settings.show_center_markers = false;
        settings.desaturate_unselected = false;
        return settings;
    }

    lfs::Result<std::array<vr::EyeFrame, 2>> RenderingManager::renderVrEyes(
        GraphicsContext& context, SceneManager& scene,
        const std::array<lfs::rendering::FrameView, 2>& views) {
        const auto unavailable = [](std::string message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Unavailable,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        };
        last_graphics_context_ = &context;
        // The desktop mirror of the previous frame still samples the eye images.
        if (!context.waitForSubmittedFrames())
            return unavailable("Could not retire the desktop VR mirror.");
        auto snapshot = scene.buildRenderState({.current_geometry = true});
        if (!hasRenderableGaussians(snapshot.combined_model))
            return unavailable("VR requires a visible Gaussian model in edit mode.");
        // LoD files hold hierarchy nodes besides the renderable leaves.
        const auto leaves = lodLeafRenderView(*snapshot.combined_model);
        const auto& model = leaves ? *leaves : *snapshot.combined_model;
        if (!scene_renderer_)
            scene_renderer_ = createSceneRenderer(context);
        if (!scene_renderer_)
            return unavailable("VR needs the Vulkan viewport renderer.");
        rendererTrainingInterop(*scene_renderer_).setLiveSubmitCallback({});
        const auto settings = vrRenderSettings(getSettings());
        std::array<vr::EyeFrame, 2> eyes{};
        for (std::size_t i = 0; i < eyes.size(); ++i) {
            if (!vr_render_targets_[i].valid())
                vr_render_targets_[i] = render_targets_.allocate();
            Viewport viewport(static_cast<std::size_t>(views[i].size.x), static_cast<std::size_t>(views[i].size.y));
            viewport.setViewMatrix(views[i].rotation, views[i].translation);
            const FrameContext frame{
                .viewport = viewport,
                .scene_manager = &scene,
                .model = &model,
                .scene_state = snapshot,
                .settings = settings,
                .render_size = views[i].size,
                .viewport_pos = {0, 0},
                .gaussian_selection_visible = false,
                .cursor_preview = {},
                .gizmo = {},
                .view_panels = {},
            };
            auto request = buildViewportRenderRequest(frame, views[i].size);
            request.frame_view = views[i];
            request.frame_view.background_color = settings.background_color;
            request.overlay = {};
            request.transparent_background = false;
            request.require_exact_depth = false;
            request.raster_backend = lfs::rendering::normalizeViewerRasterBackend(request.raster_backend, request.gut);
            request.gut = lfs::rendering::isGutBackend(request.raster_backend);
            const auto result = scene_renderer_->render(model, request, false, vr_render_targets_[i]);
            if (!result)
                return unavailable(result.error());
            eyes[i] = {.image = result->image,
                       .image_view = result->image_view,
                       .image_layout = result->image_layout,
                       .image_generation = result->generation,
                       .completion_timeline = result->completion_semaphore,
                       .completion_value = result->completion_value,
                       .size = result->size,
                       .alloc_size = result->alloc_size,
                       .flip_y = result->flip_y,
                       .rendered = true};
        }
        retainVksplatScratch();
        return eyes;
    }

    Result<std::shared_ptr<core::Tensor>> RenderingManager::renderPreviewImageWithState(
        SceneManager* const scene_manager,
        const lfs::core::SplatData& model,
        SceneRenderState scene_state,
        const glm::mat3& rotation,
        const glm::vec3& position,
        const float focal_length_mm,
        const int width,
        const int height,
        const bool render_lock_held,
        std::optional<lfs::rendering::CameraIntrinsics> intrinsics_override,
        std::optional<bool> orthographic_override,
        std::optional<float> ortho_scale_override,
        std::optional<glm::vec3> background_color_override,
        const PreviewImageReadback readback,
        const float rasterization_scale,
        const bool equirectangular,
        const RenderSettings* settings_override,
        const bool gaussian_selection_visible,
        const bool deterministic_export) {
        const auto readback_config =
            previewImageReadbackConfig(readback, background_color_override.has_value());

        // Output purpose, not pixel format, selects stable ties. Interactive
        // thumbnails explicitly retain the fast sort and cold-frame warmup.
        auto rendered = renderPreviewImageToPreviewSlotWithState(
            settings_override ? *settings_override : getSettings(),
            scene_manager,
            model,
            scene_state,
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock_held,
            intrinsics_override,
            {},
            {},
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            readback_config.transparent_background_override,
            rasterization_scale,
            deterministic_export, readback_config.capture_float_color, equirectangular,
            gaussian_selection_visible);
        if (!rendered) {
            if (!intrinsics_override && isTileInstanceOverflow(rendered.error()) &&
                height > kMinPreviewSubdivisionHeight) {
                return renderPreviewImageTiledWithState(
                    scene_manager,
                    model,
                    std::move(scene_state),
                    rotation,
                    position,
                    focal_length_mm,
                    width,
                    height,
                    render_lock_held,
                    background_color_override,
                    orthographic_override,
                    ortho_scale_override,
                    readback,
                    rasterization_scale, equirectangular, deterministic_export);
            }
            LOG_ERROR("Gaussian preview image render failed: {}", rendered.error());
            return previewImageFailure(ErrorCode::InvalidArgument, "Invalid preview dimensions");
        }

        if (readback_config.capture_float_color)
            return scene_renderer_->readLinearColorImage(preview_render_target_);
        std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> image =
            std::unexpected("unsupported preview image readback format");
        if (readback_config.dtype == lfs::core::DataType::UInt8 &&
            readback_config.channels == 4) {
            image = scene_renderer_->readOutputImageRgba8(
                preview_render_target_);
        } else if (readback_config.dtype == lfs::core::DataType::UInt8) {
            image = scene_renderer_->readOutputImageRgb8(
                preview_render_target_);
        } else {
            image = scene_renderer_->readOutputImage(
                preview_render_target_);
        }
        if (!image) {
            LOG_ERROR("Gaussian preview image readback failed: {}", image.error());
            return previewImageFailure(ErrorCode::Unavailable, rendered.error());
        }
        return std::move(*image);
    }

    std::expected<void, std::string> RenderingManager::renderPreviewImageToPreviewSlotWithState(
        const RenderSettings& settings,
        SceneManager* const scene_manager,
        const lfs::core::SplatData& source_model,
        SceneRenderState scene_state,
        const glm::mat3& rotation,
        const glm::vec3& position,
        const float focal_length_mm,
        const int width,
        const int height,
        const bool render_lock_held,
        std::optional<lfs::rendering::CameraIntrinsics> intrinsics_override,
        const glm::ivec2 subregion_origin,
        const glm::ivec2 subregion_full_size,
        std::optional<bool> orthographic_override,
        std::optional<float> ortho_scale_override,
        std::optional<glm::vec3> background_color_override,
        std::optional<bool> transparent_background_override,
        const float rasterization_scale,
        const bool deterministic_export, const bool capture_float_color, const bool equirectangular,
        const bool gaussian_selection_visible) {
        if (width <= 0 || height <= 0) {
            return std::unexpected("invalid preview render dimensions");
        }
        if (!last_graphics_context_) {
            return std::unexpected("no Vulkan context is available");
        }
        if (last_graphics_context_->terminalState() != RendererTerminalState::Running) {
            return std::unexpected("renderer is unavailable after a GPU failure; restart LichtFeld Studio");
        }
        const auto leaf_view = lodLeafRenderView(source_model);
        const lfs::core::SplatData& model = leaf_view ? *leaf_view : source_model;
        if (!hasRenderableGaussians(&model)) {
            return std::unexpected("no renderable Gaussian model is available");
        }
        if (!scene_state.combined_model || scene_state.combined_model == &source_model) {
            scene_state.combined_model = &model;
        }

        RenderSettings preview_settings = settings;
        preview_settings.focal_length_mm = std::clamp(
            focal_length_mm,
            lfs::rendering::MIN_FOCAL_LENGTH_MM,
            lfs::rendering::MAX_FOCAL_LENGTH_MM);
        preview_settings.split_view_mode = SplitViewMode::Disabled;
        preview_settings.equirectangular = equirectangular;
        if (background_color_override) {
            preview_settings.background_color = *background_color_override;
        }
        if (orthographic_override) {
            preview_settings.orthographic = *orthographic_override;
        }
        if (ortho_scale_override && std::isfinite(*ortho_scale_override) && *ortho_scale_override > 0.0f) {
            preview_settings.ortho_scale = *ortho_scale_override;
        }

        Viewport preview_viewport(
            static_cast<std::size_t>(width),
            static_cast<std::size_t>(height));
        preview_viewport.setViewMatrix(rotation, position);

        FrameContext frame_ctx{
            .viewport = preview_viewport,
            .render_lock_held = render_lock_held,
            .scene_manager = scene_manager,
            .model = &model,
            .scene_state = std::move(scene_state),
            .settings = preview_settings,
            .render_size = {width, height},
            .viewport_pos = {0, 0},
            .gaussian_selection_visible = gaussian_selection_visible,
            .cursor_preview = {},
            .gizmo = {},
            .view_panels = {},
        };

        auto request = buildViewportRenderRequest(frame_ctx, frame_ctx.render_size);
        request.capture_float_color = capture_float_color;
        request.require_complete_output = true;
        if (capture_float_color)
            request.depth_view = false;
        if (transparent_background_override) {
            request.transparent_background = *transparent_background_override;
        }
        const auto supplied_intrinsics_override = intrinsics_override;
        request.deterministic_export = deterministic_export;
        request.frame_view.intrinsics_override = std::move(intrinsics_override);
        if (supplied_intrinsics_override) {
            request.frame_view.containment_intrinsics = *supplied_intrinsics_override;
        }
        request.frame_view.subregion_origin = subregion_origin;
        request.frame_view.subregion_full_size = subregion_full_size;
        request.frame_view.rasterization_scale = rasterization_scale;
        request.raster_backend =
            lfs::rendering::normalizeViewerRasterBackend(request.raster_backend, request.gut);
        request.gut = lfs::rendering::isGutBackend(request.raster_backend);
        if (!lfs::rendering::isVkSplatBackend(request.raster_backend)) {
            return std::unexpected(std::format(
                "unsupported raster backend '{}'",
                lfs::rendering::gaussianRasterBackendId(request.raster_backend)));
        }

        if (!scene_renderer_) {
            scene_renderer_ = createSceneRenderer(*last_graphics_context_);
        }

        if (auto* const effects = customEffects()) {
            effects->select_programs = [this, effects](const std::string_view host,
                                                       const lfs::rendering::ViewportRenderRequest&) -> Status {
                if (export_draw_)
                    return selectExportPrograms(host);
                if (shader_service_ && shader_service_->hasBindings() && shader_service_->hasApplicableBindings(sequencer_custom_reveal_))
                    shader_service_->demandLive(std::array{std::string(host)}, last_complete_);
                effects->usePrograms(live_programs_);
                return {};
            };
        }
        // Preview/export output is read back right after this call, so the
        // renderer returns only a complete frame for this view.
        auto render_result = scene_renderer_->render(
            model,
            request,
            false,
            preview_render_target_,
            false,
            deterministic_export);
        if (!render_result) {
            return std::unexpected(render_result.error());
        }
        return {};
    }

    Result<std::shared_ptr<core::Tensor>> RenderingManager::renderPreviewImageTiledWithState(
        SceneManager* const scene_manager,
        const lfs::core::SplatData& model,
        SceneRenderState scene_state,
        const glm::mat3& rotation,
        const glm::vec3& position,
        const float focal_length_mm,
        const int width,
        const int height,
        const bool render_lock_held,
        std::optional<glm::vec3> background_color_override,
        std::optional<bool> orthographic_override,
        std::optional<float> ortho_scale_override,
        const PreviewImageReadback readback,
        const float rasterization_scale,
        const bool equirectangular,
        const bool deterministic_export) {
        if (width <= 0 || height <= 0) {
            return previewImageFailure(ErrorCode::InvalidArgument, "Invalid tiled preview dimensions");
        }
        const auto readback_config =
            previewImageReadbackConfig(readback, background_color_override.has_value());

        const int tile_width = width;
        const int tile_height_limit = previewTileHeightForWidth(tile_width);
        if (tile_height_limit <= 0) {
            return previewImageFailure(ErrorCode::InvalidArgument, "No valid preview tile height");
        }

        LOG_INFO("Gaussian preview image {}x{} uses tiled render readback: tile_width={} max_tile_height={}",
                 width,
                 height,
                 tile_width,
                 tile_height_limit);

        auto output = lfs::core::Tensor::empty(
            {static_cast<std::size_t>(height),
             static_cast<std::size_t>(width),
             static_cast<std::size_t>(readback_config.channels)},
            lfs::core::Device::CPU,
            readback_config.dtype);
        if (!output.is_valid()) {
            LOG_TRACE("Gaussian preview tiled render failed to allocate output tensor");
            return previewImageFailure(ErrorCode::ResourceExhausted, "Tiled preview output allocation failed");
        }

        int band_height_limit = tile_height_limit;
        // #1574 1-deep export pipelining: submit ticket for band N, render band N+1,
        // then wait ticket N (memcpy on deliver), submit N+1, ... At most one outstanding
        // export ticket. With the 3-deep OutputSlotRing + cell pin, Preview reuse of the
        // sourced ring cell blocks until that ticket retires — so a second outstanding
        // export ticket is unnecessary for source-image safety.
        std::optional<std::uint64_t> outstanding_export_ticket;
        for (int tile_y = 0; tile_y < height;) {
            int tile_height = std::min(band_height_limit, height - tile_y);
            const auto intrinsics = equirectangular ? std::nullopt : std::optional{previewTileIntrinsics(width, height, focal_length_mm)};
            while (true) {
                auto rendered = renderPreviewImageToPreviewSlotWithState(
                    getSettings(),
                    scene_manager,
                    model,
                    scene_state,
                    rotation,
                    position,
                    focal_length_mm,
                    tile_width,
                    tile_height,
                    render_lock_held,
                    intrinsics,
                    {0, tile_y},
                    {width, height},
                    orthographic_override,
                    ortho_scale_override,
                    background_color_override,
                    readback_config.transparent_background_override,
                    rasterization_scale,
                    deterministic_export, readback_config.capture_float_color, equirectangular);
                if (rendered) {
                    break;
                }
                if (!isTileInstanceOverflow(rendered.error()) ||
                    tile_height <= kMinPreviewSubdivisionHeight) {
                    LOG_TRACE("Gaussian preview tiled render failed at tile y={} height={}: {}",
                              tile_y,
                              tile_height,
                              rendered.error());
                    if (outstanding_export_ticket) {
                        (void)scene_renderer_->waitReadbackTicket(*outstanding_export_ticket);
                    }
                    return previewImageFailure(ErrorCode::Unavailable, rendered.error());
                }
                tile_height = std::max(kMinPreviewSubdivisionHeight,
                                       (tile_height / 2) / kPreviewTileHeightAlignment * kPreviewTileHeightAlignment);
                band_height_limit = tile_height;
                LOG_WARN("Gaussian preview band overflow at y={}; retrying with height={}",
                         tile_y,
                         tile_height);
            }
            if (readback_config.capture_float_color) {
                // Consume the owned, synchronized float band before raster scratch
                // is reused. The existing byte pipeline remains asynchronous.
                auto image = scene_renderer_->readLinearColorImage(preview_render_target_);
                if (!image)
                    return std::move(image).error();
                output.slice(0, size_t(tile_y), size_t(tile_y + tile_height)).copy_from(**image);
                tile_y += tile_height;
                continue;
            }
            // After render of band N: wait prior band's copy (if any), then submit band N.
            if (outstanding_export_ticket) {
                auto waited = scene_renderer_->waitReadbackTicket(*outstanding_export_ticket);
                if (!waited) {
                    LOG_TRACE("Gaussian preview tiled prior-band readback failed at tile y={}: {}",
                              tile_y,
                              waited.error());
                    return previewImageFailure(ErrorCode::Unavailable, waited.error());
                }
                outstanding_export_ticket.reset();
            }
            auto ticket = scene_renderer_->submitReadOutputImageIntoCpuHwcTicket(
                preview_render_target_,
                output,
                0,
                tile_y);
            if (!ticket) {
                LOG_TRACE("Gaussian preview tiled readback submit failed at tile y={} height={}: {}",
                          tile_y,
                          tile_height,
                          ticket.error());
                return previewImageFailure(ErrorCode::Unavailable, ticket.error());
            }
            outstanding_export_ticket = *ticket;
            tile_y += tile_height;
        }
        if (outstanding_export_ticket) {
            auto waited = scene_renderer_->waitReadbackTicket(*outstanding_export_ticket);
            if (!waited) {
                LOG_TRACE("Gaussian preview tiled final-band readback failed: {}", waited.error());
                return previewImageFailure(ErrorCode::Unavailable, waited.error());
            }
        }

        return std::make_shared<lfs::core::Tensor>(std::move(output));
    }

    float RenderingManager::getDepthAtPixel(ViewId view, const int x, const int y,
                                            const std::optional<SplitViewPanelId> panel, const bool nonblocking) const {
        const float cached_depth = viewState(view).viewport_artifact_service_.sampleLinearDepthAt(
            x,
            y, viewState(view).frame_lifecycle_service_.lastViewportSize(),
            panel, nonblocking);
        if (cached_depth > 0.0f || viewState(view).viewport_artifact_service_.hasDepthSampler()) {
            return cached_depth;
        }

        if (!scene_renderer_ || !last_graphics_context_) {
            return -1.0f;
        }

        RenderTargetId target = viewState(view).main_render_target_;

        glm::ivec2 source_size = viewState(view).frame_lifecycle_service_.lastViewportSize();

        const auto depth = scene_renderer_->sampleDepthAtPixel(
            SceneRenderer::DepthSampleRequest{
                .pixel = {x, y},
                .source_size = source_size,
                .target = target,
            });
        if (!depth) {
            LOG_TRACE("Scene depth sample failed: {}", depth.error());
            return -1.0f;
        }
        return *depth;
    }

    float RenderingManager::renderExpectedDepthAtPixel(const ExpectedDepthSampleRequest& request) {
        if (!request.scene_manager ||
            !request.viewport ||
            request.render_size.x <= 0 ||
            request.render_size.y <= 0 ||
            request.pixel.x < 0 ||
            request.pixel.y < 0 ||
            request.pixel.x >= request.render_size.x ||
            request.pixel.y >= request.render_size.y) {
            return -1.0f;
        }

        auto render_lock = acquireLiveModelRenderLock(request.scene_manager);
        const auto settings = settingsForView(request.view);
        SceneRenderState scene_state;
        const lfs::core::SplatData* model = nullptr;
        if (splitViewUsesPLYComparison(settings.split_view_mode)) {
            scene_state = request.scene_manager->buildRenderState({.metadata_only = true});
            attachCustomEffect(scene_state, *request.scene_manager, scene_time_);
            const auto& scene = request.scene_manager->getScene();
            const auto sample = resolvePlyComparisonDepthSample(
                scene,
                settings.split_view_offset,
                request.panel.value_or(SplitViewPanelId::Left));
            if (sample.uses_owned_node_model && sample.node && hasRenderableGaussians(sample.model)) {
                scopeSceneRenderStateToVisibleSplatNode(
                    scene_state,
                    scene,
                    *sample.node,
                    sample.visible_index,
                    scene_coords::nodeVisualizerWorldTransform(scene, sample.node->id));
                model = sample.model;
            } else if (hasRenderableGaussians(sample.model)) {
                scene_state.combined_model = sample.model;
                scene_state.transform_indices = scene.peekTransformIndices();
                scene_state.node_visibility_mask.assign(scene_state.model_transforms.size(), false);
                if (sample.visible_index >= 0 &&
                    static_cast<size_t>(sample.visible_index) < scene_state.node_visibility_mask.size()) {
                    scene_state.node_visibility_mask[static_cast<size_t>(sample.visible_index)] = true;
                }
                model = sample.model;
            }
        } else {
            scene_state = request.scene_manager->buildRenderState();
            attachCustomEffect(scene_state, *request.scene_manager, scene_time_);
            model = scene_state.combined_model;
        }
        if (!hasRenderableGaussians(model)) {
            return -1.0f;
        }

        auto rendered = renderDepthCaptureToPreviewSlotWithState(
            settings,
            request.scene_manager,
            *model,
            std::move(scene_state),
            request.viewport->camera.R,
            request.viewport->camera.t,
            request.focal_length_mm,
            request.render_size.x,
            request.render_size.y,
            render_lock.has_value(),
            true,
            std::nullopt,
            request.orthographic,
            request.ortho_scale);
        if (!rendered) {
            LOG_TRACE("Expected-depth pixel render failed: {}", rendered.error());
            return -1.0f;
        }

        auto depth = scene_renderer_->readPreviewDepth(
            preview_render_target_);
        if (!depth) {
            LOG_TRACE("Expected-depth pixel readback failed: {}", depth.error());
            return -1.0f;
        }

        return sampleDepthTensorAt(**depth, request.pixel).value_or(-1.0f);
    }

} // namespace lfs::vis
