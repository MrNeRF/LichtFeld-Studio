/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_geometry.hpp"
#include "viewport_reference_renderer.hpp"

#include "core/executable_path.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "internal/resource_paths.hpp"
#include "split_view_tensor_program.hpp"
#include "tensor_frame_uploads.hpp"
#include "view_render_state.hpp"
#include "viewport_compose_program.hpp"
#include "viewport_grid_program.hpp"
#include "viewport_overlay_program.hpp"
#include "viewport_reference_state.hpp"
#include "viewport_tensor_meshes.hpp"
#include "viewport_vignette_program.hpp"
#include "window/graphics_context.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace lfs::vis {
    namespace {
        using Module = lfs::core::GpuKernelModule;
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        FramebufferRect framebufferRect(const ViewportFrameDesc& desc, const glm::vec2 position,
                                        const glm::vec2 size) {
            return scaledFramebufferRect(position, size, desc.framebuffer_scale, glm::ivec2(desc.framebuffer_extent));
        }

        FramebufferRect framebufferRect(const ViewportFrameDesc& desc) {
            return framebufferRect(desc, desc.viewport_pos, desc.viewport_size);
        }

        struct ImageLayout {
            std::uint32_t width = 1;
            std::uint32_t height = 1;
            std::uint32_t channels = 4;
            bool chw = false;
            bool floating_point = false;
            bool valid = false;
        };

        ImageLayout imageLayout(const Tensor& image) {
            ImageLayout result;
            if (!image.is_valid() || image.ndim() != 3 || !image.is_contiguous())
                return result;
            if (image.size(2) == 3 || image.size(2) == 4) {
                result.width = static_cast<std::uint32_t>(image.size(1));
                result.height = static_cast<std::uint32_t>(image.size(0));
                result.channels = static_cast<std::uint32_t>(image.size(2));
            } else if (image.size(0) == 3 || image.size(0) == 4) {
                result.width = static_cast<std::uint32_t>(image.size(2));
                result.height = static_cast<std::uint32_t>(image.size(1));
                result.channels = static_cast<std::uint32_t>(image.size(0));
                result.chw = true;
            } else {
                return result;
            }
            result.floating_point = image.dtype() == DataType::Float32;
            result.valid = result.width > 0 && result.height > 0 &&
                           (result.floating_point || image.dtype() == DataType::UInt8);
            return result;
        }

        struct alignas(16) ComposeParameters {
            std::uint64_t destination = 0;
            std::uint64_t scene = 0;
            std::uint32_t destination_width = 0;
            std::uint32_t destination_height = 0;
            std::int32_t viewport_x = 0;
            std::int32_t viewport_y = 0;
            std::uint32_t viewport_width = 0;
            std::uint32_t viewport_height = 0;
            std::uint32_t source_width = 1;
            std::uint32_t source_height = 1;
            std::uint32_t source_allocation_width = 1;
            std::uint32_t source_allocation_height = 1;
            std::uint32_t source_channels = 4;
            std::uint32_t source_float = 0;
            std::uint32_t source_chw = 0;
            std::uint32_t flip_y = 0;
            std::uint32_t has_scene = 0;
            std::uint32_t over_destination = 0;
            std::array<float, 4> background{0, 0, 0, 1};
        };
        static_assert(offsetof(ComposeParameters, background) == 80);
        static_assert(sizeof(ComposeParameters) == 96);

        struct alignas(16) VignetteParameters {
            std::uint64_t destination = 0;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::int32_t viewport_x = 0;
            std::int32_t viewport_y = 0;
            std::uint32_t viewport_width = 0;
            std::uint32_t viewport_height = 0;
            float intensity = 0;
            float radius = 0;
            float softness = 0;
            std::uint32_t padding = 0;
        };
        static_assert(sizeof(VignetteParameters) == 48);

        struct alignas(16) SplitViewParameters {
            std::uint64_t destination = 0;
            std::uint64_t left = 0;
            std::uint64_t right = 0;
            std::uint32_t destination_width = 0;
            std::uint32_t destination_height = 0;
            std::int32_t viewport_x = 0;
            std::int32_t viewport_y = 0;
            std::uint32_t viewport_width = 0;
            std::uint32_t viewport_height = 0;
            std::int32_t content_x = 0;
            std::int32_t content_y = 0;
            std::uint32_t content_width = 0;
            std::uint32_t content_height = 0;
            std::uint32_t left_width = 0;
            std::uint32_t left_height = 0;
            std::uint32_t left_channels = 0;
            std::uint32_t left_float = 0;
            std::uint32_t left_chw = 0;
            std::uint32_t right_width = 0;
            std::uint32_t right_height = 0;
            std::uint32_t right_channels = 0;
            std::uint32_t right_float = 0;
            std::uint32_t right_chw = 0;
            float split_position = 0.5f;
            float left_start = 0.0f;
            float left_end = 1.0f;
            float right_start = 0.0f;
            float right_end = 1.0f;
            std::uint32_t left_normalize = 0;
            std::uint32_t right_normalize = 0;
            std::uint32_t left_flip_y = 0;
            std::uint32_t right_flip_y = 0;
            std::uint32_t left_filter = 0;
            std::uint32_t right_filter = 0;
            std::uint32_t loss_visualization = 0;
            std::uint32_t padding = 0;
            std::uint32_t vector_padding = 0;
            std::array<float, 2> left_uv_scale{1, 1};
            std::array<float, 2> left_uv_clamp{1, 1};
            std::array<float, 2> right_uv_scale{1, 1};
            std::array<float, 2> right_uv_clamp{1, 1};
            std::array<float, 2> left_texcoord_scale{1, 1};
            std::array<float, 2> left_texcoord_offset{0, 0};
            std::array<float, 2> right_texcoord_scale{1, 1};
            std::array<float, 2> right_texcoord_offset{0, 0};
            std::array<float, 4> background{0, 0, 0, 1};
        };
        static_assert(offsetof(SplitViewParameters, destination_width) == 24);
        static_assert(offsetof(SplitViewParameters, left_uv_scale) == 160);
        static_assert(offsetof(SplitViewParameters, background) == 224);
        static_assert(sizeof(SplitViewParameters) == 240);

        // Matches viewport_overlay.slang's Parameters.
        struct alignas(16) OverlayParameters {
            std::uint64_t vertices = 0;
            std::uint64_t instances = 0;
            std::uint64_t texture = 0;
            std::uint64_t depth = 0;
            std::array<float, 16> view{};
            std::array<float, 4> viewport_rect{};
            std::array<float, 4> depth_params{};
            std::array<float, 4> uv_region{1, 1, 1, 1};
            std::array<float, 4> panel{};
            std::array<float, 4> projection{};
            std::array<float, 4> tint{1, 1, 1, 1};
            std::array<float, 4> effects{};
            std::array<std::uint32_t, 4> sizes{};
            std::array<std::uint32_t, 4> flags{};
        };
        static_assert(offsetof(OverlayParameters, view) == 32);
        static_assert(offsetof(OverlayParameters, sizes) == 208);
        static_assert(sizeof(OverlayParameters) == 240);

        // Matches viewport_grid.slang's Parameters.
        struct alignas(16) GridParameters {
            std::array<float, 4> view_position_plane{};
            std::array<float, 4> opacity{};
            std::array<float, 4> near_origin{}, near_x{}, near_y{};
            std::array<float, 4> far_origin{}, far_x{}, far_y{};
        };
        static_assert(sizeof(GridParameters) == 128);

        constexpr std::uint32_t kFrustumVertexCount = 48;
        constexpr float kFrustumLineThickness = 1.5f;

        std::array<float, 4> array3(const glm::vec3& value, const float w = 0.0f) {
            return {value.x, value.y, value.z, w};
        }

        GridParameters gridParameters(const ViewportGridOverlay& grid) {
            const auto corners = gridFrustumCorners(grid.view, grid.projection, grid.orthographic);
            return {
                .view_position_plane = array3(grid.view_position, float(std::clamp(grid.plane, 0, 2))),
                .opacity = {std::clamp(grid.opacity, 0.0f, 1.0f), 0, 0, 0},
                .near_origin = array3(corners.near_origin),
                .near_x = array3(corners.near_x),
                .near_y = array3(corners.near_y),
                .far_origin = array3(corners.far_origin),
                .far_x = array3(corners.far_x),
                .far_y = array3(corners.far_y),
            };
        }

        std::array<float, 4> array(const glm::vec4& value) {
            return {value.x, value.y, value.z, value.w};
        }
    } // namespace

    std::optional<std::uint64_t> referenceSceneOutputGeneration(const ViewRenderState&) {
        return std::nullopt;
    }

    void clearViewportReferenceOutput(ViewRenderState&) {}

    struct ViewportReferenceResources::Impl {};
    ViewportReferenceResources::ViewportReferenceResources() : impl_(std::make_unique<Impl>()) {}
    ViewportReferenceResources::~ViewportReferenceResources() = default;

    struct ViewportReferenceRenderer::Impl {
        GraphicsContext* graphics = nullptr;
        std::unique_ptr<Module> compose_program;
        std::unique_ptr<Module> split_program;
        std::unique_ptr<Module> vignette_program;
        std::unique_ptr<Module> overlay_program;
        std::unique_ptr<Module> grid_program;
        Tensor dummy;
        Tensor dummy_records;
        Tensor dummy_depth;
        TensorFrameUploads uploads;
        TensorMeshPass meshes;
        SceneUpscalerSelection upscaler{};
        // The environment map is loaded once per path, like SharedViewportGpuAssets.
        std::filesystem::path environment_path;
        Tensor environment_map;
        // Frustum instances change rarely; the GUI keeps one shared block alive.
        const void* frustum_source = nullptr;
        std::uint64_t frustum_generation = 0;
        std::size_t frustum_count = 0;
        Tensor frustum_instances;

        // One recorded overlay draw. Owns its tensors and parameters until the
        // batch is submitted, because Draw only references them.
        struct OverlayDraw {
            std::string_view vertex, fragment;
            OverlayParameters parameters;
            Tensor vertices, instances, depth;
            std::shared_ptr<const Tensor> texture;
            std::uint32_t vertex_count = 0, instance_count = 1;
            FramebufferRect rect;
        };
        std::deque<OverlayDraw> overlays;

        bool ensureProgram() {
            if (compose_program)
                return true;
            auto compose = Module::load(viewport_compose_program_entries());
            auto split = Module::load(split_view_tensor_program_entries());
            auto vignette = Module::load(viewport_vignette_program_entries());
            auto overlay = Module::load(viewport_overlay_program_entries());
            auto grid = Module::load(viewport_grid_program_entries());
            if (!compose || !split || !vignette || !overlay || !grid) {
                const auto detail = !compose    ? compose.error().detail()
                                    : !split    ? split.error().detail()
                                    : !vignette ? vignette.error().detail()
                                    : !overlay  ? overlay.error().detail()
                                                : grid.error().detail();
                LOG_ERROR("Could not load tensor viewport compositor: {}", detail);
                return false;
            }
            compose_program = std::move(*compose);
            split_program = std::move(*split);
            vignette_program = std::move(*vignette);
            overlay_program = std::move(*overlay);
            grid_program = std::move(*grid);
            dummy = Tensor::full({1, 1, 4}, 255, Device::GPU, DataType::UInt8);
            dummy_records = Tensor::zeros({1, 4}, Device::GPU, DataType::Float32);
            dummy_depth = Tensor::zeros({1}, Device::GPU, DataType::Float32);
            return overlay_program->supports_raster() && grid_program->supports_raster();
        }

        bool compose(const GraphicsFrame& frame, const ViewportFrameDesc& desc,
                     const bool over_destination) {
            Tensor* destination = graphics ? graphics->finalImageTensor(frame) : nullptr;
            if (!destination || !destination->is_valid())
                return false;
            const auto rect = framebufferRect(desc);
            const Tensor* scene = desc.scene_image && desc.scene_image->is_valid()
                                      ? desc.scene_image.get()
                                      : &dummy;
            const ImageLayout layout = imageLayout(*scene);
            const bool has_scene = scene != &dummy && layout.valid;
            const auto valid_width = desc.scene_image_size.x > 0
                                         ? static_cast<std::uint32_t>(desc.scene_image_size.x)
                                         : layout.width;
            const auto valid_height = desc.scene_image_size.y > 0
                                          ? static_cast<std::uint32_t>(desc.scene_image_size.y)
                                          : layout.height;
            ComposeParameters parameters{
                .destination_width = static_cast<std::uint32_t>(destination->size(1)),
                .destination_height = static_cast<std::uint32_t>(destination->size(0)),
                .viewport_x = rect.x,
                .viewport_y = rect.y,
                .viewport_width = rect.width,
                .viewport_height = rect.height,
                .source_width = std::min(valid_width, layout.width),
                .source_height = std::min(valid_height, layout.height),
                .source_allocation_width = layout.width,
                .source_allocation_height = layout.height,
                .source_channels = layout.channels,
                .source_float = layout.floating_point ? 1u : 0u,
                .source_chw = layout.chw ? 1u : 0u,
                .flip_y = desc.scene_image_flip_y ? 1u : 0u,
                .has_scene = has_scene ? 1u : 0u,
                .over_destination = over_destination ? 1u : 0u,
                .background = {desc.background_color.r, desc.background_color.g,
                               desc.background_color.b, 1.0f},
            };
            const std::array bindings{
                Module::Binding{0, destination, Module::Access::ReadWrite},
                Module::Binding{8, scene},
            };
            auto result = compose_program->dispatch({
                .function = "composeViewport",
                .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                .groups = {Module::groups_for(parameters.destination_width, 64),
                           parameters.destination_height, 1},
            });
            if (!result) {
                LOG_ERROR("Tensor viewport composition failed: {}", result.error().detail());
                return false;
            }
            return true;
        }

        bool composeSplit(const GraphicsFrame& frame, const ViewportFrameDesc& desc) {
            Tensor* destination = graphics ? graphics->finalImageTensor(frame) : nullptr;
            const auto& split = desc.split_view;
            if (!destination || !destination->is_valid() || !split.left.image || !split.right.image)
                return false;
            const ImageLayout left_layout = imageLayout(*split.left.image);
            const ImageLayout right_layout = imageLayout(*split.right.image);
            if (!left_layout.valid || !right_layout.valid ||
                split.left.image->device() != Device::GPU ||
                split.right.image->device() != Device::GPU) {
                return false;
            }

            const auto viewport = framebufferRect(desc);
            glm::ivec4 content = split.content_rect;
            const glm::ivec2 coordinates = split.coordinate_extent;
            if (coordinates.x > 0 && coordinates.y > 0) {
                const float scale_x = static_cast<float>(viewport.width) /
                                      static_cast<float>(coordinates.x);
                const float scale_y = static_cast<float>(viewport.height) /
                                      static_cast<float>(coordinates.y);
                const int right = static_cast<int>(std::lround(
                    static_cast<float>(content.x + content.z) * scale_x));
                const int bottom = static_cast<int>(std::lround(
                    static_cast<float>(content.y + content.w) * scale_y));
                content.x = static_cast<int>(std::lround(static_cast<float>(content.x) * scale_x));
                content.y = static_cast<int>(std::lround(static_cast<float>(content.y) * scale_y));
                content.z = std::max(right - content.x, 1);
                content.w = std::max(bottom - content.y, 1);
            }
            content.x += viewport.x;
            content.y += viewport.y;

            const auto vec2 = [](const glm::vec2 value) {
                return std::array<float, 2>{value.x, value.y};
            };
            SplitViewParameters parameters{
                .destination_width = static_cast<std::uint32_t>(destination->size(1)),
                .destination_height = static_cast<std::uint32_t>(destination->size(0)),
                .viewport_x = viewport.x,
                .viewport_y = viewport.y,
                .viewport_width = viewport.width,
                .viewport_height = viewport.height,
                .content_x = content.x,
                .content_y = content.y,
                .content_width = static_cast<std::uint32_t>(std::max(content.z, 1)),
                .content_height = static_cast<std::uint32_t>(std::max(content.w, 1)),
                .left_width = left_layout.width,
                .left_height = left_layout.height,
                .left_channels = left_layout.channels,
                .left_float = left_layout.floating_point ? 1u : 0u,
                .left_chw = left_layout.chw ? 1u : 0u,
                .right_width = right_layout.width,
                .right_height = right_layout.height,
                .right_channels = right_layout.channels,
                .right_float = right_layout.floating_point ? 1u : 0u,
                .right_chw = right_layout.chw ? 1u : 0u,
                .split_position = split.split_position,
                .left_start = split.left.start_position,
                .left_end = split.left.end_position,
                .right_start = split.right.start_position,
                .right_end = split.right.end_position,
                .left_normalize = split.left.normalize_x_to_panel ? 1u : 0u,
                .right_normalize = split.right.normalize_x_to_panel ? 1u : 0u,
                .left_flip_y = split.left.flip_y ? 1u : 0u,
                .right_flip_y = split.right.flip_y ? 1u : 0u,
                .left_filter = split.left.spatial_filter ? 1u : 0u,
                .right_filter = split.right.spatial_filter ? 1u : 0u,
                .loss_visualization = split.loss_visualization ? 1u : 0u,
                .left_uv_scale = vec2(split.left.uv_scale),
                .left_uv_clamp = vec2(split.left.uv_clamp_max),
                .right_uv_scale = vec2(split.right.uv_scale),
                .right_uv_clamp = vec2(split.right.uv_clamp_max),
                .left_texcoord_scale = vec2(split.left.texcoord_scale),
                .left_texcoord_offset = vec2(split.left.texcoord_offset),
                .right_texcoord_scale = vec2(split.right.texcoord_scale),
                .right_texcoord_offset = vec2(split.right.texcoord_offset),
                .background = {split.background.r, split.background.g, split.background.b, 1.0f},
            };
            const std::array bindings{
                Module::Binding{0, destination, Module::Access::ReadWrite},
                Module::Binding{8, split.left.image.get()},
                Module::Binding{16, split.right.image.get()},
            };
            auto result = split_program->dispatch({
                .function = "composeSplitView",
                .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                .groups = {Module::groups_for(parameters.destination_width, 64),
                           parameters.destination_height, 1},
            });
            if (!result) {
                LOG_ERROR("Tensor split-view composition failed: {}", result.error().detail());
                return false;
            }
            return true;
        }

        template <std::size_t Records>
        Tensor uploadRecords(const std::vector<std::array<std::array<float, 4>, Records>>& records) {
            if (records.empty())
                return {};
            return uploads.upload(std::as_bytes(std::span(records)), {records.size() * Records, 4},
                                  DataType::Float32);
        }

        // Splat depth binding shared by world overlays; see ViewportDepth.
        void bindDepth(OverlayDraw& draw, const ViewportFrameDesc& desc, const bool world) {
            const auto& depth = desc.depth_blit;
            const bool available = world && depth.depth && depth.depth->is_valid() &&
                                   depth.depth->device() == Device::GPU && depth.depth->ndim() == 2;
            draw.depth = available ? *depth.depth : Tensor{};
            draw.parameters.depth_params = {available ? 1.0f : 0.0f, depth.flip_y ? 1.0f : 0.0f, 0, 0};
            draw.parameters.uv_region = {depth.uv_scale.x, depth.uv_scale.y, depth.uv_clamp_max.x, depth.uv_clamp_max.y};
            if (available) {
                draw.parameters.sizes[2] = static_cast<std::uint32_t>(depth.depth->size(1));
                draw.parameters.sizes[3] = static_cast<std::uint32_t>(depth.depth->size(0));
            }
        }

        OverlayDraw& addOverlay(std::string_view vertex, std::string_view fragment,
                                const FramebufferRect rect, const ViewportFrameDesc& desc,
                                const bool world) {
            auto& draw = overlays.emplace_back();
            draw.vertex = vertex;
            draw.fragment = fragment;
            draw.rect = rect;
            draw.parameters.viewport_rect = {float(rect.x), float(rect.y), float(rect.width), float(rect.height)};
            bindDepth(draw, desc, world);
            return draw;
        }

        void addTriangles(const std::span<const ViewportOverlayVertex> vertices,
                          const FramebufferRect rect, const ViewportFrameDesc& desc) {
            if (vertices.empty())
                return;
            std::vector<std::array<std::array<float, 4>, 2>> records;
            records.reserve(vertices.size());
            for (const auto& vertex : vertices)
                records.push_back({{{vertex.position.x, vertex.position.y, 0, 0}, array(vertex.color)}});
            auto& draw = addOverlay("overlayVertex", "overlayFragment", rect, desc, false);
            draw.vertices = uploadRecords(records);
            draw.vertex_count = static_cast<std::uint32_t>(vertices.size());
            draw.parameters.flags[3] = 2;
        }

        void addShapes(const std::span<const ViewportShapeOverlayVertex> vertices,
                       const FramebufferRect rect, const ViewportFrameDesc& desc, const bool world) {
            if (vertices.empty())
                return;
            std::vector<std::array<std::array<float, 4>, 5>> records;
            records.reserve(vertices.size());
            for (const auto& vertex : vertices)
                records.push_back({{{vertex.position.x, vertex.position.y, vertex.screen_position.x, vertex.screen_position.y},
                                    {vertex.p0.x, vertex.p0.y, vertex.p1.x, vertex.p1.y},
                                    array(vertex.color),
                                    array(vertex.params),
                                    {vertex.view_depth, 0, 0, 0}}});
            auto& draw = addOverlay("shapeVertex", "shapeFragment", rect, desc, world);
            draw.vertices = uploadRecords(records);
            draw.vertex_count = static_cast<std::uint32_t>(vertices.size());
            draw.parameters.flags[3] = 5;
        }

        void addTextured(const std::span<const ViewportTexturedOverlay> textured,
                         const FramebufferRect rect, const ViewportFrameDesc& desc, const bool world) {
            for (const auto& overlay : textured) {
                if (!overlay.image)
                    continue;
                const ImageLayout layout = imageLayout(*overlay.image);
                if (!layout.valid || layout.channels != 4)
                    continue;
                std::shared_ptr<const Tensor> image = overlay.image;
                if (image->device() != Device::GPU)
                    image = std::make_shared<Tensor>(uploads.upload(
                        std::as_bytes(std::span(static_cast<const std::byte*>(image->data_ptr()), image->bytes())),
                        image->shape(), image->dtype()));
                std::vector<std::array<std::array<float, 4>, 2>> records;
                records.reserve(overlay.vertices.size());
                for (const auto& vertex : overlay.vertices)
                    records.push_back({{{vertex.position.x, vertex.position.y, vertex.uv.x, vertex.uv.y},
                                        {vertex.view_depth, 0, 0, 0}}});
                auto& draw = addOverlay("texturedVertex", "texturedFragment", rect, desc, world);
                draw.vertices = uploadRecords(records);
                draw.texture = std::move(image);
                draw.vertex_count = static_cast<std::uint32_t>(overlay.vertices.size());
                draw.parameters.tint = array(overlay.tint_opacity);
                draw.parameters.effects = array(overlay.effects);
                draw.parameters.sizes[0] = layout.width;
                draw.parameters.sizes[1] = layout.height;
                draw.parameters.flags = {layout.floating_point ? 1u : 0u, layout.chw ? 1u : 0u, 0, 2};
            }
        }

        void addFrusta(const ViewportFrameDesc& desc) {
            const auto& instances = desc.frustum_overlay_data ? desc.frustum_overlay_data->frustum_instances
                                                              : desc.frustum_instances;
            const auto& batches = desc.frustum_overlay_data ? desc.frustum_overlay_data->frustum_batches
                                                            : desc.frustum_batches;
            if (instances.empty() || batches.empty())
                return;
            static_assert(sizeof(ViewportFrustumInstance) == 5 * 4 * sizeof(float));
            const void* source = desc.frustum_overlay_data ? static_cast<const void*>(desc.frustum_overlay_data.get())
                                                           : static_cast<const void*>(instances.data());
            const auto generation = desc.frustum_overlay_data ? desc.frustum_overlay_data->generation : 0;
            if (!frustum_instances.is_valid() || source != frustum_source || generation != frustum_generation ||
                instances.size() != frustum_count || !desc.frustum_overlay_data) {
                frustum_instances = uploads.upload(std::as_bytes(std::span(instances)),
                                                   {instances.size() * 5, 4}, DataType::Float32);
                frustum_source = source;
                frustum_generation = generation;
                frustum_count = instances.size();
            }
            for (const auto& batch : batches) {
                if (batch.instance_count == 0 || batch.first_instance + batch.instance_count > instances.size())
                    continue;
                const auto rect = framebufferRect(desc, batch.viewport_pos, batch.viewport_size);
                if (rect.width == 0 || rect.height == 0)
                    continue;
                auto& draw = addOverlay("frustumVertex", "shapeFragment", rect, desc, true);
                draw.instances = frustum_instances;
                draw.vertex_count = kFrustumVertexCount;
                draw.instance_count = batch.instance_count;
                std::memcpy(draw.parameters.view.data(), &batch.view[0][0], sizeof(draw.parameters.view));
                draw.parameters.depth_params[2] = kFrustumLineThickness;
                draw.parameters.depth_params[3] = batch.equirectangular ? 2.0f : (batch.orthographic ? 1.0f : 0.0f);
                draw.parameters.panel = {batch.viewport_pos.x, batch.viewport_pos.y, batch.viewport_size.x, batch.viewport_size.y};
                draw.parameters.projection = {batch.render_size.x, batch.render_size.y, batch.focal_x, batch.focal_y};
                draw.parameters.flags[2] = batch.first_instance;
            }
        }

        void addPivots(const std::span<const ViewportPivotOverlay> pivots, const FramebufferRect rect,
                       const ViewportFrameDesc& desc) {
            for (const auto& pivot : pivots) {
                auto& draw = addOverlay("pivotVertex", "pivotFragment", rect, desc, false);
                draw.vertex_count = 6;
                draw.parameters.tint = {pivot.color.r, pivot.color.g, pivot.color.b, std::clamp(pivot.opacity, 0.0f, 1.0f)};
                draw.parameters.effects = {pivot.center_ndc.x, pivot.center_ndc.y, pivot.size_ndc.x, pivot.size_ndc.y};
            }
        }

        const Tensor* environmentMap(const std::filesystem::path& path) {
            if (path == environment_path)
                return environment_map.is_valid() ? &environment_map : nullptr;
            environment_path = path;
            environment_map = {};
            const auto resolved = resolveEnvironmentMapPath(path);
            auto [pixels, width, height, channels] = lfs::core::load_image_float(resolved);
            if (!pixels || width <= 0 || height <= 0 || channels <= 0) {
                if (pixels)
                    lfs::core::free_image_float(pixels);
                LOG_WARN("Tensor compositor could not read environment map {}", lfs::core::path_to_utf8(resolved));
                return nullptr;
            }
            const std::size_t count = static_cast<std::size_t>(width) * height;
            std::vector<float> rgba(count * 4);
            for (std::size_t i = 0; i < count; ++i) {
                const float r = pixels[i * channels];
                rgba[i * 4] = r;
                rgba[i * 4 + 1] = channels >= 2 ? pixels[i * channels + 1] : r;
                rgba[i * 4 + 2] = channels >= 3 ? pixels[i * channels + 2] : r;
                rgba[i * 4 + 3] = 1.0f;
            }
            lfs::core::free_image_float(pixels);
            // Half precision, like the Vulkan environment texture.
            const auto half = Tensor::from_blob(rgba.data(), {std::size_t(height), std::size_t(width), 4},
                                                Device::CPU, DataType::Float32)
                                  .to(DataType::Float16)
                                  .contiguous();
            environment_map = uploads.upload(
                std::as_bytes(std::span(static_cast<const std::byte*>(half.data_ptr()), half.bytes())),
                half.shape(), DataType::Float16);
            return &environment_map;
        }

        bool drawEnvironment(Tensor& destination, const ViewportFrameDesc& desc, const FramebufferRect rect) {
            const auto& environment = desc.environment;
            if (!environment.enabled || environment.map_path.empty())
                return false;
            const Tensor* map = environmentMap(environment.map_path);
            if (!map)
                return false;
            auto& draw = addOverlay("environmentVertex", "environmentFragment", rect, desc, false);
            draw.texture = std::make_shared<Tensor>(*map);
            draw.vertex_count = 6;
            const glm::mat4 rotation(environment.camera_to_world);
            std::memcpy(draw.parameters.view.data(), &rotation[0][0], sizeof(draw.parameters.view));
            draw.parameters.projection = array(environment.intrinsics);
            draw.parameters.panel = {environment.viewport_size.x, environment.viewport_size.y,
                                     environment.exposure, environment.rotation_radians};
            draw.parameters.effects = {environment.equirectangular_view ? 1.0f : 0.0f, 0, 0, 0};
            draw.parameters.sizes[0] = static_cast<std::uint32_t>(map->size(1));
            draw.parameters.sizes[1] = static_cast<std::uint32_t>(map->size(0));
            flushOverlays(destination);
            return true;
        }

        void flushOverlays(Tensor& destination) {
            if (overlays.empty())
                return;
            std::vector<std::array<Module::Binding, 4>> bindings(overlays.size());
            std::vector<Module::Draw> draws;
            draws.reserve(overlays.size());
            for (std::size_t i = 0; i < overlays.size(); ++i) {
                auto& overlay = overlays[i];
                bindings[i] = {Module::Binding{0, overlay.vertices.is_valid() ? &overlay.vertices : &dummy_records},
                               Module::Binding{8, overlay.instances.is_valid() ? &overlay.instances : &dummy_records},
                               Module::Binding{16, overlay.texture ? overlay.texture.get() : &dummy},
                               Module::Binding{24, overlay.depth.is_valid() ? &overlay.depth : &dummy_depth}};
                const auto& rect = overlay.rect;
                draws.push_back({.vertex = overlay.vertex,
                                 .fragment = overlay.fragment,
                                 .arguments = {std::as_bytes(std::span(&overlay.parameters, 1)), bindings[i]},
                                 .color = &destination,
                                 .vertex_count = overlay.vertex_count,
                                 .instance_count = overlay.instance_count,
                                 .scissor = Module::Scissor{std::uint32_t(rect.x), std::uint32_t(rect.y), rect.width, rect.height},
                                 .viewport = Module::Viewport{float(rect.x), float(rect.y), float(rect.width), float(rect.height)},
                                 .blend = Module::Blend::StraightAlpha,
                                 .depth_compare = Module::Compare::Always,
                                 .depth_write = false});
            }
            auto result = overlay_program->draw_batch(draws);
            if (!result)
                LOG_ERROR("Tensor viewport overlays ({} draws) failed: {}", draws.size(), result.error().detail());
            overlays.clear();
        }

        void drawGrids(Tensor& destination, const ViewportFrameDesc& desc) {
            std::vector<ViewportGridOverlay> grids = desc.grid_overlays;
            if (grids.empty() && desc.grid_enabled)
                grids.push_back({.viewport_pos = desc.viewport_pos,
                                 .viewport_size = desc.viewport_size,
                                 .render_size = {std::max(static_cast<int>(std::lround(desc.viewport_size.x)), 1),
                                                 std::max(static_cast<int>(std::lround(desc.viewport_size.y)), 1)},
                                 .view = desc.grid_view,
                                 .projection = desc.grid_projection,
                                 .view_projection = desc.grid_view_projection,
                                 .view_position = desc.grid_view_position,
                                 .plane = desc.grid_plane,
                                 .opacity = desc.grid_opacity,
                                 .orthographic = desc.grid_orthographic});
            std::vector<GridParameters> parameters;
            std::vector<Module::Draw> draws;
            parameters.reserve(grids.size());
            draws.reserve(grids.size());
            for (const auto& grid : grids) {
                if (grid.viewport_size.x <= 0.0f || grid.viewport_size.y <= 0.0f || grid.render_size.x <= 0 ||
                    grid.render_size.y <= 0 || grid.opacity <= 0.0f)
                    continue;
                const auto rect = framebufferRect(desc, grid.viewport_pos, grid.viewport_size);
                if (rect.width == 0 || rect.height == 0)
                    continue;
                parameters.push_back(gridParameters(grid));
                draws.push_back({.vertex = "gridVertex",
                                 .fragment = "gridFragment",
                                 .arguments = {std::as_bytes(std::span(&parameters.back(), 1)), {}},
                                 .color = &destination,
                                 .vertex_count = 6,
                                 .scissor = Module::Scissor{std::uint32_t(rect.x), std::uint32_t(rect.y), rect.width, rect.height},
                                 .viewport = Module::Viewport{float(rect.x), float(rect.y), float(rect.width), float(rect.height)},
                                 .blend = Module::Blend::StraightAlpha,
                                 .depth_compare = Module::Compare::Always,
                                 .depth_write = false});
            }
            if (draws.empty())
                return;
            auto result = grid_program->draw_batch(draws);
            if (!result)
                LOG_ERROR("Tensor viewport grid failed: {}", result.error().detail());
        }

        void vignette(Tensor& destination, const ViewportFrameDesc& desc) {
            if (!desc.vignette_enabled || desc.vignette_intensity <= 0.0f)
                return;
            const auto rect = framebufferRect(desc);
            if (rect.width == 0 || rect.height == 0)
                return;
            VignetteParameters parameters{
                .width = static_cast<std::uint32_t>(destination.size(1)),
                .height = static_cast<std::uint32_t>(destination.size(0)),
                .viewport_x = rect.x,
                .viewport_y = rect.y,
                .viewport_width = rect.width,
                .viewport_height = rect.height,
                .intensity = desc.vignette_intensity,
                .radius = desc.vignette_radius,
                .softness = desc.vignette_softness,
            };
            const std::array bindings{
                Module::Binding{0, &destination, Module::Access::ReadWrite}};
            auto result = vignette_program->dispatch({
                .function = "applyVignette",
                .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                .groups = {Module::groups_for(rect.width, 64), rect.height, 1},
            });
            if (!result)
                LOG_ERROR("Tensor viewport vignette failed: {}", result.error().detail());
        }

        // Same order as the Vulkan viewport pass graph.
        void record(const GraphicsFrame& frame, const ViewportFrameDesc& desc) {
            Tensor* destination = graphics->finalImageTensor(frame);
            if (!destination)
                return;
            const auto rect = framebufferRect(desc);
            if (rect.width == 0 || rect.height == 0)
                return;
            const bool environment = !desc.split_view.enabled && drawEnvironment(*destination, desc, rect);
            const bool composed = desc.split_view.enabled ? composeSplit(frame, desc)
                                                          : compose(frame, desc, environment);
            if (!composed)
                return;
            meshes.record(*destination, desc,
                          Module::Scissor{std::uint32_t(rect.x), std::uint32_t(rect.y), rect.width, rect.height},
                          uploads);
            const std::size_t post_count = std::min<std::size_t>(
                desc.post_ui_overlay_vertex_count, desc.overlay_triangles.size());
            const std::size_t base_count = desc.overlay_triangles.size() - post_count;
            const auto* overlay_data = desc.overlay_triangles.data();
            const auto& world_textures = desc.frustum_overlay_data
                                             ? desc.frustum_overlay_data->textured_overlays
                                             : desc.textured_overlays;
            addTextured(world_textures, rect, desc, true);
            addTriangles({overlay_data, base_count}, rect, desc);
            addShapes(desc.shape_overlay_triangles, rect, desc, true);
            addFrusta(desc);
            addPivots(desc.pivot_overlays, rect, desc);
            flushOverlays(*destination);
            drawGrids(*destination, desc);
            vignette(*destination, desc);
            addShapes(desc.ui_shape_overlay_triangles, rect, desc, false);
            addTextured(desc.ui_textured_overlays, rect, desc, false);
            if (post_count > 0)
                addTriangles({overlay_data + base_count, post_count}, rect, desc);
            flushOverlays(*destination);
        }
    };

    ViewportReferenceRenderer::ViewportReferenceRenderer(
        std::shared_ptr<ViewportReferenceResources>)
        : impl_(std::make_unique<Impl>()) {}
    ViewportReferenceRenderer::~ViewportReferenceRenderer() = default;

    bool ViewportReferenceRenderer::initialize(GraphicsContext& graphics) {
        impl_->graphics = &graphics;
        return impl_->ensureProgram();
    }

    void ViewportReferenceRenderer::prepare(GraphicsContext&, const ViewportFrameDesc& desc,
                                            ViewRenderState& view) {
        const bool available = static_cast<std::uint32_t>(desc.scene_upscaler) <
                                   static_cast<std::uint32_t>(SceneUpscalerBackend::FirstExternal) ||
                               (isMetalFxBackend(desc.scene_upscaler) && metalFxBackendAvailable(desc.scene_upscaler));
        impl_->upscaler = resolveSceneUpscalerSelection(desc.scene_upscaler,
                                                        available && !desc.scene_upscaler_mode_unsupported && !view.scene_upscaler_runtime_failed_,
                                                        desc.scene_upscaler_mode_unsupported ? SceneUpscalerFallback::UnsupportedMode : SceneUpscalerFallback::RuntimeUnavailable);
    }

    void ViewportReferenceRenderer::record(const GraphicsFrame& frame,
                                           const ViewportFrameDesc& desc) {
        try {
            impl_->record(frame, desc);
        } catch (const std::exception& error) {
            LOG_ERROR("Tensor viewport compositor failed: {}", error.what());
        }
    }
    void ViewportReferenceRenderer::recordFrame(const GraphicsFrame& frame,
                                                const ViewportFrameDesc& desc,
                                                ViewRenderState&) {
        record(frame, desc);
    }
    void ViewportReferenceRenderer::prepareImport(GraphicsContext& graphics,
                                                  const ViewportFrameDesc& desc,
                                                  ViewRenderState& view,
                                                  ViewportReferenceRenderer*) {
        prepare(graphics, desc, view);
    }
    void ViewportReferenceRenderer::discardImportMesh(std::uint64_t) {}
    SceneUpscalerSelection ViewportReferenceRenderer::sceneUpscalerSelection() const {
        return impl_->upscaler;
    }

    void snapshotViewportReference(const ViewRenderState& view, ViewportFrameDesc& desc) {
        desc.scene_image = view.vulkan_viewport_image_;
        desc.scene_image_size = view.vulkan_viewport_image_size_;
        desc.scene_image_alloc_size = view.vulkan_viewport_image_alloc_size_;
        desc.scene_image_flip_y = view.vulkan_viewport_image_flip_y_;
        desc.depth_blit = {.depth = view.viewport_depth_image_};
        desc.environment = view.viewport_environment_;
        desc.split_view = view.split_view_;
        desc.mesh_view_projection = view.viewport_meshes_.view_projection;
        desc.mesh_camera_position = view.viewport_meshes_.camera_position;
        desc.mesh_items = view.viewport_meshes_.items;
        desc.scene_outputs = {{
            .target = view.main_render_target_,
            .color = desc.scene_image,
            .depth = view.viewport_depth_image_,
            .size = desc.scene_image_size,
            .allocation_size = desc.scene_image_alloc_size,
            .generation = view.vulkan_viewport_image_generation_,
            .flip_y = desc.scene_image_flip_y,
        }};
    }

    void prepareViewportReference(GraphicsContext&, ViewRenderState&, bool, bool) {}
    void clearViewportReference(ViewRenderState&) {}
    void shutdownViewportReference(ViewRenderState&, GraphicsContext*) {}
    std::size_t viewportReferenceFramesInFlight(GraphicsContext&) { return 3; }
    std::size_t viewportReferenceMemoryBytes(GraphicsContext&, std::size_t, std::size_t) {
        return 0;
    }

    std::unique_ptr<ViewportReferenceRenderer> createViewportReferenceRenderer(
        GraphicsContext&, std::shared_ptr<ViewportReferenceResources>& resources) {
        if (!resources)
            resources = std::make_shared<ViewportReferenceResources>();
        return std::make_unique<ViewportReferenceRenderer>(resources);
    }
} // namespace lfs::vis
