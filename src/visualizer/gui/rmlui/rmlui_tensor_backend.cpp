/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rmlui_tensor_backend.hpp"

#include "core/gpu_kernel_module.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "gui/rmlui/rml_image_file.hpp"
#include "gui/rmlui/tensor_frosted_glass.hpp"
#include "gui/ui_texture.hpp"
#include "rendering/tensor_frame_uploads.hpp"
#include "rmlui_composite_program.hpp"
#include "rmlui_draw_program.hpp"
#include "rmlui_mask_program.hpp"
#include "window/graphics_context.hpp"

#include <RmlUi/Core/Matrix4.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <optional>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::vis::gui {
    namespace {
        using Module = lfs::core::GpuKernelModule;
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        struct Geometry {
            Tensor positions;
            Tensor colors;
            Tensor texcoords;
            uint32_t vertices = 0;
        };
        struct Texture {
            std::shared_ptr<const Tensor> image;
            std::string name;
            // Generated textures (glyph atlases) sample nearest, like the Vulkan renderer.
            bool linear = true;
        };
        struct Layer {
            Tensor image;
        };
        struct alignas(8) DrawParameters {
            uint64_t positions = 0;
            uint64_t colors = 0;
            uint64_t texcoords = 0;
            uint64_t texture = 0;
            uint64_t mask = 0;
            uint64_t pointer_padding = 0;
            std::array<float, 16> transform{};
            std::array<float, 2> translation{};
            std::array<uint32_t, 2> texture_size{1, 1};
            std::array<uint32_t, 2> target_size{1, 1};
            uint32_t textured = 0;
            uint32_t mask_enabled = 0;
            float mask_value = 1;
            uint32_t linear_filter = 0;
            uint64_t uniform_padding = 0;
        };
        static_assert(offsetof(DrawParameters, transform) == 48);
        static_assert(sizeof(DrawParameters) == 160);

        struct alignas(8) CompositeParameters {
            uint64_t source = 0;
            uint64_t destination = 0;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t replace = 0;
            uint32_t padding = 0;
        };
        struct alignas(8) MaskParameters {
            uint64_t coverage = 0;
            uint64_t mask = 0;
            uint32_t width = 0;
            uint32_t height = 0;
        };

        std::optional<std::uintptr_t> tensorUrlId(std::string_view source) {
            constexpr std::string_view prefix = "lfs-tensor://";
            if (!source.starts_with(prefix))
                return std::nullopt;
            const auto end = source.find('?', prefix.size());
            const auto value = source.substr(prefix.size(), end - prefix.size());
            std::uintptr_t id = 0;
            const auto [last, error] = std::from_chars(value.data(), value.data() + value.size(), id);
            return error == std::errc{} && last == value.data() + value.size()
                       ? std::optional{id}
                       : std::nullopt;
        }
    } // namespace

    struct TensorRmlUiRenderer::Impl {
        GraphicsContext* graphics = nullptr;
        std::unique_ptr<Module> draw_program;
        std::unique_ptr<Module> composite_program;
        std::unique_ptr<Module> mask_program;
        Tensor* base = nullptr;
        Tensor dummy_texture;
        Tensor clip_mask;
        Rml::Matrix4f transform = Rml::Matrix4f::Identity();
        Rml::Matrix4f context = Rml::Matrix4f::Identity();
        // Same scissor model as RenderInterface_VK: RmlUi regions are context
        // coordinates offset by the context translation, then intersected with
        // the context clip and the cache-capture area.
        Rml::Vector2f context_offset{0, 0};
        bool transform_enabled = false;
        bool rml_scissor_enabled = false;
        Module::Scissor rml_scissor{};
        std::optional<Module::Scissor> context_clip;
        std::optional<Module::Scissor> capture_area;
        bool mask_enabled = false;
        bool transformed_scissor = false;
        std::vector<Rml::LayerHandle> layer_stack{0};
        std::unordered_map<Rml::LayerHandle, std::unique_ptr<Layer>> layers;
        Rml::LayerHandle next_layer = 1;
        std::size_t geometry_bytes = 0;
        std::size_t texture_bytes = 0;
        TensorFrostedGlassBackdrop frosted_glass;
        uint64_t texture_generation = 0;
        bool preview_used = false;

        // Draws into one target are submitted together. Each entry owns its
        // tensors, so geometry and textures may be released before the flush.
        struct PendingDraw {
            DrawParameters parameters;
            Tensor positions, colors, texcoords;
            std::shared_ptr<const Tensor> texture;
            const Tensor* mask = nullptr;
            std::string_view fragment;
            uint32_t vertices = 0;
            Module::Scissor scissor;
        };
        std::vector<PendingDraw> pending;
        Tensor* pending_target = nullptr;
        bool pending_clear = false;
        std::array<float, 4> pending_clear_value{};
        std::vector<Tensor> free_layers;
        lfs::vis::TensorFrameUploads uploads;

        Tensor* target(Rml::LayerHandle handle) {
            if (handle == 0)
                return base;
            const auto found = layers.find(handle);
            return found == layers.end() ? nullptr : &found->second->image;
        }
        Tensor* target() { return target(layer_stack.back()); }

        static Module::Scissor intersect(const Module::Scissor& a, const Module::Scissor& b) {
            const auto left = std::max(a.x, b.x), top = std::max(a.y, b.y);
            const auto right = std::min(a.x + a.width, b.x + b.width);
            const auto bottom = std::min(a.y + a.height, b.y + b.height);
            return {left, top, right > left ? right - left : 0u, bottom > top ? bottom - top : 0u};
        }

        Module::Scissor clampedRect(float x1, float y1, float x2, float y2) const {
            const int width = base ? static_cast<int>(base->size(1)) : 0;
            const int height = base ? static_cast<int>(base->size(0)) : 0;
            const int left = std::clamp(static_cast<int>(std::floor(x1)), 0, width);
            const int top = std::clamp(static_cast<int>(std::floor(y1)), 0, height);
            const int right = std::clamp(static_cast<int>(std::ceil(x2)), left, width);
            const int bottom = std::clamp(static_cast<int>(std::ceil(y2)), top, height);
            return {static_cast<uint32_t>(left), static_cast<uint32_t>(top),
                    static_cast<uint32_t>(right - left), static_cast<uint32_t>(bottom - top)};
        }

        Module::Scissor drawScissor() const {
            const auto width = base ? static_cast<uint32_t>(base->size(1)) : 0u;
            const auto height = base ? static_cast<uint32_t>(base->size(0)) : 0u;
            Module::Scissor rect{0, 0, width, height};
            if (context_clip)
                rect = intersect(rect, *context_clip);
            if (rml_scissor_enabled && !transformed_scissor)
                rect = intersect(rect, rml_scissor);
            if (capture_area)
                rect = intersect(rect, *capture_area);
            return rect;
        }

        bool ensureMask() {
            if (!base)
                return false;
            const auto shape = lfs::core::TensorShape{base->size(0), base->size(1), 4};
            if (!clip_mask.is_valid() || clip_mask.shape() != shape)
                clip_mask = Tensor::zeros(shape, Device::GPU, DataType::UInt8);
            return true;
        }

        void flush() {
            if (pending.empty() || !pending_target) {
                pending.clear();
                pending_target = nullptr;
                pending_clear = false;
                return;
            }
            std::vector<std::array<Module::Binding, 5>> bindings(pending.size());
            std::vector<Module::Draw> draws;
            draws.reserve(pending.size());
            for (std::size_t i = 0; i < pending.size(); ++i) {
                auto& entry = pending[i];
                const Tensor* texture = entry.texture ? entry.texture.get() : &dummy_texture;
                bindings[i] = {Module::Binding{0, &entry.positions}, Module::Binding{8, &entry.colors},
                               Module::Binding{16, &entry.texcoords}, Module::Binding{24, texture},
                               Module::Binding{32, entry.mask}};
                draws.push_back({.vertex = "uiVertex",
                                 .fragment = entry.fragment,
                                 .arguments = {std::as_bytes(std::span(&entry.parameters, 1)), bindings[i]},
                                 .color = pending_target,
                                 .vertex_count = entry.vertices,
                                 .scissor = entry.scissor,
                                 .blend = entry.fragment == "maskFragment" ? Module::Blend::Opaque
                                                                           : Module::Blend::PremultipliedAlpha,
                                 .depth_compare = Module::Compare::Always,
                                 .depth_write = false,
                                 .clear_color = i == 0 && pending_clear,
                                 .color_clear = pending_clear_value});
            }
            auto result = draw_program->draw_batch(draws);
            if (!result)
                LOG_ERROR("Tensor RmlUi draw batch of {} failed: {}", draws.size(), result.error().detail());
            pending.clear();
            pending_target = nullptr;
            pending_clear = false;
        }

        bool render(Geometry& geometry, Rml::Vector2f translation, Texture* texture,
                    Tensor& destination, std::string_view fragment = "uiFragment",
                    float mask_value = 1, bool clear = false,
                    std::array<float, 4> clear_value = {0, 0, 0, 0}) {
            if (!draw_program || geometry.vertices == 0)
                return false;
            if (pending_target != &destination || clear)
                flush();
            const bool masked = fragment == "uiFragment" && (mask_enabled || transformed_scissor) && ensureMask();
            PendingDraw entry;
            auto& parameters = entry.parameters;
            const auto matrix = Rml::Matrix4f::ProjectOrtho(
                                    0.0f, float(destination.size(1)), float(destination.size(0)),
                                    0.0f, -10000.0f, 10000.0f) *
                                context * transform;
            std::memcpy(parameters.transform.data(), matrix.data(), sizeof(parameters.transform));
            parameters.translation = {translation.x, translation.y};
            parameters.target_size = {uint32_t(destination.size(1)), uint32_t(destination.size(0))};
            parameters.textured = texture ? 1u : 0u;
            parameters.mask_enabled = masked ? 1u : 0u;
            parameters.mask_value = mask_value;
            parameters.linear_filter = texture && texture->linear ? 1u : 0u;
            // UiTextures replace their tensor on every upload; resolve the
            // current one so live previews never draw a stale image.
            if (texture && !texture->name.empty())
                if (const auto id = tensorUrlId(texture->name))
                    if (auto current = uiTextureImage(*id)) {
                        texture->image = std::move(current);
                        preview_used = true;
                    }
            if (texture && texture->image && texture->image->is_valid()) {
                parameters.texture_size = {uint32_t(texture->image->size(1)),
                                           uint32_t(texture->image->size(0))};
                entry.texture = texture->image;
            }
            entry.positions = geometry.positions;
            entry.colors = geometry.colors;
            entry.texcoords = geometry.texcoords;
            entry.mask = masked ? &clip_mask : &dummy_texture;
            entry.fragment = fragment;
            entry.vertices = geometry.vertices;
            entry.scissor = drawScissor();
            if (pending.empty()) {
                pending_target = &destination;
                pending_clear = clear;
                pending_clear_value = clear_value;
            }
            pending.push_back(std::move(entry));
            return true;
        }

        Tensor acquireLayer() {
            const auto shape = base->shape();
            for (auto it = free_layers.begin(); it != free_layers.end(); ++it) {
                if (it->shape() == shape) {
                    Tensor layer = std::move(*it);
                    free_layers.erase(it);
                    layer.zero_();
                    return layer;
                }
            }
            auto layer = Tensor::zeros(shape, Device::GPU, DataType::UInt8);
            layer.set_name("ui.layer");
            return layer;
        }

        void recycleLayers() {
            for (auto& [handle, layer] : layers)
                free_layers.push_back(std::move(layer->image));
            layers.clear();
            if (free_layers.size() > 2)
                free_layers.erase(free_layers.begin(), free_layers.end() - 2);
        }
    };

    TensorRmlUiRenderer::TensorRmlUiRenderer() : impl_(std::make_unique<Impl>()) {}
    TensorRmlUiRenderer::~TensorRmlUiRenderer() { shutdown(); }

    bool TensorRmlUiRenderer::initialize(GraphicsContext& graphics) {
        impl_->graphics = &graphics;
        auto draw = Module::load(rmlui_draw_program_entries());
        auto composite = Module::load(rmlui_composite_program_entries());
        auto mask = Module::load(rmlui_mask_program_entries());
        if (!draw || !composite || !mask) {
            const auto detail = !draw        ? draw.error().detail()
                                : !composite ? composite.error().detail()
                                             : mask.error().detail();
            LOG_ERROR("Could not load tensor RmlUi program: {}", detail);
            return false;
        }
        impl_->draw_program = std::move(*draw);
        impl_->composite_program = std::move(*composite);
        impl_->mask_program = std::move(*mask);
        impl_->dummy_texture = Tensor::full({1, 1, 4}, 255, Device::GPU, DataType::UInt8);
        return impl_->draw_program->supports_raster();
    }

    void TensorRmlUiRenderer::shutdown() {
        if (!impl_)
            return;
        impl_->pending.clear();
        impl_->pending_target = nullptr;
        impl_->free_layers.clear();
        impl_->layers.clear();
        impl_->layer_stack = {0};
        impl_->clip_mask = {};
        impl_->dummy_texture = {};
        impl_->draw_program.reset();
        impl_->composite_program.reset();
        impl_->mask_program.reset();
        impl_->frosted_glass.reset();
        impl_->base = nullptr;
        impl_->graphics = nullptr;
    }

    bool TensorRmlUiRenderer::beginFrame(const GraphicsFrame& frame) {
        if (!impl_->graphics || !impl_->draw_program || !impl_->composite_program ||
            !impl_->mask_program)
            return false;
        impl_->base = impl_->graphics->finalImageTensor(frame);
        impl_->layer_stack = {0};
        impl_->pending.clear();
        impl_->pending_target = nullptr;
        impl_->recycleLayers();
        impl_->preview_used = false;
        resetContextRenderState();
        return impl_->base && impl_->base->is_valid();
    }

    void TensorRmlUiRenderer::endFrame() {
        impl_->flush();
        impl_->base = nullptr;
        impl_->recycleLayers();
        impl_->layer_stack = {0};
    }

    void TensorRmlUiRenderer::resetContextRenderState() {
        // Cache-capture mode survives: the manager resets between
        // beginCacheCapture and PushLayer/Render.
        impl_->transform = Rml::Matrix4f::Identity();
        impl_->transform_enabled = false;
        impl_->context = Rml::Matrix4f::Identity();
        impl_->context_offset = {0, 0};
        impl_->context_clip.reset();
        impl_->rml_scissor_enabled = false;
        impl_->transformed_scissor = false;
        impl_->mask_enabled = false;
    }
    void TensorRmlUiRenderer::setContextOffset(float x, float y) {
        impl_->context_offset = {std::round(x), std::round(y)};
        impl_->context = Rml::Matrix4f::Translate(impl_->context_offset.x, impl_->context_offset.y, 0.0f);
    }
    void TensorRmlUiRenderer::setContextClipRect(float x1, float y1, float x2, float y2) {
        impl_->context_clip = x2 <= x1 || y2 <= y1 ? Module::Scissor{}
                                                   : impl_->clampedRect(x1, y1, x2, y2);
    }

    Rml::CompiledGeometryHandle TensorRmlUiRenderer::CompileGeometry(
        Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) {
        if (indices.empty() || indices.size() % 3)
            return {};
        std::vector<std::array<float, 2>> positions(indices.size()), texcoords(indices.size());
        std::vector<std::array<float, 4>> colors(indices.size());
        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || std::size_t(indices[i]) >= vertices.size())
                return {};
            const auto& vertex = vertices[indices[i]];
            positions[i] = {vertex.position.x, vertex.position.y};
            texcoords[i] = {vertex.tex_coord.x, vertex.tex_coord.y};
            colors[i] = {vertex.colour.red / 255.0f, vertex.colour.green / 255.0f,
                         vertex.colour.blue / 255.0f, vertex.colour.alpha / 255.0f};
        }
        try {
            auto result = std::make_unique<Geometry>();
            auto& uploads = impl_->uploads;
            result->positions = uploads.upload(std::span<const std::array<float, 2>>(positions),
                                               {indices.size(), 2}, DataType::Float32);
            result->colors = uploads.upload(std::span<const std::array<float, 4>>(colors),
                                            {indices.size(), 4}, DataType::Float32);
            result->texcoords = uploads.upload(std::span<const std::array<float, 2>>(texcoords),
                                               {indices.size(), 2}, DataType::Float32);
            result->positions.set_name("ui.geometry.positions");
            result->colors.set_name("ui.geometry.colors");
            result->texcoords.set_name("ui.geometry.texcoords");
            result->vertices = uint32_t(indices.size());
            impl_->geometry_bytes += result->positions.bytes() + result->colors.bytes() + result->texcoords.bytes();
            return reinterpret_cast<Rml::CompiledGeometryHandle>(result.release());
        } catch (const std::exception& error) {
            LOG_ERROR("Could not compile tensor RmlUi geometry: {}", error.what());
            return {};
        }
    }

    void TensorRmlUiRenderer::RenderGeometry(Rml::CompiledGeometryHandle handle,
                                             Rml::Vector2f translation,
                                             Rml::TextureHandle texture) {
        auto* geometry = reinterpret_cast<Geometry*>(handle);
        auto* destination = impl_->target();
        if (geometry && destination)
            impl_->render(*geometry, translation, reinterpret_cast<Texture*>(texture), *destination);
    }
    void TensorRmlUiRenderer::ReleaseGeometry(Rml::CompiledGeometryHandle handle) {
        std::unique_ptr<Geometry> geometry(reinterpret_cast<Geometry*>(handle));
        if (geometry)
            impl_->geometry_bytes -= std::min(impl_->geometry_bytes,
                                              geometry->positions.bytes() + geometry->colors.bytes() + geometry->texcoords.bytes());
    }

    Rml::TextureHandle TensorRmlUiRenderer::GenerateTexture(Rml::Span<const Rml::byte> source,
                                                            Rml::Vector2i dimensions) {
        if (dimensions.x <= 0 || dimensions.y <= 0 ||
            source.size() != std::size_t(dimensions.x) * dimensions.y * 4)
            return {};
        try {
            auto texture = std::make_unique<Texture>();
            texture->linear = false;
            auto image = std::make_shared<Tensor>(impl_->uploads.upload(
                std::as_bytes(std::span(source.data(), source.size())),
                {std::size_t(dimensions.y), std::size_t(dimensions.x), 4}, DataType::UInt8));
            image->set_name("ui.texture");
            texture->image = std::move(image);
            impl_->texture_bytes += texture->image->bytes();
            ++impl_->texture_generation;
            return reinterpret_cast<Rml::TextureHandle>(texture.release());
        } catch (const std::exception& error) {
            LOG_ERROR("Could not generate tensor RmlUi texture: {}", error.what());
            return {};
        }
    }

    Rml::TextureHandle TensorRmlUiRenderer::LoadTexture(Rml::Vector2i& dimensions,
                                                        const Rml::String& source) {
        if (const auto id = tensorUrlId(source)) {
            auto image = uiTextureImage(*id);
            if (!image)
                return {};
            dimensions = {int(image->size(1)), int(image->size(0))};
            auto texture = std::make_unique<Texture>();
            texture->image = std::move(image);
            texture->name = source;
            return reinterpret_cast<Rml::TextureHandle>(texture.release());
        }
        auto image = loadRmlImageFile(source);
        if (!image)
            return {};
        dimensions = {image->width, image->height};
        const auto handle = GenerateTexture({image->rgba.data(), image->rgba.size()}, dimensions);
        if (auto* texture = reinterpret_cast<Texture*>(handle))
            texture->linear = true; // loaded images sample like the Vulkan linear sampler
        return handle;
    }

    void TensorRmlUiRenderer::ReleaseTexture(Rml::TextureHandle handle) {
        std::unique_ptr<Texture> texture(reinterpret_cast<Texture*>(handle));
        if (texture && texture->image)
            impl_->texture_bytes -= std::min(impl_->texture_bytes, texture->image->bytes());
    }

    void TensorRmlUiRenderer::EnableScissorRegion(bool enable) {
        impl_->rml_scissor_enabled = enable;
        if (!enable)
            impl_->transformed_scissor = false;
    }
    void TensorRmlUiRenderer::SetScissorRegion(Rml::Rectanglei region) {
        if (!impl_->rml_scissor_enabled)
            return;
        if (!impl_->transform_enabled) {
            impl_->transformed_scissor = false;
            impl_->rml_scissor = impl_->clampedRect(
                float(region.Left()) + impl_->context_offset.x, float(region.Top()) + impl_->context_offset.y,
                float(region.Right()) + impl_->context_offset.x, float(region.Bottom()) + impl_->context_offset.y);
            return;
        }
        // A transformed region is a quad, not a rectangle. Like the Vulkan
        // stencil path it replaces the clip mask with the region's coverage.
        std::array<Rml::Vertex, 4> vertices{};
        vertices[0].position = Rml::Vector2f(region.TopLeft());
        vertices[1].position = Rml::Vector2f(region.TopRight());
        vertices[2].position = Rml::Vector2f(region.BottomRight());
        vertices[3].position = Rml::Vector2f(region.BottomLeft());
        constexpr std::array indices{0, 2, 1, 0, 3, 2};
        impl_->transformed_scissor = false;
        const bool mask_enabled = impl_->mask_enabled;
        if (const auto geometry = CompileGeometry({vertices.data(), vertices.size()}, {indices.data(), indices.size()})) {
            RenderToClipMask(Rml::ClipMaskOperation::Set, geometry, {});
            ReleaseGeometry(geometry);
        }
        impl_->mask_enabled = mask_enabled;
        impl_->transformed_scissor = true;
    }
    void TensorRmlUiRenderer::EnableClipMask(bool enable) { impl_->mask_enabled = enable; }

    void TensorRmlUiRenderer::RenderToClipMask(Rml::ClipMaskOperation operation,
                                               Rml::CompiledGeometryHandle handle,
                                               Rml::Vector2f translation) {
        auto* geometry = reinterpret_cast<Geometry*>(handle);
        if (!geometry || !impl_->ensureMask())
            return;
        // Queued draws sample the current mask.
        impl_->flush();
        impl_->mask_enabled = false;
        const bool inverse = operation == Rml::ClipMaskOperation::SetInverse;
        if (operation == Rml::ClipMaskOperation::Intersect) {
            Tensor coverage = Tensor::empty(impl_->clip_mask.shape(), Device::GPU, DataType::UInt8);
            impl_->render(*geometry, translation, nullptr, coverage, "maskFragment", 1, true);
            impl_->flush();
            MaskParameters parameters{.width = uint32_t(coverage.size(1)),
                                      .height = uint32_t(coverage.size(0))};
            const std::array bindings{Module::Binding{0, &coverage},
                                      Module::Binding{8, &impl_->clip_mask, Module::Access::ReadWrite}};
            auto result = impl_->mask_program->dispatch({.function = "intersectMask",
                                                         .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                                                         .groups = {Module::groups_for(coverage.size(1), 64), uint32_t(coverage.size(0)), 1}});
            if (!result)
                LOG_ERROR("Tensor RmlUi clip intersection failed: {}", result.error().detail());
        } else {
            impl_->render(*geometry, translation, nullptr, impl_->clip_mask, "maskFragment",
                          inverse ? 0.0f : 1.0f, true,
                          inverse ? std::array<float, 4>{1, 1, 1, 1}
                                  : std::array<float, 4>{0, 0, 0, 0});
            impl_->flush();
        }
        impl_->mask_enabled = true;
    }

    Rml::LayerHandle TensorRmlUiRenderer::PushLayer() {
        if (!impl_->base)
            return {};
        const auto handle = impl_->next_layer++;
        auto layer = std::make_unique<Layer>();
        layer->image = impl_->acquireLayer();
        impl_->layers.emplace(handle, std::move(layer));
        impl_->layer_stack.push_back(handle);
        return handle;
    }

    void TensorRmlUiRenderer::CompositeLayers(Rml::LayerHandle source_handle,
                                              Rml::LayerHandle destination_handle,
                                              Rml::BlendMode blend,
                                              Rml::Span<const Rml::CompiledFilterHandle> filters) {
        auto* source = impl_->target(source_handle);
        auto* destination = impl_->target(destination_handle);
        if (!source || !destination)
            return;
        impl_->flush();
        if (!filters.empty()) {
            static bool warned = false;
            if (!std::exchange(warned, true))
                LOG_WARN("RmlUi layer filters are not implemented by the tensor UI renderer");
        }
        CompositeParameters parameters{.width = uint32_t(source->size(1)),
                                       .height = uint32_t(source->size(0)),
                                       .replace = blend == Rml::BlendMode::Replace};
        const std::array bindings{Module::Binding{0, source},
                                  Module::Binding{8, destination, Module::Access::ReadWrite}};
        auto result = impl_->composite_program->dispatch({.function = "composite",
                                                          .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                                                          .groups = {Module::groups_for(source->size(1), 64), uint32_t(source->size(0)), 1}});
        if (!result)
            LOG_ERROR("Tensor RmlUi layer composite failed: {}", result.error().detail());
    }
    void TensorRmlUiRenderer::PopLayer() {
        if (impl_->layer_stack.size() <= 1)
            return;
        // The popped layer was composited already; queued work targets it only
        // if nothing flushed since, so flush before handing it back to the pool.
        impl_->flush();
        const auto handle = impl_->layer_stack.back();
        impl_->layer_stack.pop_back();
        if (const auto found = impl_->layers.find(handle); found != impl_->layers.end()) {
            impl_->free_layers.push_back(std::move(found->second->image));
            impl_->layers.erase(found);
        }
    }

    Rml::TextureHandle TensorRmlUiRenderer::saveLayerRegionAsTexture(UiPixelRect region,
                                                                     Rml::TextureHandle reuse) {
        auto* source = impl_->target();
        if (!source || region.width == 0 || region.height == 0)
            return {};
        impl_->flush();
        const int x0 = std::clamp(region.x, 0, int(source->size(1)));
        const int y0 = std::clamp(region.y, 0, int(source->size(0)));
        const int x1 = std::clamp(region.x + int(region.width), x0, int(source->size(1)));
        const int y1 = std::clamp(region.y + int(region.height), y0, int(source->size(0)));
        if (x1 == x0 || y1 == y0)
            return {};
        auto* texture = reinterpret_cast<Texture*>(reuse);
        if (!texture)
            texture = new Texture;
        // A full-width row range is already contiguous and would alias the
        // layer, which is recycled after PopLayer; the cache needs its own copy.
        texture->image = std::make_shared<Tensor>(source->slice({{y0, y1}, {x0, x1}, {0, 4}}).clone());
        return reinterpret_cast<Rml::TextureHandle>(texture);
    }
    Rml::TextureHandle TensorRmlUiRenderer::SaveLayerAsTexture() {
        const auto rect = impl_->drawScissor();
        return saveLayerRegionAsTexture({int(rect.x), int(rect.y), rect.width, rect.height}, {});
    }

    void TensorRmlUiRenderer::SetTransform(const Rml::Matrix4f* transform) {
        impl_->transform_enabled = transform != nullptr;
        impl_->transform = transform ? *transform : Rml::Matrix4f::Identity();
    }

    void TensorRmlUiRenderer::renderTextureQuad(Rml::TextureHandle texture, float x, float y,
                                                float width, float height) {
        std::array<Rml::Vertex, 4> vertices{};
        const Rml::ColourbPremultiplied white(255, 255, 255, 255);
        vertices[0] = {{x, y}, white, {0, 0}};
        vertices[1] = {{x + width, y}, white, {1, 0}};
        vertices[2] = {{x + width, y + height}, white, {1, 1}};
        vertices[3] = {{x, y + height}, white, {0, 1}};
        constexpr std::array indices{0, 1, 2, 0, 2, 3};
        const auto geometry = CompileGeometry({vertices.data(), vertices.size()},
                                              {indices.data(), indices.size()});
        RenderGeometry(geometry, {}, texture);
        ReleaseGeometry(geometry);
    }

    bool TensorRmlUiRenderer::renderFrostedGlass(
        const std::span<const UiFrostedGlassRegion> regions) {
        if (!impl_->base || impl_->target() != impl_->base || regions.empty())
            return false;
        impl_->flush();
        if (auto status = impl_->frosted_glass.update(*impl_->base); !status) {
            LOG_ERROR("Tensor frosted glass backdrop failed: {}", status.error().detail());
            return false;
        }

        Texture backdrop;
        backdrop.image = std::make_shared<Tensor>(impl_->frosted_glass.image());
        backdrop.linear = true;
        constexpr float refraction_inset = 1.25f;
        const float target_width = static_cast<float>(impl_->base->size(1));
        const float target_height = static_cast<float>(impl_->base->size(0));
        resetContextRenderState();
        const auto draw_backdrop = [&](const float x1, const float y1,
                                       const float x2, const float y2) {
            if (x2 <= x1 || y2 <= y1)
                return;
            setContextClipRect(x1, y1, x2, y2);
            renderTextureQuad(reinterpret_cast<Rml::TextureHandle>(&backdrop),
                              -refraction_inset, -refraction_inset,
                              target_width + 2.0f * refraction_inset,
                              target_height + 2.0f * refraction_inset);
        };
        std::vector<TensorFrostedGlassRegion> tensor_regions;
        tensor_regions.reserve(regions.size());
        for (const auto& region : regions)
            tensor_regions.push_back({region.x, region.y, region.width, region.height, region.radius});
        for (const auto& rect : frostedGlassClipRects(tensor_regions, target_width, target_height))
            draw_backdrop(rect.left, rect.top, rect.right, rect.bottom);
        resetContextRenderState();
        return true;
    }
    void TensorRmlUiRenderer::beginCacheCapture(int x, int y, int width, int height) {
        if (width <= 0 || height <= 0) {
            endCacheCapture();
            return;
        }
        impl_->capture_area = impl_->clampedRect(float(x), float(y), float(x + width), float(y + height));
    }
    void TensorRmlUiRenderer::endCacheCapture() {
        impl_->capture_area.reset();
    }
    uint64_t TensorRmlUiRenderer::previewTextureGeneration() const {
        return impl_->texture_generation + uiTextureGeneration();
    }
    bool TensorRmlUiRenderer::currentContextUsedPreviewTexture() const { return impl_->preview_used; }
    UiRendererMemoryStatistics TensorRmlUiRenderer::memoryStatistics() const {
        const std::size_t backdrop_bytes = impl_->frosted_glass.bytes();
        std::size_t layer_bytes = 0;
        for (const auto& [_, layer] : impl_->layers)
            layer_bytes += layer->image.bytes();
        for (const auto& layer : impl_->free_layers)
            layer_bytes += layer.bytes();
        auto& profiler = lfs::diagnostics::VramProfiler::instance();
        profiler.recordCurrentBytes("metal.ui", "compiled geometry", impl_->geometry_bytes,
                                    lfs::diagnostics::VramAllocationMethod::Metal);
        profiler.recordCurrentBytes("metal.ui", "textures", impl_->texture_bytes,
                                    lfs::diagnostics::VramAllocationMethod::Metal);
        profiler.recordCurrentBytes("metal.ui", "layers", layer_bytes,
                                    lfs::diagnostics::VramAllocationMethod::Metal);
        profiler.recordCurrentBytes("metal.ui", "frosted glass", backdrop_bytes,
                                    lfs::diagnostics::VramAllocationMethod::Metal);
        return {.block_bytes = impl_->geometry_bytes + impl_->texture_bytes + backdrop_bytes,
                .allocation_bytes = impl_->geometry_bytes + impl_->texture_bytes + backdrop_bytes};
    }
} // namespace lfs::vis::gui
