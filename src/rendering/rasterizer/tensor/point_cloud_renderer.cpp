/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "point_cloud_renderer.hpp"

#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_upload.hpp"
#include "point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>
#include <format>
#include <initializer_list>
#include <limits>

namespace lfs::rendering {
    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using M = core::GpuKernelModule;

        constexpr size_t kSceneObjectBytes = 96;
        constexpr size_t kPaletteEntries = 257;

        struct DrawParameters {
            uint64_t pointers[11]{};
            std::array<uint32_t, 2> extent{};
            std::array<uint32_t, 2> padding{};
        };
        static_assert(sizeof(DrawParameters) == 104);
        static_assert(offsetof(DrawParameters, extent) == 88);

        struct InlineDrawParameters {
            uint64_t pointers[10]{};
            PointParameters point;
            std::array<uint32_t, 2> extent{};
            std::array<uint32_t, 2> padding{};
        };
        static_assert(sizeof(InlineDrawParameters) == 352);
        static_assert(offsetof(InlineDrawParameters, point) == 80);
        static_assert(offsetof(InlineDrawParameters, extent) == 336);

        lfs::Result<void> failure(std::string detail) {
            return lfs::Result<void>::failure(lfs::make_error({
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Rendering,
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
    } // namespace

    struct SplatPointRenderer::Impl {
        core::GpuBackend backend;
        std::unique_ptr<M> module;
        bool buffered_parameters = false;
        Tensor objects, palette, point_parameters;
        Tensor rgba, raster_depth, depth_rgba, linear;
        std::deque<core::TensorUpload> uploads;

        explicit Impl(const core::GpuBackend value) : backend(value) {}

        void upload(Tensor& destination, const std::span<const std::byte> bytes) {
            std::erase_if(uploads, [](core::TensorUpload& slot) { return slot.poll(); });
            if (!destination.is_valid() || destination.bytes() != bytes.size())
                destination = Tensor::empty({bytes.size()}, Device::GPU, DataType::UInt8);
            uploads.emplace_back().enqueue_in_batch(destination, bytes);
        }

        void reserve(const uint32_t width, const uint32_t height) {
            const auto wrong = [&](const Tensor& tensor, const std::initializer_list<size_t> shape, const DataType dtype) {
                if (!tensor.is_valid() || tensor.dtype() != dtype || core::gpu_backend_of(tensor) != backend || tensor.ndim() != shape.size())
                    return true;
                size_t axis = 0;
                for (const auto size : shape)
                    if (tensor.size(axis++) != size)
                        return true;
                return false;
            };
            if (wrong(rgba, {height, width, 4}, DataType::UInt8))
                rgba = Tensor::empty({height, width, 4}, Device::GPU, DataType::UInt8);
            if (wrong(raster_depth, {height, width}, DataType::Float32))
                raster_depth = Tensor::empty({height, width}, Device::GPU, DataType::Float32);
            if (wrong(depth_rgba, {height, width, 4}, DataType::Float32))
                depth_rgba = Tensor::empty({height, width, 4}, Device::GPU, DataType::Float32);
            if (wrong(linear, {height, width}, DataType::Float32))
                linear = Tensor::empty({height, width}, Device::GPU, DataType::Float32);
        }
    };

    SplatPointRenderer::SplatPointRenderer(const core::GpuBackend backend)
        : impl_(std::make_unique<Impl>(backend)) {}
    SplatPointRenderer::~SplatPointRenderer() = default;

    lfs::Result<void> SplatPointRenderer::render(const SplatPointInputs& in,
                                                 const PointParameters& point,
                                                 const uint32_t width, const uint32_t height,
                                                 const std::array<float, 4> background) {
        auto& s = *impl_;
        const auto tensor_valid = [&](const Tensor* tensor) {
            return tensor && tensor->is_valid() && tensor->device() == Device::GPU &&
                   core::gpu_backend_of(*tensor) == s.backend && tensor->is_contiguous();
        };
        if (!tensor_valid(in.positions) || !tensor_valid(in.colors) ||
            in.positions->dtype() != DataType::Float32 || in.colors->dtype() != DataType::Float32 ||
            in.positions->ndim() != 2 || in.colors->ndim() != 2 ||
            in.positions->size(1) != 3 || in.colors->size(1) != 3 ||
            in.positions->size(0) != in.colors->size(0))
            return failure(std::format("Point rendering needs contiguous {} Float32 [N,3] positions/colors on one backend (positions={}, colors={})",
                                       core::gpu_backend_name(s.backend),
                                       in.positions ? in.positions->shape().str() : "missing",
                                       in.colors ? in.colors->shape().str() : "missing"));

        const size_t count = in.positions->size(0);
        if (!width || !height || count > std::numeric_limits<uint32_t>::max())
            return failure(std::format("Point rendering extent/count is unsupported (extent={}x{}, points={}, maximum={})",
                                       width, height, count, std::numeric_limits<uint32_t>::max()));
        const auto mask_valid = [&](const Tensor* tensor) {
            return !tensor || !tensor->is_valid() ||
                   (tensor_valid(tensor) && tensor->bytes() >= count &&
                    (tensor->dtype() == DataType::UInt8 || tensor->dtype() == DataType::Bool));
        };
        if (!mask_valid(in.selection) || !mask_valid(in.preview) || !mask_valid(in.deleted))
            return failure(std::format("Point masks must be contiguous byte tensors with at least {} entries on {}",
                                       count, core::gpu_backend_name(s.backend)));
        if (in.transform_indices && in.transform_indices->is_valid() &&
            (!tensor_valid(in.transform_indices) || in.transform_indices->dtype() != DataType::Int32 ||
             in.transform_indices->bytes() < count * sizeof(int32_t)))
            return failure(std::format("Point transform indices must be contiguous Int32 with at least {} entries on {}",
                                       count, core::gpu_backend_name(s.backend)));
        const uint32_t flags = point.counts[2];
        if (((flags & 16u) && (!in.transform_indices || !in.transform_indices->is_valid())) ||
            ((flags & 32u) && (!in.selection || !in.selection->is_valid())) ||
            ((flags & 64u) && (!in.preview || !in.preview->is_valid())) ||
            ((flags & 512u) && (!in.deleted || !in.deleted->is_valid())))
            return failure(std::format("Point parameter flags require missing tensor inputs (flags={}, indices={}, selection={}, preview={}, deleted={})",
                                       flags, in.transform_indices && in.transform_indices->is_valid(),
                                       in.selection && in.selection->is_valid(), in.preview && in.preview->is_valid(),
                                       in.deleted && in.deleted->is_valid()));
        if (in.objects.size() % kSceneObjectBytes || in.objects.size() / kSceneObjectBytes != point.counts[0])
            return failure(std::format("Point scene objects do not match parameters (bytes={}, record_bytes={}, records={}, parameter_count={})",
                                       in.objects.size(), kSceneObjectBytes, in.objects.size() / kSceneObjectBytes, point.counts[0]));
        const bool selection_enabled = (flags & (32u | 64u)) != 0u;
        if (selection_enabled && in.selection_palette.size() < kPaletteEntries * sizeof(std::array<float, 4>))
            return failure(std::format("Point selection needs at least {} float4 palette records (bytes={}, required={})",
                                       kPaletteEntries, in.selection_palette.size(), kPaletteEntries * sizeof(std::array<float, 4>)));
        if (!point.counts[3])
            return failure("Point maximum size must be nonzero");

        const core::GpuBackendScope scope(s.backend);
        if (!s.module) {
            auto loaded = M::load(point_cloud_entries(), s.backend);
            if (!loaded)
                return lfs::Result<void>::failure(std::move(loaded).error());
            s.module = std::move(*loaded);
            for (const auto& entry : point_cloud_entries())
                if (entry.backend == s.backend && entry.name == "pointVertex")
                    s.buffered_parameters = entry.parameter_bytes == sizeof(DrawParameters);
        }
        s.reserve(width, height);
        if (!in.objects.empty())
            s.upload(s.objects, in.objects);
        if (selection_enabled)
            s.upload(s.palette, in.selection_palette);

        const bool buffered = s.buffered_parameters;
        if (buffered)
            s.upload(s.point_parameters, std::as_bytes(std::span(&point, 1)));
        DrawParameters parameters{.extent = {width, height}};
        InlineDrawParameters inline_parameters{.point = point, .extent = {width, height}};
        const auto parameter_bytes = buffered ? std::as_bytes(std::span(&parameters, 1))
                                              : std::as_bytes(std::span(&inline_parameters, 1));
        const auto present = [](const Tensor* tensor) { return tensor && tensor->is_valid() ? tensor : nullptr; };
        const std::array draw_bindings{
            M::Binding{0, count ? in.positions : nullptr},
            M::Binding{8, count ? in.colors : nullptr},
            M::Binding{16, in.objects.empty() ? nullptr : &s.objects},
            M::Binding{24, present(in.transform_indices)},
            M::Binding{32, present(in.selection)},
            M::Binding{40, present(in.preview)},
            M::Binding{48, selection_enabled ? &s.palette : nullptr},
            M::Binding{56, present(in.deleted)},
            M::Binding{64, nullptr},
            M::Binding{72, nullptr},
            M::Binding{80, &s.point_parameters},
        };
        auto color_draw = M::Draw{
            .vertex = "pointVertex",
            .fragment = "pointColorFragment",
            .arguments = {parameter_bytes, std::span(draw_bindings).first(buffered ? 11 : 10)},
            .color = &s.rgba,
            .depth = &s.raster_depth,
            .vertex_count = 6,
            .instance_count = uint32_t(count),
            .depth_compare = M::Compare::Less,
            .depth_write = true,
            .clear_color = true,
            .color_clear = background,
            .clear_depth = true,
            .depth_clear = 1.0f,
        };
        if (auto result = s.module->draw(color_draw); !result)
            return result;

        auto depth_draw = color_draw;
        depth_draw.fragment = "pointDepthFragment";
        depth_draw.color = &s.depth_rgba;
        depth_draw.depth_compare = M::Compare::LessEqual;
        depth_draw.depth_write = false;
        depth_draw.color_clear = {-1.0f, 0.0f, 0.0f, 0.0f};
        depth_draw.clear_depth = false;
        if (auto result = s.module->draw(depth_draw); !result)
            return result;

        const std::array copy_bindings{
            M::Binding{0, nullptr},
            M::Binding{8, nullptr},
            M::Binding{16, nullptr},
            M::Binding{24, nullptr},
            M::Binding{32, nullptr},
            M::Binding{40, nullptr},
            M::Binding{48, nullptr},
            M::Binding{56, nullptr},
            M::Binding{64, &s.depth_rgba},
            M::Binding{72, &s.linear, M::Access::ReadWrite},
            M::Binding{80, nullptr},
        };
        return s.module->dispatch({
            .function = "extractPointDepth",
            .arguments = {parameter_bytes, std::span(copy_bindings).first(buffered ? 11 : 10)},
            .groups = {(width + 15u) / 16u, (height + 15u) / 16u, 1},
            .group = {16, 16, 1},
        });
    }

    const core::Tensor& SplatPointRenderer::color() const { return impl_->rgba; }
    const core::Tensor& SplatPointRenderer::linear_depth() const { return impl_->linear; }
} // namespace lfs::rendering
