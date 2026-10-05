/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_projector.hpp"

#include "core/gpu_kernel_module.hpp"
#include "core/tensor_upload.hpp"
#include "splat_project.hpp"

#include <deque>
#include <format>
#include <span>

namespace lfs::rendering {
    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using M = core::GpuKernelModule;
        constexpr auto RW = M::Access::ReadWrite;
        constexpr size_t kSceneObjectBytes = 96, kOverlayParameterBytes = 207 * 16;

        struct Parameters {
            uint64_t pointers[22] = {};
            uint32_t sh_storage = 0, sh_degree = 0, primitive_mode = 0, tight_bounds = 0;
        };
        static_assert(sizeof(Parameters) == 192);

        lfs::Result<void> failure(std::string detail) {
            return lfs::Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument,
                                                     .domain = ErrorDomain::Rendering,
                                                     .detail = std::move(detail),
                                                     .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    } // namespace

    struct SplatProjector::Impl {
        core::GpuBackend backend;
        std::unique_ptr<M> module;
        Tensor frame, layout, objects, overlay_parameters, overlay_flags, node_mask;
        Tensor lod_indices, logical_indices, lod_levels, lod_weights;
        std::deque<core::TensorUpload> uploads;

        void upload(Tensor& destination, const std::span<const std::byte> bytes) {
            std::erase_if(uploads, [](core::TensorUpload& slot) { return slot.poll(); });
            if (!destination.is_valid() || destination.bytes() != bytes.size())
                destination = Tensor::empty({bytes.size()}, Device::GPU, DataType::UInt8);
            uploads.emplace_back().enqueue_in_batch(destination, bytes);
        }
    };

    SplatProjector::SplatProjector(const core::GpuBackend backend) : impl_(std::make_unique<Impl>()) {
        impl_->backend = backend;
    }
    SplatProjector::~SplatProjector() = default;

    lfs::Result<void> SplatProjector::project(const SplatSources& in, const SplatProjection& projection, const uint32_t degree,
                                         const SplatPrimitive primitive, const bool tight_bounds, Tensor& projected, Tensor* gut,
                                         const SplatOverlayInputs* overlay, const SplatLodCut* lod) {
        auto& s = *impl_;
        const auto draw_count = lod ? lod->size : in.count;
        const auto cut_array = [&](const Tensor* tensor) { return !tensor || tensor->bytes() >= size_t(draw_count) * 4; };
        if (degree > 3 || (degree && !in.sh_rest) || !in.means || !in.opacity || !in.sh0 ||
            (lod && (!lod->indices || !cut_array(lod->indices) || !cut_array(lod->logical_indices) || !cut_array(lod->levels) ||
                     !cut_array(lod->weights) || (lod->count && lod->count->bytes() < 4))) ||
            (primitive != SplatPrimitive::Points && (!in.scales || !in.rotations)) ||
            projected.bytes() < size_t(draw_count) * 64 || (primitive == SplatPrimitive::Gut && (!gut || gut->bytes() < size_t(draw_count) * 64)) ||
            (in.storage == SplatShStorage::RadSigned8) != (in.page_splats != 0) || in.page_splats % 32 || in.objects.size() % kSceneObjectBytes ||
            (in.objects.size() > kSceneObjectBytes && !in.object_indices) ||
            (overlay && overlay->parameters.size() != kOverlayParameterBytes))
            return failure(std::format("Splat projection inputs are incomplete (count={}, draw_count={}, degree={}, primitive={}, storage={}, projected_bytes={}, gut_bytes={}, objects={}, overlay_bytes={})",
                                       in.count, draw_count, degree, uint32_t(primitive), uint32_t(in.storage), projected.bytes(), gut ? gut->bytes() : 0,
                                       in.objects.size() / kSceneObjectBytes, overlay ? overlay->parameters.size() : 0));
        if (in.count == 0 || draw_count == 0)
            return {};
        const core::GpuBackendScope scope(s.backend);
        if (!s.module) {
            auto loaded = M::load(splat_project_entries(), s.backend);
            if (!loaded)
                return lfs::Result<void>::failure(std::move(loaded).error());
            s.module = std::move(*loaded);
        }
        const auto object_count = uint32_t(in.objects.size() / kSceneObjectBytes);
        const std::array<uint32_t, 12> layout{in.count, in.layout_rest, in.deleted ? 1u : 0u, object_count,
                                              in.half_attributes ? 1u : 0u, overlay ? 1u : 0u, in.object_indices ? 1u : 0u, draw_count,
                                              lod ? 1u | (lod->logical_indices ? 2u : 0u) | (lod->levels ? 4u : 0u) | (lod->weights ? 8u : 0u) |
                                                        (lod->debug ? 16u : 0u) | (lod->count ? 32u : 0u)
                                                  : 0u,
                                              in.page_splats, lod && lod->logical_count ? lod->logical_count : in.count,
                                              in.deleted_count ? in.deleted_count : in.count};
        s.upload(s.frame, std::as_bytes(std::span(&projection, 1)));
        s.upload(s.layout, std::as_bytes(std::span(layout)));
        if (object_count)
            s.upload(s.objects, in.objects);
        if (overlay) {
            s.upload(s.overlay_parameters, overlay->parameters);
            if (!overlay->node_mask.empty())
                s.upload(s.node_mask, overlay->node_mask);
            if (!s.overlay_flags.is_valid() || s.overlay_flags.numel() < draw_count)
                s.overlay_flags = Tensor::empty({draw_count}, Device::GPU, DataType::UInt32);
        }

        const auto when = [overlay](const Tensor& tensor) { return overlay ? &tensor : nullptr; };
        const Parameters parameters{.sh_storage = uint32_t(in.storage), .sh_degree = degree,
                                    .primitive_mode = uint32_t(primitive), .tight_bounds = tight_bounds ? 1u : 0u};
        const std::array bindings{
            M::Binding{0, in.means}, M::Binding{8, in.scales}, M::Binding{16, in.rotations}, M::Binding{24, in.opacity},
            M::Binding{32, in.sh0}, M::Binding{40, degree ? in.sh_rest : nullptr},
            M::Binding{48, degree && (in.storage == SplatShStorage::Q16 || in.storage == SplatShStorage::RadSigned8) ? in.sh_bounds : nullptr}, M::Binding{56, in.deleted},
            M::Binding{64, &projected, RW}, M::Binding{72, &s.frame}, M::Binding{80, &s.layout},
            M::Binding{88, object_count ? in.object_indices : nullptr}, M::Binding{96, object_count ? &s.objects : nullptr}, M::Binding{104, when(s.overlay_parameters)},
            M::Binding{112, when(s.overlay_flags), RW}, M::Binding{120, overlay && !overlay->node_mask.empty() ? &s.node_mask : nullptr}, M::Binding{128, primitive == SplatPrimitive::Gut ? gut : nullptr, RW},
            M::Binding{136, lod ? lod->indices : nullptr}, M::Binding{144, lod ? lod->logical_indices : nullptr},
            M::Binding{152, lod ? lod->levels : nullptr}, M::Binding{160, lod ? lod->weights : nullptr},
            M::Binding{168, lod ? lod->count : nullptr}};
        return s.module->dispatch({.function = "project_splats",
                                   .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                                   .groups = {M::groups_for(draw_count, 256), 1, 1},
                                   .group = {256, 1, 1}});
    }
    const Tensor& SplatProjector::overlay_parameters() const { return impl_->overlay_parameters; }
    const Tensor& SplatProjector::overlay_flags() const { return impl_->overlay_flags; }
    SplatLodCut SplatProjector::upload_cut(const SplatLodInputs& cut) {
        auto& s = *impl_;
        const core::GpuBackendScope scope(s.backend);
        const auto stage = [&](Tensor& tensor, const std::span<const std::byte> bytes) -> const Tensor* {
            if (bytes.empty())
                return nullptr;
            s.upload(tensor, bytes);
            return &tensor;
        };
        return {.indices = stage(s.lod_indices, std::as_bytes(cut.indices)),
                .logical_indices = stage(s.logical_indices, std::as_bytes(cut.logical_indices)),
                .levels = stage(s.lod_levels, std::as_bytes(cut.levels)),
                .weights = stage(s.lod_weights, std::as_bytes(cut.weights)),
                .size = uint32_t(cut.indices.size()),
                .debug = cut.debug,
                .logical_count = cut.logical_count};
    }
} // namespace lfs::rendering
