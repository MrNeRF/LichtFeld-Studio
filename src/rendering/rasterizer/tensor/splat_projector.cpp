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
        const Tensor* logical_source = nullptr;

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
                                              const SplatOverlayInputs* overlay, const SplatLodInputs* lod) {
        auto& s = *impl_;
        const auto draw_count = lod ? uint32_t(lod->indices.size()) : in.count;
        const auto cut_array = [&](const size_t size) { return size == 0 || size == draw_count; };
        if (degree > 3 || (degree && !in.sh_rest) || !in.means || !in.opacity || !in.sh0 ||
            (lod && (!cut_array(lod->logical_indices.size()) || !cut_array(lod->levels.size()) || !cut_array(lod->weights.size()))) ||
            (primitive != SplatPrimitive::Points && (!in.scales || !in.rotations)) ||
            projected.bytes() < size_t(draw_count) * 64 || (primitive == SplatPrimitive::Gut && (!gut || gut->bytes() < size_t(draw_count) * 64)) ||
            in.storage == SplatShStorage::RadSigned8 || in.objects.size() % kSceneObjectBytes ||
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
                                              lod ? 1u | (lod->logical_indices.empty() ? 0u : 2u) | (lod->levels.empty() ? 0u : 4u) |
                                                        (lod->weights.empty() ? 0u : 8u) | (lod->debug ? 16u : 0u)
                                                  : 0u,
                                              0u, lod && lod->logical_count ? lod->logical_count : in.count,
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
        if (lod) {
            s.upload(s.lod_indices, std::as_bytes(lod->indices));
            if (!lod->logical_indices.empty())
                s.upload(s.logical_indices, std::as_bytes(lod->logical_indices));
            if (!lod->levels.empty())
                s.upload(s.lod_levels, std::as_bytes(lod->levels));
            if (!lod->weights.empty())
                s.upload(s.lod_weights, std::as_bytes(lod->weights));
        }
        s.logical_source = lod ? (lod->logical_indices.empty() ? &s.lod_indices : &s.logical_indices) : nullptr;
        const auto cut = [lod](const Tensor& tensor, const size_t size) { return lod && size ? &tensor : nullptr; };
        const auto when = [overlay](const Tensor& tensor) { return overlay ? &tensor : nullptr; };
        const Parameters parameters{.sh_storage = uint32_t(in.storage), .sh_degree = degree, .primitive_mode = uint32_t(primitive), .tight_bounds = tight_bounds ? 1u : 0u};
        const std::array bindings{
            M::Binding{0, in.means}, M::Binding{8, in.scales}, M::Binding{16, in.rotations}, M::Binding{24, in.opacity},
            M::Binding{32, in.sh0}, M::Binding{40, degree ? in.sh_rest : nullptr},
            M::Binding{48, degree && in.storage == SplatShStorage::Q16 ? in.sh_bounds : nullptr}, M::Binding{56, in.deleted},
            M::Binding{64, &projected, RW}, M::Binding{72, &s.frame}, M::Binding{80, &s.layout},
            M::Binding{88, object_count ? in.object_indices : nullptr}, M::Binding{96, object_count ? &s.objects : nullptr}, M::Binding{104, when(s.overlay_parameters)},
            M::Binding{112, when(s.overlay_flags), RW}, M::Binding{120, overlay && !overlay->node_mask.empty() ? &s.node_mask : nullptr}, M::Binding{128, primitive == SplatPrimitive::Gut ? gut : nullptr, RW},
            M::Binding{136, cut(s.lod_indices, lod ? lod->indices.size() : 0)},
            M::Binding{144, cut(s.logical_indices, lod ? lod->logical_indices.size() : 0)},
            M::Binding{152, cut(s.lod_levels, lod ? lod->levels.size() : 0)},
            M::Binding{160, cut(s.lod_weights, lod ? lod->weights.size() : 0)}, M::Binding{168, nullptr}};
        return s.module->dispatch({.function = "project_splats",
                                   .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                                   .groups = {M::groups_for(draw_count, 256), 1, 1},
                                   .group = {256, 1, 1}});
    }
    const Tensor& SplatProjector::overlay_parameters() const { return impl_->overlay_parameters; }
    const Tensor& SplatProjector::overlay_flags() const { return impl_->overlay_flags; }
    const Tensor& SplatProjector::logical_ids() const {
        static const Tensor none;
        return impl_->logical_source ? *impl_->logical_source : none;
    }
} // namespace lfs::rendering
