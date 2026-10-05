/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_selection_query.hpp"

#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_upload.hpp"
#include "splat_selection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <format>
#include <limits>
#include <span>
#include <string>

namespace lfs::rendering {
    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using M = core::GpuKernelModule;
        constexpr auto RW = M::Access::ReadWrite;

        struct Parameters {
            uint64_t pointers[14] = {};
        };
        static_assert(sizeof(Parameters) == 112);

        lfs::Result<void> failure(std::string detail) {
            return lfs::Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument,
                                                          .domain = ErrorDomain::Rendering,
                                                          .detail = std::move(detail),
                                                          .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    } // namespace

    struct SplatSelectionQuery::Impl {
        core::GpuBackend backend;
        std::unique_ptr<M> module;
        Tensor parameter_buffer, transforms, visibility, primitives, polygon_vertices, polygon_mask;
        std::deque<core::TensorUpload> uploads;

        void upload(Tensor& destination, const std::span<const std::byte> bytes) {
            std::erase_if(uploads, [](core::TensorUpload& slot) { return slot.poll(); });
            if (!destination.is_valid() || destination.bytes() != bytes.size())
                destination = Tensor::empty({bytes.size()}, Device::GPU, DataType::UInt8);
            uploads.emplace_back().enqueue_in_batch(destination, bytes);
        }

        void upload(Tensor& destination, const SplatSelectionParameters& parameters) {
            upload(destination, std::as_bytes(std::span(&parameters, 1)));
        }

        void enqueue(Tensor& destination, const std::span<const std::byte> bytes) {
            std::erase_if(uploads, [](core::TensorUpload& slot) { return slot.poll(); });
            uploads.emplace_back().enqueue_in_batch(destination, bytes);
        }
    };

    SplatSelectionQuery::SplatSelectionQuery(const core::GpuBackend backend) : impl_(std::make_unique<Impl>()) {
        impl_->backend = backend;
    }
    SplatSelectionQuery::~SplatSelectionQuery() = default;

    lfs::Result<void> SplatSelectionQuery::query(const SplatSelectionInputs& in,
                                                 const SplatSelectionParameters& input,
                                                 Tensor& output,
                                                 Tensor* ring_pick) {
        auto& s = *impl_;
        const auto& p = input;
        if (!p.image[0] || !p.image[1] || p.image[2] > 2 || p.image[3] > 1 ||
            p.source[1] > 3 || p.payload[0] > 1 || p.payload[1] > 1 || p.scene[1] > 1 ||
            !std::isfinite(p.intrinsics[0]) || !std::isfinite(p.intrinsics[1]) || p.intrinsics[0] <= 0 || p.intrinsics[1] <= 0 ||
            !std::isfinite(p.intrinsics[2]) || !std::isfinite(p.intrinsics[3]) || !std::isfinite(p.ring[0]) || !std::isfinite(p.ring[1]) || p.ring[1] <= 0 ||
            p.scene[3] > p.source[0] || p.scene[0] > uint32_t(std::numeric_limits<int32_t>::max()) ||
            uint64_t(p.aabb[2]) * p.aabb[3] > std::numeric_limits<uint32_t>::max())
            return failure(std::format("Invalid tensor selection parameters (image={}x{}, camera={}, gut={}, source={}, shape={}, payload=({}, {}), objects={}, deleted={}, focal=({}, {}), ring=({}, {}))",
                                       p.image[0], p.image[1], p.image[2], p.image[3], p.source[0], p.source[1], p.payload[0], p.payload[1],
                                       p.scene[0], p.scene[3], p.intrinsics[0], p.intrinsics[1], p.ring[0], p.ring[1]));
        for (size_t col = 0; col < 4; ++col)
            for (size_t row = 0; row < 4; ++row)
                if (!std::isfinite(p.world_to_camera[col * 4 + row]))
                    return failure(std::format("Invalid tensor selection camera matrix (column={}, row={}, value={})",
                                               col, row, p.world_to_camera[col * 4 + row]));

        const bool polygon = p.source[1] == 2, ring = p.source[1] == 3;
        if ((polygon && p.source[3] < 3) || (!polygon && !p.source[2]) ||
            p.aabb[0] > p.image[0] || p.aabb[2] > p.image[0] - p.aabb[0] ||
            p.aabb[1] > p.image[1] || p.aabb[3] > p.image[1] - p.aabb[1])
            return failure(std::format("Invalid tensor selection shape extent (shape={}, primitives={}, vertices={}, origin=({}, {}), extent={}x{}, image={}x{})",
                                       p.source[1], p.source[2], p.source[3], p.aabb[0], p.aabb[1], p.aabb[2], p.aabb[3], p.image[0], p.image[1]));

        const auto validate_tensor = [&](const char* name, const Tensor* tensor, const size_t bytes, const size_t alignment) -> lfs::Result<void> {
            if (!bytes)
                return {};
            const bool valid = tensor && tensor->is_valid();
            const bool gpu = valid && tensor->device() == Device::GPU;
            const bool backend_matches = gpu && core::gpu_backend_of(*tensor) == s.backend;
            const bool contiguous = valid && tensor->is_contiguous();
            const size_t offset = valid ? tensor->storage_offset() * core::dtype_size(tensor->dtype()) : 0;
            const size_t length = valid ? tensor->bytes() : 0;
            const bool aligned = valid && offset % alignment == 0;
            if (!valid || !gpu || !backend_matches || !contiguous || !aligned || length < bytes)
                return failure(std::format("Invalid tensor selection buffer extent or alignment (name={}, offset={}, length={}, required_bytes={}, alignment={}, valid={}, gpu={}, backend_matches={}, contiguous={})",
                                           name, offset, length, bytes, alignment, valid, gpu, backend_matches, contiguous));
            return {};
        };
        const auto validate_span = [&](const char* name, const std::span<const std::byte> bytes, const size_t required) -> lfs::Result<void> {
            if (required && bytes.size() < required)
                return failure(std::format("Invalid tensor selection host buffer extent (name={}, length={}, required_bytes={})",
                                           name, bytes.size(), required));
            return {};
        };

        const size_t count = p.source[0];
        if (auto r = validate_tensor("means", in.means, count * 12, 4); !r)
            return r;
        if (p.image[3] || ring) {
            if (auto r = validate_tensor("log_scales", in.log_scales, count * (p.payload[0] ? 6 : 12), p.payload[0] ? 2 : 4); !r)
                return r;
            if (auto r = validate_tensor("rotations", in.rotations, count * (p.payload[0] ? 8 : 16), p.payload[0] ? 8 : 16); !r)
                return r;
        }
        if (ring)
            if (auto r = validate_tensor("opacity", in.opacity, count * (p.payload[0] ? 2 : 4), p.payload[0] ? 2 : 4); !r)
                return r;
        if (auto r = validate_tensor("deleted", in.deleted, p.scene[3], 1); !r)
            return r;
        if (auto r = validate_span("transforms", in.transforms, size_t(p.scene[0]) * 64); !r)
            return r;
        if (auto r = validate_tensor("transform_indices", in.transform_indices, p.scene[1] ? count * 4 : 0, 4); !r)
            return r;
        if (auto r = validate_span("visibility", in.visibility, p.scene[2]); !r)
            return r;
        if (auto r = validate_span("primitives", in.primitives, polygon ? 0 : size_t(p.source[2]) * 16); !r)
            return r;
        if (auto r = validate_span("polygon_vertices", in.polygon_vertices, polygon ? size_t(p.source[3]) * 8 : 0); !r)
            return r;
        if (auto r = validate_tensor("output", &output, count, 1); !r)
            return r;
        if (auto r = validate_tensor("ring_pick", ring_pick, ring ? 8 : 0, 4); !r)
            return r;
        if (!count)
            return {};

        const core::GpuBackendScope scope(s.backend);
        if (!s.module) {
            auto loaded = M::load(splat_selection_entries(), s.backend);
            if (!loaded)
                return lfs::Result<void>::failure(std::move(loaded).error());
            s.module = std::move(*loaded);
        }
        s.upload(s.parameter_buffer, p);
        const auto stage = [&](Tensor& destination, const std::span<const std::byte> bytes, const size_t required) -> const Tensor* {
            if (!required)
                return nullptr;
            s.upload(destination, bytes.first(required));
            return &destination;
        };
        const auto* transform_buffer = stage(s.transforms, in.transforms, size_t(p.scene[0]) * 64);
        const auto* visibility_buffer = stage(s.visibility, in.visibility, p.scene[2]);
        const auto* primitive_buffer = stage(s.primitives, in.primitives, polygon ? 0 : size_t(p.source[2]) * 16);
        const auto* vertex_buffer = stage(s.polygon_vertices, in.polygon_vertices, polygon ? size_t(p.source[3]) * 8 : 0);
        const size_t mask_bytes = size_t(p.aabb[2]) * p.aabb[3];
        if (polygon && mask_bytes && (!s.polygon_mask.is_valid() || s.polygon_mask.bytes() < mask_bytes))
            s.polygon_mask = Tensor::empty({mask_bytes}, Device::GPU, DataType::UInt8);

        const Parameters parameters;
        const bool geometry = p.image[3] || ring;
        const std::array bindings{
            M::Binding{0, &s.parameter_buffer}, M::Binding{8, in.means}, M::Binding{16, geometry ? in.log_scales : nullptr},
            M::Binding{24, geometry ? in.rotations : nullptr}, M::Binding{32, ring ? in.opacity : nullptr},
            M::Binding{40, p.scene[3] ? in.deleted : nullptr}, M::Binding{48, transform_buffer},
            M::Binding{56, p.scene[1] ? in.transform_indices : nullptr}, M::Binding{64, visibility_buffer},
            M::Binding{72, primitive_buffer}, M::Binding{80, vertex_buffer},
            M::Binding{88, polygon && mask_bytes ? &s.polygon_mask : nullptr, RW}, M::Binding{96, &output, RW},
            M::Binding{104, ring ? ring_pick : nullptr, RW}};
        const auto arguments = M::Arguments{std::as_bytes(std::span(&parameters, 1)), bindings};
        if (polygon && p.aabb[2] && p.aabb[3])
            if (auto r = s.module->dispatch({.function = "polygon_coverage",
                                             .arguments = arguments,
                                             .groups = {M::groups_for(p.aabb[2], 8), M::groups_for(p.aabb[3], 8), 1},
                                             .group = {8, 8, 1}});
                !r)
                return r;
        if (ring) {
            const std::array<uint32_t, 2> empty_pick{0xffffffffu, 0xffffffffu};
            auto pick_bytes = ring_pick->view_as(DataType::UInt8).flatten().slice(0, 0, 8);
            s.enqueue(pick_bytes, std::as_bytes(std::span(empty_pick)));
        }
        for (uint32_t phase = ring ? 1 : 0; phase <= (ring ? 3u : 0u); ++phase) {
            auto phase_parameters = p;
            phase_parameters.payload[2] = phase;
            s.upload(s.parameter_buffer, phase_parameters);
            if (auto r = s.module->dispatch({.function = "selection_query",
                                             .arguments = arguments,
                                             .groups = {M::groups_for(count, 256), 1, 1},
                                             .group = {256, 1, 1}});
                !r)
                return r;
        }
        return {};
    }
} // namespace lfs::rendering
