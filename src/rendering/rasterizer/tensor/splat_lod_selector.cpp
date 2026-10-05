/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_lod_selector.hpp"

#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_upload.hpp"
#include "scratch_arena.hpp"
#include "splat_lod.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <format>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace lfs::rendering {
    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using M = core::GpuKernelModule;
        constexpr auto RW = M::Access::ReadWrite;

        struct Parameters {
            uint64_t pointers[13] = {};
            uint32_t budget_pass = 0, padding = 0;
        };
        static_assert(sizeof(Parameters) == 112);

        lfs::Result<void> failure(std::string detail) {
            return lfs::Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument,
                                                          .domain = ErrorDomain::Rendering,
                                                          .detail = std::move(detail),
                                                          .detection = LFS_SOURCE_SITE_CURRENT()}));
        }

        std::unique_ptr<M> load(const core::GpuBackend backend) {
            auto loaded = M::load(splat_lod_entries(), backend);
            if (!loaded)
                throw std::runtime_error(format_for_developer(loaded.error()));
            return std::move(*loaded);
        }
    } // namespace

    struct SplatLodSelector::Impl {
        core::GpuBackend backend;
        std::unique_ptr<M> module;
        uint32_t capacity = 0, source_count = 0, chunks = 0;
        Tensor counts, indices, logical_indices, weights, levels, touches, uniforms;
        std::deque<core::TensorUpload> uploads;

        explicit Impl(const core::GpuBackend b) : backend(b), module(load(b)) {}

        void upload(const SplatLodParameters& parameters) {
            std::erase_if(uploads, [](core::TensorUpload& slot) { return slot.poll(); });
            if (!uniforms.is_valid())
                uniforms = Tensor::empty({sizeof(parameters)}, Device::GPU, DataType::UInt8);
            uploads.emplace_back().enqueue_in_batch(uniforms, std::as_bytes(std::span(&parameters, 1)));
        }
    };

    SplatLodSelector::SplatLodSelector(const core::GpuBackend backend) : impl_(std::make_unique<Impl>(backend)) {}
    SplatLodSelector::~SplatLodSelector() = default;

    lfs::Result<void> SplatLodSelector::reserve(const uint32_t capacity, const uint32_t source_count, const uint32_t chunks) {
        auto& s = *impl_;
        if (!capacity || !source_count || !chunks)
            return failure(std::format("A tensor LOD reservation requires nonzero extents (capacity={}, source_count={}, chunks={})",
                                       capacity, source_count, chunks));
        const core::GpuBackendScope scope(s.backend);
        if (!s.counts.is_valid())
            s.counts = Tensor::zeros({8}, Device::GPU, DataType::UInt32);
        if (capacity > s.capacity) {
            std::tie(s.indices, s.logical_indices, s.weights, s.levels) =
                std::tuple_cat(carve_arena<4>({size_t(capacity) * 4, size_t(capacity) * 4,
                                               size_t(capacity) * 4, size_t(capacity) * 4}));
            s.capacity = capacity;
        }
        if (chunks > s.chunks) {
            s.touches = Tensor::zeros({chunks}, Device::GPU, DataType::UInt32);
            s.chunks = chunks;
        }
        s.source_count = std::max(s.source_count, source_count);
        return {};
    }

    lfs::Result<void> SplatLodSelector::select(const SplatLodTree& tree, const SplatLodParameters& input) {
        auto& s = *impl_;
        if (!input.node_count || !input.physical_node_count || input.physical_node_count > s.source_count ||
            !input.output_capacity || input.output_capacity > s.capacity || !input.chunk_splats || input.logical_chunk_count > s.chunks ||
            !std::isfinite(input.pixel_scale_limit) || input.pixel_scale_limit <= 0)
            return failure(std::format("Invalid tensor GPU LOD selection (nodes={}, physical_nodes={}, source_count={}, capacity={}, reserved_capacity={}, chunk_splats={}, logical_chunks={}, reserved_chunks={}, pixel_scale_limit={})",
                                       input.node_count, input.physical_node_count, s.source_count, input.output_capacity,
                                       s.capacity, input.chunk_splats, input.logical_chunk_count, s.chunks, input.pixel_scale_limit));
        const float scalars[] = {input.object_scale, input.behind_camera_penalty, input.cone_foveation,
                                 input.cone_dot0, input.cone_dot, input.cone_blend_denominator, input.cone_tail_valid,
                                 input.outside_view_foveation, input.viewport_half_tan_x, input.viewport_half_tan_y,
                                 input.ortho_half_width, input.ortho_half_height};
        for (const float value : scalars)
            if (!std::isfinite(value) || value < -1e-6f)
                return failure(std::format("Invalid tensor GPU LOD projection parameter (value={})", value));
        for (const auto& row : {input.view_row0, input.view_row1, input.view_row2})
            for (size_t c = 0; c < row.size(); ++c)
                if (!std::isfinite(row[c]))
                    return failure(std::format("Invalid tensor GPU LOD view matrix (column={}, value={})", c, row[c]));
        if (!input.object_scale || input.behind_camera_penalty > 1 || input.cone_foveation > 1 ||
            input.outside_view_foveation > 1 || input.cone_dot0 > 1 || input.cone_dot > input.cone_dot0 ||
            input.viewport_foveation > 1 || input.orthographic > 1 || input.budget_pass ||
            input.logical_chunk_count != (size_t(input.node_count) + input.chunk_splats - 1) / input.chunk_splats)
            return failure(std::format("Invalid tensor GPU LOD traversal parameter (object_scale={}, behind_penalty={}, cone_foveation={}, outside_foveation={}, cone_dot0={}, cone_dot={}, viewport_foveation={}, orthographic={}, budget_pass={}, logical_chunks={}, nodes={}, chunk_splats={})",
                                       input.object_scale, input.behind_camera_penalty, input.cone_foveation,
                                       input.outside_view_foveation, input.cone_dot0, input.cone_dot, input.viewport_foveation,
                                       input.orthographic, input.budget_pass, input.logical_chunk_count,
                                       input.node_count, input.chunk_splats));

        const auto check = [&](const char* name, const Tensor* tensor, const size_t required) -> lfs::Result<void> {
            const bool valid = tensor && tensor->is_valid();
            const bool gpu = valid && tensor->device() == Device::GPU;
            const bool backend_matches = gpu && core::gpu_backend_of(*tensor) == s.backend;
            const bool contiguous = valid && tensor->is_contiguous();
            const bool aligned = valid && tensor->storage_offset() * core::dtype_size(tensor->dtype()) % 4 == 0;
            const size_t length = valid ? tensor->bytes() : 0;
            if (!valid || !gpu || !backend_matches || !contiguous || !aligned || length < required)
                return failure(std::format("Invalid tensor LOD tree buffer (name={}, valid={}, gpu={}, backend_matches={}, contiguous={}, aligned={}, length={}, required_bytes={})",
                                           name, valid, gpu, backend_matches, contiguous, aligned, length, required));
            return {};
        };
        const size_t pages = (size_t(input.physical_node_count) + input.chunk_splats - 1) / input.chunk_splats;
        if (auto r = check("bounds", tree.bounds, size_t(input.physical_node_count) * 8); !r)
            return r;
        if (auto r = check("links", tree.links, size_t(input.physical_node_count) * 12); !r)
            return r;
        if (auto r = check("chunk_to_page", tree.chunk_to_page, size_t(input.logical_chunk_count) * 4); !r)
            return r;
        if (auto r = check("page_to_chunk", tree.page_to_chunk, pages * 4); !r)
            return r;
        if (auto r = check("page_age", tree.page_age, pages * 4); !r)
            return r;
        if (auto r = check("page_frames", tree.page_frames, pages * 64); !r)
            return r;

        const core::GpuBackendScope scope(s.backend);
        s.counts.zero_();
        s.touches.zero_();
        s.upload(input);
        const std::array bindings{
            M::Binding{0, tree.bounds}, M::Binding{8, tree.links}, M::Binding{16, tree.chunk_to_page},
            M::Binding{24, &s.counts, RW}, M::Binding{32, &s.indices, RW}, M::Binding{40, &s.logical_indices, RW},
            M::Binding{48, &s.weights, RW}, M::Binding{56, &s.touches, RW}, M::Binding{64, &s.levels, RW},
            M::Binding{72, tree.page_age}, M::Binding{80, tree.page_frames}, M::Binding{88, tree.page_to_chunk},
            M::Binding{96, &s.uniforms}};
        const auto dispatch = [&](const uint32_t pass, const bool indirect) -> lfs::Result<void> {
            const Parameters parameters{.budget_pass = pass};
            M::Dispatch call{.function = "select_lod",
                             .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                             .groups = {pass ? 1u : M::groups_for(input.physical_node_count, 128), 1, 1},
                             .group = {128, 1, 1}};
            if (indirect) {
                call.indirect = &s.counts;
                call.indirect_offset = 3;
            }
            return s.module->dispatch(call);
        };
        if (auto r = dispatch(0, false); !r)
            return r;
        if (input.output_capacity >= input.physical_node_count)
            return {};
        // Each gate examines the prior cut; retry traversal stays GPU-indirect.
        for (uint32_t pass = 1; pass <= 16; ++pass) {
            if (auto r = dispatch(pass, false); !r)
                return r;
            if (auto r = dispatch(17, true); !r)
                return r;
        }
        return {};
    }

    const Tensor& SplatLodSelector::counts() const { return impl_->counts; }
    const Tensor& SplatLodSelector::indices() const { return impl_->indices; }
    const Tensor& SplatLodSelector::logical_indices() const { return impl_->logical_indices; }
    const Tensor& SplatLodSelector::weights() const { return impl_->weights; }
    const Tensor& SplatLodSelector::levels() const { return impl_->levels; }
    const Tensor& SplatLodSelector::touches() const { return impl_->touches; }
} // namespace lfs::rendering
