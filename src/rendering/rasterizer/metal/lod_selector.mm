/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lod_selector.hpp"
#include "core/memory_pressure.hpp"
#include "frame_budget.hpp"
#include "lod_shader_source.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <format>
#include <stdexcept>

namespace lfs::rendering::metal {
    namespace {
        void check(BufferSlice slice, size_t bytes, id<MTLDevice> device) {
            if (!slice.buffer || slice.buffer.device != device || slice.offset % 4 ||
                slice.offset > slice.buffer.length || bytes > slice.buffer.length - slice.offset)
                throw std::invalid_argument(std::format("Invalid native LOD tree buffer (offset={}, length={}, required_bytes={}, same_device={})", slice.offset, slice.buffer.length, bytes, slice.buffer.device == device));
        }
        id<MTLBuffer> allocate(id<MTLDevice> device, size_t size, MTLResourceOptions storage) {
            if (!frameFitsWorkingSet(device.currentAllocatedSize, size, device.recommendedMaxWorkingSetSize))
                throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = size, .label = "viewer.lod", .operation = "lod.buffer.allocate"});
            auto buffer = [device newBufferWithLength:std::max<size_t>(16, size) options:storage];
            if (!buffer)
                throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = size, .label = "viewer.lod", .operation = "lod.buffer.allocate"});
            return buffer;
        }
    } // namespace
    struct LodCutFrame::Impl {
        id<MTLDevice> device;
        uint32_t capacity, source_count, chunks;
        id<MTLBuffer> counts, indices, logical, weights, levels, touches;
        std::atomic_bool in_flight{false}, completed{false};
    };
    LodCutFrame::LodCutFrame(id<MTLDevice> device, uint32_t capacity, uint32_t source_count, uint32_t chunks)
        : impl_(std::make_shared<Impl>()) {
        if (!device || !capacity || !source_count || !chunks)
            throw std::invalid_argument(std::format("Invalid Metal LOD cut capacity (device_present={}, capacity={}, source_count={}, chunks={})", device != nil, capacity, source_count, chunks));
        auto& f = *impl_;
        f.device = device;
        f.capacity = capacity;
        f.source_count = source_count;
        f.chunks = chunks;
        f.counts = allocate(device, 32, MTLResourceStorageModeShared);
        f.indices = allocate(device, size_t(capacity) * 4, MTLResourceStorageModePrivate);
        f.logical = allocate(device, size_t(capacity) * 4, MTLResourceStorageModePrivate);
        f.weights = allocate(device, size_t(capacity) * 4, MTLResourceStorageModePrivate);
        f.levels = allocate(device, size_t(capacity) * 4, MTLResourceStorageModePrivate);
        f.touches = allocate(device, size_t(chunks) * 4, MTLResourceStorageModeShared);
    }
    LodCutFrame::~LodCutFrame() = default;
    LodSelection LodCutFrame::selection(bool debug) const {
        const auto& f = *impl_;
        return {{f.indices}, {f.logical}, {f.levels}, {f.weights}, f.capacity, f.source_count, true, debug, {f.counts}};
    }
    bool LodCutFrame::busy() const { return impl_->in_flight.load(std::memory_order_acquire); }
    LodCutStatus LodCutFrame::status() const {
        if (busy() || !impl_->completed.load(std::memory_order_acquire))
            throw std::logic_error(std::format("Metal LOD status requires successful GPU completion (busy={}, completed={})", busy(), impl_->completed.load(std::memory_order_acquire)));
        const auto data = static_cast<const uint32_t*>(impl_->counts.contents);
        return {data[0], data[1], std::bit_cast<float>(data[2])};
    }
    BufferSlice LodCutFrame::touches() const { return {impl_->touches}; }
    struct LodSelector::Impl {
        id<MTLDevice> device;
        id<MTLComputePipelineState> pipeline;
    };
    LodSelector::LodSelector(id<MTLDevice> device) : impl_(std::make_unique<Impl>()) {
        if (!device)
            throw std::invalid_argument(std::format("Metal LOD selector requires a device (device_present={})", device != nil));
        impl_->device = device;
        auto options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion2_4;
        if (@available(macOS 15.0, iOS 18.0, *))
            options.mathMode = MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError* error = nil;
        auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalLodSelectorSource] options:options error:&error];
        if (!library)
            throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal LOD shader compilation failed");
        impl_->pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"select_lod"] error:&error];
        if (!impl_->pipeline)
            throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal LOD pipeline creation failed");
    }
    LodSelector::~LodSelector() = default;
    void LodSelector::encode(id<MTLCommandBuffer> command, const LodTreeBuffers& tree,
                             const LodParameters& input, LodCutFrame& frame) {
        const auto f = frame.impl_;
        if (!command || command.device != impl_->device || f->device != impl_->device ||
            command.status != MTLCommandBufferStatusNotEnqueued || !input.node_count || !input.physical_node_count ||
            input.physical_node_count > f->source_count || input.output_capacity != f->capacity || !input.chunk_splats ||
            input.logical_chunk_count != f->chunks || !std::isfinite(input.pixel_scale_limit) || input.pixel_scale_limit <= 0)
            throw std::invalid_argument(std::format("Invalid native GPU LOD selection (command_status={}, command_device_matches={}, frame_device_matches={}, nodes={}, physical_nodes={}, source_count={}, capacity={}, reserved_capacity={}, chunk_splats={}, logical_chunks={}, reserved_chunks={}, pixel_scale_limit={})", long(command.status), command.device == impl_->device, f->device == impl_->device, input.node_count, input.physical_node_count, f->source_count, input.output_capacity, f->capacity, input.chunk_splats, input.logical_chunk_count, f->chunks, input.pixel_scale_limit));
        const float scalars[] = {input.object_scale, input.behind_camera_penalty, input.cone_foveation,
                                 input.cone_dot0, input.cone_dot, input.cone_blend_denominator, input.cone_tail_valid,
                                 input.outside_view_foveation, input.viewport_half_tan_x, input.viewport_half_tan_y,
                                 input.ortho_half_width, input.ortho_half_height};
        for (float value : scalars)
            if (!std::isfinite(value) || value < -1e-6f)
                throw std::invalid_argument(std::format("Invalid native GPU LOD projection parameter (value={})", value));
        for (const auto row : {input.view_row0, input.view_row1, input.view_row2})
            for (size_t c = 0; c < 4; ++c)
                if (!std::isfinite(row[c]))
                    throw std::invalid_argument(std::format("Invalid native GPU LOD view matrix (column={}, value={})", c, float(row[c])));
        if (!input.object_scale || input.behind_camera_penalty > 1 || input.cone_foveation > 1 ||
            input.outside_view_foveation > 1 || input.cone_dot0 > 1 || input.cone_dot > input.cone_dot0 ||
            input.viewport_foveation > 1 || input.orthographic > 1 || input.budget_pass ||
            input.logical_chunk_count != (size_t(input.node_count) + input.chunk_splats - 1) / input.chunk_splats)
            throw std::invalid_argument(std::format("Invalid native GPU LOD traversal parameter (object_scale={}, behind_penalty={}, cone_foveation={}, outside_foveation={}, cone_dot0={}, cone_dot={}, viewport_foveation={}, orthographic={}, budget_pass={}, logical_chunks={}, nodes={}, chunk_splats={})", input.object_scale, input.behind_camera_penalty, input.cone_foveation, input.outside_view_foveation, input.cone_dot0, input.cone_dot, input.viewport_foveation, input.orthographic, input.budget_pass, input.logical_chunk_count, input.node_count, input.chunk_splats));
        const size_t pages = (size_t(input.physical_node_count) + input.chunk_splats - 1) / input.chunk_splats;
        check(tree.bounds, size_t(input.physical_node_count) * 8, impl_->device);
        check(tree.links, size_t(input.physical_node_count) * 12, impl_->device);
        check(tree.chunk_to_page, size_t(f->chunks) * 4, impl_->device);
        check(tree.page_to_chunk, pages * 4, impl_->device);
        check(tree.page_age, pages * 4, impl_->device);
        check(tree.page_frames, pages * 64, impl_->device);
        if (f->in_flight.exchange(true, std::memory_order_acq_rel))
            throw std::logic_error(std::format("Metal LOD reservation is still in flight (capacity={}, source_count={}, command_status={})", f->capacity, f->source_count, long(command.status)));
        f->completed.store(false, std::memory_order_release);
        [command addCompletedHandler:^(id<MTLCommandBuffer> finished) {
            f->completed.store(finished.status == MTLCommandBufferStatusCompleted, std::memory_order_release);
            f->in_flight.store(false, std::memory_order_release);
        }];
        auto clear = [command blitCommandEncoder];
        if (!clear)
            throw std::runtime_error(std::format("Metal LOD clear encoder unavailable (command_status={}, counts_bytes={}, touches_bytes={})", long(command.status), f->counts.length, f->touches.length));
        [clear fillBuffer:f->counts range:NSMakeRange(0, f->counts.length) value:0];
        [clear fillBuffer:f->touches range:NSMakeRange(0, f->touches.length) value:0];
        [clear endEncoding];
        const std::array<BufferSlice, 12> buffers = {tree.bounds, tree.links, tree.chunk_to_page, {f->counts}, {f->indices}, {f->logical}, {f->weights}, {f->touches}, {f->levels}, tree.page_age, tree.page_frames, tree.page_to_chunk};
        const auto dispatch = [&](uint32_t pass, bool indirect) {
            auto p = input;
            p.budget_pass = pass;
            auto encoder = [command computeCommandEncoder];
            if (!encoder)
                throw std::runtime_error(std::format("Metal LOD compute encoder unavailable (command_status={}, pass={}, indirect={})", long(command.status), pass, indirect));
            encoder.label = @"LichtFeld GPU LOD threshold selection";
            [encoder setComputePipelineState:impl_->pipeline];
            [encoder setBytes:&p length:sizeof(p) atIndex:0];
            for (NSUInteger n = 0; n < buffers.size(); ++n)
                [encoder setBuffer:buffers[n].buffer offset:buffers[n].offset atIndex:n + 1];
            if (indirect)
                [encoder dispatchThreadgroupsWithIndirectBuffer:f->counts indirectBufferOffset:12 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            else
                [encoder dispatchThreadgroups:MTLSizeMake(pass ? (1u) : (input.physical_node_count + 127ull) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        };
        dispatch(0, false);
        if (input.output_capacity >= input.physical_node_count)
            return; // One append per resident node cannot overflow.
        // Each gate examines the completed prior cut. A retry dispatch is GPU
        // indirect (zero groups when complete), and the final threshold retains
        // roots. No count readback, CPU traversal or host queue wait.
        for (uint32_t pass = 1; pass <= 16; ++pass) {
            dispatch(pass, false);
            dispatch(17, true);
        }
    }
} // namespace lfs::rendering::metal
