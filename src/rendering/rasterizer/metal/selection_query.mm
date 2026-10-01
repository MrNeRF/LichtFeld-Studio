/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "selection_query.hpp"
#include "selection_shader_source.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace lfs::rendering::metal {
    struct SelectionQuery::Impl {
        id<MTLDevice> device;
        id<MTLComputePipelineState> query, polygon;
        id<MTLBuffer> dummy;
    };
    SelectionQuery::SelectionQuery(id<MTLDevice> device) : impl_(std::make_unique<Impl>()) {
        if (!device)
            throw std::invalid_argument("Metal selection query requires a device");
        auto& i = *impl_;
        i.device = device;
        auto options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion2_4;
        // Selection boundaries and GUT signed weights require finite checks
        // and the same safe arithmetic policy as the native raster pipelines.
        if (@available(macOS 15.0, iOS 18.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
            options.fastMathEnabled = NO;
        }
        NSError* error = nil;
        auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSelectionSource] options:options error:&error];
        if (!library)
            throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal selection shader compilation failed");
        i.query = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"selection_query"] error:&error];
        i.polygon = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"polygon_coverage"] error:&error];
        if (!i.query || !i.polygon)
            throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal selection pipeline creation failed");
        i.dummy = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        if (!i.dummy)
            throw std::bad_alloc();
    }
    SelectionQuery::~SelectionQuery() = default;
    void SelectionQuery::encode(id<MTLCommandBuffer> command, const SelectionBuffers& buffers, SelectionParameters p) {
        auto& i = *impl_;
        if (!command || command.commandQueue.device != i.device || !p.image.x || !p.image.y || p.image.z > 2 || p.image.w > 1 ||
            p.source.y > uint32_t(SelectionShape::Ring) || p.payload.x > 1 || p.payload.y > 1 || p.scene.y > 1 ||
            !std::isfinite(p.intrinsics.x) || !std::isfinite(p.intrinsics.y) || p.intrinsics.x <= 0 || p.intrinsics.y <= 0 ||
            !std::isfinite(p.intrinsics.z) || !std::isfinite(p.intrinsics.w) || !std::isfinite(p.ring.x) || !std::isfinite(p.ring.y) || p.ring.y <= 0 ||
            p.scene.w > p.source.x || p.scene.x > uint32_t(std::numeric_limits<int32_t>::max()) ||
            uint64_t(p.aabb.z) * p.aabb.w > std::numeric_limits<uint32_t>::max() || (!p.image.w && p.image.z == 2))
            throw std::invalid_argument("Invalid native Metal selection parameters");
        for (size_t col = 0; col < 4; ++col)
            for (size_t row = 0; row < 4; ++row)
                if (!std::isfinite(p.world_to_camera.columns[col][row]))
                    throw std::invalid_argument("Invalid native Metal selection camera matrix");
        const bool polygon = p.source.y == uint32_t(SelectionShape::Polygon), ring = p.source.y == uint32_t(SelectionShape::Ring);
        if ((polygon && p.source.w < 3) || (!polygon && !p.source.z) || p.aabb.x > p.image.x || p.aabb.z > p.image.x - p.aabb.x ||
            p.aabb.y > p.image.y || p.aabb.w > p.image.y - p.aabb.y)
            throw std::invalid_argument("Invalid native Metal selection shape extent");
        const auto validate = [&](BufferSlice slice, size_t bytes, size_t alignment) {
            if (!bytes)
                return;
            if (!slice.buffer || slice.buffer.device != i.device || slice.offset % alignment || slice.offset > slice.buffer.length || bytes > slice.buffer.length - slice.offset)
                throw std::invalid_argument("Invalid native Metal selection buffer extent or alignment");
        };
        const size_t count = p.source.x;
        validate(buffers.means, count * 12, 4);
        if (p.image.w || ring) {
            validate(buffers.log_scales, count * (p.payload.x ? 6 : 12), p.payload.x ? 2 : 4);
            validate(buffers.rotations, count * (p.payload.x ? 8 : 16), p.payload.x ? 8 : 16);
        }
        if (ring)
            validate(buffers.opacity, count * (p.payload.x ? 2 : 4), p.payload.x ? 2 : 4);
        validate(buffers.deleted, p.scene.w, 1);
        validate(buffers.transforms, size_t(p.scene.x) * 64, 16);
        validate(buffers.transform_indices, p.scene.y ? count * 4 : 0, 4);
        validate(buffers.visibility, p.scene.z, 1);
        validate(buffers.primitives, polygon ? 0 : size_t(p.source.z) * 16, 16);
        validate(buffers.polygon_vertices, polygon ? size_t(p.source.w) * 8 : 0, 4);
        validate(buffers.polygon_mask, polygon ? size_t(p.aabb.z) * p.aabb.w : 0, 1);
        validate(buffers.output, count, 1);
        validate(buffers.ring_pick, ring ? 8 : 0, 4);
        if (!count)
            return;
        const auto bind = [&](id<MTLComputeCommandEncoder> encoder, BufferSlice slice, NSUInteger index) {
            [encoder setBuffer:slice.buffer ?: i.dummy offset:slice.buffer ? slice.offset : 0 atIndex:index];
        };
        if (polygon && p.aabb.z && p.aabb.w) {
            auto encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:i.polygon];
            [encoder setBytes:&p length:sizeof(p) atIndex:0];
            bind(encoder, buffers.polygon_vertices, 1);
            bind(encoder, buffers.polygon_mask, 2);
            [encoder dispatchThreadgroups:MTLSizeMake(p.aabb.z / 8 + (p.aabb.z % 8 != 0), p.aabb.w / 8 + (p.aabb.w % 8 != 0), 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            [encoder endEncoding];
        }
        if (ring) {
            auto clear = [command blitCommandEncoder];
            [clear fillBuffer:buffers.ring_pick.buffer range:NSMakeRange(buffers.ring_pick.offset, 8) value:255];
            [clear endEncoding];
        }
        const std::array<BufferSlice, 12> slices{buffers.means, buffers.log_scales, buffers.rotations, buffers.opacity, buffers.deleted,
                                                 buffers.transforms, buffers.transform_indices, buffers.visibility, buffers.primitives,
                                                 buffers.polygon_mask, buffers.output, buffers.ring_pick};
        for (uint32_t phase = ring ? 1 : 0; phase <= (ring ? 3u : 0u); ++phase) {
            p.payload.z = phase;
            auto encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:i.query];
            [encoder setBytes:&p length:sizeof(p) atIndex:0];
            for (size_t n = 0; n < slices.size(); ++n)
                bind(encoder, slices[n], n + 1);
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
        }
    }
} // namespace lfs::rendering::metal
