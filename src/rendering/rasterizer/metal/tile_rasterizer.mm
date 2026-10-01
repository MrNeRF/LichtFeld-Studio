/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "tile_rasterizer.hpp"
#include "tile_shader_source.hpp"
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace lfs::rendering::metal {
    namespace {
        uint32_t ceil_div(uint32_t n, uint32_t d) { return n / d + (n % d != 0); }
        struct alignas(16) RasterParameters {
            uint32_t count, width, height, columns, tiles, capacity, mode, unused;
            simd_float4 background;
            simd_float4 render_origin;
            simd_float4 intrinsics, clip;
            simd_uint4 camera;
            simd_float4 panorama;
            simd_uint4 mask_limits;
        };
        static_assert(sizeof(RasterParameters) == 144);
        struct SortParameters {
            uint32_t blocks, shift;
        };
        id<MTLBuffer> allocate(id<MTLDevice> device, size_t bytes,
                               MTLResourceOptions options = MTLResourceStorageModePrivate) {
            bytes = std::max(bytes, sizeof(ProjectedSplat));
            if (bytes > device.maxBufferLength)
                throw std::length_error("Metal viewer buffer exceeds device limit");
            auto buffer = [device newBufferWithLength:bytes options:options];
            if (!buffer)
                throw std::runtime_error("Metal viewer GPU allocation failed");
            return buffer;
        }
        id<MTLTexture> texture(id<MTLDevice> device, uint32_t width, uint32_t height, MTLPixelFormat format) {
            auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                 width:width
                                                                                height:height
                                                                             mipmapped:NO];
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            auto result = [device newTextureWithDescriptor:descriptor];
            if (!result)
                throw std::runtime_error("Metal viewer texture allocation failed");
            return result;
        }
        struct ScanLevel {
            id<MTLBuffer> sums, offsets;
            uint32_t count;
        };
        using ScanStorage = std::vector<ScanLevel>;
        ScanStorage reserve_scan(id<MTLDevice> device, uint32_t n, size_t element_bytes = 8) {
            ScanStorage levels;
            for (;;) {
                const auto groups = ceil_div(n, 256);
                levels.push_back({allocate(device, size_t(groups) * element_bytes), allocate(device, size_t(groups) * element_bytes), n});
                if (groups <= 1)
                    break;
                n = groups;
            }
            return levels;
        }
    } // namespace

    struct RasterFrame::Impl {
        id<MTLDevice> device;
        uint32_t width, height, max_splats, capacity, tiles, columns, sort_blocks;
        id<MTLBuffer> counts, offsets, status, histogram, histogram_offsets, ranges, dispatch_args;
        std::array<id<MTLBuffer>, 2> keys, indices;
        id<MTLTexture> color, depth, pick;
        ScanStorage count_scan, histogram_scan;
        std::atomic_bool in_flight{false};
        std::atomic_bool completed{false};
    };
    RasterFrame::RasterFrame(id<MTLDevice> device, uint32_t width, uint32_t height,
                             uint32_t max_splats, uint32_t max_instances) : impl_(std::make_shared<Impl>()) {
        if (!device || !width || !height || width > 16384 || height > 16384 || !max_instances)
            throw std::invalid_argument("Invalid Metal viewer frame reservation");
        auto& f = *impl_;
        f.device = device;
        f.width = width;
        f.height = height;
        f.max_splats = max_splats;
        f.capacity = max_instances;
        f.columns = ceil_div(width, 16);
        f.tiles = f.columns * ceil_div(height, 16);
        f.sort_blocks = ceil_div(max_instances, 2048);
        f.counts = allocate(device, size_t(max_splats) * 8);
        f.offsets = allocate(device, size_t(max_splats) * 8);
        f.status = allocate(device, sizeof(RasterStatus), MTLResourceStorageModeShared);
        f.dispatch_args = allocate(device, 6 * sizeof(uint32_t));
        f.ranges = allocate(device, size_t(f.tiles) * 8);
        for (int i = 0; i < 2; ++i) {
            f.keys[i] = allocate(device, size_t(max_instances) * 8);
            f.indices[i] = allocate(device, size_t(max_instances) * 4);
        }
        const uint32_t histogram_size = f.sort_blocks * 256;
        f.histogram = allocate(device, size_t(histogram_size) * 4);
        f.histogram_offsets = allocate(device, size_t(histogram_size) * 4);
        f.count_scan = reserve_scan(device, max_splats);
        f.histogram_scan = reserve_scan(device, histogram_size, 4);
        f.color = texture(device, width, height, MTLPixelFormatRGBA16Float);
        f.depth = texture(device, width, height, MTLPixelFormatRGBA32Float);
        f.pick = texture(device, width, height, MTLPixelFormatR32Uint);
    }
    RasterFrame::~RasterFrame() = default;
    bool RasterFrame::busy() const { return impl_->in_flight.load(std::memory_order_acquire); }
    RasterStatus RasterFrame::status() const {
        if (busy() || !impl_->completed.load(std::memory_order_acquire))
            throw std::logic_error("Metal viewer status requires successful GPU completion");
        return *static_cast<const RasterStatus*>(impl_->status.contents);
    }
    id<MTLTexture> RasterFrame::color() const { return impl_->color; }
    id<MTLTexture> RasterFrame::depth() const { return impl_->depth; }
    id<MTLTexture> RasterFrame::pick() const { return impl_->pick; }
    id<MTLBuffer> RasterFrame::statusBuffer() const { return impl_->status; }

    struct TileRasterizer::Impl {
        id<MTLDevice> device;
        std::map<std::string, id<MTLComputePipelineState>> pipelines;
        id<MTLLibrary> library;
        std::mutex blend_mutex;
        std::map<std::pair<uint32_t, uint32_t>, id<MTLComputePipelineState>> blend_pipelines;
        id<MTLComputePipelineState> blendPipeline(uint32_t mode, uint32_t flags) {
            std::lock_guard lock(blend_mutex);
            const auto key = std::pair{mode, flags};
            if (const auto found = blend_pipelines.find(key); found != blend_pipelines.end())
                return found->second;
            auto constants = [MTLFunctionConstantValues new];
            [constants setConstantValue:&mode type:MTLDataTypeUInt atIndex:0];
            [constants setConstantValue:&flags type:MTLDataTypeUInt atIndex:1];
            NSError* error = nil;
            auto function = [library newFunctionWithName:@"tile_blend" constantValues:constants error:&error];
            if (!function)
                throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal blend specialization failed");
            auto state = [device newComputePipelineStateWithFunction:function error:&error];
            if (!state)
                throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal blend pipeline failed");
            if (state.threadExecutionWidth != 32 || state.maxTotalThreadsPerThreadgroup < 64)
                throw std::runtime_error("Metal blend specialization requires SIMD32 and 64-thread groups");
            blend_pipelines.emplace(key, state);
            return state;
        }
        id<MTLComputeCommandEncoder> begin(id<MTLCommandBuffer> command, const char* name) {
            auto encoder = [command computeCommandEncoder];
            if (!encoder)
                throw std::runtime_error("Could not encode Metal viewer compute pass");
            encoder.label = [NSString stringWithUTF8String:name];
            [encoder setComputePipelineState:pipelines.at(name)];
            return encoder;
        }
        void scan(id<MTLCommandBuffer> command, id<MTLBuffer> input, id<MTLBuffer> output,
                  uint32_t count, const ScanStorage& storage, bool narrow = false, size_t level = 0) {
            if (!count)
                return;
            const auto& scratch = storage.at(level);
            const auto groups = ceil_div(count, 256);
            auto encoder = begin(command, narrow ? "scan32_blocks" : "scan_blocks");
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBuffer:output offset:0 atIndex:1];
            [encoder setBuffer:scratch.sums offset:0 atIndex:2];
            [encoder setBytes:&count length:4 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            if (groups <= 1)
                return;
            scan(command, scratch.sums, scratch.offsets, groups, storage, narrow, level + 1);
            encoder = begin(command, narrow ? "scan32_add" : "scan_add");
            [encoder setBuffer:output offset:0 atIndex:0];
            [encoder setBuffer:scratch.offsets offset:0 atIndex:1];
            [encoder setBytes:&count length:4 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
        }
    };
    TileRasterizer::TileRasterizer(id<MTLDevice> device) : impl_(std::make_unique<Impl>()) {
        if (!device)
            throw std::invalid_argument("Metal viewer requires a device");
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
        auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:kTileRasterizerSource]
                                            options:options
                                              error:&error];
        if (!library)
            throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal tile shader compilation failed");
        impl_->library = library;
        for (const char* name : {"tile_counts", "scan_blocks", "scan_add", "scan32_blocks", "scan32_add", "tile_status", "tile_instances",
                                 "tile_histogram", "tile_scatter", "tile_ranges"}) {
            auto function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            auto state = [device newComputePipelineStateWithFunction:function error:&error];
            if (!state)
                throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal tile pipeline failed");
            if (state.threadExecutionWidth != 32 || state.maxTotalThreadsPerThreadgroup < 256)
                throw std::runtime_error("Metal viewer requires SIMD32 and 256-thread groups");
            impl_->pipelines.emplace(name, state);
        }
    }
    TileRasterizer::~TileRasterizer() = default;
    void TileRasterizer::encode(id<MTLCommandBuffer> command, BufferSlice projected, uint32_t count,
                                RasterMode mode, simd_float4 background, RasterFrame& frame, const OverlayBuffers& overlay, BufferSlice gut, const Projection& projection, const LodSelection& lod, bool omit_saturating_color) {
        auto f = frame.impl_;
        if (!command || command.device != impl_->device || f->device != impl_->device ||
            command.status != MTLCommandBufferStatusNotEnqueued || count > f->max_splats || uint32_t(mode) > 3)
            throw std::invalid_argument("Invalid Metal viewer frame submission");
        for (int i = 0; i < 4; ++i)
            if (!std::isfinite(background[i]) || background[i] < 0 || (i == 3 && background[i] > 1))
                throw std::invalid_argument("Invalid Metal viewer background");
        if (count && (!projected.buffer || projected.buffer.device != impl_->device || projected.offset % 16 ||
                      projected.offset > projected.buffer.length ||
                      size_t(count) * sizeof(ProjectedSplat) > projected.buffer.length - projected.offset))
            throw std::invalid_argument("Invalid Metal viewer projected splat buffer");
        if (mode == RasterMode::Gut && count &&
            (!gut.buffer || gut.buffer.device != impl_->device || gut.offset % 16 ||
             gut.offset > gut.buffer.length || size_t(count) * sizeof(GutSplat) > gut.buffer.length - gut.offset))
            throw std::invalid_argument("Invalid native 3DGUT geometry buffer");
        if (mode == RasterMode::Gut) {
            for (int component = 0; component < 4; ++component)
                if (!std::isfinite(projection.intrinsics[component]) || !std::isfinite(projection.clip_scale[component]))
                    throw std::invalid_argument("Invalid native 3DGUT ray parameters");
            if (projection.intrinsics.x <= 0 || projection.intrinsics.y <= 0 || projection.clip_scale.x <= 0 || projection.extent.z > uint32_t(CameraModel::Equirectangular))
                throw std::invalid_argument("Invalid native 3DGUT camera");
            if (projection.extent.z == uint32_t(CameraModel::Equirectangular)) {
                for (int component = 0; component < 4; ++component)
                    if (!std::isfinite(projection.panorama[component]))
                        throw std::invalid_argument("Invalid native 3DGUT panorama");
                const auto panorama = projection.panorama;
                if (panorama.x < f->width || panorama.y < f->height || panorama.x > 65535 || panorama.y > 65535 ||
                    panorama.z < 0 || panorama.w < 0 || panorama.z + f->width > panorama.x || panorama.w + f->height > panorama.y)
                    throw std::invalid_argument("Invalid native 3DGUT panorama subregion");
            }
        }
        const bool expected_depth = projection.rasterization.y == 1.f;
        if (!std::isfinite(projection.rasterization.y) || (projection.rasterization.y != 0.f && !expected_depth) ||
            (expected_depth && (!std::isfinite(projection.rasterization.z) || projection.rasterization.z <= 0)))
            throw std::invalid_argument("Invalid Metal expected-depth capture parameters");
        const auto logical = lod.logical_indices.buffer ? lod.logical_indices : lod.indices;
        if (lod.enabled && (lod.count != count || !lod.source_count ||
                            (count && (!logical.buffer || logical.buffer.device != impl_->device || logical.offset % 4 ||
                                       logical.offset > logical.buffer.length || size_t(count) * 4 > logical.buffer.length - logical.offset))))
            throw std::invalid_argument("Invalid native Metal LOD identifier mapping");
        const uint32_t logical_count = lod.enabled ? (lod.logical_count ? lod.logical_count : lod.source_count) : count;
        const uint32_t selection_count = overlay.selection.buffer ? (overlay.selection_count ? overlay.selection_count : logical_count) : 0;
        const uint32_t preview_count = overlay.preview.buffer ? (overlay.preview_count ? overlay.preview_count : logical_count) : 0;
        const auto check_mask = [&](BufferSlice mask, uint32_t extent) {
            if (extent && (mask.buffer.device != impl_->device || mask.offset > mask.buffer.length || extent > mask.buffer.length - mask.offset))
                throw std::invalid_argument("Metal logical selection mask exceeds its resident storage");
        };
        check_mask(overlay.selection, selection_count);
        check_mask(overlay.preview, preview_count);
        const RasterParameters p{count, f->width, f->height, f->columns, f->tiles, f->capacity, uint32_t(mode), (overlay.parameter_count ? 1u : 0u) | (expected_depth ? 2u : 0u) | (projection.rasterization.w == 1.f && projection.display.z == 0 ? 4u : 0u) | (lod.enabled ? 8u : 0u) | (projection.display.z == 1.f ? 16u : 0u) | (omit_saturating_color ? 32u : 0u), background, overlay.render_origin, projection.intrinsics, {projection.clip_scale.x, expected_depth ? projection.rasterization.z : projection.clip_scale.y, projection.clip_scale.z, projection.clip_scale.w}, projection.extent, projection.panorama, {selection_count, preview_count, 0, 0}};
        // Compile/cache before reserving the frame or encoding any work. A
        // specialization failure cannot strand its busy flag or partial scratch.
        const auto blend_pipeline = impl_->blendPipeline(uint32_t(mode), p.unused);
        if (f->in_flight.exchange(true, std::memory_order_acq_rel))
            throw std::logic_error("Metal viewer frame reservation is still in flight");
        f->completed.store(false, std::memory_order_release);
        // A committed command retains the frame and all scratch until its completion.
        [command addCompletedHandler:^(id<MTLCommandBuffer> finished) {
            f->completed.store(finished.status == MTLCommandBufferStatusCompleted, std::memory_order_release);
            f->in_flight.store(false, std::memory_order_release);
        }];
        const auto dispatch = [](id<MTLComputeCommandEncoder> e, uint32_t n) {
            [e dispatchThreadgroups:MTLSizeMake(ceil_div(n, 256), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
        };
        const auto set_projected = [&](id<MTLComputeCommandEncoder> e) {
            [e setBuffer:count ? projected.buffer : f->counts offset:count ? projected.offset : 0 atIndex:0];
        };
        if (count) {
            auto e = impl_->begin(command, "tile_counts");
            set_projected(e);
            [e setBuffer:f->counts offset:0 atIndex:1];
            [e setBytes:&p length:sizeof(p) atIndex:2];
            dispatch(e, count);
            impl_->scan(command, f->counts, f->offsets, count, f->count_scan);
        }
        auto e = impl_->begin(command, "tile_status");
        [e setBuffer:f->counts offset:0 atIndex:0];
        [e setBuffer:f->offsets offset:0 atIndex:1];
        [e setBuffer:f->status offset:0 atIndex:2];
        [e setBytes:&p length:sizeof(p) atIndex:3];
        [e setBuffer:f->dispatch_args offset:0 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [e endEncoding];
        if (count) {
            e = impl_->begin(command, "tile_instances");
            set_projected(e);
            [e setBuffer:f->offsets offset:0 atIndex:1];
            [e setBuffer:f->status offset:0 atIndex:2];
            [e setBuffer:f->keys[0] offset:0 atIndex:3];
            [e setBuffer:f->indices[0] offset:0 atIndex:4];
            [e setBytes:&p length:sizeof(p) atIndex:5];
            dispatch(e, count);
            const uint32_t passes = 4 + (std::bit_width(f->tiles - 1) + 7) / 8;
            for (uint32_t pass = 0; pass < passes; ++pass) {
                const SortParameters sort{f->sort_blocks, pass * 8};
                const uint32_t src = pass % 2, dst = 1 - src;
                // Only live sort groups run. Inactive histogram blocks must still
                // be zeroed because the fixed-stride prefix scan spans the reservation.
                auto clear_histogram = [command blitCommandEncoder];
                [clear_histogram fillBuffer:f->histogram range:NSMakeRange(0, f->histogram.length) value:0];
                [clear_histogram endEncoding];
                e = impl_->begin(command, "tile_histogram");
                [e setBuffer:f->keys[src] offset:0 atIndex:0];
                [e setBuffer:f->histogram offset:0 atIndex:1];
                [e setBuffer:f->status offset:0 atIndex:2];
                [e setBytes:&sort length:sizeof(sort) atIndex:3];
                [e dispatchThreadgroupsWithIndirectBuffer:f->dispatch_args indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding];
                impl_->scan(command, f->histogram, f->histogram_offsets, f->sort_blocks * 256, f->histogram_scan, true);
                e = impl_->begin(command, "tile_scatter");
                [e setBuffer:f->keys[src] offset:0 atIndex:0];
                [e setBuffer:f->indices[src] offset:0 atIndex:1];
                [e setBuffer:f->keys[dst] offset:0 atIndex:2];
                [e setBuffer:f->indices[dst] offset:0 atIndex:3];
                [e setBuffer:f->histogram_offsets offset:0 atIndex:4];
                [e setBuffer:f->status offset:0 atIndex:5];
                [e setBytes:&sort length:sizeof(sort) atIndex:6];
                [e dispatchThreadgroupsWithIndirectBuffer:f->dispatch_args indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding];
            }
        }
        auto clear = [command blitCommandEncoder];
        [clear fillBuffer:f->ranges range:NSMakeRange(0, f->ranges.length) value:0];
        [clear endEncoding];
        if (count) {
            e = impl_->begin(command, "tile_ranges");
            const uint32_t sorted = (4 + (std::bit_width(f->tiles - 1) + 7) / 8) % 2;
            [e setBuffer:f->keys[sorted] offset:0 atIndex:0];
            [e setBuffer:f->ranges offset:0 atIndex:1];
            [e setBuffer:f->status offset:0 atIndex:2];
            [e dispatchThreadgroupsWithIndirectBuffer:f->dispatch_args indirectBufferOffset:3 * sizeof(uint32_t) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
        }
        e = [command computeCommandEncoder];
        if (!e)
            throw std::runtime_error("Could not encode Metal viewer blend pass");
        e.label = @"tile_blend";
        [e setComputePipelineState:blend_pipeline];
        set_projected(e);
        const uint32_t sorted = (4 + (std::bit_width(f->tiles - 1) + 7) / 8) % 2;
        [e setBuffer:gut.buffer ?: f->counts offset:gut.buffer ? gut.offset : 0 atIndex:10];
        [e setBuffer:f->indices[sorted] offset:0 atIndex:1];
        [e setBuffer:f->ranges offset:0 atIndex:2];
        [e setBuffer:f->status offset:0 atIndex:3];
        [e setBytes:&p length:sizeof(p) atIndex:4];
        const std::array<BufferSlice, 5> overlays = {overlay.parameters, overlay.flags, overlay.selection, overlay.preview, overlay.colors};
        for (NSUInteger j = 0; j < overlays.size(); ++j)
            [e setBuffer:overlays[j].buffer ?: f->counts offset:overlays[j].buffer ? overlays[j].offset : 0 atIndex:5 + j];
        [e setBuffer:lod.enabled && logical.buffer ? logical.buffer : f->counts offset:lod.enabled && logical.buffer ? logical.offset : 0 atIndex:11];
        [e setBytes:&logical_count length:sizeof(logical_count) atIndex:12];
        [e setTexture:f->color atIndex:0];
        [e setTexture:f->depth atIndex:1];
        [e setTexture:f->pick atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(size_t(f->tiles) * 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
    }
} // namespace lfs::rendering::metal
