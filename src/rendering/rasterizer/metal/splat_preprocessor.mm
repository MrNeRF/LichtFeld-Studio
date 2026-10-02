/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_preprocessor.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "gpu_profile.hpp"
#include "shader_source.hpp"

#include <array>
#include <cmath>
#include <format>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace lfs::rendering::metal {
    namespace {
        static_assert(core::kShReorderSize == 32);
        static_assert(core::kShMaxCoeffsRest == 15);
        static_assert(core::sh_value_quant::kBlockSize == 256);

        std::runtime_error failure(const char* operation, NSError* error) {
            return std::runtime_error(std::string(operation) + ": " +
                                      (error.localizedDescription.UTF8String ?: "Metal operation failed"));
        }

        void check_slice(BufferSlice view, size_t bytes, NSUInteger alignment,
                         id<MTLDevice> device, const char* name) {
            if (!view.buffer || view.buffer.device != device || view.offset % alignment ||
                view.offset > view.buffer.length || bytes > view.buffer.length - view.offset)
                throw std::invalid_argument(std::format("Invalid Metal splat buffer: {} (offset={}, length={}, required_bytes={}, alignment={}, same_device={})", name, view.offset, view.buffer.length, bytes, alignment, view.buffer.device == device));
        }

        void check_projection(const Projection& p) {
            const auto finite_matrix = [](simd_float4x4 matrix) {
                for (int column = 0; column < 4; ++column)
                    for (int row = 0; row < 4; ++row)
                        if (!std::isfinite(matrix.columns[column][row]))
                            return false;
                return matrix.columns[0].w == 0 && matrix.columns[1].w == 0 &&
                       matrix.columns[2].w == 0 && matrix.columns[3].w == 1;
            };
            bool finite = true;
            for (int i = 0; i < 4; ++i)
                finite = finite && std::isfinite(p.intrinsics[i]) &&
                         std::isfinite(p.clip_scale[i]) && std::isfinite(p.camera_local[i]) && std::isfinite(p.display[i]);
            if (!finite || !finite_matrix(p.model_to_world) || !finite_matrix(p.world_to_camera) ||
                !p.extent.x || !p.extent.y || p.extent.x > 65535 || p.extent.y > 65535 ||
                p.extent.z > uint32_t(CameraModel::Equirectangular) || p.extent.w > 1 || p.intrinsics.x <= 0 || p.intrinsics.y <= 0 || p.clip_scale.x <= 0 ||
                p.clip_scale.y <= p.clip_scale.x || p.clip_scale.z <= 0 || p.clip_scale.w < 0 ||
                (p.display.z != 0 && p.display.z != 1) || (p.display.w != 0 && p.display.w != 1) ||
                !std::isfinite(p.rasterization.w) || (p.rasterization.w != 0.f && p.rasterization.w != 1.f))
                throw std::invalid_argument(std::format("Invalid Metal splat projection (extent={}x{}, camera={}, orthographic={}, focal=({}, {}), clip=({}, {}), scale={}, finite={})", p.extent.x, p.extent.y, p.extent.z, p.extent.w, p.intrinsics.x, p.intrinsics.y, p.clip_scale.x, p.clip_scale.y, p.clip_scale.z, finite));
            if (p.extent.z == uint32_t(CameraModel::Equirectangular) || (p.display.w == 1 && (p.panorama.x != 0 || p.panorama.y != 0))) {
                for (int i = 0; i < 4; ++i)
                    if (!std::isfinite(p.panorama[i]))
                        throw std::invalid_argument(std::format("Invalid Metal full-camera dimensions (component={}, value={})", i, p.panorama[i]));
                if (p.panorama.x < p.extent.x || p.panorama.y < p.extent.y ||
                    p.panorama.x > 65535 || p.panorama.y > 65535 ||
                    p.panorama.z < 0 || p.panorama.w < 0 ||
                    p.panorama.z + p.extent.x > p.panorama.x || p.panorama.w + p.extent.y > p.panorama.y)
                    throw std::invalid_argument(std::format("Invalid Metal full-camera subregion (camera={}x{}, origin=({}, {}), extent={}x{})", p.panorama.x, p.panorama.y, p.panorama.z, p.panorama.w, p.extent.x, p.extent.y));
            } else if (p.display.w == 1 && (p.panorama.z != 0 || p.panorama.w != 0)) {
                throw std::invalid_argument(std::format("Metal GS subregions require full-camera dimensions (origin=({}, {}), camera={}x{})", p.panorama.z, p.panorama.w, p.panorama.x, p.panorama.y));
            }
        }
    } // namespace

    struct SplatPreprocessor::Impl {
        id<MTLDevice> device;
        id<MTLLibrary> library;
        id<MTLBuffer> empty;
        std::mutex mutex;
        std::map<uint32_t, id<MTLComputePipelineState>> pipelines;

        id<MTLComputePipelineState> pipeline(ShStorage storage, uint32_t degree, PrimitiveMode mode, bool tight_bounds) {
            const uint32_t format = static_cast<uint32_t>(storage), primitive = static_cast<uint32_t>(mode);
            if (format > 4 || degree > 3 || primitive > 3)
                throw std::invalid_argument(std::format("Unsupported Metal splat specialization (storage={}, degree={}, primitive={}, tight_bounds={})", format, degree, primitive, tight_bounds));
            const uint32_t key = (format * 16 + degree * 4 + primitive) | (tight_bounds ? 128u : 0u);
            std::lock_guard lock(mutex);
            if (auto found = pipelines.find(key); found != pipelines.end())
                return found->second;
            MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
            [constants setConstantValue:&format type:MTLDataTypeUInt atIndex:0];
            [constants setConstantValue:&degree type:MTLDataTypeUInt atIndex:1];
            [constants setConstantValue:&primitive type:MTLDataTypeUInt atIndex:2];
            [constants setConstantValue:&tight_bounds type:MTLDataTypeBool atIndex:3];
            NSError* error = nil;
            id<MTLFunction> function = [library newFunctionWithName:@"project_splats" constantValues:constants error:&error];
            if (!function)
                throw failure("Compile Metal splat specialization", error);
            auto descriptor = [MTLComputePipelineDescriptor new];
            descriptor.computeFunction = function;
            // Every projection dispatch uses 256 lanes. Bound compiler and
            // validator resource planning to that actual threadgroup size.
            descriptor.maxTotalThreadsPerThreadgroup = 256;
            id<MTLComputePipelineState> state = [device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
            if (!state)
                throw failure("Create Metal splat pipeline", error);
            pipelines.emplace(key, state);
            return state;
        }
    };

    SplatPreprocessor::SplatPreprocessor(id<MTLDevice> device) : impl_(std::make_unique<Impl>()) {
        if (!device)
            throw std::invalid_argument(std::format("Metal viewer requires a device (device_present={})", device != nil));
        impl_->device = device;
        MTLCompileOptions* options = [MTLCompileOptions new];
        // Q16 fma and near-plane finite checks are contractual.
        if (@available(macOS 15.0, iOS 18.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        options.languageVersion = MTLLanguageVersion2_4;
        NSError* error = nil;
        impl_->library = [device newLibraryWithSource:[NSString stringWithUTF8String:kSplatPreprocessorSource]
                                              options:options
                                                error:&error];
        if (!impl_->library)
            throw failure("Compile Metal viewer projection", error);
        const std::array<uint32_t, 24> zero{};
        impl_->empty = [device newBufferWithBytes:zero.data() length:sizeof(zero) options:MTLResourceStorageModeShared];
        if (!impl_->empty)
            throw std::runtime_error(std::format("Cannot allocate Metal placeholder buffer (bytes={})", sizeof(zero)));
    }
    SplatPreprocessor::~SplatPreprocessor() = default;

    void SplatPreprocessor::prepare(ShStorage storage, uint32_t degree, PrimitiveMode mode, bool tight_bounds) {
        (void)impl_->pipeline(storage, degree, mode, tight_bounds);
    }

    void SplatPreprocessor::encode(id<MTLCommandBuffer> command, const SplatBuffers& in,
                                   const Projection& projection, uint32_t degree, PrimitiveMode mode, BufferSlice output, const SceneBuffers& scene, const OverlayBuffers& overlay, BufferSlice gut_output, const LodSelection& lod, GpuProfile* profile, bool tight_bounds) {
        if (!command || command.commandQueue.device != impl_->device || command.status != MTLCommandBufferStatusNotEnqueued)
            throw std::invalid_argument(std::format("Metal viewer requires an uncommitted command buffer on the same device (present={}, same_device={}, status={})", command != nil, command.commandQueue.device == impl_->device, long(command.status)));
        if (degree > 3 || (in.layout_rest != 0 && in.layout_rest != 3 && in.layout_rest != 8 && in.layout_rest != 15) ||
            core::sh_rest_coefficients_for_degree(degree) > in.layout_rest || static_cast<uint32_t>(in.storage) > 4 ||
            static_cast<uint32_t>(mode) > 3)
            throw std::invalid_argument(std::format("Active SH degree does not fit resident Metal storage (degree={}, rest={}, storage={}, primitive={})", degree, in.layout_rest, uint32_t(in.storage), uint32_t(mode)));
        check_projection(projection);
        if (!in.count)
            return;
        const bool rad = in.storage == ShStorage::RadSigned8;
        if (rad && (!in.rad_page_splats || in.rad_page_splats % 32))
            throw std::invalid_argument(std::format("RAD pool page size must be an explicit multiple of the SH cell width (page_splats={}, cell_width=32)", in.rad_page_splats));
        const size_t n = in.count, draw_count = lod.enabled ? lod.count : n;
        const uint32_t logical_count = lod.enabled && lod.logical_count ? lod.logical_count : in.count;
        const uint32_t deleted_count = in.deleted_count ? in.deleted_count : in.count;
        if (lod.enabled && lod.source_count != in.count)
            throw std::invalid_argument(std::format("Metal resident LOD source extent mismatch (lod_source={}, resident={})", lod.source_count, in.count));
        const std::array<BufferSlice, 4> lod_buffers = {lod.indices, lod.logical_indices, lod.levels, lod.weights};
        for (size_t j = 0; j < lod_buffers.size(); ++j) {
            const auto input = lod_buffers[j];
            if (!lod.enabled || (!input.buffer && j))
                continue;
            if (draw_count)
                check_slice(input, draw_count * 4, 4, impl_->device, "LOD selection");
            if (input.buffer && (input.buffer == output.buffer || input.buffer == gut_output.buffer || input.buffer == overlay.flags.buffer))
                throw std::invalid_argument(std::format("Metal LOD input overlaps projection output (input_index={}, offset={}, length={}, projected_overlap={}, gut_overlap={}, flags_overlap={})", j, input.offset, input.buffer.length, input.buffer == output.buffer, input.buffer == gut_output.buffer, input.buffer == overlay.flags.buffer));
        }
        if (lod.enabled && lod.counter.buffer) {
            check_slice(lod.counter, 4, 4, impl_->device, "GPU LOD count");
            if (lod.counter.buffer == output.buffer || lod.counter.buffer == gut_output.buffer || lod.counter.buffer == overlay.flags.buffer)
                throw std::invalid_argument(std::format("Metal GPU LOD counter overlaps projection output (offset={}, length={}, projected_overlap={}, gut_overlap={}, flags_overlap={})", lod.counter.offset, lod.counter.buffer.length, lod.counter.buffer == output.buffer, lod.counter.buffer == gut_output.buffer, lod.counter.buffer == overlay.flags.buffer));
        }
        if (!draw_count)
            return;
        if (overlay.parameter_count) {
            if (overlay.parameter_count != 207)
                throw std::invalid_argument(std::format("Metal overlay parameter ABI mismatch (actual={}, expected=207)", overlay.parameter_count));
            check_slice(overlay.parameters, 207 * 16, 16, impl_->device, "overlay parameters");
            check_slice(overlay.flags, draw_count * 4, 4, impl_->device, "overlay flags");
            if (overlay.node_count)
                check_slice(overlay.node_mask, overlay.node_count, 1, impl_->device, "node emphasis mask");
        }
        const bool gaussians = mode != PrimitiveMode::Points;
        size_t rest_bytes = 0, bounds_bytes = 0;
        if (degree) {
            if (in.storage == ShStorage::Q16) {
                rest_bytes = core::sh_value_quant::sh_value_u16_count(n, in.layout_rest) * sizeof(uint16_t);
                bounds_bytes = core::sh_value_quant::n_bounds_for_prims(n) * 2u * sizeof(float);
            } else if (rad) {
                rest_bytes = core::sh_swizzled_float_count(n, in.layout_rest);
                bounds_bytes = ((n + in.rad_page_splats - 1) / in.rad_page_splats) * 64;
            } else if (in.storage == ShStorage::CanonicalFloat32)
                rest_bytes = n * in.layout_rest * 3u * sizeof(float);
            else
                rest_bytes = core::sh_swizzled_float_count(n, in.layout_rest) *
                             (in.storage == ShStorage::SwizzledFloat16 ? sizeof(uint16_t) : sizeof(float));
        }
        const std::array<BufferSlice, 10> inputs = {in.means, in.log_scales, in.rotations, in.opacity_logits,
                                                    in.sh0, in.sh_rest, in.sh_bounds, in.deleted, scene.object_indices, scene.objects};
        const size_t attr = (rad || in.non_sh_attrs_f16) ? 2 : 4;
        const std::array<size_t, 10> lengths = {n * 12, gaussians ? n * (rad ? 8 : 3 * attr) : 0, gaussians ? n * 4 * attr : 0, n * attr, n * (rad ? 8 : 12),
                                                rest_bytes, bounds_bytes, in.deleted.buffer ? deleted_count : 0, scene.count && scene.object_indices.buffer ? size_t(logical_count) * 4 : 0, size_t(scene.count) * sizeof(SceneObject)};
        const std::array<NSUInteger, 10> alignments = {4, rad ? 8u : attr, 4 * attr, attr, rad ? 8u : 4u, 4, rad ? 16u : 8u, 1, 4, 16};
        const char* names[] = {"means", "scales", "rotations", "opacity", "SH0", "SH rest", "SH bounds", "deleted mask", "object indices", "scene objects"};
        check_slice(output, draw_count * sizeof(ProjectedSplat), 16, impl_->device, "projection output");
        if (mode == PrimitiveMode::Gut) {
            check_slice(gut_output, draw_count * sizeof(GutSplat), 16, impl_->device, "3DGUT output");
            if (gut_output.buffer == output.buffer)
                throw std::invalid_argument(std::format("Metal 3DGUT geometry overlaps projection output (geometry_offset={}, projection_offset={}, draw_count={})", gut_output.offset, output.offset, draw_count));
            for (const auto input : inputs)
                if (input.buffer && input.buffer == gut_output.buffer)
                    throw std::invalid_argument(std::format("Metal 3DGUT geometry overlaps source storage (input_offset={}, geometry_offset={}, input_length={}, draw_count={})", input.offset, gut_output.offset, input.buffer.length, draw_count));
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!lengths[i])
                continue;
            check_slice(inputs[i], lengths[i], alignments[i], impl_->device, names[i]);
            if (inputs[i].buffer == output.buffer && inputs[i].offset < output.offset + draw_count * sizeof(ProjectedSplat) &&
                output.offset < inputs[i].offset + lengths[i])
                throw std::invalid_argument(std::format("Metal projection output overlaps input (input={}, input_offset={}, input_bytes={}, output_offset={}, output_bytes={})", names[i], inputs[i].offset, lengths[i], output.offset, draw_count * sizeof(ProjectedSplat)));
        }
        if (scene.count > 1 && !scene.object_indices.buffer)
            throw std::invalid_argument(std::format("Multiple Metal scene objects require primitive indices (objects={}, logical_splats={})", scene.count, logical_count));
        // Resolve/compile before opening an encoder so failure leaves the command usable.
        auto pipeline = impl_->pipeline(in.storage, degree, mode, tight_bounds);
        id<MTLComputeCommandEncoder> encoder = profiledCompute(command, profile, GpuStage::Projection);
        if (!encoder)
            throw std::runtime_error(std::format("Cannot create Metal projection encoder (source_count={}, draw_count={}, degree={}, primitive={})", n, draw_count, degree, uint32_t(mode)));
        encoder.label = @"LichtFeld splat projection";
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger i = 0; i < inputs.size(); ++i)
            [encoder setBuffer:lengths[i] ? inputs[i].buffer : impl_->empty offset:lengths[i] ? inputs[i].offset : 0 atIndex:i < 8 ? i : i + 3];
        [encoder setBuffer:output.buffer offset:output.offset atIndex:8];
        [encoder setBytes:&projection length:sizeof(projection) atIndex:9];
        const std::array<uint32_t, 12> layout = {in.count, in.layout_rest, in.deleted.buffer ? 1u : 0u, scene.count, (rad || in.non_sh_attrs_f16) ? 1u : 0u, overlay.parameter_count ? 1u : 0u, scene.object_indices.buffer ? 1u : 0u, uint32_t(draw_count),
                                                 lod.enabled ? (1u | (lod.logical_indices.buffer ? 2u : 0u) | (lod.levels.buffer ? 4u : 0u) | (lod.weights.buffer ? 8u : 0u) | (lod.debug ? 16u : 0u) | (lod.counter.buffer ? 32u : 0u)) : 0u, in.rad_page_splats, logical_count, deleted_count};
        [encoder setBytes:layout.data() length:sizeof(layout) atIndex:10];
        const std::array<BufferSlice, 3> overlays = {overlay.parameters, overlay.flags, overlay.node_mask};
        for (NSUInteger j = 0; j < overlays.size(); ++j)
            [encoder setBuffer:overlays[j].buffer ?: impl_->empty offset:overlays[j].buffer ? overlays[j].offset : 0 atIndex:13 + j];
        [encoder setBuffer:gut_output.buffer ?: impl_->empty offset:gut_output.buffer ? gut_output.offset : 0 atIndex:16];
        for (NSUInteger j = 0; j < lod_buffers.size(); ++j)
            [encoder setBuffer:lod.enabled && lod_buffers[j].buffer ? lod_buffers[j].buffer : impl_->empty
                        offset:lod.enabled && lod_buffers[j].buffer ? lod_buffers[j].offset : 0
                       atIndex:17 + j];
        [encoder setBuffer:lod.enabled && lod.counter.buffer ? lod.counter.buffer : impl_->empty
                    offset:lod.enabled && lod.counter.buffer ? lod.counter.offset : 0
                   atIndex:21];
        const NSUInteger width = std::min(NSUInteger(256), pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(draw_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
    }
} // namespace lfs::rendering::metal
