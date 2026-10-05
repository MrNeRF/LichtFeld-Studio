/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_scene_upscaler.hpp"
#include "core/tensor.hpp"
#include "core/tensor_metal_reader.hpp"
#include "tensor_scene_temporal.hpp"
#import <MetalFX/MetalFX.h>
#include <cstring>
#include <cmath>
#include <stdexcept>

namespace lfs::vis {
    namespace {
        // MetalFX consumes textures, whereas the native compositor consumes
        // packed tensors. Both conversions stay on the GPU and participate in
        // MetalTensorReader's producer/consumer timeline and storage lifetime.
        constexpr auto conversionSource = R"(
#include <metal_stdlib>
using namespace metal;
struct Params { uint4 layout; uint4 extent; float4 depth; };
kernel void pack(device const uchar* color [[buffer(0)]],
                 device const float* depth [[buffer(1)]],
                 device const float2* motion [[buffer(2)]],
                 constant Params& p [[buffer(3)]],
                 texture2d<float, access::write> ct [[texture(0)]],
                 texture2d<float, access::write> dt [[texture(1)]],
                 texture2d<float, access::write> mt [[texture(2)]],
                 uint2 xy [[thread_position_in_grid]]) {
    if (any(xy >= p.extent.xy)) return;
    uint2 source = xy;
    if (p.extent.z) source.y = p.extent.y - 1 - xy.y;
    uint idx = (source.y * p.layout.x + source.x) * p.layout.z;
    float4 c = p.layout.w ? float4(((device const float*)color)[idx], ((device const float*)color)[idx+1], ((device const float*)color)[idx+2], ((device const float*)color)[idx+3]) : float4(color[idx],color[idx+1],color[idx+2],color[idx+3])/255.0f;
    ct.write(c, xy);
    if (p.extent.w) {
        float z = depth[source.y * p.layout.y + source.x];
        float n = p.depth.x, f = p.depth.y;
        float raster = (!isfinite(z) || z <= 0) ? 1.0f :
            (p.depth.z ? (z-n)/(f-n) : f/(f-n)*(1.0f-n/max(z,n)));
        dt.write(float4(clamp(raster,0.0f,1.0f)),xy);
        mt.write(float4(motion[xy.y*p.extent.x+xy.x],0,0),xy);
    }
}
kernel void unpack(texture2d<float, access::read> source [[texture(0)]],
                   texture2d<float, access::sample> coverage [[texture(1)]],
                   device float4* output [[buffer(0)]],
                   constant float2& jitter [[buffer(1)]],
                   uint2 xy [[thread_position_in_grid]]) {
    if (xy.x >= source.get_width() || xy.y >= source.get_height()) return;
    constexpr sampler linearClamp(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(xy)+.5f)/float2(source.get_width(),source.get_height()) +
                jitter/float2(coverage.get_width(),coverage.get_height());
    // MetalFX reconstructs RGB and writes opaque alpha. Preserve scene
    // coverage explicitly for environment/mesh composition and transparent capture.
    float4 value = source.read(xy);
    value.a = coverage.sample(linearClamp,uv).a;
    output[xy.y*source.get_width()+xy.x] = value;
}
)";
        lfs::Error error(std::string detail) {
            return lfs::make_error({.code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::Rendering, .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        id<MTLTexture> texture(id<MTLDevice> device, MTLPixelFormat format,
                              glm::ivec2 size, MTLTextureUsage usage) {
            auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                width:size.x height:size.y mipmapped:NO];
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.usage = usage;
            auto result = [device newTextureWithDescriptor:descriptor];
            if (!result) throw std::runtime_error("MetalFX texture allocation failed");
            return result;
        }
        bool validColor(const core::Tensor& color, glm::ivec2 extent) {
            return color.is_valid() && color.device() == core::Device::GPU &&
                core::gpu_backend_of(color) == core::GpuBackend::Metal && color.is_contiguous() &&
                color.ndim() == 3 && color.size(0) >= size_t(extent.y) &&
                color.size(1) >= size_t(extent.x) && color.size(2) == 4 &&
                (color.dtype() == core::DataType::UInt8 || color.dtype() == core::DataType::Float32);
        }
    }

    bool metalFxBackendAvailable(SceneUpscalerBackend backend) {
        if (!isMetalFxBackend(backend)) return false;
        @autoreleasepool {
            auto device = MTLCreateSystemDefaultDevice();
            return device && (backend == SceneUpscalerBackend::MetalFxSpatial
                ? [MTLFXSpatialScalerDescriptor supportsDevice:device]
                : [MTLFXTemporalScalerDescriptor supportsDevice:device]);
        }
    }

    struct MetalSceneUpscaler::Impl {
        struct Feature {
            SceneUpscalerBackend backend;
            glm::ivec2 input, output;
            id<MTLFXSpatialScaler> spatial;
            id<MTLFXTemporalScaler> temporal;
            id<MTLTexture> color, depth, motion, result;
        };
        core::MetalTensorReader reader;
        rendering::TensorSceneTemporalKernels kernels{core::GpuBackend::Metal};
        SceneTemporalCoordinator coordinator;
        std::array<std::shared_ptr<Feature>, size_t(TemporalViewId::Count)> features;
        id<MTLComputePipelineState> pack, unpack;

        void ensureProgram() {
            if (pack && unpack) return;
            NSError* failure = nil;
            auto library = [reader.device() newLibraryWithSource:@(conversionSource) options:nil error:&failure];
            if (!library) throw std::runtime_error(failure.localizedDescription.UTF8String);
            pack = [reader.device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"pack"] error:&failure];
            unpack = [reader.device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"unpack"] error:&failure];
            if (!pack || !unpack) throw std::runtime_error("MetalFX conversion pipeline creation failed");
        }
        std::shared_ptr<Feature> makeFeature(SceneUpscalerBackend backend, glm::ivec2 input, glm::ivec2 output) {
            auto f = std::make_shared<Feature>();
            f->backend = backend; f->input = input; f->output = output;
            auto device = reader.device();
            MTLTextureUsage colorUsage, resultUsage;
            if (backend == SceneUpscalerBackend::MetalFxSpatial) {
                auto d = [MTLFXSpatialScalerDescriptor new];
                d.inputWidth = input.x; d.inputHeight = input.y;
                d.outputWidth = output.x; d.outputHeight = output.y;
                d.colorTextureFormat = MTLPixelFormatRGBA16Float;
                d.outputTextureFormat = MTLPixelFormatRGBA16Float;
                d.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
                f->spatial = [d newSpatialScalerWithDevice:device];
                if (!f->spatial) throw std::runtime_error("MetalFX Spatial feature creation failed");
                colorUsage = f->spatial.colorTextureUsage; resultUsage = f->spatial.outputTextureUsage;
            } else {
                auto d = [MTLFXTemporalScalerDescriptor new];
                d.inputWidth = input.x; d.inputHeight = input.y;
                d.outputWidth = output.x; d.outputHeight = output.y;
                d.colorTextureFormat = MTLPixelFormatRGBA16Float;
                d.outputTextureFormat = MTLPixelFormatRGBA16Float;
                d.depthTextureFormat = MTLPixelFormatR32Float;
                d.motionTextureFormat = MTLPixelFormatRG32Float;
                d.autoExposureEnabled = NO;
                // The shared motion kernel uses unjittered projection matrices.
                // Keep the descriptor default (unjittered vectors) on macOS 26.
                f->temporal = [d newTemporalScalerWithDevice:device];
                if (!f->temporal) throw std::runtime_error("MetalFX Temporal feature creation failed");
                colorUsage = f->temporal.colorTextureUsage; resultUsage = f->temporal.outputTextureUsage;
                f->depth = texture(device, MTLPixelFormatR32Float, input, f->temporal.depthTextureUsage | MTLTextureUsageShaderWrite);
                f->motion = texture(device, MTLPixelFormatRG32Float, input, f->temporal.motionTextureUsage | MTLTextureUsageShaderWrite);
            }
            f->color = texture(device, MTLPixelFormatRGBA16Float, input, colorUsage | MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead);
            f->result = texture(device, MTLPixelFormatRGBA16Float, output, resultUsage | MTLTextureUsageShaderRead);
            return f;
        }
    };

    MetalSceneUpscaler::MetalSceneUpscaler() : impl_(std::make_unique<Impl>()) {}
    MetalSceneUpscaler::~MetalSceneUpscaler() = default;

    lfs::Result<TensorSceneTemporalResult> MetalSceneUpscaler::resolve(
        SceneUpscalerBackend backend, const TensorSceneTemporalRequest& r) {
        const auto slot = size_t(r.view);
        const bool temporal = backend == SceneUpscalerBackend::MetalFxTemporal;
        if (!isMetalFxBackend(backend) || slot >= size_t(TemporalViewId::Count) ||
            r.render_extent.x <= 0 || r.render_extent.y <= 0 ||
            r.output_extent.x < r.render_extent.x || r.output_extent.y < r.render_extent.y ||
            !r.color || !validColor(*r.color, r.render_extent))
            return error("MetalFX requires a contiguous four-channel Metal image and valid upscale extents");
        if (temporal && (!r.depth || !r.depth->is_valid() ||
            core::gpu_backend_of(*r.depth) != core::GpuBackend::Metal ||
            r.depth->dtype() != core::DataType::Float32 || !r.depth->is_contiguous() ||
            r.frame.view.size != r.render_extent || r.frame.output_extent != r.output_extent ||
            r.frame.view.near_plane <= 0 || r.frame.view.far_plane <= r.frame.view.near_plane ||
            r.depth->ndim() != 2 || r.depth->size(0) != size_t(r.render_extent.y) ||
            r.depth->size(1) != size_t(r.render_extent.x) || !validTemporalFrameInput(r.frame)))
            return error("MetalFX Temporal requires aligned view-space depth and a valid frame contract");
        PreparedSceneTemporalFrame prepared;
        try {
            @autoreleasepool {
                const core::GpuBackendScope scope(core::GpuBackend::Metal);
                impl_->ensureProgram();
                auto f = impl_->features[slot];
                if (!f || f->backend != backend || f->input != r.render_extent || f->output != r.output_extent) {
                    f = impl_->makeFeature(backend, r.render_extent, r.output_extent);
                    impl_->coordinator.reset(r.view, TemporalResetReason::RenderSize);
                }
                core::Tensor motion;
                if (temporal) {
                    prepared = impl_->coordinator.prepare({.view = r.view,
                        .requirements = {.depth = true, .motion = true, .jitter = !r.frame.view.orthographic,
                                         .history_color = true},
                        .frame = r.frame, .render_extent = r.render_extent, .output_extent = r.output_extent});
                    if (!prepared.active()) throw std::runtime_error("MetalFX temporal preparation failed");
                    auto projections = makeTemporalMotionViewProjectionPair(prepared.frame);
                    if (!projections) throw std::runtime_error("MetalFX motion projection unavailable");
                    rendering::TensorSceneMotionParameters params;
                    auto inverse = glm::inverse(projections->current);
                    std::memcpy(params.inverse_current_view_projection.data(), &inverse, sizeof(inverse));
                    std::memcpy(params.previous_view_projection.data(), &projections->previous, sizeof(projections->previous));
                    params.render_info = {uint32_t(r.render_extent.x), uint32_t(r.render_extent.y), 0,
                                          r.frame.view.orthographic ? 2u : 1u};
                    params.depth_info = {r.frame.view.near_plane, r.frame.view.far_plane, r.flip_y ? 1.f : 0.f, 0};
                    auto generated = impl_->kernels.motion(*r.depth, params, motion);
                    if (!generated) { impl_->coordinator.discard(prepared); return std::move(generated).error(); }
                }
                auto result = std::make_shared<core::Tensor>(core::Tensor::empty(
                    {size_t(r.output_extent.y), size_t(r.output_extent.x), 4}, core::Device::GPU, core::DataType::Float32));
                const std::array<const core::Tensor*, 3> inputs{r.color.get(), temporal ? r.depth.get() : nullptr,
                                                            temporal ? &motion : nullptr};
                const std::array<core::Tensor*, 1> outputs{result.get()};
                const auto jitter = sceneTemporalJitterPixels(prepared.frame.current_jitter, r.render_extent, false);
                auto command = impl_->reader.submitWrites(inputs, outputs,
                    [&](id<MTLCommandBuffer> command, auto in, auto out) {
                        struct Params { std::array<uint32_t,4> layout, extent; std::array<float,4> depth; };
                        const Params params{{uint32_t(r.color->size(1)), temporal ? uint32_t(r.depth->size(1)) : 0u,
                                             4, r.color->dtype() == core::DataType::Float32 ? 1u : 0u},
                                            {uint32_t(r.render_extent.x),uint32_t(r.render_extent.y),r.flip_y ? 1u : 0u,temporal ? 1u : 0u},
                                            {r.frame.view.near_plane,r.frame.view.far_plane,r.frame.view.orthographic ? 1.f : 0.f,0}};
                        auto encoder = [command computeCommandEncoder];
                        if (!encoder) throw std::runtime_error("MetalFX input encoder unavailable");
                        [encoder setComputePipelineState:impl_->pack];
                        for (NSUInteger i=0;i<in.size();++i) [encoder setBuffer:in[i].buffer offset:in[i].offset atIndex:i];
                        [encoder setBytes:&params length:sizeof(params) atIndex:3];
                        [encoder setTexture:f->color atIndex:0]; [encoder setTexture:f->depth atIndex:1]; [encoder setTexture:f->motion atIndex:2];
                        [encoder dispatchThreads:MTLSizeMake(r.render_extent.x,r.render_extent.y,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
                        [encoder endEncoding];
                        if (temporal) {
                            auto scaler = f->temporal;
                            scaler.colorTexture = f->color; scaler.depthTexture = f->depth;
                            scaler.motionTexture = f->motion; scaler.outputTexture = f->result;
                            scaler.inputContentWidth = r.render_extent.x; scaler.inputContentHeight = r.render_extent.y;
                            scaler.jitterOffsetX = jitter.x; scaler.jitterOffsetY = jitter.y;
                            scaler.motionVectorScaleX = 1; scaler.motionVectorScaleY = 1;
                            scaler.preExposure = 1; scaler.depthReversed = NO;
                            scaler.reset = !prepared.frame.history_valid;
                            [scaler encodeToCommandBuffer:command];
                        } else {
                            f->spatial.colorTexture = f->color; f->spatial.outputTexture = f->result;
                            f->spatial.inputContentWidth = r.render_extent.x; f->spatial.inputContentHeight = r.render_extent.y;
                            [f->spatial encodeToCommandBuffer:command];
                        }
                        encoder = [command computeCommandEncoder];
                        if (!encoder) throw std::runtime_error("MetalFX output encoder unavailable");
                        [encoder setComputePipelineState:impl_->unpack]; [encoder setTexture:f->result atIndex:0];
                        [encoder setTexture:f->color atIndex:1];
                        const glm::vec2 alphaJitter = temporal ? jitter : glm::vec2(0);
                        [encoder setBytes:&alphaJitter length:sizeof(alphaJitter) atIndex:1];
                        [encoder setBuffer:out[0].buffer offset:out[0].offset atIndex:0];
                        [encoder dispatchThreads:MTLSizeMake(r.output_extent.x,r.output_extent.y,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
                        [encoder endEncoding];
                        // Retain the old feature through completion, including a resize/reset/release.
                        [command addCompletedHandler:^(id<MTLCommandBuffer>) { (void)f; }];
                    });
                if (!command) throw std::runtime_error("MetalFX command was not submitted");
                if (temporal && !impl_->coordinator.commit(prepared, SceneHistoryStorage::MetalFx))
                    throw std::runtime_error("MetalFX history commit failed");
                if (!temporal) impl_->coordinator.reset(r.view);
                impl_->features[slot] = std::move(f);
                return TensorSceneTemporalResult{.color = std::move(result), .sequence = temporal ? prepared.frame.sequence+1 : 0,
                    .reset_reasons = temporal ? prepared.frame.reset_reasons : TemporalResetReason::None};
            }
        } catch (const std::exception& e) {
            if (prepared.active()) impl_->coordinator.discard(prepared);
            reset(r.view);
            return error(e.what());
        }
    }
    void MetalSceneUpscaler::reset(TemporalViewId view) {
        const auto slot = size_t(view);
        if (slot >= impl_->features.size()) return;
        impl_->coordinator.reset(view);
        impl_->features[slot].reset();
    }
    void MetalSceneUpscaler::resetAll() {
        impl_->coordinator.resetAll(); impl_->features.fill({});
    }
} // namespace lfs::vis
