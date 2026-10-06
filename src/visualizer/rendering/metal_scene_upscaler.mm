/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_scene_upscaler.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_metal_reader.hpp"
#include "tensor_scene_temporal.hpp"
#include "tensor_scene_texture.hpp"
#import <MetalFX/MetalFX.h>
#include <cstring>
#include <cmath>
#include <stdexcept>

namespace lfs::vis {
    namespace {
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
                core::tensor_supports_metal_access(color) && color.is_contiguous() &&
                color.ndim() == 3 && color.size(0) >= size_t(extent.y) &&
                color.size(1) >= size_t(extent.x) && color.size(2) == 4 &&
                (color.dtype() == core::DataType::UInt8 || color.dtype() == core::DataType::Float32);
        }
    }

    bool metalFxBackendAvailable(SceneUpscalerBackend backend) {
        if (!isMetalFxBackend(backend)) return false;
        static const auto support = [] {
            @autoreleasepool {
                auto device = MTLCreateSystemDefaultDevice();
                return std::array<bool, 2>{device && [MTLFXSpatialScalerDescriptor supportsDevice:device],
                                          device && [MTLFXTemporalScalerDescriptor supportsDevice:device]};
            }
        }();
        return support[backend == SceneUpscalerBackend::MetalFxSpatial ? 0 : 1];
    }

    struct MetalSceneUpscaler::Impl {
        struct Feature {
            SceneUpscalerBackend backend;
            glm::ivec2 input, output;
            id<MTLFXSpatialScaler> spatial;
            id<MTLFXTemporalScaler> temporal;
            id<MTLTexture> color, depth, motion, result;
            core::Tensor packed_color, packed_depth, packed_motion, packed_result, motion_tensor;
            std::array<uint32_t, 4> pitches;
            std::array<std::shared_ptr<core::Tensor>, 2> outputs;
            std::shared_ptr<core::Tensor> acquireOutput() {
                for (auto& buffer : outputs) {
                    if (buffer && buffer.use_count() == 1) return buffer;
                    if (!buffer) {
                        buffer = std::make_shared<core::Tensor>(core::Tensor::empty(
                            {size_t(output.y), size_t(output.x), 4}, core::Device::GPU, core::DataType::Float32));
                        return buffer;
                    }
                }
                // A caller retaining both outputs must not see them overwritten.
                // Keep the persistent pool bounded; this exceptional output is not cached.
                return std::make_shared<core::Tensor>(core::Tensor::empty(
                    {size_t(output.y), size_t(output.x), 4}, core::Device::GPU, core::DataType::Float32));
            }
        };
        core::MetalTensorReader reader;
        rendering::TensorSceneTemporalKernels kernels{core::GpuBackend::Metal};
        SceneTemporalCoordinator coordinator;
        std::array<std::shared_ptr<Feature>, size_t(TemporalViewId::Count)> features;
        rendering::TensorSceneTextureKernels conversion{core::GpuBackend::Metal};
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
            const auto pitch = [](uint32_t bytes) { return (bytes + 255u) & ~255u; };
            f->pitches = {pitch(input.x * 8), pitch(input.x * 4), pitch(input.x * 8), pitch(output.x * 8)};
            f->packed_color = core::Tensor::empty({size_t(input.y), size_t(f->pitches[0])}, core::Device::GPU, core::DataType::UInt8);
            f->packed_result = core::Tensor::empty({size_t(output.y), size_t(f->pitches[3])}, core::Device::GPU, core::DataType::UInt8);
            if (backend == SceneUpscalerBackend::MetalFxTemporal) {
                f->packed_depth = core::Tensor::empty({size_t(input.y), size_t(f->pitches[1])}, core::Device::GPU, core::DataType::UInt8);
                f->packed_motion = core::Tensor::empty({size_t(input.y), size_t(f->pitches[2])}, core::Device::GPU, core::DataType::UInt8);
            }
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
            !core::tensor_supports_metal_access(*r.depth) ||
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
                auto f = impl_->features[slot];
                if (!f || f->backend != backend || f->input != r.render_extent || f->output != r.output_extent) {
                    f = impl_->makeFeature(backend, r.render_extent, r.output_extent);
                    impl_->coordinator.reset(r.view, TemporalResetReason::RenderSize);
                }
                auto& motion = f->motion_tensor;
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
                auto result = f->acquireOutput();
                const auto jitter = sceneTemporalJitterPixels(prepared.frame.current_jitter, r.render_extent, false);
                rendering::SceneTextureParameters params{
                    .extents = {uint32_t(r.render_extent.x), uint32_t(r.render_extent.y), uint32_t(r.output_extent.x), uint32_t(r.output_extent.y)},
                    .layout = {uint32_t(r.color->size(1)), r.color->dtype() == core::DataType::Float32 ? 1u : 0u, r.flip_y ? 1u : 0u, temporal ? 1u : 0u},
                    .pitches = f->pitches,
                    .depth = {r.frame.view.near_plane, r.frame.view.far_plane, r.frame.view.orthographic ? 1.f : 0.f, temporal ? float(r.depth->size(1)) : 0.f},
                    .jitter = {temporal ? jitter.x : 0.f, temporal ? jitter.y : 0.f, 0, 0}};
                if (auto packed = impl_->conversion.dispatch(false, params,
                    {r.color.get(), temporal ? r.depth.get() : nullptr, temporal ? &motion : nullptr,
                     &f->packed_color, temporal ? &f->packed_depth : nullptr, temporal ? &f->packed_motion : nullptr, nullptr, nullptr}); !packed)
                    throw std::runtime_error(lfs::format_for_developer(packed.error()));
                const std::array<const core::Tensor*, 3> inputs{&f->packed_color, temporal ? &f->packed_depth : nullptr, temporal ? &f->packed_motion : nullptr};
                const std::array<core::Tensor*, 1> outputs{&f->packed_result};
                auto command = impl_->reader.submitWrites(inputs, outputs,
                    [&](id<MTLCommandBuffer> command, auto in, auto out) {
                        auto blit = [command blitCommandEncoder];
                        if (!blit) throw std::runtime_error("MetalFX input blit unavailable");
                        const auto copy = [&](size_t i, id<MTLTexture> destination) {
                            [blit copyFromBuffer:in[i].buffer sourceOffset:in[i].offset
                                sourceBytesPerRow:f->pitches[i] sourceBytesPerImage:f->pitches[i]*r.render_extent.y
                                sourceSize:MTLSizeMake(r.render_extent.x, r.render_extent.y, 1)
                                toTexture:destination destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
                        };
                        copy(0, f->color);
                        if (temporal) { copy(1, f->depth); copy(2, f->motion); }
                        [blit endEncoding];
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
                        blit = [command blitCommandEncoder];
                        if (!blit) throw std::runtime_error("MetalFX output blit unavailable");
                        [blit copyFromTexture:f->result sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                            sourceSize:MTLSizeMake(r.output_extent.x, r.output_extent.y, 1)
                            toBuffer:out[0].buffer destinationOffset:out[0].offset
                            destinationBytesPerRow:f->pitches[3] destinationBytesPerImage:f->pitches[3]*r.output_extent.y];
                        [blit endEncoding];
                        // Retain the old feature through completion, including a resize/reset/release.
                        [command addCompletedHandler:^(id<MTLCommandBuffer>) { (void)f; }];
                    });
                if (!command) throw std::runtime_error("MetalFX command was not submitted");
                if (auto unpacked = impl_->conversion.dispatch(true, params,
                    {r.color.get(), nullptr, nullptr, &f->packed_result, nullptr, nullptr, result.get(), nullptr}); !unpacked)
                    throw std::runtime_error(lfs::format_for_developer(unpacked.error()));
                if (temporal && !impl_->coordinator.commit(prepared, SceneHistoryStorage::MetalFx))
                    throw std::runtime_error("MetalFX history commit failed");
                if (!temporal) impl_->coordinator.reset(r.view);
                impl_->features[slot] = std::move(f);
                return TensorSceneTemporalResult{.color = std::move(result), .sequence = temporal ? prepared.frame.sequence + 1 : 0,
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
