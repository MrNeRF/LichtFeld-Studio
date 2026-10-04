/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../gpu_program.hpp"
#include "metal_context.hpp"
#include "core/tensor_metal_reader.hpp"
#include <format>
#include <map>
#include <mutex>

namespace lfs::core::internal {
    namespace {
        using Module = GpuKernelModule;
        API_AVAILABLE_BEGIN(macos(26.0))

        class Program final : public GpuProgram {
        public:
            explicit Program(std::span<const Module::Entry> entries) {
                for (const auto& entry : entries) {
                    if (entry.backend != GpuBackend::Metal) continue;
                    MTLCompileOptions* options = [MTLCompileOptions new];
                    options.languageVersion = MTLLanguageVersion3_1;
                    options.mathMode = MTLMathModeSafe;
                    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
                    NSString* source = [[NSString alloc] initWithBytes:entry.code.data() length:entry.code.size() encoding:NSUTF8StringEncoding];
                    NSError* error = nil;
                    id<MTLLibrary> library = [reader_.device() newLibraryWithSource:source options:options error:&error];
                    NSString* name = [[NSString alloc] initWithBytes:entry.name.data() length:entry.name.size() encoding:NSUTF8StringEncoding];
                    id<MTLFunction> function = [library newFunctionWithName:name];
                    if (!function)
                        throw TensorError(std::format("Slang Metal entry '{}' could not load: {}", entry.name,
                                                      error ? error.localizedDescription.UTF8String : "entry missing"));
                    functions_.emplace(std::pair{std::string(entry.name), entry.stage}, function);
                }
            }

            bool supports_raster() const override { return true; }
            uint64_t address(const Tensor& tensor) override {
                const auto located = metal::acquire_context()->locate(storage_ref(tensor));
                return located.address + located.offset;
            }

            void dispatch(const Module::Dispatch& launch, const ProgramArguments& arguments) override {
                std::lock_guard lock(mutex_);
                auto& pipeline = compute_[std::string(launch.function)];
                if (!pipeline) {
                    NSError* error = nil;
                    pipeline = [reader_.device() newComputePipelineStateWithFunction:functions_.at({std::string(launch.function), Module::Stage::Compute}) error:&error];
                    if (!pipeline)
                        throw TensorError(std::format("Slang Metal compute pipeline '{}': {}", launch.function, error.localizedDescription.UTF8String));
                }
                const auto encode = [&](id<MTLCommandBuffer> command, std::span<const MetalTensorView> reads, std::span<const MetalTensorView> writes) {
                    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:pipeline];
                    if (!arguments.parameters.empty())
                        [encoder setBytes:arguments.parameters.data() length:arguments.parameters.size() atIndex:0];
                    for (const auto& view : reads) [encoder useResource:view.buffer usage:MTLResourceUsageRead];
                    for (const auto& view : writes) [encoder useResource:view.buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                    [encoder dispatchThreadgroups:MTLSizeMake(launch.groups[0], launch.groups[1], launch.groups[2])
                             threadsPerThreadgroup:MTLSizeMake(launch.group[0], launch.group[1], launch.group[2])];
                    [encoder endEncoding];
                };
                submit(arguments, {}, encode);
            }

            void draw(const Module::Draw& draw, const ProgramArguments& arguments) override {
                std::lock_guard lock(mutex_);
                const auto width = draw.color->size(1), height = draw.color->size(0);
                const bool bytes = draw.color->dtype() == DataType::UInt8;
                const MTLPixelFormat format = bytes ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatRGBA32Float;
                const auto color = texture(width, height, format);
                const auto depth = draw.depth ? texture(width, height, MTLPixelFormatDepth32Float) : nil;
                const auto pipeline = render_pipeline(draw, format);
                MTLDepthStencilDescriptor* depth_desc = [MTLDepthStencilDescriptor new];
                depth_desc.depthCompareFunction = !draw.depth ? MTLCompareFunctionAlways :
                                                   draw.depth_compare == Module::Compare::Less ? MTLCompareFunctionLess :
                                                   draw.depth_compare == Module::Compare::LessEqual ? MTLCompareFunctionLessEqual : MTLCompareFunctionAlways;
                depth_desc.depthWriteEnabled = draw.depth && draw.depth_write;
                const auto depth_state = [reader_.device() newDepthStencilStateWithDescriptor:depth_desc];
                std::vector<Tensor*> attachments{draw.color};
                if (draw.depth) attachments.push_back(draw.depth);
                submit(arguments, attachments, [&](id<MTLCommandBuffer> command, std::span<const MetalTensorView> reads, std::span<const MetalTensorView> writes) {
                    const auto color_view = writes[arguments.writes.size()];
                    const auto depth_view = draw.depth ? writes.back() : MetalTensorView{};
                    const auto color_row = width * (bytes ? 4 : 16), depth_row = width * 4;
                    id<MTLBlitCommandEncoder> upload = [command blitCommandEncoder];
                    if (!draw.clear_color)
                        [upload copyFromBuffer:color_view.buffer sourceOffset:color_view.offset sourceBytesPerRow:color_row sourceBytesPerImage:color_row * height
                                    sourceSize:MTLSizeMake(width, height, 1) toTexture:color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
                    if (draw.depth && !draw.clear_depth)
                        [upload copyFromBuffer:depth_view.buffer sourceOffset:depth_view.offset sourceBytesPerRow:depth_row sourceBytesPerImage:depth_row * height
                                    sourceSize:MTLSizeMake(width, height, 1) toTexture:depth destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
                    [upload endEncoding];

                    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                    pass.colorAttachments[0].texture = color;
                    pass.colorAttachments[0].loadAction = draw.clear_color ? MTLLoadActionClear : MTLLoadActionLoad;
                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                    pass.colorAttachments[0].clearColor = MTLClearColorMake(draw.color_clear[0], draw.color_clear[1], draw.color_clear[2], draw.color_clear[3]);
                    if (depth) {
                        pass.depthAttachment.texture = depth;
                        pass.depthAttachment.loadAction = draw.clear_depth ? MTLLoadActionClear : MTLLoadActionLoad;
                        pass.depthAttachment.storeAction = MTLStoreActionStore;
                        pass.depthAttachment.clearDepth = draw.depth_clear;
                    }
                    id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
                    [encoder setRenderPipelineState:pipeline];
                    [encoder setDepthStencilState:depth_state];
                    [encoder setCullMode:MTLCullModeNone];
                    if (!arguments.parameters.empty()) {
                        [encoder setVertexBytes:arguments.parameters.data() length:arguments.parameters.size() atIndex:0];
                        [encoder setFragmentBytes:arguments.parameters.data() length:arguments.parameters.size() atIndex:0];
                    }
                    for (const auto& view : reads) [encoder useResource:view.buffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex | MTLRenderStageFragment];
                    for (size_t i = 0; i < arguments.writes.size(); ++i)
                        [encoder useResource:writes[i].buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite stages:MTLRenderStageVertex | MTLRenderStageFragment];
                    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:draw.first_vertex vertexCount:draw.vertex_count instanceCount:draw.instance_count];
                    [encoder endEncoding];

                    id<MTLBlitCommandEncoder> download = [command blitCommandEncoder];
                    [download copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
                                      toBuffer:color_view.buffer destinationOffset:color_view.offset destinationBytesPerRow:color_row destinationBytesPerImage:color_row * height];
                    if (depth)
                        [download copyFromTexture:depth sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
                                          toBuffer:depth_view.buffer destinationOffset:depth_view.offset destinationBytesPerRow:depth_row destinationBytesPerImage:depth_row * height];
                    [download endEncoding];
                });
            }

        private:
            void submit(const ProgramArguments& arguments, std::span<Tensor* const> attachments, const MetalTensorReader::EncodeWrite& encode) {
                std::vector<Tensor*> writes;
                for (auto* tensor : arguments.writes) writes.push_back(const_cast<Tensor*>(tensor));
                writes.insert(writes.end(), attachments.begin(), attachments.end());
                if (writes.empty()) {
                    (void)reader_.submit(arguments.reads, [&](id<MTLCommandBuffer> command, std::span<const MetalTensorView> reads) { encode(command, reads, {}); });
                } else {
                    (void)reader_.submitWrites(arguments.reads, writes, encode);
                }
            }

            id<MTLTexture> texture(size_t width, size_t height, MTLPixelFormat format) {
                MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
                desc.storageMode = MTLStorageModePrivate;
                desc.usage = MTLTextureUsageRenderTarget;
                const auto result = [reader_.device() newTextureWithDescriptor:desc];
                if (!result) throw TensorError(std::format("Metal raster attachment allocation failed: {}x{}, format {}", width, height, uint64_t(format)));
                return result;
            }

            id<MTLRenderPipelineState> render_pipeline(const Module::Draw& draw, MTLPixelFormat format) {
                const auto key = std::tuple{std::string(draw.vertex), std::string(draw.fragment), uint64_t(format), draw.blend, draw.depth != nullptr};
                auto& pipeline = render_[key];
                if (pipeline) return pipeline;
                MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
                desc.vertexFunction = functions_.at({std::string(draw.vertex), Module::Stage::Vertex});
                desc.fragmentFunction = functions_.at({std::string(draw.fragment), Module::Stage::Fragment});
                auto* attachment = desc.colorAttachments[0];
                attachment.pixelFormat = format;
                attachment.blendingEnabled = draw.blend != Module::Blend::Opaque;
                attachment.sourceRGBBlendFactor = draw.blend == Module::Blend::StraightAlpha ? MTLBlendFactorSourceAlpha : MTLBlendFactorOne;
                attachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
                attachment.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                desc.depthAttachmentPixelFormat = draw.depth ? MTLPixelFormatDepth32Float : MTLPixelFormatInvalid;
                NSError* error = nil;
                pipeline = [reader_.device() newRenderPipelineStateWithDescriptor:desc error:&error];
                if (!pipeline) throw TensorError(std::format("Slang Metal raster pipeline '{}'/ '{}': {}", draw.vertex, draw.fragment, error.localizedDescription.UTF8String));
                return pipeline;
            }

            MetalTensorReader reader_;
            std::mutex mutex_;
            std::map<std::pair<std::string, Module::Stage>, id<MTLFunction>> functions_;
            std::map<std::string, id<MTLComputePipelineState>> compute_;
            std::map<std::tuple<std::string, std::string, uint64_t, Module::Blend, bool>, id<MTLRenderPipelineState>> render_;
        };
        API_AVAILABLE_END
    } // namespace

    std::unique_ptr<GpuProgram> make_metal_program(std::span<const GpuKernelModule::Entry> entries) {
        if (@available(macOS 26.0, *)) return std::make_unique<Program>(entries);
        throw TensorError("Slang Metal programs require macOS 26 and the Metal tensor backend");
    }
} // namespace lfs::core::internal
