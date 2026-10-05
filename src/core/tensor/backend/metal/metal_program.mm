/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../gpu_program.hpp"
#include "metal_context.hpp"
#include "core/tensor_metal_reader.hpp"
#include <format>
#include <map>
#include <tuple>
#include <mutex>

namespace lfs::core::internal {
    namespace {
        using Module = GpuKernelModule;
        API_AVAILABLE_BEGIN(macos(26.0))

        // Every program submits on one queue, so a raster attachment can be
        // shared by all of them: Metal orders a reused texture after its
        // previous pass on the same queue. Intentionally never destroyed.
        struct SharedRaster {
            MetalTensorReader reader;
            std::mutex mutex;
            std::map<std::tuple<size_t, size_t, uint64_t>, id<MTLTexture>> attachments;
        };
        SharedRaster& shared_raster() {
            static auto* shared = new SharedRaster;
            return *shared;
        }

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

            // Compute records on the tensor queue's batched encoder, like the
            // tensor ops: no command buffer or queue hop per dispatch.
            void dispatch(const Module::Dispatch& launch, const ProgramArguments& arguments) override {
                id<MTLComputePipelineState> pipeline;
                {
                    std::lock_guard lock(mutex_);
                    auto& cached = compute_[std::string(launch.function)];
                    if (!cached) {
                        NSError* error = nil;
                        cached = [context_->device() newComputePipelineStateWithFunction:functions_.at({std::string(launch.function), Module::Stage::Compute}) error:&error];
                        if (!cached)
                            throw TensorError(std::format("Slang Metal compute pipeline '{}': {}", launch.function, error.localizedDescription.UTF8String));
                    }
                    pipeline = cached;
                }
                std::vector<StorageRef> uses;
                uses.reserve(arguments.reads.size() + arguments.writes.size());
                for (const auto* list : {&arguments.reads, &arguments.writes})
                    for (const Tensor* tensor : *list)
                        uses.push_back(storage_ref(*tensor));
                context_->dispatch(uses, {.pipeline = pipeline,
                                          .buffers = {},
                                          .params = arguments.parameters,
                                          .grid = MTLSizeMake(launch.groups[0], launch.groups[1], launch.groups[2]),
                                          .group_size = MTLSizeMake(launch.group[0], launch.group[1], launch.group[2])});
            }

            void draw(std::span<const Module::Draw> draws, std::span<const ProgramArguments> arguments) override {
                std::lock_guard lock(mutex_);
                const auto& first = draws.front();
                const auto width = first.color->size(1), height = first.color->size(0);
                const bool bytes = first.color->dtype() == DataType::UInt8;
                const MTLPixelFormat format = bytes ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatRGBA32Float;
                const auto depth = first.depth ? attachment(width, height, MTLPixelFormatDepth32Float) : nil;
                std::vector<id<MTLRenderPipelineState>> pipelines;
                std::vector<id<MTLDepthStencilState>> depth_states;
                pipelines.reserve(draws.size());
                depth_states.reserve(draws.size());
                ProgramArguments merged;
                for (size_t i = 0; i < draws.size(); ++i) {
                    pipelines.push_back(render_pipeline(draws[i], format));
                    depth_states.push_back(depth_state(draws[i]));
                    merged.reads.insert(merged.reads.end(), arguments[i].reads.begin(), arguments[i].reads.end());
                    merged.writes.insert(merged.writes.end(), arguments[i].writes.begin(), arguments[i].writes.end());
                }
                std::vector<Tensor*> attachments{first.color};
                if (first.depth) attachments.push_back(first.depth);
                submit(merged, attachments, [&](id<MTLCommandBuffer> command, std::span<const MetalTensorView> reads, std::span<const MetalTensorView> writes) {
                    const auto color_view = writes[merged.writes.size()];
                    const auto depth_view = first.depth ? writes.back() : MetalTensorView{};
                    const auto color_row = width * (bytes ? 4 : 16), depth_row = width * 4;
                    // Render straight into the color tensor when its storage can
                    // back a linear texture; otherwise through a cached attachment.
                    id<MTLTexture> color = linear_texture(color_view, width, height, format, color_row);
                    const bool copy_color = color == nil;
                    // Attachments are reused: every submission shares one queue, so
                    // Metal orders a reused texture after its previous pass.
                    if (copy_color)
                        color = attachment(width, height, format);
                    if ((copy_color && !first.clear_color) || (first.depth && !first.clear_depth)) {
                        id<MTLBlitCommandEncoder> upload = [command blitCommandEncoder];
                        if (copy_color && !first.clear_color)
                            [upload copyFromBuffer:color_view.buffer sourceOffset:color_view.offset sourceBytesPerRow:color_row sourceBytesPerImage:color_row * height
                                        sourceSize:MTLSizeMake(width, height, 1) toTexture:color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
                        if (first.depth && !first.clear_depth)
                            [upload copyFromBuffer:depth_view.buffer sourceOffset:depth_view.offset sourceBytesPerRow:depth_row sourceBytesPerImage:depth_row * height
                                        sourceSize:MTLSizeMake(width, height, 1) toTexture:depth destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
                        [upload endEncoding];
                    }

                    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                    pass.colorAttachments[0].texture = color;
                    pass.colorAttachments[0].loadAction = first.clear_color ? MTLLoadActionClear : MTLLoadActionLoad;
                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                    pass.colorAttachments[0].clearColor = MTLClearColorMake(first.color_clear[0], first.color_clear[1], first.color_clear[2], first.color_clear[3]);
                    if (depth) {
                        pass.depthAttachment.texture = depth;
                        pass.depthAttachment.loadAction = first.clear_depth ? MTLLoadActionClear : MTLLoadActionLoad;
                        pass.depthAttachment.storeAction = MTLStoreActionStore;
                        pass.depthAttachment.clearDepth = first.depth_clear;
                    }
                    id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
                    // Metal's viewport keeps NDC +Y up, so CCW in NDC is CCW on screen.
                    [encoder setFrontFacingWinding:MTLWindingCounterClockwise];
                    for (const auto& view : reads) [encoder useResource:view.buffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex | MTLRenderStageFragment];
                    for (size_t i = 0; i < merged.writes.size(); ++i)
                        [encoder useResource:writes[i].buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite stages:MTLRenderStageVertex | MTLRenderStageFragment];
                    const MTLScissorRect full{0, 0, static_cast<NSUInteger>(width), static_cast<NSUInteger>(height)};
                    for (size_t i = 0; i < draws.size(); ++i) {
                        const auto& draw = draws[i];
                        if (draw.vertex_count == 0) continue;
                        const auto viewport = draw.viewport.value_or(Module::Viewport{0, 0, float(width), float(height)});
                        [encoder setViewport:MTLViewport{viewport.x, viewport.y, viewport.width, viewport.height, 0, 1}];
                        [encoder setRenderPipelineState:pipelines[i]];
                        [encoder setCullMode:draw.cull == Module::Cull::Back ? MTLCullModeBack
                                             : draw.cull == Module::Cull::Front ? MTLCullModeFront
                                                                                : MTLCullModeNone];
                        [encoder setDepthStencilState:depth_states[i]];
                        if (draw.scissor) {
                            const auto& rect = *draw.scissor;
                            [encoder setScissorRect:MTLScissorRect{rect.x, rect.y, rect.width, rect.height}];
                        } else {
                            [encoder setScissorRect:full];
                        }
                        const auto& parameters = arguments[i].parameters;
                        if (!parameters.empty()) {
                            [encoder setVertexBytes:parameters.data() length:parameters.size() atIndex:0];
                            [encoder setFragmentBytes:parameters.data() length:parameters.size() atIndex:0];
                        }
                        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:draw.first_vertex vertexCount:draw.vertex_count instanceCount:draw.instance_count];
                    }
                    [encoder endEncoding];

                    if (copy_color || depth) {
                        id<MTLBlitCommandEncoder> download = [command blitCommandEncoder];
                        if (copy_color)
                            [download copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
                                              toBuffer:color_view.buffer destinationOffset:color_view.offset destinationBytesPerRow:color_row destinationBytesPerImage:color_row * height];
                        if (depth)
                            [download copyFromTexture:depth sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
                                              toBuffer:depth_view.buffer destinationOffset:depth_view.offset destinationBytesPerRow:depth_row destinationBytesPerImage:depth_row * height];
                        [download endEncoding];
                    }
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

            id<MTLTexture> attachment(size_t width, size_t height, MTLPixelFormat format) {
                auto& shared = shared_raster();
                std::lock_guard lock(shared.mutex);
                const auto key = std::tuple{width, height, uint64_t(format)};
                // Resizes leave stale sizes behind; in-flight commands retain theirs.
                if (!shared.attachments.contains(key) && shared.attachments.size() >= 8)
                    shared.attachments.clear();
                auto& cached = shared.attachments[key];
                if (!cached) cached = texture(width, height, format);
                return cached;
            }

            // A render target aliasing the tensor's own storage, or nil when the
            // storage misses the device's linear-texture alignment.
            id<MTLTexture> linear_texture(const MetalTensorView& view, size_t width, size_t height, MTLPixelFormat format, size_t row) {
                const NSUInteger alignment = [reader_.device() minimumLinearTextureAlignmentForPixelFormat:format];
                if (alignment == 0 || view.offset % alignment || row % alignment)
                    return nil;
                MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
                desc.resourceOptions = view.buffer.resourceOptions;
                desc.usage = MTLTextureUsageRenderTarget;
                return [view.buffer newTextureWithDescriptor:desc offset:view.offset bytesPerRow:row];
            }

            id<MTLDepthStencilState> depth_state(const Module::Draw& draw) {
                const auto key = std::tuple{draw.depth != nullptr, draw.depth_compare, draw.depth_write};
                auto& state = depth_states_[key];
                if (state) return state;
                MTLDepthStencilDescriptor* desc = [MTLDepthStencilDescriptor new];
                desc.depthCompareFunction = !draw.depth ? MTLCompareFunctionAlways :
                                            draw.depth_compare == Module::Compare::Less ? MTLCompareFunctionLess :
                                            draw.depth_compare == Module::Compare::LessEqual ? MTLCompareFunctionLessEqual : MTLCompareFunctionAlways;
                desc.depthWriteEnabled = draw.depth && draw.depth_write;
                state = [reader_.device() newDepthStencilStateWithDescriptor:desc];
                return state;
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

            std::shared_ptr<metal::Context> context_ = metal::acquire_context();
            MetalTensorReader& reader_ = shared_raster().reader;
            std::mutex mutex_;
            std::map<std::pair<std::string, Module::Stage>, id<MTLFunction>> functions_;
            std::map<std::string, id<MTLComputePipelineState>> compute_;
            std::map<std::tuple<std::string, std::string, uint64_t, Module::Blend, bool>, id<MTLRenderPipelineState>> render_;
            std::map<std::tuple<bool, Module::Compare, bool>, id<MTLDepthStencilState>> depth_states_;
        };
        API_AVAILABLE_END
    } // namespace

    std::unique_ptr<GpuProgram> make_metal_program(std::span<const GpuKernelModule::Entry> entries) {
        if (@available(macOS 26.0, *)) return std::make_unique<Program>(entries);
        throw TensorError("Slang Metal programs require macOS 26 and the Metal tensor backend");
    }
} // namespace lfs::core::internal
