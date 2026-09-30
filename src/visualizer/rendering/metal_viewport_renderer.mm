/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_viewport_renderer.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_metal_reader.hpp"
#include "frame_budget.hpp"
#include "metal_present_source.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "splat_preprocessor.hpp"
#include "tile_rasterizer.hpp"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <vulkan/vulkan_metal.h>

namespace lfs::vis {
    namespace {
        using namespace rendering::metal;
        using Slot = VksplatViewportRenderer::OutputSlot;
        lfs::Error nativeError(std::string message, lfs::ErrorCode code = lfs::ErrorCode::Internal,
                               core::SourceSite site = LFS_SOURCE_SITE_CURRENT()) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = std::move(message),
                .detection = site,
            });
        }
        lfs::Error nativeError(const std::exception& error,
                               core::SourceSite site = LFS_SOURCE_SITE_CURRENT()) {
            if (const auto* structured = dynamic_cast<const lfs::Exception*>(&error))
                return structured->error();
            const auto code = dynamic_cast<const std::invalid_argument*>(&error) ? lfs::ErrorCode::InvalidArgument
                              : dynamic_cast<const std::bad_alloc*>(&error)      ? lfs::ErrorCode::ResourceExhausted
                                                                                 : lfs::ErrorCode::Internal;
            return nativeError(error.what(), code, site);
        }
        std::atomic<uint64_t> generation{uint64_t{1} << 63};
        std::atomic<uint64_t> native_ticket_serial{0};
        uint64_t reserveNativeTicket() {
            auto value = native_ticket_serial.load(std::memory_order_relaxed);
            for (;;) {
                if (value == ((uint64_t{1} << 63) - 1))
                    throw std::runtime_error("Native readback ticket identity exhausted");
                if (native_ticket_serial.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
                    return (uint64_t{1} << 63) | (value + 1);
            }
        }
        void check(VkResult r, const char* label) {
            if (r != VK_SUCCESS)
                throw std::runtime_error(std::string(label) + ": " + std::to_string(r));
        }
        simd_float4x4 matrix(const glm::mat4& m) {
            simd_float4x4 result;
            static_assert(sizeof(result) == sizeof(m));
            std::memcpy(&result, &m, sizeof(m));
            return result;
        }
        struct PresentParameters {
            float exposure;
            uint32_t tone, transparent, has_previous;
            float depth_min, depth_max;
            uint32_t depth_view, depth_mode;
            simd_float4 background;
            simd_uint4 capture;
        };
        struct PointParameters {
            simd_float4x4 view_projection, view, crop_to_local;
            simd_float4 crop_min, crop_max, voxel_focal_ortho;
            simd_uint4 counts;
        };
        static_assert(sizeof(PointParameters) == 256);
        struct Image {
            VkDevice device = VK_NULL_HANDLE;
            id<MTLTexture> texture;
            VkImage image = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            ~Image() {
                if (view)
                    vkDestroyImageView(device, view, nullptr);
                if (image)
                    vkDestroyImage(device, image, nullptr);
            }
            void init(VulkanContext& context, id<MTLDevice> metal, uint32_t w, uint32_t h,
                      MTLPixelFormat native_format, VkFormat format) {
                device = context.device();
                auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native_format width:w height:h mipmapped:NO];
                descriptor.storageMode = MTLStorageModePrivate;
                descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
                texture = [metal newTextureWithDescriptor:descriptor];
                if (!texture)
                    throw lfs::Exception(nativeError("Metal viewport texture allocation failed", lfs::ErrorCode::ResourceExhausted));
                VkImportMetalTextureInfoEXT imported{VK_STRUCTURE_TYPE_IMPORT_METAL_TEXTURE_INFO_EXT};
                imported.plane = VK_IMAGE_ASPECT_PLANE_0_BIT;
                imported.mtlTexture = texture;
                VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                info.pNext = &imported;
                info.imageType = VK_IMAGE_TYPE_2D;
                info.format = format;
                info.extent = {w, h, 1};
                info.mipLevels = 1;
                info.arrayLayers = 1;
                info.samples = VK_SAMPLE_COUNT_1_BIT;
                info.tiling = VK_IMAGE_TILING_OPTIMAL;
                info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                check(vkCreateImage(device, &info, nullptr, &image), "Import native viewport texture");
                VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view_info.image = image;
                view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view_info.format = format;
                view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                check(vkCreateImageView(device, &view_info, nullptr, &view), "Native viewport image view");
                if (!context.transitionImageLayoutImmediate(image, VK_IMAGE_LAYOUT_UNDEFINED,
                                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {}))
                    throw std::runtime_error(context.lastError());
            }
        };
        struct Frame {
            glm::ivec2 size{};
            uint32_t count = 0, capacity = 0;
            uint64_t generation = 0, consumer_serial = 0, producer_value = 0;
            uint64_t request_key = 0;
            std::unique_ptr<RasterFrame> raster;
            id<MTLBuffer> projected, gut_geometry, objects, overlay_parameters, overlay_flags, overlay_nodes, selection_colors;
            id<MTLCommandBuffer> command;
            id<MTLTexture> point_depth;
            bool points = false;
            Image color, depth;
        };
    } // namespace
    struct MetalViewportRenderer::Impl {
        VulkanContext* context = nullptr;
        core::MetalTensorReader reader;
        SplatPreprocessor preprocessor{reader.device()};
        TileRasterizer rasterizer{reader.device()};
        id<MTLComputePipelineState> present;
        id<MTLRenderPipelineState> point_pipeline;
        id<MTLDepthStencilState> point_depth_state;
        id<MTLSharedEvent> event;
        VkSemaphore completion = VK_NULL_HANDLE;
        uint64_t serial = 0;
        std::array<std::array<std::unique_ptr<Frame>, 3>, 4> frames;
        std::array<Frame*, 4> latest{};
        std::array<size_t, 4> next{};
        std::array<uint32_t, 4> needed_capacity{};
        struct Readback {
            id<MTLBuffer> buffer;
            id<MTLCommandBuffer> command;
            id<MTLCommandBuffer> producer;
            void* destination = nullptr;
            glm::ivec2 size;
            size_t width, channels;
            int x, y;
            bool depth, floating;
        };
        mutable std::mutex readback_mutex;
        mutable std::map<uint64_t, Readback> readbacks;
        mutable uint64_t next_readback = 0;
        id<MTLCommandQueue> readback_queue = [reader.device() newCommandQueue];
        id<MTLSharedEvent> readback_event = [reader.device() newSharedEvent];
        ~Impl() {
            // Teardown only: the event covers every native producer before device idle.
            if (context && completion) {
                try {
                    wait(serial);
                    if (!context->deviceWaitIdle())
                        throw std::runtime_error(context->lastError());
                } catch (...) {
                    // Never destroy textures still referenced by an unretired device.
                    for (auto& group : frames)
                        for (auto& frame : group)
                            (void)frame.release();
                    return;
                }
                for (auto& group : frames)
                    for (auto& frame : group)
                        frame.reset();
                vkDestroySemaphore(context->device(), completion, nullptr);
            }
        }
        void wait(uint64_t value) const {
            if (!value)
                return;
            VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            info.semaphoreCount = 1;
            info.pSemaphores = &completion;
            info.pValues = &value;
            check(vkWaitSemaphores(context->device(), &info, 5000000000ull), "Native viewport completion");
        }
        void initialize(VulkanContext& ctx) {
            if (context) {
                if (context != &ctx)
                    throw std::invalid_argument("Metal viewport context changed without reset");
                return;
            }
            auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(vkGetDeviceProcAddr(ctx.device(), "vkExportMetalObjectsEXT"));
            if (!export_objects)
                throw std::runtime_error("Presentation does not support VK_EXT_metal_objects");
            VkExportMetalDeviceInfoEXT native_device{VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT};
            VkExportMetalObjectsInfoEXT exports{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
            exports.pNext = &native_device;
            export_objects(ctx.device(), &exports);
            if (!native_device.mtlDevice || native_device.mtlDevice.registryID != reader.device().registryID)
                throw std::runtime_error("Metal tensors and desktop presentation must use the same GPU");
            NSError* error = nil;
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion2_4;
            auto library = [reader.device() newLibraryWithSource:[NSString stringWithUTF8String:kMetalPresentSource] options:options error:&error];
            if (!library)
                throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal pipeline creation failed");
            present = [reader.device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"present_viewer"] error:&error];
            if (!present)
                throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal pipeline creation failed");
            auto point_descriptor = [MTLRenderPipelineDescriptor new];
            point_descriptor.vertexFunction = [library newFunctionWithName:@"point_vertex"];
            point_descriptor.fragmentFunction = [library newFunctionWithName:@"point_fragment"];
            point_descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            point_descriptor.colorAttachments[1].pixelFormat = MTLPixelFormatR32Float;
            point_descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
            point_pipeline = [reader.device() newRenderPipelineStateWithDescriptor:point_descriptor error:&error];
            if (!point_pipeline)
                throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal pipeline creation failed");
            auto depth_descriptor = [MTLDepthStencilDescriptor new];
            depth_descriptor.depthCompareFunction = MTLCompareFunctionLess;
            depth_descriptor.depthWriteEnabled = YES;
            point_depth_state = [reader.device() newDepthStencilStateWithDescriptor:depth_descriptor];
            if (!point_depth_state)
                throw std::runtime_error("Metal point depth state unavailable");
            event = [reader.device() newSharedEvent];
            if (!event)
                throw lfs::Exception(nativeError("Metal presentation timeline allocation failed", lfs::ErrorCode::ResourceExhausted));
            VkImportMetalSharedEventInfoEXT imported{VK_STRUCTURE_TYPE_IMPORT_METAL_SHARED_EVENT_INFO_EXT};
            imported.mtlSharedEvent = event;
            VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            type.pNext = &imported;
            type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            info.pNext = &type;
            check(vkCreateSemaphore(ctx.device(), &info, nullptr, &completion), "Native presentation timeline");
            context = &ctx;
        }
        Frame& acquire(Slot output, const rendering::ViewportRenderRequest& request, uint32_t count, bool points = false) {
            const size_t slot = static_cast<size_t>(output);
            auto& frame = frames[slot][next[slot]++ % 3];
            uint32_t capacity = std::max(needed_capacity[slot], static_cast<uint32_t>(std::min<uint64_t>(uint64_t(count) * 16 + 4096, 16u * 1024u * 1024u)));
            // A failed encode may leave a reservation without a submitted producer.
            // Such a frame has no readable GPU status and must be recreated.
            if (frame && !frame->command) {
                if (latest[slot] == frame.get())
                    latest[slot] = nullptr;
                frame.reset();
            }
            if (frame) {
                wait(frame->producer_value);
                if (frame->command.status == MTLCommandBufferStatusError)
                    throw std::runtime_error(frame->command.error.localizedDescription.UTF8String ?: "Metal command failed");
                if (!context->waitForRetiredFrameSubmitSerial(frame->consumer_serial))
                    throw std::runtime_error(context->lastError());
                if (frame->raster && frame->raster->busy())
                    [frame->command waitUntilCompleted];
                const auto status = frame->raster ? frame->raster->status() : RasterStatus{};
                if (status.error != RasterError::None) {
                    if (status.required_instances > std::numeric_limits<uint32_t>::max())
                        throw std::runtime_error("Metal viewport instance count exceeds 32-bit indexing");
                    capacity = std::max(capacity, static_cast<uint32_t>(status.required_instances));
                }
                if (frame->points == points && frame->size == request.frame_view.size && frame->count >= count && frame->capacity >= capacity)
                    return *frame;
                if (latest[slot] == frame.get())
                    latest[slot] = nullptr;
                frame.reset();
            }
            // Account for all Metal allocations on the shared device, including
            // resident tensors and Vulkan presentation. Fail before a large growth
            // can exhaust unified memory; retain the last completed output.
            const auto device = reader.device();
            const auto reservation = frameReservationBytes(request.frame_view.size.x,
                                                           request.frame_view.size.y, count, capacity, points);
            if (!frameFitsWorkingSet(device.currentAllocatedSize, reservation, device.recommendedMaxWorkingSetSize))
                throw lfs::Exception(nativeError("Metal viewport reservation exceeds the recommended GPU working set", lfs::ErrorCode::ResourceExhausted));
            frame = std::make_unique<Frame>();
            auto& f = *frame;
            f.size = request.frame_view.size;
            f.count = count;
            f.capacity = capacity;
            f.generation = ++generation;
            f.points = points;
            if (!points) {
                f.raster = std::make_unique<RasterFrame>(device, f.size.x, f.size.y, count, capacity);
                f.projected = [device newBufferWithLength:std::max<size_t>(16, size_t(count) * sizeof(ProjectedSplat)) options:MTLResourceStorageModePrivate];
                if (!f.projected)
                    throw lfs::Exception(nativeError("Metal projected buffer allocation failed", lfs::ErrorCode::ResourceExhausted));
            } else {
                auto depth_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:f.size.x height:f.size.y mipmapped:NO];
                depth_descriptor.storageMode = MTLStorageModePrivate;
                depth_descriptor.usage = MTLTextureUsageRenderTarget;
                f.point_depth = [device newTextureWithDescriptor:depth_descriptor];
                if (!f.point_depth)
                    throw lfs::Exception(nativeError("Metal point depth allocation failed", lfs::ErrorCode::ResourceExhausted));
            }
            f.color.init(*context, device, f.size.x, f.size.y, MTLPixelFormatRGBA8Unorm, VK_FORMAT_R8G8B8A8_UNORM);
            f.depth.init(*context, device, f.size.x, f.size.y, MTLPixelFormatR32Float, VK_FORMAT_R32_SFLOAT);
            return f;
        }
        id<MTLBuffer> readTexture(Frame& f, Image& image, size_t bytes_per_pixel, glm::ivec2 pixel = {-1, -1}) const {
            wait(f.producer_value);
            if (f.command.status == MTLCommandBufferStatusError)
                throw std::runtime_error(f.command.error.localizedDescription.UTF8String ?: "Metal command failed");
            auto queue = [reader.device() newCommandQueue];
            auto command = [queue commandBuffer];
            const bool sample = pixel.x >= 0;
            const size_t width = sample ? 1 : f.size.x, height = sample ? 1 : f.size.y;
            const size_t row = width * bytes_per_pixel;
            auto result = [reader.device() newBufferWithLength:row * height options:MTLResourceStorageModeShared];
            if (!queue || !command || !result)
                throw lfs::Exception(nativeError("Metal readback allocation failed", lfs::ErrorCode::ResourceExhausted));
            auto blit = [command blitCommandEncoder];
            [blit copyFromTexture:image.texture
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(sample ? pixel.x : 0, sample ? pixel.y : 0, 0)
                              sourceSize:MTLSizeMake(width, height, 1)
                                toBuffer:result
                       destinationOffset:0
                  destinationBytesPerRow:row
                destinationBytesPerImage:row * height];
            [blit endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error("Metal output readback failed");
            return result;
        }
    };
    MetalViewportRenderer::MetalViewportRenderer() : impl_(std::make_unique<Impl>()) {}
    MetalViewportRenderer::~MetalViewportRenderer() = default;
    bool MetalViewportRenderer::supportsPoints(const PointCloudVulkanRenderer::RenderRequest& r) {
        const auto resident = [](const core::Tensor* t) { return t && t->is_valid() && t->is_contiguous() && core::gpu_backend_of(*t) == core::GpuBackend::Metal; };
        if (!resident(r.positions) || !resident(r.colors) || r.positions->dtype() != core::DataType::Float32 ||
            r.colors->dtype() != core::DataType::Float32 || r.positions->ndim() != 2 || r.colors->ndim() != 2 ||
            r.positions->size(1) != 3 || r.colors->size(1) != 3 || r.positions->size(0) != r.colors->size(0))
            return false;
        const size_t count = r.positions->size(0);
        for (auto t : {r.selection_mask, r.preview_selection_mask, r.deleted_mask})
            if (t && t->is_valid() && (!resident(t) || t->bytes() < count || (t->dtype() != core::DataType::UInt8 && t->dtype() != core::DataType::Bool)))
                return false;
        if (r.transform_indices && r.transform_indices->is_valid() && (!resident(r.transform_indices) || r.transform_indices->bytes() < count * 4 || r.transform_indices->dtype() != core::DataType::Int32))
            return false;
        return count <= std::numeric_limits<uint32_t>::max() && r.size.x > 0 && r.size.y > 0;
    }
    lfs::Result<PointCloudVulkanRenderer::RenderResult> MetalViewportRenderer::renderPoints(
        VulkanContext& context, const PointCloudVulkanRenderer::RenderRequest& r, PointCloudVulkanRenderer::OutputSlot output) {
        try {
            if (!supportsPoints(r))
                throw std::invalid_argument("Unsupported native Metal point request");
            auto& i = *impl_;
            std::lock_guard lock(i.readback_mutex);
            i.initialize(context);
            for (auto frame : i.latest)
                if (frame)
                    frame->consumer_serial = std::max(frame->consumer_serial, context.lastFrameSubmitSerial());
            const auto slot = static_cast<Slot>(output);
            rendering::ViewportRenderRequest request;
            request.frame_view.size = r.size;
            auto& f = i.acquire(slot, request, uint32_t(r.positions->size(0)), true);
            const auto valid = [](const core::Tensor* t) { return t && t->is_valid(); };
            const size_t nodes = r.model_transforms ? r.model_transforms->size() : 0;
            if (nodes > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("Metal point object count exceeds indexing");
            const auto allocate = [&](id<MTLBuffer> __strong& buffer, size_t bytes) {
                if (!buffer || buffer.length < bytes)
                    buffer = [i.reader.device() newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
                if (!buffer)
                    throw lfs::Exception(nativeError("Metal point allocation failed", lfs::ErrorCode::ResourceExhausted));
            };
            allocate(f.objects, std::max<size_t>(1, nodes) * sizeof(SceneObject));
            auto objects = static_cast<SceneObject*>(f.objects.contents);
            for (size_t n = 0; n < nodes; ++n) {
                const bool visible = !r.node_visibility_mask || n >= r.node_visibility_mask->size() || (*r.node_visibility_mask)[n];
                objects[n] = {matrix((*r.model_transforms)[n]), {}, {uint32_t(visible), 0, 0, 0}};
            }
            const auto palette = r.selection_colors ? *r.selection_colors : rendering::defaultSelectionColorTable();
            allocate(f.selection_colors, sizeof(palette));
            std::memcpy(f.selection_colors.contents, palette.data(), sizeof(palette));
            uint32_t flags = (r.orthographic ? 8u : 0u) | (valid(r.transform_indices) ? 16u : 0u) |
                             (valid(r.selection_mask) ? 32u : 0u) | (valid(r.preview_selection_mask) ? 64u : 0u) |
                             (r.preview_selection_additive ? 128u : 0u) | (uint32_t(r.depth_visualization_mode) == 1 ? 256u : 0u) |
                             (valid(r.deleted_mask) ? 512u : 0u);
            PointParameters p{matrix(r.view_projection), matrix(r.view), matrix(glm::mat4(1)), {}, {}, {r.voxel_size * r.scaling_modifier, r.focal_y, float(r.size.y) / std::max(r.ortho_scale, 1e-5f), float(r.depth_view)}, {uint32_t(nodes), 0, flags, 511}};
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(context.physicalDevice(), &props);
            p.counts.w = uint32_t(std::min(511.f, props.limits.pointSizeRange[1]));
            if (r.crop) {
                p.counts.z |= 1u | (r.crop->inverse ? 2u : 0u) | (r.crop->desaturate ? 4u : 0u);
                p.crop_to_local = matrix(r.crop->to_local);
                p.crop_min = {r.crop->min.x, r.crop->min.y, r.crop->min.z, 0};
                p.crop_max = {r.crop->max.x, r.crop->max.y, r.crop->max.z, 0};
            } else if (r.crop_ellipsoid) {
                p.counts.z |= 1025u | (r.crop_ellipsoid->inverse ? 2u : 0u) | (r.crop_ellipsoid->desaturate ? 4u : 0u);
                p.crop_to_local = matrix(r.crop_ellipsoid->to_local);
                p.crop_min = {r.crop_ellipsoid->radii.x, r.crop_ellipsoid->radii.y, r.crop_ellipsoid->radii.z, 0};
            }
            p.crop_min.w = r.depth_view_min;
            p.crop_max.w = r.depth_view_max;
            std::array<const core::Tensor*, 6> tensors = {r.positions, r.colors, r.transform_indices, r.selection_mask, r.preview_selection_mask, r.deleted_mask};
            if (i.serial == std::numeric_limits<uint64_t>::max())
                throw std::runtime_error("Metal point timeline exhausted");
            const uint64_t serial = i.serial + 1;
            const auto event = i.event;
            f.command = i.reader.submit(tensors, [&](id<MTLCommandBuffer> command, std::span<const core::MetalTensorView> views) {
                if (i.next_readback)
                    [command encodeWaitForEvent:i.readback_event value:i.next_readback];
                auto pass = [MTLRenderPassDescriptor new];
                pass.colorAttachments[0].texture = f.color.texture;
                pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                pass.colorAttachments[0].clearColor = MTLClearColorMake(r.background_color.x, r.background_color.y, r.background_color.z, r.transparent_background ? 0 : 1);
                pass.colorAttachments[1].texture = f.depth.texture;
                pass.colorAttachments[1].loadAction = MTLLoadActionClear;
                pass.colorAttachments[1].storeAction = MTLStoreActionStore;
                pass.colorAttachments[1].clearColor = MTLClearColorMake(-1, 0, 0, 0);
                pass.depthAttachment.texture = f.point_depth;
                pass.depthAttachment.loadAction = MTLLoadActionClear;
                pass.depthAttachment.storeAction = MTLStoreActionDontCare;
                pass.depthAttachment.clearDepth = 1;
                auto encoder = [command renderCommandEncoderWithDescriptor:pass];
                if (!encoder)
                    throw std::runtime_error("Metal point render encoder failed");
                [encoder setRenderPipelineState:i.point_pipeline];
                [encoder setDepthStencilState:i.point_depth_state];
                const NSUInteger bindings[] = {0, 1, 3, 4, 5, 7};
                for (size_t n = 0; n < views.size(); ++n)
                    [encoder setVertexBuffer:views[n].buffer ?: f.objects offset:views[n].buffer ? views[n].offset : 0 atIndex:bindings[n]];
                [encoder setVertexBuffer:f.objects offset:0 atIndex:2];
                [encoder setVertexBuffer:f.selection_colors offset:0 atIndex:6];
                [encoder setVertexBytes:&p length:sizeof(p) atIndex:8];
                [encoder setFragmentBytes:&p length:sizeof(p) atIndex:0];
                [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:r.positions->size(0)];
                [encoder endEncoding];
                [command encodeSignalEvent:event value:serial];
                [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    if (completed.status == MTLCommandBufferStatusError && event.signaledValue < serial)
                        event.signaledValue = serial;
                }];
            });
            f.producer_value = serial;
            i.serial = serial;
            f.consumer_serial = context.lastFrameSubmitSerial() + 1;
            i.latest[static_cast<size_t>(slot)] = &f;
            // Preserve the existing synchronous point-cloud presentation contract.
            // The Gaussian path continues to expose its asynchronous GPU timeline.
            i.wait(serial);
            [f.command waitUntilCompleted];
            if (f.command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error("Metal point raster failed");
            return PointCloudVulkanRenderer::RenderResult{.image = f.color.image, .image_view = f.color.view, .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .generation = f.generation, .depth_image = f.depth.image, .depth_image_view = f.depth.view, .depth_image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .depth_generation = f.generation, .size = f.size, .flip_y = false, .viewer_backend = rendering::ViewerBackend::Metal};
        } catch (const std::exception& e) { return nativeError(e); }
    }
    bool MetalViewportRenderer::supports(const core::SplatData& model, const rendering::ViewportRenderRequest& r) {
        // Every unsupported contract is routed to the existing renderer; never silently
        // drop filters, display settings or editor overlays from a requested frame.
        const auto resident_mask = [&](const core::Tensor* mask) { return !mask || !mask->is_valid() ||
                                                                          (core::gpu_backend_of(*mask) == core::GpuBackend::Metal && mask->is_contiguous() && mask->bytes() >= size_t(model.size()) &&
                                                                           (mask->dtype() == core::DataType::UInt8 || mask->dtype() == core::DataType::Bool)); };
        const auto indices = r.scene.transform_indices.get();
        const size_t objects = r.scene.model_transforms ? r.scene.model_transforms->size() : 0;
        const bool indexed = indices && indices->is_valid();
        if ((objects > 1 && !indexed) || (indexed &&
                                          (core::gpu_backend_of(*indices) != core::GpuBackend::Metal || !indices->is_contiguous() ||
                                           indices->dtype() != core::DataType::Int32 || indices->bytes() < size_t(model.size()) * 4)))
            return false;
        return resident_mask(r.overlay.emphasis.mask.get()) && resident_mask(r.overlay.emphasis.transient_mask.mask) &&
               core::gpu_backend_of(model.means_raw()) == core::GpuBackend::Metal &&
               (!r.equirectangular || r.gut) && r.splat_render_profile == 0 &&
               !r.lod_indices && !r.lod_gpu_traversal.enabled && !r.lod_debug_mode &&
               model.means_raw().dtype() == core::DataType::Float32 && model.sh0_raw().dtype() == core::DataType::Float32 &&
               ((model.scaling_raw().dtype() == core::DataType::Float32 && model.rotation_raw().dtype() == core::DataType::Float32 &&
                 model.opacity_raw().dtype() == core::DataType::Float32) ||
                model.non_sh_attrs_f16());
    }
    lfs::Result<VksplatViewportRenderer::RenderResult> MetalViewportRenderer::render(
        VulkanContext& context, const core::SplatData& model, const rendering::ViewportRenderRequest& request, Slot slot, bool expected_depth) {
        try {
            auto& i = *impl_;
            std::lock_guard lock(i.readback_mutex);
            i.initialize(context);
            // Cached outputs may be sampled for many GUI frames without a new
            // raster submission. Stamp their most recent graphics consumer before
            // replacing latest, rather than retiring against the first consumer.
            const auto graphics_serial = context.lastFrameSubmitSerial();
            for (auto frame : i.latest)
                if (frame)
                    frame->consumer_serial = std::max(frame->consumer_serial, graphics_serial);
            if (model.size() > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("Metal primitive count exceeds indexing capacity");
            auto previous = i.latest[static_cast<size_t>(slot)];
            if (previous && previous->points)
                previous = nullptr;
            if (previous && !previous->raster->busy()) {
                const auto status = previous->raster->status();
                if (status.required_instances > std::numeric_limits<uint32_t>::max())
                    throw std::runtime_error("Metal instance indexing overflow");
                if (status.error != RasterError::None)
                    i.needed_capacity[static_cast<size_t>(slot)] = static_cast<uint32_t>(status.required_instances);
            }
            if (i.frames[static_cast<size_t>(slot)][i.next[static_cast<size_t>(slot)] % 3].get() == previous)
                previous = nullptr;
            auto& f = i.acquire(slot, request, static_cast<uint32_t>(model.size()));
            if (request.gut && (!f.gut_geometry || f.gut_geometry.length < size_t(model.size()) * sizeof(GutSplat))) {
                const size_t bytes = std::max<size_t>(16, size_t(model.size()) * sizeof(GutSplat));
                const auto device = i.reader.device();
                if (!frameFitsWorkingSet(device.currentAllocatedSize, bytes, device.recommendedMaxWorkingSetSize))
                    throw lfs::Exception(nativeError("Metal 3DGUT reservation exceeds the recommended GPU working set", lfs::ErrorCode::ResourceExhausted));
                f.gut_geometry = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
                if (!f.gut_geometry)
                    throw lfs::Exception(nativeError("Metal 3DGUT geometry allocation failed", lfs::ErrorCode::ResourceExhausted));
            }
            const int node_degree = request.scene.node_active_sh_degrees.empty() ? model.get_active_sh_degree() : *std::max_element(request.scene.node_active_sh_degrees.begin(), request.scene.node_active_sh_degrees.end());
            const int max_degree = std::clamp(node_degree, 0, std::min(3, model.get_max_sh_degree()));
            const uint32_t degree = static_cast<uint32_t>(std::clamp(request.sh_degree, 0, max_degree));
            const auto storage = model.shN_value_quantized() ? ShStorage::Q16 : model.shN_ieee_f16() ? ShStorage::SwizzledFloat16
                                                                                                     : ShStorage::SwizzledFloat32;
            Projection projection{};
            projection.model_to_world = matrix(glm::mat4(1));
            auto view = glm::mat4(glm::transpose(rendering::dataCameraToWorldFromVisualizerRotation(request.frame_view.rotation)));
            view[3] = glm::vec4(-glm::mat3(view) * request.frame_view.translation, 1);
            projection.world_to_camera = matrix(view);
            const auto camera = request.frame_view.translation;
            projection.camera_local = {camera.x, camera.y, camera.z, 1};
            auto intrinsics = request.frame_view.getCameraIntrinsics();
            projection.intrinsics = {intrinsics.focal_x, intrinsics.focal_y,
                                     intrinsics.center_x - request.frame_view.subregion_origin.x, intrinsics.center_y - request.frame_view.subregion_origin.y};
            // Viewer raster clipping differs from the desktop projection matrix's
            // near/far planes. Derive the reference near threshold at configure.
            projection.clip_scale = {kViewerNearClip, std::numeric_limits<float>::max(), request.scaling_modifier, request.mip_filter ? .1f : .3f};
            projection.extent = {uint32_t(f.size.x), uint32_t(f.size.y), uint32_t(request.equirectangular ? CameraModel::Equirectangular : request.frame_view.orthographic ? CameraModel::Orthographic
                                                                                                                                                                           : CameraModel::Perspective),
                                 uint32_t(request.mip_filter)};
            projection.rasterization = {request.frame_view.rasterization_scale, expected_depth ? 1.f : 0.f, request.frame_view.far_plane, 0};
            const auto panorama_size = request.frame_view.cameraSize();
            projection.panorama = {float(panorama_size.x), float(panorama_size.y), float(request.frame_view.subregion_origin.x), float(request.frame_view.subregion_origin.y)};
            SceneBuffers scene{};
            OverlayBuffers overlay{};
            if (request.scene.model_transforms && !request.scene.model_transforms->empty()) {
                const auto& transforms = *request.scene.model_transforms;
                if (transforms.size() > 1 && (!request.scene.transform_indices || !request.scene.transform_indices->is_valid()))
                    throw std::invalid_argument("Multiple Metal scene transforms require primitive indices");
                const size_t bytes = transforms.size() * sizeof(SceneObject);
                if (!f.objects || f.objects.length < bytes)
                    f.objects = [i.reader.device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!f.objects)
                    throw lfs::Exception(nativeError("Metal scene object allocation failed", lfs::ErrorCode::ResourceExhausted));
                auto objects = static_cast<SceneObject*>(f.objects.contents);
                for (size_t n = 0; n < transforms.size(); ++n) {
                    const auto local = glm::inverse(transforms[n]) * glm::vec4(camera, 1);
                    const bool visible = n >= request.scene.node_visibility_mask.size() || request.scene.node_visibility_mask[n];
                    const int active = n < request.scene.node_active_sh_degrees.size() ? request.scene.node_active_sh_degrees[n] : degree;
                    objects[n] = {matrix(transforms[n]), {local.x, local.y, local.z, 1}, {uint32_t(visible), uint32_t(std::clamp(active, 0, 3)), 0, 0}};
                }
                scene.objects = {f.objects, 0};
                scene.count = static_cast<uint32_t>(transforms.size());
            }
            const auto selection = request.overlay.emphasis.mask.get();
            const auto preview = request.overlay.emphasis.transient_mask.mask;
            const bool selection_enabled = request.overlay.has_selection && selection && selection->is_valid();
            const bool preview_enabled = preview && preview->is_valid();
            const size_t node_count = request.overlay.emphasis.emphasized_node_mask.size();
            const bool needs_overlay = request.filters.crop_region || request.filters.ellipsoid_region ||
                                       !request.filters.crop_regions.empty() || !request.filters.ellipsoid_regions.empty() ||
                                       request.filters.view_volume || selection_enabled || preview_enabled || node_count ||
                                       request.overlay.emphasis.dim_non_emphasized || request.overlay.emphasis.flash_intensity > 0 ||
                                       request.overlay.emphasis.focused_gaussian_id >= 0 || request.overlay.cursor.enabled ||
                                       request.overlay.markers.show_rings || request.overlay.markers.show_center_markers;
            if (needs_overlay) {
                static_assert(detail::ParamCount == 207);
                static_assert(detail::ViewWindow == 206 && detail::SelectionFlags == 24 && detail::EmphasisFlags == 20);
                const auto params = detail::buildOverlayParamsCpuFloats(request, selection_enabled, preview_enabled, scene.count > 0, node_count, false);
                if (!params)
                    throw std::runtime_error(params.error());
                const auto allocate = [&](id<MTLBuffer> __strong& buffer, size_t bytes, MTLResourceOptions options) {
                    if (!buffer || buffer.length < bytes)
                        buffer = [i.reader.device() newBufferWithLength:std::max<size_t>(bytes, 16) options:options];
                    if (!buffer)
                        throw lfs::Exception(nativeError("Metal overlay allocation failed", lfs::ErrorCode::ResourceExhausted));
                };
                allocate(f.overlay_parameters, params->size() * sizeof(float), MTLResourceStorageModeShared);
                std::memcpy(f.overlay_parameters.contents, params->data(), params->size() * sizeof(float));
                allocate(f.overlay_flags, size_t(model.size()) * 4, MTLResourceStorageModePrivate);
                allocate(f.overlay_nodes, node_count, MTLResourceStorageModeShared);
                auto nodes = static_cast<uint8_t*>(f.overlay_nodes.contents);
                for (size_t n = 0; n < node_count; ++n)
                    nodes[n] = request.overlay.emphasis.emphasized_node_mask[n];
                allocate(f.selection_colors, sizeof(request.overlay.selection_colors), MTLResourceStorageModeShared);
                std::memcpy(f.selection_colors.contents, request.overlay.selection_colors.data(), sizeof(request.overlay.selection_colors));
                overlay = {{f.overlay_parameters, 0}, {f.overlay_flags, 0}, {}, {}, {f.overlay_nodes, 0}, {f.selection_colors, 0}, 207, uint32_t(node_count)};
                overlay.render_origin = {float(request.frame_view.subregion_origin.x), float(request.frame_view.subregion_origin.y), 0, 0};
            }
            std::array<const core::Tensor*, 11> tensors{&model.means_raw(), &model.scaling_raw(), &model.rotation_raw(),
                                                        &model.opacity_raw(), &model.sh0_raw(), degree ? &model.shN_raw() : nullptr,
                                                        degree ? &model.shN_value_bounds() : nullptr, &model.deleted(), scene.count ? request.scene.transform_indices.get() : nullptr,
                                                        selection_enabled ? selection : nullptr, preview_enabled ? preview : nullptr};
            if (i.serial == std::numeric_limits<uint64_t>::max())
                throw std::runtime_error("Metal viewport timeline exhausted");
            const uint64_t serial = i.serial + 1;
            const auto background = request.frame_view.background_color;
            const auto event = i.event;
            const PresentParameters present{request.color_exposure, uint32_t(request.color_tonemapping), uint32_t(request.transparent_background), uint32_t(previous != nullptr), request.depth_view_min, request.depth_view_max, uint32_t(request.depth_view), uint32_t(request.depth_visualization_mode), {background.x, background.y, background.z, 1}, {uint32_t(expected_depth), 0, 0, 0}};
            uint64_t key = 1469598103934665603ull;
            const auto hash = [&](const void* bytes, size_t length) {
                const auto data = static_cast<const uint8_t*>(bytes);
                for (size_t n = 0; n < length; ++n) {
                    key ^= data[n];
                    key *= 1099511628211ull;
                }
            };
            // Temporal reconstruction jitters only the draw intrinsics. It must
            // not look like scene/camera invalidation and request fresh content
            // forever. Retain the jitter in projection, exclude it from this key.
            auto stable_projection = projection;
            const auto camera_size = request.frame_view.cameraSize();
            const auto containment = request.frame_view.containment_intrinsics;
            stable_projection.intrinsics.z = containment ? containment->center_x : float(camera_size.x) * .5f;
            stable_projection.intrinsics.w = containment ? containment->center_y : float(camera_size.y) * .5f;
            stable_projection.intrinsics.z -= request.frame_view.subregion_origin.x;
            stable_projection.intrinsics.w -= request.frame_view.subregion_origin.y;
            hash(&stable_projection, sizeof(stable_projection));
            hash(&present, 12);
            hash(&present.depth_min, sizeof(PresentParameters) - 16);
            const auto mask_version = model.deleted_mask_version();
            hash(&mask_version, sizeof(mask_version));
            hash(&request.gut, sizeof(request.gut));
            if (scene.count)
                hash(f.objects.contents, scene.count * sizeof(SceneObject));
            if (needs_overlay) {
                hash(f.overlay_parameters.contents, 207 * 16);
                hash(f.selection_colors.contents, sizeof(request.overlay.selection_colors));
                if (node_count)
                    hash(f.overlay_nodes.contents, node_count);
            }
            for (const auto tensor : tensors)
                if (tensor && tensor->is_valid()) {
                    const auto pointer = tensor->data_ptr();
                    hash(&pointer, sizeof(pointer));
                }
            const bool refine = !previous || previous->request_key != key || previous->raster->busy() ||
                                previous->raster->status().error != RasterError::None;
            f.command = i.reader.submit(tensors, [&](id<MTLCommandBuffer> command, std::span<const core::MetalTensorView> views) {
                // A blit on the readback queue may still sample a recycled slot.
                // GPU ordering protects it without waiting on the host each frame.
                if (i.next_readback)
                    [command encodeWaitForEvent:i.readback_event value:i.next_readback];
                auto slice = [&](size_t n) { return BufferSlice{views[n].buffer, views[n].offset}; };
                SplatBuffers inputs{slice(0), slice(1), slice(2), slice(3), slice(4), slice(5), slice(6), slice(7),
                                    uint32_t(model.size()), uint32_t(model.max_sh_coeffs_rest()), storage, model.non_sh_attrs_f16()};
                scene.object_indices = slice(8);
                overlay.selection = selection_enabled ? slice(9) : BufferSlice{};
                overlay.preview = preview_enabled ? slice(10) : BufferSlice{};
                i.preprocessor.encode(command, inputs, projection, degree, request.gut ? PrimitiveMode::Gut : PrimitiveMode::Gaussian, {f.projected, 0}, scene, overlay, request.gut ? BufferSlice{f.gut_geometry, 0} : BufferSlice{});
                i.rasterizer.encode(command, {f.projected, 0}, uint32_t(model.size()), request.gut ? RasterMode::Gut : RasterMode::Gaussian,
                                    {background.x, background.y, background.z, request.transparent_background ? 0.f : 1.f}, *f.raster, overlay, request.gut ? BufferSlice{f.gut_geometry, 0} : BufferSlice{}, projection);
                auto encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:i.present];
                [encoder setTexture:f.raster->color() atIndex:0];
                [encoder setTexture:f.raster->depth() atIndex:1];
                [encoder setTexture:f.color.texture atIndex:2];
                [encoder setTexture:f.depth.texture atIndex:3];
                [encoder setTexture:previous ? previous->color.texture : f.raster->color() atIndex:4];
                [encoder setTexture:previous ? previous->depth.texture : f.raster->depth() atIndex:5];
                [encoder setBytes:&present length:sizeof(present) atIndex:0];
                [encoder setBuffer:f.raster->statusBuffer() offset:0 atIndex:1];
                [encoder dispatchThreads:MTLSizeMake(f.size.x, f.size.y, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                [encoder endEncoding];
                [command encodeSignalEvent:event value:serial];
                [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    // Failed read-only producers must release presentation waits too;
                    // their typed command error is inspected before resource reuse.
                    if (completed.status == MTLCommandBufferStatusError && event.signaledValue < serial)
                        event.signaledValue = serial;
                }];
            });
            f.producer_value = serial;
            f.request_key = key;
            i.serial = serial;
            f.consumer_serial = slot == Slot::Preview ? 0 : context.lastFrameSubmitSerial() + 1;
            i.latest[static_cast<size_t>(slot)] = &f;
            return VksplatViewportRenderer::RenderResult{.image = f.color.image, .image_view = f.color.view, .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .generation = f.generation, .depth_image = f.depth.image, .depth_image_view = f.depth.view, .depth_image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .depth_generation = f.generation, .size = f.size, .alloc_size = f.size, .flip_y = false, .completion_semaphore = i.completion, .completion_value = serial, .lod_streaming_active = refine, .viewer_backend = rendering::ViewerBackend::Metal};
        } catch (const std::exception& e) { return nativeError(e); }
    }
    glm::ivec2 MetalViewportRenderer::size(Slot slot) const {
        auto f = impl_->latest[static_cast<size_t>(slot)];
        return f ? f->size : glm::ivec2{};
    }
    lfs::Result<uint64_t> MetalViewportRenderer::submitReadback(
        Slot slot, core::Tensor& destination, int x, int y, bool depth) const {
        try {
            auto& i = *impl_;
            std::lock_guard lock(i.readback_mutex);
            auto f = i.latest[static_cast<size_t>(slot)];
            if (!f)
                throw std::runtime_error("Metal readback slot is empty");
            // Abandoned commands retain staging until completion but no host pointer.
            std::erase_if(i.readbacks, [](const auto& entry) { return !entry.second.destination &&
                                                                      (entry.second.command.status == MTLCommandBufferStatusCompleted || entry.second.command.status == MTLCommandBufferStatusError); });
            if (i.readbacks.size() >= ReadbackTicketRing::kRingSize)
                throw std::runtime_error("Metal readback ring is full; retire or abandon a ticket");
            if (!destination.is_valid() || destination.device() != core::Device::CPU || !destination.is_contiguous() ||
                x < 0 || y < 0 || (depth ? (destination.dtype() != core::DataType::Float32 || destination.ndim() != 2 || destination.size(0) != size_t(f->size.y) || destination.size(1) != size_t(f->size.x)) : (destination.ndim() != 3 || destination.size(2) < 3 || destination.size(2) > 4 || size_t(x) + f->size.x > destination.size(1) || size_t(y) + f->size.y > destination.size(0) || (destination.dtype() != core::DataType::Float32 && destination.dtype() != core::DataType::UInt8))))
                throw std::invalid_argument("Invalid Metal readback destination");
            if (!i.readback_queue || !i.readback_event || i.next_readback == ((uint64_t{1} << 63) - 1))
                throw std::runtime_error("Metal readback timeline unavailable or exhausted");
            const uint64_t serial = i.next_readback + 1;
            const auto command = [i.readback_queue commandBuffer];
            const size_t row = size_t(f->size.x) * 4;
            const auto buffer = [i.reader.device() newBufferWithLength:row * f->size.y options:MTLResourceStorageModeShared];
            if (!command || !buffer)
                throw lfs::Exception(nativeError("Metal readback allocation failed", lfs::ErrorCode::ResourceExhausted));
            [command encodeWaitForEvent:i.event value:f->producer_value];
            auto blit = [command blitCommandEncoder];
            if (!blit)
                throw std::runtime_error("Metal readback encoder unavailable");
            [blit copyFromTexture:depth ? f->depth.texture : f->color.texture
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(f->size.x, f->size.y, 1)
                                toBuffer:buffer
                       destinationOffset:0
                  destinationBytesPerRow:row
                destinationBytesPerImage:row * f->size.y];
            [blit endEncoding];
            const auto event = i.readback_event;
            [command encodeSignalEvent:event value:serial];
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                if (completed.status == MTLCommandBufferStatusError && event.signaledValue < serial)
                    event.signaledValue = serial;
            }];
            const uint64_t ticket = reserveNativeTicket();
            i.readbacks.emplace(ticket, Impl::Readback{buffer, command, f->command, destination.data_ptr(), f->size,
                                                       destination.size(1), depth ? 1 : destination.size(2), x, y, depth, destination.dtype() == core::DataType::Float32});
            i.next_readback = serial;
            [command commit];
            return ticket;
        } catch (const std::exception& e) { return nativeError(e); }
    }
    lfs::Result<VksplatViewportRenderer::ReadbackTicketStatus> MetalViewportRenderer::pollReadback(uint64_t ticket, bool wait) const {
        auto& i = *impl_;
        std::lock_guard lock(i.readback_mutex);
        auto it = i.readbacks.find(ticket);
        if (it == i.readbacks.end())
            return nativeError("Unknown Metal readback ticket", lfs::ErrorCode::NotFound);
        auto& r = it->second;
        if (wait)
            [r.command waitUntilCompleted];
        if (r.command.status == MTLCommandBufferStatusError) {
            const std::string error = r.command.error.localizedDescription.UTF8String ?: "Metal readback command failed";
            i.readbacks.erase(it);
            return nativeError(error);
        }
        if (r.command.status != MTLCommandBufferStatusCompleted)
            return VksplatViewportRenderer::ReadbackTicketStatus::NotReady;
        if (r.producer.status == MTLCommandBufferStatusError) {
            i.readbacks.erase(it);
            return nativeError("Metal readback producer failed");
        }
        if (!r.destination) {
            i.readbacks.erase(it);
            return nativeError("Metal readback ticket abandoned", lfs::ErrorCode::Cancelled);
        }
        if (r.depth)
            std::memcpy(r.destination, r.buffer.contents, size_t(r.size.x) * r.size.y * 4);
        else {
            const auto src = static_cast<const uint8_t*>(r.buffer.contents);
            for (int row = 0; row < r.size.y; ++row)
                for (int col = 0; col < r.size.x; ++col)
                    for (size_t c = 0; c < r.channels; ++c) {
                        const auto value = src[(size_t(row) * r.size.x + col) * 4 + c];
                        const auto offset = ((size_t(row + r.y) * r.width + col + r.x) * r.channels + c);
                        if (r.floating)
                            static_cast<float*>(r.destination)[offset] = float(value) / 255;
                        else
                            static_cast<uint8_t*>(r.destination)[offset] = value;
                    }
        }
        i.readbacks.erase(it);
        return VksplatViewportRenderer::ReadbackTicketStatus::Ready;
    }
    void MetalViewportRenderer::abandonReadback(uint64_t ticket) const {
        std::lock_guard lock(impl_->readback_mutex);
        const auto it = impl_->readbacks.find(ticket);
        if (it != impl_->readbacks.end())
            it->second.destination = nullptr;
    }
    size_t MetalViewportRenderer::outstandingReadbacks() const {
        std::lock_guard lock(impl_->readback_mutex);
        return std::count_if(impl_->readbacks.begin(), impl_->readbacks.end(), [](const auto& entry) { return entry.second.destination != nullptr; });
    }
    lfs::Status MetalViewportRenderer::release(Slot slot) {
        try {
            auto& i = *impl_;
            std::lock_guard lock(i.readback_mutex);
            if (!i.context)
                return {};
            i.wait(i.serial);
            for (const auto& [ticket, r] : i.readbacks)
                [r.command waitUntilCompleted];
            // Includes cached outputs sampled after their initial producer frame.
            if (!i.context->waitForRetiredFrameSubmitSerial(i.context->lastFrameSubmitSerial()))
                throw std::runtime_error(i.context->lastError());
            const size_t index = static_cast<size_t>(slot);
            i.latest[index] = nullptr;
            for (auto& frame : i.frames[index])
                frame.reset();
            i.next[index] = 0;
            i.needed_capacity[index] = 0;
            return {};
        } catch (const std::exception& e) { return lfs::Status::failure(nativeError(e)); }
    }
    lfs::Result<bool> MetalViewportRenderer::outputComplete(Slot slot) const {
        try {
            auto frame = impl_->latest[static_cast<size_t>(slot)];
            if (!frame)
                throw lfs::Exception(nativeError("Metal output slot is empty", lfs::ErrorCode::FailedPrecondition));
            impl_->wait(frame->producer_value);
            [frame->command waitUntilCompleted];
            if (frame->command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error("Metal output command failed");
            return !frame->raster || frame->raster->status().error == RasterError::None;
        } catch (const std::exception& error) {
            return nativeError(error);
        }
    }
    lfs::Status MetalViewportRenderer::readColor(Slot slot, core::Tensor& destination, int x, int y) const {
        try {
            auto f = impl_->latest[static_cast<size_t>(slot)];
            if (!f)
                throw lfs::Exception(nativeError("Metal output slot is empty", lfs::ErrorCode::FailedPrecondition));
            if (destination.device() != core::Device::CPU || !destination.is_contiguous() || destination.ndim() != 3 ||
                destination.size(2) < 3 || destination.size(2) > 4 || x < 0 || y < 0 ||
                size_t(x) + f->size.x > destination.size(1) || size_t(y) + f->size.y > destination.size(0) ||
                (destination.dtype() != core::DataType::Float32 && destination.dtype() != core::DataType::UInt8))
                throw std::invalid_argument("Invalid Metal HWC readback destination");
            auto buffer = impl_->readTexture(*f, f->color, 4);
            const auto source = static_cast<const uint8_t*>(buffer.contents);
            const size_t channels = destination.size(2), width = destination.size(1);
            for (int row = 0; row < f->size.y; ++row)
                for (int col = 0; col < f->size.x; ++col)
                    for (size_t c = 0; c < channels; ++c) {
                        const size_t offset = ((row + y) * width + col + x) * channels + c;
                        const auto value = source[(row * f->size.x + col) * 4 + c];
                        if (destination.dtype() == core::DataType::Float32)
                            destination.ptr<float>()[offset] = float(value) / 255;
                        else
                            destination.ptr<uint8_t>()[offset] = value;
                    }
            return {};
        } catch (const std::exception& e) { return lfs::Status::failure(nativeError(e)); }
    }
    lfs::Result<float> MetalViewportRenderer::readDepth(const VksplatViewportRenderer::DepthSampleRequest& r) const {
        try {
            auto f = impl_->latest[static_cast<size_t>(r.output_slot)];
            if (!f)
                throw lfs::Exception(nativeError("Metal depth slot is empty", lfs::ErrorCode::FailedPrecondition));
            auto p = r.pixel;
            if (r.source_size.x > 0 && r.source_size.y > 0)
                p = glm::clamp(glm::ivec2(glm::round((glm::vec2(p) + .5f) * glm::vec2(f->size) / glm::vec2(r.source_size) - .5f)), glm::ivec2(0), f->size - 1);
            if (p.x < 0 || p.y < 0 || p.x >= f->size.x || p.y >= f->size.y)
                return -1.f;
            auto buffer = impl_->readTexture(*f, f->depth, 4, p);
            const float value = static_cast<const float*>(buffer.contents)[0];
            return value > 0 && value < 1e9f ? value : -1.f;
        } catch (const std::exception& e) { return nativeError(e); }
    }
} // namespace lfs::vis
