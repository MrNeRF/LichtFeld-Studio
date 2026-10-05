/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_viewport_renderer.hpp"
#include "core/logger.hpp"
#include "core/gpu_elapsed.hpp"
#include "core/memory_pressure.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_metal_reader.hpp"
#include "core/tensor_upload.hpp"
#include "metal_frame_budget.hpp"
#include "metal_present_source.hpp"
#include "metal_rad_pager.hpp"
#include "generic_readback_ticket_ring.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "scene_overlay_params.hpp"
#include "splat_lod_selector.hpp"
#include "splat_projector.hpp"
#include "splat_rasterizer.hpp"
#include "splat_selection_query.hpp"
#ifdef LFS_GRAPHICS_VULKAN
#include "vulkan_scene_output.hpp"
#include "window/vulkan_context.hpp"
#endif
#include <algorithm>
#include <glm/gtc/type_ptr.hpp>
#include <simd/simd.h>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <deque>
#include <format>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#ifdef LFS_GRAPHICS_VULKAN
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_metal.h>
#endif

namespace lfs::vis {
    namespace {
        // A quarter more than measured, so a moving camera does not overflow again at once.
        uint32_t withGrowthHeadroom(const uint64_t required) {
            return static_cast<uint32_t>(std::min<uint64_t>(required + required / 4, std::numeric_limits<uint32_t>::max()));
        }
    } // namespace
    namespace {
        using Slot = RenderTargetId;
        using rendering::metal::frameFitsWorkingSet;
        using rendering::metal::viewportOutputReservationBytes;
        enum class CameraModel : uint32_t { Perspective,
                                            Orthographic,
                                            Equirectangular };
        constexpr float kViewerNearClip = rendering::kSplatNearClip;
        // GPU records shared with the tensor programs (splat_project.slang,
        // splat_types.slang, splat_lod.slang).
        struct alignas(16) SceneObject {
            simd_float4x4 model_to_world;
            simd_float4 camera_local;
            simd_uint4 flags; // visible, maximum active SH degree
        };
        static_assert(sizeof(SceneObject) == 96);
        struct RasterStatus {
            uint64_t required_instances = 0;
            uint32_t error = 0, blend_threads = 0, maximum_tile_instances = 0, padding = 0;
        };
        static_assert(sizeof(RasterStatus) == 24);
        struct LodCutStatus {
            uint32_t selected = 0, overflow = 0;
            float threshold_multiplier = 1;
        };
        bool nativeStorage(const core::Tensor& tensor) {
            return core::tensor_supports_metal_access(tensor);
        }
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
            if (const auto* allocation = dynamic_cast<const core::MemoryAllocationError*>(&error))
                return core::to_error(allocation->failure(), site);
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
                    throw std::runtime_error(std::format("Native readback ticket identity exhausted (serial={}, limit={})", value, (uint64_t{1} << 63) - 1));
                if (native_ticket_serial.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
                    return (uint64_t{1} << 63) | (value + 1);
            }
        }
#ifdef LFS_GRAPHICS_VULKAN
        void check(VkResult r, const char* label) {
            if (r != VK_SUCCESS)
                throw std::runtime_error(std::string(label) + ": " + std::to_string(r));
        }
#endif
        simd_float4x4 matrix(const glm::mat4& m) {
            simd_float4x4 result;
            static_assert(sizeof(result) == sizeof(m));
            std::memcpy(&result, &m, sizeof(m));
            return result;
        }
        struct PointParameters {
            simd_float4x4 view_projection, view, crop_to_local;
            simd_float4 crop_min, crop_max, voxel_focal_ortho;
            simd_uint4 counts;
        };
        static_assert(sizeof(PointParameters) == 256);
#ifdef LFS_GRAPHICS_VULKAN
        struct Image {
            VkDevice device = VK_NULL_HANDLE;
            VmaAllocator allocator = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            id<MTLTexture> texture;
            VkImage image = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            ~Image() {
                if (view)
                    vkDestroyImageView(device, view, nullptr);
                if (image)
                    vmaDestroyImage(allocator, image, allocation);
            }
            void init(VulkanContext& context, id<MTLDevice> metal, uint32_t w, uint32_t h,
                      MTLPixelFormat native_format, VkFormat format) {
                device = context.device();
                allocator = context.allocator();
                auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
                    vkGetDeviceProcAddr(device, "vkExportMetalObjectsEXT"));
                if (!allocator || !export_objects)
                    throw std::runtime_error(std::format("Metal viewport image export is unavailable (allocator_present={}, export_function_present={})", allocator != VK_NULL_HANDLE, export_objects != nullptr));
                // MoltenVK owns the image and its backing allocation. Importing
                // an independently allocated texture bypasses that ownership and
                // can omit residency when the compositor uses argument buffers.
                VkExportMetalObjectCreateInfoEXT export_texture{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT};
                export_texture.exportObjectType = VK_EXPORT_METAL_OBJECT_TYPE_METAL_TEXTURE_BIT_EXT;
                VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                info.pNext = &export_texture;
                info.imageType = VK_IMAGE_TYPE_2D;
                info.format = format;
                info.extent = {w, h, 1};
                info.mipLevels = 1;
                info.arrayLayers = 1;
                info.samples = VK_SAMPLE_COUNT_1_BIT;
                info.tiling = VK_IMAGE_TILING_OPTIMAL;
                info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                VmaAllocationCreateInfo memory_info{};
                memory_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                const auto allocated = vmaCreateImage(allocator, &info, &memory_info, &image, &allocation, nullptr);
                if (allocated != VK_SUCCESS) {
                    const auto code = allocated == VK_ERROR_OUT_OF_HOST_MEMORY || allocated == VK_ERROR_OUT_OF_DEVICE_MEMORY
                                          ? lfs::ErrorCode::ResourceExhausted : lfs::ErrorCode::Internal;
                    throw lfs::Exception(nativeError(std::format("Vulkan-owned Metal viewport image allocation failed (result={}, extent={}x{}, format={}, allocated={}, recommended={})", int(allocated), w, h, int(format), metal.currentAllocatedSize, metal.recommendedMaxWorkingSetSize), code));
                }
                VkExportMetalTextureInfoEXT native_texture{VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT};
                native_texture.image = image;
                native_texture.plane = VK_IMAGE_ASPECT_COLOR_BIT;
                VkExportMetalObjectsInfoEXT exports{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
                exports.pNext = &native_texture;
                export_objects(device, &exports);
                texture = native_texture.mtlTexture;
                if (!texture || texture.device.registryID != metal.registryID || texture.pixelFormat != native_format)
                    throw std::runtime_error(std::format("Vulkan viewport texture export does not match Metal rendering (texture_present={}, texture_device={}, render_device={}, texture_format={}, requested_format={}, extent={}x{})", texture != nil, texture.device.registryID, metal.registryID, uint64_t(texture.pixelFormat), uint64_t(native_format), w, h));
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
#else
        // Native output for the tensor compositor, which copies it into tensors.
        struct Image {
            id<MTLTexture> texture;
            void init(MetalViewportPresentation&, id<MTLDevice> metal, uint32_t w, uint32_t h,
                      MTLPixelFormat native_format) {
                auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native_format width:w height:h mipmapped:NO];
                descriptor.storageMode = MTLStorageModePrivate;
                descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
                texture = [metal newTextureWithDescriptor:descriptor];
                if (!texture)
                    throw lfs::Exception(nativeError(std::format("Metal viewport image allocation failed (extent={}x{}, format={}, allocated={}, recommended={})", w, h, uint64_t(native_format), metal.currentAllocatedSize, metal.recommendedMaxWorkingSetSize), lfs::ErrorCode::ResourceExhausted));
            }
        };
#endif
        struct Frame {
            glm::ivec2 size{};
            uint32_t count = 0, capacity = 0;
            uint64_t generation = 0, consumer_serial = 0, producer_value = 0;
            id<MTLBuffer> objects, overlay_parameters, overlay_nodes, selection_colors;
            bool rad_bootstrap = false;
            uint64_t rad_signature = 0;
            id<MTLCommandBuffer> command;
            id<MTLTexture> point_depth;
            bool gpu_lod_active = false;
            uint64_t gpu_tree_signature = 0;
            uint32_t gpu_capacity = 0, gpu_source_count = 0, gpu_chunks = 0;
            bool points = false;
            // Host copies of the tensor results, read after completion: the
            // RasterStatus, and the LOD cut's counts (selected, overflow,
            // threshold) and chunk touches. Tensor storage may be device-only.
            id<MTLBuffer> raster_status, lod_counts, lod_touches;
            // Opt-in GPU timestamps at the tensor stage boundaries.
            std::unique_ptr<core::GpuElapsed> profile;
            std::vector<std::string> profile_stages;
            RasterStatus rasterStatus() const {
                return raster_status ? *static_cast<const RasterStatus*>(raster_status.contents) : RasterStatus{};
            }
            bool lodResultReady() const {
                return gpu_lod_active && lod_counts && command.status == MTLCommandBufferStatusCompleted;
            }
            LodCutStatus lodStatus() const {
                const auto counts = static_cast<const uint32_t*>(lod_counts.contents);
                return {counts[0], counts[1], std::bit_cast<float>(counts[2])};
            }
            Image color, depth;
        };
    } // namespace
    struct MetalViewportRenderer::Impl {
        MetalViewportPresentation* context = nullptr;
        struct LodTree {
            // bounds, links, chunk_to_page, page_age, page_frames, page_to_chunk
            std::array<core::Tensor, 6> tensors;
            uint64_t signature = 0, last_used = 0;
            uint32_t nodes = 0, chunks = 0, roots = 0;
        };
        std::map<const core::SplatLodTree*, LodTree> lod_trees;
        std::unique_ptr<rendering::SplatSelectionQuery> selection_query;
        core::GpuBackend selection_backend{};
        // Selection requests carry their own camera, not a render-target ID.
        // A dedicated pager avoids borrowing another view's RAD/model workspace.
        std::unique_ptr<MetalRadPager> selection_pager;
        MetalRadPager::Settings rad_settings;
        core::MetalTensorReader reader;
        std::unique_ptr<rendering::SplatProjector> projector;
        core::GpuBackend projector_backend{};
        core::Tensor projected, gut_geometry;
        bool profiling_enabled = false;
        std::function<void()> retry_callback;
        id<MTLRenderPipelineState> point_pipeline;
        id<MTLDepthStencilState> point_depth_state;
        id<MTLSharedEvent> event;
#ifdef LFS_GRAPHICS_VULKAN
        VkSemaphore completion = VK_NULL_HANDLE;
        // The Vulkan compositor samples published images inside graphics frames.
        uint64_t consumerSerial() const { return context->lastFrameSubmitSerial() + (context->hasActiveFrame() ? 1 : 0); }
        uint64_t publishedConsumerSerial() const { return context->lastFrameSubmitSerial(); }
        uint64_t retiredConsumerSerial() const { return context->retiredFrameSubmitSerial(); }
        void waitForConsumer(uint64_t serial) const {
            if (!context->waitForRetiredFrameSubmitSerial(serial))
                throw std::runtime_error(context->lastError());
        }
#else
        // Tensor copies are the only consumers; they run on the render queue.
        uint64_t consumerSerial() const { return 0; }
        uint64_t publishedConsumerSerial() const { return 0; }
        uint64_t retiredConsumerSerial() const { return std::numeric_limits<uint64_t>::max(); }
        void waitForConsumer(uint64_t) const {}
#endif
        void configurePager(MetalRadPager& pager, const core::SplatData& model) {
#ifdef LFS_GRAPHICS_VULKAN
            pager.configure(model, context->device(), completion, rad_settings);
#else
            // Uploads wait for this renderer's completion event directly.
            pager.configure(model, nullptr, (__bridge void*)event, rad_settings);
#endif
        }
        uint64_t serial = 0;
        struct TargetState {
            std::array<std::unique_ptr<Frame>, 3> frames;
            Frame* latest = nullptr;
            size_t next = 0;
            uint32_t needed_capacity = 0;
            // Instances an overflowing frame needed, recorded on GPU completion so
            // the retry sizes from it instead of waiting for that frame's slot.
            std::shared_ptr<std::atomic<uint64_t>> overflow_required = std::make_shared<std::atomic<uint64_t>>(0);
            std::unique_ptr<MetalRadPager> pager;
            // Raster scratch and display outputs, the LOD selector and RAD page
            // maps, all on the backend of the resident splats.
            std::unique_ptr<rendering::SplatRasterizer> raster;
            std::unique_ptr<rendering::SplatLodSelector> lod;
            core::GpuBackend backend{};
            std::array<core::Tensor, 3> page_maps; // chunk_to_page, page_age, page_to_chunk
            std::deque<core::TensorUpload> uploads;
        };
        std::unordered_map<RenderTargetId, TargetState, RenderTargetIdHash> targets;
        std::unordered_set<RenderTargetId, RenderTargetIdHash> released_targets;
        struct RetiredTarget {
            TargetState state;
            uint64_t consumer = 0;
        };
        std::vector<RetiredTarget> retired_targets;
        TargetState& target(Slot id) {
            if (!id.valid() || released_targets.contains(id))
                throw std::invalid_argument(std::format("Invalid or released Metal render target (target={}, released={})", id.value, released_targets.contains(id)));
            return targets[id];
        }
        Frame* latestFrame(Slot id) const {
            const auto found = targets.find(id);
            return found == targets.end() ? nullptr : found->second.latest;
        }
        void stampConsumers() {
            const auto consumer = publishedConsumerSerial();
            for (auto& [id, state] : targets)
                if (state.latest)
                    state.latest->consumer_serial = std::max(state.latest->consumer_serial, consumer);
        }
        void drainRetiredTargets() {
            const auto consumer = retiredConsumerSerial();
            std::erase_if(retired_targets, [&](const auto& retired) {
                if (retired.consumer > consumer)
                    return false;
                for (const auto& frame : retired.state.frames)
                    if (frame && frame->command && frame->command.status != MTLCommandBufferStatusCompleted &&
                        frame->command.status != MTLCommandBufferStatusError)
                        return false;
                return true;
            });
        }
        void retireTarget(Slot id, bool permanent) {
            if (!id.valid())
                throw std::invalid_argument(std::format("Invalid Metal render target (target={})", id.value));
            if (const auto found = targets.find(id); found != targets.end()) {
                // Include the recording graphics frame: it may not have a
                // submission serial yet, but still references these textures.
                uint64_t consumer = context ? consumerSerial() : 0;
                for (const auto& frame : found->second.frames)
                    if (frame)
                        consumer = std::max(consumer, frame->consumer_serial);
                retired_targets.push_back({std::move(found->second), consumer});
                targets.erase(found);
            }
            for (auto& [ticket, readback] : readbacks)
                if (readback.target == id) {
                    readback.destination = nullptr;
                    readback.target_released = true;
                }
            if (targets.empty()) {
                // A new scene starts with no high-water reservation; tensors stay
                // alive until their last GPU reader completes.
                projected = gut_geometry = {};
            }
            if (permanent)
                released_targets.insert(id);
        }
        struct Readback {
            id<MTLBuffer> buffer;
            id<MTLCommandBuffer> command;
            id<MTLCommandBuffer> producer;
            void* destination = nullptr;
            glm::ivec2 size;
            size_t width, channels;
            int x, y;
            bool depth, floating;
            RenderTargetId target{};
            bool target_released = false;
        };
        mutable std::mutex readback_mutex;
        mutable std::map<uint64_t, Readback> readbacks;
        mutable uint64_t next_readback = 0;
        id<MTLCommandQueue> readback_queue = [reader.device() newCommandQueue];
        id<MTLSharedEvent> readback_event = [reader.device() newSharedEvent];
#ifdef LFS_GRAPHICS_VULKAN
        ~Impl() {
            // Teardown only: the event covers every native producer before device idle.
            if (context && completion) {
                try {
                    wait(serial);
                    if (!context->deviceWaitIdle())
                        throw std::runtime_error(context->lastError());
                } catch (...) {
                    // Never destroy textures still referenced by an unretired device.
                    for (auto& [id, state] : targets)
                        for (auto& frame : state.frames)
                            (void)frame.release();
                    for (auto& retired : retired_targets)
                        for (auto& frame : retired.state.frames)
                            (void)frame.release();
                    return;
                }
                targets.clear();
                retired_targets.clear();
                // Upload queues import this consumer semaphore. Retire their decode
                // jobs and queues before destroying the presentation event.
                selection_pager.reset();
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
                    throw std::invalid_argument(std::format("Metal viewport context changed without reset (existing_context={:#x}, requested_context={:#x})", reinterpret_cast<uintptr_t>(context), reinterpret_cast<uintptr_t>(&ctx)));
                return;
            }
            auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(vkGetDeviceProcAddr(ctx.device(), "vkExportMetalObjectsEXT"));
            if (!export_objects)
                throw std::runtime_error(std::format("Presentation does not support VK_EXT_metal_objects (device={:#x}, export_function_present={})", reinterpret_cast<uintptr_t>(ctx.device()), export_objects != nullptr));
            VkExportMetalDeviceInfoEXT native_device{VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT};
            VkExportMetalObjectsInfoEXT exports{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
            exports.pNext = &native_device;
            export_objects(ctx.device(), &exports);
            if (!native_device.mtlDevice || native_device.mtlDevice.registryID != reader.device().registryID)
                throw std::runtime_error(std::format("Metal tensors and desktop presentation must use the same GPU (presentation_registry={}, tensor_registry={}, presentation_device_present={})", native_device.mtlDevice.registryID, reader.device().registryID, native_device.mtlDevice != nil));
            NSError* error = nil;
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion2_4;
            auto library = [reader.device() newLibraryWithSource:[NSString stringWithUTF8String:kMetalPresentSource] options:options error:&error];
            if (!library)
                throw std::runtime_error(std::format("Metal presentation shader compilation failed (error_code={}, error_domain={}, error={})", long(error.code), error.domain.UTF8String ?: "none", error.localizedDescription.UTF8String ?: "none"));
            auto point_descriptor = [MTLRenderPipelineDescriptor new];
            point_descriptor.vertexFunction = [library newFunctionWithName:@"point_vertex"];
            point_descriptor.fragmentFunction = [library newFunctionWithName:@"point_fragment"];
            point_descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            point_descriptor.colorAttachments[1].pixelFormat = MTLPixelFormatR32Float;
            point_descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
            point_pipeline = [reader.device() newRenderPipelineStateWithDescriptor:point_descriptor error:&error];
            if (!point_pipeline)
                throw std::runtime_error(std::format("Metal point pipeline creation failed (error_code={}, error_domain={}, error={})", long(error.code), error.domain.UTF8String ?: "none", error.localizedDescription.UTF8String ?: "none"));
            auto depth_descriptor = [MTLDepthStencilDescriptor new];
            depth_descriptor.depthCompareFunction = MTLCompareFunctionLess;
            depth_descriptor.depthWriteEnabled = YES;
            point_depth_state = [reader.device() newDepthStencilStateWithDescriptor:depth_descriptor];
            if (!point_depth_state)
                throw std::runtime_error(std::format("Metal point depth state unavailable (device={}, compare_function={}, depth_write={})", reader.device().name.UTF8String, uint32_t(depth_descriptor.depthCompareFunction), bool(depth_descriptor.depthWriteEnabled)));
            event = [reader.device() newSharedEvent];
            if (!event)
                throw lfs::Exception(nativeError(std::format("Metal presentation timeline allocation failed (device={}, allocated={}, recommended={})", reader.device().name.UTF8String, reader.device().currentAllocatedSize, reader.device().recommendedMaxWorkingSetSize), lfs::ErrorCode::ResourceExhausted));
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
#else
        ~Impl() {
            // Teardown only: never destroy textures a submitted command still uses.
            if (context && event) {
                try {
                    wait(serial);
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): leak rather than free textures of an unretired command.
                    for (auto& [id, state] : targets)
                        for (auto& frame : state.frames)
                            (void)frame.release();
                    for (auto& retired : retired_targets)
                        for (auto& frame : retired.state.frames)
                            (void)frame.release();
                    return;
                }
                targets.clear();
                retired_targets.clear();
                selection_pager.reset();
            }
        }
        void wait(uint64_t value) const {
            if (!value)
                return;
            if (![event waitUntilSignaledValue:value timeoutMS:5000])
                throw std::runtime_error(std::format("Native viewport completion timed out after 5 s (value={}, signaled={})", value, event.signaledValue));
        }
        void initialize(MetalViewportPresentation& ctx) {
            if (context) {
                if (context != &ctx)
                    throw std::invalid_argument(std::format("Metal viewport context changed without reset (existing_context={:#x}, requested_context={:#x})", reinterpret_cast<uintptr_t>(context), reinterpret_cast<uintptr_t>(&ctx)));
                return;
            }
            NSError* error = nil;
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion2_4;
            auto library = [reader.device() newLibraryWithSource:[NSString stringWithUTF8String:kMetalPresentSource] options:options error:&error];
            if (!library)
                throw std::runtime_error(std::format("Metal presentation shader compilation failed (error_code={}, error_domain={}, error={})", long(error.code), error.domain.UTF8String ?: "none", error.localizedDescription.UTF8String ?: "none"));
            auto point_descriptor = [MTLRenderPipelineDescriptor new];
            point_descriptor.vertexFunction = [library newFunctionWithName:@"point_vertex"];
            point_descriptor.fragmentFunction = [library newFunctionWithName:@"point_fragment"];
            point_descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            point_descriptor.colorAttachments[1].pixelFormat = MTLPixelFormatR32Float;
            point_descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
            point_pipeline = [reader.device() newRenderPipelineStateWithDescriptor:point_descriptor error:&error];
            if (!point_pipeline)
                throw std::runtime_error(std::format("Metal point pipeline creation failed (error_code={}, error_domain={}, error={})", long(error.code), error.domain.UTF8String ?: "none", error.localizedDescription.UTF8String ?: "none"));
            auto depth_descriptor = [MTLDepthStencilDescriptor new];
            depth_descriptor.depthCompareFunction = MTLCompareFunctionLess;
            depth_descriptor.depthWriteEnabled = YES;
            point_depth_state = [reader.device() newDepthStencilStateWithDescriptor:depth_descriptor];
            if (!point_depth_state)
                throw std::runtime_error(std::format("Metal point depth state unavailable (device={}, compare_function={}, depth_write={})", reader.device().name.UTF8String, uint32_t(depth_descriptor.depthCompareFunction), bool(depth_descriptor.depthWriteEnabled)));
            event = [reader.device() newSharedEvent];
            if (!event)
                throw lfs::Exception(nativeError(std::format("Metal presentation timeline allocation failed (device={}, allocated={}, recommended={})", reader.device().name.UTF8String, reader.device().currentAllocatedSize, reader.device().recommendedMaxWorkingSetSize), lfs::ErrorCode::ResourceExhausted));
            context = &ctx;
        }
#endif
        LodTree& prepareLodTree(const core::SplatData& model) {
            const auto& tree = *model.lod_tree;
            const size_t n = tree.total_nodes(), chunk_size = core::SplatLodTree::kChunkSplats;
            uint64_t signature = 1469598103934665603ull;
            const auto mix = [&](uint64_t word) {signature^=word;signature*=1099511628211ull; };
            mix(n);
            mix(tree.centers.size());
            mix(tree.sizes.size());
            mix(tree.lod_level.size());
            mix(reinterpret_cast<uintptr_t>(tree.child_count.data()));
            mix(reinterpret_cast<uintptr_t>(tree.child_start.data()));
            mix(reinterpret_cast<uintptr_t>(tree.centers.data()));
            mix(reinterpret_cast<uintptr_t>(tree.sizes.data()));
            mix(reinterpret_cast<uintptr_t>(tree.lod_level.data()));
            mix(reinterpret_cast<uintptr_t>(model.means_raw().data_ptr()));
            mix(reinterpret_cast<uintptr_t>(model.scaling_raw().data_ptr()));
            if (auto cached = lod_trees.find(&tree); cached != lod_trees.end() && cached->second.signature == signature) {
                cached->second.last_used = serial;
                return cached->second;
            }
            if (!n || n > size_t(model.size()) || n > std::numeric_limits<uint32_t>::max() || tree.child_start.size() < n || tree.child_count.size() < n)
                throw std::invalid_argument(std::format("Native GPU LOD requires a resident ordered hierarchy (nodes={}, model_size={}, child_starts={}, child_counts={})", n, model.size(), tree.child_start.size(), tree.child_count.size()));
            const size_t chunks = (n + chunk_size - 1) / chunk_size;
            std::vector<uint32_t> parents(n, 0xffffffffu), links(n * 3), bounds(n * 2), maps(chunks), age(chunks, 0);
            uint32_t roots = 0;
            for (size_t node = 0; node < n; ++node) {
                const size_t first = tree.child_start[node], count = tree.child_count[node];
                if (count && (first <= node || first >= n || count > n - first))
                    throw std::invalid_argument(std::format("Invalid native LOD child range (node={}, first_child={}, children={}, nodes={})", node, first, count, n));
                for (size_t child = first; child < first + count; ++child) {
                    if (parents[child] != 0xffffffffu)
                        throw std::invalid_argument(std::format("Native LOD node has multiple parents (child={}, existing_parent={}, new_parent={})", child, parents[child], node));
                    parents[child] = uint32_t(node);
                }
            }
            core::Tensor means_cpu, scales_cpu;
            if (tree.centers.size() < n)
                means_cpu = model.means_raw().cpu();
            if (tree.sizes.size() < n)
                scales_cpu = model.scaling_raw().to(core::DataType::Float32).cpu();
            const auto center = [&](size_t node) { return tree.centers.size() >= n ? tree.centers[node] : glm::vec3(means_cpu.ptr<float>()[node * 3], means_cpu.ptr<float>()[node * 3 + 1], means_cpu.ptr<float>()[node * 3 + 2]); };
            const auto size = [&](size_t node) { return tree.sizes.size() >= n ? tree.sizes[node] : 2.f * std::exp(std::max({scales_cpu.ptr<float>()[node * 3], scales_cpu.ptr<float>()[node * 3 + 1], scales_cpu.ptr<float>()[node * 3 + 2]})); };
            std::vector<simd_float4> frames(chunks * 4);
            for (size_t page = 0; page < chunks; ++page) {
                maps[page] = uint32_t(page);
                const size_t begin = page * chunk_size, end = std::min(begin + chunk_size, n);
                glm::vec3 lo(std::numeric_limits<float>::max()), hi(std::numeric_limits<float>::lowest());
                float log_lo = INFINITY, log_hi = -INFINITY;
                for (size_t node = begin; node < end; ++node) {
                    const auto c = center(node);
                    const float extent = size(node);
                    if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z) || !std::isfinite(extent) || extent <= 0)
                        throw std::invalid_argument(std::format("Invalid native LOD node bounds (node={}, center=[{},{},{}], extent={})", node, c.x, c.y, c.z, extent));
                    lo = glm::min(lo, c);
                    hi = glm::max(hi, c);
                    const float value = std::log(std::max(extent, 1e-8f));
                    log_lo = std::min(log_lo, value);
                    log_hi = std::max(log_hi, value);
                }
                const auto extent = glm::max(hi - lo, glm::vec3(0));
                const float log_range = std::max(log_hi - log_lo, 0.f);
                frames[page * 4 + 1] = {lo.x, lo.y, lo.z, log_lo};
                frames[page * 4 + 2] = {extent.x, extent.y, extent.z, log_range};
                const auto quant = [](float value, float base, float range) { return range > 0 ? uint32_t(std::lround(std::clamp((value - base) / range, 0.f, 1.f) * 65535.f)) : 0u; };
                for (size_t node = begin; node < end; ++node) {
                    const auto c = center(node);
                    bounds[node * 2] = quant(c.x, lo.x, extent.x) | (quant(c.y, lo.y, extent.y) << 16u);
                    bounds[node * 2 + 1] = quant(c.z, lo.z, extent.z) | (quant(std::log(std::max(size(node), 1e-8f)), log_lo, log_range) << 16u);
                    links[node * 3] = tree.child_start[node];
                    links[node * 3 + 1] = uint32_t(tree.child_count[node]) | (uint32_t(node < tree.lod_level.size() ? tree.lod_level[node] : 0) << 16u);
                    links[node * 3 + 2] = parents[node];
                    if (parents[node] == 0xffffffffu)
                        ++roots;
                }
            }
            LodTree metadata;
            metadata.signature = signature;
            metadata.nodes = uint32_t(n);
            metadata.chunks = uint32_t(chunks);
            metadata.roots = roots;
            metadata.last_used = serial;
            // Uploaded beside the splats; the tensors keep them until their last reader completes.
            const auto backend = model.means_raw().device() == core::Device::GPU ? core::gpu_backend_of(model.means_raw()) : std::nullopt;
            if (!backend)
                throw std::invalid_argument(std::format("GPU LOD requires resident GPU splats (device={}, nodes={})", int(model.means_raw().device()), n));
            const core::GpuBackendScope scope(*backend);
            const auto upload = [](const void* data, size_t bytes) {
                return core::Tensor::from_blob(const_cast<void*>(data), {bytes}, core::Device::CPU, core::DataType::UInt8).to(core::Device::GPU);
            };
            metadata.tensors = {upload(bounds.data(), bounds.size() * 4), upload(links.data(), links.size() * 4),
                                upload(maps.data(), maps.size() * 4), upload(age.data(), age.size() * 4),
                                upload(frames.data(), frames.size() * 16), upload(maps.data(), maps.size() * 4)};
            // Views can use independent models. Bound the shared metadata
            // cache; command buffers retain evicted resources until GPU completion.
            if (!lod_trees.contains(&tree) && lod_trees.size() >= 4) {
                auto oldest = std::min_element(lod_trees.begin(), lod_trees.end(), [](const auto& a, const auto& b) { return a.second.last_used < b.second.last_used; });
                lod_trees.erase(oldest);
            }
            return lod_trees.insert_or_assign(&tree, std::move(metadata)).first->second;
        }
        // `tensor` names the backend of the resident splats when the tensor
        // rasterizer draws the frame.
        Frame& acquire(Slot output, const rendering::ViewportRenderRequest& request, uint32_t count, bool points = false) {
            auto& state = target(output);
            // The published image may be cached by the compositor even after
            // its last GPU consumer completes. Never recycle it until another
            // frame has been submitted and published successfully.
            size_t index = state.next++ % state.frames.size();
            if (state.frames[index].get() == state.latest && state.latest)
                index = state.next++ % state.frames.size();
            auto& frame = state.frames[index];
            if (const uint64_t reported = state.overflow_required->exchange(0, std::memory_order_acq_rel))
                state.needed_capacity = std::max(state.needed_capacity, withGrowthHeadroom(reported));
            // Start near typical tile coverage and grow from measured demand: an
            // overflowing frame keeps the last image and requests a redraw that fits.
            uint32_t capacity = std::max(state.needed_capacity, static_cast<uint32_t>(std::min<uint64_t>(uint64_t(count) * 2 + 4096, 16u * 1024u * 1024u)));
            // A failed encode has no readable status. Replace it transactionally
            // too, so an admission/allocation failure cannot discard owned slots.
            if (frame && frame->command) {
                wait(frame->producer_value);
                if (frame->command.status == MTLCommandBufferStatusError)
                    throw std::runtime_error(std::format("Metal command failed while acquiring a viewport frame (status={}, producer={}, consumer={}, error_code={}, error={})", long(frame->command.status), frame->producer_value, frame->consumer_serial, long(frame->command.error.code), frame->command.error.localizedDescription.UTF8String ?: "none"));
                waitForConsumer(frame->consumer_serial);
                [frame->command waitUntilCompleted];
                if (const auto status = frame->rasterStatus(); status.error) {
                    if (status.required_instances > std::numeric_limits<uint32_t>::max())
                        throw std::runtime_error(std::format("Metal viewport instance count exceeds 32-bit indexing (required={}, capacity={})", status.required_instances, capacity));
                    capacity = std::max(capacity, withGrowthHeadroom(status.required_instances));
                }
                if (frame->points == points && frame->size == request.frame_view.size && frame->count >= count && frame->capacity >= capacity)
                    return *frame;
            }
            // Account for all Metal allocations on the shared device, including
            // resident tensors and Vulkan presentation. Fail before a large growth
            // can exhaust unified memory; retain the last completed output.
            const auto device = reader.device();
            const auto outputs = viewportOutputReservationBytes(request.frame_view.size.x, request.frame_view.size.y, points);
            if (!frameFitsWorkingSet(device.currentAllocatedSize, outputs, device.recommendedMaxWorkingSetSize))
                throw lfs::Exception(nativeError(std::format("Metal viewport reservation exceeds the recommended GPU working set (extent={}x{}, count={}, capacity={}, reservation={}, allocated={}, recommended={})", request.frame_view.size.x, request.frame_view.size.y, count, capacity, outputs, device.currentAllocatedSize, device.recommendedMaxWorkingSetSize), lfs::ErrorCode::ResourceExhausted));
            auto candidate = std::make_unique<Frame>();
            auto& f = *candidate;
            f.size = request.frame_view.size;
            f.count = count;
            f.capacity = capacity;
            f.generation = ++generation;
            f.points = points;
            if (!points) {
                f.raster_status = [device newBufferWithLength:sizeof(RasterStatus) options:MTLResourceStorageModeShared];
                if (!f.raster_status)
                    throw lfs::Exception(nativeError("Metal raster status staging allocation failed", lfs::ErrorCode::ResourceExhausted));
            } else {
                auto depth_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:f.size.x height:f.size.y mipmapped:NO];
                depth_descriptor.storageMode = MTLStorageModePrivate;
                depth_descriptor.usage = MTLTextureUsageRenderTarget;
                f.point_depth = [device newTextureWithDescriptor:depth_descriptor];
                if (!f.point_depth)
                    throw lfs::Exception(nativeError(std::format("Metal point depth allocation failed (extent={}x{})", f.size.x, f.size.y), lfs::ErrorCode::ResourceExhausted));
            }
#ifdef LFS_GRAPHICS_VULKAN
            f.color.init(*context, device, f.size.x, f.size.y, MTLPixelFormatRGBA8Unorm, VK_FORMAT_R8G8B8A8_UNORM);
            f.depth.init(*context, device, f.size.x, f.size.y, MTLPixelFormatR32Float, VK_FORMAT_R32_SFLOAT);
#else
            f.color.init(*context, device, f.size.x, f.size.y, MTLPixelFormatRGBA8Unorm);
            f.depth.init(*context, device, f.size.x, f.size.y, MTLPixelFormatR32Float);
#endif
            // Commit only a fully allocated replacement. Exceptions destroy the
            // unpublished candidate and leave existing slot ownership intact.
            frame = std::move(candidate);
            return f;
        }
        id<MTLBuffer> readTexture(Frame& f, Image& image, size_t bytes_per_pixel, glm::ivec2 pixel = {-1, -1}) const {
            wait(f.producer_value);
            if (f.command.status == MTLCommandBufferStatusError)
                throw std::runtime_error(std::format("Metal command failed before texture readback (status={}, producer={}, extent={}x{}, error_code={}, error={})", long(f.command.status), f.producer_value, f.size.x, f.size.y, long(f.command.error.code), f.command.error.localizedDescription.UTF8String ?: "none"));
            auto queue = readback_queue;
            auto command = [queue commandBuffer];
            const bool sample = pixel.x >= 0;
            const size_t width = sample ? 1 : f.size.x, height = sample ? 1 : f.size.y;
            const size_t row = width * bytes_per_pixel;
            auto result = [reader.device() newBufferWithLength:row * height options:MTLResourceStorageModeShared];
            if (!queue || !command || !result)
                throw lfs::Exception(nativeError(std::format("Metal readback allocation failed (bytes={}, queue_present={}, command_present={})", row * height, queue != nil, command != nil), lfs::ErrorCode::ResourceExhausted));
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
                throw std::runtime_error(std::format("Metal output readback failed (status={}, extent={}x{}, bytes={}, error={})", long(command.status), width, height, row * height, command.error.localizedDescription.UTF8String ?: "none"));
            return result;
        }
    };
    MetalViewportRenderer::MetalViewportRenderer() : impl_(std::make_unique<Impl>()) {}
    MetalViewportRenderer::~MetalViewportRenderer() = default;
    void MetalViewportRenderer::setRetryCallback(std::function<void()> callback) {
        std::lock_guard lock(impl_->readback_mutex);
        impl_->retry_callback = std::move(callback);
    }
    void MetalViewportRenderer::setLodSettings(size_t splats, float fraction, uint32_t fade) {
        std::lock_guard lock(impl_->readback_mutex);
        impl_->rad_settings = {splats, fraction, fade};
    }
    bool MetalViewportRenderer::supportsSelection(const core::SplatData& model, const SceneRenderer::SelectionMaskRequest& request) {
        const bool rad_preview = model.lod_tree && model.lod_tree->rad_source.valid() &&
                                 model.means_raw().device() == core::Device::CPU && core::tensor_backend_supports_metal_access();
        const auto resident = [&](const core::Tensor& tensor) {
            return tensor.is_valid() && tensor.is_contiguous() &&
                   (nativeStorage(tensor) || (rad_preview && tensor.device() == core::Device::CPU));
        };
        const auto& means = model.means_raw();
        if (!resident(means) || means.dtype() != core::DataType::Float32 || means.ndim() != 2 || means.size(1) != 3)
            return false;
        const bool geometry = request.gut || request.shape == SceneRenderer::SelectionMaskShape::Ring;
        if (geometry && (!resident(model.scaling_raw()) || !resident(model.rotation_raw()) ||
                         !((model.scaling_raw().dtype() == core::DataType::Float32 && model.rotation_raw().dtype() == core::DataType::Float32 && model.opacity_raw().dtype() == core::DataType::Float32) || model.non_sh_attrs_f16())))
            return false;
        if (request.shape == SceneRenderer::SelectionMaskShape::Ring && !resident(model.opacity_raw()))
            return false;
        if (model.deleted().is_valid() && (!resident(model.deleted()) ||
                                           (model.deleted().dtype() != core::DataType::Bool && model.deleted().dtype() != core::DataType::UInt8)))
            return false;
        const auto indices = request.scene.transform_indices.get();
        return !indices || !indices->is_valid() ||
               (indices->is_contiguous() && nativeStorage(*indices) &&
                indices->dtype() == core::DataType::Int32 && indices->bytes() >= size_t(model.size()) * 4);
    }
    lfs::Result<core::Tensor> MetalViewportRenderer::buildSelectionMask(MetalViewportPresentation& context, const core::SplatData& model,
                                                                        const SceneRenderer::SelectionMaskRequest& request) {
        try {
            auto& i = *impl_;
            std::lock_guard lock(i.readback_mutex);
            i.initialize(context);
            if (request.picked_ring_id_out)
                *request.picked_ring_id_out = 0xffffffffu;
            const size_t n = model.size();
            const bool polygon = request.shape == SceneRenderer::SelectionMaskShape::Polygon;
            const bool ring = request.shape == SceneRenderer::SelectionMaskShape::Ring;
            if (!supportsSelection(model, request) || !n || n > std::numeric_limits<uint32_t>::max() ||
                request.frame_view.size.x <= 0 || request.frame_view.size.y <= 0 ||
                request.primitives.size() > std::numeric_limits<uint32_t>::max() || request.polygon_vertices.size() > std::numeric_limits<uint32_t>::max() ||
                (polygon ? request.polygon_vertices.size() < 3 : request.primitives.empty()))
                throw std::invalid_argument(std::format("Invalid native Metal selection request (splats={}, extent={}x{}, primitives={}, vertices={}, polygon={})", n, request.frame_view.size.x, request.frame_view.size.y, request.primitives.size(), request.polygon_vertices.size(), polygon));
            const bool geometry = request.gut || ring;
            // A paged model selects against its resident preview prefix.
            std::array<const core::Tensor*, 6> tensors{&model.means_raw(), geometry ? &model.scaling_raw() : nullptr,
                                                       geometry ? &model.rotation_raw() : nullptr, ring ? &model.opacity_raw() : nullptr,
                                                       &model.deleted(), nullptr};
            if (model.means_raw().device() == core::Device::CPU) {
                auto& pager = i.selection_pager;
                if (!pager)
                    pager = std::make_unique<MetalRadPager>(i.reader.device());
                i.configurePager(*pager, model);
                const auto& prefix = pager->preview(model);
                tensors[0] = &prefix[0];
                if (geometry) {
                    tensors[1] = &prefix[1];
                    tensors[2] = &prefix[2];
                }
                if (ring)
                    tensors[3] = &prefix[3];
                tensors[4] = &pager->deleted(model);
            }
            // Editor masks stay on the backend of the selected splats.
            const auto backend = core::gpu_backend_of(*tensors[0]);
            if (tensors[0]->device() != core::Device::GPU || !backend)
                throw std::invalid_argument(std::format("Selection requires resident GPU splats (device={}, splats={})", int(tensors[0]->device()), n));
            const core::GpuBackendScope scope(*backend);
            if (!frameFitsWorkingSet(i.reader.device().currentAllocatedSize, n, i.reader.device().recommendedMaxWorkingSetSize))
                throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = n, .label = "viewer.selection.result", .operation = "viewer.buffer.allocate"});
            auto output = core::Tensor::empty({n}, core::Device::GPU, core::DataType::Bool);
            rendering::SplatSelectionParameters parameters;
            auto view = glm::mat4(glm::transpose(rendering::dataCameraToWorldFromVisualizerRotation(request.frame_view.rotation)));
            view[3] = glm::vec4(-glm::mat3(view) * request.frame_view.translation, 1);
            std::memcpy(parameters.world_to_camera.data(), glm::value_ptr(view), sizeof(parameters.world_to_camera));
            const auto intrinsics = request.frame_view.getCameraIntrinsics();
            parameters.intrinsics = {intrinsics.focal_x, intrinsics.focal_y, intrinsics.center_x, intrinsics.center_y};
            parameters.image = {uint32_t(request.frame_view.size.x), uint32_t(request.frame_view.size.y),
                                uint32_t(request.equirectangular ? CameraModel::Equirectangular : request.frame_view.orthographic ? CameraModel::Orthographic
                                                                                                                                   : CameraModel::Perspective),
                                uint32_t(request.gut)};
            parameters.source = {uint32_t(n), uint32_t(request.shape), uint32_t(request.primitives.size()), uint32_t(request.polygon_vertices.size())};
            parameters.payload = {uint32_t(model.non_sh_attrs_f16()), uint32_t(request.mip_filter), 0, 0};
            parameters.ring = {request.ring_width, kViewerNearClip, 0, 0};
            if (polygon) {
                glm::vec2 lo(std::numeric_limits<float>::max()), hi(std::numeric_limits<float>::lowest());
                for (const auto vertex : request.polygon_vertices) {
                    if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y))
                        throw std::invalid_argument(std::format("Invalid selection polygon vertex (x={}, y={})", vertex.x, vertex.y));
                    lo = glm::min(lo, vertex);
                    hi = glm::max(hi, vertex);
                }
                const auto size = request.frame_view.size;
                // Clamp in floating point before integer conversion, including
                // huge off-screen gestures; no out-of-range C++ float cast.
                const glm::ivec2 begin(glm::clamp(glm::floor(lo), glm::vec2(0), glm::vec2(size)));
                const glm::ivec2 end(glm::clamp(glm::ceil(hi), glm::vec2(0), glm::vec2(size)));
                if (end.x <= begin.x || end.y <= begin.y)
                    return core::Tensor::zeros({n}, core::Device::GPU, core::DataType::Bool);
                parameters.aabb = {uint32_t(begin.x), uint32_t(begin.y), uint32_t(end.x - begin.x), uint32_t(end.y - begin.y)};
            }
            const auto indices = request.scene.transform_indices.get();
            const bool indexed = indices && indices->is_valid();
            const size_t transforms = request.scene.model_transforms ? request.scene.model_transforms->size() : 0;
            if (transforms > std::numeric_limits<int32_t>::max() || request.scene.node_visibility_mask.size() > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument(std::format("Selection scene exceeds transform indexing limits (transforms={}, indexed={})", transforms, indexed));
            if (transforms)
                for (const auto& transform : *request.scene.model_transforms)
                    for (size_t col = 0; col < 4; ++col)
                        for (size_t row = 0; row < 4; ++row)
                            if (!std::isfinite(transform[col][row]))
                                throw std::invalid_argument(std::format("Invalid selection object matrix (column={}, row={}, value={})", col, row, transform[col][row]));
            if (!polygon)
                for (const auto primitive : request.primitives)
                    for (size_t axis = 0; axis < 4; ++axis)
                        if (!std::isfinite(primitive[axis]))
                            throw std::invalid_argument(std::format("Invalid selection primitive (axis={}, value={})", axis, primitive[axis]));
            const std::vector<uint8_t> visibility(request.scene.node_visibility_mask.begin(), request.scene.node_visibility_mask.end());
            tensors[5] = indexed ? indices : nullptr;
            parameters.scene = {uint32_t(transforms), uint32_t(indexed), uint32_t(visibility.size()), uint32_t(std::min(n, tensors[4]->bytes()))};
            rendering::SplatSelectionInputs inputs{tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]->bytes() ? tensors[4] : nullptr, tensors[5]};
            if (transforms)
                inputs.transforms = std::as_bytes(std::span(*request.scene.model_transforms));
            inputs.visibility = std::as_bytes(std::span(visibility));
            if (!polygon)
                inputs.primitives = std::as_bytes(std::span(request.primitives));
            else
                inputs.polygon_vertices = std::as_bytes(std::span(request.polygon_vertices));
            if (!i.selection_query || i.selection_backend != *backend) {
                i.selection_query = std::make_unique<rendering::SplatSelectionQuery>(*backend);
                i.selection_backend = *backend;
            }
            auto pick = ring ? core::Tensor::empty({2}, core::Device::GPU, core::DataType::UInt32) : core::Tensor{};
            if (auto queried = i.selection_query->query(inputs, parameters, output, ring ? &pick : nullptr); !queried)
                throw lfs::Exception(queried.error());
            if (ring && request.picked_ring_id_out) {
                const auto host = pick.to(core::Device::CPU);
                const auto* words = static_cast<const uint32_t*>(host.data_ptr());
                if (words[0] != 0xffffffffu && words[1] < n)
                    *request.picked_ring_id_out = words[1];
            }
            return output;
        } catch (const std::exception& error) { return nativeError(error); }
    }
    bool MetalViewportRenderer::supportsPoints(const PointSceneRenderer::RenderRequest& r) {
        const auto resident = [](const core::Tensor* t) { return t && t->is_valid() && t->is_contiguous() && nativeStorage(*t); };
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
    lfs::Result<PointSceneRenderer::RenderResult> MetalViewportRenderer::renderPoints(
        MetalViewportPresentation& context, const PointSceneRenderer::RenderRequest& r, RenderTargetId output) {
        try {
            if (!supportsPoints(r))
                throw std::invalid_argument(std::format("Unsupported native Metal point request (extent={}x{}, positions_shape={}, positions_dtype={}, positions_contiguous={}, colors_shape={}, colors_dtype={}, colors_contiguous={})", r.size.x, r.size.y, r.positions ? r.positions->shape().str() : "missing", r.positions ? int(r.positions->dtype()) : -1, r.positions && r.positions->is_contiguous(), r.colors ? r.colors->shape().str() : "missing", r.colors ? int(r.colors->dtype()) : -1, r.colors && r.colors->is_contiguous()));
            auto& i = *impl_;
            std::lock_guard lock(i.readback_mutex);
            i.initialize(context);
            i.drainRetiredTargets();
            i.stampConsumers();
            const auto slot = output;
            (void)i.target(slot);
            rendering::ViewportRenderRequest request;
            request.frame_view.size = r.size;
            auto& f = i.acquire(slot, request, uint32_t(r.positions->size(0)), true);
            const auto valid = [](const core::Tensor* t) { return t && t->is_valid(); };
            const size_t nodes = r.model_transforms ? r.model_transforms->size() : 0;
            if (nodes > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error(std::format("Metal point object count exceeds indexing (nodes={}, max={})", nodes, std::numeric_limits<uint32_t>::max()));
            const auto allocate = [&](id<MTLBuffer> __strong& buffer, size_t bytes) {
                if (!buffer || buffer.length < bytes)
                    buffer = [i.reader.device() newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
                if (!buffer)
                    throw lfs::Exception(nativeError(std::format("Metal point allocation failed (bytes={}, allocated={}, recommended={})", bytes, i.reader.device().currentAllocatedSize, i.reader.device().recommendedMaxWorkingSetSize), lfs::ErrorCode::ResourceExhausted));
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
#ifdef LFS_GRAPHICS_VULKAN
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(context.physicalDevice(), &props);
            p.counts.w = uint32_t(std::min(511.f, props.limits.pointSizeRange[1]));
#else
            p.counts.w = 511; // Metal clamps point_size to 511
#endif
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
                throw std::runtime_error(std::format("Metal point timeline exhausted (serial={})", i.serial));
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
                    throw std::runtime_error(std::format("Metal point render encoder failed (command_status={}, extent={}x{}, points={})", long(command.status), f.size.x, f.size.y, r.positions->size(0)));
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
                const auto completion_event = event;
                const auto completion_value = serial;
                [command encodeSignalEvent:completion_event value:completion_value];
                [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    if (completed.status == MTLCommandBufferStatusError && completion_event.signaledValue < completion_value)
                        completion_event.signaledValue = completion_value;
                }];
                const auto completion_retry = i.retry_callback;
                if (completion_retry) {
                    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                        if (completed.status == MTLCommandBufferStatusError)
                            completion_retry();
                    }];
                }
            });
            f.producer_value = serial;
            i.serial = serial;
            f.consumer_serial = i.consumerSerial();
            i.target(slot).latest = &f;
#ifndef LFS_GRAPHICS_VULKAN
            return PointSceneRenderer::RenderResult{.generation = f.generation, .depth_generation = f.generation, .size = f.size, .completion_value = serial, .viewer_backend = rendering::ViewerBackend::Metal};
#else
            return PointSceneRenderer::RenderResult{.image = sceneImageHandle(f.color.image), .image_view = sceneImageViewHandle(f.color.view), .image_layout = sceneImageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL), .generation = f.generation, .depth_image = sceneImageHandle(f.depth.image), .depth_image_view = sceneImageViewHandle(f.depth.view), .depth_image_layout = sceneImageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL), .depth_generation = f.generation, .size = f.size, .flip_y = false, .completion_semaphore = sceneTimelineHandle(i.completion), .completion_value = serial, .viewer_backend = rendering::ViewerBackend::Metal};
#endif
        } catch (const std::exception& e) { return nativeError(e); }
    }
    bool MetalViewportRenderer::supports(const core::SplatData& model, const rendering::ViewportRenderRequest& r) {
        // Validate the native scene contract before encoding. Unsupported requests
        // report an error; filters, settings and overlays are never silently dropped.
        // Short byte masks have an unselected suffix; both native overlay
        // stages guard logical IDs against the exact resident mask extent.
        const auto resident_mask = [](const core::Tensor* mask) { return !mask || !mask->is_valid() ||
                                                                         (nativeStorage(*mask) && mask->is_contiguous() && mask->bytes() > 0 &&
                                                                          (mask->dtype() == core::DataType::UInt8 || mask->dtype() == core::DataType::Bool)); };
        // A fully resident RAD can use the ordinary source path. Paging is
        // required only for a GPU hierarchy cut or a partial resident preview.
        const bool rad = model.lod_tree && model.lod_tree->rad_source.valid() &&
                         (r.lod_gpu_traversal.enabled || model.lod_tree->total_nodes() > size_t(model.size()));
        const size_t logical_count = rad ? model.lod_tree->total_nodes() : size_t(model.size());
        const auto indices = r.scene.transform_indices.get();
        const size_t objects = r.scene.model_transforms ? r.scene.model_transforms->size() : 0;
        const bool indexed = indices && indices->is_valid();
        if ((objects > 1 && !indexed) || (indexed &&
                                          (!nativeStorage(*indices) || !indices->is_contiguous() ||
                                           indices->dtype() != core::DataType::Int32 || indices->bytes() < logical_count * 4)))
            return false;
        return resident_mask(r.overlay.emphasis.mask.get()) && resident_mask(r.overlay.emphasis.transient_mask.mask) &&
               (nativeStorage(model.means_raw()) ||
                (rad && model.means_raw().device() == core::Device::CPU && core::tensor_backend_supports_metal_access())) &&
               (r.splat_render_profile == 0 || r.splat_render_profile == 1) &&
               (!r.lod_gpu_traversal.enabled || (model.lod_tree && model.lod_tree->has_tree() &&
                                                 r.lod_gpu_traversal.node_count == model.lod_tree->total_nodes() && (rad || model.lod_tree->total_nodes() <= size_t(model.size())) &&
                                                 r.lod_gpu_traversal.output_capacity > 0 && r.lod_gpu_traversal.output_capacity <= std::numeric_limits<uint32_t>::max())) &&
               (!rad || (model.lod_tree->rad_source.chunk_size >= core::SplatLodTree::kChunkSplats && model.lod_tree->rad_source.chunk_size % core::SplatLodTree::kChunkSplats == 0)) &&
               (!r.lod_indices || r.lod_count <= std::numeric_limits<uint32_t>::max()) &&
               model.means_raw().dtype() == core::DataType::Float32 && model.sh0_raw().dtype() == core::DataType::Float32 &&
               ((model.scaling_raw().dtype() == core::DataType::Float32 && model.rotation_raw().dtype() == core::DataType::Float32 &&
                 model.opacity_raw().dtype() == core::DataType::Float32) ||
                model.non_sh_attrs_f16());
    }
    lfs::Result<SceneRenderer::RenderResult> MetalViewportRenderer::render(
        MetalViewportPresentation& context, const core::SplatData& model, const rendering::ViewportRenderRequest& request, Slot slot, bool expected_depth, bool wait_for_pages) {
        try {
            auto& i = *impl_;
            std::lock_guard lock(i.readback_mutex);
            i.initialize(context);
            // Cached outputs may be sampled for many GUI frames without a new
            // raster submission. Stamp their most recent graphics consumer before
            // replacing latest, rather than retiring against the first consumer.
            i.drainRetiredTargets();
            i.stampConsumers();
            auto& state = i.target(slot);
            if (model.size() > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error(std::format("Metal primitive count exceeds indexing capacity (splats={}, max={})", model.size(), std::numeric_limits<uint32_t>::max()));
            const bool rad = model.lod_tree && model.lod_tree->rad_source.valid() &&
                             (request.lod_gpu_traversal.enabled || model.lod_tree->total_nodes() > size_t(model.size()));
            MetalRadPager* pager = nullptr;
            Impl::LodTree paged_tree;
            if (rad) {
                if (!state.pager)
                    state.pager = std::make_unique<MetalRadPager>(i.reader.device());
                pager = state.pager.get();
                i.configurePager(*pager, model);
                const Frame* completed = nullptr;
                for (const auto& candidate : state.frames)
                    if (candidate && candidate->lodResultReady() && candidate->rad_signature == pager->signature() &&
                        (!completed || candidate->producer_value > completed->producer_value))
                        completed = candidate.get();
                pager->advance(completed ? std::span<const uint32_t>(static_cast<const uint32_t*>(completed->lod_touches.contents), pager->cache().snapshot().logical_chunks)
                                         : std::span<const uint32_t>{});
                if (wait_for_pages)
                    pager->waitForRoot();
                paged_tree.nodes = pager->nodes();
                paged_tree.chunks = uint32_t(pager->cache().snapshot().logical_chunks);
                paged_tree.roots = 1;
                paged_tree.signature = pager->signature();
            }
            const bool gpu_lod = request.lod_gpu_traversal.enabled && (!pager || pager->rootReady());
            auto* gpu_tree = gpu_lod ? (pager ? &paged_tree : &i.prepareLodTree(model)) : nullptr;
            const uint32_t source_count = pager && gpu_lod ? pager->physicalNodes() : uint32_t(model.size());
            const uint32_t logical_count = pager ? pager->nodes() : source_count;
            const uint32_t draw_count = gpu_lod ? uint32_t(std::clamp<size_t>(request.lod_gpu_traversal.output_capacity, gpu_tree->roots, std::min(gpu_tree->nodes, source_count)))
                                                : uint32_t(pager ? model.size() : request.lod_indices ? request.lod_count
                                                                                                      : model.size());
            const int node_degree = request.scene.node_active_sh_degrees.empty() ? model.get_active_sh_degree() : *std::max_element(request.scene.node_active_sh_degrees.begin(), request.scene.node_active_sh_degrees.end());
            const int max_degree = std::clamp(node_degree, 0, std::min(3, model.get_max_sh_degree()));
            const uint32_t degree = static_cast<uint32_t>(std::clamp(request.sh_degree, 0, max_degree));
            // The resident inputs: the model, a paged model's preview prefix, or its page pool.
            const auto selection = request.overlay.emphasis.mask.get();
            const auto preview = request.overlay.emphasis.transient_mask.mask;
            const bool selection_enabled = request.overlay.has_selection && selection && selection->is_valid();
            const bool preview_enabled = preview && preview->is_valid();
            std::array<const core::Tensor*, 8> tensors{&model.means_raw(), &model.scaling_raw(), &model.rotation_raw(),
                                                       &model.opacity_raw(), &model.sh0_raw(), degree ? &model.shN_raw() : nullptr,
                                                       degree ? &model.shN_value_bounds() : nullptr, &model.deleted()};
            if (pager) {
                if (gpu_lod) {
                    const auto& pool = pager->pool();
                    const std::array<size_t, 7> order{0, 4, 3, 5, 1, 2, 6};
                    for (size_t n = 0; n < order.size(); ++n)
                        tensors[n] = &pool.regions[order[n]];
                } else {
                    const auto& prefix = pager->preview(model);
                    for (size_t n = 0; n < prefix.size(); ++n)
                        tensors[n] = &prefix[n];
                }
                tensors[7] = &pager->deleted(model);
            }
            // Every pass runs on the backend of the resident splats.
            const auto backend = tensors[0]->device() == core::Device::GPU ? core::gpu_backend_of(*tensors[0]) : std::nullopt;
            const auto beside_splats = [&](const core::Tensor* tensor) { return !tensor || core::gpu_backend_of(*tensor) == backend; };
            if (!backend || !beside_splats(selection_enabled ? selection : nullptr) || !beside_splats(preview_enabled ? preview : nullptr))
                throw std::invalid_argument(std::format("Viewport rendering requires GPU splats and masks on one backend (device={}, selection={}, preview={})",
                                                        int(tensors[0]->device()), selection_enabled, preview_enabled));
            const core::GpuBackendScope scope(*backend);
            if (!state.raster || state.backend != *backend) {
                state.raster = std::make_unique<rendering::SplatRasterizer>(*backend);
                state.lod.reset();
                state.backend = *backend;
            }
            if (!i.projector || i.projector_backend != *backend) {
                i.projector = std::make_unique<rendering::SplatProjector>(*backend);
                i.projector_backend = *backend;
            }
            auto& f = i.acquire(slot, request, draw_count);
            f.gpu_lod_active = gpu_lod;
            f.rad_bootstrap = pager && !gpu_lod;
            f.rad_signature = pager ? pager->signature() : 0;
            // One rasterizer serves every frame of the target's ring: follow
            // this frame's extent and capacity (no-op when unchanged).
            if (auto reserved = state.raster->reserve(draw_count, uint32_t(f.size.x), uint32_t(f.size.y), f.capacity); !reserved)
                throw lfs::Exception(reserved.error());
            const auto storage = pager && gpu_lod ? rendering::SplatShStorage::RadSigned8 : model.shN_value_quantized() ? rendering::SplatShStorage::Q16
                                                                                    : model.shN_ieee_f16()          ? rendering::SplatShStorage::SwizzledFloat16
                                                                                                                    : rendering::SplatShStorage::SwizzledFloat32;
            const bool rad_pool = storage == rendering::SplatShStorage::RadSigned8;
            rendering::SplatProjection projection;
            const auto store = [](std::array<float, 16>& to, const glm::mat4& m) { std::memcpy(to.data(), glm::value_ptr(m), sizeof(to)); };
            store(projection.model_to_world, glm::mat4(1));
            auto view = glm::mat4(glm::transpose(rendering::dataCameraToWorldFromVisualizerRotation(request.frame_view.rotation)));
            view[3] = glm::vec4(-glm::mat3(view) * request.frame_view.translation, 1);
            store(projection.world_to_camera, view);
            const auto camera = request.frame_view.translation;
            projection.camera_local = {camera.x, camera.y, camera.z, 1};
            const auto intrinsics = request.frame_view.getCameraIntrinsics();
            // Rectilinear projection and GUT rays use crop-local pixels.
            // Preserve full-camera dimensions/origin separately for binning,
            // overlays, and periodic panorama projection.
            projection.intrinsics = {intrinsics.focal_x, intrinsics.focal_y,
                                     intrinsics.center_x - float(request.frame_view.subregion_origin.x),
                                     intrinsics.center_y - float(request.frame_view.subregion_origin.y)};
            // Viewer raster clipping differs from the desktop projection matrix's
            // near/far planes. Derive the reference near threshold at configure.
            const bool portal = request.splat_render_profile == 1;
            const bool spark = (pager || gpu_lod || (request.lod_indices && request.lod_count)) && model.lod_tree && model.lod_tree->lod_opacity_encoded;
            const bool portal_math = portal && !spark;
            const bool mip = request.mip_filter && !(portal_math && request.gut);
            const float dilation = portal_math && !request.gut ? .075f : mip ? .1f
                                                                             : .3f;
            const auto camera_model = request.equirectangular             ? CameraModel::Equirectangular
                                      : request.frame_view.orthographic ? CameraModel::Orthographic
                                                                        : CameraModel::Perspective;
            projection.clip_scale = {kViewerNearClip, std::numeric_limits<float>::max(), request.scaling_modifier, dilation};
            projection.extent = {uint32_t(f.size.x), uint32_t(f.size.y), uint32_t(camera_model), uint32_t(mip)};
            projection.rasterization = {request.frame_view.rasterization_scale, expected_depth ? 1.f : 0.f, request.frame_view.far_plane, float(request.splat_render_profile)};
            projection.display = {float(request.color_tonemapping), request.color_exposure, spark ? 1.f : 0.f, request.gut ? 0.f : 1.f};
            const auto panorama_size = request.frame_view.cameraSize();
            projection.panorama = {float(panorama_size.x), float(panorama_size.y), float(request.frame_view.subregion_origin.x), float(request.frame_view.subregion_origin.y)};
            std::array<float, 2> render_origin{float(request.frame_view.subregion_origin.x), float(request.frame_view.subregion_origin.y)};
            const auto allocate = [&](id<MTLBuffer> __strong& buffer, size_t bytes) {
                if (!buffer || buffer.length < bytes)
                    buffer = [i.reader.device() newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
                if (!buffer)
                    throw lfs::Exception(nativeError(std::format("Metal frame staging allocation failed (bytes={}, target={})", bytes, slot.value), lfs::ErrorCode::ResourceExhausted));
            };
            size_t scene_count = 0;
            if (request.scene.model_transforms && !request.scene.model_transforms->empty()) {
                const auto& transforms = *request.scene.model_transforms;
                if (transforms.size() > 1 && (!request.scene.transform_indices || !request.scene.transform_indices->is_valid()))
                    throw std::invalid_argument(std::format("Multiple Metal scene transforms require primitive indices (transforms={}, draw_count={})", request.scene.model_transforms->size(), draw_count));
                allocate(f.objects, transforms.size() * sizeof(SceneObject));
                auto objects = static_cast<SceneObject*>(f.objects.contents);
                for (size_t n = 0; n < transforms.size(); ++n) {
                    const auto local = glm::inverse(transforms[n]) * glm::vec4(camera, 1);
                    const bool visible = n >= request.scene.node_visibility_mask.size() || request.scene.node_visibility_mask[n];
                    const int active = n < request.scene.node_active_sh_degrees.size() ? request.scene.node_active_sh_degrees[n] : degree;
                    objects[n] = {matrix(transforms[n]), {local.x, local.y, local.z, 1}, {uint32_t(visible), uint32_t(std::clamp(active, 0, 3)), 0, 0}};
                }
                scene_count = transforms.size();
            }
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
                const auto params = detail::buildOverlayParamsCpuFloats(request, selection_enabled, preview_enabled, scene_count > 0, node_count, false);
                if (!params)
                    throw std::runtime_error(params.error());
                allocate(f.overlay_parameters, params->size() * sizeof(float));
                std::memcpy(f.overlay_parameters.contents, params->data(), params->size() * sizeof(float));
                allocate(f.overlay_nodes, node_count);
                auto nodes = static_cast<uint8_t*>(f.overlay_nodes.contents);
                for (size_t n = 0; n < node_count; ++n)
                    nodes[n] = request.overlay.emphasis.emphasized_node_mask[n];
                allocate(f.selection_colors, sizeof(request.overlay.selection_colors));
                std::memcpy(f.selection_colors.contents, request.overlay.selection_colors.data(), sizeof(request.overlay.selection_colors));
                render_origin = {projection.panorama[2], projection.panorama[3]};
            }
            const auto host_bytes = [](id<MTLBuffer> buffer, size_t bytes) { return std::span(static_cast<const std::byte*>(buffer.contents), bytes); };
            const auto resident = [](const core::Tensor* tensor) { return tensor && tensor->is_valid() && tensor->numel() ? tensor : nullptr; };
            rendering::SplatSources sources;
            sources.means = tensors[0];
            sources.scales = tensors[1];
            sources.rotations = tensors[2];
            sources.opacity = tensors[3];
            sources.sh0 = tensors[4];
            sources.sh_rest = degree ? tensors[5] : nullptr;
            sources.sh_bounds = resident(tensors[6]);
            sources.deleted = resident(tensors[7]);
            sources.count = source_count;
            sources.layout_rest = uint32_t(model.max_sh_coeffs_rest());
            sources.storage = storage;
            sources.half_attributes = rad_pool || model.non_sh_attrs_f16();
            sources.page_splats = rad_pool ? uint32_t(core::SplatLodTree::kChunkSplats) : 0;
            sources.deleted_count = sources.deleted ? uint32_t(std::min<size_t>(sources.deleted->bytes(), std::numeric_limits<uint32_t>::max())) : 0;
            if (scene_count) {
                sources.objects = host_bytes(f.objects, scene_count * sizeof(SceneObject));
                sources.object_indices = request.scene.transform_indices.get();
            }
            const bool transparent = request.transparent_background;
            const rendering::SplatOverlayInputs overlay_inputs{needs_overlay ? host_bytes(f.overlay_parameters, 207 * 16) : std::span<const std::byte>{},
                                                               node_count ? host_bytes(f.overlay_nodes, node_count) : std::span<const std::byte>{}};
            if (request.gut && (!i.gut_geometry.is_valid() || i.gut_geometry.bytes() < size_t(draw_count) * 64 || core::gpu_backend_of(i.gut_geometry) != *backend))
                i.gut_geometry = core::Tensor::empty({std::max<size_t>(16, size_t(draw_count) * 64)}, core::Device::GPU, core::DataType::UInt8);
            if (!i.projected.is_valid() || i.projected.bytes() < size_t(draw_count) * 64 || core::gpu_backend_of(i.projected) != *backend)
                i.projected = core::Tensor::empty({std::max<size_t>(16, size_t(draw_count) * 64)}, core::Device::GPU, core::DataType::UInt8);
            // A LOD cut selected on the host: physical indices, plus optional
            // logical IDs, debug levels and transition weights.
            const auto cut = [&](const auto* values) { return values ? std::span(values, draw_count) : std::span<std::remove_const_t<std::remove_pointer_t<decltype(values)>> const>{}; };
            const bool host_cut = !gpu_lod && request.lod_indices != nullptr;
            const auto host_cut_tensors = host_cut ? i.projector->upload_cut({cut(request.lod_indices), cut(request.lod_logical_indices), cut(request.lod_levels),
                                                                             cut(request.lod_weights), request.lod_debug_mode, uint32_t(model.size())})
                                                   : rendering::SplatLodCut{};
            // A GPU-selected cut: the LOD selector writes indices and its count.
            rendering::SplatLodCut gpu_cut{};
            if (gpu_lod) {
                if (!state.lod)
                    state.lod = std::make_unique<rendering::SplatLodSelector>(*backend);
                if (auto reserved = state.lod->reserve(draw_count, source_count, gpu_tree->chunks); !reserved)
                    throw lfs::Exception(reserved.error());
                rendering::SplatLodParameters lod_parameters;
                const auto& p = request.lod_gpu_traversal;
                lod_parameters.node_count = gpu_tree->nodes;
                lod_parameters.physical_node_count = source_count;
                lod_parameters.output_capacity = draw_count;
                lod_parameters.logical_chunk_count = gpu_tree->chunks;
                lod_parameters.chunk_splats = uint32_t(core::SplatLodTree::kChunkSplats);
                rendering::SplatLodTree tree;
                if (pager) {
                    lod_parameters.current_frame = uint32_t(pager->cache().frameIndex());
                    lod_parameters.fade_frames = wait_for_pages ? 0 : pager->fadeFrames();
                    const auto& pool = pager->pool();
                    const auto& snapshot = pager->cache().snapshot();
                    const std::array<std::span<const uint32_t>, 3> maps{snapshot.chunk_to_page, snapshot.page_resident_frame, snapshot.page_to_chunk};
                    std::erase_if(state.uploads, [](core::TensorUpload& upload) { return upload.poll(); });
                    for (size_t n = 0; n < maps.size(); ++n) {
                        auto& map = state.page_maps[n];
                        const size_t bytes = std::max<size_t>(4, maps[n].size_bytes());
                        if (!map.is_valid() || map.bytes() != bytes || core::gpu_backend_of(map) != *backend)
                            map = core::Tensor::zeros({bytes}, core::Device::GPU, core::DataType::UInt8);
                        if (!maps[n].empty())
                            state.uploads.emplace_back().enqueue_in_batch(map, std::as_bytes(maps[n]));
                    }
                    tree = {&pool.regions[7], &pool.regions[8], &state.page_maps[0], &state.page_maps[1], &pool.regions[6], &state.page_maps[2]};
                } else {
                    const auto& t = gpu_tree->tensors;
                    tree = {&t[0], &t[1], &t[2], &t[3], &t[4], &t[5]};
                }
                lod_parameters.pixel_scale_limit = p.pixel_scale_limit;
                lod_parameters.object_scale = p.object_scale;
                lod_parameters.behind_camera_penalty = p.behind_camera_penalty;
                lod_parameters.cone_foveation = p.cone_foveation;
                lod_parameters.cone_dot0 = std::cos(glm::radians(std::clamp(p.cone_inner_degrees, 0.f, 180.f) * .5f));
                lod_parameters.cone_dot = std::min(lod_parameters.cone_dot0, std::cos(glm::radians(std::clamp(p.cone_outer_degrees, 0.f, 180.f) * .5f)));
                lod_parameters.cone_blend_denominator = lod_parameters.cone_dot0 - lod_parameters.cone_dot;
                lod_parameters.cone_tail_valid = lod_parameters.cone_dot >= 1e-6f ? 1.f : 0.f;
                for (size_t row = 0; row < 3; ++row) {
                    auto& dest = row == 0 ? lod_parameters.view_row0 : row == 1 ? lod_parameters.view_row1
                                                                                : lod_parameters.view_row2;
                    for (size_t col = 0; col < 4; ++col)
                        dest[col] = p.object_to_view[col][row];
                }
                lod_parameters.outside_view_foveation = std::clamp(p.outside_view_foveation, 0.f, 1.f);
                lod_parameters.viewport_half_tan_x = p.viewport_half_tan_x;
                lod_parameters.viewport_half_tan_y = p.viewport_half_tan_y;
                lod_parameters.ortho_half_width = p.ortho_half_width;
                lod_parameters.ortho_half_height = p.ortho_half_height;
                lod_parameters.viewport_foveation = uint32_t(p.viewport_foveation);
                lod_parameters.orthographic = uint32_t(p.orthographic);
                if (auto selected = state.lod->select(tree, lod_parameters); !selected)
                    throw lfs::Exception(selected.error());
                const auto& selector = *state.lod;
                gpu_cut = {&selector.indices(), &selector.logical_indices(), &selector.levels(), &selector.weights(), &selector.counts(),
                           draw_count, request.lod_debug_mode, logical_count};
                f.gpu_capacity = draw_count;
                f.gpu_source_count = source_count;
                f.gpu_tree_signature = gpu_tree->signature;
                f.gpu_chunks = gpu_tree->chunks;
            }
            const auto* lod_cut = gpu_lod ? &gpu_cut : host_cut ? &host_cut_tensors : nullptr;
            // Projection flags of the native derivation: Spark opacity, portal edge math.
            const bool spark_opacity = projection.display[2] == 1.f;
            const bool portal_edges = projection.rasterization[3] == 1.f && projection.display[2] == 0.f;
            const bool tight = !transparent && !needs_overlay && !request.gut && !spark_opacity && !portal_math;
            const auto mark = [&f](const char* stage) {
                if (!f.profile)
                    return;
                if (!f.profile->mark(f.profile_stages.size() + 1, nullptr))
                    throw std::runtime_error(std::format("GPU timestamp failed (stage={})", stage));
                f.profile_stages.emplace_back(stage);
            };
            f.profile = i.profiling_enabled ? std::make_unique<core::GpuElapsed>(*backend, 32) : nullptr;
            f.profile_stages.clear();
            if (f.profile && !f.profile->mark(0, nullptr))
                throw std::runtime_error("GPU timestamp failed (stage=begin)");
            state.raster->set_stage_marker(f.profile ? std::function<void(const char*)>(mark) : nullptr);
            if (auto projected = i.projector->project(sources, projection, degree, request.gut ? rendering::SplatPrimitive::Gut : rendering::SplatPrimitive::Gaussian,
                                                      tight, i.projected, request.gut ? &i.gut_geometry : nullptr, needs_overlay ? &overlay_inputs : nullptr, lod_cut);
                !projected)
                throw lfs::Exception(projected.error());
            mark("projection");
            rendering::SplatRasterParameters raster;
            raster.count = draw_count;
            raster.width = uint32_t(f.size.x);
            raster.height = uint32_t(f.size.y);
            raster.columns = (raster.width + 15) / 16;
            raster.tiles = raster.columns * ((raster.height + 15) / 16);
            raster.capacity = f.capacity;
            raster.mode = uint32_t(request.gut ? rendering::SplatRasterMode::Gut : rendering::SplatRasterMode::Gaussian);
            // Rings use the macro reference's half footprint, unless a separate
            // median (depth view) or precise transparent blending is needed.
            // 3DGUT keeps the legacy Vulkan chain's color update and starts with
            // 64-thread groups; the rasterizer narrows dense frames itself.
            const bool gut = request.gut;
            const bool macro_half = request.overlay.markers.show_rings && !transparent && !request.depth_view && !gut && !spark_opacity;
            const bool precise_transparent = transparent && !gut && !spark_opacity;
            raster.flags = (needs_overlay ? 1u : 0u) | (expected_depth ? 2u : 0u) | (portal_edges ? 4u : 0u) | (lod_cut ? 8u : 0u) |
                           (spark_opacity ? 16u : 0u) | (gut && !spark_opacity ? 32u : 0u) | (macro_half ? 64u : 0u) |
                           (gut || spark_opacity ? 0u : 128u) | (request.depth_view ? 2048u : 0u) | (transparent ? 0u : 4096u) |
                           (precise_transparent ? 16384u : 0u) | (camera_model == CameraModel::Equirectangular && !gut ? 8192u : 0u);
            const auto mask_extent = [](const core::Tensor* mask) { return uint32_t(std::min<size_t>(mask->bytes(), std::numeric_limits<uint32_t>::max())); };
            raster.mask_limits = {selection_enabled ? mask_extent(selection) : 0u, preview_enabled ? mask_extent(preview) : 0u, 0, 0};
            const auto background = request.frame_view.background_color;
            raster.background = {background.x, background.y, background.z, transparent ? 0.f : 1.f};
            raster.render_origin = {render_origin[0], render_origin[1], 0, 0};
            raster.intrinsics = projection.intrinsics;
            raster.clip = {projection.clip_scale[0], expected_depth ? projection.rasterization[2] : projection.clip_scale[1], projection.clip_scale[2], projection.clip_scale[3]};
            raster.camera = projection.extent;
            raster.panorama = projection.panorama;
            const rendering::SplatRasterOverlay raster_overlay{&i.projector->overlay_parameters(), &i.projector->overlay_flags(),
                                                               selection_enabled ? selection : nullptr, preview_enabled ? preview : nullptr,
                                                               needs_overlay ? host_bytes(f.selection_colors, sizeof(request.overlay.selection_colors)) : std::span<const std::byte>{}};
            const rendering::SplatRasterLogical raster_logical{lod_cut && lod_cut->logical_indices ? lod_cut->logical_indices : lod_cut ? lod_cut->indices : nullptr,
                                                               logical_count};
            if (auto rasterized = state.raster->rasterize(i.projected, gut ? &i.gut_geometry : nullptr, draw_count, rendering::SplatRasterMode(raster.mode), raster,
                                                          needs_overlay ? &raster_overlay : nullptr, lod_cut ? &raster_logical : nullptr);
                !rasterized)
                throw lfs::Exception(rasterized.error());
            rendering::SplatPresentParameters presented;
            presented.exposure = request.color_exposure;
            presented.tone = portal ? 0u : uint32_t(request.color_tonemapping);
            presented.transparent = uint32_t(transparent);
            presented.depth_min = request.depth_view_min;
            presented.depth_max = request.depth_view_max;
            presented.depth_view = uint32_t(request.depth_view);
            presented.depth_mode = uint32_t(request.depth_visualization_mode);
            presented.background = {background.x, background.y, background.z, 1};
            presented.capture = {uint32_t(expected_depth), 0, 0, 0};
            if (auto shown = state.raster->present(presented); !shown)
                throw lfs::Exception(shown.error());
            state.raster->set_stage_marker(nullptr);
            // LOD statistics and the RAD pager read the cut's counts and chunk
            // touches after completion.
            if (gpu_lod) {
                allocate(f.lod_touches, size_t(gpu_tree->chunks) * 4);
                allocate(f.lod_counts, 32);
            } else {
                f.lod_counts = nil;
            }
            if (i.serial == std::numeric_limits<uint64_t>::max())
                throw std::runtime_error(std::format("Metal viewport timeline exhausted (serial={})", i.serial));
            const uint64_t serial = i.serial + 1;
            const auto event = i.event;
            const auto& rasterizer = *state.raster;
            const std::array<const core::Tensor*, 5> outputs{&rasterizer.rgba(), &rasterizer.linear_depth(), &rasterizer.status(),
                                                             gpu_lod ? &state.lod->touches() : nullptr, gpu_lod ? &state.lod->counts() : nullptr};
            // One native command publishes the tensor outputs into the viewport
            // images and signals presentation.
            f.command = i.reader.submit(outputs, [&](id<MTLCommandBuffer> command, std::span<const core::MetalTensorView> views) {
                // A blit on the readback queue may still sample a recycled slot.
                // GPU ordering protects it without waiting on the host each frame.
                if (i.next_readback)
                    [command encodeWaitForEvent:i.readback_event value:i.next_readback];
                const NSUInteger width = raster.width, height = raster.height;
                auto blit = [command blitCommandEncoder];
                [blit copyFromBuffer:views[0].buffer sourceOffset:views[0].offset sourceBytesPerRow:width * 4 sourceBytesPerImage:width * height * 4
                          sourceSize:MTLSizeMake(width, height, 1) toTexture:f.color.texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
                [blit copyFromBuffer:views[1].buffer sourceOffset:views[1].offset sourceBytesPerRow:width * 4 sourceBytesPerImage:width * height * 4
                          sourceSize:MTLSizeMake(width, height, 1) toTexture:f.depth.texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
                [blit copyFromBuffer:views[2].buffer sourceOffset:views[2].offset toBuffer:f.raster_status destinationOffset:0 size:sizeof(RasterStatus)];
                if (gpu_lod) {
                    [blit copyFromBuffer:views[3].buffer sourceOffset:views[3].offset toBuffer:f.lod_touches destinationOffset:0 size:size_t(gpu_tree->chunks) * 4];
                    [blit copyFromBuffer:views[4].buffer sourceOffset:views[4].offset toBuffer:f.lod_counts destinationOffset:0 size:32];
                }
                [blit endEncoding];
                const auto completion_value = serial;
                [command encodeSignalEvent:event value:completion_value];
                [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    // Failed producers must release presentation waits too;
                    // their typed command error is inspected before resource reuse.
                    if (completed.status == MTLCommandBufferStatusError && event.signaledValue < completion_value)
                        event.signaledValue = completion_value;
                }];
                // Record overflow demand for the retry, which redraws at a capacity that fits.
                const auto completion_retry = i.retry_callback;
                const auto overflow_required = state.overflow_required;
                const id<MTLBuffer> status_buffer = f.raster_status;
                [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    const auto* status = static_cast<const RasterStatus*>(status_buffer.contents);
                    if (completed.status != MTLCommandBufferStatusError && status->error) {
                        uint64_t seen = overflow_required->load(std::memory_order_relaxed);
                        while (status->required_instances > seen &&
                               !overflow_required->compare_exchange_weak(seen, status->required_instances, std::memory_order_acq_rel)) {
                        }
                    }
                    if (completion_retry && (completed.status == MTLCommandBufferStatusError || status->error))
                        completion_retry();
                }];
            });
            const bool refine = pager && (pager->pending() || !pager->rootReady());
            if (pager && gpu_lod)
                pager->noteRendererCompletion(serial);
            f.producer_value = serial;
            i.serial = serial;
            f.consumer_serial = i.consumerSerial();
            state.latest = &f;
#ifndef LFS_GRAPHICS_VULKAN
            return SceneRenderer::RenderResult{.generation = f.generation, .depth_generation = f.generation, .size = f.size, .alloc_size = f.size, .completion_value = serial, .lod_streaming_active = refine, .viewer_backend = rendering::ViewerBackend::Metal};
#else
            return SceneRenderer::RenderResult{.image = sceneImageHandle(f.color.image), .image_view = sceneImageViewHandle(f.color.view), .image_layout = sceneImageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL), .generation = f.generation, .depth_image = sceneImageHandle(f.depth.image), .depth_image_view = sceneImageViewHandle(f.depth.view), .depth_image_layout = sceneImageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL), .depth_generation = f.generation, .size = f.size, .alloc_size = f.size, .flip_y = false, .completion_semaphore = sceneTimelineHandle(i.completion), .completion_value = serial, .lod_streaming_active = refine, .viewer_backend = rendering::ViewerBackend::Metal};
#endif
        } catch (const std::exception& e) { return nativeError(e); }
    }
    glm::ivec2 MetalViewportRenderer::size(Slot slot) const {
        std::lock_guard lock(impl_->readback_mutex);
        auto f = impl_->latestFrame(slot);
        return f ? f->size : glm::ivec2{};
    }
#ifndef LFS_GRAPHICS_VULKAN
    lfs::Status MetalViewportRenderer::copyOutputs(Slot slot, core::Tensor& color, core::Tensor* depth) const {
        try {
            auto& i = *impl_;
            std::lock_guard lock(i.readback_mutex);
            auto f = i.latestFrame(slot);
            if (!f)
                throw std::runtime_error(std::format("Metal output copy slot is empty (target={}, targets={})", slot.value, i.targets.size()));
            const auto width = size_t(f->size.x), height = size_t(f->size.y);
            if (!color.is_valid() || !nativeStorage(color) || !color.is_contiguous() || color.dtype() != core::DataType::UInt8 ||
                color.ndim() != 3 || color.size(0) != height || color.size(1) != width || color.size(2) != 4 ||
                (depth && (!depth->is_valid() || !nativeStorage(*depth) || !depth->is_contiguous() || depth->dtype() != core::DataType::Float32 ||
                           depth->ndim() != 2 || depth->size(0) != height || depth->size(1) != width)))
                throw std::invalid_argument(std::format("Metal output copy needs UInt8 [{},{},4] color and Float32 [{},{}] depth (color={}, depth={})",
                                                        height, width, height, width, color.shape().str(), depth ? depth->shape().str() : "none"));
            std::array<core::Tensor*, 2> outputs{&color, depth};
            const std::span<core::Tensor* const> written(outputs.data(), depth ? 2 : 1);
            // Same queue as the render: Metal orders this blit after it, and a
            // later render into the recycled texture after this blit.
            auto command = i.reader.submitWrites({}, written, [&](id<MTLCommandBuffer> command, std::span<const core::MetalTensorView>, std::span<const core::MetalTensorView> out) {
                auto blit = [command blitCommandEncoder];
                [blit copyFromTexture:f->color.texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
                             toBuffer:out[0].buffer destinationOffset:out[0].offset destinationBytesPerRow:width * 4 destinationBytesPerImage:width * height * 4];
                if (depth)
                    [blit copyFromTexture:f->depth.texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, 1)
                                 toBuffer:out[1].buffer destinationOffset:out[1].offset destinationBytesPerRow:width * 4 destinationBytesPerImage:width * height * 4];
                [blit endEncoding];
            });
            if (!command)
                throw std::runtime_error(std::format("Metal output copy was not submitted (target={}, extent={}x{})", slot.value, width, height));
            return {};
        } catch (const std::exception& error) { return lfs::Status::failure(nativeError(error)); }
    }
#endif
    lfs::Result<uint64_t> MetalViewportRenderer::submitReadback(
        Slot slot, core::Tensor& destination, int x, int y, bool depth) const {
        try {
            auto& i = *impl_;
            std::lock_guard lock(i.readback_mutex);
            auto f = i.latestFrame(slot);
            if (!f)
                throw std::runtime_error(std::format("Metal readback slot is empty (target={}, targets={})", slot.value, i.targets.size()));
            // Abandoned commands retain staging until completion but no host pointer.
            std::erase_if(i.readbacks, [](const auto& entry) { return !entry.second.destination &&
                                                                      (entry.second.command.status == MTLCommandBufferStatusCompleted || entry.second.command.status == MTLCommandBufferStatusError); });
            if (i.readbacks.size() >= BasicReadbackTicketRing<void*>::kRingSize)
                throw std::runtime_error(std::format("Metal readback ring is full; retire or abandon a ticket (active_tickets={}, capacity={}, target={})", i.readbacks.size(), BasicReadbackTicketRing<void*>::kRingSize, slot.value));
            if (!destination.is_valid() || destination.device() != core::Device::CPU || !destination.is_contiguous() ||
                x < 0 || y < 0 || (depth ? (destination.dtype() != core::DataType::Float32 || destination.ndim() != 2 || destination.size(0) != size_t(f->size.y) || destination.size(1) != size_t(f->size.x)) : (destination.ndim() != 3 || destination.size(2) < 3 || destination.size(2) > 4 || size_t(x) + f->size.x > destination.size(1) || size_t(y) + f->size.y > destination.size(0) || (destination.dtype() != core::DataType::Float32 && destination.dtype() != core::DataType::UInt8))))
                throw std::invalid_argument(std::format("Invalid Metal readback destination (shape={}, dtype={}, device={}, contiguous={}, origin=[{},{}], extent={}x{}, depth={})", destination.shape().str(), int(destination.dtype()), int(destination.device()), destination.is_contiguous(), x, y, f->size.x, f->size.y, depth));
            if (!i.readback_queue || !i.readback_event || i.next_readback == ((uint64_t{1} << 63) - 1))
                throw std::runtime_error(std::format("Metal readback timeline unavailable or exhausted (queue_present={}, event_present={}, serial={})", i.readback_queue != nil, i.readback_event != nil, i.next_readback));
            const uint64_t serial = i.next_readback + 1;
            const auto command = [i.readback_queue commandBuffer];
            const size_t row = size_t(f->size.x) * 4;
            const auto buffer = [i.reader.device() newBufferWithLength:row * f->size.y options:MTLResourceStorageModeShared];
            if (!command || !buffer)
                throw lfs::Exception(nativeError(std::format("Metal readback allocation failed (target={}, bytes={}, command_present={})", slot.value, row * f->size.y, command != nil), lfs::ErrorCode::ResourceExhausted));
            [command encodeWaitForEvent:i.event value:f->producer_value];
            auto blit = [command blitCommandEncoder];
            if (!blit)
                throw std::runtime_error(std::format("Metal readback encoder unavailable (command_status={}, target={}, bytes={})", long(command.status), slot.value, row * f->size.y));
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
                                                       destination.size(1), depth ? 1 : destination.size(2), x, y, depth, destination.dtype() == core::DataType::Float32, slot});
            i.next_readback = serial;
            [command commit];
            return ticket;
        } catch (const std::exception& e) { return nativeError(e); }
    }
    lfs::Result<SceneRenderer::ReadbackTicketStatus> MetalViewportRenderer::pollReadback(uint64_t ticket, bool wait) const {
        auto& i = *impl_;
        std::lock_guard lock(i.readback_mutex);
        auto it = i.readbacks.find(ticket);
        if (it == i.readbacks.end())
            return nativeError(std::format("Unknown Metal readback ticket (ticket={}, active_tickets={})", ticket, i.readbacks.size()), lfs::ErrorCode::NotFound);
        auto& r = it->second;
        if (r.target_released) {
            if (wait)
                [r.command waitUntilCompleted];
            if (r.command.status == MTLCommandBufferStatusCompleted || r.command.status == MTLCommandBufferStatusError)
                i.readbacks.erase(it);
            return SceneRenderer::ReadbackTicketStatus::Failed;
        }
        if (wait)
            [r.command waitUntilCompleted];
        if (r.command.status == MTLCommandBufferStatusError) {
            const std::string error = r.command.error.localizedDescription.UTF8String ?: "Metal readback command failed";
            i.readbacks.erase(it);
            return nativeError(error);
        }
        if (r.command.status != MTLCommandBufferStatusCompleted)
            return SceneRenderer::ReadbackTicketStatus::NotReady;
        if (r.producer.status == MTLCommandBufferStatusError) {
            const auto error = std::format("Metal readback producer failed (ticket={}, status={}, error={})", ticket, long(r.producer.status), r.producer.error.localizedDescription.UTF8String ?: "none");
            i.readbacks.erase(it);
            return nativeError(error);
        }
        if (!r.destination) {
            i.readbacks.erase(it);
            return nativeError(std::format("Metal readback ticket abandoned (ticket={})", ticket), lfs::ErrorCode::Cancelled);
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
        return SceneRenderer::ReadbackTicketStatus::Ready;
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
            i.retireTarget(slot, true);
            // Native producers and the graphics consumer retire independently.
            // Never wait for another view or destroy an active-frame texture.
            if (i.context)
                i.drainRetiredTargets();
            return {};
        } catch (const std::exception& error) { return lfs::Status::failure(nativeError(error)); }
    }
    lfs::Status MetalViewportRenderer::releaseAll() {
        try {
            auto& i = *impl_;
            std::lock_guard lock(i.readback_mutex);
            while (!i.targets.empty())
                i.retireTarget(i.targets.begin()->first, false);
            if (i.context)
                i.drainRetiredTargets();
            return {};
        } catch (const std::exception& error) { return lfs::Status::failure(nativeError(error)); }
    }
    SceneRenderer::GpuLodSelectionStatus MetalViewportRenderer::gpuLodSelectionStatus(Slot slot) const {
        auto& i = *impl_;
        std::lock_guard lock(i.readback_mutex);
        SceneRenderer::GpuLodSelectionStatus status;
        const auto latest = i.latestFrame(slot);
        if (!latest || !latest->gpu_lod_active)
            return status;
        status.active = true;
        status.capacity = latest->gpu_capacity;
        const Frame* completed = nullptr;
        for (const auto& candidate : i.target(slot).frames) {
            if (candidate && candidate->lodResultReady() && candidate->gpu_tree_signature == latest->gpu_tree_signature &&
                (!completed || candidate->producer_value > completed->producer_value))
                completed = candidate.get();
        }
        if (!completed)
            return status;
        const auto cut = completed->lodStatus();
        status.capacity = completed->gpu_capacity;
        status.selected = std::min(cut.selected, completed->gpu_capacity);
        status.overflow = cut.overflow;
        status.pixel_scale_feedback = cut.threshold_multiplier;
        const auto* touches = static_cast<const uint32_t*>(completed->lod_touches.contents);
        status.resident_chunks = status.chunk_count = status.pool_pages = completed->gpu_chunks;
        const auto& pager = i.target(slot).pager;
        if (pager && completed->rad_signature == pager->signature()) {
            status.resident_chunks = pager->cache().snapshot().resident_chunks;
            status.pool_pages = pager->cache().snapshot().physical_pages;
            status.streaming_jobs = pager->cache().outstandingWorkCount() + size_t(pager->pending());
            status.deferred_requests = pager->cache().deferredRequestCount();
            status.admission_frozen = pager->frozen();
        }
        for (size_t n = 0; n < status.chunk_count; ++n) {
            status.touched_chunks += touches[n] != 0;
            status.miss_chunks += touches[n] != 0 && touches[n] != 0xffffffffu;
        }
        return status;
    }
    void MetalViewportRenderer::setProfilingEnabled(bool enabled) {
        impl_->profiling_enabled = enabled;
    }
    lfs::Result<MetalViewportRenderer::FrameDiagnostics> MetalViewportRenderer::frameDiagnostics(Slot slot) const {
        try {
            const auto complete = outputComplete(slot);
            if (!complete)
                throw lfs::Exception(complete.error());
            if (!*complete)
                throw lfs::Exception(nativeError(std::format("Native diagnostics require a complete frame (target={}, complete={})", slot.value, *complete), lfs::ErrorCode::FailedPrecondition));
            const auto* frame = impl_->latestFrame(slot);
            FrameDiagnostics result;
            result.input_splats = frame->count;
            result.reserved_instances = frame->capacity;
            if (!frame->points) {
                const auto status = frame->rasterStatus();
                result.required_instances = status.required_instances;
                result.blend_threads = status.blend_threads;
                result.maximum_tile_instances = status.maximum_tile_instances;
            }
            result.gpu_command_ms = (frame->command.GPUEndTime - frame->command.GPUStartTime) * 1000.;
            if (!std::isfinite(result.gpu_command_ms) || result.gpu_command_ms < 0)
                throw std::runtime_error(std::format("Invalid native command GPU interval (start={}, end={}, duration_ms={})", frame->command.GPUStartTime, frame->command.GPUEndTime, result.gpu_command_ms));
            if (frame->profile) {
                // Stage buckets: projection, instances, sort, blend, present.
                const auto bucket = [](const std::string& stage) -> size_t {
                    if (stage == "projection")
                        return 0;
                    if (stage == "source" || stage == "counts" || stage == "instances")
                        return 1;
                    if (stage == "sort" || stage == "ranges")
                        return 2;
                    return stage == "present" ? 4 : 3;
                };
                result.counter_timestamps_available = true;
                for (size_t n = 0; n < frame->profile_stages.size(); ++n) {
                    const auto ms = frame->profile->milliseconds(n, n + 1);
                    if (!ms) {
                        result.counter_timestamps_available = false;
                        break;
                    }
                    result.gpu_stage_ms[bucket(frame->profile_stages[n])] += *ms;
                }
            }
            return result;
        } catch (const std::exception& error) {
            return nativeError(error);
        }
    }
    lfs::Result<bool> MetalViewportRenderer::outputComplete(Slot slot) const {
        try {
            std::lock_guard lock(impl_->readback_mutex);
            auto frame = impl_->latestFrame(slot);
            if (!frame)
                throw lfs::Exception(nativeError(std::format("Metal output slot is empty (target={})", slot.value), lfs::ErrorCode::FailedPrecondition));
            impl_->wait(frame->producer_value);
            [frame->command waitUntilCompleted];
            if (frame->command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error(std::format("Metal output command failed (target={}, command_status={}, error={})", slot.value, long(frame->command.status), frame->command.error.localizedDescription.UTF8String ?: "none"));
            if (frame->rad_bootstrap)
                return false;
            if (frame->lodResultReady() && frame->lodStatus().overflow)
                return false;
            return frame->rasterStatus().error == 0;
        } catch (const std::exception& error) {
            return nativeError(error);
        }
    }
    lfs::Status MetalViewportRenderer::readColor(Slot slot, core::Tensor& destination, int x, int y) const {
        try {
            std::lock_guard lock(impl_->readback_mutex);
            auto f = impl_->latestFrame(slot);
            if (!f)
                throw lfs::Exception(nativeError(std::format("Metal output slot is empty (target={})", slot.value), lfs::ErrorCode::FailedPrecondition));
            if (destination.device() != core::Device::CPU || !destination.is_contiguous() || destination.ndim() != 3 ||
                destination.size(2) < 3 || destination.size(2) > 4 || x < 0 || y < 0 ||
                size_t(x) + f->size.x > destination.size(1) || size_t(y) + f->size.y > destination.size(0) ||
                (destination.dtype() != core::DataType::Float32 && destination.dtype() != core::DataType::UInt8))
                throw std::invalid_argument(std::format("Invalid Metal HWC readback destination (shape={}, dtype={}, device={}, contiguous={}, origin=[{},{}], extent={}x{})", destination.shape().str(), int(destination.dtype()), int(destination.device()), destination.is_contiguous(), x, y, f->size.x, f->size.y));
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
    lfs::Result<float> MetalViewportRenderer::readDepth(const SceneRenderer::DepthSampleRequest& r) const {
        try {
            std::lock_guard lock(impl_->readback_mutex);
            auto f = impl_->latestFrame(r.target);
            if (!f)
                throw lfs::Exception(nativeError(std::format("Metal depth slot is empty (target={}, pixel=[{},{}])", r.target.value, r.pixel.x, r.pixel.y), lfs::ErrorCode::FailedPrecondition));
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
