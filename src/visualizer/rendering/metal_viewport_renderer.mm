/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_viewport_renderer.hpp"
#include "core/memory_pressure.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_metal_reader.hpp"
#include "frame_budget.hpp"
#include "gpu_profile.hpp"
#include "lod_selector.hpp"
#include "metal_present_source.hpp"
#include "metal_rad_pager.hpp"
#include "readback_ticket_ring.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "scene_overlay_params.hpp"
#include "selection_query.hpp"
#include "splat_preprocessor.hpp"
#include "tile_rasterizer.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vulkan/vulkan_metal.h>

namespace lfs::vis {
    namespace {
        using namespace rendering::metal;
        using Slot = RenderTargetId;
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
                    throw lfs::Exception(nativeError(std::format("Metal viewport texture allocation failed (extent={}x{}, format={}, allocated={}, recommended={})", w, h, uint64_t(native_format), metal.currentAllocatedSize, metal.recommendedMaxWorkingSetSize), lfs::ErrorCode::ResourceExhausted));
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
            std::unique_ptr<RasterFrame> raster;
            std::unique_ptr<GpuProfile> gpu_profile;
            id<MTLBuffer> projected, gut_geometry, objects, overlay_parameters, overlay_flags, overlay_nodes, selection_colors;
            std::array<id<MTLBuffer>, 4> lod_buffers;
            std::array<id<MTLBuffer>, 3> page_maps;
            bool rad_bootstrap = false;
            uint64_t rad_signature = 0;
            id<MTLCommandBuffer> command;
            id<MTLTexture> point_depth;
            std::unique_ptr<LodCutFrame> gpu_lod;
            bool gpu_lod_active = false;
            uint64_t gpu_tree_signature = 0;
            uint32_t gpu_capacity = 0, gpu_source_count = 0, gpu_chunks = 0;
            bool points = false;
            Image color, depth;
        };
    } // namespace
    struct MetalViewportRenderer::Impl {
        VulkanContext* context = nullptr;
        struct NativeLodTree {
            LodTreeBuffers buffers;
            uint64_t signature = 0, last_used = 0;
            uint32_t nodes = 0, chunks = 0, roots = 0;
        };
        std::map<const core::SplatLodTree*, NativeLodTree> lod_trees;
        std::unique_ptr<LodSelector> lod_selector;
        std::unique_ptr<SelectionQuery> selection_query;
        // Selection requests carry their own camera, not a render-target ID.
        // A dedicated pager avoids borrowing another view's RAD/model workspace.
        std::unique_ptr<MetalRadPager> selection_pager;
        MetalRadPager::Settings rad_settings;
        core::MetalTensorReader reader;
        SplatPreprocessor preprocessor{reader.device()};
        TileRasterizer rasterizer{reader.device()};
        bool profiling_enabled = false;
        std::function<void()> retry_callback;
        id<MTLComputePipelineState> present;
        id<MTLRenderPipelineState> point_pipeline;
        id<MTLDepthStencilState> point_depth_state;
        id<MTLSharedEvent> event;
        VkSemaphore completion = VK_NULL_HANDLE;
        uint64_t serial = 0;
        struct TargetState {
            std::array<std::unique_ptr<Frame>, 3> frames;
            Frame* latest = nullptr;
            size_t next = 0;
            uint32_t needed_capacity = 0;
            std::unique_ptr<MetalRadPager> pager;
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
            const auto consumer = context->lastFrameSubmitSerial();
            for (auto& [id, state] : targets)
                if (state.latest)
                    state.latest->consumer_serial = std::max(state.latest->consumer_serial, consumer);
        }
        void drainRetiredTargets() {
            const auto consumer = context->retiredFrameSubmitSerial();
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
                uint64_t consumer = context ? context->lastFrameSubmitSerial() + (context->hasActiveFrame() ? 1 : 0) : 0;
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
        NativeLodTree& prepareLodTree(const core::SplatData& model) {
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
            NativeLodTree metadata;
            metadata.signature = signature;
            metadata.nodes = uint32_t(n);
            metadata.chunks = uint32_t(chunks);
            metadata.roots = roots;
            metadata.last_used = serial;
            const auto upload = [&](const void* data, size_t bytes) {
                const auto device = reader.device();
                if (!frameFitsWorkingSet(device.currentAllocatedSize, bytes, device.recommendedMaxWorkingSetSize))
                    throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = bytes, .label = "viewer.lod.metadata", .operation = "viewer.buffer.allocate"});
                auto buffer = [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
                if (!buffer)
                    throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = bytes, .label = "viewer.lod.metadata", .operation = "viewer.buffer.allocate"});
                return BufferSlice{buffer};
            };
            metadata.buffers = {upload(bounds.data(), bounds.size() * 4), upload(links.data(), links.size() * 4), upload(maps.data(), maps.size() * 4),
                                upload(age.data(), age.size() * 4), upload(frames.data(), frames.size() * 16), upload(maps.data(), maps.size() * 4)};
            // Views can use independent models. Bound the shared metadata
            // cache; command buffers retain evicted resources until GPU completion.
            if (!lod_trees.contains(&tree) && lod_trees.size() >= 4) {
                auto oldest = std::min_element(lod_trees.begin(), lod_trees.end(), [](const auto& a, const auto& b) { return a.second.last_used < b.second.last_used; });
                lod_trees.erase(oldest);
            }
            return lod_trees.insert_or_assign(&tree, std::move(metadata)).first->second;
        }
        Frame& acquire(Slot output, const rendering::ViewportRenderRequest& request, uint32_t count, bool points = false) {
            auto& state = target(output);
            auto& frame = state.frames[state.next++ % 3];
            uint32_t capacity = std::max(state.needed_capacity, static_cast<uint32_t>(std::min<uint64_t>(uint64_t(count) * 16 + 4096, 16u * 1024u * 1024u)));
            // A failed encode may leave a reservation without a submitted producer.
            // Such a frame has no readable GPU status and must be recreated.
            if (frame && !frame->command) {
                if (state.latest == frame.get())
                    state.latest = nullptr;
                frame.reset();
            }
            if (frame) {
                wait(frame->producer_value);
                if (frame->command.status == MTLCommandBufferStatusError)
                    throw std::runtime_error(frame->command.error.localizedDescription.UTF8String ?: "Metal command failed");
                if (!context->waitForRetiredFrameSubmitSerial(frame->consumer_serial))
                    throw std::runtime_error(context->lastError());
                if ((frame->raster && frame->raster->busy()) || (frame->gpu_lod && frame->gpu_lod->busy()))
                    [frame->command waitUntilCompleted];
                const auto status = frame->raster ? frame->raster->status() : RasterStatus{};
                if (status.error != RasterError::None) {
                    if (status.required_instances > std::numeric_limits<uint32_t>::max())
                        throw std::runtime_error(std::format("Metal viewport instance count exceeds 32-bit indexing (required={}, capacity={})", status.required_instances, capacity));
                    capacity = std::max(capacity, static_cast<uint32_t>(status.required_instances));
                }
                if (frame->points == points && frame->size == request.frame_view.size && frame->count >= count && frame->capacity >= capacity)
                    return *frame;
                if (state.latest == frame.get())
                    state.latest = nullptr;
                frame.reset();
            }
            // Account for all Metal allocations on the shared device, including
            // resident tensors and Vulkan presentation. Fail before a large growth
            // can exhaust unified memory; retain the last completed output.
            const auto device = reader.device();
            const auto reservation = frameReservationBytes(request.frame_view.size.x,
                                                           request.frame_view.size.y, count, capacity, points);
            if (!frameFitsWorkingSet(device.currentAllocatedSize, reservation, device.recommendedMaxWorkingSetSize))
                throw lfs::Exception(nativeError(std::format("Metal viewport reservation exceeds the recommended GPU working set (extent={}x{}, count={}, capacity={}, reservation={}, allocated={}, recommended={})", request.frame_view.size.x, request.frame_view.size.y, count, capacity, reservation, device.currentAllocatedSize, device.recommendedMaxWorkingSetSize), lfs::ErrorCode::ResourceExhausted));
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
                    throw lfs::Exception(nativeError(std::format("Metal projected buffer allocation failed (count={}, bytes={})", count, size_t(count) * sizeof(ProjectedSplat)), lfs::ErrorCode::ResourceExhausted));
            } else {
                auto depth_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:f.size.x height:f.size.y mipmapped:NO];
                depth_descriptor.storageMode = MTLStorageModePrivate;
                depth_descriptor.usage = MTLTextureUsageRenderTarget;
                f.point_depth = [device newTextureWithDescriptor:depth_descriptor];
                if (!f.point_depth)
                    throw lfs::Exception(nativeError(std::format("Metal point depth allocation failed (extent={}x{})", f.size.x, f.size.y), lfs::ErrorCode::ResourceExhausted));
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
    lfs::Result<core::Tensor> MetalViewportRenderer::buildSelectionMask(VulkanContext& context, const core::SplatData& model,
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
            // Editor masks stay on the model's tensor backend even when their
            // producer is the native Metal renderer.
            const auto backend = model.means_raw().device() == core::Device::GPU
                                     ? core::gpu_backend_of(model.means_raw()).value_or(core::default_gpu_backend())
                                     : core::default_gpu_backend();
            const core::GpuBackendScope scope(backend);
            if (!frameFitsWorkingSet(i.reader.device().currentAllocatedSize, n, i.reader.device().recommendedMaxWorkingSetSize))
                throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = n, .label = "viewer.selection.result", .operation = "viewer.buffer.allocate"});
            auto output = core::Tensor::empty({n}, core::Device::GPU, core::DataType::Bool);
            SelectionParameters parameters;
            auto view = glm::mat4(glm::transpose(rendering::dataCameraToWorldFromVisualizerRotation(request.frame_view.rotation)));
            view[3] = glm::vec4(-glm::mat3(view) * request.frame_view.translation, 1);
            parameters.world_to_camera = matrix(view);
            const auto intrinsics = request.frame_view.getCameraIntrinsics();
            parameters.intrinsics = {intrinsics.focal_x, intrinsics.focal_y, intrinsics.center_x, intrinsics.center_y};
            parameters.image = {uint32_t(request.frame_view.size.x), uint32_t(request.frame_view.size.y), uint32_t(request.equirectangular ? CameraModel::Equirectangular : request.frame_view.orthographic ? CameraModel::Orthographic
                                                                                                                                                                                                            : CameraModel::Perspective),
                                uint32_t(request.gut)};
            parameters.source = {uint32_t(n), uint32_t(request.shape), uint32_t(request.primitives.size()), uint32_t(request.polygon_vertices.size())};
            parameters.payload = {uint32_t(model.non_sh_attrs_f16()), uint32_t(request.mip_filter), 0, 0};
            parameters.ring.x = request.ring_width;
            if (polygon) {
                glm::vec2 lo(std::numeric_limits<float>::max()), hi(std::numeric_limits<float>::lowest());
                for (const auto vertex : request.polygon_vertices) {
                    if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y))
                        throw std::invalid_argument(std::format("Invalid native selection polygon vertex (x={}, y={})", vertex.x, vertex.y));
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
            const auto upload = [&](const void* data, size_t bytes) -> BufferSlice {
                if (!bytes)
                    return {};
                if (!frameFitsWorkingSet(i.reader.device().currentAllocatedSize, bytes, i.reader.device().recommendedMaxWorkingSetSize))
                    throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = bytes, .label = "viewer.selection.upload", .operation = "viewer.buffer.allocate"});
                auto buffer = [i.reader.device() newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
                if (!buffer)
                    throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = bytes, .label = "viewer.selection.upload", .operation = "viewer.buffer.allocate"});
                return {buffer};
            };
            SelectionBuffers buffers;
            if (transforms > std::numeric_limits<int32_t>::max() || request.scene.node_visibility_mask.size() > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument(std::format("Native selection scene exceeds transform indexing limits (transforms={}, indexed={})", transforms, indexed));
            if (transforms) {
                for (const auto& transform : *request.scene.model_transforms)
                    for (size_t col = 0; col < 4; ++col)
                        for (size_t row = 0; row < 4; ++row)
                            if (!std::isfinite(transform[col][row]))
                                throw std::invalid_argument(std::format("Invalid native selection object matrix (column={}, row={}, value={})", col, row, transform[col][row]));
                buffers.transforms = upload(request.scene.model_transforms->data(), transforms * sizeof(glm::mat4));
            }
            std::vector<uint8_t> visibility(request.scene.node_visibility_mask.begin(), request.scene.node_visibility_mask.end());
            buffers.visibility = upload(visibility.data(), visibility.size());
            parameters.scene = {uint32_t(transforms), uint32_t(indexed), uint32_t(visibility.size()), 0};
            if (!polygon) {
                for (const auto primitive : request.primitives)
                    for (size_t axis = 0; axis < 4; ++axis)
                        if (!std::isfinite(primitive[axis]))
                            throw std::invalid_argument(std::format("Invalid native selection primitive (axis={}, value={})", axis, primitive[axis]));
                buffers.primitives = upload(request.primitives.data(), request.primitives.size() * sizeof(glm::vec4));
            } else {
                buffers.polygon_vertices = upload(request.polygon_vertices.data(), request.polygon_vertices.size() * sizeof(glm::vec2));
                const size_t bytes = size_t(parameters.aabb.z) * parameters.aabb.w;
                if (bytes > std::numeric_limits<uint32_t>::max())
                    throw std::invalid_argument(std::format("Native selection polygon exceeds shader indexing limits (bytes={}, width={}, height={})", bytes, uint32_t(parameters.aabb.z), uint32_t(parameters.aabb.w)));
                if (!frameFitsWorkingSet(i.reader.device().currentAllocatedSize, bytes, i.reader.device().recommendedMaxWorkingSetSize))
                    throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = bytes, .label = "viewer.selection.polygon", .operation = "viewer.buffer.allocate"});
                auto mask = [i.reader.device() newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
                if (!mask)
                    throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = bytes, .label = "viewer.selection.polygon", .operation = "viewer.buffer.allocate"});
                buffers.polygon_mask = {mask};
            }
            if (ring) {
                auto pick = [i.reader.device() newBufferWithLength:8 options:MTLResourceStorageModeShared];
                if (!pick)
                    throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = 8, .label = "viewer.selection.pick", .operation = "viewer.buffer.allocate"});
                buffers.ring_pick = {pick};
            }
            const bool geometry = request.gut || ring;
            std::array<const core::Tensor*, 6> tensors{&model.means_raw(), geometry ? &model.scaling_raw() : nullptr,
                                                       geometry ? &model.rotation_raw() : nullptr, ring ? &model.opacity_raw() : nullptr,
                                                       &model.deleted(), indexed ? indices : nullptr};
            if (model.means_raw().device() == core::Device::CPU) {
                auto& pager = i.selection_pager;
                if (!pager)
                    pager = std::make_unique<MetalRadPager>(i.reader.device());
                pager->configure(model, context, i.completion, i.rad_settings);
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
            if (!i.selection_query)
                i.selection_query = std::make_unique<SelectionQuery>(i.reader.device());
            const std::array<core::Tensor*, 1> outputs{&output};
            auto command = i.reader.submitWrites(tensors, outputs, [&](id<MTLCommandBuffer> command, std::span<const core::MetalTensorView> input, std::span<const core::MetalTensorView> out) {
                const auto slice = [&](size_t index) { return BufferSlice{input[index].buffer, input[index].offset}; };
                buffers.means = slice(0);
                buffers.log_scales = slice(1);
                buffers.rotations = slice(2);
                buffers.opacity = slice(3);
                buffers.deleted = slice(4);
                buffers.transform_indices = slice(5);
                buffers.output = {out[0].buffer, out[0].offset};
                parameters.scene.w = uint32_t(std::min(n, input[4].bytes));
                i.selection_query->encode(command, buffers, parameters);
            });
            if (ring && request.picked_ring_id_out) {
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error(std::format("Native Metal ring query failed (splats={}, status={}, error={})", n, long(command.status), command.error.localizedDescription.UTF8String ?: "none"));
                const auto pick = reinterpret_cast<const uint32_t*>(static_cast<const char*>(buffers.ring_pick.buffer.contents) + buffers.ring_pick.offset);
                if (pick[0] != 0xffffffffu && pick[1] < n)
                    *request.picked_ring_id_out = pick[1];
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
        VulkanContext& context, const PointSceneRenderer::RenderRequest& r, RenderTargetId output) {
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
            });
            f.producer_value = serial;
            i.serial = serial;
            f.consumer_serial = context.lastFrameSubmitSerial() + (context.hasActiveFrame() ? 1 : 0);
            i.target(slot).latest = &f;
            // Preserve the existing synchronous point-cloud presentation contract.
            // The Gaussian path continues to expose its asynchronous GPU timeline.
            i.wait(serial);
            [f.command waitUntilCompleted];
            if (f.command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error(std::format("Metal point raster failed (command_status={}, error={})", long(f.command.status), f.command.error.localizedDescription.UTF8String ?: "none"));
            return PointSceneRenderer::RenderResult{.image = f.color.image, .image_view = f.color.view, .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .generation = f.generation, .depth_image = f.depth.image, .depth_image_view = f.depth.view, .depth_image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .depth_generation = f.generation, .size = f.size, .flip_y = false, .viewer_backend = rendering::ViewerBackend::Metal};
        } catch (const std::exception& e) { return nativeError(e); }
    }
    bool MetalViewportRenderer::supports(const core::SplatData& model, const rendering::ViewportRenderRequest& r) {
        // Every unsupported contract is routed to the existing renderer; never silently
        // drop filters, display settings or editor overlays from a requested frame.
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
        VulkanContext& context, const core::SplatData& model, const rendering::ViewportRenderRequest& request, Slot slot, bool expected_depth, bool wait_for_pages) {
        try {
            auto& i = *impl_;
            std::lock_guard lock(i.readback_mutex);
            i.initialize(context);
            // Cached outputs may be sampled for many GUI frames without a new
            // raster submission. Stamp their most recent graphics consumer before
            // replacing latest, rather than retiring against the first consumer.
            i.drainRetiredTargets();
            i.stampConsumers();
            (void)i.target(slot);
            if (model.size() > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error(std::format("Metal primitive count exceeds indexing capacity (splats={}, max={})", model.size(), std::numeric_limits<uint32_t>::max()));
            auto previous = i.latestFrame(slot);
            if (previous && previous->points)
                previous = nullptr;
            if (previous && !previous->raster->busy()) {
                const auto status = previous->raster->status();
                if (status.required_instances > std::numeric_limits<uint32_t>::max())
                    throw std::runtime_error(std::format("Metal instance indexing overflow (required_instances={}, max={})", status.required_instances, std::numeric_limits<uint32_t>::max()));
                if (status.error != RasterError::None)
                    i.target(slot).needed_capacity = static_cast<uint32_t>(status.required_instances);
            }
            if (i.target(slot).frames[i.target(slot).next % 3].get() == previous)
                previous = nullptr;
            const bool rad = model.lod_tree && model.lod_tree->rad_source.valid() &&
                             (request.lod_gpu_traversal.enabled || model.lod_tree->total_nodes() > size_t(model.size()));
            MetalRadPager* pager = nullptr;
            Impl::NativeLodTree paged_tree;
            if (rad) {
                auto& owner = i.target(slot).pager;
                if (!owner)
                    owner = std::make_unique<MetalRadPager>(i.reader.device());
                pager = owner.get();
                pager->configure(model, context, i.completion, i.rad_settings);
                const Frame* completed = nullptr;
                for (const auto& candidate : i.target(slot).frames)
                    if (candidate && candidate->gpu_lod_active && candidate->gpu_lod && !candidate->gpu_lod->busy() &&
                        candidate->rad_signature == pager->signature() && candidate->command.status == MTLCommandBufferStatusCompleted &&
                        (!completed || candidate->producer_value > completed->producer_value))
                        completed = candidate.get();
                const auto touches = completed ? completed->gpu_lod->touches() : BufferSlice{};
                pager->advance(touches.buffer ? std::span<const uint32_t>(static_cast<const uint32_t*>(touches.buffer.contents), pager->cache().snapshot().logical_chunks) : std::span<const uint32_t>{});
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
            const uint32_t draw_count = gpu_lod ? uint32_t(std::clamp<size_t>(request.lod_gpu_traversal.output_capacity, gpu_tree->roots, std::min(gpu_tree->nodes, source_count))) : uint32_t(pager ? model.size() : request.lod_indices ? request.lod_count
                                                                                                                                                                                                                                          : model.size());
            auto& f = i.acquire(slot, request, draw_count);
            if (i.profiling_enabled && !f.gpu_profile)
                f.gpu_profile = std::make_unique<GpuProfile>(i.reader.device());
            auto* profile = i.profiling_enabled ? f.gpu_profile.get() : nullptr;
            if (profile)
                profile->reset();
            f.gpu_lod_active = gpu_lod;
            f.rad_bootstrap = pager && !gpu_lod;
            f.rad_signature = pager ? pager->signature() : 0;
            LodSelection lod{};
            LodParameters lod_parameters{};
            if (gpu_lod) {
                if (!i.lod_selector)
                    i.lod_selector = std::make_unique<LodSelector>(i.reader.device());
                if (!f.gpu_lod || f.gpu_capacity != draw_count || f.gpu_source_count != source_count || f.gpu_tree_signature != gpu_tree->signature) {
                    f.gpu_lod = std::make_unique<LodCutFrame>(i.reader.device(), draw_count, source_count, gpu_tree->chunks);
                    f.gpu_capacity = draw_count;
                    f.gpu_source_count = source_count;
                    f.gpu_tree_signature = gpu_tree->signature;
                    f.gpu_chunks = gpu_tree->chunks;
                }
                lod = f.gpu_lod->selection(request.lod_debug_mode);
                lod.logical_count = logical_count;
                const auto& p = request.lod_gpu_traversal;
                lod_parameters.node_count = gpu_tree->nodes;
                lod_parameters.physical_node_count = source_count;
                lod_parameters.output_capacity = draw_count;
                lod_parameters.logical_chunk_count = gpu_tree->chunks;
                lod_parameters.chunk_splats = uint32_t(core::SplatLodTree::kChunkSplats);
                if (pager) {
                    lod_parameters.current_frame = uint32_t(pager->cache().frameIndex());
                    lod_parameters.fade_frames = wait_for_pages ? 0 : pager->fadeFrames();
                    const auto& snapshot = pager->cache().snapshot();
                    const std::array<std::span<const uint32_t>, 3> maps{snapshot.chunk_to_page, snapshot.page_resident_frame, snapshot.page_to_chunk};
                    for (size_t n = 0; n < maps.size(); ++n) {
                        const size_t bytes = std::max<size_t>(16, maps[n].size_bytes());
                        if (!f.page_maps[n] || f.page_maps[n].length < bytes)
                            f.page_maps[n] = [i.reader.device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                        if (!f.page_maps[n])
                            throw core::MemoryAllocationError({.domain = core::MemoryDomain::MetalDevice, .requested_bytes = bytes, .label = "viewer.rad.maps", .operation = "viewer.buffer.allocate"});
                        std::memcpy(f.page_maps[n].contents, maps[n].data(), maps[n].size_bytes());
                    }
                    gpu_tree->buffers.chunk_to_page = {f.page_maps[0]};
                    gpu_tree->buffers.page_age = {f.page_maps[1]};
                    gpu_tree->buffers.page_to_chunk = {f.page_maps[2]};
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
            } else if (!pager && request.lod_indices) {
                lod.enabled = true;
                lod.debug = request.lod_debug_mode;
                lod.count = draw_count;
                lod.source_count = uint32_t(model.size());
                const std::array<const void*, 4> sources = {request.lod_indices, request.lod_logical_indices, request.lod_levels, request.lod_weights};
                std::array<BufferSlice*, 4> destinations = {&lod.indices, &lod.logical_indices, &lod.levels, &lod.weights};
                for (size_t n = 0; n < sources.size(); ++n) {
                    if (!sources[n])
                        continue;
                    const size_t bytes = std::max<size_t>(16, size_t(draw_count) * 4);
                    auto& buffer = f.lod_buffers[n];
                    if (!buffer || buffer.length < bytes) {
                        const auto device = i.reader.device();
                        if (!frameFitsWorkingSet(device.currentAllocatedSize, bytes, device.recommendedMaxWorkingSetSize))
                            throw lfs::Exception(nativeError(std::format("Metal LOD selection exceeds the recommended GPU working set (bytes={}, allocated={}, recommended={})", bytes, device.currentAllocatedSize, device.recommendedMaxWorkingSetSize), lfs::ErrorCode::ResourceExhausted));
                        buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                    }
                    if (!buffer)
                        throw lfs::Exception(nativeError(std::format("Metal LOD selection allocation failed (bytes={}, draw_count={}, buffer_index={})", bytes, draw_count, n), lfs::ErrorCode::ResourceExhausted));
                    if (draw_count)
                        std::memcpy(buffer.contents, sources[n], size_t(draw_count) * 4);
                    *destinations[n] = {buffer, 0};
                }
            }
            if (request.gut && (!f.gut_geometry || f.gut_geometry.length < size_t(draw_count) * sizeof(GutSplat))) {
                const size_t bytes = std::max<size_t>(16, size_t(draw_count) * sizeof(GutSplat));
                const auto device = i.reader.device();
                if (!frameFitsWorkingSet(device.currentAllocatedSize, bytes, device.recommendedMaxWorkingSetSize))
                    throw lfs::Exception(nativeError(std::format("Metal 3DGUT reservation exceeds the recommended GPU working set (bytes={}, allocated={}, recommended={})", bytes, device.currentAllocatedSize, device.recommendedMaxWorkingSetSize), lfs::ErrorCode::ResourceExhausted));
                f.gut_geometry = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
                if (!f.gut_geometry)
                    throw lfs::Exception(nativeError(std::format("Metal 3DGUT geometry allocation failed (bytes={}, draw_count={}, allocated={}, recommended={})", bytes, draw_count, device.currentAllocatedSize, device.recommendedMaxWorkingSetSize), lfs::ErrorCode::ResourceExhausted));
            }
            const int node_degree = request.scene.node_active_sh_degrees.empty() ? model.get_active_sh_degree() : *std::max_element(request.scene.node_active_sh_degrees.begin(), request.scene.node_active_sh_degrees.end());
            const int max_degree = std::clamp(node_degree, 0, std::min(3, model.get_max_sh_degree()));
            const uint32_t degree = static_cast<uint32_t>(std::clamp(request.sh_degree, 0, max_degree));
            const auto storage = pager && gpu_lod ? ShStorage::RadSigned8 : model.shN_value_quantized() ? ShStorage::Q16
                                                                        : model.shN_ieee_f16()          ? ShStorage::SwizzledFloat16
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
                                     intrinsics.center_x, intrinsics.center_y};
            // Viewer raster clipping differs from the desktop projection matrix's
            // near/far planes. Derive the reference near threshold at configure.
            const bool portal = request.splat_render_profile == 1;
            const bool spark = (pager || gpu_lod || (request.lod_indices && request.lod_count)) && model.lod_tree && model.lod_tree->lod_opacity_encoded;
            const bool portal_math = portal && !spark;
            const bool mip = request.mip_filter && !(portal_math && request.gut);
            const float dilation = portal_math && !request.gut ? .075f : mip ? .1f
                                                                             : .3f;
            projection.clip_scale = {kViewerNearClip, std::numeric_limits<float>::max(), request.scaling_modifier, dilation};
            projection.extent = {uint32_t(f.size.x), uint32_t(f.size.y), uint32_t(request.equirectangular ? CameraModel::Equirectangular : request.frame_view.orthographic ? CameraModel::Orthographic
                                                                                                                                                                           : CameraModel::Perspective),
                                 uint32_t(mip)};
            projection.rasterization = {request.frame_view.rasterization_scale, expected_depth ? 1.f : 0.f, request.frame_view.far_plane, float(request.splat_render_profile)};
            projection.display = {float(request.color_tonemapping), request.color_exposure, spark ? 1.f : 0.f, request.gut ? 0.f : 1.f};
            const auto panorama_size = request.frame_view.cameraSize();
            projection.panorama = {float(panorama_size.x), float(panorama_size.y), float(request.frame_view.subregion_origin.x), float(request.frame_view.subregion_origin.y)};
            SceneBuffers scene{};
            OverlayBuffers overlay{};
            if (request.scene.model_transforms && !request.scene.model_transforms->empty()) {
                const auto& transforms = *request.scene.model_transforms;
                if (transforms.size() > 1 && (!request.scene.transform_indices || !request.scene.transform_indices->is_valid()))
                    throw std::invalid_argument(std::format("Multiple Metal scene transforms require primitive indices (transforms={}, draw_count={})", request.scene.model_transforms->size(), draw_count));
                const size_t bytes = transforms.size() * sizeof(SceneObject);
                if (!f.objects || f.objects.length < bytes)
                    f.objects = [i.reader.device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!f.objects)
                    throw lfs::Exception(nativeError(std::format("Metal scene object allocation failed (objects={}, bytes={})", scene.count, size_t(scene.count) * sizeof(SceneObject)), lfs::ErrorCode::ResourceExhausted));
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
                        throw lfs::Exception(nativeError(std::format("Metal overlay allocation failed (bytes={}, options={}, target={})", bytes, uint64_t(options), slot.value), lfs::ErrorCode::ResourceExhausted));
                };
                allocate(f.overlay_parameters, params->size() * sizeof(float), MTLResourceStorageModeShared);
                std::memcpy(f.overlay_parameters.contents, params->data(), params->size() * sizeof(float));
                allocate(f.overlay_flags, size_t(draw_count) * 4, MTLResourceStorageModePrivate);
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
            std::vector<const core::Tensor*> inputs_tensors(tensors.begin(), tensors.end());
            if (pager && gpu_lod) {
                inputs_tensors.push_back(&pager->pool().regions[7]);
                inputs_tensors.push_back(&pager->pool().regions[8]);
            }
            if (i.serial == std::numeric_limits<uint64_t>::max())
                throw std::runtime_error(std::format("Metal viewport timeline exhausted (serial={})", i.serial));
            const uint64_t serial = i.serial + 1;
            const auto background = request.frame_view.background_color;
            const auto event = i.event;
            const PresentParameters present{request.color_exposure, portal ? 0u : uint32_t(request.color_tonemapping), uint32_t(request.transparent_background), uint32_t(previous != nullptr), request.depth_view_min, request.depth_view_max, uint32_t(request.depth_view), uint32_t(request.depth_visualization_mode), {background.x, background.y, background.z, 1}, {uint32_t(expected_depth), 0, 0, 0}};
            const bool refine = pager && (pager->pending() || !pager->rootReady());
            f.command = i.reader.submit(inputs_tensors, [&](id<MTLCommandBuffer> command, std::span<const core::MetalTensorView> views) {
                // A blit on the readback queue may still sample a recycled slot.
                // GPU ordering protects it without waiting on the host each frame.
                if (i.next_readback)
                    [command encodeWaitForEvent:i.readback_event value:i.next_readback];
                auto slice = [&](size_t n) { return BufferSlice{views[n].buffer, views[n].offset}; };
                if (pager && gpu_lod) {
                    gpu_tree->buffers.bounds = slice(11);
                    gpu_tree->buffers.links = slice(12);
                    gpu_tree->buffers.page_frames = slice(6);
                }
                if (gpu_lod)
                    i.lod_selector->encode(command, gpu_tree->buffers, lod_parameters, *f.gpu_lod);
                SplatBuffers inputs{slice(0), slice(1), slice(2), slice(3), slice(4), slice(5), slice(6), slice(7),
                                    source_count, uint32_t(model.max_sh_coeffs_rest()), storage, model.non_sh_attrs_f16()};
                if (pager && gpu_lod)
                    inputs.rad_page_splats = uint32_t(core::SplatLodTree::kChunkSplats);
                inputs.deleted_count = uint32_t(std::min<size_t>(views[7].bytes, std::numeric_limits<uint32_t>::max()));
                scene.object_indices = slice(8);
                overlay.selection = selection_enabled ? slice(9) : BufferSlice{};
                overlay.preview = preview_enabled ? slice(10) : BufferSlice{};
                overlay.selection_count = selection_enabled ? uint32_t(std::min<size_t>(views[9].bytes, std::numeric_limits<uint32_t>::max())) : 0;
                overlay.preview_count = preview_enabled ? uint32_t(std::min<size_t>(views[10].bytes, std::numeric_limits<uint32_t>::max())) : 0;
                i.preprocessor.encode(command, inputs, projection, degree, request.gut ? PrimitiveMode::Gut : PrimitiveMode::Gaussian, {f.projected, 0}, scene, overlay, request.gut ? BufferSlice{f.gut_geometry, 0} : BufferSlice{}, lod, profile,
                                      !request.gut && !spark && !portal_math && !overlay.parameter_count &&
                                          !request.transparent_background && !request.overlay.markers.show_rings);
                i.rasterizer.encode(command, {f.projected, 0}, draw_count, request.gut ? RasterMode::Gut : RasterMode::Gaussian,
                                    {background.x, background.y, background.z, request.transparent_background ? 0.f : 1.f}, *f.raster, overlay, request.gut ? BufferSlice{f.gut_geometry, 0} : BufferSlice{}, projection, lod, request.gut && !spark, (request.transparent_background || request.overlay.markers.show_rings) && !request.gut && !spark, profile, request.depth_view);
                auto encoder = profiledCompute(command, profile, GpuStage::Present);
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
                const auto completion_event = event;
                const auto completion_value = serial;
                [command encodeSignalEvent:completion_event value:completion_value];
                [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    // Failed read-only producers must release presentation waits too;
                    // their typed command error is inspected before resource reuse.
                    if (completed.status == MTLCommandBufferStatusError && completion_event.signaledValue < completion_value)
                        completion_event.signaledValue = completion_value;
                }];
                const auto completion_retry = i.retry_callback;
                const auto completion_status = f.raster->statusBuffer();
                if (completion_retry) {
                    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                        if (completed.status == MTLCommandBufferStatusError ||
                            static_cast<const RasterStatus*>(completion_status.contents)->error != RasterError::None)
                            completion_retry();
                    }];
                }
            });
            if (pager && gpu_lod)
                pager->noteRendererCompletion(serial);
            f.producer_value = serial;
            i.serial = serial;
            f.consumer_serial = context.lastFrameSubmitSerial() + (context.hasActiveFrame() ? 1 : 0);
            i.target(slot).latest = &f;
            return SceneRenderer::RenderResult{.image = f.color.image, .image_view = f.color.view, .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .generation = f.generation, .depth_image = f.depth.image, .depth_image_view = f.depth.view, .depth_image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .depth_generation = f.generation, .size = f.size, .alloc_size = f.size, .flip_y = false, .completion_semaphore = i.completion, .completion_value = serial, .lod_streaming_active = refine, .viewer_backend = rendering::ViewerBackend::Metal};
        } catch (const std::exception& e) { return nativeError(e); }
    }
    glm::ivec2 MetalViewportRenderer::size(Slot slot) const {
        std::lock_guard lock(impl_->readback_mutex);
        auto f = impl_->latestFrame(slot);
        return f ? f->size : glm::ivec2{};
    }
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
            if (i.readbacks.size() >= ReadbackTicketRing::kRingSize)
                throw std::runtime_error(std::format("Metal readback ring is full; retire or abandon a ticket (active_tickets={}, capacity={}, target={})", i.readbacks.size(), ReadbackTicketRing::kRingSize, slot.value));
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
            if (candidate && candidate->gpu_lod_active && candidate->gpu_lod && !candidate->gpu_lod->busy() &&
                candidate->gpu_tree_signature == latest->gpu_tree_signature &&
                candidate->command.status == MTLCommandBufferStatusCompleted &&
                (!completed || candidate->producer_value > completed->producer_value))
                completed = candidate.get();
        }
        if (!completed)
            return status;
        const auto cut = completed->gpu_lod->status();
        status.capacity = completed->gpu_capacity;
        status.selected = std::min(cut.selected, completed->gpu_capacity);
        status.overflow = cut.overflow;
        status.pixel_scale_feedback = cut.threshold_multiplier;
        const auto touches = completed->gpu_lod->touches();
        status.resident_chunks = status.chunk_count = status.pool_pages = completed->gpu_chunks;
        const auto& pager = i.target(slot).pager;
        if (pager && completed->rad_signature == pager->signature()) {
            status.resident_chunks = pager->cache().snapshot().resident_chunks;
            status.pool_pages = pager->cache().snapshot().physical_pages;
            status.streaming_jobs = pager->cache().outstandingWorkCount() + size_t(pager->pending());
            status.deferred_requests = pager->cache().deferredRequestCount();
            status.admission_frozen = pager->frozen();
        }
        const auto values = static_cast<const uint32_t*>(touches.buffer.contents);
        for (size_t n = 0; n < status.chunk_count; ++n) {
            status.touched_chunks += values[n] != 0;
            status.miss_chunks += values[n] != 0 && values[n] != 0xffffffffu;
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
            if (frame->raster) {
                const auto status = frame->raster->status();
                result.required_instances = status.required_instances;
                result.blend_threads = status.blend_threads;
            }
            result.gpu_command_ms = (frame->command.GPUEndTime - frame->command.GPUStartTime) * 1000.;
            if (!std::isfinite(result.gpu_command_ms) || result.gpu_command_ms < 0)
                throw std::runtime_error(std::format("Invalid native command GPU interval (start={}, end={}, duration_ms={})", frame->command.GPUStartTime, frame->command.GPUEndTime, result.gpu_command_ms));
            if (impl_->profiling_enabled && frame->gpu_profile) {
                result.counter_timestamps_available = frame->gpu_profile->available();
                result.gpu_stage_ms = frame->gpu_profile->resolve();
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
            if (frame->gpu_lod && frame->gpu_lod_active && frame->gpu_lod->status().overflow)
                return false;
            return !frame->raster || frame->raster->status().error == RasterError::None;
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
