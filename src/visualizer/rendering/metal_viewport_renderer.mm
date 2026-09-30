/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_viewport_renderer.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_metal_reader.hpp"
#include "metal_present_source.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "splat_preprocessor.hpp"
#include "tile_rasterizer.hpp"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vulkan/vulkan_metal.h>

namespace lfs::vis {
    namespace {
        using namespace rendering::metal;
        using Slot = VksplatViewportRenderer::OutputSlot;
        std::atomic<uint64_t> generation{uint64_t{1} << 63};
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
        };
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
                descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                texture = [metal newTextureWithDescriptor:descriptor];
                if (!texture)
                    throw std::runtime_error("Metal viewport texture allocation failed");
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
            id<MTLBuffer> projected, objects;
            id<MTLCommandBuffer> command;
            Image color, depth;
        };
    } // namespace
    struct MetalViewportRenderer::Impl {
        VulkanContext* context = nullptr;
        core::MetalTensorReader reader;
        SplatPreprocessor preprocessor{reader.device()};
        TileRasterizer rasterizer{reader.device()};
        id<MTLComputePipelineState> present;
        id<MTLSharedEvent> event;
        VkSemaphore completion = VK_NULL_HANDLE;
        uint64_t serial = 0;
        std::array<std::array<std::unique_ptr<Frame>, 3>, 4> frames;
        std::array<Frame*, 4> latest{};
        std::array<size_t, 4> next{};
        std::array<uint32_t, 4> needed_capacity{};
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
                throw std::runtime_error(error.localizedDescription.UTF8String);
            present = [reader.device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"present_viewer"] error:&error];
            if (!present)
                throw std::runtime_error(error.localizedDescription.UTF8String);
            event = [reader.device() newSharedEvent];
            if (!event)
                throw std::runtime_error("Metal presentation timeline allocation failed");
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
        Frame& acquire(Slot output, const rendering::ViewportRenderRequest& request, uint32_t count) {
            const size_t slot = static_cast<size_t>(output);
            auto& frame = frames[slot][next[slot]++ % 3];
            uint32_t capacity = std::max(needed_capacity[slot], static_cast<uint32_t>(std::min<uint64_t>(uint64_t(count) * 16 + 4096, 16u * 1024u * 1024u)));
            if (frame) {
                wait(frame->producer_value);
                if (frame->command.status == MTLCommandBufferStatusError)
                    throw std::runtime_error(frame->command.error.localizedDescription.UTF8String);
                if (!context->waitForRetiredFrameSubmitSerial(frame->consumer_serial))
                    throw std::runtime_error(context->lastError());
                if (frame->raster->busy())
                    [frame->command waitUntilCompleted];
                const auto status = frame->raster->status();
                if (status.error != RasterError::None) {
                    if (status.required_instances > std::numeric_limits<uint32_t>::max())
                        throw std::runtime_error("Metal viewport instance count exceeds 32-bit indexing");
                    capacity = std::max(capacity, static_cast<uint32_t>(status.required_instances));
                }
                if (frame->size == request.frame_view.size && frame->count >= count && frame->capacity >= capacity)
                    return *frame;
                if (latest[slot] == frame.get())
                    latest[slot] = nullptr;
                frame.reset();
            }
            frame = std::make_unique<Frame>();
            auto& f = *frame;
            f.size = request.frame_view.size;
            f.count = count;
            f.capacity = capacity;
            f.generation = ++generation;
            auto device = reader.device();
            f.raster = std::make_unique<RasterFrame>(device, f.size.x, f.size.y, count, capacity);
            f.projected = [device newBufferWithLength:std::max<size_t>(16, size_t(count) * sizeof(ProjectedSplat)) options:MTLResourceStorageModePrivate];
            if (!f.projected)
                throw std::runtime_error("Metal projected buffer allocation failed");
            f.color.init(*context, device, f.size.x, f.size.y, MTLPixelFormatRGBA8Unorm, VK_FORMAT_R8G8B8A8_UNORM);
            f.depth.init(*context, device, f.size.x, f.size.y, MTLPixelFormatR32Float, VK_FORMAT_R32_SFLOAT);
            return f;
        }
        id<MTLBuffer> readTexture(Frame& f, Image& image, size_t bytes_per_pixel, glm::ivec2 pixel = {-1, -1}) const {
            wait(f.producer_value);
            if (f.command.status == MTLCommandBufferStatusError)
                throw std::runtime_error(f.command.error.localizedDescription.UTF8String);
            auto queue = [reader.device() newCommandQueue];
            auto command = [queue commandBuffer];
            const bool sample = pixel.x >= 0;
            const size_t width = sample ? 1 : f.size.x, height = sample ? 1 : f.size.y;
            const size_t row = width * bytes_per_pixel;
            auto result = [reader.device() newBufferWithLength:row * height options:MTLResourceStorageModeShared];
            if (!queue || !command || !result)
                throw std::runtime_error("Metal readback allocation failed");
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
    bool MetalViewportRenderer::supports(const core::SplatData& model, const rendering::ViewportRenderRequest& r) {
        // Every unsupported contract is routed to the existing renderer; never silently
        // drop filters, display settings or editor overlays from a requested frame.
        return core::gpu_backend_of(model.means_raw()) == core::GpuBackend::Metal &&
               !r.gut && !r.equirectangular && r.splat_render_profile == 0 &&
               !r.lod_indices && !r.lod_gpu_traversal.enabled && !r.lod_debug_mode &&
               !r.filters.crop_region && !r.filters.ellipsoid_region && r.filters.crop_regions.empty() &&
               r.filters.ellipsoid_regions.empty() && !r.filters.view_volume && !r.filters.screen_window &&
               !r.overlay.has_selection && !r.overlay.emphasis.mask && !r.overlay.emphasis.transient_mask.mask &&
               r.overlay.emphasis.emphasized_node_mask.empty() && !r.overlay.emphasis.dim_non_emphasized &&
               r.overlay.emphasis.flash_intensity == 0 && r.overlay.emphasis.focused_gaussian_id < 0 &&
               !r.overlay.cursor.enabled && !r.overlay.markers.show_rings && !r.overlay.markers.show_center_markers &&
               model.means_raw().dtype() == core::DataType::Float32 && model.sh0_raw().dtype() == core::DataType::Float32 &&
               ((model.scaling_raw().dtype() == core::DataType::Float32 && model.rotation_raw().dtype() == core::DataType::Float32 &&
                 model.opacity_raw().dtype() == core::DataType::Float32) ||
                model.non_sh_attrs_f16());
    }
    std::expected<VksplatViewportRenderer::RenderResult, std::string> MetalViewportRenderer::render(
        VulkanContext& context, const core::SplatData& model, const rendering::ViewportRenderRequest& request, Slot slot) {
        try {
            auto& i = *impl_;
            i.initialize(context);
            if (model.size() > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("Metal primitive count exceeds indexing capacity");
            auto previous = i.latest[static_cast<size_t>(slot)];
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
            projection.clip_scale = {request.frame_view.near_plane, request.frame_view.far_plane, request.scaling_modifier, .3f};
            projection.extent = {uint32_t(f.size.x), uint32_t(f.size.y), uint32_t(request.frame_view.orthographic), uint32_t(request.mip_filter)};
            SceneBuffers scene{};
            if (request.scene.model_transforms && !request.scene.model_transforms->empty()) {
                const auto& transforms = *request.scene.model_transforms;
                if (!request.scene.transform_indices)
                    throw std::runtime_error("Metal scene transforms require primitive indices");
                const size_t bytes = transforms.size() * sizeof(SceneObject);
                if (!f.objects || f.objects.length < bytes)
                    f.objects = [i.reader.device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!f.objects)
                    throw std::runtime_error("Metal scene object allocation failed");
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
            std::array<const core::Tensor*, 9> tensors{&model.means_raw(), &model.scaling_raw(), &model.rotation_raw(),
                                                       &model.opacity_raw(), &model.sh0_raw(), degree ? &model.shN_raw() : nullptr,
                                                       degree ? &model.shN_value_bounds() : nullptr, &model.deleted(), scene.count ? request.scene.transform_indices.get() : nullptr};
            if (i.serial == std::numeric_limits<uint64_t>::max())
                throw std::runtime_error("Metal viewport timeline exhausted");
            const uint64_t serial = i.serial + 1;
            const auto background = request.frame_view.background_color;
            const auto event = i.event;
            const PresentParameters present{request.color_exposure, uint32_t(request.color_tonemapping), uint32_t(request.transparent_background), uint32_t(previous != nullptr), request.depth_view_min, request.depth_view_max, uint32_t(request.depth_view), uint32_t(request.depth_visualization_mode), {background.x, background.y, background.z, 1}};
            uint64_t key = 1469598103934665603ull;
            const auto hash = [&](const void* bytes, size_t length) {
                const auto data = static_cast<const uint8_t*>(bytes);
                for (size_t n = 0; n < length; ++n) {
                    key ^= data[n];
                    key *= 1099511628211ull;
                }
            };
            hash(&projection, sizeof(projection));
            hash(&present, 12);
            hash(&present.depth_min, sizeof(PresentParameters) - 16);
            const auto mask_version = model.deleted_mask_version();
            hash(&mask_version, sizeof(mask_version));
            if (scene.count)
                hash(f.objects.contents, scene.count * sizeof(SceneObject));
            for (const auto tensor : tensors)
                if (tensor && tensor->is_valid()) {
                    const auto pointer = tensor->data_ptr();
                    hash(&pointer, sizeof(pointer));
                }
            const bool refine = !previous || previous->request_key != key || previous->raster->busy() ||
                                previous->raster->status().error != RasterError::None;
            f.command = i.reader.submit(tensors, [&](id<MTLCommandBuffer> command, std::span<const core::MetalTensorView> views) {
                auto slice = [&](size_t n) { return BufferSlice{views[n].buffer, views[n].offset}; };
                SplatBuffers inputs{slice(0), slice(1), slice(2), slice(3), slice(4), slice(5), slice(6), slice(7),
                                    uint32_t(model.size()), uint32_t(model.max_sh_coeffs_rest()), storage, model.non_sh_attrs_f16()};
                scene.object_indices = slice(8);
                i.preprocessor.encode(command, inputs, projection, degree, PrimitiveMode::Gaussian, {f.projected, 0}, scene);
                i.rasterizer.encode(command, {f.projected, 0}, uint32_t(model.size()), RasterMode::Gaussian,
                                    {background.x, background.y, background.z, request.transparent_background ? 0.f : 1.f}, *f.raster);
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
            return VksplatViewportRenderer::RenderResult{.image = f.color.image, .image_view = f.color.view, .image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .generation = f.generation, .depth_image = f.depth.image, .depth_image_view = f.depth.view, .depth_image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .depth_generation = f.generation, .size = f.size, .alloc_size = f.size, .flip_y = false, .completion_semaphore = i.completion, .completion_value = serial, .lod_streaming_active = refine};
        } catch (const std::exception& e) { return std::unexpected(e.what()); }
    }
    glm::ivec2 MetalViewportRenderer::size(Slot slot) const {
        auto f = impl_->latest[static_cast<size_t>(slot)];
        return f ? f->size : glm::ivec2{};
    }
    std::expected<void, std::string> MetalViewportRenderer::readColor(Slot slot, core::Tensor& destination, int x, int y) const {
        try {
            auto f = impl_->latest[static_cast<size_t>(slot)];
            if (!f)
                throw std::runtime_error("Metal output slot is empty");
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
        } catch (const std::exception& e) { return std::unexpected(e.what()); }
    }
    std::expected<float, std::string> MetalViewportRenderer::readDepth(const VksplatViewportRenderer::DepthSampleRequest& r) const {
        try {
            auto f = impl_->latest[static_cast<size_t>(r.output_slot)];
            if (!f)
                throw std::runtime_error("Metal depth slot is empty");
            auto p = r.pixel;
            if (r.source_size.x > 0 && r.source_size.y > 0)
                p = glm::clamp(glm::ivec2(glm::round((glm::vec2(p) + .5f) * glm::vec2(f->size) / glm::vec2(r.source_size) - .5f)), glm::ivec2(0), f->size - 1);
            if (p.x < 0 || p.y < 0 || p.x >= f->size.x || p.y >= f->size.y)
                return -1.f;
            auto buffer = impl_->readTexture(*f, f->depth, 4, p);
            const float value = static_cast<const float*>(buffer.contents)[0];
            return value > 0 && value < 1e9f ? value : -1.f;
        } catch (const std::exception& e) { return std::unexpected(e.what()); }
    }
} // namespace lfs::vis
