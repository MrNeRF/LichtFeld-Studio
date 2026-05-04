/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_viewport_renderer.hpp"

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "viewport/vksplat_compose.comp.spv.h"
#include "vksplat_input_packer.hpp"
#include "window/vulkan_result.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <glm/glm.hpp>
#include <map>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis {
    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        [[nodiscard]] std::string vkError(const char* const operation, const VkResult result) {
            return std::format("{} failed: {}", operation, vkResultToString(result));
        }

        [[nodiscard]] std::map<std::string, std::string> makeVkSplatSpirvPaths() {
            const std::filesystem::path root{LFS_VKSPLAT_SPV_DIR};
            return {
                {"projection_forward", (root / "generated/projection_forward.spv").string()},
                {"generate_keys", (root / "generated/generate_keys.spv").string()},
                {"compute_tile_ranges", (root / "generated/compute_tile_ranges.spv").string()},
                {"setup_dispatch_indirect", (root / "generated/setup_dispatch_indirect.spv").string()},
                {"rasterize_forward", (root / "generated/rasterize_forward.spv").string()},
                {"cumsum_single_pass", (root / "generated/cumsum_single_pass.spv").string()},
                {"cumsum_block_scan", (root / "generated/cumsum_block_scan.spv").string()},
                {"cumsum_scan_block_sums", (root / "generated/cumsum_scan_block_sums.spv").string()},
                {"cumsum_add_block_offsets", (root / "generated/cumsum_add_block_offsets.spv").string()},
                {"radix_sort/upsweep", (root / "radix_sort/upsweep.spv").string()},
                {"radix_sort/spine", (root / "radix_sort/spine.spv").string()},
                {"radix_sort/downsweep", (root / "radix_sort/downsweep.spv").string()},
            };
        }

        [[nodiscard]] std::array<float, 16> rowMajorMat4(const glm::mat4& matrix) {
            std::array<float, 16> row_major{};
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    row_major[static_cast<std::size_t>(row * 4 + col)] = matrix[col][row];
                }
            }
            return row_major;
        }

        [[nodiscard]] std::array<float, 16> multiplyRowMajorMat4(const std::array<float, 16>& a,
                                                                 const std::array<float, 16>& b) {
            std::array<float, 16> result{};
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    float value = 0.0f;
                    for (int k = 0; k < 4; ++k) {
                        value += a[static_cast<std::size_t>(row * 4 + k)] *
                                 b[static_cast<std::size_t>(k * 4 + col)];
                    }
                    result[static_cast<std::size_t>(row * 4 + col)] = value;
                }
            }
            return result;
        }

        [[nodiscard]] VksplatViewportRenderer::ModelInputSnapshot makeModelInputSnapshot(
            const lfs::core::SplatData& splat_data) {
            const auto tensor_ptr = [](const Tensor& tensor) -> const void* {
                return tensor.is_valid() ? tensor.data_ptr() : nullptr;
            };
            const auto tensor_bytes = [](const Tensor& tensor) -> std::size_t {
                return tensor.is_valid() ? tensor.bytes() : 0;
            };

            const Tensor& means = splat_data.means_raw();
            const Tensor& scaling = splat_data.scaling_raw();
            const Tensor& rotation = splat_data.rotation_raw();
            const Tensor& opacity = splat_data.opacity_raw();
            const Tensor& sh0 = splat_data.sh0_raw();
            const Tensor& shn = splat_data.shN_raw();
            return VksplatViewportRenderer::ModelInputSnapshot{
                .model = &splat_data,
                .count = static_cast<std::size_t>(splat_data.size()),
                .max_sh_degree = splat_data.get_max_sh_degree(),
                .means = tensor_ptr(means),
                .scaling = tensor_ptr(scaling),
                .rotation = tensor_ptr(rotation),
                .opacity = tensor_ptr(opacity),
                .sh0 = tensor_ptr(sh0),
                .shn = tensor_ptr(shn),
                .means_bytes = tensor_bytes(means),
                .scaling_bytes = tensor_bytes(scaling),
                .rotation_bytes = tensor_bytes(rotation),
                .opacity_bytes = tensor_bytes(opacity),
                .sh0_bytes = tensor_bytes(sh0),
                .shn_bytes = tensor_bytes(shn),
            };
        }

        struct ComposePushConstants {
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint32_t pad0 = 0;
            std::uint32_t pad1 = 0;
            glm::vec4 background{0.0f, 0.0f, 0.0f, 1.0f};
        };

    } // namespace

    struct VksplatViewportRenderer::ComposePipeline {
        VkShaderModule shader_module = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        void destroy(VkDevice device) {
            if (device == VK_NULL_HANDLE) {
                return;
            }
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pipeline, nullptr);
            }
            if (pipeline_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            }
            if (descriptor_set_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
            }
            if (shader_module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, shader_module, nullptr);
            }
            *this = {};
        }
    };

    VksplatViewportRenderer::VksplatViewportRenderer() = default;

    VksplatViewportRenderer::~VksplatViewportRenderer() {
        reset();
    }

    void VksplatViewportRenderer::reset() {
        if (context_ && context_->device() != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(context_->device());
        }
        // Detach our managed VkBuffers from buffers_ before the renderer's
        // cleanupBuffers runs so it does not vkDestroyBuffer them out from
        // under us.
        if (initialized_) {
            detachManagedBuffers();
            renderer_.cleanupBuffers(buffers_);
            renderer_.cleanup();
        }
        for (auto& slot : cuda_inputs_) {
            slot.interop.reset();
            if (context_) {
                context_->destroyExternalBuffer(slot.buffer);
            }
            slot = {};
        }
        for (auto& snap : ring_uploaded_) {
            snap = {};
        }
        for (auto& timeline : upload_timelines_) {
            timeline.cuda_semaphore.reset();
            if (context_) {
                context_->destroyExternalSemaphore(timeline.vk_semaphore);
            }
            timeline.vk_semaphore = {};
            timeline.value = 0;
        }
        if (context_) {
            if (output_image_.image != VK_NULL_HANDLE) {
                context_->imageBarriers().forgetImage(output_image_.image);
            }
            context_->destroyExternalImage(output_image_);
            if (compose_) {
                compose_->destroy(context_->device());
            }
        }
        compose_.reset();
        buffers_ = {};
        output_size_ = {0, 0};
        output_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        initialized_ = false;
        context_ = nullptr;
    }

    void VksplatViewportRenderer::detachManagedBuffers() {
        const auto detach = [](_VulkanBuffer& dev) {
            dev.buffer = VK_NULL_HANDLE;
            dev.allocation = VK_NULL_HANDLE;
            dev.allocSize = 0;
            dev.size = 0;
            dev.offset = 0;
        };
        detach(buffers_.xyz_ws.deviceBuffer);
        detach(buffers_.rotations.deviceBuffer);
        detach(buffers_.scales_opacs.deviceBuffer);
        detach(buffers_.sh_coeffs.deviceBuffer);
    }

    void VksplatViewportRenderer::plugRingInputs(const std::size_t ring_slot, const std::size_t num_splats) {
        assert(ring_slot < cuda_inputs_.size());
        auto& slot = cuda_inputs_[ring_slot];
        // All four region views share one VkBuffer / one device allocation; only
        // (offset, size) differs per binding. allocation is left null because the
        // CudaInputSlot owns it.
        const auto plug = [&](_VulkanBuffer& dev, std::size_t region) {
            dev.buffer = slot.buffer.buffer;
            dev.allocation = VK_NULL_HANDLE;
            dev.allocSize = static_cast<std::size_t>(slot.buffer.allocation_size);
            dev.offset = slot.region_offset[region];
            dev.size = slot.region_bytes[region];
        };
        plug(buffers_.xyz_ws.deviceBuffer, 0);
        plug(buffers_.rotations.deviceBuffer, 1);
        plug(buffers_.scales_opacs.deviceBuffer, 2);
        plug(buffers_.sh_coeffs.deviceBuffer, 3);

        // Resize host-shadow vectors so the rasterizer's bookkeeping (which
        // calls byteLength()) keeps matching the device-side payload. The host
        // vectors are not read by the rasterizer; only their size() matters
        // when the renderer cross-checks element counts.
        buffers_.xyz_ws.resize(slot.region_bytes[0] / sizeof(float));
        buffers_.rotations.resize(slot.region_bytes[1] / sizeof(float));
        buffers_.scales_opacs.resize(slot.region_bytes[2] / sizeof(float));
        buffers_.sh_coeffs.resize(slot.region_bytes[3] / sizeof(float));

        buffers_.num_splats = num_splats;
        buffers_.num_indices = 0;
        buffers_.is_unsorted_1 = true;
    }

    std::expected<void, std::string> VksplatViewportRenderer::ensureCudaInputSlot(
        VulkanContext& context,
        CudaInputSlot& slot,
        const std::size_t required_bytes,
        const char* const debug_name) {
        if (required_bytes == 0) {
            return std::unexpected(std::format("VkSplat slot '{}' requested zero-byte allocation", debug_name));
        }
        if (slot.buffer.buffer != VK_NULL_HANDLE && slot.buffer.allocation_size >= required_bytes) {
            return {};
        }
        // Re-allocate. The old VkBuffer is still referenced by buffers_ via the
        // four offset views — those are reset by the caller after this returns.
        slot.interop.reset();
        context.destroyExternalBuffer(slot.buffer);

        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (!context.createExternalBuffer(static_cast<VkDeviceSize>(required_bytes), usage, slot.buffer)) {
            return std::unexpected(std::format("VkSplat external buffer '{}' allocation failed: {}",
                                               debug_name,
                                               context.lastError()));
        }
        const auto native = context.releaseExternalBufferNativeHandle(slot.buffer);
        if (!VulkanContext::externalNativeHandleValid(native)) {
            context.destroyExternalBuffer(slot.buffer);
            return std::unexpected(std::format("VkSplat external buffer '{}' returned invalid native handle",
                                               debug_name));
        }
        lfs::rendering::CudaVulkanExternalBufferImport import{
            .memory_handle = native,
            .allocation_size = static_cast<std::size_t>(slot.buffer.allocation_size),
            .size = static_cast<std::size_t>(slot.buffer.size),
            .dedicated_allocation = context.externalMemoryDedicatedAllocationEnabled(),
        };
        if (!slot.interop.init(import)) {
            const std::string err = slot.interop.lastError();
            context.destroyExternalBuffer(slot.buffer);
            return std::unexpected(std::format("VkSplat external buffer '{}' CUDA import failed: {}",
                                               debug_name,
                                               err));
        }
        return {};
    }

    std::expected<void, std::string> VksplatViewportRenderer::ensureInitialized(VulkanContext& context) {
        if (context_ != nullptr && context_ != &context) {
            reset();
        }
        context_ = &context;
        if (initialized_) {
            return {};
        }
        try {
            // Submit the splat dispatch chain on the dedicated async-compute queue
            // when the device exposes one (NVIDIA family 2, AMD family 1, etc.). The
            // existing per-frame timeline-semaphore wait that gates the swapchain pass
            // on the rasterizer's output already provides cross-queue ordering, so
            // graphics-queue work (RmlUi, viewport overlays) can overlap the splat
            // compute pass with no additional synchronization.
            const bool use_async_compute = context.hasDedicatedComputeQueue();
            renderer_.initializeExternal(makeVkSplatSpirvPaths(),
                                         context.instance(),
                                         context.physicalDevice(),
                                         context.device(),
                                         use_async_compute ? context.computeQueue()
                                                           : context.graphicsQueue(),
                                         use_async_compute ? context.computeQueueFamily()
                                                           : context.graphicsQueueFamily(),
                                         context.allocator());
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat initialization failed: {}", e.what()));
        }

        // Per-ring-slot upload timeline: a Vulkan-exportable timeline semaphore
        // imported into CUDA so we can signal CUDA-side after the upload's
        // cudaMemcpyAsync and have Vulkan compute wait on it — replacing the
        // per-frame cudaStreamSynchronize that previously blocked the CPU.
        for (auto& timeline : upload_timelines_) {
            if (!context.createExternalTimelineSemaphore(0, timeline.vk_semaphore)) {
                return std::unexpected(std::format(
                    "VkSplat upload timeline semaphore creation failed: {}",
                    context.lastError()));
            }
            const auto handle = context.releaseExternalSemaphoreNativeHandle(timeline.vk_semaphore);
            if (!VulkanContext::externalNativeHandleValid(handle)) {
                context.destroyExternalSemaphore(timeline.vk_semaphore);
                return std::unexpected("VkSplat upload timeline semaphore export failed");
            }
            lfs::rendering::CudaVulkanExternalSemaphoreImport import{};
            import.semaphore_handle = handle;
            import.initial_value = timeline.vk_semaphore.initial_value;
            if (!timeline.cuda_semaphore.init(import)) {
                std::string err = timeline.cuda_semaphore.lastError();
                context.destroyExternalSemaphore(timeline.vk_semaphore);
                return std::unexpected(std::format(
                    "VkSplat upload timeline semaphore CUDA import failed: {}", err));
            }
            timeline.value = 0;
        }

        initialized_ = true;
        return {};
    }

    bool VksplatViewportRenderer::inputsResident(const lfs::core::SplatData& splat_data,
                                                 const std::size_t ring_slot) const {
        if (ring_slot >= ring_uploaded_.size())
            return false;
        const auto current = makeModelInputSnapshot(splat_data);
        return ring_uploaded_[ring_slot].valid() && ring_uploaded_[ring_slot] == current;
    }

    std::expected<void, std::string> VksplatViewportRenderer::uploadInputs(
        VulkanContext& context,
        const lfs::core::SplatData& splat_data,
        const int active_sh_degree,
        const std::size_t ring_slot) {
        (void)active_sh_degree;
        const std::size_t n = static_cast<std::size_t>(splat_data.size());
        if (n == 0) {
            return std::unexpected("VkSplat cannot render an empty model");
        }

        assert(context.externalMemoryInteropEnabled());
        assert(ring_slot < cuda_inputs_.size());
        auto& slot = cuda_inputs_[ring_slot];

        auto packed = vksplat::packDeviceInputs(splat_data);
        if (!packed) {
            return std::unexpected(packed.error());
        }

        const std::array<std::size_t, kInputRegionCount> region_bytes{
            static_cast<std::size_t>(packed->xyz_ws.bytes()),
            static_cast<std::size_t>(packed->rotations.bytes()),
            static_cast<std::size_t>(packed->scales_opacs.bytes()),
            static_cast<std::size_t>(packed->sh_coeffs.bytes()),
        };

        // Lay out the four regions back-to-back, padding each to kRegionAlignment
        // so the resulting offsets are valid for VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        // bindings on every conformant device. Driver-required alignment is at
        // most 256 bytes (often less) — overshooting here costs ≤ 1 KiB per ring.
        std::array<std::size_t, kInputRegionCount> region_offset{};
        std::size_t cursor = 0;
        for (std::size_t i = 0; i < kInputRegionCount; ++i) {
            region_offset[i] = cursor;
            const std::size_t aligned = (region_bytes[i] + kRegionAlignment - 1) /
                                        kRegionAlignment * kRegionAlignment;
            cursor += aligned;
        }
        const std::size_t total_bytes = cursor;

        if (auto ok = ensureCudaInputSlot(context, slot, total_bytes, "vksplat_inputs"); !ok)
            return std::unexpected(ok.error());

        slot.region_offset = region_offset;
        slot.region_bytes = region_bytes;
        slot.total_live_bytes = total_bytes;

        // Single CUDA-imported VkBuffer — four offset-targeted memcpys. Half the
        // setup cost (1 import / 1 mapped pointer) of the prior 4-buffer scheme;
        // the 4× cudaMemcpyAsync calls overlap on the upload stream as before.
        const cudaStream_t stream = packed->xyz_ws.stream();
        const auto copy_region = [&](std::size_t i, const lfs::core::Tensor& t) {
            return slot.interop.copyFromTensor(t, region_bytes[i], region_offset[i], stream);
        };
        if (!copy_region(0, packed->xyz_ws) ||
            !copy_region(1, packed->rotations) ||
            !copy_region(2, packed->scales_opacs) ||
            !copy_region(3, packed->sh_coeffs)) {
            return std::unexpected(std::format("VkSplat CUDA buffer copy failed: {}",
                                               slot.interop.lastError()));
        }

        // Cross-API handoff: signal CUDA-side after the memcpys complete, queue
        // a Vulkan-side wait so the next vksplat compute submit waits on it
        // before reading the buffers. No CPU stall.
        auto& timeline = upload_timelines_[ring_slot];
        const std::uint64_t signal_value = ++timeline.value;
        if (!timeline.cuda_semaphore.cudaSignal(signal_value, stream)) {
            return std::unexpected(std::format("VkSplat CUDA upload signal failed: {}",
                                               timeline.cuda_semaphore.lastError()));
        }
        context.addFrameTimelineWait(timeline.vk_semaphore.semaphore,
                                     signal_value,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        ring_uploaded_[ring_slot] = makeModelInputSnapshot(splat_data);
        return {};
    }

    std::expected<void, std::string> VksplatViewportRenderer::ensureOutputImage(VulkanContext& context,
                                                                                const glm::ivec2 size) {
        if (output_image_.image != VK_NULL_HANDLE && output_size_ == size) {
            return {};
        }
        if (output_image_.image != VK_NULL_HANDLE) {
            context.imageBarriers().forgetImage(output_image_.image);
        }
        context.destroyExternalImage(output_image_);
        output_size_ = {0, 0};
        output_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        const VkExtent2D extent{
            .width = static_cast<std::uint32_t>(size.x),
            .height = static_cast<std::uint32_t>(size.y),
        };
        if (!context.createExternalImage(extent, VK_FORMAT_R8G8B8A8_UNORM, output_image_)) {
            return std::unexpected(context.lastError());
        }
        context.imageBarriers().registerImage(output_image_.image,
                                              VK_IMAGE_ASPECT_COLOR_BIT,
                                              VK_IMAGE_LAYOUT_UNDEFINED,
                                              /*external=*/true);
        output_size_ = size;
        ++output_generation_;
        return {};
    }

    std::expected<void, std::string> VksplatViewportRenderer::ensureComposePipeline(VulkanContext& context) {
        if (compose_ && compose_->pipeline != VK_NULL_HANDLE) {
            return {};
        }
        compose_ = std::make_unique<ComposePipeline>();
        VkDevice device = context.device();
        VkResult result = VK_SUCCESS;

        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = sizeof(viewport_shaders::kVkSplatComposeCompSpv);
        shader_info.pCode = viewport_shaders::kVkSplatComposeCompSpv;
        result = vkCreateShaderModule(device, &shader_info, nullptr, &compose_->shader_module);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateShaderModule(VkSplat compose)", result));
        }

        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &compose_->descriptor_set_layout);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateDescriptorSetLayout(VkSplat compose)", result));
        }

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.size = sizeof(ComposePushConstants);
        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &compose_->descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        result = vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &compose_->pipeline_layout);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreatePipelineLayout(VkSplat compose)", result));
        }

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = compose_->shader_module;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage;
        pipeline_info.layout = compose_->pipeline_layout;
        result = vkCreateComputePipelines(device, context.pipelineCache(), 1, &pipeline_info, nullptr, &compose_->pipeline);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateComputePipelines(VkSplat compose)", result));
        }
        return {};
    }

    std::expected<void, std::string> VksplatViewportRenderer::composePixelState(
        VulkanContext& context,
        VkCommandBuffer cmd,
        const VulkanGSRendererUniforms& uniforms,
        const glm::vec3& background) {
        if (auto ok = ensureComposePipeline(context); !ok) {
            return ok;
        }

        const bool has_pixel_state = buffers_.num_indices > 0 &&
                                     buffers_.pixel_state.deviceBuffer.buffer != VK_NULL_HANDLE &&
                                     buffers_.pixel_state.deviceBuffer.size > 0;
        if (!has_pixel_state) {
            context.imageBarriers().transitionImage(cmd,
                                                    output_image_.image,
                                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue clear{{background.r, background.g, background.b, 1.0f}};
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.levelCount = 1;
            range.layerCount = 1;
            vkCmdClearColorImage(cmd,
                                 output_image_.image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &clear,
                                 1,
                                 &range);
            context.imageBarriers().transitionImage(cmd,
                                                    output_image_.image,
                                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            output_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ++output_generation_;
            return {};
        }

        VkDescriptorBufferInfo pixel_info{};
        pixel_info.buffer = buffers_.pixel_state.deviceBuffer.buffer;
        pixel_info.range = buffers_.pixel_state.deviceBuffer.size;
        VkDescriptorImageInfo image_info{};
        image_info.imageView = output_image_.view;
        image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &pixel_info;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &image_info;

        VkBufferMemoryBarrier2 pixel_barrier{};
        pixel_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        pixel_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        pixel_barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        pixel_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        pixel_barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        pixel_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pixel_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pixel_barrier.buffer = buffers_.pixel_state.deviceBuffer.buffer;
        pixel_barrier.size = buffers_.pixel_state.deviceBuffer.size;
        VkDependencyInfo pixel_dep{};
        pixel_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        pixel_dep.bufferMemoryBarrierCount = 1;
        pixel_dep.pBufferMemoryBarriers = &pixel_barrier;
        vkCmdPipelineBarrier2(cmd, &pixel_dep);
        context.imageBarriers().transitionImage(cmd,
                                                output_image_.image,
                                                VK_IMAGE_ASPECT_COLOR_BIT,
                                                VK_IMAGE_LAYOUT_GENERAL);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compose_->pipeline);
        context.vkCmdPushDescriptorSet()(cmd,
                                         VK_PIPELINE_BIND_POINT_COMPUTE,
                                         compose_->pipeline_layout,
                                         0,
                                         static_cast<std::uint32_t>(writes.size()),
                                         writes.data());
        ComposePushConstants push{
            .width = uniforms.image_width,
            .height = uniforms.image_height,
            .background = glm::vec4(background, 1.0f),
        };
        vkCmdPushConstants(cmd,
                           compose_->pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(push),
                           &push);
        vkCmdDispatch(cmd,
                      _CEIL_DIV(uniforms.image_width, 16),
                      _CEIL_DIV(uniforms.image_height, 16),
                      1);
        context.imageBarriers().transitionImage(cmd,
                                                output_image_.image,
                                                VK_IMAGE_ASPECT_COLOR_BIT,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        output_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ++output_generation_;
        return {};
    }

    std::expected<VksplatViewportRenderer::RenderResult, std::string> VksplatViewportRenderer::render(
        VulkanContext& context,
        const lfs::core::SplatData& splat_data,
        const lfs::rendering::ViewportRenderRequest& request,
        const bool force_input_upload) {
        const glm::ivec2 size = request.frame_view.size;
        if (size.x <= 0 || size.y <= 0) {
            return std::unexpected("VkSplat received an invalid viewport size");
        }
        if (request.frame_view.orthographic) {
            return std::unexpected("VkSplat forward path supports pinhole cameras, not orthographic cameras");
        }
        if (request.equirectangular) {
            return std::unexpected("VkSplat forward path supports pinhole cameras, not equirectangular cameras");
        }
        assert(context.externalMemoryInteropEnabled());

        const int active_sh_degree = std::clamp(request.sh_degree, 0, std::min(3, splat_data.get_max_sh_degree()));
        if (auto ok = ensureInitialized(context); !ok) {
            return std::unexpected(ok.error());
        }

        const std::size_t ring_slot = context.currentFrameSlot() % kInputRingSize;
        assert(context.framesInFlight() == kInputRingSize);

        if (force_input_upload || !inputsResident(splat_data, ring_slot)) {
            if (auto ok = uploadInputs(context, splat_data, active_sh_degree, ring_slot); !ok) {
                return std::unexpected(ok.error());
            }
            // Drop the deferred-readback high-water-mark whenever the model identity
            // changes — a fresh model can have a wildly different num_indices range,
            // and stale estimates risk under-sizing the sort buffers (or wasting VRAM
            // if oversized). The next frame re-seeds via heuristic and grows from there.
            renderer_.resetNumIndicesEstimate();
        }
        plugRingInputs(ring_slot, static_cast<std::size_t>(splat_data.size()));
        if (auto ok = ensureOutputImage(context, size); !ok) {
            return std::unexpected(ok.error());
        }

        VulkanGSRendererUniforms uniforms{};
        uniforms.image_width = static_cast<std::uint32_t>(size.x);
        uniforms.image_height = static_cast<std::uint32_t>(size.y);
        uniforms.grid_width = _CEIL_DIV(uniforms.image_width, TILE_WIDTH);
        uniforms.grid_height = _CEIL_DIV(uniforms.image_height, TILE_HEIGHT);
        uniforms.num_splats = static_cast<std::uint32_t>(buffers_.num_splats);
        uniforms.active_sh = static_cast<std::uint32_t>(active_sh_degree);
        uniforms.camera_model = 0;

        if (request.frame_view.intrinsics_override) {
            const auto& intrinsics = *request.frame_view.intrinsics_override;
            uniforms.fx = intrinsics.focal_x;
            uniforms.fy = intrinsics.focal_y;
            uniforms.cx = intrinsics.center_x;
            uniforms.cy = intrinsics.center_y;
        } else {
            const auto [fx, fy] = lfs::rendering::computePixelFocalLengths(
                size, request.frame_view.focal_length_mm);
            uniforms.fx = fx;
            uniforms.fy = fy;
            uniforms.cx = static_cast<float>(size.x) * 0.5f;
            uniforms.cy = static_cast<float>(size.y) * 0.5f;
        }

        const glm::mat3 camera_to_world =
            lfs::rendering::dataCameraToWorldFromVisualizerRotation(request.frame_view.rotation);
        const glm::mat3 world_to_camera = glm::transpose(camera_to_world);
        const glm::vec3 translation = -world_to_camera * request.frame_view.translation;

        std::array<float, 16> row_major_view{};
        row_major_view[15] = 1.0f;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                row_major_view[static_cast<std::size_t>(row * 4 + col)] = world_to_camera[col][row];
            }
        }
        row_major_view[3] = translation.x;
        row_major_view[7] = translation.y;
        row_major_view[11] = translation.z;
        // VkSplat doesn't yet rebake per-gaussian transform indices, so it can
        // only fold a single model transform into the view matrix. With one or
        // more visible nodes, apply the first transform — the data→visualizer
        // axes correction is identical across paste-cloned nodes, so this keeps
        // the typical multi-node case (paste, same-source duplicates) oriented
        // correctly. Truly heterogeneous per-node transforms still need a CUDA
        // backend until vksplat grows transform indexing.
        if (request.scene.model_transforms && !request.scene.model_transforms->empty()) {
            row_major_view = multiplyRowMajorMat4(row_major_view, rowMajorMat4(request.scene.model_transforms->front()));
        }
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                uniforms.world_view_transform[4 * row + col] =
                    row_major_view[static_cast<std::size_t>(4 * col + row)];
            }
        }

        std::expected<void, std::string> compose_status;
        try {
            auto batch = DeviceGuard(&renderer_);
            renderer_.executeProjectionForward(uniforms, buffers_);
            renderer_.executeCalculateIndexBufferOffset(buffers_);
            if (buffers_.num_indices > 0) {
                renderer_.executeGenerateKeys(uniforms, buffers_);
                renderer_.executeSort(uniforms, buffers_, -1);
                renderer_.executeComputeTileRanges(uniforms, buffers_);
                renderer_.executeRasterizeForward(uniforms, buffers_);
            }
            // Record compose into the rasterizer's batch so the entire frame
            // submits and waits exactly once instead of fence-blocking twice.
            compose_status = composePixelState(
                context, renderer_.activeCommandBuffer(), uniforms, request.frame_view.background_color);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat forward pass failed: {}", e.what()));
        }
        if (!compose_status) {
            return std::unexpected(compose_status.error());
        }

        return RenderResult{
            .image = output_image_.image,
            .image_view = output_image_.view,
            .image_layout = output_layout_,
            .generation = output_generation_,
            .size = size,
            .flip_y = false,
        };
    }

} // namespace lfs::vis
