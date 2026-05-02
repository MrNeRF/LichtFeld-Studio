/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_viewport_renderer.hpp"

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "viewport/vksplat_compose.comp.spv.h"
#include "vksplat_input_packer.hpp"

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
            return std::format("{} failed: {}", operation, static_cast<int>(result));
        }

        [[nodiscard]] std::map<std::string, std::string> makeVkSplatSpirvPaths() {
            const std::filesystem::path root{LFS_VKSPLAT_SPV_DIR};
            return {
                {"projection_forward", (root / "generated/projection_forward.spv").string()},
                {"generate_keys", (root / "generated/generate_keys.spv").string()},
                {"compute_tile_ranges", (root / "generated/compute_tile_ranges.spv").string()},
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

        [[nodiscard]] VkAccessFlags accessForLayout(const VkImageLayout layout) {
            switch (layout) {
            case VK_IMAGE_LAYOUT_UNDEFINED:
                return 0;
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                return VK_ACCESS_TRANSFER_WRITE_BIT;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return VK_ACCESS_SHADER_READ_BIT;
            case VK_IMAGE_LAYOUT_GENERAL:
                return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            default:
                return VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            }
        }

        [[nodiscard]] VkPipelineStageFlags stageForLayout(const VkImageLayout layout) {
            switch (layout) {
            case VK_IMAGE_LAYOUT_UNDEFINED:
                return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                return VK_PIPELINE_STAGE_TRANSFER_BIT;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            case VK_IMAGE_LAYOUT_GENERAL:
                return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            default:
                return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            }
        }

        void imageBarrier(VkCommandBuffer command_buffer,
                          VkImage image,
                          const VkImageLayout old_layout,
                          const VkImageLayout new_layout,
                          const VkAccessFlags dst_access,
                          const VkPipelineStageFlags dst_stage) {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcAccessMask = accessForLayout(old_layout);
            barrier.dstAccessMask = dst_access;
            barrier.oldLayout = old_layout;
            barrier.newLayout = new_layout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(command_buffer,
                                 stageForLayout(old_layout),
                                 dst_stage,
                                 0,
                                 0,
                                 nullptr,
                                 0,
                                 nullptr,
                                 1,
                                 &barrier);
        }

        [[nodiscard]] std::expected<void, std::string> submitAndWait(VulkanContext& context,
                                                                     VkCommandBuffer command_buffer,
                                                                     VkFence fence) {
            VkResult result = vkEndCommandBuffer(command_buffer);
            if (result != VK_SUCCESS) {
                return std::unexpected(vkError("vkEndCommandBuffer", result));
            }
            result = vkResetFences(context.device(), 1, &fence);
            if (result != VK_SUCCESS) {
                return std::unexpected(vkError("vkResetFences", result));
            }
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command_buffer;
            result = vkQueueSubmit(context.graphicsQueue(), 1, &submit, fence);
            if (result != VK_SUCCESS) {
                return std::unexpected(vkError("vkQueueSubmit", result));
            }
            result = vkWaitForFences(context.device(), 1, &fence, VK_TRUE, UINT64_MAX);
            if (result != VK_SUCCESS) {
                return std::unexpected(vkError("vkWaitForFences", result));
            }
            return {};
        }
    } // namespace

    struct VksplatViewportRenderer::ComposePipeline {
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkShaderModule shader_module = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
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
            if (descriptor_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            }
            if (descriptor_set_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
            }
            if (shader_module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, shader_module, nullptr);
            }
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, fence, nullptr);
            }
            if (command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, command_pool, nullptr);
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
        if (context_) {
            context_->destroyExternalImage(output_image_);
            if (compose_) {
                compose_->destroy(context_->device());
            }
        }
        compose_.reset();
        buffers_ = {};
        uploaded_inputs_ = {};
        output_size_ = {0, 0};
        output_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        cuda_inputs_supported_ = true;
        initialized_ = false;
        context_ = nullptr;
    }

    void VksplatViewportRenderer::detachManagedBuffers() {
        const auto detach = [](_VulkanBuffer& dev) {
            dev.buffer = VK_NULL_HANDLE;
            dev.memory = VK_NULL_HANDLE;
            dev.allocSize = 0;
            dev.size = 0;
        };
        detach(buffers_.xyz_ws.deviceBuffer);
        detach(buffers_.rotations.deviceBuffer);
        detach(buffers_.scales_opacs.deviceBuffer);
        detach(buffers_.sh_coeffs.deviceBuffer);
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
        // Re-allocate. The old VkBuffer is still referenced by buffers_ — that
        // pointer will be reset by the caller after this returns.
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
        slot.element_size = sizeof(float);
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
            renderer_.initializeExternal(makeVkSplatSpirvPaths(),
                                         context.instance(),
                                         context.physicalDevice(),
                                         context.device(),
                                         context.graphicsQueue(),
                                         context.graphicsQueueFamily());
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat initialization failed: {}", e.what()));
        }
        initialized_ = true;
        return {};
    }

    bool VksplatViewportRenderer::inputsResident(const lfs::core::SplatData& splat_data) const {
        return uploaded_inputs_.valid() &&
               buffers_.num_splats > 0 &&
               buffers_.xyz_ws.deviceBuffer.buffer != VK_NULL_HANDLE &&
               buffers_.sh_coeffs.deviceBuffer.buffer != VK_NULL_HANDLE &&
               buffers_.rotations.deviceBuffer.buffer != VK_NULL_HANDLE &&
               buffers_.scales_opacs.deviceBuffer.buffer != VK_NULL_HANDLE &&
               uploaded_inputs_ == makeModelInputSnapshot(splat_data);
    }

    std::expected<void, std::string> VksplatViewportRenderer::uploadInputs(
        VulkanContext& context,
        const lfs::core::SplatData& splat_data,
        const int active_sh_degree) {
        (void)active_sh_degree;
        const std::size_t n = static_cast<std::size_t>(splat_data.size());
        if (n == 0) {
            return std::unexpected("VkSplat cannot render an empty model");
        }

        if (cuda_inputs_supported_ && context.externalMemoryInteropEnabled()) {
            auto packed = vksplat::packDeviceInputs(splat_data);
            if (!packed) {
                return std::unexpected(packed.error());
            }

            const std::size_t xyz_bytes = static_cast<std::size_t>(packed->xyz_ws.bytes());
            const std::size_t rot_bytes = static_cast<std::size_t>(packed->rotations.bytes());
            const std::size_t so_bytes = static_cast<std::size_t>(packed->scales_opacs.bytes());
            const std::size_t sh_bytes = static_cast<std::size_t>(packed->sh_coeffs.bytes());

            auto& xyz_slot = cuda_inputs_[0];
            auto& rot_slot = cuda_inputs_[1];
            auto& so_slot = cuda_inputs_[2];
            auto& sh_slot = cuda_inputs_[3];

            const auto setup = [&](CudaInputSlot& slot, std::size_t bytes, const char* name) {
                return ensureCudaInputSlot(context, slot, bytes, name);
            };
            if (auto ok = setup(xyz_slot, xyz_bytes, "xyz_ws"); !ok) {
                cuda_inputs_supported_ = false;
                return std::unexpected(ok.error());
            }
            if (auto ok = setup(rot_slot, rot_bytes, "rotations"); !ok) {
                cuda_inputs_supported_ = false;
                return std::unexpected(ok.error());
            }
            if (auto ok = setup(so_slot, so_bytes, "scales_opacs"); !ok) {
                cuda_inputs_supported_ = false;
                return std::unexpected(ok.error());
            }
            if (auto ok = setup(sh_slot, sh_bytes, "sh_coeffs"); !ok) {
                cuda_inputs_supported_ = false;
                return std::unexpected(ok.error());
            }

            const cudaStream_t stream = packed->xyz_ws.stream();
            if (!xyz_slot.interop.copyFromTensor(packed->xyz_ws, xyz_bytes, stream) ||
                !rot_slot.interop.copyFromTensor(packed->rotations, rot_bytes, stream) ||
                !so_slot.interop.copyFromTensor(packed->scales_opacs, so_bytes, stream) ||
                !sh_slot.interop.copyFromTensor(packed->sh_coeffs, sh_bytes, stream)) {
                cuda_inputs_supported_ = false;
                return std::unexpected(std::format("VkSplat CUDA buffer copy failed: {}",
                                                   sh_slot.interop.lastError()));
            }
            const cudaError_t sync = stream != nullptr ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
            if (sync != cudaSuccess) {
                cuda_inputs_supported_ = false;
                return std::unexpected(std::format("VkSplat CUDA stream sync failed: {} ({})",
                                                   cudaGetErrorName(sync),
                                                   cudaGetErrorString(sync)));
            }

            const auto plug = [](_VulkanBuffer& dev, const VulkanContext::ExternalBuffer& src, std::size_t live_bytes) {
                dev.buffer = src.buffer;
                dev.memory = src.memory;
                dev.allocSize = static_cast<std::size_t>(src.allocation_size);
                dev.size = live_bytes;
            };
            plug(buffers_.xyz_ws.deviceBuffer, xyz_slot.buffer, xyz_bytes);
            plug(buffers_.rotations.deviceBuffer, rot_slot.buffer, rot_bytes);
            plug(buffers_.scales_opacs.deviceBuffer, so_slot.buffer, so_bytes);
            plug(buffers_.sh_coeffs.deviceBuffer, sh_slot.buffer, sh_bytes);

            // Resize host-shadow vectors so the rasterizer's bookkeeping (which
            // calls byteLength()) keeps matching the device-side payload. The
            // host vectors are not read by the rasterizer; only their size()
            // matters when the renderer cross-checks element counts.
            buffers_.xyz_ws.resize(xyz_bytes / sizeof(float));
            buffers_.rotations.resize(rot_bytes / sizeof(float));
            buffers_.scales_opacs.resize(so_bytes / sizeof(float));
            buffers_.sh_coeffs.resize(sh_bytes / sizeof(float));

            buffers_.num_splats = n;
            buffers_.num_indices = 0;
            buffers_.is_unsorted_1 = true;
            uploaded_inputs_ = makeModelInputSnapshot(splat_data);
            return {};
        }

        // Fallback: legacy host packer + staging upload.
        if (auto ok = vksplat::packHostInputs(splat_data,
                                              buffers_.xyz_ws,
                                              buffers_.rotations,
                                              buffers_.scales_opacs,
                                              buffers_.sh_coeffs);
            !ok) {
            return std::unexpected(ok.error());
        }

        buffers_.num_splats = n;
        buffers_.num_indices = 0;
        buffers_.is_unsorted_1 = true;
        try {
            renderer_.copyToDevice(buffers_.xyz_ws);
            renderer_.copyToDevice(buffers_.sh_coeffs);
            renderer_.copyToDevice(buffers_.rotations);
            renderer_.copyToDevice(buffers_.scales_opacs);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat input upload failed: {}", e.what()));
        }
        uploaded_inputs_ = makeModelInputSnapshot(splat_data);
        return {};
    }

    std::expected<void, std::string> VksplatViewportRenderer::ensureOutputImage(VulkanContext& context,
                                                                                const glm::ivec2 size) {
        if (output_image_.image != VK_NULL_HANDLE && output_size_ == size) {
            return {};
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

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = context.graphicsQueueFamily();
        VkResult result = vkCreateCommandPool(device, &pool_info, nullptr, &compose_->command_pool);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateCommandPool(VkSplat compose)", result));
        }

        VkCommandBufferAllocateInfo command_info{};
        command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_info.commandPool = compose_->command_pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        result = vkAllocateCommandBuffers(device, &command_info, &compose_->command_buffer);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkAllocateCommandBuffers(VkSplat compose)", result));
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        result = vkCreateFence(device, &fence_info, nullptr, &compose_->fence);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateFence(VkSplat compose)", result));
        }

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
        layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &compose_->descriptor_set_layout);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateDescriptorSetLayout(VkSplat compose)", result));
        }

        std::array<VkDescriptorPoolSize, 2> pool_sizes{};
        pool_sizes[0] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1};
        pool_sizes[1] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1};
        VkDescriptorPoolCreateInfo descriptor_pool_info{};
        descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptor_pool_info.maxSets = 1;
        descriptor_pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        descriptor_pool_info.pPoolSizes = pool_sizes.data();
        result = vkCreateDescriptorPool(device, &descriptor_pool_info, nullptr, &compose_->descriptor_pool);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateDescriptorPool(VkSplat compose)", result));
        }

        VkDescriptorSetAllocateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set_info.descriptorPool = compose_->descriptor_pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &compose_->descriptor_set_layout;
        result = vkAllocateDescriptorSets(device, &set_info, &compose_->descriptor_set);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkAllocateDescriptorSets(VkSplat compose)", result));
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
        const VulkanGSRendererUniforms& uniforms,
        const glm::vec3& background) {
        if (auto ok = ensureComposePipeline(context); !ok) {
            return ok;
        }

        VkDevice device = context.device();
        VkResult result = vkResetCommandPool(device, compose_->command_pool, 0);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkResetCommandPool(VkSplat compose)", result));
        }

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(compose_->command_buffer, &begin);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkBeginCommandBuffer(VkSplat compose)", result));
        }

        const bool has_pixel_state = buffers_.num_indices > 0 &&
                                     buffers_.pixel_state.deviceBuffer.buffer != VK_NULL_HANDLE &&
                                     buffers_.pixel_state.deviceBuffer.size > 0;
        if (!has_pixel_state) {
            imageBarrier(compose_->command_buffer,
                         output_image_.image,
                         output_layout_,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkClearColorValue clear{{background.r, background.g, background.b, 1.0f}};
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.levelCount = 1;
            range.layerCount = 1;
            vkCmdClearColorImage(compose_->command_buffer,
                                 output_image_.image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &clear,
                                 1,
                                 &range);
            imageBarrier(compose_->command_buffer,
                         output_image_.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_ACCESS_SHADER_READ_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            if (auto ok = submitAndWait(context, compose_->command_buffer, compose_->fence); !ok) {
                return ok;
            }
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
        writes[0].dstSet = compose_->descriptor_set;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &pixel_info;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = compose_->descriptor_set;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &image_info;
        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkBufferMemoryBarrier pixel_barrier{};
        pixel_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        pixel_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        pixel_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        pixel_barrier.srcQueueFamilyIndex = context.graphicsQueueFamily();
        pixel_barrier.dstQueueFamilyIndex = context.graphicsQueueFamily();
        pixel_barrier.buffer = buffers_.pixel_state.deviceBuffer.buffer;
        pixel_barrier.size = buffers_.pixel_state.deviceBuffer.size;
        vkCmdPipelineBarrier(compose_->command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             1,
                             &pixel_barrier,
                             0,
                             nullptr);
        imageBarrier(compose_->command_buffer,
                     output_image_.image,
                     output_layout_,
                     VK_IMAGE_LAYOUT_GENERAL,
                     VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(compose_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compose_->pipeline);
        vkCmdBindDescriptorSets(compose_->command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                compose_->pipeline_layout,
                                0,
                                1,
                                &compose_->descriptor_set,
                                0,
                                nullptr);
        ComposePushConstants push{
            .width = uniforms.image_width,
            .height = uniforms.image_height,
            .background = glm::vec4(background, 1.0f),
        };
        vkCmdPushConstants(compose_->command_buffer,
                           compose_->pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(push),
                           &push);
        vkCmdDispatch(compose_->command_buffer,
                      _CEIL_DIV(uniforms.image_width, 16),
                      _CEIL_DIV(uniforms.image_height, 16),
                      1);
        imageBarrier(compose_->command_buffer,
                     output_image_.image,
                     VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        if (auto ok = submitAndWait(context, compose_->command_buffer, compose_->fence); !ok) {
            return ok;
        }
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
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat viewport path requires Vulkan external memory interop");
        }

        const int active_sh_degree = std::clamp(request.sh_degree, 0, std::min(3, splat_data.get_max_sh_degree()));
        if (auto ok = ensureInitialized(context); !ok) {
            return std::unexpected(ok.error());
        }
        if (force_input_upload || !inputsResident(splat_data)) {
            if (auto ok = uploadInputs(context, splat_data, active_sh_degree); !ok) {
                return std::unexpected(ok.error());
            }
        }
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
        if (request.scene.model_transforms && request.scene.model_transforms->size() == 1) {
            row_major_view = multiplyRowMajorMat4(row_major_view, rowMajorMat4(request.scene.model_transforms->front()));
        }
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                uniforms.world_view_transform[4 * row + col] =
                    row_major_view[static_cast<std::size_t>(4 * col + row)];
            }
        }

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
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat forward pass failed: {}", e.what()));
        }

        if (auto ok = composePixelState(context, uniforms, request.frame_view.background_color); !ok) {
            return std::unexpected(ok.error());
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
