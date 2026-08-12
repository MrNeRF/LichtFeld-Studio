/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_runtime.hpp"

#include "conv.comp.spv.h"
#include "conv_1x1.comp.spv.h"
#include "conv_transpose.comp.spv.h"
#include "conv_transpose_tiled.comp.spv.h"
#include "conv_tiled.comp.spv.h"
#include "elementwise.comp.spv.h"
#include "layer_norm.comp.spv.h"
#include "matmul.comp.spv.h"
#include "matmul_small_k.comp.spv.h"
#include "matmul_tiled.comp.spv.h"
#include "reduce.comp.spv.h"
#include "reduce_serial.comp.spv.h"
#include "softmax.comp.spv.h"
#include "transform.comp.spv.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <optional>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace lfs::onnx_vulkan::detail {
    namespace {
        constexpr VkDeviceSize kUploadChunkBytes = 64ull * 1024ull * 1024ull;

        [[nodiscard]] Error vk_error(const std::string_view operation, const VkResult result) {
            return {ErrorCode::VulkanFailure,
                    std::string(operation) + " failed with VkResult " + std::to_string(result)};
        }

        [[nodiscard]] Error unavailable(std::string message) {
            return {ErrorCode::VulkanUnavailable, std::move(message)};
        }

        [[nodiscard]] fs::path default_cache_path(const VkPhysicalDeviceProperties& properties) {
            fs::path root;
#ifdef _WIN32
            if (const char* local = std::getenv("LOCALAPPDATA"); local && local[0])
                root = local;
#else
            if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0])
                root = xdg;
            else if (const char* home = std::getenv("HOME"); home && home[0])
                root = fs::path(home) / ".cache";
#endif
            if (root.empty())
                return {};
            return root / "lichtfeld" / "onnx-vulkan" /
                   (std::to_string(properties.vendorID) + "-" +
                    std::to_string(properties.deviceID) + "-" +
                    std::to_string(properties.driverVersion) + ".bin");
        }

        struct Candidate {
            VkPhysicalDevice device = VK_NULL_HANDLE;
            VkPhysicalDeviceProperties properties{};
            std::uint32_t queue_family = 0;
            int score = 0;
        };

        template <std::size_t N>
        [[nodiscard]] std::span<const std::uint32_t> words(const std::uint32_t (&value)[N]) {
            return value;
        }
    } // namespace

    Buffer::Buffer(Buffer&& other) noexcept
        : device(std::exchange(other.device, VK_NULL_HANDLE)),
          buffer(std::exchange(other.buffer, VK_NULL_HANDLE)),
          memory(std::exchange(other.memory, VK_NULL_HANDLE)),
          size(std::exchange(other.size, 0)),
          mapped(std::exchange(other.mapped, nullptr)) {}

    Buffer& Buffer::operator=(Buffer&& other) noexcept {
        if (this == &other)
            return *this;
        this->~Buffer();
        new (this) Buffer(std::move(other));
        return *this;
    }

    Buffer::~Buffer() {
        if (mapped && device && memory)
            vkUnmapMemory(device, memory);
        if (buffer && device)
            vkDestroyBuffer(device, buffer, nullptr);
        if (memory && device)
            vkFreeMemory(device, memory, nullptr);
    }

    VulkanRuntime::~VulkanRuntime() {
        if (device_)
            vkDeviceWaitIdle(device_);
        save_pipeline_cache();
        for (const auto pipeline : pipelines_)
            if (pipeline)
                vkDestroyPipeline(device_, pipeline, nullptr);
        if (pipeline_cache_)
            vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
        if (pipeline_layout_)
            vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (descriptor_layout_)
            vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
        if (command_pool_)
            vkDestroyCommandPool(device_, command_pool_, nullptr);
        if (device_)
            vkDestroyDevice(device_, nullptr);
        if (instance_)
            vkDestroyInstance(instance_, nullptr);
    }

    std::expected<std::unique_ptr<VulkanRuntime>, Error>
    VulkanRuntime::create(const SessionOptions& options) {
        auto runtime = std::unique_ptr<VulkanRuntime>(new VulkanRuntime);
        if (auto initialized = runtime->initialize(options); !initialized)
            return std::unexpected(initialized.error());
        return runtime;
    }

    std::expected<void, Error> VulkanRuntime::initialize(const SessionOptions& options) {
        std::uint32_t loader_version = VK_API_VERSION_1_0;
        if (const auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
                vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion")))
            enumerate_version(&loader_version);
        if (loader_version < VK_API_VERSION_1_1)
            return std::unexpected(unavailable("Vulkan 1.1 or newer is required"));
        const VkApplicationInfo application{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "lfs_onnx_vulkan",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "lfs_onnx_vulkan",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_1,
        };
        const VkInstanceCreateInfo instance_info{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &application,
        };
        if (const auto result = vkCreateInstance(&instance_info, nullptr, &instance_); result != VK_SUCCESS)
            return std::unexpected(vk_error("vkCreateInstance", result));

        std::uint32_t physical_count = 0;
        if (const auto result = vkEnumeratePhysicalDevices(instance_, &physical_count, nullptr); result != VK_SUCCESS)
            return std::unexpected(vk_error("vkEnumeratePhysicalDevices", result));
        if (physical_count == 0)
            return std::unexpected(unavailable("no Vulkan physical device is available"));
        std::vector<VkPhysicalDevice> physical_devices(physical_count);
        if (const auto result = vkEnumeratePhysicalDevices(instance_, &physical_count, physical_devices.data());
            result != VK_SUCCESS)
            return std::unexpected(vk_error("vkEnumeratePhysicalDevices", result));

        std::vector<Candidate> candidates;
        for (const auto physical : physical_devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(physical, &properties);
            if (VK_API_VERSION_MAJOR(properties.apiVersion) < 1 ||
                (VK_API_VERSION_MAJOR(properties.apiVersion) == 1 &&
                 VK_API_VERSION_MINOR(properties.apiVersion) < 1))
                continue;
            std::uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
            for (std::uint32_t family = 0; family < family_count; ++family) {
                if ((families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0 || families[family].queueCount == 0)
                    continue;
                int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2000 :
                            properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1000 : 100;
                if ((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0)
                    score += 100;
                candidates.push_back({physical, properties, family, score});
                break;
            }
        }
        if (candidates.empty())
            return std::unexpected(unavailable("no Vulkan 1.1 compute device is available"));
        std::ranges::sort(candidates, [](const Candidate& lhs, const Candidate& rhs) {
            return lhs.score > rhs.score;
        });
        const auto selected = options.vulkan_device.value_or(0);
        if (selected >= candidates.size())
            return std::unexpected(unavailable("Vulkan device index " + std::to_string(selected) +
                                               " is out of range; " + std::to_string(candidates.size()) +
                                               " compute device(s) are available"));
        const auto& candidate = candidates[selected];
        physical_device_ = candidate.device;
        queue_family_ = candidate.queue_family;
        device_name_ = candidate.properties.deviceName;
        storage_alignment_ = std::max<VkDeviceSize>(1, candidate.properties.limits.minStorageBufferOffsetAlignment);
        maximum_storage_range_ = candidate.properties.limits.maxStorageBufferRange;
        maximum_group_count_x_ = candidate.properties.limits.maxComputeWorkGroupCount[0];
        maximum_group_count_y_ = candidate.properties.limits.maxComputeWorkGroupCount[1];
        maximum_group_count_z_ = candidate.properties.limits.maxComputeWorkGroupCount[2];
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);
        if (candidate.properties.limits.maxComputeWorkGroupInvocations < 256 ||
            candidate.properties.limits.maxComputeWorkGroupSize[0] < 256 ||
            candidate.properties.limits.maxComputeWorkGroupSize[1] < 16)
            return std::unexpected(unavailable(
                "Vulkan device does not support the 256-invocation FP32 workgroups required by MoGe"));

        constexpr float priority = 1.0f;
        const VkDeviceQueueCreateInfo queue_info{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queue_family_,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
        const VkDeviceCreateInfo device_info{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_info,
        };
        if (const auto result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_); result != VK_SUCCESS)
            return std::unexpected(vk_error("vkCreateDevice", result));
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

        const VkCommandPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family_,
        };
        if (const auto result = vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_); result != VK_SUCCESS)
            return std::unexpected(vk_error("vkCreateCommandPool", result));

        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index] = {.binding = index,
                               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               .descriptorCount = 1,
                               .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
        }
        const VkDescriptorSetLayoutCreateInfo descriptor_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        };
        if (const auto result = vkCreateDescriptorSetLayout(device_, &descriptor_info, nullptr, &descriptor_layout_);
            result != VK_SUCCESS)
            return std::unexpected(vk_error("vkCreateDescriptorSetLayout", result));
        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_layout_,
        };
        if (const auto result = vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout_);
            result != VK_SUCCESS)
            return std::unexpected(vk_error("vkCreatePipelineLayout", result));

        pipeline_cache_path_ = options.pipeline_cache_path.empty()
                                   ? default_cache_path(candidate.properties)
                                   : options.pipeline_cache_path;
        std::vector<std::byte> cache_data;
        if (!pipeline_cache_path_.empty()) {
            std::error_code ec;
            const auto cache_size = fs::file_size(pipeline_cache_path_, ec);
            if (!ec && cache_size <= 64ull * 1024ull * 1024ull) {
                cache_data.resize(static_cast<std::size_t>(cache_size));
                std::ifstream stream(pipeline_cache_path_, std::ios::binary);
                stream.read(reinterpret_cast<char*>(cache_data.data()), static_cast<std::streamsize>(cache_data.size()));
                if (!stream)
                    cache_data.clear();
            }
        }
        VkPipelineCacheCreateInfo cache_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .initialDataSize = cache_data.size(),
            .pInitialData = cache_data.data(),
        };
        auto cache_result = vkCreatePipelineCache(device_, &cache_info, nullptr, &pipeline_cache_);
        if (cache_result != VK_SUCCESS && !cache_data.empty()) {
            cache_info.initialDataSize = 0;
            cache_info.pInitialData = nullptr;
            cache_result = vkCreatePipelineCache(device_, &cache_info, nullptr, &pipeline_cache_);
        }
        if (cache_result != VK_SUCCESS)
            return std::unexpected(vk_error("vkCreatePipelineCache", cache_result));
        return create_pipelines();
    }

    std::expected<void, Error> VulkanRuntime::create_pipelines() {
        const std::array<std::span<const std::uint32_t>, static_cast<std::size_t>(Kernel::Count)> code{
            words(shaders::kElementwiseSpv), words(shaders::kTransformSpv), words(shaders::kMatMulSpv),
            words(shaders::kMatMulTiledSpv), words(shaders::kMatMulSmallKSpv),
            words(shaders::kConvSpv), words(shaders::kConv1x1Spv),
            words(shaders::kConvTiledSpv), words(shaders::kConvTransposeSpv),
            words(shaders::kConvTransposeTiledSpv), words(shaders::kReduceSpv),
            words(shaders::kReduceSerialSpv), words(shaders::kSoftmaxSpv),
            words(shaders::kLayerNormSpv),
        };
        for (std::size_t index = 0; index < code.size(); ++index) {
            const VkShaderModuleCreateInfo module_info{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = code[index].size_bytes(),
                .pCode = code[index].data(),
            };
            VkShaderModule module = VK_NULL_HANDLE;
            if (const auto result = vkCreateShaderModule(device_, &module_info, nullptr, &module); result != VK_SUCCESS)
                return std::unexpected(vk_error("vkCreateShaderModule", result));
            const VkPipelineShaderStageCreateInfo stage{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "main",
            };
            const VkComputePipelineCreateInfo pipeline_info{
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage = stage,
                .layout = pipeline_layout_,
            };
            const auto result = vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_info, nullptr,
                                                         &pipelines_[index]);
            vkDestroyShaderModule(device_, module, nullptr);
            if (result != VK_SUCCESS)
                return std::unexpected(vk_error("vkCreateComputePipelines", result));
        }
        return {};
    }

    std::expected<std::uint32_t, Error>
    VulkanRuntime::memory_type(const std::uint32_t bits,
                               const VkMemoryPropertyFlags required,
                               const VkMemoryPropertyFlags preferred) const {
        std::optional<std::uint32_t> fallback;
        for (std::uint32_t index = 0; index < memory_properties_.memoryTypeCount; ++index) {
            if ((bits & (1u << index)) == 0)
                continue;
            const auto flags = memory_properties_.memoryTypes[index].propertyFlags;
            if ((flags & required) != required)
                continue;
            if ((flags & preferred) == preferred)
                return index;
            if (!fallback)
                fallback = index;
        }
        if (fallback)
            return *fallback;
        return std::unexpected(unavailable("no Vulkan memory type satisfies the required properties"));
    }

    std::expected<Buffer, Error>
    VulkanRuntime::create_buffer(const VkDeviceSize requested_size,
                                 const VkBufferUsageFlags usage,
                                 const VkMemoryPropertyFlags required,
                                 const VkMemoryPropertyFlags preferred) const {
        Buffer result;
        result.device = device_;
        result.size = std::max<VkDeviceSize>(requested_size, 4);
        const VkBufferCreateInfo buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = result.size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        if (const auto status = vkCreateBuffer(device_, &buffer_info, nullptr, &result.buffer); status != VK_SUCCESS)
            return std::unexpected(vk_error("vkCreateBuffer", status));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, result.buffer, &requirements);
        auto type = memory_type(requirements.memoryTypeBits, required, preferred);
        if (!type)
            return std::unexpected(type.error());
        const VkMemoryAllocateInfo allocation{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex = *type,
        };
        if (const auto status = vkAllocateMemory(device_, &allocation, nullptr, &result.memory); status != VK_SUCCESS)
            return std::unexpected(vk_error("vkAllocateMemory", status));
        if (const auto status = vkBindBufferMemory(device_, result.buffer, result.memory, 0); status != VK_SUCCESS)
            return std::unexpected(vk_error("vkBindBufferMemory", status));
        if ((required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
            if (const auto status = vkMapMemory(device_, result.memory, 0, result.size, 0, &result.mapped);
                status != VK_SUCCESS)
                return std::unexpected(vk_error("vkMapMemory", status));
        }
        return result;
    }

    std::expected<void, Error>
    VulkanRuntime::immediate_submit(const std::function<void(VkCommandBuffer)>& record) const {
        const VkCommandBufferAllocateInfo allocation{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer command = VK_NULL_HANDLE;
        if (const auto result = vkAllocateCommandBuffers(device_, &allocation, &command); result != VK_SUCCESS)
            return std::unexpected(vk_error("vkAllocateCommandBuffers", result));
        const VkCommandBufferBeginInfo begin{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        auto result = vkBeginCommandBuffer(command, &begin);
        if (result == VK_SUCCESS) {
            record(command);
            result = vkEndCommandBuffer(command);
        }
        if (result == VK_SUCCESS) {
            const VkSubmitInfo submit{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &command,
            };
            result = vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE);
        }
        if (result == VK_SUCCESS)
            result = vkQueueWaitIdle(queue_);
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
        if (result != VK_SUCCESS)
            return std::unexpected(vk_error("immediate Vulkan submission", result));
        return {};
    }

    std::expected<void, Error>
    VulkanRuntime::upload(Buffer& destination,
                          const VkDeviceSize destination_offset,
                          const std::span<const std::byte> bytes) const {
        if (destination_offset > destination.size || bytes.size() > destination.size - destination_offset)
            return std::unexpected(Error{ErrorCode::VulkanFailure, "buffer upload exceeds destination range"});
        if (bytes.empty())
            return {};
        const auto staging_size = std::min<VkDeviceSize>(bytes.size(), kUploadChunkBytes);
        auto staging = create_buffer(staging_size,
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (!staging)
            return std::unexpected(staging.error());
        for (VkDeviceSize copied = 0; copied < bytes.size();) {
            const auto count = std::min<VkDeviceSize>(staging->size, bytes.size() - copied);
            std::memcpy(staging->mapped, bytes.data() + copied, static_cast<std::size_t>(count));
            auto submitted = immediate_submit([&](const VkCommandBuffer command) {
                const VkBufferCopy region{.srcOffset = 0,
                                          .dstOffset = destination_offset + copied,
                                          .size = count};
                vkCmdCopyBuffer(command, staging->buffer, destination.buffer, 1, &region);
            });
            if (!submitted)
                return std::unexpected(submitted.error());
            copied += count;
        }
        return {};
    }

    std::expected<VkDescriptorPool, Error>
    VulkanRuntime::create_descriptor_pool(const std::uint32_t max_sets) const {
        const VkDescriptorPoolSize size{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = max_sets * 5,
        };
        const VkDescriptorPoolCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = max_sets,
            .poolSizeCount = 1,
            .pPoolSizes = &size,
        };
        VkDescriptorPool pool = VK_NULL_HANDLE;
        if (const auto result = vkCreateDescriptorPool(device_, &info, nullptr, &pool); result != VK_SUCCESS)
            return std::unexpected(vk_error("vkCreateDescriptorPool", result));
        return pool;
    }

    std::expected<VkDescriptorSet, Error>
    VulkanRuntime::allocate_descriptor_set(const VkDescriptorPool pool) const {
        const VkDescriptorSetAllocateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_layout_,
        };
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (const auto result = vkAllocateDescriptorSets(device_, &info, &set); result != VK_SUCCESS)
            return std::unexpected(vk_error("vkAllocateDescriptorSets", result));
        return set;
    }

    void VulkanRuntime::update_descriptor_set(const VkDescriptorSet set,
                                               const std::array<BufferBinding, 5>& bindings) const {
        std::array<VkDescriptorBufferInfo, 5> infos{};
        std::array<VkWriteDescriptorSet, 5> writes{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            infos[index] = {.buffer = bindings[index].buffer,
                            .offset = bindings[index].offset,
                            .range = std::max<VkDeviceSize>(4, bindings[index].range)};
            writes[index] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = set,
                             .dstBinding = index,
                             .descriptorCount = 1,
                             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .pBufferInfo = &infos[index]};
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void VulkanRuntime::bind_and_dispatch(const VkCommandBuffer command,
                                          const Kernel kernel,
                                          const VkDescriptorSet descriptor_set,
                                          const std::uint64_t invocation_count,
                                          const std::array<std::uint32_t, 3> workgroups) const {
        if (invocation_count == 0)
            return;
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[static_cast<std::size_t>(kernel)]);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                                &descriptor_set, 0, nullptr);
        if (workgroups[0] != 0) {
            vkCmdDispatch(command, workgroups[0], workgroups[1], workgroups[2]);
            return;
        }
        const std::uint32_t local_size = kernel == Kernel::Elementwise || kernel == Kernel::Transform
                                             ? 256
                                             : 64;
        const auto group_count = (invocation_count + local_size - 1) / local_size;
        const auto groups_x = static_cast<std::uint32_t>(std::min<std::uint64_t>(group_count, maximum_group_count_x_));
        const auto groups_y64 = (group_count + groups_x - 1) / groups_x;
        const auto groups_y = static_cast<std::uint32_t>(std::min<std::uint64_t>(groups_y64, maximum_group_count_y_));
        vkCmdDispatch(command, groups_x, groups_y, 1);
    }

    void VulkanRuntime::save_pipeline_cache() const noexcept {
        if (!device_ || !pipeline_cache_ || pipeline_cache_path_.empty())
            return;
        std::size_t size = 0;
        if (vkGetPipelineCacheData(device_, pipeline_cache_, &size, nullptr) != VK_SUCCESS || size == 0)
            return;
        std::vector<std::byte> bytes(size);
        if (vkGetPipelineCacheData(device_, pipeline_cache_, &size, bytes.data()) != VK_SUCCESS)
            return;
        std::error_code ec;
        fs::create_directories(pipeline_cache_path_.parent_path(), ec);
        if (ec)
            return;
        std::ofstream stream(pipeline_cache_path_, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

} // namespace lfs::onnx_vulkan::detail
