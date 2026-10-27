/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/tensor_impl.hpp"
#include "../backend_driver.hpp"
#include "../backend_work_queue.hpp"
#include "../native_export.hpp"
#include "../tensor_vulkan_interop.hpp"
#include "core/assert.hpp"
#include "core/gpu_device_selection.hpp"
#include "core/vulkan_helpers.hpp"
#include "vk_backend_ops.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"
#include "vk_recorder.hpp"
#include <atomic>
#include <charconv>
#include <cstring>
#include <format>

namespace lfs::core::internal {
    namespace {
        class VulkanDriver final : public BackendDriver {
        public:
            void* current_target() const noexcept override { return nullptr; }
            BackendRendererAccess renderer_access() const noexcept override {
#ifdef __APPLE__
                return {true, false, true};
#else
                return {true, false, false};
#endif
            }

            std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_interop(VulkanInteropDevice device) override { return make_vulkan_vulkan_interop(device); }
            SplatPublication publication() const override { return SplatPublication::Shared; }
            std::unique_ptr<BackendWorkQueue> make_work_queue(const BackendQueueCreateInfo& info) override { return make_vulkan_work_queue(info); }
            std::array<uintptr_t, 2> bind_target(void* target) override {
                const auto previous = reinterpret_cast<uintptr_t>(getCurrentCUDAStream());
                if (target) {
                    auto context = acquire_vulkan_context();
                    const auto id = reinterpret_cast<uint64_t>(target);
                    if (!context->recorders().owns_queue(id))
                        throw std::invalid_argument("Tensor execution target is not a Vulkan queue");
                    context->recorders().bind_queue(id);
                    return {previous, 1};
                }
                setCurrentCUDAStream(nullptr);
                return {previous, 0};
            }
            void unbind_target(std::array<uintptr_t, 2> token) noexcept override {
                if (token[1]) {
                    if (auto context = try_live_vulkan_context())
                        context->recorders().unbind_queue();
                } else
                    setCurrentCUDAStream(reinterpret_cast<cudaStream_t>(token[0]));
            }
            void bridge_targets(void* consumer, void* producer) override { acquire_vulkan_context()->recorders().bridge_queues(reinterpret_cast<uint64_t>(consumer), reinterpret_cast<uint64_t>(producer)); }
            const char* name() const noexcept override { return "Vulkan"; }
            bool available() override { return vulkan_backend_probe_available(); }
            bool live() noexcept override { return vulkan_backend_live(); }
            [[noreturn]] void throw_unavailable() override {
                if (vulkan_backend_lost()) {
                    throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
                        .code = lfs::ErrorCode::DeviceLost,
                        .domain = lfs::ErrorDomain::Vulkan,
                        .user_message = "Vulkan tensor backend device lost; shut the backend down to create a new context",
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                    }));
                }
                throw TensorError(std::format("GPU backend '{}' is unavailable", name()));
            }
            int device_count() override { return vulkan_device_count(); }
            int selected_device() override { return 0; }
            void bind_thread() override {}
            void set_selection(int) noexcept override {}
            std::optional<std::string> validate_options(const TensorBackendOptions&, int& ordinal) override {
                ordinal = -1;
                return {};
            }
            GpuBackendOps& activate() override {
                static VulkanBackendOps* const ops = new VulkanBackendOps();
                return *ops;
            }
            MemoryInfo memory_info(bool) override { return activate().stats(); }
            size_t allocation_bytes(size_t) override { throw std::invalid_argument("Allocation sizes are unknown on the Vulkan backend"); }
            TensorExportDevice dlpack_device(const Tensor& tensor) override { return vulkan_dlpack_device(tensor); }
            Tensor prepare_dlpack_export(const Tensor& tensor, std::optional<int64_t> consumer) override { return vulkan_prepare_dlpack_export(tensor, consumer); }
            BackendExportCapabilities export_capabilities() const noexcept override { return vulkan_dlpack_capabilities(); }
            std::vector<VulkanHardwareDevice> enumerate_hardware() override {

                std::vector<VulkanHardwareDevice> result;
                VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
                app.pApplicationName = "LichtFeld hardware discovery";
                app.apiVersion = VK_API_VERSION_1_3;
                VkInstance instance = VK_NULL_HANDLE;
                if (create_vulkan_instance(app, {}, {}, nullptr, 0, &instance) != VK_SUCCESS)
                    return result;
                struct InstanceGuard {
                    VkInstance instance;
                    ~InstanceGuard() { vkDestroyInstance(instance, nullptr); }
                } guard{instance};
                const auto devices = enumerate_vulkan_physical_devices(instance);
                if (devices.count_result != VK_SUCCESS || devices.devices_result != VK_SUCCESS)
                    return result;
                for (uint32_t index = 0; index < devices.devices.size(); ++index) {
                    VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
                    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                    props.pNext = &ids;
                    vkGetPhysicalDeviceProperties2(devices.devices[index], &props);
                    std::string uuid;
                    constexpr char hex[] = "0123456789abcdef";
                    for (const auto byte : ids.deviceUUID) {
                        uuid += hex[byte >> 4];
                        uuid += hex[byte & 15];
                    }
                    auto tensor = check_vulkan_feature_requirements(devices.devices[index]);
                    auto rendering = check_vulkan_feature_requirements(
                        devices.devices[index], {.viewer_shaders = true, .window_renderer = true});
                    if (props.properties.apiVersion < VK_API_VERSION_1_3) {
                        tensor.missing = "Vulkan 1.3 is required";
                        rendering.missing = tensor.missing;
                    }
                    result.push_back({props.properties.deviceName, std::move(uuid),
                                      std::move(tensor.missing), std::move(rendering.missing), index,
                                      props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU});
                    auto& entry = result.back();
                    entry.vendor_id = props.properties.vendorID;
                    entry.software = props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;

                    entry.shared_memory = props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
                    VkPhysicalDeviceMemoryProperties memory{};
                    vkGetPhysicalDeviceMemoryProperties(devices.devices[index], &memory);
                    for (uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap)
                        if (memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                            entry.total_memory_bytes += memory.memoryHeaps[heap].size;
                }
                return result;
            }
            std::optional<GpuDeviceInfo> device_info(int ordinal) override {
                if (ordinal > 0)
                    return std::nullopt;

                const auto context = internal::try_live_vulkan_context();
                if (!context || context->dead()) {
                    return std::nullopt;
                }
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(context->physical_device(), &properties);
                GpuDeviceInfo result{.name = properties.deviceName};
                VkPhysicalDeviceIDProperties identity{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
                VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                properties2.pNext = &identity;
                vkGetPhysicalDeviceProperties2(context->physical_device(), &properties2);
                std::copy_n(identity.deviceUUID, result.uuid.size(), result.uuid.begin());
                const auto& memory = context->memory_properties();
                for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
                    if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                        result.total_memory_bytes += memory.memoryHeaps[i].size;
                    }
                }
                // Process-budget support belongs to the physical device, independently
                // of the allocator's enabled budget policy.
                static std::mutex capability_mutex;
                static uint64_t capability_context = 0;
                static bool supports_budget = false;
                {
                    std::lock_guard lock(capability_mutex);
                    if (capability_context != context->context_id()) {
                        uint32_t count = 0;
                        if (vkEnumerateDeviceExtensionProperties(context->physical_device(), nullptr, &count, nullptr) != VK_SUCCESS)
                            return result;
                        std::vector<VkExtensionProperties> extensions(count);
                        if (vkEnumerateDeviceExtensionProperties(context->physical_device(), nullptr, &count, extensions.data()) != VK_SUCCESS)
                            return result;
                        supports_budget = std::any_of(extensions.begin(), extensions.begin() + count, [](const auto& extension) {
                            return std::strcmp(extension.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
                        });
                        capability_context = context->context_id();
                    }
                    result.supports_process_memory_budget = supports_budget;
                }
                if (result.supports_process_memory_budget) {
                    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
                    VkPhysicalDeviceMemoryProperties2 memory_info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
                    memory_info.pNext = &budget;
                    vkGetPhysicalDeviceMemoryProperties2(context->physical_device(), &memory_info);
                    for (uint32_t i = 0; i < memory_info.memoryProperties.memoryHeapCount; ++i) {
                        if ((memory_info.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                            result.process_memory_budget_bytes += budget.heapBudget[i];
                            result.process_memory_used_bytes += budget.heapUsage[i];
                        }
                    }
                }
                return result;
            }
            lfs::Status shutdown() override {

                try {
                    internal::backend_ops(GpuBackend::Vulkan).shutdown();
                } catch (...) {
                    return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                        .code = lfs::ErrorCode::Internal,
                        .domain = lfs::ErrorDomain::Vulkan,
                        .user_message = "Vulkan tensor backend shutdown failed",
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                    }));
                }
                return {};
            }
        };
    } // namespace
    void register_vulkan_backend(BackendRegistry& registry) {
        static VulkanDriver driver;
        registry[static_cast<size_t>(GpuBackend::Vulkan)] = &driver;
    }
} // namespace lfs::core::internal
