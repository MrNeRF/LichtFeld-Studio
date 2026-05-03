#include "gs_pipeline.h"
#include "perf_timer.h"

#include <fstream>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

static const size_t MAX_UNIFORM_SIZE = 192;

static const uint32_t MAX_TIMESTAMP_QUERY_COUNT = 48;

#if defined(__SSE2__) || defined(_MSC_VER)
#define SSE2_AVAILABLE 1
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

std::vector<uint32_t> loadSpirv(std::string spirv_path) {
// Load the SPIR-V file
#ifdef WIN32
    // replace "/" with "\\"
    size_t start_pos = 0;
    while ((start_pos = spirv_path.find("/", start_pos)) != std::string::npos) {
        spirv_path.replace(start_pos, 1, "\\");
        start_pos += 1;
    }
#endif

    std::ifstream file(spirv_path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open file: " + spirv_path);

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> spirv_code(fileSize / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char*>(spirv_code.data()), fileSize))
        throw std::runtime_error("Failed to read file: " + spirv_path);

    return spirv_code;
}

VulkanGSPipeline::VulkanGSPipeline() : instance(VK_NULL_HANDLE),
                                       physical_device(VK_NULL_HANDLE),
                                       device(VK_NULL_HANDLE),
                                       command_queue(VK_NULL_HANDLE),
                                       command_pool(VK_NULL_HANDLE),
                                       command_buffer(VK_NULL_HANDLE),
                                       fence(VK_NULL_HANDLE),
                                       timestamp_query_pool(VK_NULL_HANDLE),
                                       queue_family_index(UINT32_MAX) {
}

VulkanGSPipeline::~VulkanGSPipeline() {
    if (commandBatchInProgress)
        endCommandBatch(false);
    cleanup();
}

void VulkanGSPipeline::initialize(int device_id) {
    owns_instance = true;
    owns_device = true;

#if ENABLE_VULKAN_VALIDATION_LAYER
    do {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        enableValidationLayer = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp("VK_LAYER_KHRONOS_validation", layerProperties.layerName) == 0) {
                enableValidationLayer = true;
                break;
            }
        }

        if (!enableValidationLayer)
            fprintf(stderr, "WARNING: Vulkan validation layer not available");
    } while (0);
#endif

    createInstance();
    selectPhysicalDevice(device_id);
    createDevice();
    createCommandPool();
    createFence();
    createQueryPools();

    commandBatchInProgress = false;
}

void VulkanGSPipeline::initializeExternal(VkInstance external_instance,
                                          VkPhysicalDevice external_physical_device,
                                          VkDevice external_device,
                                          VkQueue external_queue,
                                          uint32_t external_queue_family_index,
                                          VmaAllocator external_allocator) {
    cleanup();
    if (external_instance == VK_NULL_HANDLE ||
        external_physical_device == VK_NULL_HANDLE ||
        external_device == VK_NULL_HANDLE ||
        external_queue == VK_NULL_HANDLE ||
        external_queue_family_index == UINT32_MAX ||
        external_allocator == VK_NULL_HANDLE) {
        _THROW_ERROR("initializeExternal received an invalid Vulkan handle");
    }

    instance = external_instance;
    physical_device = external_physical_device;
    device = external_device;
    command_queue = external_queue;
    queue_family_index = external_queue_family_index;
    allocator = external_allocator;
    owns_instance = false;
    owns_device = false;
    owns_allocator = false;

    populateDeviceInfo(physical_device);
    createCommandPool();
    createFence();
    createQueryPools();

    commandBatchInProgress = false;
}

void VulkanGSPipeline::cleanupBuffers(VulkanGSPipelineBuffers& buffers) {
    HOST_GUARD;
#define _(name)                                   \
    {                                             \
        destroyBuffer(buffers.name.deviceBuffer); \
        buffers.name.clear();                     \
        buffers.name.shrink_to_fit();             \
    }
    _(xyz_ws)
    _(sh_coeffs)
    _(rotations)
    _(scales_opacs)
        _(tiles_touched) _(rect_tile_space) _(radii) _(xy_vs) _(depths) _(inv_cov_vs_opacity) _(rgb)
            _(index_buffer_offset) _(sorting_keys_1) _(sorting_keys_2) _(sorting_gauss_idx_1) _(sorting_gauss_idx_2) _(tile_ranges)
                _(pixel_state) _(n_contributors)
                    _(_cumsum_blockSums) _(_cumsum_blockSums2) _(_sorting_histogram) _(_sorting_histogram_cumsum)
#undef _
}

void VulkanGSPipeline::cleanup() {
    HOST_GUARD;

    if (stager.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, stager.buffer, stager.allocation);
        stager.buffer = VK_NULL_HANDLE;
        stager.allocation = VK_NULL_HANDLE;
        stager.allocSize = 0;
    }

    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);

        for (_ComputePipeline* pipeline : all_compute_pipelines)
            destroyComputePipeline(*pipeline);
        all_compute_pipelines.clear();

        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, fence, nullptr);
            fence = VK_NULL_HANDLE;
        }
        if (timestamp_query_pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, timestamp_query_pool, nullptr);
            timestamp_query_pool = VK_NULL_HANDLE;
        }

        if (command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
            command_buffer = VK_NULL_HANDLE;
        }
        if (command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, command_pool, nullptr);
            command_pool = VK_NULL_HANDLE;
        }

        if (owns_allocator && allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator);
        }
        allocator = VK_NULL_HANDLE;

        if (owns_device) {
            vkDestroyDevice(device, nullptr);
        }
        device = VK_NULL_HANDLE;
    }

    if (instance != VK_NULL_HANDLE) {
        if (owns_instance) {
            vkDestroyInstance(instance, nullptr);
        }
        instance = VK_NULL_HANDLE;
    }
    physical_device = VK_NULL_HANDLE;
    command_queue = VK_NULL_HANDLE;
    queue_family_index = UINT32_MAX;
    owns_instance = true;
    owns_device = true;
    owns_allocator = true;
}

void VulkanGSPipeline::createInstance() {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = VK_NULL_HANDLE;
    app_info.pApplicationName = "Vulkan Gaussian Rasterization";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pNext = VK_NULL_HANDLE;
    create_info.pApplicationInfo = &app_info;

    if (enableValidationLayer) {
        static const char* validation_layer_name = "VK_LAYER_KHRONOS_validation";
        create_info.enabledLayerCount = 1u;
        create_info.ppEnabledLayerNames = &validation_layer_name;
    }

    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS)
        _THROW_ERROR("Failed to create instance");
}

void VulkanGSPipeline::selectPhysicalDevice(int device_id) {
    static constexpr int kANSIDefault = 0;
    static constexpr int kANSIRed = 91;
    static constexpr int kANSIGreen = 32;
    static constexpr int kANSIOrange = 93;

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
        const char* message = "Could not find any physical device.";
        printf("\033[%dm%s\033[m\n", kANSIRed, message);
        throw std::runtime_error(message);
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    struct SelectedDevice {
        int idx = -1;
        VkPhysicalDevice device = VK_NULL_HANDLE;
        uint32_t queueFamilyIdx = UINT32_MAX;
        DeviceInfo deviceInfo;
    };
    std::vector<SelectedDevice> viableDevices;
    std::vector<SelectedDevice> softViableDevices;

    const auto& [minMaxGroups, minMaxThreads, minSharedMemory] = getDeviceRequirement();

    printf(
        "Device Requirement: subgroup>=%d, maxGroups>=[%u %u %u], maxThreads>=[%u %u %u], maxShared>=%u, I16|I64|F32Atomic \n",
        (int)SUBGROUP_SIZE,
        minMaxGroups[0], minMaxGroups[1], minMaxGroups[2],
        minMaxThreads[0], minMaxThreads[1], minMaxThreads[2],
        minSharedMemory);
    fflush(stdout);

    for (size_t i = 0; i < devices.size(); i++) {
        VkPhysicalDevice& device = devices[i];

        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        auto& limits = deviceProperties.limits;
        uint32_t maxGroupsX = limits.maxComputeWorkGroupCount[0];
        uint32_t maxGroupsY = limits.maxComputeWorkGroupCount[1];
        uint32_t maxGroupsZ = limits.maxComputeWorkGroupCount[2];
        uint32_t maxThreadsX = limits.maxComputeWorkGroupSize[0];
        uint32_t maxThreadsY = limits.maxComputeWorkGroupSize[1];
        uint32_t maxThreadsZ = limits.maxComputeWorkGroupSize[2];
        bool validGroupCount[4] = {
            maxGroupsX >= minMaxGroups[0],
            maxGroupsY >= minMaxGroups[1],
            maxGroupsZ >= minMaxGroups[2],
        };
        validGroupCount[3] = (validGroupCount[0] && validGroupCount[1] && validGroupCount[2]);
        bool validGroupSize[4] = {
            maxThreadsX >= minMaxThreads[0],
            maxThreadsY >= minMaxThreads[1],
            maxThreadsZ >= minMaxThreads[2],
        };
        validGroupSize[3] = (validGroupSize[0] && validGroupSize[1] && validGroupSize[2]);
        bool validSharedSize = limits.maxComputeSharedMemorySize >= minSharedMemory;

        // check subgroup size support
        VkPhysicalDeviceSubgroupProperties subgroupProperties{};
        subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        subgroupProperties.pNext = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties2KHR deviceProperties2{};
        deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        deviceProperties2.pNext = &subgroupProperties;
        vkGetPhysicalDeviceProperties2(device, &deviceProperties2);
        bool validSubgroupSize = subgroupProperties.subgroupSize >= SUBGROUP_SIZE;

        // check compute pipeline.support
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
        uint32_t queueFamilyIdx = (uint32_t)(-1);
        for (uint32_t i = 0; i < queue_families.size(); i++) {
            if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                (queue_families[i].timestampValidBits != 0))
                queueFamilyIdx = i;
        }
        bool validQueueFamily = ((int32_t)queueFamilyIdx != -1);

        // check feature support
        bool hasInt16 = true, hasInt64 = true, hasFloat32AtomicAdd = true;
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features{};
        atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        atomic_float_features.pNext = VK_NULL_HANDLE;
        VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.pNext = &atomic_float_features;
        vkGetPhysicalDeviceFeatures2(device, &deviceFeatures2);
        if (deviceFeatures2.features.shaderInt16 == VK_FALSE)
            hasInt16 = false;
        if (deviceFeatures2.features.shaderInt64 == VK_FALSE)
            hasInt64 = false;
        if (atomic_float_features.shaderBufferFloat32AtomicAdd == VK_FALSE)
            hasFloat32AtomicAdd = false;

        DeviceVendor vendor = DeviceVendor::Unknown;
        if (deviceProperties.vendorID == 0x10DE)
            vendor = DeviceVendor::NVIDIA;
        else if (deviceProperties.vendorID == 0x1002 || deviceProperties.vendorID == 0x1022)
            vendor = DeviceVendor::AMD;
        else if (deviceProperties.vendorID == 0x8086)
            vendor = DeviceVendor::Intel_R_;
        else if (deviceProperties.vendorID == 0x13B5)
            vendor = DeviceVendor::ARM;
        else if (deviceProperties.vendorID == 0x5143)
            vendor = DeviceVendor::Qualcomm;

        bool softViable = validSubgroupSize && validGroupSize[3] && validQueueFamily && hasInt16;
        bool viable = softViable && validGroupCount[3] && validSharedSize && hasInt64 && hasFloat32AtomicAdd;
        SelectedDevice deviceInfo{
            (int)i,
            device,
            queueFamilyIdx,
            {subgroupProperties.subgroupSize,
             limits.maxComputeSharedMemorySize,
             maxGroupsX, maxGroupsY, maxGroupsZ,
             maxThreadsX, maxThreadsY, maxThreadsZ,
             hasInt16, hasInt64, hasFloat32AtomicAdd,
             vendor, deviceProperties.vendorID,
             deviceProperties.deviceName}};
        bool is_excluded_device = (device_id >= 0 && device_id != (int)i);
        if (viable && !is_excluded_device)
            viableDevices.push_back(deviceInfo);
        else if (softViable && !is_excluded_device)
            softViableDevices.push_back(deviceInfo);

        printf(
            "[%d] %s - \033[%dm%s\033[m\n "
            " subgroup=\033[%dm%u\033[m, "
            "maxGroups=[\033[%dm%u\033[m \033[%dm%u\033[m \033[%dm%u\033[m], "
            "maxThreads=[\033[%dm%u\033[m \033[%dm%u\033[m \033[%dm%u\033[m], "
            "maxShared=\033[%dm%u\033[m, "
            "\033[%dmI16\033[m|\033[%dmI64\033[m|\033[%dmF32Atomic\033[m\n",
            (int)i, deviceProperties.deviceName,
            viable ? kANSIGreen : softViable ? kANSIOrange
                                             : kANSIRed,
            viable ? "VIABLE" : softViable ? "POSSIBLY VIABLE"
                                           : "NOT VIABLE",
            validSubgroupSize ? kANSIDefault : kANSIRed, subgroupProperties.subgroupSize,
            validGroupCount[0] ? kANSIDefault : kANSIOrange, maxGroupsX,
            validGroupCount[1] ? kANSIDefault : kANSIOrange, maxGroupsY,
            validGroupCount[2] ? kANSIDefault : kANSIOrange, maxGroupsZ,
            validGroupSize[0] ? kANSIDefault : kANSIRed, maxThreadsX,
            validGroupSize[1] ? kANSIDefault : kANSIRed, maxThreadsY,
            validGroupSize[2] ? kANSIDefault : kANSIRed, maxThreadsZ,
            validSharedSize ? kANSIDefault : kANSIOrange, limits.maxComputeSharedMemorySize,
            hasInt16 ? kANSIDefault : kANSIRed,
            hasInt64 ? kANSIDefault : kANSIOrange,
            hasFloat32AtomicAdd ? kANSIDefault : kANSIOrange);
        if (softViable) {
            if (!hasInt64)
                printf("  \033[%dm%s\033[m\n", kANSIOrange, "WARNING: To use this device, shaders must be compiled with USE_EMULATED_INT64=1.");
            if (!hasFloat32AtomicAdd)
                printf("  \033[%dm%s\033[m\n", kANSIOrange, "WARNING: To use this device, shaders must be compiled with USE_EMULATED_F32_ATOMIC=1.");
            if (!validGroupCount[3])
                printf("  \033[%dm%s\033[m\n", kANSIOrange, "WARNING: This device may not work if you want to train a large scene.");
            if (!validSharedSize)
                printf("  \033[%dm%s\033[m\n", kANSIOrange, "WARNING: This device does not have sufficient shared memory, which can involve undefined behavior.");
        }
        fflush(stdout);
    }

    SelectedDevice device;
    if (!viableDevices.empty())
        device = viableDevices[0];
    else if (!softViableDevices.empty())
        device = softViableDevices[0];
    else {
        const char* message = "Could not find a viable physical device.";
        printf("\033[%dm%s\033[m\n", kANSIRed, message);
        if (device_id >= 0)
            printf("\033[%dmNote: Device [%d] is requested, but it %s.\033[m\n",
                   kANSIOrange, device_id, device_id >= (int)devices.size() ? "does not exist" : "is not viable");
        throw std::runtime_error(message);
    }

    this->physical_device = device.device;
    this->queue_family_index = device.queueFamilyIdx;
    this->deviceInfo = device.deviceInfo;
    printf("Using device [\033[%dm%d\033[m]%s\n",
           viableDevices.empty() ? kANSIOrange : kANSIDefault,
           device.idx,
           viableDevices.empty() ? " (\033[93mPOSSIBLY VIABLE\033[m)" : "");
    if (!deviceInfo.hasFloat32AtomicAdd)
        printf("\033[%dm%s\033[m\n", kANSIOrange, "WARNING: Float32AtomicAdd is not available. Make sure shaders are compiled with USE_EMULATED_F32_ATOMIC=1.");
    if (!deviceInfo.hasInt64)
        printf("\033[%dm%s\033[m\n", kANSIOrange, "WARNING: Int64 is not available. Make sure shaders are compiled with USE_EMULATED_INT64=1.");
    printf("\n");
    fflush(stdout);
}

void VulkanGSPipeline::populateDeviceInfo(VkPhysicalDevice selected_physical_device) {
    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(selected_physical_device, &deviceProperties);
    const auto& limits = deviceProperties.limits;

    VkPhysicalDeviceSubgroupProperties subgroupProperties{};
    subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2KHR deviceProperties2{};
    deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProperties2.pNext = &subgroupProperties;
    vkGetPhysicalDeviceProperties2(selected_physical_device, &deviceProperties2);

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
    atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.pNext = &atomicFloatFeatures;
    vkGetPhysicalDeviceFeatures2(selected_physical_device, &deviceFeatures2);

    DeviceVendor vendor = DeviceVendor::Unknown;
    if (deviceProperties.vendorID == 0x10DE)
        vendor = DeviceVendor::NVIDIA;
    else if (deviceProperties.vendorID == 0x1002 || deviceProperties.vendorID == 0x1022)
        vendor = DeviceVendor::AMD;
    else if (deviceProperties.vendorID == 0x8086)
        vendor = DeviceVendor::Intel_R_;
    else if (deviceProperties.vendorID == 0x13B5)
        vendor = DeviceVendor::ARM;
    else if (deviceProperties.vendorID == 0x5143)
        vendor = DeviceVendor::Qualcomm;

    deviceInfo = {
        subgroupProperties.subgroupSize,
        limits.maxComputeSharedMemorySize,
        limits.maxComputeWorkGroupCount[0],
        limits.maxComputeWorkGroupCount[1],
        limits.maxComputeWorkGroupCount[2],
        limits.maxComputeWorkGroupSize[0],
        limits.maxComputeWorkGroupSize[1],
        limits.maxComputeWorkGroupSize[2],
        deviceFeatures2.features.shaderInt16 == VK_TRUE,
        deviceFeatures2.features.shaderInt64 == VK_TRUE,
        atomicFloatFeatures.shaderBufferFloat32AtomicAdd == VK_TRUE,
        vendor,
        deviceProperties.vendorID,
        deviceProperties.deviceName,
    };
}

std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>>
VulkanGSPipeline::get_device_info() const {
    std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>> result;
    result["subgroup_size"] = deviceInfo.subgroupSize;
    result["shared_size"] = deviceInfo.sharedSize;
    result["max_groups"] = std::vector<uint32_t>{deviceInfo.maxGroupsX, deviceInfo.maxGroupsY, deviceInfo.maxGroupsZ};
    result["max_threads"] = std::vector<uint32_t>{deviceInfo.maxThreadsX, deviceInfo.maxThreadsY, deviceInfo.maxThreadsZ};
    result["has_int16"] = deviceInfo.hasInt16;
    result["has_int64"] = deviceInfo.hasInt64;
    result["has_float32_atomic_add"] = deviceInfo.hasFloat32AtomicAdd;
    result["vendor"] = deviceInfo.vendorId;
    result["name"] = deviceInfo.name;
    return result;
}

void VulkanGSPipeline::createDevice() {
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.pNext = VK_NULL_HANDLE;
    queue_create_info.queueFamilyIndex = queue_family_index;
    queue_create_info.queueCount = 1;

    float queue_priority = 1.0f;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures enabledFeatures = {};
    enabledFeatures.shaderInt16 = VK_TRUE;
    if (deviceInfo.hasInt64)
        enabledFeatures.shaderInt64 = VK_TRUE;

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features = {};
    atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    atomic_float_features.shaderBufferFloat32AtomicAdd = VK_TRUE;
    atomic_float_features.pNext = VK_NULL_HANDLE;

    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeControlFeatures = {};
    subgroupSizeControlFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    subgroupSizeControlFeatures.subgroupSizeControl = VK_TRUE;
    subgroupSizeControlFeatures.computeFullSubgroups = VK_TRUE;
    subgroupSizeControlFeatures.pNext = VK_NULL_HANDLE;
    if (deviceInfo.hasFloat32AtomicAdd)
        subgroupSizeControlFeatures.pNext = &atomic_float_features;

    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.queueCreateInfoCount = 1;
    create_info.pEnabledFeatures = &enabledFeatures;

    std::vector<const char*> device_extensions = {
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
    };
    create_info.enabledExtensionCount = (uint32_t)device_extensions.size();
    create_info.ppEnabledExtensionNames = device_extensions.data();
    create_info.pNext = &subgroupSizeControlFeatures;

    if (vkCreateDevice(physical_device, &create_info, nullptr, &device) != VK_SUCCESS) {
        _THROW_ERROR("Failed to create device");
    }

    vkGetDeviceQueue(device, queue_family_index, 0, &command_queue);
}

void VulkanGSPipeline::createCommandPool() {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_index;

    if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
        _THROW_ERROR("Failed to create command pool");

    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) != VK_SUCCESS)
        _THROW_ERROR("Failed to allocate command buffer");
}

void VulkanGSPipeline::createFence() {
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;

    VkResult result = vkCreateFence(device, &fenceInfo, nullptr, &fence);
    if (result != VK_SUCCESS)
        _THROW_ERROR("Failed to create fence");
}

void VulkanGSPipeline::createQueryPools() {
    // timestamp
    VkQueryPoolCreateInfo queryPoolCreateInfo = {};
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = MAX_TIMESTAMP_QUERY_COUNT;
    if (vkCreateQueryPool(device, &queryPoolCreateInfo, nullptr, &timestamp_query_pool) != VK_SUCCESS)
        _THROW_ERROR("Failed to create timestamp query pool");
}

void VulkanGSPipeline::createShaderModule(const std::vector<uint32_t>& spirv_code, VkShaderModule* pShaderModule) {
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_code.size() * sizeof(uint32_t);
    create_info.pCode = spirv_code.data();

    if (vkCreateShaderModule(device, &create_info, nullptr, pShaderModule) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");
}

void VulkanGSPipeline::beginCommandBatch() {
    if (commandBatchInProgress)
        _THROW_ERROR("Command batch already in progress");
    commandBatchInProgress = true;

    PerfTimer::hostToc();

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS)
        _THROW_ERROR("Failed to begin command buffer for batch");

    vkCmdResetQueryPool(command_buffer, timestamp_query_pool, 0, MAX_TIMESTAMP_QUERY_COUNT);
    PerfTimer::popMarkers(this);
}

void VulkanGSPipeline::endCommandBatch(bool use_fence) {
    if (!commandBatchInProgress)
        _THROW_ERROR("No command batch in progress");

    if (timestampNumWritten > 0) {
        while (timestampStackDepth > 0)
            PerfTimer::pushMarker(this);
    }

    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        _THROW_ERROR("Failed to end command buffer for batch");
    }

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    if (vkQueueSubmit(command_queue, 1, &submit_info,
                      use_fence ? fence : VK_NULL_HANDLE) != VK_SUCCESS) {
        _THROW_ERROR("Failed to submit batch");
    }

    commandBatchInProgress = false;

    if (use_fence) {
#if SSE2_AVAILABLE
#if ENABLE_ASSERTION
        constexpr unsigned long long kTimeout = 0x100000000ull;
        auto time0 = __rdtsc();
#endif
        while (vkGetFenceStatus(device, fence) != VK_SUCCESS) {
            _mm_pause();
#if ENABLE_ASSERTION
            if (__rdtsc() - time0 >= kTimeout) {
                // _THROW_ERROR("Fence timed out");
                printf("\033[91m%s\033[m\n", "Timed out.");
                std::terminate(); // note that this is often in destructor
            }
#endif
        }
#else
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
#endif
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS)
            _THROW_ERROR("Failed to reset fence");
    } else
        vkQueueWaitIdle(command_queue);

    PerfTimer::hostTic();

    if (timestampNumWritten > 0) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physical_device, &deviceProperties);
        double timestampPeriod = deviceProperties.limits.timestampPeriod;

        std::vector<uint64_t> timestamps(timestampNumWritten);
        vkGetQueryPoolResults(
            device, timestamp_query_pool,
            0, timestampNumWritten,
            sizeof(uint64_t) * timestampNumWritten,
            timestamps.data(), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        std::vector<double> times(timestampNumWritten);
        for (uint32_t i = 0; i < timestampNumWritten; i++)
            times[i] = 1e-9 * double(timestamps[i] - timestamps[0]) * timestampPeriod;
        auto time_updates = PerfTimer::update(times);
        for (auto& callback : timerCallbacks)
            callback(time_updates);

        timestampNumWritten = 0;
    }
}

bool VulkanGSPipeline::writeTimestamp(int delta) {
    if (!commandBatchInProgress)
        _THROW_ERROR("writeTimestamp requires command batch in progress");
    if (timestampNumWritten >= MAX_TIMESTAMP_QUERY_COUNT)
        _THROW_ERROR("Too many timestamps written");
    if (delta != 1 && delta != -1)
        _THROW_ERROR("delta in writeTimestamp must be 1 or -1");
    if (delta == -1 && timestampStackDepth == 0)
        _THROW_ERROR("attempt to write exit timestamp while stack is empty");
    vkCmdWriteTimestamp(
        command_buffer,
        // delta == 1 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        timestamp_query_pool, timestampNumWritten);
    timestampNumWritten += 1;
    timestampStackDepth += delta;
    return true;
}

bool VulkanGSPipeline::writeTimestampNoExcept(int delta) {
    if (!commandBatchInProgress)
        return false;
    if (timestampNumWritten >= MAX_TIMESTAMP_QUERY_COUNT)
        return false;
    if (delta != 1 && delta != -1)
        return false;
    if (delta == -1 && timestampStackDepth == 0)
        return false;
    vkCmdWriteTimestamp(
        command_buffer,
        // delta == 1 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        timestamp_query_pool, timestampNumWritten);
    timestampNumWritten += 1;
    timestampStackDepth += delta;
    return true;
}

VkAccessFlags toAccessMask(VulkanGSPipeline::BarrierMask barrierMask) {
    VkAccessFlags result = (VkAccessFlags)0;
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_TRANSFER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_TRANSFER_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_SHADER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_ACCESS_SHADER_WRITE_BIT;
    return result;
}

VkPipelineStageFlags toStageMask(VulkanGSPipeline::BarrierMask barrierMask) {
    VkPipelineStageFlags result = (VkPipelineStageFlags)0;
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE)
        result |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (barrierMask == VulkanGSPipeline::HOST_READ ||
        barrierMask == VulkanGSPipeline::HOST_WRITE ||
        barrierMask == VulkanGSPipeline::HOST_READ_WRITE)
        result |= VK_PIPELINE_STAGE_HOST_BIT;
    return result;
}

void VulkanGSPipeline::memoryBarrier(
    VulkanGSPipeline::BarrierMask srcMask,
    VulkanGSPipeline::BarrierMask dstMask) {
    if (!commandBatchInProgress)
        return;

    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = toAccessMask(srcMask);
    barrier.dstAccessMask = toAccessMask(dstMask);

    vkCmdPipelineBarrier(
        command_buffer,
        toStageMask(srcMask), toStageMask(dstMask),
        0,           // dependencyFlags
        1, &barrier, // memory barriers
        0, nullptr,  // buffer barriers
        0, nullptr   // image barriers
    );
}

void VulkanGSPipeline::bufferMemoryBarrier(
    const std::vector<std::pair<_VulkanBuffer, VulkanGSPipeline::BarrierMask>>& buffers,
    VulkanGSPipeline::BarrierMask dstMask) {
    if (!commandBatchInProgress)
        return;

    std::vector<VkBufferMemoryBarrier> barriers;
    barriers.reserve(buffers.size());
    VkPipelineStageFlags srcStageFlags = (VkPipelineStageFlags)0;
    for (auto& [buffer, srcMask] : buffers) {
        if (buffer.buffer == VK_NULL_HANDLE)
            continue;
        VkBufferMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = toAccessMask(srcMask);
        barrier.dstAccessMask = toAccessMask(dstMask);
        barrier.srcQueueFamilyIndex = queue_family_index;
        barrier.dstQueueFamilyIndex = queue_family_index;
        barrier.buffer = buffer.buffer;
        barrier.offset = 0;
        barrier.size = buffer.size;
        barriers.push_back(barrier);
        srcStageFlags |= toStageMask(srcMask);
    }
    if (barriers.empty())
        return;

    vkCmdPipelineBarrier(
        command_buffer,
        srcStageFlags, toStageMask(dstMask),
        0,                                          // dependencyFlags
        0, nullptr,                                 // memory barriers
        (uint32_t)barriers.size(), barriers.data(), // buffer barriers
        0, nullptr                                  // image barriers
    );
}

// Compute pipeline

void VulkanGSPipeline::createComputeDescriptorSetLayout(_ComputePipeline& pipeline) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(pipeline.buffer_layouts.size());

    for (int i : pipeline.buffer_layouts) {
        VkDescriptorSetLayoutBinding binding;
        binding.binding = i;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binding.pImmutableSamplers = nullptr;
        bindings.push_back(binding);
    }

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &pipeline.descriptor_set_layout) != VK_SUCCESS)
        _THROW_ERROR("Failed to create descriptor set layout");
}

void VulkanGSPipeline::createComputeDescriptorPool(_ComputePipeline& pipeline) {
    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = (uint32_t)(pipeline.buffer_layouts.size());

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;

    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &pipeline.descriptor_pool) != VK_SUCCESS)
        _THROW_ERROR("Failed to create descriptor pool");
}

void VulkanGSPipeline::updateComputeDescriptorSet(_ComputePipeline& pipeline, const std::vector<_VulkanBuffer>& data_buffers) {
    if (pipeline.descriptor_set == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = pipeline.descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &pipeline.descriptor_set_layout;

        if (vkAllocateDescriptorSets(device, &alloc_info, &pipeline.descriptor_set) != VK_SUCCESS)
            _THROW_ERROR("Failed to allocate descriptor sets while updating");
    }

    size_t num_buffers = pipeline.buffer_layouts.size();
    std::vector<VkWriteDescriptorSet> descriptor_writes(num_buffers);
    std::vector<VkDescriptorBufferInfo> buffer_infos(num_buffers);

    int idx = 0;
    for (int i : pipeline.buffer_layouts) {
        if (data_buffers[i].buffer == VK_NULL_HANDLE)
            _THROW_ERROR("Buffer " + std::to_string(i) + " is NULL");
        buffer_infos[idx].buffer = data_buffers[i].buffer;
        buffer_infos[idx].offset = 0;
        // buffer_infos[idx].range = data_buffers[i].size;
        buffer_infos[idx].range = data_buffers[i].allocSize;

        descriptor_writes[idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[idx].dstSet = pipeline.descriptor_set;
        descriptor_writes[idx].dstBinding = i;
        descriptor_writes[idx].dstArrayElement = 0;
        descriptor_writes[idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_writes[idx].descriptorCount = 1;
        descriptor_writes[idx].pBufferInfo = &buffer_infos[idx];

        idx++;
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
}

void VulkanGSPipeline::createComputePipeline(_ComputePipeline& pipeline, const std::string& spirv_path, uint32_t min_shared_memory, bool compatible_subgroup_size) {

    if (min_shared_memory > this->deviceInfo.sharedSize) {
        pipeline.shader = VK_NULL_HANDLE;
        return;
    }

    createShaderModule(loadSpirv(spirv_path), &pipeline.shader);
    createComputeDescriptorSetLayout(pipeline);

    // Create push constant range for uniforms
    VkPushConstantRange push_constant_range = {};
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = (uint32_t)MAX_UNIFORM_SIZE;

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &pipeline.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;

    if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline.pipeline_layout) != VK_SUCCESS) {
        _THROW_ERROR("Failed to create pipeline set layout");
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT req = {};
    req.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    req.requiredSubgroupSize = SUBGROUP_SIZE; // 32

    VkPipelineShaderStageCreateInfo compute_shader_stage_info = {};
    compute_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute_shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute_shader_stage_info.module = pipeline.shader;
    compute_shader_stage_info.pName = "main";
    if (compatible_subgroup_size && (deviceInfo.subgroupSize != SUBGROUP_SIZE ||
                                     deviceInfo.vendor == DeviceVendor::Intel_R_))
        compute_shader_stage_info.pNext = &req;

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = pipeline.pipeline_layout;
    pipeline_info.stage = compute_shader_stage_info;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline.pipeline) != VK_SUCCESS)
        _THROW_ERROR("Failed to create compute pipeline");

    createComputeDescriptorPool(pipeline);

    all_compute_pipelines.push_back(&pipeline);
}

void VulkanGSPipeline::executeCompute(
    std::vector<std::pair<size_t, size_t>> dims,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline& pipeline,
    const std::vector<_VulkanBuffer>& buffers) {
    if (uniformSize > MAX_UNIFORM_SIZE)
        _THROW_ERROR("Maximum uniform size exceeded");

    if (pipeline.buffers != buffers) {
        HOST_GUARD;
        updateComputeDescriptorSet(pipeline, buffers);
        pipeline.buffers = buffers;
    }

    DEVICE_GUARD;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout, 0, 1, &pipeline.descriptor_set, 0, nullptr);

    // Push constants for uniforms
    if (uniformsPtr) {
        vkCmdPushConstants(
            command_buffer,
            pipeline.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, (uint32_t)uniformSize, uniformsPtr);
    }

    // Dispatch compute shader
    while (dims.size() < 3)
        dims.push_back({1, 1});
    uint32_t nGroupsX = (uint32_t)_CEIL_DIV(dims[0].first, dims[0].second);
    uint32_t nGroupsY = (uint32_t)_CEIL_DIV(dims[1].first, dims[1].second);
    uint32_t nGroupsZ = (uint32_t)_CEIL_DIV(dims[2].first, dims[2].second);
    if (nGroupsX > deviceInfo.maxGroupsX ||
        nGroupsY > deviceInfo.maxGroupsY ||
        nGroupsZ > deviceInfo.maxGroupsZ)
        _THROW_ERROR("Cannot launch compute kernel, too many groups: [" +
                     std::to_string(nGroupsX) + " " +
                     std::to_string(nGroupsY) + " " +
                     std::to_string(nGroupsZ) + "] > [" +
                     std::to_string(deviceInfo.maxGroupsX) + " " +
                     std::to_string(deviceInfo.maxGroupsY) + " " +
                     std::to_string(deviceInfo.maxGroupsZ) + "]");
    vkCmdDispatch(command_buffer, nGroupsX, nGroupsY, nGroupsZ);
}

void VulkanGSPipeline::destroyComputePipeline(_ComputePipeline& pipeline) {
    if (pipeline.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, pipeline.descriptor_pool, nullptr);
        pipeline.descriptor_pool = VK_NULL_HANDLE;
    }
    if (pipeline.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, pipeline.descriptor_set_layout, nullptr);
        pipeline.descriptor_set_layout = VK_NULL_HANDLE;
    }
    if (pipeline.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        pipeline.pipeline = VK_NULL_HANDLE;
    }
    if (pipeline.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipeline.pipeline_layout, nullptr);
        pipeline.pipeline_layout = VK_NULL_HANDLE;
    }
    if (pipeline.shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, pipeline.shader, nullptr);
        pipeline.shader = VK_NULL_HANDLE;
    }
}
