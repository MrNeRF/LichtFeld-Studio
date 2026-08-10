/* L0 probe: does NVML compute-process memory include Vulkan DEVICE_LOCAL allocations?
 *
 * Method (design HUD_DESIGN.md §3.11):
 *   1. Create CUDA primary context, sample NVML per-PID compute process memory.
 *   2. Allocate a large DEVICE_LOCAL VkDeviceMemory (and separately host-visible).
 *   3. Resample NVML after each allocation.
 *   4. Allocate a known-size cudaMalloc block as a positive control.
 *   5. Compare deltas: if NVML tracks Vulkan DEVICE_LOCAL growth, root F stays in the spine.
 *
 * Build:
 *   g++ -O2 -std=c++17 l0_nvml_probe.cpp -o l0_nvml_probe \
 *     -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
 *     -lcudart -ldl -lvulkan -lnvidia-ml
 */

#include <cuda_runtime.h>
#include <nvml.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kVulkanDeviceLocalBytes = 512ull * 1024ull * 1024ull; // 512 MiB
constexpr std::size_t kVulkanHostVisibleBytes = 128ull * 1024ull * 1024ull;  // 128 MiB
constexpr std::size_t kCudaBytes = 256ull * 1024ull * 1024ull;               // 256 MiB

[[nodiscard]] const char* vk_err(VkResult r) {
    switch (r) {
    case VK_SUCCESS:
        return "VK_SUCCESS";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
        return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
        return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
        return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
        return "VK_ERROR_INCOMPATIBLE_DRIVER";
    default:
        return "VK_OTHER";
    }
}

[[nodiscard]] std::size_t nvml_process_used_bytes(nvmlDevice_t device, unsigned int pid) {
    // Use compute-running processes (same API as gpu_memory_query.cpp).
    unsigned int count = 128;
    std::vector<nvmlProcessInfo_t> procs(count);
    nvmlReturn_t st =
        nvmlDeviceGetComputeRunningProcesses(device, &count, procs.data());
    if (st == NVML_ERROR_INSUFFICIENT_SIZE) {
        procs.resize(count);
        st = nvmlDeviceGetComputeRunningProcesses(device, &count, procs.data());
    }
    if (st != NVML_SUCCESS) {
        std::fprintf(stderr, "nvmlDeviceGetComputeRunningProcesses failed: %s\n",
                     nvmlErrorString(st));
        return 0;
    }
    for (unsigned int i = 0; i < count; ++i) {
        if (procs[i].pid == pid) {
            return static_cast<std::size_t>(procs[i].usedGpuMemory);
        }
    }
    return 0;
}

[[nodiscard]] std::size_t nvml_graphics_process_used_bytes(nvmlDevice_t device,
                                                            unsigned int pid) {
    unsigned int count = 128;
    std::vector<nvmlProcessInfo_t> procs(count);
    nvmlReturn_t st =
        nvmlDeviceGetGraphicsRunningProcesses(device, &count, procs.data());
    if (st == NVML_ERROR_INSUFFICIENT_SIZE) {
        procs.resize(count);
        st = nvmlDeviceGetGraphicsRunningProcesses(device, &count, procs.data());
    }
    if (st != NVML_SUCCESS) {
        std::fprintf(stderr, "nvmlDeviceGetGraphicsRunningProcesses failed: %s\n",
                     nvmlErrorString(st));
        return 0;
    }
    for (unsigned int i = 0; i < count; ++i) {
        if (procs[i].pid == pid) {
            return static_cast<std::size_t>(procs[i].usedGpuMemory);
        }
    }
    return 0;
}

[[nodiscard]] std::size_t cuda_used_bytes() {
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b < free_b) {
        return 0;
    }
    return total_b - free_b;
}

struct Sample {
    const char* label = "";
    std::size_t nvml_compute = 0;
    std::size_t nvml_graphics = 0;
    std::size_t cuda_device_used = 0;
};

void print_sample(const Sample& s) {
    std::printf(
        "%-36s  nvml_compute=%8.2f MiB  nvml_graphics=%8.2f MiB  cudaMemGetInfo_used=%8.2f MiB\n",
        s.label,
        s.nvml_compute / (1024.0 * 1024.0),
        s.nvml_graphics / (1024.0 * 1024.0),
        s.cuda_device_used / (1024.0 * 1024.0));
}

Sample take_sample(const char* label, nvmlDevice_t device, unsigned int pid) {
    // Give the driver a moment to publish accounting.
    cudaDeviceSynchronize();
    Sample s;
    s.label = label;
    s.nvml_compute = nvml_process_used_bytes(device, pid);
    s.nvml_graphics = nvml_graphics_process_used_bytes(device, pid);
    s.cuda_device_used = cuda_used_bytes();
    print_sample(s);
    return s;
}

[[nodiscard]] int find_memory_type(VkPhysicalDevice phys,
                                   std::uint32_t type_bits,
                                   VkMemoryPropertyFlags required,
                                   VkMemoryPropertyFlags preferred_not = 0) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    int best = -1;
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0) {
            continue;
        }
        const auto flags = props.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) {
            continue;
        }
        if (preferred_not != 0 && (flags & preferred_not) == preferred_not) {
            // Keep as fallback but prefer types without preferred_not.
            if (best < 0) {
                best = static_cast<int>(i);
            }
            continue;
        }
        return static_cast<int>(i);
    }
    return best;
}

struct VkState {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
};

bool init_vulkan(VkState& vk) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "l0_nvml_probe";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkResult r = vkCreateInstance(&ici, nullptr, &vk.instance);
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateInstance failed: %s (%d)\n", vk_err(r), r);
        return false;
    }

    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(vk.instance, &count, nullptr);
    if (count == 0) {
        std::fprintf(stderr, "no Vulkan physical devices\n");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(vk.instance, &count, devices.data());

    // Prefer a discrete NVIDIA GPU.
    for (auto d : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.vendorID == 0x10de) {
            vk.phys = d;
            std::printf("Vulkan physical device: %s (vendor=0x%x device=0x%x)\n",
                        props.deviceName, props.vendorID, props.deviceID);
            break;
        }
    }
    if (vk.phys == VK_NULL_HANDLE) {
        vk.phys = devices[0];
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(vk.phys, &props);
        std::printf("Vulkan physical device (fallback): %s\n", props.deviceName);
    }

    std::uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk.phys, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(vk.phys, &qcount, qprops.data());
    vk.queue_family = 0;
    for (std::uint32_t i = 0; i < qcount; ++i) {
        if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            vk.queue_family = i;
            break;
        }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = vk.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    r = vkCreateDevice(vk.phys, &dci, nullptr, &vk.device);
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateDevice failed: %s\n", vk_err(r));
        return false;
    }
    return true;
}

[[nodiscard]] VkDeviceMemory alloc_vk(VkState& vk,
                                      std::size_t bytes,
                                      VkMemoryPropertyFlags required,
                                      VkMemoryPropertyFlags preferred_not,
                                      int* out_type_index,
                                      VkMemoryPropertyFlags* out_flags) {
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = bytes;
    // typeBits: all types initially; we pick by property flags.
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(vk.phys, &props);
    std::uint32_t type_bits = (1u << props.memoryTypeCount) - 1u;
    const int type = find_memory_type(vk.phys, type_bits, required, preferred_not);
    if (type < 0) {
        std::fprintf(stderr, "no memory type for required flags 0x%x\n", required);
        return VK_NULL_HANDLE;
    }
    mai.memoryTypeIndex = static_cast<std::uint32_t>(type);
    *out_type_index = type;
    *out_flags = props.memoryTypes[type].propertyFlags;

    VkDeviceMemory mem = VK_NULL_HANDLE;
    const VkResult r = vkAllocateMemory(vk.device, &mai, nullptr, &mem);
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "vkAllocateMemory(%zu) failed: %s\n", bytes, vk_err(r));
        return VK_NULL_HANDLE;
    }
    std::printf("  allocated VkDeviceMemory size=%.2f MiB type=%d flags=0x%x heap=%u\n",
                bytes / (1024.0 * 1024.0),
                type,
                *out_flags,
                props.memoryTypes[type].heapIndex);
    const auto heap_flags = props.memoryHeaps[props.memoryTypes[type].heapIndex].flags;
    std::printf("  heap flags=0x%x device_local=%s\n",
                heap_flags,
                (heap_flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "yes" : "no");
    return mem;
}

void dump_memory_types(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    std::printf("Memory heaps (%u):\n", props.memoryHeapCount);
    for (std::uint32_t i = 0; i < props.memoryHeapCount; ++i) {
        std::printf("  heap[%u] size=%.1f MiB flags=0x%x device_local=%s\n",
                    i,
                    props.memoryHeaps[i].size / (1024.0 * 1024.0),
                    props.memoryHeaps[i].flags,
                    (props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "yes"
                                                                                    : "no");
    }
    std::printf("Memory types (%u):\n", props.memoryTypeCount);
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        const auto f = props.memoryTypes[i].propertyFlags;
        std::printf("  type[%u] heap=%u flags=0x%x%s%s%s%s\n",
                    i,
                    props.memoryTypes[i].heapIndex,
                    f,
                    (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " DEVICE_LOCAL" : "",
                    (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " HOST_VISIBLE" : "",
                    (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? " HOST_COHERENT" : "",
                    (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? " HOST_CACHED" : "");
    }
}

} // namespace

int main() {
    std::printf("=== L0 NVML includes Vulkan? probe ===\n");
    std::printf("Driver path: nvmlDeviceGetComputeRunningProcesses (same as app)\n");
    std::printf("Alloc sizes: DEVICE_LOCAL=%.0f MiB HOST_VISIBLE=%.0f MiB CUDA=%.0f MiB\n",
                kVulkanDeviceLocalBytes / (1024.0 * 1024.0),
                kVulkanHostVisibleBytes / (1024.0 * 1024.0),
                kCudaBytes / (1024.0 * 1024.0));

    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice failed\n");
        return 1;
    }
    // Force primary context creation.
    void* warmup = nullptr;
    if (cudaMalloc(&warmup, 1) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc warmup failed\n");
        return 1;
    }
    cudaFree(warmup);

    char pci[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {};
    if (cudaDeviceGetPCIBusId(pci, sizeof(pci), 0) != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceGetPCIBusId failed\n");
        return 1;
    }
    std::printf("CUDA PCI bus id: %s  pid=%u\n", pci, static_cast<unsigned>(getpid()));

    if (nvmlInit_v2() != NVML_SUCCESS) {
        std::fprintf(stderr, "nvmlInit failed\n");
        return 1;
    }
    nvmlDevice_t device = nullptr;
    if (nvmlDeviceGetHandleByPciBusId_v2(pci, &device) != NVML_SUCCESS) {
        std::fprintf(stderr, "nvmlDeviceGetHandleByPciBusId failed\n");
        return 1;
    }
    const unsigned int pid = static_cast<unsigned int>(getpid());

    const Sample s0 = take_sample("0_after_cuda_context", device, pid);

    VkState vk;
    if (!init_vulkan(vk)) {
        return 1;
    }
    dump_memory_types(vk.phys);

    const Sample s1 = take_sample("1_after_vk_device", device, pid);

    int type_dl = -1;
    VkMemoryPropertyFlags flags_dl = 0;
    VkDeviceMemory mem_dl =
        alloc_vk(vk,
                 kVulkanDeviceLocalBytes,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // prefer pure device-local
                 &type_dl,
                 &flags_dl);
    if (mem_dl == VK_NULL_HANDLE) {
        return 1;
    }
    const Sample s2 = take_sample("2_after_vk_device_local_512MiB", device, pid);

    int type_hv = -1;
    VkMemoryPropertyFlags flags_hv = 0;
    VkDeviceMemory mem_hv =
        alloc_vk(vk,
                 kVulkanHostVisibleBytes,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 0,
                 &type_hv,
                 &flags_hv);
    if (mem_hv == VK_NULL_HANDLE) {
        return 1;
    }
    const Sample s3 = take_sample("3_after_vk_host_visible_128MiB", device, pid);

    void* cuda_ptr = nullptr;
    if (cudaMalloc(&cuda_ptr, kCudaBytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc %zu failed\n", kCudaBytes);
        return 1;
    }
    const Sample s4 = take_sample("4_after_cudaMalloc_256MiB", device, pid);

    // Touch Vulkan memory so it cannot be lazily uncommitted on some drivers.
    // DEVICE_LOCAL cannot be mapped; issue a no-op queue submit is overkill —
    // instead free/realloc is not needed; NVML accounts on allocate for NVIDIA.
    // Touch host-visible by mapping.
    void* mapped = nullptr;
    if (vkMapMemory(vk.device, mem_hv, 0, 4096, 0, &mapped) == VK_SUCCESS) {
        std::memset(mapped, 0xAB, 4096);
        vkUnmapMemory(vk.device, mem_hv);
    }
    const Sample s5 = take_sample("5_after_touch_host_visible", device, pid);

    auto delta = [](std::size_t a, std::size_t b) -> std::int64_t {
        return static_cast<std::int64_t>(a) - static_cast<std::int64_t>(b);
    };

    std::printf("\n=== DELTAS (MiB) ===\n");
    const auto d_vk_dl_compute = delta(s2.nvml_compute, s1.nvml_compute) / (1024.0 * 1024.0);
    const auto d_vk_dl_graphics = delta(s2.nvml_graphics, s1.nvml_graphics) / (1024.0 * 1024.0);
    const auto d_vk_dl_cuda = delta(s2.cuda_device_used, s1.cuda_device_used) / (1024.0 * 1024.0);

    const auto d_vk_hv_compute = delta(s3.nvml_compute, s2.nvml_compute) / (1024.0 * 1024.0);
    const auto d_vk_hv_graphics = delta(s3.nvml_graphics, s2.nvml_graphics) / (1024.0 * 1024.0);
    const auto d_vk_hv_cuda = delta(s3.cuda_device_used, s2.cuda_device_used) / (1024.0 * 1024.0);

    const auto d_cuda_compute = delta(s4.nvml_compute, s3.nvml_compute) / (1024.0 * 1024.0);
    const auto d_cuda_graphics = delta(s4.nvml_graphics, s3.nvml_graphics) / (1024.0 * 1024.0);
    const auto d_cuda_cuda = delta(s4.cuda_device_used, s3.cuda_device_used) / (1024.0 * 1024.0);

    std::printf("After +512 MiB Vulkan DEVICE_LOCAL:\n");
    std::printf("  d(nvml_compute)=%+.2f  d(nvml_graphics)=%+.2f  d(cudaMemGetInfo)=%+.2f\n",
                d_vk_dl_compute, d_vk_dl_graphics, d_vk_dl_cuda);
    std::printf("After +128 MiB Vulkan HOST_VISIBLE:\n");
    std::printf("  d(nvml_compute)=%+.2f  d(nvml_graphics)=%+.2f  d(cudaMemGetInfo)=%+.2f\n",
                d_vk_hv_compute, d_vk_hv_graphics, d_vk_hv_cuda);
    std::printf("After +256 MiB cudaMalloc (positive control):\n");
    std::printf("  d(nvml_compute)=%+.2f  d(nvml_graphics)=%+.2f  d(cudaMemGetInfo)=%+.2f\n",
                d_cuda_compute, d_cuda_graphics, d_cuda_cuda);

    const bool device_local_in_compute =
        d_vk_dl_compute > 256.0; // expect ~512 if included; allow half-tolerance
    const bool device_local_in_graphics = d_vk_dl_graphics > 256.0;
    const bool host_visible_in_compute = d_vk_hv_compute > 64.0;
    const bool cuda_in_compute = d_cuda_compute > 128.0;

    std::printf("\n=== VERDICT ===\n");
    std::printf("positive_control_cuda_in_nvml_compute: %s (d=%.2f MiB, expect ~256)\n",
                cuda_in_compute ? "YES" : "NO", d_cuda_compute);
    std::printf("vulkan_DEVICE_LOCAL_in_nvml_compute:   %s (d=%.2f MiB, expect ~512 if included)\n",
                device_local_in_compute ? "YES" : "NO", d_vk_dl_compute);
    std::printf("vulkan_DEVICE_LOCAL_in_nvml_graphics:  %s (d=%.2f MiB)\n",
                device_local_in_graphics ? "YES" : "NO", d_vk_dl_graphics);
    std::printf("vulkan_HOST_VISIBLE_in_nvml_compute:   %s (d=%.2f MiB)\n",
                host_visible_in_compute ? "YES" : "NO", d_vk_hv_compute);
    std::printf("host_visible_type_flags=0x%x device_local_bit=%s\n",
                flags_hv,
                (flags_hv & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "yes" : "no");

    if (device_local_in_compute) {
        std::printf("SPINE_RULE: root F (Vulkan VMA blocks) BELONGS in the NVML process sum.\n");
    } else if (device_local_in_graphics && !device_local_in_compute) {
        std::printf("SPINE_RULE: root F is NOT in compute NVML; use graphics NVML or exclude root F.\n");
    } else {
        std::printf("SPINE_RULE: root F must be EXCLUDED from the sum (out-of-ledger annotation).\n");
    }

    // Machine-readable footer for the report.
    std::printf("\nJSON_RESULT\n");
    std::printf("{\n");
    std::printf("  \"d_vk_device_local_nvml_compute_mib\": %.3f,\n", d_vk_dl_compute);
    std::printf("  \"d_vk_device_local_nvml_graphics_mib\": %.3f,\n", d_vk_dl_graphics);
    std::printf("  \"d_vk_host_visible_nvml_compute_mib\": %.3f,\n", d_vk_hv_compute);
    std::printf("  \"d_cuda_nvml_compute_mib\": %.3f,\n", d_cuda_compute);
    std::printf("  \"device_local_in_compute\": %s,\n", device_local_in_compute ? "true" : "false");
    std::printf("  \"device_local_in_graphics\": %s,\n", device_local_in_graphics ? "true" : "false");
    std::printf("  \"host_visible_in_compute\": %s,\n", host_visible_in_compute ? "true" : "false");
    std::printf("  \"cuda_in_compute\": %s,\n", cuda_in_compute ? "true" : "false");
    std::printf("  \"host_visible_flags\": %u,\n", static_cast<unsigned>(flags_hv));
    std::printf("  \"root_f_in_sum\": %s\n", device_local_in_compute ? "true" : "false");
    std::printf("}\n");

    cudaFree(cuda_ptr);
    vkFreeMemory(vk.device, mem_hv, nullptr);
    vkFreeMemory(vk.device, mem_dl, nullptr);
    vkDestroyDevice(vk.device, nullptr);
    vkDestroyInstance(vk.instance, nullptr);
    nvmlShutdown();
    return 0;
}
