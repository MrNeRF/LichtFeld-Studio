/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gpu_memory_query.hpp"
#include "core/gpu_device_info.hpp"
#include "core/host_metrics.hpp"
#include "core/tensor_backend.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
#if LFS_HAS_CUDA
#include <cuda_runtime.h>
#include <nvml.h>
#endif
#include <vector>

#ifdef _WIN32
#include <dxgi1_4.h>
#include <windows.h>
#elif defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#elif defined(__linux__)
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace lfs::vis::gui {

    namespace {

        std::string shortenGpuDeviceName(std::string name) {
            if (name.rfind("NVIDIA ", 0) == 0)
                name.erase(0, std::string_view("NVIDIA ").size());
            return name;
        }

#if LFS_HAS_CUDA && defined(_WIN32)
        // Windows: DXGI QueryVideoMemoryInfo for per-process GPU memory.
        // NVML process memory returns NVML_VALUE_NOT_AVAILABLE under WDDM, but
        // device utilization rates work and are used for the GPU% meter.
        struct DxgiMemoryState {
            IDXGIAdapter3* adapter3 = nullptr;
            bool init_done = false;
            UINT node_index = 0;

            DxgiMemoryState(const DxgiMemoryState&) = delete;
            DxgiMemoryState& operator=(const DxgiMemoryState&) = delete;
            DxgiMemoryState() = default;

            ~DxgiMemoryState() {
                // Intentionally not releasing adapter3 here.
                // At static destruction time, DXGI/DirectX runtime may already be
                // unloaded, causing a crash. The OS will clean up the COM reference
                // when the process exits anyway.
            }

            void ensureInit() {
                if (init_done)
                    return;
                init_done = true;

                IDXGIFactory1* factory = nullptr;
                if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                              reinterpret_cast<void**>(&factory))))
                    return;

                int cuda_device = 0;
                cudaGetDevice(&cuda_device);

                if (matchByLuid(factory, cuda_device)) {
                    factory->Release();
                    return;
                }

                // Identical capacities do not identify an adapter. An unavailable
                // process sample is preferable to reading another GPU's usage.
                factory->Release();
            }

            bool getProcessMemory(size_t& used, size_t& budget) {
                ensureInit();
                if (!adapter3)
                    return false;
                DXGI_QUERY_VIDEO_MEMORY_INFO mem_info{};
                if (FAILED(adapter3->QueryVideoMemoryInfo(
                        node_index, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mem_info)))
                    return false;
                used = static_cast<size_t>(mem_info.CurrentUsage);
                budget = static_cast<size_t>(mem_info.Budget);
                return true;
            }

        private:
            // Match DXGI adapter to CUDA device via LUID (exact, multi-GPU safe).
            bool matchByLuid(IDXGIFactory1* factory, int cuda_device) {
                using FnCuDeviceGetLuid = int (*)(char*, unsigned int*, int);
                HMODULE nvcuda = GetModuleHandleA("nvcuda.dll");
                if (!nvcuda)
                    return false;
                auto fn = reinterpret_cast<FnCuDeviceGetLuid>(
                    GetProcAddress(nvcuda, "cuDeviceGetLuid"));
                if (!fn)
                    return false;

                LUID cuda_luid{};
                static_assert(sizeof(LUID) == 8);
                unsigned int node_mask = 0;
                if (fn(reinterpret_cast<char*>(&cuda_luid), &node_mask, cuda_device) != 0)
                    return false;

                for (UINT i = 0;; ++i) {
                    IDXGIAdapter* adapter = nullptr;
                    if (FAILED(factory->EnumAdapters(i, &adapter)))
                        break;
                    DXGI_ADAPTER_DESC desc{};
                    if (SUCCEEDED(adapter->GetDesc(&desc)) &&
                        desc.AdapterLuid.LowPart == cuda_luid.LowPart &&
                        desc.AdapterLuid.HighPart == cuda_luid.HighPart) {
                        adapter->QueryInterface(__uuidof(IDXGIAdapter3),
                                                reinterpret_cast<void**>(&adapter3));
                        adapter->Release();
                        node_index = node_mask ? std::countr_zero(node_mask) : 0;
                        return adapter3 != nullptr;
                    }
                    adapter->Release();
                }
                return false;
            }
        };

        DxgiMemoryState& dxgiState() {
            static DxgiMemoryState s;
            return s;
        }
#endif

#if LFS_HAS_CUDA
        // NVML: process memory on Linux; utilization on Linux and Windows.
        using NvmlDevice = void*;
        enum { NVML_SUCCESS = 0 };
        constexpr int NVML_PCI_BUS_ID_LEN = 32;

        using FnNvmlInit = int (*)();
        using FnNvmlDeviceGetHandleByPciBusId = int (*)(const char*, NvmlDevice*);
        using FnNvmlDeviceGetProcesses = int (*)(NvmlDevice, unsigned int*, nvmlProcessInfo_t*);
        using FnNvmlDeviceGetMemoryInfo = int (*)(NvmlDevice, nvmlMemory_v2_t*);
        struct NvmlUtilization {
            unsigned int gpu;
            unsigned int memory;
        };
        using FnNvmlDeviceGetUtilizationRates = int (*)(NvmlDevice, NvmlUtilization*);

        struct NvmlState {
            bool initialized = false;
            NvmlDevice device = nullptr;
            unsigned int pid = 0;
#ifdef _WIN32
            HMODULE lib = nullptr;
#else
            void* lib = nullptr;
#endif
            FnNvmlDeviceGetProcesses fn_get_compute = nullptr;
            FnNvmlDeviceGetProcesses fn_get_graphics = nullptr;
            FnNvmlDeviceGetMemoryInfo fn_get_memory = nullptr;
            FnNvmlDeviceGetUtilizationRates fn_get_utilization = nullptr;

            NvmlState() {
#ifdef _WIN32
                lib = LoadLibraryA("nvml.dll");
                if (!lib)
                    return;
                auto load = [this](const char* name) -> void* {
                    return reinterpret_cast<void*>(GetProcAddress(lib, name));
                };
#else
                lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
                if (!lib)
                    lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
                if (!lib)
                    return;
                auto load = [this](const char* name) -> void* {
                    return dlsym(lib, name);
                };
#endif

                auto fn_init = reinterpret_cast<FnNvmlInit>(load("nvmlInit_v2"));
                auto fn_get_handle = reinterpret_cast<FnNvmlDeviceGetHandleByPciBusId>(
                    load("nvmlDeviceGetHandleByPciBusId_v2"));
                fn_get_compute = reinterpret_cast<FnNvmlDeviceGetProcesses>(
                    load("nvmlDeviceGetComputeRunningProcesses_v3"));
                fn_get_graphics = reinterpret_cast<FnNvmlDeviceGetProcesses>(
                    load("nvmlDeviceGetGraphicsRunningProcesses_v3"));
                fn_get_memory = reinterpret_cast<FnNvmlDeviceGetMemoryInfo>(
                    load("nvmlDeviceGetMemoryInfo_v2"));
                fn_get_utilization = reinterpret_cast<FnNvmlDeviceGetUtilizationRates>(
                    load("nvmlDeviceGetUtilizationRates"));

                // Utilization only needs init + handle + getUtilizationRates.
                // Process memory also needs get_procs (Linux path).
                if (!fn_init || !fn_get_handle)
                    return;
                if (fn_init() != NVML_SUCCESS)
                    return;

                int cuda_device = 0;
                cudaGetDevice(&cuda_device);
                char pci_bus_id[NVML_PCI_BUS_ID_LEN];
                if (cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cuda_device) != cudaSuccess)
                    return;
                if (fn_get_handle(pci_bus_id, &device) != NVML_SUCCESS)
                    return;

#ifdef _WIN32
                pid = GetCurrentProcessId();
#else
                pid = static_cast<unsigned int>(getpid());
#endif
                initialized = true;
            }

            size_t getProcessMemory(FnNvmlDeviceGetProcesses fn) const {
                if (!initialized)
                    return 0;
                if (!fn)
                    return 0;
                std::vector<nvmlProcessInfo_t> procs(64);
                auto count = static_cast<unsigned int>(procs.size());
                auto status = fn(device, &count, procs.data());
                if (status == NVML_ERROR_INSUFFICIENT_SIZE) {
                    procs.resize(count);
                    status = fn(device, &count, procs.data());
                }
                if (status != NVML_SUCCESS)
                    return 0;
                std::vector<GpuProcessUsage> usage(count);
                for (unsigned int i = 0; i < count; ++i)
                    usage[i] = {procs[i].pid, procs[i].usedGpuMemory};
                return parseGpuProcessBytes(pid, std::span(usage.data(), count));
            }

            bool getDeviceMemory(size_t& used, size_t& total) const {
                if (!initialized || !fn_get_memory)
                    return false;
                nvmlMemory_v2_t memory{};
                memory.version = nvmlMemory_v2;
                if (fn_get_memory(device, &memory) != NVML_SUCCESS)
                    return false;
                used = static_cast<size_t>(memory.used);
                total = static_cast<size_t>(memory.total);
                return total >= used && total > 0;
            }

            float getUtilization() const {
                if (!initialized || !fn_get_utilization)
                    return -1.f;
                NvmlUtilization utilization{};
                if (fn_get_utilization(device, &utilization) != NVML_SUCCESS)
                    return -1.f;
                return static_cast<float>(utilization.gpu);
            }
        };

        NvmlState& nvmlState() {
            static NvmlState s;
            return s;
        }
#endif

#ifdef __APPLE__
        struct AcceleratorStats {
            // Resident GPU memory of all clients. "Alloc system memory" also
            // counts reserved address space and routinely exceeds the working set.
            size_t in_use = 0;
            float utilization = -1.f;
        };

        // Apple GPUs publish device-wide counters on their IOAccelerator
        // service, readable without privileges.
        std::optional<AcceleratorStats> readAcceleratorStats() {
            io_iterator_t services = IO_OBJECT_NULL;
            if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &services) !=
                KERN_SUCCESS)
                return std::nullopt;
            std::optional<AcceleratorStats> result;
            for (io_object_t service; !result && (service = IOIteratorNext(services)) != IO_OBJECT_NULL;) {
                const CFTypeRef stats = IORegistryEntryCreateCFProperty(service, CFSTR("PerformanceStatistics"),
                                                                        kCFAllocatorDefault, 0);
                IOObjectRelease(service);
                if (!stats)
                    continue;
                if (CFGetTypeID(stats) == CFDictionaryGetTypeID()) {
                    const auto read = [dict = static_cast<CFDictionaryRef>(stats)](const CFStringRef key, int64_t& out) {
                        const auto value = CFDictionaryGetValue(dict, key);
                        return value && CFGetTypeID(value) == CFNumberGetTypeID() &&
                               CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberSInt64Type, &out);
                    };
                    int64_t utilization = 0;
                    int64_t in_use = 0;
                    if (read(CFSTR("Device Utilization %"), utilization) && read(CFSTR("In use system memory"), in_use))
                        result = AcceleratorStats{.in_use = static_cast<size_t>(std::max<int64_t>(in_use, 0)),
                                                  .utilization = std::clamp(static_cast<float>(utilization), 0.f, 100.f)};
                }
                CFRelease(stats);
            }
            IOObjectRelease(services);
            return result;
        }

        // IOKit is sampled at most every 500 ms.
        std::optional<AcceleratorStats> acceleratorStats() {
            static std::mutex mutex;
            static std::optional<AcceleratorStats> cached;
            static auto last_sample = std::chrono::steady_clock::time_point{};
            std::lock_guard lock(mutex);
            const auto now = std::chrono::steady_clock::now();
            if (last_sample == std::chrono::steady_clock::time_point{} ||
                now - last_sample >= std::chrono::milliseconds(500)) {
                cached = readAcceleratorStats();
                last_sample = now;
            }
            return cached;
        }
#endif

    } // namespace

    size_t parseGpuProcessBytes(unsigned int pid,
                                std::span<const GpuProcessUsage> processes) {
        size_t result = 0;
        for (const auto& process : processes) {
            if (process.pid == pid &&
                process.bytes != std::numeric_limits<unsigned long long>::max())
                result = std::max(result, static_cast<size_t>(process.bytes));
        }
        return result;
    }

    GpuMemoryInfo selectGpuMemory(size_t compute_bytes, size_t graphics_bytes,
                                  size_t dxgi_bytes, size_t cuda_used, size_t cuda_total,
                                  size_t nvml_used, size_t nvml_total,
                                  size_t dxgi_budget, bool dxgi_valid) {
        GpuMemoryInfo info;
        // Compute and graphics APIs both report the same PID's total allocation.
        // Taking their sum would double-count CUDA/Vulkan interop memory.
        info.process_used = std::max(compute_bytes, graphics_bytes);
        info.process_valid = info.process_used > 0;
        if (!info.process_valid && (dxgi_valid || dxgi_bytes > 0)) {
            info.process_used = dxgi_bytes;
            info.process_valid = true;
        }
        if (dxgi_valid || dxgi_bytes > 0) {
            info.process_budget = dxgi_budget;
            info.process_over_budget = (dxgi_valid || dxgi_budget > 0) && dxgi_bytes > dxgi_budget;
        }
        // cudaMemGetInfo is device-wide: it must never stand in for this PID.
        if (nvml_total > 0) {
            info.total_used = std::min(nvml_used, nvml_total);
            info.total = nvml_total;
        } else if (cuda_total > 0) {
            info.total_used = std::min(cuda_used, cuda_total);
            info.total = cuda_total;
            info.device_estimated = true;
        }
        return info;
    }

    GpuMemoryInfo selectUnifiedGpuMemory(const size_t process_used, const size_t working_set,
                                         const size_t gpu_in_use, const size_t host_available) {
        GpuMemoryInfo info;
        info.unified_memory = true;
        info.process_valid = true;
        info.process_used = process_used;
        info.total = working_set;
        const size_t gpu_free = working_set - std::min(gpu_in_use, working_set);
        info.total_used = std::max(working_set - std::min(gpu_free, host_available), std::min(process_used, working_set));
        return info;
    }

    std::string formatGpuGiB(size_t bytes) {
        constexpr double gib = 1024.0 * 1024.0 * 1024.0;
        return std::format("{:.2f}", static_cast<double>(bytes) / gib);
    }

    const char* gpuProcessMemoryTooltipKey(const GpuMemoryInfo& info) {
        if (info.unified_memory)
            return "ui.vram_process_metal_tooltip";
        return info.process_valid ? "ui.vram_process_nvml_tooltip" : "ui.vram_process_unavailable_tooltip";
    }

    const char* gpuDeviceMemoryTooltipKey(const GpuMemoryInfo& info) {
        if (info.unified_memory)
            return "ui.vram_device_unified_tooltip";
        return info.device_estimated ? "ui.vram_device_cuda_tooltip" : "ui.vram_device_nvml_tooltip";
    }

    GpuMemoryInfo queryGpuMemory(const lfs::core::GpuBackend backend) {
        const auto device = lfs::core::gpu_backend_device_info(backend);
#ifdef __APPLE__
        // Metal and MoltenVK share one GPU and the host's memory.
        if (device && device->supports_process_memory_budget) {
            const auto stats = acceleratorStats();
            const auto host = lfs::core::host_metrics::memory();
            if (stats && host) {
                auto info = selectUnifiedGpuMemory(device->process_memory_used_bytes,
                                                   device->process_memory_budget_bytes, stats->in_use,
                                                   host->available_bytes);
                info.device_name = shortenGpuDeviceName(device->name);
                info.gpu_utilization_percent = stats->utilization;
                info.gpu_utilization_valid = true;
                return info;
            }
        }
#endif
        if (backend == lfs::core::GpuBackend::Vulkan || backend == lfs::core::GpuBackend::Metal) {
            GpuMemoryInfo info;
            if (device) {
                info.device_name = shortenGpuDeviceName(device->name);
                info.total = device->total_memory_bytes;
            }
            info.uses_process_budget = true;
            if (device && device->supports_process_memory_budget) {
                info.process_budget = device->process_memory_budget_bytes;
                info.process_budget_used = device->process_memory_used_bytes;
            }
            return info;
        }
        if (!device) {
            return {};
        }
#if !LFS_HAS_CUDA
        GpuMemoryInfo info;
        info.device_name = shortenGpuDeviceName(device->name);
        info.total = device->total_memory_bytes;
        return info;
#else
        // NVML and the driver are sampled at most every 500 ms.
        static std::mutex cache_mutex;
        static GpuMemoryInfo cached;
        static auto last_sample = std::chrono::steady_clock::time_point{};
        std::lock_guard lock(cache_mutex);
        const auto now = std::chrono::steady_clock::now();
        if (last_sample != std::chrono::steady_clock::time_point{} &&
            now - last_sample < std::chrono::milliseconds(500))
            return cached;

        const auto memory = lfs::core::gpu_backend_memory_info(backend);
        const bool cuda_valid = memory.total_bytes > 0 && memory.total_bytes >= memory.free_bytes;
        size_t nvml_used = 0;
        size_t nvml_total = 0;
        nvmlState().getDeviceMemory(nvml_used, nvml_total);
        size_t dxgi_bytes = 0;
        size_t dxgi_budget = 0;
        bool dxgi_valid = false;
#ifdef _WIN32
        dxgi_valid = dxgiState().getProcessMemory(dxgi_bytes, dxgi_budget);
#endif
        auto info = selectGpuMemory(nvmlState().getProcessMemory(nvmlState().fn_get_compute),
                                    nvmlState().getProcessMemory(nvmlState().fn_get_graphics),
                                    dxgi_bytes, cuda_valid ? memory.total_bytes - memory.free_bytes : 0,
                                    cuda_valid ? (device->total_memory_bytes ? std::min(memory.total_bytes, device->total_memory_bytes) : memory.total_bytes) : 0,
                                    nvml_used, nvml_total, dxgi_budget, dxgi_valid);
        info.device_name = shortenGpuDeviceName(device->name);
        info.gpu_utilization_percent = nvmlState().getUtilization();
        info.gpu_utilization_valid = info.gpu_utilization_percent >= 0.f;
        cached = info;
        last_sample = now;
        return info;
#endif
    }

    float queryGpuUtilization() {
#ifdef __APPLE__
        if (const auto stats = acceleratorStats())
            return stats->utilization;
#endif
        if (lfs::core::default_gpu_backend() == lfs::core::GpuBackend::Vulkan) {
            return -1.f;
        }
#if LFS_HAS_CUDA
        return nvmlState().getUtilization();
#else
        return -1.f;
#endif
    }

} // namespace lfs::vis::gui
