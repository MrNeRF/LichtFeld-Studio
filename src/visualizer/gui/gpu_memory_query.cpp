/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gpu_memory_query.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cuda_runtime.h>
#include <format>
#include <limits>
#include <mutex>
#include <nvml.h>

#ifdef _WIN32
#include <dxgi1_4.h>
#include <windows.h>
#else
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

#ifdef _WIN32
        // Windows: DXGI QueryVideoMemoryInfo for per-process GPU memory.
        // NVML process memory returns NVML_VALUE_NOT_AVAILABLE under WDDM, but
        // device utilization rates work and are used for the GPU% meter.
        struct DxgiMemoryState {
            IDXGIAdapter3* adapter3 = nullptr;
            bool init_done = false;

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

                matchByVram(factory, cuda_device);
                factory->Release();
            }

            size_t getProcessMemory() {
                ensureInit();
                if (!adapter3)
                    return 0;
                DXGI_QUERY_VIDEO_MEMORY_INFO mem_info{};
                if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(
                        0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mem_info)))
                    return static_cast<size_t>(mem_info.CurrentUsage);
                return 0;
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
                    if (factory->EnumAdapters(i, &adapter) == DXGI_ERROR_NOT_FOUND)
                        break;
                    DXGI_ADAPTER_DESC desc{};
                    if (SUCCEEDED(adapter->GetDesc(&desc)) &&
                        desc.AdapterLuid.LowPart == cuda_luid.LowPart &&
                        desc.AdapterLuid.HighPart == cuda_luid.HighPart) {
                        adapter->QueryInterface(__uuidof(IDXGIAdapter3),
                                                reinterpret_cast<void**>(&adapter3));
                        adapter->Release();
                        return true;
                    }
                    adapter->Release();
                }
                return false;
            }

            // Fallback: match by dedicated VRAM size (unreliable with identical GPUs).
            void matchByVram(IDXGIFactory1* factory, int cuda_device) {
                cudaDeviceProp props{};
                size_t cuda_total = 0;
                if (cudaGetDeviceProperties(&props, cuda_device) == cudaSuccess)
                    cuda_total = props.totalGlobalMem;

                for (UINT i = 0;; ++i) {
                    IDXGIAdapter* adapter = nullptr;
                    if (factory->EnumAdapters(i, &adapter) == DXGI_ERROR_NOT_FOUND)
                        break;
                    DXGI_ADAPTER_DESC desc{};
                    if (SUCCEEDED(adapter->GetDesc(&desc))) {
                        auto dxgi_vram = static_cast<size_t>(desc.DedicatedVideoMemory);
                        size_t diff = dxgi_vram > cuda_total ? dxgi_vram - cuda_total
                                                             : cuda_total - dxgi_vram;
                        constexpr size_t TOLERANCE = 512ULL * 1024 * 1024;
                        if (cuda_total > 0 && diff < TOLERANCE) {
                            adapter->QueryInterface(__uuidof(IDXGIAdapter3),
                                                    reinterpret_cast<void**>(&adapter3));
                            adapter->Release();
                            return;
                        }
                    }
                    adapter->Release();
                }
            }
        };

        DxgiMemoryState& dxgiState() {
            static DxgiMemoryState s;
            return s;
        }
#endif

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
                std::array<nvmlProcessInfo_t, 256> procs{};
                auto count = static_cast<unsigned int>(procs.size());
                if (fn(device, &count, procs.data()) != NVML_SUCCESS)
                    return 0;
                std::array<GpuProcessUsage, 256> usage{};
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
                                  size_t nvml_used, size_t nvml_total) {
        GpuMemoryInfo info;
        info.process_used = std::max(compute_bytes, graphics_bytes);
        if (info.process_used == 0)
            info.process_used = dxgi_bytes;
        if (info.process_used == 0) {
            info.process_used = cuda_used;
            info.process_estimated = true;
        }
        if (nvml_total > 0 && nvml_total >= nvml_used) {
            info.total_used = nvml_used;
            info.total = nvml_total;
        } else if (cuda_total > 0 && cuda_total >= cuda_used) {
            info.total_used = cuda_used;
            info.total = cuda_total;
            info.device_estimated = true;
        }
        return info;
    }

    std::string formatGpuGiB(size_t bytes) {
        constexpr double gib = 1024.0 * 1024.0 * 1024.0;
        return std::format("{:.2f}", static_cast<double>(bytes) / gib);
    }

    GpuMemoryInfo queryGpuMemory() {
        static std::mutex cache_mutex;
        static GpuMemoryInfo cached;
        static auto last_sample = std::chrono::steady_clock::time_point{};
        std::lock_guard lock(cache_mutex);
        const auto now = std::chrono::steady_clock::now();
        if (last_sample != std::chrono::steady_clock::time_point{} &&
            now - last_sample < std::chrono::milliseconds(500))
            return cached;
        std::string device_name;
        int cuda_device = 0;
        if (cudaGetDevice(&cuda_device) == cudaSuccess) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, cuda_device) == cudaSuccess)
                device_name = shortenGpuDeviceName(prop.name);
        }

        size_t free_mem = 0;
        size_t total_mem = 0;
        const auto cuda_valid = cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess &&
                                total_mem >= free_mem;
        size_t nvml_used = 0;
        size_t nvml_total = 0;
        nvmlState().getDeviceMemory(nvml_used, nvml_total);
        size_t dxgi_bytes = 0;
#ifdef _WIN32
        dxgi_bytes = dxgiState().getProcessMemory();
#endif
        auto info = selectGpuMemory(nvmlState().getProcessMemory(nvmlState().fn_get_compute),
                                    nvmlState().getProcessMemory(nvmlState().fn_get_graphics),
                                    dxgi_bytes, cuda_valid ? total_mem - free_mem : 0,
                                    cuda_valid ? total_mem : 0, nvml_used, nvml_total);
        info.device_name = std::move(device_name);
        info.gpu_utilization_percent = nvmlState().getUtilization();
        info.gpu_utilization_valid = info.gpu_utilization_percent >= 0.f;
        cached = info;
        last_sample = now;
        return info;
    }

    float queryGpuUtilization() {
        return nvmlState().getUtilization();
    }

} // namespace lfs::vis::gui
