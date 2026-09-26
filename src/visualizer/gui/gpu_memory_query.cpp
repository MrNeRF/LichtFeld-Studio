/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gpu_memory_query.hpp"
#include "core/gpu_device_info.hpp"
#include "core/tensor_backend.hpp"

#if LFS_HAS_CUDA
#include <cuda_runtime.h>
#endif

#ifdef _WIN32
#include <dxgi1_4.h>
#include <windows.h>
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

#if LFS_HAS_CUDA
        // NVML: process memory on Linux; utilization on Linux and Windows.
        using NvmlDevice = void*;
        enum { NVML_SUCCESS = 0 };
        constexpr int NVML_PCI_BUS_ID_LEN = 32;

        struct NvmlProcessInfo {
            unsigned int pid;
            unsigned long long usedGpuMemory;
            unsigned int gpuInstanceId;
            unsigned int computeInstanceId;
        };

        using FnNvmlInit = int (*)();
        using FnNvmlDeviceGetHandleByPciBusId = int (*)(const char*, NvmlDevice*);
        using FnNvmlDeviceGetComputeRunningProcesses = int (*)(NvmlDevice, unsigned int*, NvmlProcessInfo*);
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
            FnNvmlDeviceGetComputeRunningProcesses fn_get_procs = nullptr;
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
                fn_get_procs = reinterpret_cast<FnNvmlDeviceGetComputeRunningProcesses>(
                    load("nvmlDeviceGetComputeRunningProcesses_v3"));
                fn_get_utilization = reinterpret_cast<FnNvmlDeviceGetUtilizationRates>(
                    load("nvmlDeviceGetUtilizationRates"));

                // Utilization only needs init + handle + getUtilizationRates.
                // Process memory also needs get_procs (Linux path).
                if (!fn_init || !fn_get_handle || !fn_get_utilization)
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

#ifndef _WIN32
                pid = static_cast<unsigned int>(getpid());
#endif
                initialized = true;
            }

#ifndef _WIN32
            size_t getProcessMemory() const {
                if (!initialized || !fn_get_procs)
                    return 0;
                unsigned int count = 64;
                NvmlProcessInfo procs[64];
                if (fn_get_procs(device, &count, procs) != NVML_SUCCESS)
                    return 0;
                for (unsigned int i = 0; i < count; ++i) {
                    if (procs[i].pid == pid)
                        return static_cast<size_t>(procs[i].usedGpuMemory);
                }
                return 0;
            }
#endif

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

    } // namespace

    GpuMemoryInfo queryGpuMemory(const lfs::core::GpuBackend backend) {
        GpuMemoryInfo info;
        const auto device = lfs::core::gpu_backend_device_info(backend);
        if (device) {
            info.device_name = shortenGpuDeviceName(device->name);
            info.total = device->total_memory_bytes;
        }
        if (backend == lfs::core::GpuBackend::Vulkan || backend == lfs::core::GpuBackend::Metal) {
            info.uses_process_budget = true;
            if (device && device->supports_process_memory_budget) {
                info.process_budget = device->process_memory_budget_bytes;
                info.process_budget_used = device->process_memory_used_bytes;
            }
            return info;
        }
        if (!device) {
            return info;
        }
#if !LFS_HAS_CUDA
        return info;
#else
        const auto memory = lfs::core::gpu_backend_memory_info(backend);
        info.total = memory.total_bytes;
        info.total_used = memory.total_bytes >= memory.free_bytes
                              ? memory.total_bytes - memory.free_bytes
                              : 0;
#ifdef _WIN32
        info.process_used = dxgiState().getProcessMemory();
#else
        info.process_used = nvmlState().getProcessMemory();
#endif
        info.gpu_utilization_percent = nvmlState().getUtilization();
        info.gpu_utilization_valid = info.gpu_utilization_percent >= 0.f;
        if (info.process_used > info.total)
            info.process_used = 0;

        return info;
#endif
    }

    float queryGpuUtilization() {
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
