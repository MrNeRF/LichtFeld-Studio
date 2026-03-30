/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gpu_memory_query.hpp"

#include <cuda_runtime.h>

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
        // NVML returns NVML_VALUE_NOT_AVAILABLE under WDDM.
        struct DxgiMemoryState {
            IDXGIAdapter3* adapter3 = nullptr;
            bool init_done = false;

            DxgiMemoryState(const DxgiMemoryState&) = delete;
            DxgiMemoryState& operator=(const DxgiMemoryState&) = delete;
            DxgiMemoryState() = default;

            ~DxgiMemoryState() {
                if (adapter3)
                    adapter3->Release();
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
#else
        // Linux: NVML per-process memory works correctly.
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

        struct NvmlState {
            bool initialized = false;
            NvmlDevice device = nullptr;
            unsigned int pid = 0;
            void* lib = nullptr;
            FnNvmlDeviceGetComputeRunningProcesses fn_get_procs = nullptr;

            NvmlState() {
                lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
                if (!lib)
                    lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
                if (!lib)
                    return;

                auto load = [this](const char* name) -> void* {
                    return dlsym(lib, name);
                };

                auto fn_init = reinterpret_cast<FnNvmlInit>(load("nvmlInit_v2"));
                auto fn_get_handle = reinterpret_cast<FnNvmlDeviceGetHandleByPciBusId>(
                    load("nvmlDeviceGetHandleByPciBusId_v2"));
                fn_get_procs = reinterpret_cast<FnNvmlDeviceGetComputeRunningProcesses>(
                    load("nvmlDeviceGetComputeRunningProcesses_v3"));

                if (!fn_init || !fn_get_handle || !fn_get_procs)
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

                pid = static_cast<unsigned int>(getpid());
                initialized = true;
            }

            size_t getProcessMemory() const {
                if (!initialized)
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
        };

        NvmlState& nvmlState() {
            static NvmlState s;
            return s;
        }
#endif

    } // namespace

    GpuMemoryInfo queryGpuMemory() {
        GpuMemoryInfo info;

        int cuda_device = 0;
        if (cudaGetDevice(&cuda_device) == cudaSuccess) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, cuda_device) == cudaSuccess)
                info.device_name = shortenGpuDeviceName(prop.name);
        }

        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        info.total = total_mem;
        info.total_used = total_mem - free_mem;
#ifdef _WIN32
        info.process_used = dxgiState().getProcessMemory();
#else
        info.process_used = nvmlState().getProcessMemory();
#endif
        if (info.process_used > info.total)
            info.process_used = 0;

        return info;
    }

} // namespace lfs::vis::gui
