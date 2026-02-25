/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gpu_memory_query.hpp"

#include <cuda_runtime.h>

#ifdef _WIN32
#include <process.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace lfs::vis::gui {

    namespace {

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
#ifdef _WIN32
            HMODULE lib = nullptr;
#else
            void* lib = nullptr;
#endif
            FnNvmlDeviceGetComputeRunningProcesses fn_get_procs = nullptr;

            NvmlState() {
#ifdef _WIN32
                lib = LoadLibraryA("nvml.dll");
#else
                lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
                if (!lib)
                    lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
#endif
                if (!lib)
                    return;

                auto load = [this](const char* name) -> void* {
#ifdef _WIN32
                    return reinterpret_cast<void*>(GetProcAddress(lib, name));
#else
                    return dlsym(lib, name);
#endif
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

#ifdef _WIN32
                pid = static_cast<unsigned int>(_getpid());
#else
                pid = static_cast<unsigned int>(getpid());
#endif
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

    } // namespace

    GpuMemoryInfo queryGpuMemory() {
        GpuMemoryInfo info;

        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        info.total = total_mem;
        info.total_used = total_mem - free_mem;
        info.process_used = nvmlState().getProcessMemory();
        if (info.process_used > info.total)
            info.process_used = 0;

        return info;
    }

} // namespace lfs::vis::gui
