/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "diagnostics/gpu_backend.hpp"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#ifndef _WIN32
#include <dlfcn.h>
#include <nvml.h>
#include <unistd.h>
#endif

namespace lfs::diagnostics {
    namespace {
#ifndef _WIN32
        // NVML finds the device through the CUDA runtime's PCI bus id.
        struct NvmlMemorySample {
            std::size_t process = 0;
            std::size_t used = 0;
            std::size_t total = 0;
        };

        [[nodiscard]] NvmlMemorySample nvml_memory_sample() {
            using Device = void*;
            struct DeviceMemoryInfo {
                unsigned long long total;
                unsigned long long free;
                unsigned long long used;
            };
            struct Api {
                void* lib = nullptr;
                Device device = nullptr;
                std::mutex mutex;
                int (*handle)(const char*, Device*) = nullptr;
                int (*processes)(Device, unsigned int*, nvmlProcessInfo_t*) = nullptr;
                int (*graphics)(Device, unsigned int*, nvmlProcessInfo_t*) = nullptr;
                int (*memory)(Device, DeviceMemoryInfo*) = nullptr;
                Api() {
                    lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
                    if (!lib)
                        return;
                    const auto init = reinterpret_cast<int (*)()>(dlsym(lib, "nvmlInit_v2"));
                    handle = reinterpret_cast<int (*)(const char*, Device*)>(
                        dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2"));
                    processes = reinterpret_cast<int (*)(Device, unsigned int*, nvmlProcessInfo_t*)>(
                        dlsym(lib, "nvmlDeviceGetComputeRunningProcesses_v3"));
                    graphics = reinterpret_cast<int (*)(Device, unsigned int*, nvmlProcessInfo_t*)>(
                        dlsym(lib, "nvmlDeviceGetGraphicsRunningProcesses_v3"));
                    memory = reinterpret_cast<int (*)(Device, DeviceMemoryInfo*)>(
                        dlsym(lib, "nvmlDeviceGetMemoryInfo"));
                    if (!init || !handle || !processes || init() != 0) {
                        handle = nullptr;
                        return;
                    }
                }
                Device getDevice() {
                    std::lock_guard lock(mutex);
                    if (device)
                        return device;
                    if (!handle)
                        return nullptr;
                    int cuda_device = 0;
                    char bus_id[32]{};
                    if (cudaGetDevice(&cuda_device) != cudaSuccess ||
                        cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), cuda_device) != cudaSuccess ||
                        handle(bus_id, &device) != 0)
                        device = nullptr;
                    return device;
                }
            };
            static Api api;
            NvmlMemorySample sample;
            const auto device = api.getDevice();
            if (!device)
                return sample;
            if (api.memory) {
                DeviceMemoryInfo info{};
                if (api.memory(device, &info) == 0) {
                    sample.used = static_cast<std::size_t>(info.used);
                    sample.total = static_cast<std::size_t>(info.total);
                }
            }
            for (const auto query : {api.processes, api.graphics}) {
                if (!query)
                    continue;
                std::vector<nvmlProcessInfo_t> info(64);
                auto count = static_cast<unsigned int>(info.size());
                auto status = query(device, &count, info.data());
                if (status == NVML_ERROR_INSUFFICIENT_SIZE) {
                    info.resize(count);
                    status = query(device, &count, info.data());
                }
                if (status != NVML_SUCCESS)
                    continue;
                for (unsigned int i = 0; i < count; ++i) {
                    if (info[i].pid == static_cast<unsigned int>(getpid()) &&
                        info[i].usedGpuMemory != NVML_VALUE_NOT_AVAILABLE)
                        sample.process = std::max(sample.process,
                                                  static_cast<std::size_t>(info[i].usedGpuMemory));
                }
            }
            return sample;
        }
#endif
        bool cuda_context_live() noexcept {
            static const auto get_state = [] {
                void* entry = nullptr;
                if (cudaGetDriverEntryPointByVersion("cuDevicePrimaryCtxGetState", &entry, 7000,
                                                     cudaEnableDefault) != cudaSuccess)
                    return decltype(&cuDevicePrimaryCtxGetState){};
                return reinterpret_cast<decltype(&cuDevicePrimaryCtxGetState)>(entry);
            }();
            int device = 0;
            unsigned flags = 0;
            int active = 0;
            return get_state && cudaGetDevice(&device) == cudaSuccess &&
                   get_state(device, &flags, &active) == CUDA_SUCCESS && active != 0;
        }

        [[nodiscard]] bool sample_cuda_used_bytes(std::size_t& used_bytes,
                                                  std::size_t* total_bytes = nullptr) noexcept {
            if (!cuda_context_live())
                return false;
            std::size_t free_bytes = 0;
            std::size_t total = 0;
            if (cudaMemGetInfo(&free_bytes, &total) != cudaSuccess || total < free_bytes) {
                return false;
            }
            used_bytes = total - free_bytes;
            if (total_bytes) {
                *total_bytes = total;
            }
            return true;
        }

        void sample_process(VramProcessSnapshot& process) {
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            const bool cuda_live = cuda_context_live();
            if (cuda_live && cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && total_bytes >= free_bytes) {
                process.cuda_used = total_bytes - free_bytes;
                process.cuda_total = total_bytes;
                process.cuda_memory_valid = true;
            }
#ifndef _WIN32
            {
                const auto usage = nvml_memory_sample();
                process.process_used = usage.process;
                process.total = usage.total ? usage.total : process.cuda_total;
                process.total_used = std::min(usage.total ? usage.used : process.cuda_used, process.total);
                process.process_memory_valid = usage.process > 0;
            }
#endif

#if CUDART_VERSION >= 12080
            int device = 0;
            if (cuda_live && cudaGetDevice(&device) == cudaSuccess) {
                cudaMemPool_t pool = nullptr;
                if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
                    std::uint64_t used = 0;
                    std::uint64_t reserved = 0;
                    if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used) == cudaSuccess &&
                        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved) == cudaSuccess) {
                        process.cuda_pool_used = static_cast<std::size_t>(used);
                        process.cuda_pool_reserved = static_cast<std::size_t>(reserved);
                        process.cuda_pool_valid = true;
                    }
                }
            }
#endif
        }

        std::optional<std::size_t> process_memory_bytes() {
#ifndef _WIN32
            if (const auto bytes = nvml_memory_sample().process)
                return bytes;
#endif
            return std::nullopt;
        }

        void* create_event() noexcept {
            cudaEvent_t event = nullptr;
            return cudaEventCreateWithFlags(&event, cudaEventDefault) == cudaSuccess ? event : nullptr;
        }

        bool record_event(void* event, void* stream) noexcept {
            return cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)) == cudaSuccess;
        }

        GpuEventStatus elapsed_time(void* start, void* stop, float& elapsed_ms) noexcept {
            const auto status = cudaEventQuery(static_cast<cudaEvent_t>(stop));
            if (status == cudaErrorNotReady)
                return GpuEventStatus::Pending;
            if (status == cudaSuccess && cudaEventElapsedTime(&elapsed_ms,
                                                              static_cast<cudaEvent_t>(start), static_cast<cudaEvent_t>(stop)) == cudaSuccess)
                return GpuEventStatus::Ready;
            return GpuEventStatus::Failed;
        }

        const GpuDiagnosticsBackend kCudaBackend{
            cuda_context_live, sample_cuda_used_bytes, sample_process, process_memory_bytes,
            create_event, record_event, elapsed_time};
        [[maybe_unused]] const bool kRegistered = register_gpu_diagnostics_backend(kCudaBackend);
    } // namespace
} // namespace lfs::diagnostics
