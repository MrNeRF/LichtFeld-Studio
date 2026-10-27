/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/tensor_impl.hpp"
#include "../backend_driver.hpp"
#include "../gpu_backend_ops.hpp"
#include "../native_export.hpp"
#include "../tensor_vulkan_interop.hpp"
#include "core/assert.hpp"
#include "core/cuda/driver_entry.hpp"
#include "core/cuda_error.hpp"
#include "core/gpu_device_selection.hpp"
#include <atomic>
#include <charconv>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <format>

namespace lfs::core::internal {
    namespace {
        class CudaDriver final : public BackendDriver {
            std::atomic<int> selection_{-1};

        public:
            std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_interop(VulkanInteropDevice device) override { return make_cuda_driver_vulkan_interop(device); }
            SplatPublication publication() const override { return cuda_vulkan_direct_interop() ? SplatPublication::RendererStorage : SplatPublication::Copied; }
            const char* name() const noexcept override { return "CUDA"; }
            bool available() override {
                static const bool available = [] {
                    int count = 0;
                    if (cudaGetDeviceCount(&count) != cudaSuccess) {
                        (void)cudaGetLastError();
                        return false;
                    }
                    return count > 0;
                }();
                return available;
            }
            bool live() noexcept override {
                if (!available())
                    return false;
                int ordinal = 0, active = 0;
                CUdevice device{};
                unsigned flags = 0;
                return cudaGetDevice(&ordinal) == cudaSuccess &&
                       LFS_CUDA_DRIVER(cuDeviceGet)(&device, ordinal) == CUDA_SUCCESS &&
                       LFS_CUDA_DRIVER(cuDevicePrimaryCtxGetState)(device, &flags, &active) == CUDA_SUCCESS && active != 0;
            }
            [[noreturn]] void throw_unavailable() override {
                throw TensorError(std::format("GPU backend '{}' is unavailable", name()));
            }
            int device_count() override {
                int count = 0;
                LFS_CUDA_CHECK_MSG(cudaGetDeviceCount(&count), "tensor CUDA device count");
                return count;
            }
            std::vector<VulkanHardwareDevice> enumerate_hardware() override {

                std::vector<VulkanHardwareDevice> devices;
                int count = 0;
                if (cudaGetDeviceCount(&count) != cudaSuccess) {
                    (void)cudaGetLastError();
                    return devices;
                }
                constexpr char hex[] = "0123456789abcdef";
                for (int index = 0; index < count; ++index) {
                    cudaDeviceProp properties{};
                    if (cudaGetDeviceProperties(&properties, index) != cudaSuccess) {
                        (void)cudaGetLastError();
                        continue;
                    }
                    VulkanHardwareDevice item;
                    item.name = properties.name;
                    item.vendor_id = 0x10de;
                    item.index = index;
                    item.discrete = !properties.integrated;
                    item.shared_memory = properties.integrated;
                    item.total_memory_bytes = properties.totalGlobalMem;
                    for (const unsigned char byte : properties.uuid.bytes) {
                        item.uuid += hex[byte >> 4];
                        item.uuid += hex[byte & 15];
                    }
#ifdef LFS_MIN_SM
                    if (properties.major * 10 + properties.minor < LFS_MIN_SM)
                        item.tensor_unavailable_reason = std::format("Requires compute capability {}.{}", LFS_MIN_SM / 10, LFS_MIN_SM % 10);
#endif
                    devices.push_back(std::move(item));
                }
                return devices;
            }
            int selected_device() override {

                const int selected = selection_.load(std::memory_order_acquire);
                if (selected >= 0)
                    return selected;
                int current = 0;
                if (cudaGetDevice(&current) != cudaSuccess)
                    (void)cudaGetLastError();
                return current;
            }
            void bind_thread() override {

                const int selected = selection_.load(std::memory_order_acquire);
                if (selected < 0)
                    return; // Preserve the existing automatic/default device behavior.
                int current = -1;
                LFS_CUDA_CHECK_MSG(cudaGetDevice(&current), "query selected CUDA worker device");
                if (current != selected)
                    LFS_CUDA_CHECK_MSG(cudaSetDevice(selected), "activate selected CUDA worker device");
            }
            MemoryInfo memory_info(bool include_pool_stats) override {

                MemoryInfo result = MemoryInfo::cuda();
#if CUDART_VERSION >= 12080
                if (include_pool_stats) {
                    int device = 0;
                    cudaMemPool_t pool = nullptr;
                    if (cudaGetDevice(&device) == cudaSuccess &&
                        cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess && pool != nullptr) {
                        uint64_t value = 0;
                        if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &value) == cudaSuccess)
                            result.pool_used_current = static_cast<size_t>(value);
                        if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &value) == cudaSuccess)
                            result.pool_reserved_current = static_cast<size_t>(value);
                        if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &value) == cudaSuccess)
                            result.pool_used_high = static_cast<size_t>(value);
                        if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &value) == cudaSuccess)
                            result.pool_reserved_high = static_cast<size_t>(value);
                    }
                }
#else
                (void)include_pool_stats;
#endif
                return result;
            }
            std::optional<GpuDeviceInfo> device_info(int ordinal) override {
                const int device_index = ordinal < 0 ? selected_device() : ordinal;
                cudaDeviceProp properties{};
                if (cudaGetDeviceProperties(&properties, device_index) != cudaSuccess)
                    return std::nullopt;
                GpuDeviceInfo result{.name = properties.name,
                                     .total_memory_bytes = properties.totalGlobalMem};
                static_assert(sizeof(properties.uuid.bytes) == result.uuid.size());
                std::memcpy(result.uuid.data(), properties.uuid.bytes, result.uuid.size());
                result.compute_capability_major = properties.major;
                return result;
            }
            std::optional<std::string> validate_options(const TensorBackendOptions& options, int& ordinal) override {
                ordinal = -1;
                if (options.cuda_device.empty())
                    return {};
                const auto devices = enumerate_hardware();
                const auto selected = find_gpu_device<VulkanHardwareDevice>(devices, options.cuda_device);
                if (!selected || !devices[*selected].tensor_unavailable_reason.empty())
                    return "Selected CUDA GPU is unavailable or incompatible";
                ordinal = devices[*selected].index;
                return {};
            }
            void set_selection(int ordinal) noexcept override { selection_.store(ordinal, std::memory_order_release); }
            GpuBackendOps& activate() override {
                (void)tensor_backend_options();
                bind_thread();
                static CudaBackendOps* const ops = new CudaBackendOps();
                return *ops;
            }
            size_t allocation_bytes(size_t bytes) override { return cuda_allocation_size(bytes); }
            lfs::Status shutdown() override { return {}; }
            TensorExportDevice dlpack_device(const Tensor& tensor) override { return cuda_dlpack_device(tensor); }
            Tensor prepare_dlpack_export(const Tensor& tensor, std::optional<int64_t> consumer) override { return cuda_prepare_dlpack_export(tensor, consumer); }
            BackendExportCapabilities export_capabilities() const noexcept override { return {.dlpack = true}; }
        };
    } // namespace
    void register_cuda_backend(BackendRegistry& registry) {
        static CudaDriver driver;
        registry[static_cast<size_t>(GpuBackend::CUDA)] = &driver;
    }
} // namespace lfs::core::internal
