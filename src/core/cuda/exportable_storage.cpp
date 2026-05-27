/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/exportable_storage.hpp"

#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <format>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lfs::core {

    namespace {

#ifdef _WIN32
        constexpr CUmemAllocationHandleType kCudaHandleType = CU_MEM_HANDLE_TYPE_WIN32;
#else
        constexpr CUmemAllocationHandleType kCudaHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif

        constexpr const char* kProfilerLabel = "splat.exportable_block";

        std::string cu_error(CUresult r) {
            const char* name = nullptr;
            const char* desc = nullptr;
            cuGetErrorName(r, &name);
            cuGetErrorString(r, &desc);
            return std::format("CUDA driver error {}: {}",
                               name ? name : "?",
                               desc ? desc : "?");
        }

        std::size_t align_up(std::size_t value, std::size_t alignment) {
            return ((value + alignment - 1) / alignment) * alignment;
        }

        bool vmm_supported(int device) {
            int supported = 0;
            const CUresult r = cuDeviceGetAttribute(
                &supported,
                CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                device);
            return r == CUDA_SUCCESS && supported != 0;
        }

        struct OwnedAllocation {
            CUdeviceptr va = 0;
            std::size_t mapped_size = 0;
            CUmemGenericAllocationHandle mem_handle = 0;
            bool mapped = false;
            bool created = false;
            bool reserved = false;
            ExportNativeHandle native = ExportNativeHandle{};
            bool native_valid = false;
        };

        void teardown(OwnedAllocation& a) {
            if (a.mapped) {
                cuMemUnmap(a.va, a.mapped_size);
                a.mapped = false;
            }
            if (a.created) {
                cuMemRelease(a.mem_handle);
                a.created = false;
            }
            if (a.reserved) {
                cuMemAddressFree(a.va, a.mapped_size);
                a.reserved = false;
            }
            if (a.native_valid) {
#ifdef _WIN32
                CloseHandle(a.native);
#else
                if (a.native >= 0) {
                    ::close(a.native);
                }
#endif
                a.native_valid = false;
            }
        }

    } // namespace

    std::expected<std::shared_ptr<ExportableBlock>, std::string>
    allocateExportableDeviceBlock(std::size_t size, int device) {
        if (size == 0) {
            return std::unexpected("allocateExportableDeviceBlock: size must be non-zero");
        }
        if (!vmm_supported(device)) {
            return std::unexpected(std::format(
                "allocateExportableDeviceBlock: device {} does not support virtual memory management",
                device));
        }

        // CUDA VMM allocations require a current context. cudaSetDevice ensures one exists.
        if (const auto err = cudaSetDevice(device); err != cudaSuccess) {
            return std::unexpected(std::format("cudaSetDevice({}) failed: {}",
                                               device,
                                               cudaGetErrorString(err)));
        }

        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        prop.requestedHandleTypes = kCudaHandleType;

        std::size_t granularity = 0;
        if (const auto r = cuMemGetAllocationGranularity(&granularity, &prop,
                                                         CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            r != CUDA_SUCCESS) {
            return std::unexpected("cuMemGetAllocationGranularity failed: " + cu_error(r));
        }
        if (granularity == 0) {
            granularity = std::size_t(2) << 20;
        }
        const std::size_t aligned_size = align_up(size, granularity);

        OwnedAllocation a{};
        a.mapped_size = aligned_size;

        if (const auto r = cuMemAddressReserve(&a.va, aligned_size, 0, 0, 0);
            r != CUDA_SUCCESS) {
            return std::unexpected("cuMemAddressReserve failed: " + cu_error(r));
        }
        a.reserved = true;

        if (const auto r = cuMemCreate(&a.mem_handle, aligned_size, &prop, 0);
            r != CUDA_SUCCESS) {
            teardown(a);
            return std::unexpected("cuMemCreate (exportable) failed: " + cu_error(r));
        }
        a.created = true;

        if (const auto r = cuMemMap(a.va, aligned_size, 0, a.mem_handle, 0);
            r != CUDA_SUCCESS) {
            teardown(a);
            return std::unexpected("cuMemMap failed: " + cu_error(r));
        }
        a.mapped = true;

        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (const auto r = cuMemSetAccess(a.va, aligned_size, &access, 1);
            r != CUDA_SUCCESS) {
            teardown(a);
            return std::unexpected("cuMemSetAccess failed: " + cu_error(r));
        }

        // Zero the whole exportable block once. Capacity-backed tensor views can
        // expose slack rows before training fills them, and Vulkan reads those
        // views directly in the zero-copy path.
        if (const auto err = cudaMemset(reinterpret_cast<void*>(a.va), 0, aligned_size);
            err != cudaSuccess) {
            teardown(a);
            return std::unexpected(std::format("cudaMemset on exportable block failed: {}",
                                               cudaGetErrorString(err)));
        }

#ifdef _WIN32
        void* native = nullptr;
        if (const auto r = cuMemExportToShareableHandle(&native, a.mem_handle,
                                                        kCudaHandleType, 0);
            r != CUDA_SUCCESS) {
            teardown(a);
            return std::unexpected("cuMemExportToShareableHandle (Win32) failed: " + cu_error(r));
        }
        a.native = native;
#else
        int native = -1;
        if (const auto r = cuMemExportToShareableHandle(&native, a.mem_handle,
                                                        kCudaHandleType, 0);
            r != CUDA_SUCCESS) {
            teardown(a);
            return std::unexpected("cuMemExportToShareableHandle (fd) failed: " + cu_error(r));
        }
        a.native = native;
#endif
        a.native_valid = true;

        void* const device_ptr = reinterpret_cast<void*>(a.va);

        // Deliberately do NOT recordAllocation here. The six splat tensor views
        // are published per-tensor via record_splat_vram_breakdown (trainer.cpp)
        // through recordCurrentBytes; recording the underlying block here too
        // would double-count the same physical memory under whichever scope was
        // active at allocation time. The block stays visible via the model.*
        // rows. Visibility of the block itself comes from the cuda.exportable
        // process-snapshot field surfaced by the HUD.
        diagnostics::VramProfiler::instance().setExportableSplatBytes(aligned_size);

        auto* block = new ExportableBlock{
            .device_ptr = device_ptr,
            .size = aligned_size,
            .handle = ExportHandle{.native = a.native, .size = aligned_size},
        };

        LOG_INFO("Exportable CUDA block: device_ptr={} size={} MiB granularity={}",
                 device_ptr,
                 aligned_size >> 20,
                 granularity);

        return std::shared_ptr<ExportableBlock>(
            block,
            [owned = a](ExportableBlock* p) mutable {
                diagnostics::VramProfiler::instance().setExportableSplatBytes(0);
                teardown(owned);
                delete p;
            });
    }

} // namespace lfs::core
