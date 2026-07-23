/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/exportable_storage.hpp"

#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <format>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lfs::core {

    namespace {

#ifdef _WIN32
        constexpr CUmemAllocationHandleType kCudaHandleType = CU_MEM_HANDLE_TYPE_WIN32;
        constexpr const char* kCudaHandleTypeName = "WIN32";
#else
        constexpr CUmemAllocationHandleType kCudaHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        constexpr const char* kCudaHandleTypeName = "POSIX_FILE_DESCRIPTOR";
#endif

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
            LFS_CUDA_BREADCRUMB("exportable.cuDeviceGetAttribute.vmm");
            const CUresult r = cuDeviceGetAttribute(
                &supported,
                CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                device);
            return r == CUDA_SUCCESS && supported != 0;
        }

        bool export_handle_supported(int device) {
            int supported = 0;
#ifdef _WIN32
            constexpr CUdevice_attribute handle_attribute =
                CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED;
#else
            constexpr CUdevice_attribute handle_attribute =
                CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED;
#endif
            LFS_CUDA_BREADCRUMB("exportable.cuDeviceGetAttribute.handle_type");
            const CUresult r = cuDeviceGetAttribute(&supported, handle_attribute, device);
            return r == CUDA_SUCCESS && supported != 0;
        }

        void log_cleanup_error(std::string_view operation, CUresult result) {
            if (result != CUDA_SUCCESS) {
                LOG_ERROR("{} failed during exportable CUDA cleanup: {}", operation, cu_error(result));
            }
        }

        // Owns a CUDA VMM allocation: a fixed virtual reservation (va,
        // reserved_size) with `mapped_size` physical committed under it. The
        // virtual address is stable for the lifetime of the allocation; only the
        // physical handle / committed size change when growing.
        struct OwnedAllocation {
            CUmemAllocationProp prop{};
            int device = 0;
            std::size_t granularity = 0;
            CUdeviceptr va = 0;
            std::size_t reserved_size = 0;
            std::size_t mapped_size = 0;
            CUmemGenericAllocationHandle mem_handle = 0;
            bool mapped = false;
            bool created = false;
            bool reserved = false;
            ExportNativeHandle native = ExportNativeHandle{};
            bool native_valid = false;
#ifdef _WIN32
            SECURITY_ATTRIBUTES security_attributes{};
#endif
        };

        void close_native(OwnedAllocation& a) {
            if (a.native_valid) {
#ifdef _WIN32
                CloseHandle(a.native);
#else
                if (a.native >= 0) {
                    ::close(a.native);
                }
#endif
                a.native_valid = false;
                a.native = ExportNativeHandle{};
            }
        }

        // Releases the committed physical memory (unmap + release handle + close
        // exported handle) while keeping the virtual reservation intact.
        void release_physical(OwnedAllocation& a) {
            close_native(a);
            if (a.mapped) {
                LFS_CUDA_BREADCRUMB_ARGS(
                    "exportable.cuMemUnmap.physical", 0, a.va, a.mapped_size);
                const CUresult unmap_result = cuMemUnmap(a.va, a.mapped_size);
                log_cleanup_error("cuMemUnmap", unmap_result);
                a.mapped = false;
            }
            if (a.created) {
                LFS_CUDA_BREADCRUMB_ARGS(
                    "exportable.cuMemRelease.physical", 0, a.va, a.mapped_size);
                const CUresult release_result = cuMemRelease(a.mem_handle);
                log_cleanup_error("cuMemRelease", release_result);
                a.created = false;
                a.mem_handle = 0;
            }
            a.mapped_size = 0;
        }

        void teardown(OwnedAllocation& a) {
            release_physical(a);
            if (a.reserved) {
                LFS_CUDA_BREADCRUMB_ARGS(
                    "exportable.cuMemAddressFree.reservation", 0, a.va, a.reserved_size);
                const CUresult free_result = cuMemAddressFree(a.va, a.reserved_size);
                log_cleanup_error("cuMemAddressFree", free_result);
                a.reserved = false;
                a.va = 0;
                a.reserved_size = 0;
            }
        }

        // Commits `aligned_size` physical bytes under the (already reserved) va,
        // sets device R/W access, zeroes it, and exports an OS handle. Requires
        // a.va valid and aligned_size <= a.reserved_size. On failure the physical
        // is rolled back (the reservation is left intact for the caller).
        std::expected<void, std::string> commit_physical(OwnedAllocation& a, std::size_t aligned_size) {
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemCreate.commit", a.va, 0, aligned_size);
            if (const auto r = cuMemCreate(&a.mem_handle, aligned_size, &a.prop, 0); r != CUDA_SUCCESS) {
                return std::unexpected("cuMemCreate (exportable) failed: " + cu_error(r));
            }
            a.created = true;

            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemMap.commit", a.va, 0, aligned_size);
            if (const auto r = cuMemMap(a.va, aligned_size, 0, a.mem_handle, 0); r != CUDA_SUCCESS) {
                release_physical(a);
                return std::unexpected("cuMemMap failed: " + cu_error(r));
            }
            a.mapped = true;
            a.mapped_size = aligned_size;

            CUmemAccessDesc access{};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = a.device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemSetAccess.commit", a.va, 0, aligned_size);
            if (const auto r = cuMemSetAccess(a.va, aligned_size, &access, 1); r != CUDA_SUCCESS) {
                release_physical(a);
                return std::unexpected("cuMemSetAccess failed: " + cu_error(r));
            }

            // Zero the committed block. Capacity-backed tensor views can expose
            // slack rows before training fills them, and Vulkan reads those views
            // directly in the zero-copy path.
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cudaMemset.zero_fill", a.va, 0, aligned_size);
            if (const auto err = cudaMemset(reinterpret_cast<void*>(a.va), 0, aligned_size); err != cudaSuccess) {
                release_physical(a);
                return std::unexpected(std::format("cudaMemset on exportable block failed: {}",
                                                   cudaGetErrorString(err)));
            }
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cudaStreamSynchronize.zero_fill", a.va, 0, aligned_size);
            if (const auto err = cudaStreamSynchronize(nullptr); err != cudaSuccess) {
                release_physical(a);
                return std::unexpected(std::format(
                    "cudaStreamSynchronize after exportable zero-fill failed: {} ({})",
                    cudaGetErrorName(err),
                    cudaGetErrorString(err)));
            }

#ifdef _WIN32
            void* native = nullptr;
#else
            int native = -1;
#endif
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemExportToShareableHandle.commit", 0, a.va, aligned_size);
            if (const auto r = cuMemExportToShareableHandle(&native, a.mem_handle, kCudaHandleType, 0);
                r != CUDA_SUCCESS) {
                release_physical(a);
                return std::unexpected("cuMemExportToShareableHandle failed: " + cu_error(r));
            }
            a.native = native;
            a.native_valid = true;
            return {};
        }

    } // namespace

    std::expected<std::shared_ptr<ExportableBlock>, std::string>
    allocateExportableDeviceBlock(std::size_t size, int device, bool track_splat_bytes,
                                  std::size_t reserve_bytes) {
        if (size == 0) {
            return std::unexpected("allocateExportableDeviceBlock: size must be non-zero");
        }
        if (!vmm_supported(device)) {
            return std::unexpected(std::format(
                "allocateExportableDeviceBlock: device {} does not support virtual memory management",
                device));
        }
        if (!export_handle_supported(device)) {
            return std::unexpected(std::format(
                "allocateExportableDeviceBlock: device {} does not support CUDA VMM export handle type {}",
                device,
                kCudaHandleTypeName));
        }

        // CUDA VMM allocations require a current context. cudaSetDevice ensures one exists.
        LFS_CUDA_BREADCRUMB("exportable.cudaSetDevice.allocate");
        if (const auto err = cudaSetDevice(device); err != cudaSuccess) {
            return std::unexpected(std::format("cudaSetDevice({}) failed: {}",
                                               device,
                                               cudaGetErrorString(err)));
        }

        auto owned = std::make_shared<OwnedAllocation>();
        owned->device = device;
        owned->prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        owned->prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        owned->prop.location.id = device;
        owned->prop.requestedHandleTypes = kCudaHandleType;
#ifdef _WIN32
        owned->security_attributes.nLength = sizeof(SECURITY_ATTRIBUTES);
        owned->security_attributes.lpSecurityDescriptor = nullptr;
        owned->security_attributes.bInheritHandle = FALSE;
        // CUDA 12.8 documents POBJECT_ATTRIBUTES here, while NVIDIA's memMapIPCDrv
        // sample uses SECURITY_ATTRIBUTES; the sample form is field-verified on Windows.
        owned->prop.win32HandleMetaData = &owned->security_attributes;
#endif

        LFS_CUDA_BREADCRUMB("exportable.cuMemGetAllocationGranularity.allocate");
        if (const auto r = cuMemGetAllocationGranularity(&owned->granularity, &owned->prop,
                                                         CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            r != CUDA_SUCCESS) {
            return std::unexpected("cuMemGetAllocationGranularity failed: " + cu_error(r));
        }
        if (owned->granularity == 0) {
            owned->granularity = std::size_t(2) << 20;
        }

        const std::size_t aligned_size = align_up(size, owned->granularity);
        const std::size_t reserved_size = align_up(std::max(reserve_bytes, size), owned->granularity);

        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cuMemAddressReserve.allocate", 0, 0, reserved_size);
        if (const auto r = cuMemAddressReserve(&owned->va, reserved_size, 0, 0, 0); r != CUDA_SUCCESS) {
            return std::unexpected("cuMemAddressReserve failed: " + cu_error(r));
        }
        owned->reserved = true;
        owned->reserved_size = reserved_size;

        if (auto ok = commit_physical(*owned, aligned_size); !ok) {
            teardown(*owned);
            return std::unexpected(ok.error());
        }
        register_cuda_address_range(
            reinterpret_cast<void*>(owned->va), aligned_size, "exportable-splat-block");

        if (track_splat_bytes) {
            diagnostics::VramProfiler::instance().setExportableSplatBytes(aligned_size);
        }

        auto* block = new ExportableBlock{
            .device_ptr = reinterpret_cast<void*>(owned->va),
            .size = aligned_size,
            .handle = ExportHandle{.native = owned->native, .size = aligned_size},
            .state = owned,
        };

        LOG_INFO("Exportable CUDA block: device_ptr={} committed={} MiB reserved={} MiB granularity={}",
                 block->device_ptr,
                 aligned_size >> 20,
                 reserved_size >> 20,
                 owned->granularity);

        return std::shared_ptr<ExportableBlock>(
            block,
            [owned, track_splat_bytes](ExportableBlock* p) mutable {
                if (track_splat_bytes) {
                    diagnostics::VramProfiler::instance().setExportableSplatBytes(0);
                }
                LFS_CUDA_BREADCRUMB_ARGS(
                    "exportable.cudaDeviceSynchronize.destroy", 0,
                    reinterpret_cast<uintptr_t>(p->device_ptr), p->size);
                if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
                    LOG_ERROR(
                        "cudaDeviceSynchronize before exportable CUDA teardown failed: {} ({})",
                        cudaGetErrorName(err),
                        cudaGetErrorString(err));
                }
                unregister_cuda_address_range(p->device_ptr);
                teardown(*owned);
                delete p;
            });
    }

    std::expected<bool, std::string>
    growExportableDeviceBlock(const std::shared_ptr<ExportableBlock>& block, std::size_t new_size) {
        if (!block || !block->state) {
            return std::unexpected("growExportableDeviceBlock: null block");
        }
        auto* owned = static_cast<OwnedAllocation*>(block->state.get());

        const std::size_t aligned_new = align_up(new_size, owned->granularity);
        if (aligned_new <= owned->mapped_size) {
            return false; // already large enough
        }
        if (aligned_new > owned->reserved_size) {
            return std::unexpected(std::format(
                "growExportableDeviceBlock: request {} MiB exceeds reserved {} MiB",
                aligned_new >> 20, owned->reserved_size >> 20));
        }

        LFS_CUDA_BREADCRUMB("exportable.cudaSetDevice.grow");
        if (const auto err = cudaSetDevice(owned->device); err != cudaSuccess) {
            return std::unexpected(std::format("cudaSetDevice({}) failed: {}",
                                               owned->device, cudaGetErrorString(err)));
        }

        // Grow committed physical at the larger size under the SAME virtual
        // address. The existing contents are preserved (copied) so this is safe
        // even mid-frame, when earlier bump allocations of the current frame are
        // still live. device_ptr (owned->va) stays constant, so consumers holding
        // the base pointer (the training arena) never see it move.
        const std::size_t old_size = owned->mapped_size;

        // 1. Create + map the new (larger) physical at a temporary address.
        CUmemGenericAllocationHandle new_handle = 0;
        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cuMemCreate.grow", 0, 0, aligned_new);
        if (const auto r = cuMemCreate(&new_handle, aligned_new, &owned->prop, 0); r != CUDA_SUCCESS) {
            return std::unexpected("cuMemCreate (grow) failed: " + cu_error(r));
        }
        CUdeviceptr temp_va = 0;
        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cuMemAddressReserve.grow", 0, 0, aligned_new);
        if (const auto r = cuMemAddressReserve(&temp_va, aligned_new, 0, 0, 0); r != CUDA_SUCCESS) {
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemRelease.grow_reserve_failure", 0, 0, aligned_new);
            log_cleanup_error("cuMemRelease", cuMemRelease(new_handle));
            return std::unexpected("cuMemAddressReserve (grow) failed: " + cu_error(r));
        }
        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cuMemMap.grow_temp", temp_va, 0, aligned_new);
        if (const auto r = cuMemMap(temp_va, aligned_new, 0, new_handle, 0); r != CUDA_SUCCESS) {
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemAddressFree.grow_map_failure", 0, temp_va, aligned_new);
            log_cleanup_error("cuMemAddressFree", cuMemAddressFree(temp_va, aligned_new));
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemRelease.grow_map_failure", 0, temp_va, aligned_new);
            log_cleanup_error("cuMemRelease", cuMemRelease(new_handle));
            return std::unexpected("cuMemMap (grow temp) failed: " + cu_error(r));
        }
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = owned->device;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cuMemSetAccess.grow_temp", temp_va, 0, aligned_new);
        cuMemSetAccess(temp_va, aligned_new, &access, 1);

        // 2. Zero the new slack, then copy the live contents over.
        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cudaMemset.grow", temp_va, 0, aligned_new);
        cudaMemset(reinterpret_cast<void*>(temp_va), 0, aligned_new);
        if (old_size > 0) {
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cudaMemcpy.grow", temp_va, owned->va, old_size);
            cudaMemcpy(reinterpret_cast<void*>(temp_va), reinterpret_cast<void*>(owned->va),
                       old_size, cudaMemcpyDeviceToDevice);
        }
        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cudaDeviceSynchronize.grow", temp_va, owned->va, old_size);
        cudaDeviceSynchronize(); // copy must complete before the old mapping is torn down

        // 3. Export the new handle, then release the old physical.
#ifdef _WIN32
        void* new_native = nullptr;
#else
        int new_native = -1;
#endif
        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cuMemExportToShareableHandle.grow", 0, temp_va, aligned_new);
        if (const auto r = cuMemExportToShareableHandle(&new_native, new_handle, kCudaHandleType, 0);
            r != CUDA_SUCCESS) {
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemUnmap.grow_export_failure", 0, temp_va, aligned_new);
            log_cleanup_error("cuMemUnmap", cuMemUnmap(temp_va, aligned_new));
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemAddressFree.grow_export_failure", 0, temp_va, aligned_new);
            log_cleanup_error("cuMemAddressFree", cuMemAddressFree(temp_va, aligned_new));
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemRelease.grow_export_failure", 0, temp_va, aligned_new);
            log_cleanup_error("cuMemRelease", cuMemRelease(new_handle));
            return std::unexpected("cuMemExportToShareableHandle (grow) failed: " + cu_error(r));
        }
        release_physical(*owned); // unmap old from owned->va, release old handle, close old fd
        unregister_cuda_address_range(block->device_ptr);

        // 4. Move the new physical onto the stable base address.
        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cuMemUnmap.grow_temp", 0, temp_va, aligned_new);
        log_cleanup_error("cuMemUnmap", cuMemUnmap(temp_va, aligned_new));
        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cuMemAddressFree.grow_temp", 0, temp_va, aligned_new);
        log_cleanup_error("cuMemAddressFree", cuMemAddressFree(temp_va, aligned_new));
        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cuMemMap.grow_base", owned->va, 0, aligned_new);
        if (const auto r = cuMemMap(owned->va, aligned_new, 0, new_handle, 0); r != CUDA_SUCCESS) {
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemRelease.grow_base_failure", 0, owned->va, aligned_new);
            log_cleanup_error("cuMemRelease", cuMemRelease(new_handle));
#ifdef _WIN32
            CloseHandle(new_native);
#else
            if (new_native >= 0)
                ::close(new_native);
#endif
            return std::unexpected("cuMemMap (grow base) failed: " + cu_error(r));
        }
        LFS_CUDA_BREADCRUMB_ARGS(
            "exportable.cuMemSetAccess.grow_base", owned->va, 0, aligned_new);
        cuMemSetAccess(owned->va, aligned_new, &access, 1);

        owned->mem_handle = new_handle;
        owned->created = true;
        owned->mapped = true;
        owned->mapped_size = aligned_new;
        owned->native = new_native;
        owned->native_valid = true;

        block->size = aligned_new;
        block->handle.native = owned->native;
        block->handle.size = aligned_new;
        register_cuda_address_range(
            block->device_ptr, aligned_new, "exportable-splat-block");

        LOG_INFO("Exportable CUDA block grew in place: device_ptr={} committed={} MiB reserved={} MiB",
                 block->device_ptr,
                 aligned_new >> 20,
                 owned->reserved_size >> 20);
        return true;
    }

} // namespace lfs::core
