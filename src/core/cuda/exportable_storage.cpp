/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/exportable_storage.hpp"

#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/shareable_allocation_limit.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <format>
#include <iterator>
#include <string_view>
#include <utility>
#include <vector>

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

        constexpr std::size_t kDefaultGranularity = std::size_t{2} << 20;

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
            if (alignment == 0) {
                return value;
            }
            return ((value + alignment - 1) / alignment) * alignment;
        }

        std::size_t align_down(std::size_t value, std::size_t alignment) {
            if (alignment == 0) {
                return value;
            }
            return (value / alignment) * alignment;
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

        CUmemAllocationProp make_prop(int device) {
            CUmemAllocationProp prop{};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            prop.requestedHandleTypes = kCudaHandleType;
            return prop;
        }

        struct CommittedSlice {
            CUmemGenericAllocationHandle mem_handle = 0;
            std::size_t offset = 0;
            std::size_t bytes = 0;
            ExportNativeHandle native = ExportNativeHandle{};
            bool native_valid = false;
            bool created = false;
            bool mapped = false;
        };

        struct OwnedAllocation {
            CUmemAllocationProp prop{};
            int device = 0;
            std::size_t granularity = 0;
            CUdeviceptr va = 0;
            std::size_t reserved_size = 0;
            std::vector<CommittedSlice> slices;
            bool reserved = false;
#ifdef _WIN32
            SECURITY_ATTRIBUTES security_attributes{};
#endif
        };

        void close_native(CommittedSlice& slice) {
            if (!slice.native_valid) {
                return;
            }
#ifdef _WIN32
            CloseHandle(slice.native);
#else
            if (slice.native >= 0) {
                ::close(slice.native);
            }
#endif
            slice.native_valid = false;
            slice.native = ExportNativeHandle{};
        }

        void release_slice(OwnedAllocation& a, CommittedSlice& slice) {
            close_native(slice);
            if (slice.mapped) {
                LFS_CUDA_BREADCRUMB_ARGS(
                    "exportable.cuMemUnmap.slice", a.va + slice.offset, slice.offset, slice.bytes);
                const CUresult unmap_result = cuMemUnmap(a.va + slice.offset, slice.bytes);
                log_cleanup_error("cuMemUnmap", unmap_result);
                slice.mapped = false;
            }
            if (slice.created) {
                LFS_CUDA_BREADCRUMB_ARGS(
                    "exportable.cuMemRelease.slice", a.va + slice.offset, slice.offset, slice.bytes);
                const CUresult release_result = cuMemRelease(slice.mem_handle);
                log_cleanup_error("cuMemRelease", release_result);
                slice.created = false;
                slice.mem_handle = 0;
            }
        }

        void teardown(OwnedAllocation& a) {
            for (auto& slice : a.slices) {
                release_slice(a, slice);
            }
            a.slices.clear();
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

        void publish_chunks(ExportableBlock& block, const OwnedAllocation& owned) {
            block.chunks.clear();
            block.chunks.reserve(owned.slices.size());
            std::size_t committed = 0;
            for (const auto& slice : owned.slices) {
                assert(slice.offset % owned.granularity == 0);
                assert(slice.bytes % owned.granularity == 0);
                assert(slice.bytes > 0);
                block.chunks.push_back(ExportableChunk{
                    .offset = slice.offset,
                    .bytes = slice.bytes,
                    .handle = ExportHandle{.native = slice.native, .size = slice.bytes},
                });
                committed += slice.bytes;
            }
            for (std::size_t i = 1; i < block.chunks.size(); ++i) {
                assert(block.chunks[i - 1].offset < block.chunks[i].offset);
                assert(block.chunks[i - 1].offset + block.chunks[i - 1].bytes <= block.chunks[i].offset);
            }
            block.committed_bytes = committed;
            block.reserved_bytes = owned.reserved_size;
        }

        std::vector<std::pair<std::size_t, std::size_t>>
        uncovered_ranges(const OwnedAllocation& owned, std::size_t start, std::size_t end) {
            std::vector<std::pair<std::size_t, std::size_t>> holes;
            std::size_t cursor = start;
            for (const auto& slice : owned.slices) {
                const std::size_t slice_end = slice.offset + slice.bytes;
                if (slice_end <= cursor) {
                    continue;
                }
                if (slice.offset >= end) {
                    break;
                }
                if (slice.offset > cursor) {
                    holes.emplace_back(cursor, std::min(slice.offset, end) - cursor);
                }
                cursor = std::max(cursor, slice_end);
                if (cursor >= end) {
                    return holes;
                }
            }
            if (cursor < end) {
                holes.emplace_back(cursor, end - cursor);
            }
            return holes;
        }

        std::expected<void, std::string> commit_slice(OwnedAllocation& a, std::size_t offset, std::size_t bytes) {
            assert(offset % a.granularity == 0);
            assert(bytes % a.granularity == 0);
            assert(bytes > 0);
            assert(offset + bytes <= a.reserved_size);

            CommittedSlice slice{};
            slice.offset = offset;
            slice.bytes = bytes;

            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemCreate.slice", a.va + offset, offset, bytes);
            if (const auto r = cuMemCreate(&slice.mem_handle, bytes, &a.prop, 0); r != CUDA_SUCCESS) {
                return std::unexpected("cuMemCreate (exportable chunk) failed: " + cu_error(r));
            }
            slice.created = true;

            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemMap.slice", a.va + offset, offset, bytes);
            if (const auto r = cuMemMap(a.va + offset, bytes, 0, slice.mem_handle, 0); r != CUDA_SUCCESS) {
                release_slice(a, slice);
                return std::unexpected("cuMemMap (exportable chunk) failed: " + cu_error(r));
            }
            slice.mapped = true;

            CUmemAccessDesc access{};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = a.device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemSetAccess.slice", a.va + offset, offset, bytes);
            if (const auto r = cuMemSetAccess(a.va + offset, bytes, &access, 1); r != CUDA_SUCCESS) {
                release_slice(a, slice);
                return std::unexpected("cuMemSetAccess (exportable chunk) failed: " + cu_error(r));
            }

            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cudaMemset.slice", a.va + offset, offset, bytes);
            if (const auto err = cudaMemset(reinterpret_cast<void*>(a.va + offset), 0, bytes);
                err != cudaSuccess) {
                release_slice(a, slice);
                return std::unexpected(std::format("cudaMemset on exportable chunk failed: {}",
                                                   cudaGetErrorString(err)));
            }
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cudaStreamSynchronize.slice", a.va + offset, offset, bytes);
            if (const auto err = cudaStreamSynchronize(nullptr); err != cudaSuccess) {
                release_slice(a, slice);
                return std::unexpected(std::format(
                    "cudaStreamSynchronize after exportable chunk zero-fill failed: {} ({})",
                    cudaGetErrorName(err),
                    cudaGetErrorString(err)));
            }

#ifdef _WIN32
            void* native = nullptr;
#else
            int native = -1;
#endif
            LFS_CUDA_BREADCRUMB_ARGS(
                "exportable.cuMemExportToShareableHandle.slice", a.va + offset, offset, bytes);
            if (const auto r = cuMemExportToShareableHandle(&native, slice.mem_handle, kCudaHandleType, 0);
                r != CUDA_SUCCESS) {
                release_slice(a, slice);
                return std::unexpected("cuMemExportToShareableHandle (exportable chunk) failed: " +
                                       cu_error(r));
            }
            slice.native = native;
            slice.native_valid = true;

            const auto insert_at = std::lower_bound(
                a.slices.begin(),
                a.slices.end(),
                offset,
                [](const CommittedSlice& existing, const std::size_t off) {
                    return existing.offset < off;
                });
            if (insert_at != a.slices.end()) {
                assert(insert_at->offset >= offset + bytes);
            }
            if (insert_at != a.slices.begin()) {
                const auto prev = std::prev(insert_at);
                assert(prev->offset + prev->bytes <= offset);
            }
            a.slices.insert(insert_at, slice);
            return {};
        }

    } // namespace

    std::size_t exportable_allocation_granularity(const int device) {
        CUmemAllocationProp prop = make_prop(device);
        std::size_t granularity = 0;
        const CUresult r = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (r != CUDA_SUCCESS || granularity == 0) {
            return kDefaultGranularity;
        }
        return granularity;
    }

    std::expected<void, std::string>
    commitExportableDeviceRange(const std::shared_ptr<ExportableBlock>& block, std::size_t offset,
                                std::size_t bytes) {
        if (!block || !block->state) {
            return std::unexpected("commitExportableDeviceRange: null block");
        }
        if (bytes == 0) {
            return {};
        }
        auto* owned = static_cast<OwnedAllocation*>(block->state.get());
        if (owned->granularity == 0) {
            return std::unexpected("commitExportableDeviceRange: invalid granularity");
        }
        if (offset > owned->reserved_size) {
            return std::unexpected(std::format(
                "commitExportableDeviceRange: offset {} exceeds reserved {}",
                offset,
                owned->reserved_size));
        }
        if (bytes > owned->reserved_size - offset) {
            return std::unexpected(std::format(
                "commitExportableDeviceRange: range [{}, {}) exceeds reserved {}",
                offset,
                offset + bytes,
                owned->reserved_size));
        }

        LFS_CUDA_BREADCRUMB("exportable.cudaSetDevice.commit_range");
        if (const auto err = cudaSetDevice(owned->device); err != cudaSuccess) {
            return std::unexpected(std::format("cudaSetDevice({}) failed: {}",
                                               owned->device,
                                               cudaGetErrorString(err)));
        }

        const std::size_t aligned_start = align_down(offset, owned->granularity);
        const std::size_t aligned_end = align_up(offset + bytes, owned->granularity);
        const auto holes = uncovered_ranges(*owned, aligned_start, aligned_end);
        if (holes.empty()) {
            publish_chunks(*block, *owned);
            return {};
        }

        const std::size_t chunk_limit = shareable_chunk_bytes(owned->device);
        assert(chunk_limit >= owned->granularity);
        assert(chunk_limit % owned->granularity == 0);

        std::vector<std::size_t> added_offsets;
        const auto rollback_added = [&]() {
            for (auto it = added_offsets.rbegin(); it != added_offsets.rend(); ++it) {
                const auto sit = std::find_if(
                    owned->slices.begin(),
                    owned->slices.end(),
                    [&](const CommittedSlice& slice) { return slice.offset == *it; });
                if (sit == owned->slices.end()) {
                    continue;
                }
                release_slice(*owned, *sit);
                owned->slices.erase(sit);
            }
        };
        for (const auto& [hole_off, hole_bytes] : holes) {
            std::size_t remaining_off = hole_off;
            std::size_t remaining = hole_bytes;
            while (remaining > 0) {
                const std::size_t piece = std::min(remaining, chunk_limit);
                if (auto ok = commit_slice(*owned, remaining_off, piece); !ok) {
                    rollback_added();
                    publish_chunks(*block, *owned);
                    return std::unexpected(ok.error());
                }
                added_offsets.push_back(remaining_off);
                remaining_off += piece;
                remaining -= piece;
            }
        }

        publish_chunks(*block, *owned);
        LOG_DEBUG("Exportable CUDA block: chunks appended/bound count={} committed={} MiB reserved={} MiB",
                  block->chunks.size(),
                  block->committed_bytes >> 20,
                  block->reserved_bytes >> 20);
        return {};
    }

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

        LFS_CUDA_BREADCRUMB("exportable.cudaSetDevice.allocate");
        if (const auto err = cudaSetDevice(device); err != cudaSuccess) {
            return std::unexpected(std::format("cudaSetDevice({}) failed: {}",
                                               device,
                                               cudaGetErrorString(err)));
        }

        auto owned = std::make_shared<OwnedAllocation>();
        owned->device = device;
        owned->prop = make_prop(device);
#ifdef _WIN32
        owned->security_attributes.nLength = sizeof(SECURITY_ATTRIBUTES);
        owned->security_attributes.lpSecurityDescriptor = nullptr;
        owned->security_attributes.bInheritHandle = FALSE;
        owned->prop.win32HandleMetaData = &owned->security_attributes;
#endif

        LFS_CUDA_BREADCRUMB("exportable.cuMemGetAllocationGranularity.allocate");
        if (const auto r = cuMemGetAllocationGranularity(&owned->granularity, &owned->prop,
                                                         CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            r != CUDA_SUCCESS) {
            return std::unexpected("cuMemGetAllocationGranularity failed: " + cu_error(r));
        }
        if (owned->granularity == 0) {
            owned->granularity = kDefaultGranularity;
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

        auto* block = new ExportableBlock{
            .device_ptr = reinterpret_cast<void*>(owned->va),
            .reserved_bytes = reserved_size,
            .committed_bytes = 0,
            .chunks = {},
            .state = owned,
        };

        auto holder = std::shared_ptr<ExportableBlock>(
            block,
            [owned, track_splat_bytes](ExportableBlock* p) mutable {
                if (track_splat_bytes) {
                    diagnostics::VramProfiler::instance().setExportableSplatBytes(0);
                }
                LFS_CUDA_BREADCRUMB_ARGS(
                    "exportable.cudaDeviceSynchronize.destroy", 0,
                    reinterpret_cast<uintptr_t>(p->device_ptr), p->reserved_bytes);
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

        if (auto ok = commitExportableDeviceRange(holder, 0, aligned_size); !ok) {
            return std::unexpected(ok.error());
        }

        register_cuda_address_range(
            reinterpret_cast<void*>(owned->va), reserved_size, "exportable-splat-block");

        if (track_splat_bytes) {
            diagnostics::VramProfiler::instance().setExportableSplatBytes(holder->committed_bytes);
        }

        LOG_INFO("Exportable CUDA block: device_ptr={} committed={} MiB reserved={} MiB granularity={} chunks={}",
                 holder->device_ptr,
                 holder->committed_bytes >> 20,
                 reserved_size >> 20,
                 owned->granularity,
                 holder->chunks.size());

        return holder;
    }

    std::expected<bool, std::string>
    growExportableDeviceBlock(const std::shared_ptr<ExportableBlock>& block, std::size_t new_size) {
        if (!block || !block->state) {
            return std::unexpected("growExportableDeviceBlock: null block");
        }
        auto* owned = static_cast<OwnedAllocation*>(block->state.get());

        const std::size_t aligned_new = align_up(new_size, owned->granularity);
        const std::size_t prefix = block->committedPrefixBytes();
        if (aligned_new <= prefix) {
            return false;
        }
        if (aligned_new > owned->reserved_size) {
            return std::unexpected(std::format(
                "growExportableDeviceBlock: request {} MiB exceeds reserved {} MiB",
                aligned_new >> 20, owned->reserved_size >> 20));
        }

        const std::size_t chunks_before = block->chunks.size();
        if (auto ok = commitExportableDeviceRange(block, prefix, aligned_new - prefix); !ok) {
            return std::unexpected(ok.error());
        }
        return block->chunks.size() > chunks_before;
    }

} // namespace lfs::core
