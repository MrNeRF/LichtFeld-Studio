/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/lazy_device_storage.hpp"
#include "internal/tensor_impl.hpp"

#include "core/cuda/memory_arena.hpp"
#include "core/cuda_error.hpp"
#include "core/lazy_direct_storage.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"

#include <algorithm>
#include <atomic>
#include <cuda.h>
#include <format>
#include <stdexcept>
#include <vector>

namespace lfs::core {

    namespace {
        // Commit this far past the requested prefix so growth maps memory
        // every few densification steps, not every one: 1/8 of it, at most
        // 256 MiB. The first 2 MiB are always backed, for kernels that write
        // a padded tile past the last row.
        constexpr std::size_t kPaddingBytes = 2u << 20;
        constexpr std::size_t kMaxAheadBytes = 256u << 20;

        // Commit-ahead is only a shortcut for later growth. It stays out of
        // the last eighth of the device, so it never takes the room growth
        // planning keeps for the rasterizer arena, refines and the arena
        // memory a refine took under pressure; there only the requested
        // prefix is backed.
        [[nodiscard]] bool ahead_fits(const std::size_t ahead_bytes) {
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            std::size_t owed_to_arena = 0;
            if (auto* arena = GlobalArenaManager::instance().try_get_arena()) {
                owed_to_arena = arena->get_memory_info().recommit_bytes;
            }
            return free_bytes >= ahead_bytes + owed_to_arena + total_bytes / 8 +
                                     2 * RasterizerMemoryArena::growth_headroom_bytes(total_bytes);
        }

        // Ahead pages are mapped in chunks this small so a trim can hand back
        // all but the one the rows have grown into.
        constexpr std::size_t kAheadChunkBytes = 16u << 20;

        std::atomic<std::size_t> g_committed_total{0};
        std::atomic<std::size_t> g_requested_total{0};

        std::mutex g_registry_mutex;
        std::vector<LazyDeviceStorage*> g_registry;
        // The storage this thread is committing; a trim from its pressure
        // relief must not lock it again.
        thread_local const LazyDeviceStorage* t_committing = nullptr;

        [[nodiscard]] bool device_supports_vmm() {
            static const bool supported = [] {
                int device = 0;
                if (cudaGetDevice(&device) != cudaSuccess) {
                    return false;
                }
                CUdevice cu_device = 0;
                int vmm = 0;
                if (cuDeviceGet(&cu_device, device) != CUDA_SUCCESS ||
                    cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                         cu_device) != CUDA_SUCCESS) {
                    return false;
                }
                return vmm != 0;
            }();
            return supported;
        }

        [[noreturn]] void throw_commit_oom(const std::size_t bytes, const cudaStream_t stream) {
            int device = 0;
            if (cudaGetDevice(&device) != cudaSuccess) {
                (void)cudaGetLastError();
            }
            throw MemoryAllocationError(AllocationFailure{
                .domain = MemoryDomain::CudaDevice,
                .requested_bytes = bytes,
                .device = device,
                .stream = reinterpret_cast<uintptr_t>(stream),
                .label = "tensor.lazy_commit",
                .operation = "tensor.commit_rows",
                .native_error = cudaErrorMemoryAllocation,
            });
        }
    } // namespace

    std::shared_ptr<LazyDeviceStorage> LazyDeviceStorage::create(
        const std::size_t reserved_bytes, const std::size_t committed_bytes,
        const cudaStream_t stream) {
        if (reserved_bytes < kMinReservedBytes || !device_supports_vmm()) {
            return nullptr;
        }
        auto buffer = VmmDeviceBuffer::create(reserved_bytes, "tensor.lazy_direct");
        if (!buffer) {
            LOG_DEBUG("Lazy direct storage unavailable ({}); allocating the full capacity",
                      buffer.error().user_message());
            return nullptr;
        }
        std::shared_ptr<LazyDeviceStorage> storage(new LazyDeviceStorage());
        storage->buffer_ = std::move(*buffer);
        storage->reserved_bytes_ = reserved_bytes;
        {
            std::lock_guard lock(g_registry_mutex);
            g_registry.push_back(storage.get());
        }
        storage->commit(committed_bytes, stream);
        return storage;
    }

    LazyDeviceStorage::~LazyDeviceStorage() {
        {
            std::lock_guard lock(g_registry_mutex);
            std::erase(g_registry, this);
        }
        const std::size_t committed = buffer_.committed_bytes();
        g_committed_total.fetch_sub(committed, std::memory_order_relaxed);
        g_requested_total.fetch_sub(requested_bytes_, std::memory_order_relaxed);
        Tensor::record_storage_deallocation(Tensor::StorageAccountingKind::CudaDirect, committed);
    }

    std::size_t LazyDeviceStorage::committed_bytes() const {
        std::lock_guard lock(mutex_);
        return buffer_.committed_bytes();
    }

    std::size_t LazyDeviceStorage::committed_ahead_bytes_total() noexcept {
        const std::size_t requested = g_requested_total.load(std::memory_order_relaxed);
        const std::size_t committed = g_committed_total.load(std::memory_order_relaxed);
        return committed > requested ? committed - requested : 0;
    }

    std::size_t lazy_direct_storage_ahead_bytes() noexcept {
        return LazyDeviceStorage::committed_ahead_bytes_total();
    }

    std::size_t release_lazy_direct_storage_ahead() {
        return LazyDeviceStorage::release_ahead_total();
    }

    std::size_t LazyDeviceStorage::release_ahead_total() {
        if (committed_ahead_bytes_total() == 0) {
            return 0;
        }
        // Kernels may still read rows near the tail; nothing may be in flight
        // while pages leave the address space.
        if (cudaDeviceSynchronize() != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
        std::size_t released = 0;
        std::lock_guard registry_lock(g_registry_mutex);
        for (LazyDeviceStorage* storage : g_registry) {
            if (storage == t_committing) {
                continue;
            }
            std::unique_lock lock(storage->mutex_, std::try_to_lock);
            if (lock.owns_lock()) {
                released += storage->release_ahead_locked();
            }
        }
        return released;
    }

    std::size_t LazyDeviceStorage::release_ahead_locked() {
        const std::size_t granularity = buffer_.granularity_bytes();
        // The logical reservation can end inside a physical page. Clamp before
        // rounding so decommit_tail's alignment cannot discard live tail rows.
        const std::size_t padded = std::min(reserved_bytes_, requested_bytes_ + kPaddingBytes);
        const std::size_t keep = (padded + granularity - 1) / granularity * granularity;
        const std::size_t before = buffer_.committed_bytes();
        if (before <= keep) {
            return 0;
        }
        if (const Status status = buffer_.decommit_tail(keep); !status) {
            LOG_WARN("Could not release committed-ahead rows: {}", status.error().user_message());
        }
        const std::size_t released = before - buffer_.committed_bytes();
        zeroed_bytes_ = std::min(zeroed_bytes_, buffer_.committed_bytes());
        g_committed_total.fetch_sub(released, std::memory_order_relaxed);
        Tensor::record_storage_deallocation(Tensor::StorageAccountingKind::CudaDirect, released);
        return released;
    }

    void LazyDeviceStorage::commit(const std::size_t bytes, const cudaStream_t stream) {
        std::lock_guard lock(mutex_);
        commit_locked(bytes, stream);
    }

    void LazyDeviceStorage::commit_locked(const std::size_t bytes, const cudaStream_t stream) {
        if (bytes > reserved_bytes_) {
            throw std::logic_error(std::format(
                "lazy device storage commit of {} bytes exceeds its {}-byte reservation",
                bytes, reserved_bytes_));
        }
        if (bytes > requested_bytes_) {
            g_requested_total.fetch_add(bytes - requested_bytes_, std::memory_order_relaxed);
            requested_bytes_ = bytes;
        }
        if (bytes <= buffer_.committed_bytes()) {
            return;
        }

        const std::size_t before = buffer_.committed_bytes();
        const std::size_t padded = std::min(reserved_bytes_, bytes + kPaddingBytes);
        std::size_t target = std::min(reserved_bytes_, bytes + std::min(bytes / 8, kMaxAheadBytes));
        if (target > padded && !ahead_fits(target - std::max(before, padded))) {
            target = padded;
        }
        struct CommittingScope {
            const LazyDeviceStorage* outer = t_committing;
            explicit CommittingScope(const LazyDeviceStorage* self) { t_committing = self; }
            ~CommittingScope() { t_committing = outer; }
        } committing(this);
        // A failed commit can still map some chunks; zero and count whatever
        // got committed before pressure relief.
        auto attempt = [&](const std::size_t want) {
            const Status status = buffer_.commit(want);
            return status.has_value() || buffer_.committed_bytes() >= bytes;
        };
        bool ok = attempt(padded);
        if (ok && target > padded) {
            // Separate chunks past the rows, so release_ahead_total can
            // return them; failing to map them costs nothing.
            (void)buffer_.commit(target, kAheadChunkBytes);
        }
        zero_new_pages(stream);
        if (!ok) {
            int device = 0;
            if (cudaGetDevice(&device) != cudaSuccess) {
                (void)cudaGetLastError();
            }
            const AllocationFailure failure{
                .domain = MemoryDomain::CudaDevice,
                .requested_bytes = bytes - buffer_.committed_bytes(),
                .device = device,
                .stream = reinterpret_cast<uintptr_t>(stream),
                .label = "tensor.lazy_commit",
                .operation = "tensor.commit_rows",
                .native_error = cudaErrorMemoryAllocation,
            };
            if (MemoryPressureCoordinator::instance().relieve_and_should_retry(failure)) {
                ok = attempt(bytes);
                zero_new_pages(stream);
            }
            (void)cudaGetLastError();
        }
        const std::size_t grown = buffer_.committed_bytes() - before;
        g_committed_total.fetch_add(grown, std::memory_order_relaxed);
        Tensor::record_storage_allocation(Tensor::StorageAccountingKind::CudaDirect, grown);
        if (!ok) {
            throw_commit_oom(bytes, stream);
        }
    }

    void LazyDeviceStorage::zero_new_pages(const cudaStream_t stream) {
        const std::size_t committed = buffer_.committed_bytes();
        if (committed <= zeroed_bytes_) {
            return;
        }
        LFS_CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(buffer_.data()) + zeroed_bytes_, 0,
                                       committed - zeroed_bytes_, stream));
        // Rows are read on other streams too (optimizer); new pages hold
        // whatever the device left in them until the memset lands.
        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        zeroed_bytes_ = committed;
    }

} // namespace lfs::core
