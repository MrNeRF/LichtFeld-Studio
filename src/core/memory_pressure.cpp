/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/memory_pressure.hpp"

#include "core/checked_arithmetic.hpp"
#include "core/cuda_error.hpp"
#include "core/environment.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "tensor/internal/memory_pool.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <format>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace lfs::core {

    const char* to_string(MemoryDomain domain) noexcept {
        switch (domain) {
        case MemoryDomain::CudaDevice: return "cuda-device";
        case MemoryDomain::CudaVmm: return "cuda-vmm";
        case MemoryDomain::VulkanDevice: return "vulkan-device";
        case MemoryDomain::PinnedHost: return "pinned-host";
        case MemoryDomain::PageableHost: return "pageable-host";
        }
        return "unknown";
    }

    bool is_device_heap(MemoryDomain domain) noexcept {
        return domain == MemoryDomain::CudaDevice ||
               domain == MemoryDomain::CudaVmm ||
               domain == MemoryDomain::VulkanDevice;
    }

    namespace {

        std::string format_bytes(size_t bytes) {
            constexpr double mib = 1024.0 * 1024.0;
            return std::format("{:.1f} MiB", static_cast<double>(bytes) / mib);
        }

        // Consumes only a sticky OOM before querying so an unrelated asynchronous
        // error is preserved for its real handler.
        size_t query_device_free_bytes() {
            const cudaError_t sticky = cudaPeekAtLastError();
            if (sticky == cudaErrorMemoryAllocation) {
                cudaGetLastError();
            }
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
            if (status != cudaSuccess) {
                if (status == cudaErrorMemoryAllocation) {
                    cudaGetLastError();
                }
                return 0;
            }
            return free_bytes;
        }

        size_t query_device_total_bytes() {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
                return 0;
            }
            return total_bytes;
        }

        [[noreturn]] void throw_cuda_unavailable_allocation(
            const size_t bytes,
            const cudaStream_t stream,
            const char* const label,
            const char* const operation) {
            throw MemoryAllocationError(AllocationFailure{
                .domain = MemoryDomain::CudaDevice,
                .requested_bytes = bytes,
                .alignment = 0,
                .device = -1,
                .stream = reinterpret_cast<uintptr_t>(stream),
                .label = label,
                .operation = operation,
                .native_error = cudaErrorInitializationError,
            });
        }

        void* try_allocate_direct_cuda_storage(const size_t bytes,
                                               cudaError_t* const failure_status) {
            void* ptr = nullptr;
            const auto pre_call_state = sample_cuda_pre_call_state();
            const cudaError_t status = cudaMalloc(&ptr, bytes);
            if (status == cudaSuccess) {
                return ptr;
            }
            if (failure_status) {
                *failure_status = status;
            }
            ensure_cuda_success(
                status, pre_call_state, "cudaMalloc(direct tensor storage)",
                std::format("requested_bytes={}", bytes),
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            if (ptr != nullptr) {
                const cudaError_t cleanup_status = cudaFree(ptr);
                if (cleanup_status != cudaSuccess) {
                    ensure_cuda_success(
                        cleanup_status, "cudaFree(failed direct tensor storage allocation)", {},
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                }
            }
            return nullptr;
        }

        [[noreturn]] void throw_non_oom_cuda_allocation(
            const cudaError_t status,
            const size_t bytes,
            const char* const label,
            const char* const operation) {
            if (status == cudaSuccess) {
                throw std::runtime_error(std::format(
                    "CUDA allocator returned no storage without a CUDA error: "
                    "requested_bytes={}, label='{}', operation='{}'",
                    bytes, label ? label : "", operation ? operation : ""));
            }
            throw std::runtime_error(std::format(
                "CUDA allocation failed with {} ({}): requested_bytes={}, label='{}', "
                "operation='{}'",
                cudaGetErrorName(status), cudaGetErrorString(status), bytes,
                label ? label : "", operation ? operation : ""));
        }

    } // namespace

    MemoryAllocationError::MemoryAllocationError(const AllocationFailure& failure)
        : std::runtime_error(std::format(
              "{} out of memory: failed to allocate {} for '{}' (op '{}', device {}, native {})",
              to_string(failure.domain),
              format_bytes(failure.requested_bytes),
              failure.label ? failure.label : "",
              failure.operation ? failure.operation : "",
              failure.device,
              failure.native_error)),
          failure_(failure) {}

    struct MemoryPressureCoordinator::Impl {
        struct LedgerEntry {
            const char* name;
            int priority;
            size_t logical_released;
            size_t free_after;
        };

        mutable std::mutex registry_mutex;
        std::vector<PressureClient> clients;

        std::mutex episode_mutex;
        std::atomic<uint64_t> episode_counter{0};
        std::atomic<bool> pressure_active{false};
        std::atomic<size_t> last_target_free{0};
        std::chrono::steady_clock::time_point last_recover_check{};

        mutable std::mutex status_mutex;
        std::string last_status;

        std::mutex probe_mutex;
        std::atomic<bool> probe_armed{false};
        std::function<bool(MemoryDomain, size_t)> alloc_probe;
        std::function<size_t(MemoryDomain)> free_probe;

        std::once_flag reserve_once;
        size_t reserve_bytes = 0;

        static thread_local bool in_episode;

        size_t query_free(MemoryDomain domain) {
            {
                std::lock_guard<std::mutex> lock(probe_mutex);
                if (free_probe) {
                    return free_probe(domain);
                }
            }
            if (is_device_heap(domain)) {
                return query_device_free_bytes();
            }
            return 0;
        }

        std::vector<PressureClient> snapshot(MemoryDomain domain, PressureContext context) {
            std::vector<PressureClient> matched;
            {
                std::lock_guard<std::mutex> lock(registry_mutex);
                matched.reserve(clients.size());
                for (const auto& client : clients) {
                    if (!client_reclaims_domain(client.domain, domain)) {
                        continue;
                    }
                    if (!context_allows(context, client.affinity)) {
                        continue;
                    }
                    matched.push_back(client);
                }
            }
            std::stable_sort(matched.begin(), matched.end(),
                             [](const PressureClient& a, const PressureClient& b) {
                                 return a.priority < b.priority;
                             });
            return matched;
        }

        static bool client_reclaims_domain(MemoryDomain client_domain, MemoryDomain failing) {
            if (is_device_heap(failing)) {
                return is_device_heap(client_domain);
            }
            return client_domain == failing;
        }

        static bool context_allows(PressureContext context, PressureAffinity affinity) {
            switch (affinity) {
            case PressureAffinity::ImmediateThreadSafe:
                return true;
            case PressureAffinity::RenderSafePoint:
                return context == PressureContext::RenderThread;
            case PressureAffinity::TrainingSafePoint:
                return context == PressureContext::TrainingThread;
            }
            return false;
        }
    };

    thread_local bool MemoryPressureCoordinator::Impl::in_episode = false;

    MemoryPressureCoordinator::MemoryPressureCoordinator()
        : impl_(new Impl()) {
        register_client(PressureClient{
            .name = "tensor-cuda-pool",
            .priority = 10,
            .domain = MemoryDomain::CudaDevice,
            .affinity = PressureAffinity::ImmediateThreadSafe,
            .estimate = nullptr,
            .shrink = [](const PressureRequest&) {
                CudaMemoryPool::instance().trim_cached_memory();
                return ReclaimResult{};
            },
        });
        register_client(PressureClient{
            .name = "pinned-host-cache",
            .priority = 10,
            .domain = MemoryDomain::PinnedHost,
            .affinity = PressureAffinity::ImmediateThreadSafe,
            .estimate = [](const PressureRequest&) -> size_t {
                return PinnedMemoryAllocator::instance().get_stats().cached_bytes;
            },
            .shrink = [](const PressureRequest&) {
                auto& allocator = PinnedMemoryAllocator::instance();
                const size_t before = allocator.get_stats().cached_bytes;
                allocator.empty_cache();
                return ReclaimResult{.logical_bytes_released = before}; },
        });
    }

    MemoryPressureCoordinator::~MemoryPressureCoordinator() {
        delete impl_;
    }

    MemoryPressureCoordinator& MemoryPressureCoordinator::instance() {
        static MemoryPressureCoordinator coordinator;
        return coordinator;
    }

    void* allocate_cuda_storage(const size_t bytes,
                                const cudaStream_t stream,
                                const CudaStorageMode mode,
                                const char* const label,
                                const char* const operation) {
        if (bytes == 0) {
            return nullptr;
        }
        if (cuda_is_unavailable()) [[unlikely]] {
            throw_cuda_unavailable_allocation(bytes, stream, label, operation);
        }

        auto& coordinator = MemoryPressureCoordinator::instance();
        const auto try_allocate = [&](cudaError_t* const failure_status) -> void* {
            *failure_status = cudaSuccess;
            if (coordinator.probe_should_fail(MemoryDomain::CudaDevice, bytes)) {
                *failure_status = cudaErrorMemoryAllocation;
                return nullptr;
            }
            if (mode == CudaStorageMode::Direct) {
                return try_allocate_direct_cuda_storage(bytes, failure_status);
            }
            return CudaMemoryPool::instance().try_allocate(bytes, stream, failure_status);
        };

        cudaError_t failure_status = cudaSuccess;
        void* ptr = try_allocate(&failure_status);
        if (ptr != nullptr) {
            return ptr;
        }
        if (cuda_is_unavailable()) {
            throw_cuda_unavailable_allocation(bytes, stream, label, operation);
        }
        if (failure_status != cudaErrorMemoryAllocation) {
            throw_non_oom_cuda_allocation(failure_status, bytes, label, operation);
        }

        int device = 0;
        cudaGetDevice(&device);
        const AllocationFailure failure{
            .domain = MemoryDomain::CudaDevice,
            .requested_bytes = bytes,
            .alignment = 0,
            .device = device,
            .stream = reinterpret_cast<uintptr_t>(stream),
            .label = label,
            .operation = operation,
            .native_error = cudaErrorMemoryAllocation,
        };
        if (coordinator.relieve_and_should_retry(failure)) {
            ptr = try_allocate(&failure_status);
        }
        if (ptr == nullptr) {
            if (cuda_is_unavailable()) {
                throw_cuda_unavailable_allocation(bytes, stream, label, operation);
            }
            if (failure_status != cudaErrorMemoryAllocation) {
                throw_non_oom_cuda_allocation(failure_status, bytes, label, operation);
            }
            throw MemoryAllocationError(failure);
        }
        return ptr;
    }

    size_t MemoryPressureCoordinator::reserve_bytes() const noexcept {
        std::call_once(impl_->reserve_once, [this]() {
            size_t reserve = static_cast<size_t>(512) * 1024 * 1024;
            if (const auto mb = environment::unsigned_integer<unsigned long long>("LFS_VRAM_RESERVE_MB");
                mb && *mb > 0) {
                constexpr size_t MIB = size_t{1024} * 1024;
                reserve = *mb > std::numeric_limits<size_t>::max() / MIB
                              ? std::numeric_limits<size_t>::max()
                              : static_cast<size_t>(*mb) * MIB;
            }
            const size_t total = query_device_total_bytes();
            const size_t floor = static_cast<size_t>(128) * 1024 * 1024;
            const size_t ceiling = total > 0 ? total / 4 : reserve;
            reserve = std::max(reserve, floor);
            if (ceiling > floor) {
                reserve = std::min(reserve, ceiling);
            }
            impl_->reserve_bytes = reserve;
        });
        return impl_->reserve_bytes;
    }

    void MemoryPressureCoordinator::register_client(PressureClient client) {
        std::lock_guard<std::mutex> lock(impl_->registry_mutex);
        impl_->clients.push_back(std::move(client));
    }

    size_t MemoryPressureCoordinator::run_episode(const AllocationFailure& failure,
                                                  PressureContext context) {
        if (Impl::in_episode) {
            return 0;
        }
        if (cuda_is_unavailable()) {
            return 0;
        }
        Impl::in_episode = true;
        struct Guard {
            ~Guard() { Impl::in_episode = false; }
        } guard;

        std::lock_guard<std::mutex> episode_lock(impl_->episode_mutex);
        if (cuda_is_unavailable()) {
            return 0;
        }

        const size_t reserve = reserve_bytes();
        const size_t target = saturating_add(failure.requested_bytes, reserve);
        const size_t free_before = impl_->query_free(failure.domain);

        // A concurrent episode may already have relieved the pressure.
        if (is_device_heap(failure.domain) && free_before >= target) {
            return 0;
        }

        const uint64_t episode_id = impl_->episode_counter.fetch_add(1) + 1;
        impl_->pressure_active.store(true);
        impl_->last_target_free.store(target);

        PressureRequest request{
            .domain = failure.domain,
            .requested_bytes = failure.requested_bytes,
            .target_free_bytes = target,
            .device = failure.device,
            .episode_id = episode_id,
        };

        std::vector<PressureClient> clients = impl_->snapshot(failure.domain, context);
        std::vector<Impl::LedgerEntry> ledger;
        ledger.reserve(clients.size());

        size_t free_now = free_before;
        for (const auto& client : clients) {
            if (is_device_heap(failure.domain) && free_now >= target) {
                break;
            }
            ReclaimResult result;
            try {
                if (client.shrink) {
                    result = client.shrink(request);
                }
            } catch (const std::exception& e) {
                LOG_WARN("Pressure client '{}' failed during reclaim: {}", client.name, e.what());
                continue;
            } catch (...) {
                LOG_WARN("Pressure client '{}' failed during reclaim", client.name);
                continue;
            }
            free_now = impl_->query_free(failure.domain);
            ledger.push_back({client.name, client.priority, result.logical_bytes_released, free_now});
        }

        const size_t free_after = impl_->query_free(failure.domain);
        const bool satisfied = !is_device_heap(failure.domain) || free_after >= target;
        const size_t observed_released = free_after > free_before ? free_after - free_before : 0;

        record_cuda_breadcrumb("memory-pressure.episode", __FILE__, __LINE__);

        std::string summary = std::format(
            "VRAM pressure episode #{} ({}): requested {}, reserve {}, free {} -> {}, released {} across {} client(s){}",
            episode_id, to_string(failure.domain), format_bytes(failure.requested_bytes),
            format_bytes(reserve), format_bytes(free_before), format_bytes(free_after),
            format_bytes(observed_released), ledger.size(),
            satisfied ? "" : " (still short)");
        for (const auto& entry : ledger) {
            summary += std::format("\n  - {} (prio {}): logical {}, free after {}",
                                   entry.name, entry.priority,
                                   format_bytes(entry.logical_released), format_bytes(entry.free_after));
        }

        if (satisfied) {
            LOG_WARN("{}", summary);
        } else {
            LOG_ERROR("{}", summary);
        }

        {
            std::lock_guard<std::mutex> lock(impl_->status_mutex);
            impl_->last_status = summary;
        }

        // Formatting/emitting after relief keeps the raw OOM path allocation-free.
        events::state::VramPressure{
            .domain = to_string(failure.domain),
            .requested_bytes = failure.requested_bytes,
            .freed_bytes = observed_released,
            .recovered = satisfied,
        }
            .emit();

        return observed_released;
    }

    bool MemoryPressureCoordinator::relieve_and_should_retry(const AllocationFailure& failure,
                                                             PressureContext context) {
        if (cuda_is_unavailable()) {
            return false;
        }
        const size_t target = saturating_add(failure.requested_bytes, reserve_bytes());
        const size_t freed = run_episode(failure, context);
        if (freed > 0) {
            return true;
        }
        // Device retries require reported headroom; host availability is not measured.
        if (is_device_heap(failure.domain)) {
            return impl_->query_free(failure.domain) >= target;
        }
        return false;
    }

    PreflightResult MemoryPressureCoordinator::preflight(const OperationMemoryPlan& plan,
                                                         MemoryDomain domain) const {
        PreflightResult result;
        result.safety_reserve_bytes = reserve_bytes();
        result.required_peak_bytes = saturating_add(
            saturating_add(
                saturating_add(plan.persistent_device_bytes, plan.temporary_device_bytes),
                plan.old_new_overlap_bytes),
            result.safety_reserve_bytes);

        size_t reclaimable = 0;
        PressureRequest request{
            .domain = domain,
            .requested_bytes = saturating_add(
                plan.persistent_device_bytes, plan.temporary_device_bytes),
            .target_free_bytes = result.required_peak_bytes,
            .device = 0,
            .episode_id = 0,
        };
        {
            std::lock_guard<std::mutex> lock(impl_->registry_mutex);
            for (const auto& client : impl_->clients) {
                if (!Impl::client_reclaims_domain(client.domain, domain)) {
                    continue;
                }
                if (client.estimate) {
                    reclaimable = saturating_add(reclaimable, client.estimate(request));
                }
            }
        }
        result.reclaimable_bytes = reclaimable;
        result.effective_free_bytes = saturating_add(impl_->query_free(domain), reclaimable);
        result.ok = result.effective_free_bytes >= result.required_peak_bytes;
        return result;
    }

    bool MemoryPressureCoordinator::pressure_active() const noexcept {
        return impl_->pressure_active.load();
    }

    void MemoryPressureCoordinator::maybe_recover() {
        if (cuda_is_unavailable()) {
            return;
        }
        if (!impl_->pressure_active.load()) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - impl_->last_recover_check < std::chrono::milliseconds(250)) {
            return;
        }
        impl_->last_recover_check = now;

        const size_t target = impl_->last_target_free.load();
        const size_t hysteresis = saturating_add(target, target / 5); // require 20% headroom to restore
        if (impl_->query_free(MemoryDomain::CudaDevice) >= hysteresis) {
            impl_->pressure_active.store(false);
            LOG_INFO("VRAM pressure lease released; sustained headroom restored");
        }
    }

    uint64_t MemoryPressureCoordinator::episode_count() const noexcept {
        return impl_->episode_counter.load();
    }

    std::string MemoryPressureCoordinator::last_status() const {
        std::lock_guard<std::mutex> lock(impl_->status_mutex);
        return impl_->last_status;
    }

    void MemoryPressureCoordinator::set_allocation_probe(std::function<bool(MemoryDomain, size_t)> probe) {
        std::lock_guard<std::mutex> lock(impl_->probe_mutex);
        impl_->probe_armed.store(static_cast<bool>(probe), std::memory_order_release);
        impl_->alloc_probe = std::move(probe);
    }

    bool MemoryPressureCoordinator::probe_should_fail(MemoryDomain domain, size_t bytes) const {
        if (!impl_->probe_armed.load(std::memory_order_relaxed)) {
            return false;
        }
        std::function<bool(MemoryDomain, size_t)> probe;
        {
            std::lock_guard<std::mutex> lock(impl_->probe_mutex);
            probe = impl_->alloc_probe;
        }
        return probe ? probe(domain, bytes) : false;
    }

    void MemoryPressureCoordinator::set_free_memory_probe(std::function<size_t(MemoryDomain)> probe) {
        std::lock_guard<std::mutex> lock(impl_->probe_mutex);
        impl_->free_probe = std::move(probe);
    }

    void MemoryPressureCoordinator::reset_for_testing() {
        {
            std::lock_guard<std::mutex> lock(impl_->registry_mutex);
            impl_->clients.clear();
        }
        {
            std::lock_guard<std::mutex> lock(impl_->probe_mutex);
            impl_->probe_armed.store(false, std::memory_order_release);
            impl_->alloc_probe = nullptr;
            impl_->free_probe = nullptr;
        }
        impl_->episode_counter.store(0);
        impl_->pressure_active.store(false);
        impl_->last_target_free.store(0);
        {
            std::lock_guard<std::mutex> lock(impl_->status_mutex);
            impl_->last_status.clear();
        }
    }

} // namespace lfs::core
