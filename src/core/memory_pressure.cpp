/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/memory_pressure.hpp"
#include "core/gpu_device_runtime.hpp"

#include "core/alloc_counter.hpp"
#include "core/checked_arithmetic.hpp"
#include "core/cuda_error.hpp"
#include "core/environment.hpp"
#include "core/events.hpp"
#include "core/gpu_device_info.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace lfs::core {

    namespace {

        std::string format_bytes(size_t bytes) {
            constexpr double mib = 1024.0 * 1024.0;
            return std::format("{:.1f} MiB", static_cast<double>(bytes) / mib);
        }

        // The tensor backend of a Vulkan or Metal device domain; CUDA and host
        // domains have none.
        std::optional<GpuBackend> domain_backend(const MemoryDomain domain) {
            if (domain == MemoryDomain::VulkanDevice)
                return GpuBackend::Vulkan;
            if (domain == MemoryDomain::MetalDevice)
                return GpuBackend::Metal;
            if (domain == MemoryDomain::HipDevice)
                return GpuBackend::HIP;
            return std::nullopt;
        }

        MemoryInfo query_backend_memory(const MemoryDomain domain, bool free_query) {
            const auto backend = domain_backend(domain).value_or(GpuBackend::CUDA);
            try {
                return gpu_pressure_memory_info(backend, free_query);
            } catch (const std::exception& error) {
                LOG_WARN("Cannot query {} memory headroom: {}", gpu_backend_name(backend), error.what());
                return {};
            }
        }
        size_t query_device_free_bytes(MemoryDomain domain) { return query_backend_memory(domain, true).free_bytes; }
        size_t query_device_total_bytes(MemoryDomain domain) { return query_backend_memory(domain, false).total_bytes; }
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
        std::atomic<MemoryDomain> last_failure_domain{MemoryDomain::CudaDevice};
        std::chrono::steady_clock::time_point last_recover_check{};

        std::mutex probe_mutex;
        std::atomic<bool> probe_armed{false};
        std::function<bool(MemoryDomain, size_t)> alloc_probe;
        std::function<size_t(MemoryDomain)> free_probe;

        // One reserve per backend, indexed by GpuBackend on every platform.
        static constexpr size_t kReserveSlots = kGpuBackendCount;
        std::array<std::once_flag, kReserveSlots> reserve_once;
        std::array<size_t, kReserveSlots> reserve_bytes{};

        static thread_local bool in_episode;

        size_t query_free(MemoryDomain domain) {
            {
                std::lock_guard<std::mutex> lock(probe_mutex);
                if (free_probe) {
                    return free_probe(domain);
                }
            }
            if (is_device_heap(domain)) {
                return query_device_free_bytes(domain);
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
        register_gpu_pressure_clients(*this);
    }

    MemoryPressureCoordinator::~MemoryPressureCoordinator() {
        delete impl_;
    }

    MemoryPressureCoordinator& MemoryPressureCoordinator::instance() {
        static MemoryPressureCoordinator coordinator;
        return coordinator;
    }

    size_t MemoryPressureCoordinator::reserve_bytes(const MemoryDomain domain) const noexcept {
        const size_t index = static_cast<size_t>(domain_backend(domain).value_or(GpuBackend::CUDA));
        std::call_once(impl_->reserve_once[index], [this, domain, index]() {
            size_t reserve = static_cast<size_t>(512) * 1024 * 1024;
            if (const auto mb = environment::unsigned_integer<unsigned long long>("LFS_VRAM_RESERVE_MB");
                mb && *mb > 0) {
                constexpr size_t MIB = size_t{1024} * 1024;
                reserve = *mb > std::numeric_limits<size_t>::max() / MIB
                              ? std::numeric_limits<size_t>::max()
                              : static_cast<size_t>(*mb) * MIB;
            }
            const size_t total = query_device_total_bytes(domain);
            const size_t floor = static_cast<size_t>(128) * 1024 * 1024;
            const size_t ceiling = total > 0 ? total / 4 : reserve;
            reserve = std::max(reserve, floor);
            if (ceiling > floor) {
                reserve = std::min(reserve, ceiling);
            }
            impl_->reserve_bytes[index] = reserve;
        });
        return impl_->reserve_bytes[index];
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
        if (!domain_backend(failure.domain) && cuda_is_unavailable()) {
            return 0;
        }
        Impl::in_episode = true;
        struct Guard {
            ~Guard() { Impl::in_episode = false; }
        } guard;

        std::lock_guard<std::mutex> episode_lock(impl_->episode_mutex);
        if (!domain_backend(failure.domain) && cuda_is_unavailable()) {
            return 0;
        }

        const size_t reserve = reserve_bytes(failure.domain);
        const size_t target = saturating_add(failure.requested_bytes, reserve);
        const size_t free_before = impl_->query_free(failure.domain);

        // A concurrent episode may already have relieved the pressure.
        if (is_device_heap(failure.domain) && free_before >= target) {
            return 0;
        }

        const uint64_t episode_id = impl_->episode_counter.fetch_add(1) + 1;
        // Preview degradation belongs to failures that reached the frame boundary.
        // Training allocations share the device heap, but their retry episodes must
        // neither lower viewer quality nor replace its recovery target.
        const bool viewer_failure = context == PressureContext::RenderThread &&
                                    is_device_heap(failure.domain);
        if (viewer_failure) {
            impl_->pressure_active.store(true);
            impl_->last_target_free.store(target);
            impl_->last_failure_domain.store(failure.domain);
        }

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
        if (viewer_failure && satisfied) {
            // Reclaim already freed enough for the failed viewer allocation and its
            // reserve. Retry at full quality instead of waiting for extra headroom.
            impl_->pressure_active.store(false);
        }
        const size_t observed_released = free_after > free_before ? free_after - free_before : 0;

        record_gpu_pressure_breadcrumb();

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
        if (!domain_backend(failure.domain) && cuda_is_unavailable()) {
            return false;
        }
        const size_t target = saturating_add(failure.requested_bytes, reserve_bytes(failure.domain));
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
        result.safety_reserve_bytes = reserve_bytes(domain);
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
        if (!impl_->pressure_active.load()) {
            return;
        }
        // Do not clear a lease while another thread is recording a new failure.
        std::unique_lock episode_lock(impl_->episode_mutex, std::try_to_lock);
        if (!episode_lock.owns_lock()) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - impl_->last_recover_check < std::chrono::milliseconds(250)) {
            return;
        }
        impl_->last_recover_check = now;

        const size_t target = impl_->last_target_free.load();
        const size_t hysteresis = saturating_add(target, target / 5); // require 20% headroom to restore
        const auto recovery_domain = impl_->last_failure_domain.load();
        if (impl_->query_free(recovery_domain) >= hysteresis) {
            impl_->pressure_active.store(false);
            LOG_INFO("VRAM pressure lease released; sustained headroom restored");
        }
    }

    uint64_t MemoryPressureCoordinator::episode_count() const noexcept {
        return impl_->episode_counter.load();
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
        impl_->last_failure_domain.store(MemoryDomain::CudaDevice);
        impl_->last_recover_check = {};
    }

} // namespace lfs::core
