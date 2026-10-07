/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/memory_domain.hpp"
#include "core/source_site.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

// Forward declaration only: this header must stay includable from any future
// CUDA translation unit that references AllocationFailure, so it must never
// include the host-C++23-only core/error.hpp. See to_error() below.
namespace lfs {
    class Error;
}

namespace lfs::core {

    // Typed allocation failure. Carries the structured cause across expected /
    // exception boundaries so callers can distinguish OOM from an invariant
    // failure and decide whether a retry is safe.
    class LFS_CORE_API MemoryAllocationError : public std::runtime_error {
    public:
        explicit MemoryAllocationError(const AllocationFailure& failure);

        const AllocationFailure& failure() const noexcept { return failure_; }
        MemoryDomain domain() const noexcept { return failure_.domain; }
        size_t requested_bytes() const noexcept { return failure_.requested_bytes; }

    private:
        AllocationFailure failure_;
    };

    // The safe point at which a reclaim callback may run. The coordinator only
    // dispatches a client whose affinity the calling context permits.
    enum class PressureAffinity : uint8_t {
        ImmediateThreadSafe, // safe from any thread that just failed to allocate
        TrainingSafePoint,   // only at a trainer iteration boundary
        RenderSafePoint,     // only on the render/UI thread between frames
    };

    // Which affinities the current call site is allowed to dispatch.
    enum class PressureContext : uint8_t {
        ImmediateOnly, // arbitrary allocating thread: caches only
        RenderThread,  // GUI frame boundary: immediate + render-safe clients
        TrainingThread // trainer safe point: immediate + training-safe clients
    };

    struct PressureRequest {
        MemoryDomain domain = MemoryDomain::CudaDevice;
        size_t requested_bytes = 0;
        size_t target_free_bytes = 0; // requested + safety reserve
        int device = 0;
        uint64_t episode_id = 0;
    };

    struct ReclaimResult {
        size_t logical_bytes_released = 0; // best-effort; 0 means "unknown, observe"
    };

    // A registered elasticity owner. `estimate` (optional) reports reclaimable
    // bytes for preflight; `shrink` performs the reclaim and must be safe to
    // call from any context permitted by `affinity`. Callbacks must not allocate
    // pool memory or re-enter the coordinator.
    struct PressureClient {
        const char* name = "";
        int priority = 0; // lower runs first (10 = discardable caches, 80 = spill)
        MemoryDomain domain = MemoryDomain::CudaDevice;
        PressureAffinity affinity = PressureAffinity::ImmediateThreadSafe;
        std::function<size_t(const PressureRequest&)> estimate;
        std::function<ReclaimResult(const PressureRequest&)> shrink;
    };

    // Planned device cost of a large operation, used by preflight to refuse
    // before mutation rather than fail mid-operation.
    struct OperationMemoryPlan {
        const char* operation = "";
        size_t persistent_device_bytes = 0;
        size_t temporary_device_bytes = 0;
        size_t old_new_overlap_bytes = 0;
    };

    struct PreflightResult {
        bool ok = false;
        size_t effective_free_bytes = 0;
        size_t required_peak_bytes = 0;
        size_t reclaimable_bytes = 0;
        size_t safety_reserve_bytes = 0;
    };

    // Process-wide coordinator. Owns policy and diagnostics, never memory. The
    // fast allocation path never enters it; it is invoked only after a typed OOM
    // or at a large-operation preflight boundary.
    class LFS_CORE_API MemoryPressureCoordinator {
    public:
        static MemoryPressureCoordinator& instance();

        void register_client(PressureClient client);

        // Runs one coalesced pressure episode for `failure`, dispatching clients
        // permitted by `context` in ascending priority until the observed free
        // memory reaches the target or the clients are exhausted. Reentrant calls
        // (a shrink callback that faults) are refused. Returns the observed bytes
        // freed for the failing domain.
        size_t run_episode(const AllocationFailure& failure,
                           PressureContext context = PressureContext::ImmediateOnly);

        // Runs an episode and reports whether retrying the failed allocation is
        // worthwhile (observed free grew or now meets the target).
        bool relieve_and_should_retry(const AllocationFailure& failure,
                                      PressureContext context = PressureContext::ImmediateOnly);

        // Conservative feasibility for a planned operation. Does not mutate state.
        PreflightResult preflight(const OperationMemoryPlan& plan,
                                  MemoryDomain domain = MemoryDomain::CudaDevice) const;

        // LFS_VRAM_RESERVE_MB, default 512 MiB, clamped to [128 MiB, device VRAM/4].
        size_t reserve_bytes(MemoryDomain domain) const noexcept;

        // Preview pressure lease: set only by a device allocation failure at the
        // render-thread boundary that reclaim could not satisfy. Training/host
        // failures never activate it or replace its recovery target.
        bool pressure_active() const noexcept;

        // Throttled recovery check for the render thread. Cheap no-op unless a
        // lease is active; when active it samples free memory at most a few times
        // per second and clears the lease once sustained headroom returns.
        void maybe_recover();

        uint64_t episode_count() const noexcept;

        // Test-only deterministic injection. `probe` returning true simulates a
        // native OOM at a checked allocation site for the given domain/bytes.
        void set_allocation_probe(std::function<bool(MemoryDomain, size_t)> probe);
        bool probe_should_fail(MemoryDomain domain, size_t bytes) const;

        // Test-only override of the free-memory query, so episode logic can be
        // exercised without real device pressure.
        void set_free_memory_probe(std::function<size_t(MemoryDomain)> probe);

        void reset_for_testing();

        MemoryPressureCoordinator(const MemoryPressureCoordinator&) = delete;
        MemoryPressureCoordinator& operator=(const MemoryPressureCoordinator&) = delete;

    private:
        MemoryPressureCoordinator();
        ~MemoryPressureCoordinator();

        struct Impl;
        Impl* impl_;
    };

} // namespace lfs::core
