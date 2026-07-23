/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

// Phase 6C device-fault ABI + host registry surface.
// CUDA-safe C++20 subset (ABI + device helpers + cudaError_t registry entries):
// includable from .cu and .cpp with no Error/Result. Host-only harvest→Error and
// graph-capture Unsupported helpers are gated by !__CUDACC__ (phase-6c P2).
// Frozen layout and protocol: .codex_tmp/phase-6c-device-fault-spec.md §0.1 / §1 / §9.

#include "core/export.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#if !defined(__CUDACC__)
#include "core/error.hpp"
#include "core/source_site.hpp"
#endif

namespace lfs::core {

    // Closed set frozen by phase-6c §9 Ambiguity sign-off 2. Further
    // enumerators require a future phase freeze. Stored in
    // DeviceFaultRecord::code as uint32_t; zero is the sole no-fault sentinel.
    enum class DeviceFaultCode : std::uint32_t {
        NoFault = 0,
        IndexOutOfBounds = 1,
    };

    // FROZEN VERBATIM five-field ABI (spec §0.1.2 / master §5.8). Do not add,
    // remove, reorder, or rename fields. Naturally aligned scalars only.
    struct LFS_CORE_API DeviceFaultRecord {
        std::uint32_t code; // zero = no fault; otherwise first fault wins
        std::uint32_t op_id;
        std::int64_t value;
        std::int64_t bound;
        std::uint64_t thread_id;
    };

    static_assert(std::is_trivially_copyable_v<DeviceFaultRecord>,
                  "DeviceFaultRecord crosses the nvcc/host boundary; "
                  "it must stay trivially copyable");
    static_assert(std::is_standard_layout_v<DeviceFaultRecord>,
                  "DeviceFaultRecord must remain standard-layout for offsetof ABI tests");
    static_assert(sizeof(DeviceFaultRecord) == 32,
                  "DeviceFaultRecord ABI size must stay 32 bytes on this target");
    static_assert(alignof(DeviceFaultRecord) == 8,
                  "DeviceFaultRecord natural alignment must stay 8");
    static_assert(offsetof(DeviceFaultRecord, code) == 0);
    static_assert(offsetof(DeviceFaultRecord, op_id) == 4);
    static_assert(offsetof(DeviceFaultRecord, value) == 8);
    static_assert(offsetof(DeviceFaultRecord, bound) == 16);
    static_assert(offsetof(DeviceFaultRecord, thread_id) == 24);
    static_assert(sizeof(std::uint32_t) == sizeof(unsigned int),
                  "DeviceFaultRecord::code CAS view requires uint32_t == unsigned int");

#if defined(__CUDACC__)
    // Device-side first-fault protocol (spec §1.4). Call ONLY on the failure
    // branch of a bounds (or similar) check — never on the hot success path.
    //
    // Semantics:
    //   1. atomicCAS on `code`: expected NoFault(0) → IndexOutOfBounds(1).
    //   2. If CAS wins: write op_id/value/bound/thread_id (non-atomic stores).
    //   3. If CAS loses: leave the record untouched (another thread owns first fault).
    //   4. trap_after_record: when true and this thread won the CAS, execute
    //      __trap() AFTER the record is fully written. Host publishes this bool
    //      from diagnostic_mode_enabled(DeviceTrap); device code never parses
    //      mode strings (phase-6c §9 Ruling 1). Production passes false.
    //
    // Returns true iff this thread won the first-fault CAS.
    //
    // thread_id packing (recommended): pack block/thread as
    //   (static_cast<std::uint64_t>(blockIdx.x) << 32) | threadIdx.x
    // or any stable linear tid the host Error context can document.
    __device__ inline bool device_fault_try_record_first(
        DeviceFaultRecord* record,
        const std::uint32_t op_id,
        const std::int64_t value,
        const std::int64_t bound,
        const std::uint64_t thread_id,
        const bool trap_after_record) {
        if (record == nullptr) {
            return false;
        }
        const unsigned int desired =
            static_cast<unsigned int>(DeviceFaultCode::IndexOutOfBounds);
        const unsigned int previous = atomicCAS(
            reinterpret_cast<unsigned int*>(&record->code),
            /*compare=*/0u,
            desired);
        if (previous != 0u) {
            return false;
        }
        record->op_id = op_id;
        record->value = value;
        record->bound = bound;
        record->thread_id = thread_id;
        if (trap_after_record) {
            __trap();
        }
        return true;
    }

    // Allocation-free device-side clear of a fault record. Host protocol prefers
    // enqueuing reset via device_fault_slot_enqueue_reset on the same stream
    // before a checked range; this helper exists for in-kernel test fixtures.
    __device__ inline void device_fault_clear(DeviceFaultRecord* record) {
        if (record == nullptr) {
            return;
        }
        record->code = 0;
        record->op_id = 0;
        record->value = 0;
        record->bound = 0;
        record->thread_id = 0;
    }
#endif // defined(__CUDACC__)

    // ---------------------------------------------------------------------------
    // Host-only per-stream registry (defined in device_fault.cpp).
    //
    // Storage (phase-6c §9 Ruling 2): each slot owns a dedicated cudaMalloc'd
    // DeviceFaultRecord plus pinned host staging. NEVER CudaMemoryPool memory.
    // Teardown frees via LFS_CUDA_LOG_TEARDOWN and is hooked into
    // teardown_gpu_before_exit() BEFORE Tensor::shutdown_memory_pool().
    //
    // Thread contract (spec §2.3): map create/free is mutex-protected. Reset /
    // harvest enqueue on a live stream assumes single-owner-per-stream — tensor
    // ops must not share a stream across host threads without external sync.
    // Concurrent acquire/reset/harvest for the *same* stream from multiple host
    // threads without external synchronization is a precondition violation.
    // ---------------------------------------------------------------------------

    // Lazy-allocate the per-stream slot if needed. On success writes the device
    // pointer suitable for kernel launch (stable until free/teardown).
    [[nodiscard]] LFS_CORE_API cudaError_t device_fault_slot_acquire(
        cudaStream_t stream,
        DeviceFaultRecord** out_device_record) noexcept;

    // Enqueue zeroing of the device record on `stream` (FIFO before the checked
    // kernel range). Acquires the slot if missing.
    [[nodiscard]] LFS_CORE_API cudaError_t device_fault_slot_enqueue_reset(
        cudaStream_t stream) noexcept;

    // Enqueue async D2H of the device record into the slot's host staging.
    // Does NOT synchronize (spec §1.5: no success-path D2H sync).
    [[nodiscard]] LFS_CORE_API cudaError_t device_fault_slot_enqueue_harvest(
        cudaStream_t stream) noexcept;

    // Host-side read of pinned staging AFTER an existing semantic safe-point wait
    // that already orders the harvest copy. Does not wait or synchronize.
    // Returns a zeroed NoFault record if no slot exists for `stream`.
    [[nodiscard]] LFS_CORE_API DeviceFaultRecord device_fault_slot_consume(
        cudaStream_t stream) noexcept;

    // Drain and free every registered slot (device + host). No-throw, idempotent.
    // Called from teardown_gpu_before_exit() before pool shutdown.
    LFS_CORE_API void device_fault_registry_teardown() noexcept;

    // Free all slots and allow re-acquire (unit tests). No-throw.
    LFS_CORE_API void reset_device_fault_registry_for_testing() noexcept;

    // Host-side launch-prep consumer for DiagnosticMode::DeviceTrap (spec §1.8,
    // phase-6c §9 Ruling 1). Call ONCE per checked launch site; pass the returned
    // bool into kernels as `trap_after_record`. Device code never parses mode
    // strings or calls diagnostic_mode_enabled. Cached bitmask bit-test only —
    // no getenv/alloc/lock/sync on the hot path.
    [[nodiscard]] LFS_CORE_API bool device_fault_trap_after_record_for_launch() noexcept;

#if !defined(__CUDACC__)
    // Host-only harvest → typed Error surface (phase-6c P2). Not visible to nvcc;
    // implementations live in the host TU device_fault.cpp and mirror the
    // cuda_error_typed cold-path construction idiom (make_error + Exception).

    // Builds Error{BoundsViolation, CUDA} from a nonzero DeviceFaultRecord,
    // carrying op_id/value/bound/thread_id as SmallFields. When the process-wide
    // CUDA-unavailable latch is set (cuda_is_unavailable()), returns Unavailable
    // instead — dead-context wins over BoundsViolation (§0.3 / §9 Ruling 2).
    // Precondition for the BoundsViolation path: record.code != 0.
    [[nodiscard]] LFS_CORE_API Error make_device_fault_error(
        const DeviceFaultRecord& record,
        std::string_view operation_tag,
        SourceSite location,
        std::uintptr_t stream = 0);

    // Throws lfs::Exception(make_device_fault_error(...)). When DeviceTrap is
    // enabled, reports the Error then aborts (diagnostic subprocess fatal path
    // per §1.8 preferred host trap site) — never the sole production path.
    [[noreturn]] LFS_CORE_API void throw_device_fault_error(
        const DeviceFaultRecord& record,
        std::string_view operation_tag,
        SourceSite location,
        std::uintptr_t stream = 0);

    // Host safe-point harvest: consume staging; if code != 0, throw via
    // throw_device_fault_error. Returns normally on NoFault. Does not wait.
    LFS_CORE_API void device_fault_slot_consume_or_throw(
        cudaStream_t stream,
        std::string_view operation_tag,
        SourceSite location);

    // Graph-capture typed Unsupported (spec §1.9). Domain CUDA.
    [[nodiscard]] LFS_CORE_API Error make_device_fault_graph_capture_error(
        cudaStream_t stream,
        SourceSite location);

    [[noreturn]] LFS_CORE_API void throw_device_fault_graph_capture_error(
        cudaStream_t stream,
        SourceSite location);

    class ValidatedIndexToken;

    // Declared before the class so the friend declaration inside it is not
    // the first declaration MSVC sees — attaching dllexport/dllimport only
    // on a later redeclaration is C2375 (redefinition; different linkage).
    [[nodiscard]] LFS_CORE_API ValidatedIndexToken issue_validated_index_token(
        const void* storage_identity,
        std::uint64_t mutation_version,
        int device_ordinal,
        std::uint64_t producer_event_or_range);

    // ---------------------------------------------------------------------------
    // ValidatedIndexToken (host-only; phase-6c §1.10 / P3).
    //
    // Opaque, non-copyable, movable-only. Private ctor; only the friend issuer
    // can construct a live token. Binds storage identity, mutation/version,
    // device ordinal, and producer event/range. Contiguity alone is NOT
    // validation. Stale/forged/mismatched tokens select the checked kernel path
    // (never assert / never UB). Unchecked index launchers that skip the
    // device-fault protocol require a live matching token.
    // ---------------------------------------------------------------------------
    class LFS_CORE_API ValidatedIndexToken {
    public:
        ValidatedIndexToken(const ValidatedIndexToken&) = delete;
        ValidatedIndexToken& operator=(const ValidatedIndexToken&) = delete;
        ValidatedIndexToken(ValidatedIndexToken&&) noexcept = default;
        ValidatedIndexToken& operator=(ValidatedIndexToken&&) noexcept = default;

        // True iff every bound field matches the live tensor/provenance snapshot.
        // Mismatch → caller MUST take the checked device-fault path (not assert).
        [[nodiscard]] bool matches(const void* storage_identity,
                                   std::uint64_t mutation_version,
                                   int device_ordinal,
                                   std::uint64_t producer_event_or_range) const noexcept;

        [[nodiscard]] const void* storage_identity() const noexcept { return storage_identity_; }
        [[nodiscard]] std::uint64_t mutation_version() const noexcept { return mutation_version_; }
        [[nodiscard]] int device_ordinal() const noexcept { return device_ordinal_; }
        [[nodiscard]] std::uint64_t producer_event_or_range() const noexcept {
            return producer_event_or_range_;
        }

    private:
        friend ValidatedIndexToken issue_validated_index_token(
            const void* storage_identity,
            std::uint64_t mutation_version,
            int device_ordinal,
            std::uint64_t producer_event_or_range);

        ValidatedIndexToken(const void* storage_identity,
                            std::uint64_t mutation_version,
                            int device_ordinal,
                            std::uint64_t producer_event_or_range) noexcept
            : storage_identity_(storage_identity),
              mutation_version_(mutation_version),
              device_ordinal_(device_ordinal),
              producer_event_or_range_(producer_event_or_range) {}

        const void* storage_identity_ = nullptr;
        std::uint64_t mutation_version_ = 0;
        int device_ordinal_ = -1;
        std::uint64_t producer_event_or_range_ = 0;
    };

    // Validator issuance API (friend of ValidatedIndexToken). 6C ships this
    // minimal issuer for tests + dual-path hooks; full production provenance
    // issuance for every index op is TENSOR_LIB.
    [[nodiscard]] LFS_CORE_API ValidatedIndexToken issue_validated_index_token(
        const void* storage_identity,
        std::uint64_t mutation_version,
        int device_ordinal,
        std::uint64_t producer_event_or_range);

    // Test/consumer drain: wait for `stream` (the wait the algorithm already
    // needs before interpreting results), then harvest. Does not invent a
    // success-path sync inside the tensor op itself — callers that already
    // wait (or tests that need a deterministic BoundsViolation surface) use
    // this after their existing wait expression. Internally: cudaStreamSynchronize
    // then device_fault_slot_consume_or_throw.
    LFS_CORE_API void device_fault_await_and_consume_or_throw(
        cudaStream_t stream,
        std::string_view operation_tag,
        SourceSite location);
#endif // !defined(__CUDACC__)

} // namespace lfs::core
