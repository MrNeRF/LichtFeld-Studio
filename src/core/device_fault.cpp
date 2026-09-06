/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/device_fault.hpp"

#include "core/cuda_error.hpp"
#include "core/cuda_error_typed.hpp"
#include "core/error_reporter.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <format>
#include <mutex>
#include <unordered_map>

namespace lfs::core {
    namespace {

        // The record lives in host-mapped pinned memory: the kernel writes it
        // through the device alias and the host reads it after the stream wait,
        // so the checked range needs no device-to-host copy behind the kernel
        // (that copy was scheduled 8 us late by the runtime on driver 580 and
        // doubled the cost of small Assert-mode index reads).
        struct DeviceFaultSlot {
            DeviceFaultRecord* device_record = nullptr; // device alias of the mapped record
            DeviceFaultRecord* host_staging = nullptr;  // cudaHostAlloc mapped pinned
        };

        std::mutex g_registry_mu;
        std::unordered_map<cudaStream_t, DeviceFaultSlot> g_slots;
        // Set by device_fault_registry_teardown(); cleared only by the testing reset.
        std::atomic<bool> g_registry_torn_down{false};

        void free_slot_buffers(DeviceFaultSlot& slot) noexcept {
            slot.device_record = nullptr;
            if (slot.host_staging != nullptr) {
                LFS_CUDA_LOG_TEARDOWN(cudaFreeHost(slot.host_staging),
                                      nullptr,
                                      "device_fault: free host staging");
                slot.host_staging = nullptr;
            }
        }

        // Allocate device record + pinned staging. Caller holds g_registry_mu.
        // On failure leaves *out empty and returns the failing cudaError_t.
        [[nodiscard]] cudaError_t allocate_slot_buffers(DeviceFaultSlot& out) noexcept {
            out = {};
            void* host_ptr = nullptr;
            const cudaError_t host_status = cudaHostAlloc(
                &host_ptr, sizeof(DeviceFaultRecord), cudaHostAllocMapped);
            if (host_status != cudaSuccess) {
                return host_status;
            }
            void* device_ptr = nullptr;
            const cudaError_t map_status = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
            if (map_status != cudaSuccess) {
                LFS_CUDA_LOG_TEARDOWN(cudaFreeHost(host_ptr),
                                      nullptr,
                                      "device_fault: free host record after mapping failure");
                return map_status;
            }

            out.device_record = static_cast<DeviceFaultRecord*>(device_ptr);
            out.host_staging = static_cast<DeviceFaultRecord*>(host_ptr);
            // The record starts clean so a premature consume reads NoFault; the
            // mandatory pre-range enqueue_reset zeroes it in stream order.
            std::memset(out.host_staging, 0, sizeof(DeviceFaultRecord));
            return cudaSuccess;
        }

        // Look up or create the slot for `stream`. Caller holds g_registry_mu.
        [[nodiscard]] cudaError_t get_or_create_slot_locked(
            const cudaStream_t stream,
            DeviceFaultSlot** out_slot) noexcept {
            if (out_slot == nullptr) {
                return cudaErrorInvalidValue;
            }
            *out_slot = nullptr;

            if (g_registry_torn_down.load(std::memory_order_acquire)) {
                return cudaErrorCudartUnloading;
            }

            const auto it = g_slots.find(stream);
            if (it != g_slots.end()) {
                *out_slot = &it->second;
                return cudaSuccess;
            }

            DeviceFaultSlot created;
            const cudaError_t alloc_status = allocate_slot_buffers(created);
            if (alloc_status != cudaSuccess) {
                return alloc_status;
            }

            const auto [inserted_it, inserted] = g_slots.emplace(stream, created);
            if (!inserted) {
                // Should be unreachable under the mutex; free and surface internal error.
                free_slot_buffers(created);
                return cudaErrorUnknown;
            }
            *out_slot = &inserted_it->second;
            return cudaSuccess;
        }

        void drain_all_slots_locked() noexcept {
            for (auto& entry : g_slots) {
                free_slot_buffers(entry.second);
            }
            g_slots.clear();
        }

        // Graph-capture guard (spec §1.9): no silent sync, no capture-breaking
        // enqueue. Active capture OR inability to prove a capture-free path →
        // cudaErrorStreamCaptureUnsupported (host typed path maps to
        // ErrorCode::Unsupported via make_device_fault_graph_capture_error).
        [[nodiscard]] cudaError_t reject_if_graph_capturing(
            const cudaStream_t stream) noexcept {
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            const cudaError_t query = cudaStreamIsCapturing(stream, &capture_status);
            if (query != cudaSuccess) {
                // Cannot prove capture-free — same stance as active capture.
                return cudaErrorStreamCaptureUnsupported;
            }
            if (capture_status != cudaStreamCaptureStatusNone) {
                return cudaErrorStreamCaptureUnsupported;
            }
            return cudaSuccess;
        }

    } // namespace

    cudaError_t device_fault_slot_acquire(
        const cudaStream_t stream,
        DeviceFaultRecord** out_device_record) noexcept {
        if (out_device_record == nullptr) {
            return cudaErrorInvalidValue;
        }
        *out_device_record = nullptr;

        std::lock_guard<std::mutex> lock(g_registry_mu);
        DeviceFaultSlot* slot = nullptr;
        const cudaError_t status = get_or_create_slot_locked(stream, &slot);
        if (status != cudaSuccess) {
            return status;
        }
        *out_device_record = slot->device_record;
        return cudaSuccess;
    }

    cudaError_t device_fault_slot_enqueue_reset(const cudaStream_t stream) noexcept {
        // Spec §1.9: host entry of reset rejects graph capture before any enqueue.
        const cudaError_t capture_status = reject_if_graph_capturing(stream);
        if (capture_status != cudaSuccess) {
            return capture_status;
        }

        std::lock_guard<std::mutex> lock(g_registry_mu);
        DeviceFaultSlot* slot = nullptr;
        const cudaError_t acquire_status = get_or_create_slot_locked(stream, &slot);
        if (acquire_status != cudaSuccess) {
            return acquire_status;
        }
        // The record is written only by Assert-mode kernels, and every Assert
        // operation waits on its stream and consumes the record before it
        // returns, so no writer is in flight here: the host clears the mapped
        // record directly. A device memset into host memory stalled the stream
        // (training pairs at 75c01f402 lost up to 10 s in 7000 iterations).
        std::memset(slot->host_staging, 0, sizeof(DeviceFaultRecord));
        (void)stream;
        return cudaSuccess;
    }

    cudaError_t device_fault_slot_enqueue_harvest(const cudaStream_t stream) noexcept {
        // Spec §1.9: harvest host entry also rejects graph capture (no silent sync).
        const cudaError_t capture_status = reject_if_graph_capturing(stream);
        if (capture_status != cudaSuccess) {
            return capture_status;
        }

        std::lock_guard<std::mutex> lock(g_registry_mu);
        DeviceFaultSlot* slot = nullptr;
        const cudaError_t acquire_status = get_or_create_slot_locked(stream, &slot);
        if (acquire_status != cudaSuccess) {
            return acquire_status;
        }
        // The record is host-mapped: whatever the checked range wrote is visible
        // on the host once the caller's wait on `stream` completes (spec §1.5
        // step 4 keeps the no-synchronize rule; there is nothing to copy).
        (void)slot;
        return cudaSuccess;
    }

    DeviceFaultRecord device_fault_slot_consume(const cudaStream_t stream) noexcept {
        DeviceFaultRecord clean{};
        clean.code = static_cast<std::uint32_t>(DeviceFaultCode::NoFault);
        clean.op_id = 0;
        clean.value = 0;
        clean.bound = 0;
        clean.thread_id = 0;

        std::lock_guard<std::mutex> lock(g_registry_mu);
        const auto it = g_slots.find(stream);
        if (it == g_slots.end() || it->second.host_staging == nullptr) {
            return clean;
        }
        // Host read of the mapped record after the caller's wait on the stream.
        return *it->second.host_staging;
    }

    void device_fault_registry_teardown() noexcept {
        // Idempotent: first call drains; subsequent calls are no-ops.
        if (g_registry_torn_down.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        std::lock_guard<std::mutex> lock(g_registry_mu);
        drain_all_slots_locked();
    }

    void reset_device_fault_registry_for_testing() noexcept {
        std::lock_guard<std::mutex> lock(g_registry_mu);
        drain_all_slots_locked();
        g_registry_torn_down.store(false, std::memory_order_release);
    }

    bool device_fault_trap_after_record_for_launch() noexcept {
        // Ruling 1: host queries the mode; kernels receive only this bool.
        return diagnostic_mode_enabled(DiagnosticMode::DeviceTrap);
    }

    Error make_device_fault_error(const DeviceFaultRecord& record,
                                  const std::string_view operation_tag,
                                  const SourceSite location,
                                  const std::uintptr_t stream) {
        // §0.3 / §9 Ruling 2: dead context wins over BoundsViolation.
        if (cuda_is_unavailable()) {
            return make_error(ErrorInit{
                .code = ErrorCode::Unavailable,
                .domain = ErrorDomain::CUDA,
                .detail = "device-fault harvest skipped: CUDA is unavailable in this process",
                .detection = location,
                .fields = SmallFields{}
                              .add("stream", static_cast<std::int64_t>(stream))
                              .add("operation_tag",
                                   std::string(operation_tag.empty() ? "<untagged>"
                                                                     : operation_tag)),
            });
        }

        // Mirror cuda_error_typed.cpp make_cuda_failure_seed_error construction.
        return make_error(ErrorInit{
            .code = ErrorCode::BoundsViolation,
            .domain = ErrorDomain::CUDA,
            .detail = std::format(
                "observed at device-fault safe point: {} (code={}, op_id={})",
                operation_tag.empty() ? "<untagged>" : operation_tag,
                record.code,
                record.op_id),
            .detection = location,
            .fields = SmallFields{}
                          .add("op_id", static_cast<std::int64_t>(record.op_id))
                          .add("value", record.value)
                          .add("bound", record.bound)
                          .add("thread_id", record.thread_id)
                          .add("stream", static_cast<std::int64_t>(stream))
                          .add("fault_code", static_cast<std::int64_t>(record.code)),
        });
    }

    void throw_device_fault_error(const DeviceFaultRecord& record,
                                  const std::string_view operation_tag,
                                  const SourceSite location,
                                  const std::uintptr_t stream) {
        Error error = make_device_fault_error(record, operation_tag, location, stream);
        // §1.8 preferred host trap site: after harvest, DeviceTrap → report then
        // abort (diagnostic subprocess). Production leaves DeviceTrap off.
        if (diagnostic_mode_enabled(DiagnosticMode::DeviceTrap) &&
            error.code() == ErrorCode::BoundsViolation) {
            try {
                ErrorReporter::get().report(error, ReportChannel::OwnerLog);
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): diagnostic fatal path must still abort.
            }
            std::abort();
        }
        throw Exception(std::move(error));
    }

    void device_fault_slot_consume_or_throw(const cudaStream_t stream,
                                            const std::string_view operation_tag,
                                            const SourceSite location) {
        // Unavailable latch short-circuit even when staging is clean (dead context).
        if (cuda_is_unavailable()) {
            throw_device_fault_error(DeviceFaultRecord{}, operation_tag, location,
                                     reinterpret_cast<std::uintptr_t>(stream));
        }
        const DeviceFaultRecord record = device_fault_slot_consume(stream);
        if (record.code == static_cast<std::uint32_t>(DeviceFaultCode::NoFault)) {
            return;
        }
        throw_device_fault_error(record, operation_tag, location,
                                 reinterpret_cast<std::uintptr_t>(stream));
    }

    Error make_device_fault_graph_capture_error(const cudaStream_t stream,
                                                const SourceSite location) {
        return make_error(ErrorInit{
            .code = ErrorCode::Unsupported,
            .domain = ErrorDomain::CUDA,
            .detail = "device-fault reset/harvest is unsupported under CUDA graph "
                      "capture (no silent synchronize)",
            .detection = location,
            .fields = SmallFields{}.add(
                "stream", static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(stream))),
        });
    }

    void throw_device_fault_graph_capture_error(const cudaStream_t stream,
                                                const SourceSite location) {
        throw Exception(make_device_fault_graph_capture_error(stream, location));
    }

    bool ValidatedIndexToken::matches(const void* storage_identity,
                                      const std::uint64_t mutation_version,
                                      const int device_ordinal,
                                      const std::uint64_t producer_event_or_range) const noexcept {
        return storage_identity_ == storage_identity &&
               mutation_version_ == mutation_version &&
               device_ordinal_ == device_ordinal &&
               producer_event_or_range_ == producer_event_or_range;
    }

    ValidatedIndexToken issue_validated_index_token(
        const void* storage_identity,
        const std::uint64_t mutation_version,
        const int device_ordinal,
        const std::uint64_t producer_event_or_range) {
        return ValidatedIndexToken(storage_identity, mutation_version, device_ordinal,
                                   producer_event_or_range);
    }

    void device_fault_await_and_consume_or_throw(const cudaStream_t stream,
                                                 const std::string_view operation_tag,
                                                 const SourceSite location) {
        // Consumer/test drain: the wait is the caller's semantic safe point
        // (spec §1.5 step 5). Production index_select does not call this on the
        // happy path; tests and materializing consumers do.
        const cudaError_t sync_status = cudaStreamSynchronize(stream);
        if (sync_status != cudaSuccess) {
            // Surface as a CUDA check failure rather than silently skipping harvest.
            ensure_cuda_success(sync_status, "device_fault_await_and_consume_or_throw",
                                operation_tag, location);
        }
        device_fault_slot_consume_or_throw(stream, operation_tag, location);
    }

} // namespace lfs::core
