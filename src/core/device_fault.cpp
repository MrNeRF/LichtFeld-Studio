/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/device_fault.hpp"

#include "core/tensor_cuda_interop.hpp"

#include "core/cuda_error.hpp"
#include "core/cuda_error_typed.hpp"
#include "core/error_reporter.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace lfs::core {
    namespace {

        // Kernels write the mapped record through its device alias; host reads
        // and clears occur only after the producing stream completes.
        struct DeviceFaultSlot {
            DeviceFaultRecord* device_record = nullptr; // device alias of the mapped record
            DeviceFaultRecord* host_staging = nullptr;  // cudaHostAlloc mapped pinned
            bool armed = false;
        };

        std::mutex g_registry_mu;
        std::unordered_map<cudaStream_t, DeviceFaultSlot> g_slots;
        std::uint64_t g_registry_generation = 0;
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

        // The caller holds g_registry_mu; failure leaves out empty.
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
            // Only a synchronized consumer clears a record after initialization.
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

            struct CachedSlot {
                cudaStream_t stream = nullptr;
                DeviceFaultSlot* slot = nullptr;
                std::uint64_t generation = 0;
            };
            thread_local CachedSlot cache;
            if (cache.slot != nullptr && cache.stream == stream &&
                cache.generation == g_registry_generation) {
                *out_slot = cache.slot;
                return cudaSuccess;
            }

            const auto it = g_slots.find(stream);
            if (it != g_slots.end()) {
                *out_slot = &it->second;
                cache = {stream, *out_slot, g_registry_generation};
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
            cache = {stream, *out_slot, g_registry_generation};
            return cudaSuccess;
        }

        void drain_all_slots_locked() noexcept {
            ++g_registry_generation;
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

    cudaError_t device_fault_slot_arm(cudaStream_t stream, DeviceFaultRecord** out_device_record) noexcept {
        if (out_device_record == nullptr)
            return cudaErrorInvalidValue;
        *out_device_record = nullptr;
        // The legacy null stream cannot be captured.
        if (stream != nullptr) {
            const auto capture_status = reject_if_graph_capturing(stream);
            if (capture_status != cudaSuccess)
                return capture_status;
        }
        std::lock_guard lock(g_registry_mu);
        DeviceFaultSlot* slot = nullptr;
        const auto status = get_or_create_slot_locked(stream, &slot);
        if (status != cudaSuccess)
            return status;
        slot->armed = true;
        *out_device_record = slot->device_record;
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
        const auto record = *it->second.host_staging;
        if (record.code != static_cast<std::uint32_t>(DeviceFaultCode::NoFault))
            std::memset(it->second.host_staging, 0, sizeof(DeviceFaultRecord));
        it->second.armed = false;
        return record;
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

    void device_fault_registry_consume_or_throw(const SourceSite location, bool device_synchronized) {
        std::vector<cudaStream_t> streams;
        {
            std::lock_guard lock(g_registry_mu);
            for (const auto& [stream, slot] : g_slots) {
                if (slot.armed)
                    streams.push_back(stream);
            }
        }
        std::exception_ptr failure;
        for (const auto stream : streams) {
            try {
                if (!device_synchronized) {
                    // Pool retirement waits for stream work before the handle is destroyed.
                    if (!is_stream_retired(stream)) {
                        const auto status = cudaStreamQuery(stream);
                        if (status == cudaErrorNotReady)
                            continue;
                        if (status == cudaErrorInvalidResourceHandle) {
                            (void)cudaGetLastError();
                            continue;
                        }
                        ensure_cuda_success(status, "cudaStreamQuery(device fault)", {}, location);
                    }
                }
                device_fault_slot_consume_or_throw(stream, "tensor.synchronize", location);
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): Preserve the first error while clearing all completed slots.
                if (!failure)
                    failure = std::current_exception();
            }
        }
        if (failure)
            std::rethrow_exception(failure);
    }

    Error make_device_fault_graph_capture_error(const cudaStream_t stream,
                                                const SourceSite location) {
        return make_error(ErrorInit{
            .code = ErrorCode::Unsupported,
            .domain = ErrorDomain::CUDA,
            .detail = "device-fault checking is unsupported under CUDA graph "
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
