/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "training_snapshot_service.hpp"

#include "checkpoint.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error_typed.hpp"
#include "core/logger.hpp"
#include "core/tensor_serialization_sink.hpp"
#include "strategies/istrategy.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <format>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <ostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <streambuf>
#include <thread>
#include <utility>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace lfs::training {

    namespace {

        using Clock = std::chrono::steady_clock;
        using Milliseconds =
            std::chrono::duration<double, std::milli>;

        constexpr std::uint64_t MAX_PINNED_RING_BYTES =
            512ull * 1024 * 1024;
        constexpr std::uint64_t MIN_HOST_MEMORY_RESERVE_BYTES =
            4ull * 1024 * 1024 * 1024;
        constexpr std::uint64_t HOST_MEMORY_GATE_HEADROOM_BYTES =
            768ull * 1024 * 1024;

        void require_cuda(
            const cudaError_t status,
            const std::string_view operation) {
            if (status != cudaSuccess) {
                throw std::runtime_error(std::format(
                    "{}: {}", operation,
                    cudaGetErrorString(status)));
            }
        }

        [[nodiscard]] lfs::Error snapshot_error(
            const lfs::ErrorCode code,
            std::string detail,
            const lfs::core::SourceSite source) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::Training,
                .user_message =
                    "The training snapshot could not be captured.",
                .detail = std::move(detail),
                .detection = source,
            });
        }

        std::uint64_t read_rss_bytes() {
#if defined(__linux__)
            std::ifstream input("/proc/self/status");
            std::string line;
            while (std::getline(input, line)) {
                if (!line.starts_with("VmRSS:")) {
                    continue;
                }
                unsigned long long kib = 0;
                if (std::sscanf(
                        line.c_str(), "VmRSS: %llu",
                        &kib) == 1) {
                    return static_cast<std::uint64_t>(
                               kib) *
                           1024;
                }
                break;
            }
#endif
            return 0;
        }

        struct HostMemoryInfo {
            std::uint64_t total_bytes = 0;
            std::uint64_t available_bytes = 0;
        };

        HostMemoryInfo read_host_memory_info() {
#if defined(_WIN32)
            MEMORYSTATUSEX status{};
            status.dwLength = sizeof(status);
            if (GlobalMemoryStatusEx(&status)) {
                return {
                    .total_bytes = status.ullTotalPhys,
                    .available_bytes = status.ullAvailPhys,
                };
            }
#elif defined(__linux__)
            std::ifstream input("/proc/meminfo");
            std::string line;
            HostMemoryInfo result;
            while (std::getline(input, line)) {
                unsigned long long kib = 0;
                if (std::sscanf(
                        line.c_str(), "MemTotal: %llu kB",
                        &kib) == 1) {
                    result.total_bytes =
                        static_cast<std::uint64_t>(kib) * 1024;
                } else if (std::sscanf(
                               line.c_str(),
                               "MemAvailable: %llu kB",
                               &kib) == 1) {
                    result.available_bytes =
                        static_cast<std::uint64_t>(kib) * 1024;
                }
            }
            return result;
#endif
            return {};
        }

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((target("avx2")))
#endif
        void
        non_temporal_copy(
            void* destination,
            const void* source,
            std::size_t bytes) {
            auto* dst =
                static_cast<std::uint8_t*>(destination);
            const auto* src =
                static_cast<const std::uint8_t*>(source);
            while (bytes > 0 &&
                   (reinterpret_cast<std::uintptr_t>(dst) &
                    31u) != 0) {
                *dst++ = *src++;
                --bytes;
            }
            const auto vectors = bytes / 32;
            auto* vector_dst =
                reinterpret_cast<__m256i*>(dst);
            const auto* vector_src =
                reinterpret_cast<const __m256i*>(src);
            const bool source_aligned =
                (reinterpret_cast<std::uintptr_t>(src) &
                 31u) == 0;
            for (std::size_t index = 0;
                 index < vectors; ++index) {
                const __m256i value =
                    source_aligned
                        ? _mm256_load_si256(
                              vector_src + index)
                        : _mm256_loadu_si256(
                              vector_src + index);
                _mm256_stream_si256(
                    vector_dst + index, value);
            }
            dst += vectors * 32;
            src += vectors * 32;
            bytes -= vectors * 32;
            while (bytes-- > 0) {
                *dst++ = *src++;
            }
            _mm_sfence();
        }
#else
        void non_temporal_copy(
            void* destination,
            const void* source,
            const std::size_t bytes) {
            std::memcpy(destination, source, bytes);
        }
#endif

        struct TensorLayoutWitness {
            const void* source_pointer = nullptr;
            lfs::core::TensorShape source_shape;
            lfs::core::DataType source_dtype =
                lfs::core::DataType::Float32;
            lfs::core::Device source_device =
                lfs::core::Device::CPU;
            cudaStream_t source_stream = nullptr;
            lfs::core::TensorSerializationDescriptor descriptor;
            std::uint64_t payload_offset = 0;
            std::uint64_t payload_bytes = 0;
        };

        class CountingStreamBuffer final
            : public std::streambuf {
        public:
            [[nodiscard]] std::uint64_t size() const noexcept {
                return high_water_;
            }

        protected:
            std::streamsize xsputn(
                const char*,
                const std::streamsize count) override {
                if (count < 0) {
                    return 0;
                }
                advance(static_cast<std::uint64_t>(count));
                return count;
            }

            int_type overflow(const int_type character) override {
                if (traits_type::eq_int_type(
                        character, traits_type::eof())) {
                    return traits_type::not_eof(character);
                }
                advance(1);
                return character;
            }

            pos_type seekoff(
                const off_type offset,
                const std::ios_base::seekdir direction,
                const std::ios_base::openmode mode) override {
                if ((mode & std::ios_base::out) == 0) {
                    return pos_type(off_type(-1));
                }
                std::int64_t base = 0;
                if (direction == std::ios_base::beg) {
                    base = 0;
                } else if (direction == std::ios_base::cur) {
                    base = static_cast<std::int64_t>(cursor_);
                } else if (direction == std::ios_base::end) {
                    base = static_cast<std::int64_t>(high_water_);
                }
                if (offset < -base) {
                    return pos_type(off_type(-1));
                }
                const auto next =
                    static_cast<std::uint64_t>(base + offset);
                cursor_ = next;
                high_water_ =
                    std::max(high_water_, cursor_);
                return pos_type(
                    static_cast<off_type>(cursor_));
            }

            pos_type seekpos(
                const pos_type position,
                const std::ios_base::openmode mode) override {
                return seekoff(
                    static_cast<off_type>(position),
                    std::ios_base::beg, mode);
            }

        private:
            void advance(const std::uint64_t bytes) {
                if (bytes >
                    std::numeric_limits<std::uint64_t>::max() -
                        cursor_) {
                    throw std::overflow_error(
                        "Checkpoint byte count overflows");
                }
                cursor_ += bytes;
                high_water_ =
                    std::max(high_water_, cursor_);
            }

            std::uint64_t cursor_ = 0;
            std::uint64_t high_water_ = 0;
        };

        class CountingTensorSink final
            : public lfs::core::TensorSerializationSink {
        public:
            explicit CountingTensorSink(
                std::vector<TensorLayoutWitness>& witnesses)
                : witnesses_(witnesses) {}

            void write_tensor_payload(
                std::ostream& destination,
                const lfs::core::Tensor& source,
                const lfs::core::TensorSerializationDescriptor&
                    descriptor) override {
                validate_tensor_source(source, descriptor);
                const auto position = destination.tellp();
                if (position == std::streampos(-1)) {
                    throw std::runtime_error(
                        "Cannot locate serialized tensor payload");
                }
                const auto bytes = descriptor.payload_bytes();
                witnesses_.push_back(TensorLayoutWitness{
                    .source_pointer = source.data_ptr(),
                    .source_shape = source.shape(),
                    .source_dtype = source.dtype(),
                    .source_device = source.device(),
                    .source_stream = source.stream(),
                    .descriptor = descriptor,
                    .payload_offset =
                        static_cast<std::uint64_t>(
                            static_cast<std::streamoff>(
                                position)),
                    .payload_bytes = bytes,
                });
                if (source.device() ==
                    lfs::core::Device::CUDA) {
                    if (bytes >
                        std::numeric_limits<std::uint64_t>::max() -
                            device_bytes_) {
                        throw std::overflow_error(
                            "Device snapshot byte count overflows");
                    }
                    device_bytes_ += bytes;
                }
                destination.seekp(
                    static_cast<std::streamoff>(bytes),
                    std::ios_base::cur);
            }

            [[nodiscard]] std::uint64_t
            device_bytes() const noexcept {
                return device_bytes_;
            }

            static void validate_tensor_source(
                const lfs::core::Tensor& source,
                const lfs::core::TensorSerializationDescriptor&
                    descriptor) {
                if (!source.is_valid()) {
                    throw std::runtime_error(
                        "Snapshot tensor source is invalid");
                }
                if (!source.is_contiguous()) {
                    throw std::runtime_error(
                        "Persistent snapshot tensors must be contiguous");
                }
                if (descriptor.encoding ==
                    lfs::core::TensorPayloadEncoding::
                        NativeContiguous) {
                    if (source.shape() !=
                            descriptor.serialized_shape ||
                        source.dtype() != descriptor.dtype ||
                        source.bytes() !=
                            descriptor.payload_bytes()) {
                        throw std::runtime_error(
                            "Native snapshot tensor dimensions do not match serialization");
                    }
                    return;
                }
                if (descriptor.encoding !=
                        lfs::core::TensorPayloadEncoding::
                            SwizzledShToCanonical ||
                    source.device() !=
                        lfs::core::Device::CUDA ||
                    source.dtype() !=
                        lfs::core::DataType::Float32 ||
                    source.ndim() != 1 ||
                    descriptor.serialized_shape.rank() != 3 ||
                    descriptor.serialized_shape[0] !=
                        descriptor.sh_primitives ||
                    descriptor.serialized_shape[1] !=
                        descriptor.sh_coefficients_rest ||
                    descriptor.serialized_shape[2] !=
                        lfs::core::kShChannels ||
                    source.numel() !=
                        lfs::core::sh_swizzled_float_count(
                            descriptor.sh_primitives,
                            descriptor
                                .sh_layout_coefficients_rest)) {
                    throw std::runtime_error(
                        "Swizzled SH snapshot dimensions are inconsistent");
                }
            }

        private:
            std::vector<TensorLayoutWitness>& witnesses_;
            std::uint64_t device_bytes_ = 0;
        };

        struct PieceStamp {
            lfs::core::Uuid snapshot_uuid;
            std::uint64_t offset = 0;
            std::uint64_t bytes = 0;
            bool tensor = false;
        };

        std::mutex calibration_mutex;
        double process_pinned_d2h_bytes_per_second = 0.0;

        double percentile_95(std::vector<double> values) {
            if (values.empty()) {
                return 0.0;
            }
            std::ranges::sort(values);
            const double position =
                0.95 * static_cast<double>(
                           values.size() - 1);
            const auto lower =
                static_cast<std::size_t>(position);
            const auto upper =
                std::min(lower + 1, values.size() - 1);
            const double fraction =
                position - static_cast<double>(lower);
            return values[lower] * (1.0 - fraction) +
                   values[upper] * fraction;
        }

    } // namespace

    struct PreparedTrainingSnapshot::Impl {
        lfs::core::Uuid snapshot_uuid;
        int planned_iteration = 0;
        std::uint64_t baseline_rss_bytes = 0;
        std::uint64_t checkpoint_bytes = 0;
        std::uint64_t device_snapshot_bytes = 0;
        std::vector<TensorLayoutWitness> layout;
        std::shared_ptr<std::vector<std::byte>> staging;
        TrainingSnapshotPauseMetrics metrics;
    };

    struct PendingTrainingSnapshot::Impl {
        mutable std::mutex mutex;
        std::condition_variable drained_condition;
        std::shared_ptr<std::vector<std::byte>> staging;
        TrainingSnapshotPauseMetrics metrics;
        std::vector<PieceStamp> stamps;
        std::size_t outstanding_drains = 0;
        bool issuing_complete = false;
        bool drained = false;
        bool completion_recorded = false;
        std::string error;
        Clock::time_point pause_end;
    };

    struct TrainingSnapshotService::Impl {
        struct DrainSegment {
            std::size_t pinned_offset = 0;
            std::uint64_t destination_offset = 0;
            std::size_t bytes = 0;
        };

        struct DrainTask {
            std::shared_ptr<PendingTrainingSnapshot::Impl>
                capture;
            std::vector<DrainSegment> segments;
        };

        struct RingSlot {
            void* pinned = nullptr;
            cudaEvent_t d2h_complete = nullptr;
            bool busy = false;
            std::optional<DrainTask> task;
        };

        explicit Impl(TrainingSnapshotServiceConfig value)
            : config(std::move(value)) {
            if (config.ring_slots == 0 ||
                config.band_bytes == 0 ||
                config.calibration_bytes == 0 ||
                config.calibration_iterations <= 0 ||
                config.ring_slots >
                    std::numeric_limits<std::uint64_t>::max() /
                        config.band_bytes ||
                config.ring_slots * config.band_bytes >
                    MAX_PINNED_RING_BYTES) {
                throw std::invalid_argument(
                    "Snapshot ring must be non-zero and no larger than 512 MiB");
            }
        }

        ~Impl() {
            shutdown();
        }

        void initialize_resources(
            const std::vector<TensorLayoutWitness>& layout) {
            if (!d2h_stream) {
                require_cuda(
                    cudaStreamCreateWithFlags(
                        &d2h_stream,
                        cudaStreamNonBlocking),
                    "create snapshot D2H stream");
            }
            calibrate_once(layout);
            if (slots.empty()) {
                slots.resize(config.ring_slots);
                for (auto& slot : slots) {
                    require_cuda(
                        cudaHostAlloc(
                            &slot.pinned,
                            config.band_bytes,
                            cudaHostAllocPortable),
                        "allocate snapshot pinned ring");
                    std::memset(
                        slot.pinned, 0,
                        config.band_bytes);
                    require_cuda(
                        cudaEventCreateWithFlags(
                            &slot.d2h_complete,
                            cudaEventDisableTiming),
                        "create snapshot band event");
                }
                drain_thread = std::jthread(
                    [this](std::stop_token stop) {
                        drain_loop(stop);
                    });
            }
            const bool needs_sh_scratch =
                std::ranges::any_of(
                    layout,
                    [](const TensorLayoutWitness& witness) {
                        return witness.descriptor.encoding ==
                               lfs::core::
                                   TensorPayloadEncoding::
                                       SwizzledShToCanonical;
                    });
            if (needs_sh_scratch && !sh_scratch) {
                require_cuda(
                    cudaMalloc(
                        &sh_scratch,
                        config.band_bytes),
                    "allocate bounded SH snapshot scratch");
            }
        }

        void calibrate_once(
            const std::vector<TensorLayoutWitness>& layout) {
            std::scoped_lock lock(calibration_mutex);
            if (process_pinned_d2h_bytes_per_second > 0.0) {
                measured_bandwidth =
                    process_pinned_d2h_bytes_per_second;
                return;
            }
            const auto source = std::ranges::max_element(
                layout, std::less{},
                [](const TensorLayoutWitness& witness) {
                    return witness.source_device ==
                                       lfs::core::Device::CUDA &&
                                   witness.source_pointer &&
                                   witness.payload_bytes > 0
                               ? source_raw_bytes(witness)
                               : std::uint64_t{0};
                });
            if (source == layout.end() ||
                source->source_device !=
                    lfs::core::Device::CUDA ||
                !source->source_pointer ||
                source->payload_bytes == 0) {
                measured_bandwidth =
                    std::numeric_limits<double>::infinity();
                return;
            }
            if (source->source_stream) {
                require_cuda(
                    cudaStreamSynchronize(
                        source->source_stream),
                    "sync calibration source stream");
            }
            const auto bytes = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    config.calibration_bytes,
                    source->source_device ==
                            lfs::core::Device::CUDA
                        ? source_raw_bytes(*source)
                        : 0));
            if (bytes == 0) {
                throw std::runtime_error(
                    "Snapshot calibration found no CUDA source bytes");
            }

            void* pinned = nullptr;
            cudaEvent_t begin = nullptr;
            cudaEvent_t end = nullptr;
            try {
                require_cuda(
                    cudaHostAlloc(
                        &pinned, bytes,
                        cudaHostAllocPortable),
                    "allocate D2H calibration pin");
                require_cuda(
                    cudaEventCreate(&begin),
                    "create D2H calibration begin event");
                require_cuda(
                    cudaEventCreate(&end),
                    "create D2H calibration end event");
                require_cuda(
                    cudaMemcpyAsync(
                        pinned, source->source_pointer,
                        bytes, cudaMemcpyDeviceToHost,
                        d2h_stream),
                    "warm D2H calibration");
                require_cuda(
                    cudaStreamSynchronize(d2h_stream),
                    "finish D2H calibration warmup");
                require_cuda(
                    cudaEventRecord(begin, d2h_stream),
                    "record D2H calibration begin");
                for (int iteration = 0;
                     iteration <
                     config.calibration_iterations;
                     ++iteration) {
                    require_cuda(
                        cudaMemcpyAsync(
                            pinned,
                            source->source_pointer,
                            bytes,
                            cudaMemcpyDeviceToHost,
                            d2h_stream),
                        "measure raw pinned D2H");
                }
                require_cuda(
                    cudaEventRecord(end, d2h_stream),
                    "record D2H calibration end");
                require_cuda(
                    cudaEventSynchronize(end),
                    "wait D2H calibration");
                float elapsed_ms = 0.0f;
                require_cuda(
                    cudaEventElapsedTime(
                        &elapsed_ms, begin, end),
                    "read D2H calibration time");
                if (!(elapsed_ms > 0.0f)) {
                    throw std::runtime_error(
                        "Pinned D2H calibration duration is zero");
                }
                process_pinned_d2h_bytes_per_second =
                    static_cast<double>(bytes) *
                    config.calibration_iterations /
                    (static_cast<double>(elapsed_ms) /
                     1000.0);
                measured_bandwidth =
                    process_pinned_d2h_bytes_per_second;
            } catch (...) {
                if (end)
                    cudaEventDestroy(end);
                if (begin)
                    cudaEventDestroy(begin);
                if (pinned)
                    cudaFreeHost(pinned);
                throw;
            }
            require_cuda(
                cudaEventDestroy(end),
                "destroy D2H calibration end event");
            require_cuda(
                cudaEventDestroy(begin),
                "destroy D2H calibration begin event");
            require_cuda(
                cudaFreeHost(pinned),
                "free D2H calibration pin");
        }

        static std::uint64_t source_raw_bytes(
            const TensorLayoutWitness& witness) {
            if (witness.descriptor.encoding ==
                lfs::core::TensorPayloadEncoding::
                    NativeContiguous) {
                return witness.payload_bytes;
            }
            return static_cast<std::uint64_t>(
                       lfs::core::sh_swizzled_float_count(
                           witness.descriptor.sh_primitives,
                           witness.descriptor
                               .sh_layout_coefficients_rest)) *
                   sizeof(float);
        }

        [[nodiscard]] std::size_t
        reserve_ring_slot() {
            std::unique_lock lock(ring_mutex);
            if (slots.empty()) {
                throw std::runtime_error(
                    "Snapshot ring is not initialized");
            }
            const auto slot_index =
                next_slot++ % slots.size();
            ring_condition.wait(
                lock, [&] {
                    return !slots[slot_index].busy;
                });
            auto& slot = slots[slot_index];
            if (slot.task) {
                throw std::runtime_error(
                    "Free snapshot ring slot retained a drain task");
            }
            slot.busy = true;
            return slot_index;
        }

        void issue_native_to_slot(
            const std::size_t slot_index,
            const std::size_t pinned_offset,
            const void* source,
            const std::size_t bytes) {
            validate_slot_range(
                slot_index, pinned_offset, bytes);
            require_cuda(
                cudaMemcpyAsync(
                    static_cast<std::byte*>(
                        slots[slot_index].pinned) +
                        pinned_offset,
                    source, bytes,
                    cudaMemcpyDeviceToHost,
                    d2h_stream),
                "issue packed snapshot D2H");
        }

        void issue_sh_to_slot(
            const std::size_t slot_index,
            const std::size_t pinned_offset,
            const TensorLayoutWitness& witness,
            const std::uint64_t tensor_byte_offset,
            const std::size_t bytes) {
            validate_slot_range(
                slot_index, pinned_offset, bytes);
            if (!sh_scratch ||
                tensor_byte_offset % sizeof(float) != 0 ||
                bytes % sizeof(float) != 0) {
                throw std::runtime_error(
                    "Bounded SH snapshot band is misaligned");
            }
            lfs::core::
                undo_reorder_sh_range_from_swizzled(
                    static_cast<const float*>(
                        witness.source_pointer),
                    static_cast<float*>(sh_scratch),
                    tensor_byte_offset / sizeof(float),
                    bytes / sizeof(float),
                    witness.descriptor.sh_primitives,
                    witness.descriptor.sh_coefficients_rest,
                    witness.descriptor
                        .sh_layout_coefficients_rest,
                    d2h_stream);
            require_cuda(
                cudaMemcpyAsync(
                    static_cast<std::byte*>(
                        slots[slot_index].pinned) +
                        pinned_offset,
                    sh_scratch, bytes,
                    cudaMemcpyDeviceToHost,
                    d2h_stream),
                "issue packed SH snapshot D2H");
        }

        [[nodiscard]] cudaEvent_t submit_ring_slot(
            const std::size_t slot_index,
            const std::shared_ptr<
                PendingTrainingSnapshot::Impl>& capture,
            std::vector<DrainSegment> segments) {
            if (slot_index >= slots.size() ||
                segments.empty()) {
                throw std::invalid_argument(
                    "Packed snapshot band is empty or invalid");
            }
            std::size_t packed_bytes = 0;
            for (const auto& segment : segments) {
                if (segment.pinned_offset !=
                        packed_bytes ||
                    segment.destination_offset >
                        capture->staging->size() ||
                    segment.bytes >
                        capture->staging->size() -
                            segment
                                .destination_offset) {
                    throw std::invalid_argument(
                        "Packed snapshot band segments are inconsistent");
                }
                validate_slot_range(
                    slot_index,
                    segment.pinned_offset,
                    segment.bytes);
                packed_bytes += segment.bytes;
            }
            auto& slot = slots[slot_index];
            require_cuda(
                cudaEventRecord(
                    slot.d2h_complete, d2h_stream),
                "record packed snapshot D2H band");
            {
                std::scoped_lock lock(ring_mutex);
                if (!slot.busy || slot.task) {
                    throw std::runtime_error(
                        "Packed snapshot ring slot ownership changed");
                }
                slot.task = DrainTask{
                    .capture = capture,
                    .segments = std::move(segments),
                };
                {
                    std::scoped_lock capture_lock(
                        capture->mutex);
                    ++capture->outstanding_drains;
                }
                drain_queue.push_back(slot_index);
            }
            const auto event = slot.d2h_complete;
            ring_condition.notify_all();
            return event;
        }

        void cancel_ring_slot(
            const std::size_t slot_index) noexcept {
            if (slot_index >= slots.size()) {
                return;
            }
            LFS_CUDA_LOG_TEARDOWN(
                cudaStreamSynchronize(d2h_stream),
                d2h_stream,
                "cancel unsubmitted training snapshot ring slot");
            {
                std::scoped_lock lock(ring_mutex);
                auto& slot = slots[slot_index];
                if (!slot.task) {
                    slot.busy = false;
                }
            }
            ring_condition.notify_all();
        }

        void validate_slot_range(
            const std::size_t slot_index,
            const std::size_t pinned_offset,
            const std::size_t bytes) const {
            if (slot_index >= slots.size() ||
                bytes == 0 ||
                pinned_offset > config.band_bytes ||
                bytes > config.band_bytes -
                            pinned_offset) {
                throw std::invalid_argument(
                    "Packed snapshot D2H range is invalid");
            }
        }

        void drain_loop(const std::stop_token stop) {
            while (true) {
                std::size_t slot_index = 0;
                {
                    std::unique_lock lock(ring_mutex);
                    ring_condition.wait(
                        lock, [&] {
                            return !drain_queue.empty() ||
                                   stop.stop_requested();
                        });
                    if (drain_queue.empty() &&
                        stop.stop_requested()) {
                        return;
                    }
                    slot_index = drain_queue.front();
                    drain_queue.pop_front();
                }

                auto& slot = slots[slot_index];
                auto task = std::move(*slot.task);
                std::string error;
                const auto event_status =
                    cudaEventSynchronize(
                        slot.d2h_complete);
                if (event_status == cudaSuccess) {
                    for (const auto& segment :
                         task.segments) {
                        non_temporal_copy(
                            task.capture
                                    ->staging->data() +
                                segment
                                    .destination_offset,
                            static_cast<
                                const std::byte*>(
                                slot.pinned) +
                                segment.pinned_offset,
                            segment.bytes);
                    }
                } else {
                    error = std::format(
                        "snapshot drain event: {}",
                        cudaGetErrorString(event_status));
                }

                bool finalize = false;
                {
                    std::scoped_lock lock(
                        task.capture->mutex);
                    if (!error.empty() &&
                        task.capture->error.empty()) {
                        task.capture->error =
                            std::move(error);
                    }
                    if (task.capture
                            ->outstanding_drains == 0) {
                        task.capture->error =
                            "Snapshot drain accounting underflow";
                    } else {
                        --task.capture
                              ->outstanding_drains;
                    }
                    finalize =
                        task.capture->issuing_complete &&
                        task.capture
                                ->outstanding_drains ==
                            0;
                }
                {
                    std::scoped_lock lock(ring_mutex);
                    slot.task.reset();
                    slot.busy = false;
                }
                ring_condition.notify_all();
                if (finalize) {
                    finalize_capture(task.capture);
                }
            }
        }

        void mark_issuing_complete(
            const std::shared_ptr<
                PendingTrainingSnapshot::Impl>& capture) {
            bool finalize = false;
            {
                std::scoped_lock lock(capture->mutex);
                capture->issuing_complete = true;
                finalize =
                    capture->outstanding_drains == 0;
            }
            if (finalize) {
                finalize_capture(capture);
            }
        }

        void finalize_capture(
            const std::shared_ptr<
                PendingTrainingSnapshot::Impl>& capture) {
            TrainingSnapshotPauseMetrics completed;
            bool record = false;
            {
                std::scoped_lock lock(capture->mutex);
                if (capture->completion_recorded) {
                    return;
                }
                capture->completion_recorded = true;
                capture->metrics.final_drain_ms =
                    Milliseconds(
                        Clock::now() -
                        capture->pause_end)
                        .count();
                bool consistent =
                    capture->error.empty() &&
                    !capture->stamps.empty();
                std::size_t tensor_count = 0;
                std::size_t cpu_count = 0;
                for (const auto& stamp :
                     capture->stamps) {
                    consistent =
                        consistent &&
                        stamp.snapshot_uuid ==
                            capture->metrics
                                .snapshot_uuid &&
                        stamp.offset <=
                            capture->staging->size() &&
                        stamp.bytes <=
                            capture->staging->size() -
                                stamp.offset;
                    tensor_count += stamp.tensor ? 1 : 0;
                    cpu_count += stamp.tensor ? 0 : 1;
                }
                capture->metrics.tensor_piece_count =
                    tensor_count;
                capture->metrics.cpu_piece_count =
                    cpu_count;
                capture->metrics.consistency_proven =
                    consistent && tensor_count > 0 &&
                    cpu_count > 0;
                if (!capture->metrics
                         .consistency_proven &&
                    capture->error.empty()) {
                    capture->error =
                        "Snapshot UUID consistency proof failed";
                }
                capture->drained = true;
                completed = capture->metrics;
                record = capture->error.empty();
            }
            capture->drained_condition.notify_all();
            {
                std::scoped_lock lock(metrics_mutex);
                active_capture = false;
                if (record) {
                    ++aggregate.completed_snapshots;
                    aggregate.last = completed;
                    pause_samples.push_back(
                        completed.pause_ms);
                    aggregate.pause_p95_ms =
                        percentile_95(
                            pause_samples);
                }
            }
        }

        void shutdown() {
            if (drain_thread.joinable()) {
                drain_thread.request_stop();
                ring_condition.notify_all();
                drain_thread.join();
            }
            if (sh_scratch) {
                LFS_CUDA_LOG_TEARDOWN(
                    cudaFree(sh_scratch), d2h_stream,
                    "training snapshot SH scratch teardown");
                sh_scratch = nullptr;
            }
            for (auto& slot : slots) {
                if (slot.d2h_complete) {
                    LFS_CUDA_LOG_TEARDOWN(
                        cudaEventDestroy(
                            slot.d2h_complete),
                        d2h_stream,
                        "training snapshot band event teardown");
                }
                if (slot.pinned) {
                    LFS_CUDA_LOG_TEARDOWN(
                        cudaFreeHost(slot.pinned),
                        d2h_stream,
                        "training snapshot pinned band teardown");
                }
            }
            slots.clear();
            if (d2h_stream) {
                LFS_CUDA_LOG_TEARDOWN(
                    cudaStreamDestroy(d2h_stream),
                    d2h_stream,
                    "training snapshot D2H stream teardown");
                d2h_stream = nullptr;
            }
        }

        TrainingSnapshotServiceConfig config;
        cudaStream_t d2h_stream = nullptr;
        void* sh_scratch = nullptr;
        std::vector<RingSlot> slots;
        std::size_t next_slot = 0;
        std::mutex ring_mutex;
        std::condition_variable ring_condition;
        std::deque<std::size_t> drain_queue;
        std::jthread drain_thread;
        double measured_bandwidth = 0.0;

        mutable std::mutex metrics_mutex;
        bool active_capture = false;
        TrainingSnapshotServiceMetrics aggregate;
        std::vector<double> pause_samples;
    };

    namespace {

        class MemoryStreamBuffer final
            : public std::streambuf {
        public:
            MemoryStreamBuffer(
                std::shared_ptr<
                    PendingTrainingSnapshot::Impl>
                    capture,
                const lfs::core::Uuid& snapshot_uuid)
                : capture_(std::move(capture)),
                  snapshot_uuid_(snapshot_uuid) {}

        protected:
            std::streamsize xsputn(
                const char* source,
                const std::streamsize count) override {
                if (count < 0 ||
                    static_cast<std::uint64_t>(count) >
                        capture_->staging->size() -
                            cursor_) {
                    return 0;
                }
                if (count > 0) {
                    std::memcpy(
                        capture_->staging->data() +
                            cursor_,
                        source,
                        static_cast<std::size_t>(count));
                    add_stamp(
                        cursor_,
                        static_cast<std::uint64_t>(
                            count),
                        false);
                    cursor_ +=
                        static_cast<std::uint64_t>(
                            count);
                }
                return count;
            }

            int_type overflow(const int_type character) override {
                if (traits_type::eq_int_type(
                        character, traits_type::eof())) {
                    return traits_type::not_eof(character);
                }
                const char byte =
                    traits_type::to_char_type(character);
                return xsputn(&byte, 1) == 1
                           ? character
                           : traits_type::eof();
            }

            pos_type seekoff(
                const off_type offset,
                const std::ios_base::seekdir direction,
                const std::ios_base::openmode mode) override {
                if ((mode & std::ios_base::out) == 0) {
                    return pos_type(off_type(-1));
                }
                std::int64_t base = 0;
                if (direction == std::ios_base::beg) {
                    base = 0;
                } else if (direction == std::ios_base::cur) {
                    base = static_cast<std::int64_t>(
                        cursor_);
                } else if (direction == std::ios_base::end) {
                    base = static_cast<std::int64_t>(
                        capture_->staging->size());
                }
                if (offset < -base) {
                    return pos_type(off_type(-1));
                }
                const auto next =
                    static_cast<std::uint64_t>(
                        base + offset);
                if (next >
                    capture_->staging->size()) {
                    return pos_type(off_type(-1));
                }
                cursor_ = next;
                return pos_type(
                    static_cast<off_type>(cursor_));
            }

            pos_type seekpos(
                const pos_type position,
                const std::ios_base::openmode mode) override {
                return seekoff(
                    static_cast<off_type>(position),
                    std::ios_base::beg, mode);
            }

        private:
            void add_stamp(
                const std::uint64_t offset,
                const std::uint64_t bytes,
                const bool tensor) {
                std::scoped_lock lock(capture_->mutex);
                capture_->stamps.push_back(PieceStamp{
                    .snapshot_uuid = snapshot_uuid_,
                    .offset = offset,
                    .bytes = bytes,
                    .tensor = tensor,
                });
            }

            std::shared_ptr<
                PendingTrainingSnapshot::Impl>
                capture_;
            lfs::core::Uuid snapshot_uuid_;
            std::uint64_t cursor_ = 0;
        };

        bool descriptors_equal(
            const lfs::core::TensorSerializationDescriptor& lhs,
            const lfs::core::TensorSerializationDescriptor& rhs) {
            return lhs.serialized_shape ==
                       rhs.serialized_shape &&
                   lhs.dtype == rhs.dtype &&
                   lhs.serialized_device ==
                       rhs.serialized_device &&
                   lhs.encoding == rhs.encoding &&
                   lhs.sh_primitives ==
                       rhs.sh_primitives &&
                   lhs.sh_coefficients_rest ==
                       rhs.sh_coefficients_rest &&
                   lhs.sh_layout_coefficients_rest ==
                       rhs.sh_layout_coefficients_rest;
        }

        class CaptureTensorSink final
            : public lfs::core::TensorSerializationSink {
        public:
            CaptureTensorSink(
                TrainingSnapshotService::Impl& service,
                const std::vector<TensorLayoutWitness>& layout,
                std::shared_ptr<
                    PendingTrainingSnapshot::Impl>
                    capture,
                const lfs::core::Uuid& snapshot_uuid)
                : service_(service),
                  layout_(layout),
                  capture_(std::move(capture)),
                  snapshot_uuid_(snapshot_uuid) {}

            ~CaptureTensorSink() override {
                if (active_slot_) {
                    service_.cancel_ring_slot(
                        *active_slot_);
                }
            }

            void write_tensor_payload(
                std::ostream& destination,
                const lfs::core::Tensor& source,
                const lfs::core::TensorSerializationDescriptor&
                    descriptor) override {
                if (index_ >= layout_.size()) {
                    throw std::runtime_error(
                        "Snapshot layout gained a tensor");
                }
                CountingTensorSink::validate_tensor_source(
                    source, descriptor);
                const auto position = destination.tellp();
                if (position == std::streampos(-1)) {
                    throw std::runtime_error(
                        "Cannot locate captured tensor payload");
                }
                const auto& expected = layout_[index_++];
                const auto offset =
                    static_cast<std::uint64_t>(
                        static_cast<std::streamoff>(
                            position));
                const auto bytes =
                    descriptor.payload_bytes();
                if (source.data_ptr() !=
                        expected.source_pointer ||
                    source.shape() !=
                        expected.source_shape ||
                    source.dtype() !=
                        expected.source_dtype ||
                    source.device() !=
                        expected.source_device ||
                    !descriptors_equal(
                        descriptor,
                        expected.descriptor) ||
                    offset != expected.payload_offset ||
                    bytes != expected.payload_bytes) {
                    throw std::runtime_error(
                        "Snapshot layout changed after preparation; request must be coalesced and replanned");
                }

                if (source.device() ==
                    lfs::core::Device::CPU) {
                    destination.write(
                        static_cast<const char*>(
                            source.data_ptr()),
                        static_cast<std::streamsize>(
                            bytes));
                } else {
                    append_device_tensor(
                        source, descriptor, expected,
                        offset, bytes);
                    destination.seekp(
                        static_cast<std::streamoff>(
                            bytes),
                        std::ios_base::cur);
                }
                {
                    std::scoped_lock lock(capture_->mutex);
                    capture_->stamps.push_back(PieceStamp{
                        .snapshot_uuid = snapshot_uuid_,
                        .offset = offset,
                        .bytes = bytes,
                        .tensor = true,
                    });
                }
            }

            void finish() {
                flush_active_slot();
            }

            [[nodiscard]] bool complete() const noexcept {
                return index_ == layout_.size();
            }

            [[nodiscard]] cudaEvent_t
            last_event() const noexcept {
                return last_event_;
            }

        private:
            void append_device_tensor(
                const lfs::core::Tensor& source,
                const lfs::core::
                    TensorSerializationDescriptor&
                        descriptor,
                const TensorLayoutWitness& witness,
                const std::uint64_t destination_offset,
                const std::uint64_t bytes) {
                std::uint64_t tensor_offset = 0;
                while (tensor_offset < bytes) {
                    if (!active_slot_) {
                        active_slot_ =
                            service_
                                .reserve_ring_slot();
                        active_slot_bytes_ = 0;
                        active_segments_.clear();
                    }
                    const auto available =
                        service_.config.band_bytes -
                        active_slot_bytes_;
                    auto count =
                        static_cast<std::size_t>(
                            std::min<std::uint64_t>(
                                available,
                                bytes - tensor_offset));
                    if (descriptor.encoding ==
                        lfs::core::
                            TensorPayloadEncoding::
                                SwizzledShToCanonical) {
                        count -=
                            count % sizeof(float);
                    }
                    if (count == 0) {
                        flush_active_slot();
                        continue;
                    }

                    if (descriptor.encoding ==
                        lfs::core::
                            TensorPayloadEncoding::
                                NativeContiguous) {
                        service_.issue_native_to_slot(
                            *active_slot_,
                            active_slot_bytes_,
                            static_cast<const std::byte*>(
                                source.data_ptr()) +
                                tensor_offset,
                            count);
                    } else {
                        service_.issue_sh_to_slot(
                            *active_slot_,
                            active_slot_bytes_,
                            witness, tensor_offset,
                            count);
                    }
                    active_segments_.push_back(
                        TrainingSnapshotService::Impl::
                            DrainSegment{
                                .pinned_offset =
                                    active_slot_bytes_,
                                .destination_offset =
                                    destination_offset +
                                    tensor_offset,
                                .bytes = count,
                            });
                    active_slot_bytes_ += count;
                    tensor_offset += count;
                    if (active_slot_bytes_ ==
                        service_.config.band_bytes) {
                        flush_active_slot();
                    }
                }
            }

            void flush_active_slot() {
                if (!active_slot_) {
                    return;
                }
                if (active_slot_bytes_ == 0 ||
                    active_segments_.empty()) {
                    service_.cancel_ring_slot(
                        *active_slot_);
                    active_slot_.reset();
                    throw std::runtime_error(
                        "Packed snapshot band made no progress");
                }
                last_event_ =
                    service_.submit_ring_slot(
                        *active_slot_, capture_,
                        std::move(active_segments_));
                active_slot_.reset();
                active_slot_bytes_ = 0;
                active_segments_.clear();
            }

            TrainingSnapshotService::Impl& service_;
            const std::vector<TensorLayoutWitness>& layout_;
            std::shared_ptr<
                PendingTrainingSnapshot::Impl>
                capture_;
            lfs::core::Uuid snapshot_uuid_;
            std::size_t index_ = 0;
            cudaEvent_t last_event_ = nullptr;
            std::optional<std::size_t> active_slot_;
            std::size_t active_slot_bytes_ = 0;
            std::vector<
                TrainingSnapshotService::Impl::
                    DrainSegment>
                active_segments_;
        };

    } // namespace

    PreparedTrainingSnapshot::PreparedTrainingSnapshot(
        std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}
    PreparedTrainingSnapshot::PreparedTrainingSnapshot(
        PreparedTrainingSnapshot&&) noexcept = default;
    PreparedTrainingSnapshot&
    PreparedTrainingSnapshot::operator=(
        PreparedTrainingSnapshot&&) noexcept = default;
    PreparedTrainingSnapshot::~PreparedTrainingSnapshot() =
        default;

    const lfs::core::Uuid&
    PreparedTrainingSnapshot::snapshot_uuid() const noexcept {
        return impl_->snapshot_uuid;
    }

    std::uint64_t
    PreparedTrainingSnapshot::checkpoint_bytes() const noexcept {
        return impl_->checkpoint_bytes;
    }

    PendingTrainingSnapshot::PendingTrainingSnapshot(
        std::shared_ptr<Impl> impl)
        : impl_(std::move(impl)) {}
    PendingTrainingSnapshot::PendingTrainingSnapshot(
        PendingTrainingSnapshot&&) noexcept = default;
    PendingTrainingSnapshot&
    PendingTrainingSnapshot::operator=(
        PendingTrainingSnapshot&&) noexcept = default;
    PendingTrainingSnapshot::~PendingTrainingSnapshot() =
        default;

    bool PendingTrainingSnapshot::ready() const {
        std::scoped_lock lock(impl_->mutex);
        return impl_->drained;
    }

    TrainingSnapshotPauseMetrics
    PendingTrainingSnapshot::pause_metrics() const {
        std::scoped_lock lock(impl_->mutex);
        return impl_->metrics;
    }

    lfs::Result<CapturedTrainingSnapshot>
    PendingTrainingSnapshot::wait() {
        std::unique_lock lock(impl_->mutex);
        impl_->drained_condition.wait(
            lock, [&] { return impl_->drained; });
        if (!impl_->error.empty()) {
            return snapshot_error(
                lfs::ErrorCode::DataLoss,
                impl_->error,
                LFS_SOURCE_SITE_CURRENT());
        }
        return CapturedTrainingSnapshot{
            .snapshot_uuid =
                impl_->metrics.snapshot_uuid,
            .iteration = impl_->metrics.iteration,
            .checkpoint_bytes = impl_->staging,
            .metrics = impl_->metrics,
        };
    }

    TrainingSnapshotService::TrainingSnapshotService(
        TrainingSnapshotServiceConfig config)
        : impl_(std::make_unique<Impl>(
              std::move(config))) {}

    TrainingSnapshotService::~TrainingSnapshotService() =
        default;

    lfs::Result<PreparedTrainingSnapshot>
    TrainingSnapshotService::prepare(
        const TrainingSnapshotCaptureRequest& request) {
        try {
            const auto begin = Clock::now();
            auto prepared =
                std::make_unique<
                    PreparedTrainingSnapshot::Impl>();
            prepared->snapshot_uuid =
                request.snapshot_uuid.is_nil()
                    ? lfs::core::generate_uuid_v4()
                    : request.snapshot_uuid;
            prepared->planned_iteration =
                request.iteration;

            CountingStreamBuffer buffer;
            std::ostream destination(&buffer);
            CountingTensorSink sink(prepared->layout);
            lfs::core::TensorSerializationSinkScope
                scope(sink);
            auto serialized = serialize_checkpoint(
                destination, request.iteration,
                request.strategy, request.params,
                request.bilateral_grid, request.ppisp,
                request.ppisp_controller_pool,
                request.sparsity_optimizer);
            if (!serialized) {
                return std::move(serialized)
                    .error()
                    .with_context(
                        "prepare training snapshot checkpoint layout",
                        LFS_SOURCE_SITE_CURRENT());
            }
            if (!destination ||
                buffer.size() != serialized->bytes ||
                buffer.size() == 0 ||
                buffer.size() >
                    lfs::core::
                        MAX_CHECKPOINT_FILE_BYTES) {
                return snapshot_error(
                    lfs::ErrorCode::DataLoss,
                    "Prepared checkpoint size is invalid",
                    LFS_SOURCE_SITE_CURRENT());
            }
            prepared->checkpoint_bytes =
                buffer.size();
            prepared->device_snapshot_bytes =
                sink.device_bytes();

            prepared->baseline_rss_bytes =
                read_rss_bytes();
            impl_->initialize_resources(
                prepared->layout);

            if (prepared->checkpoint_bytes >
                std::numeric_limits<std::size_t>::max()) {
                return snapshot_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "Checkpoint staging exceeds address space",
                    LFS_SOURCE_SITE_CURRENT());
            }
            const auto host_memory =
                read_host_memory_info();
            const auto reserve_bytes =
                std::max<std::uint64_t>(
                    MIN_HOST_MEMORY_RESERVE_BYTES,
                    host_memory.total_bytes / 5);
            if (prepared->checkpoint_bytes >
                std::numeric_limits<std::uint64_t>::max() -
                    reserve_bytes) {
                return snapshot_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "Snapshot host-memory requirement overflows",
                    LFS_SOURCE_SITE_CURRENT());
            }
            const auto required_host_memory =
                prepared->checkpoint_bytes + reserve_bytes;
            if (host_memory.available_bytes == 0 ||
                host_memory.available_bytes <
                    required_host_memory) {
                return snapshot_error(
                    lfs::ErrorCode::ResourceExhausted,
                    std::format(
                        "Training snapshot deferred: {} bytes available, "
                        "{} required ({} snapshot + {} reserve)",
                        host_memory.available_bytes,
                        required_host_memory,
                        prepared->checkpoint_bytes,
                        reserve_bytes),
                    LFS_SOURCE_SITE_CURRENT());
            }
            prepared->staging =
                std::make_shared<
                    std::vector<std::byte>>();
            prepared->staging->resize(
                static_cast<std::size_t>(
                    prepared->checkpoint_bytes));
            const auto rss_after = read_rss_bytes();

            prepared->metrics.snapshot_uuid =
                prepared->snapshot_uuid;
            prepared->metrics.iteration =
                request.iteration;
            prepared->metrics.checkpoint_bytes =
                prepared->checkpoint_bytes;
            prepared->metrics.device_snapshot_bytes =
                prepared->device_snapshot_bytes;
            prepared->metrics.pinned_peak_bytes =
                impl_->config.ring_slots *
                impl_->config.band_bytes;
            prepared->metrics.host_staging_bytes =
                prepared->checkpoint_bytes;
            prepared->metrics.host_rss_delta_bytes =
                rss_after >=
                        prepared->baseline_rss_bytes
                    ? rss_after -
                          prepared->baseline_rss_bytes
                    : 0;
            prepared->metrics.host_memory_available_bytes =
                host_memory.available_bytes;
            prepared->metrics.host_memory_required_bytes =
                required_host_memory;
            prepared->metrics.host_memory_preflight_passed =
                true;
            prepared->metrics.host_ram_within_gate =
                prepared->metrics.host_rss_delta_bytes <=
                prepared->checkpoint_bytes +
                    HOST_MEMORY_GATE_HEADROOM_BYTES;
            prepared->metrics.preparation_ms =
                Milliseconds(Clock::now() - begin)
                    .count();
            prepared->metrics
                .measured_pinned_d2h_bytes_per_second =
                impl_->measured_bandwidth;
            prepared->metrics.rig_gate_ms =
                std::isfinite(
                    impl_->measured_bandwidth)
                    ? static_cast<double>(
                          prepared
                              ->checkpoint_bytes) /
                          impl_->measured_bandwidth *
                          1.12 * 1000.0
                    : 0.0;
            {
                std::scoped_lock lock(
                    impl_->metrics_mutex);
                prepared->metrics
                    .cold_first_snapshot =
                    impl_->aggregate
                        .completed_snapshots ==
                    0;
            }
            return PreparedTrainingSnapshot(
                std::move(prepared));
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): normalize the exception into a typed snapshot error.
            return snapshot_error(
                lfs::ErrorCode::Internal,
                std::format(
                    "Prepare training snapshot failed: {}",
                    error.what()),
                LFS_SOURCE_SITE_CURRENT());
        }
    }

    lfs::Result<PendingTrainingSnapshot>
    TrainingSnapshotService::capture(
        PreparedTrainingSnapshot prepared,
        const TrainingSnapshotCaptureRequest& request) {
        if (!prepared.impl_) {
            return snapshot_error(
                lfs::ErrorCode::FailedPrecondition,
                "Prepared snapshot is empty",
                LFS_SOURCE_SITE_CURRENT());
        }
        if (!request.snapshot_uuid.is_nil() &&
            request.snapshot_uuid !=
                prepared.impl_->snapshot_uuid) {
            return snapshot_error(
                lfs::ErrorCode::ContractViolation,
                "Prepared snapshot UUID does not match the capture request",
                LFS_SOURCE_SITE_CURRENT());
        }
        {
            std::scoped_lock lock(impl_->metrics_mutex);
            if (impl_->active_capture) {
                return snapshot_error(
                    lfs::ErrorCode::AlreadyExists,
                    "Snapshot already in flight; newer request must coalesce",
                    LFS_SOURCE_SITE_CURRENT());
            }
            impl_->active_capture = true;
        }

        auto pending =
            std::make_shared<
                PendingTrainingSnapshot::Impl>();
        pending->staging =
            std::move(prepared.impl_->staging);
        pending->metrics =
            prepared.impl_->metrics;
        pending->metrics.iteration =
            request.iteration;
        pending->metrics.snapshot_uuid =
            prepared.impl_->snapshot_uuid;

        try {
            const auto capture_begin = Clock::now();
            const auto pause_begin =
                request.safe_point_entered_at.value_or(
                    capture_begin);
            if (pause_begin > capture_begin) {
                throw std::invalid_argument(
                    "Snapshot safe-point clock origin is in the future");
            }
            std::set<cudaStream_t> streams;
            streams.insert(impl_->d2h_stream);
            for (const auto stream :
                 request.mutating_streams) {
                if (stream) {
                    streams.insert(stream);
                }
            }
            for (const auto& witness :
                 prepared.impl_->layout) {
                if (witness.source_device ==
                        lfs::core::Device::CUDA &&
                    witness.source_stream) {
                    streams.insert(
                        witness.source_stream);
                }
            }
            for (const auto stream : streams) {
                require_cuda(
                    cudaStreamSynchronize(stream),
                    "synchronize snapshot mutating stream");
            }
            const auto sync_end = Clock::now();

            if (request.capture_additional_cpu_state) {
                auto captured =
                    request.capture_additional_cpu_state(
                        prepared.impl_->snapshot_uuid);
                if (!captured) {
                    throw std::runtime_error(
                        lfs::format_for_developer(
                            captured.error()));
                }
                std::scoped_lock lock(pending->mutex);
                pending->stamps.push_back(PieceStamp{
                    .snapshot_uuid =
                        prepared.impl_->snapshot_uuid,
                    .offset = 0,
                    .bytes = 0,
                    .tensor = false,
                });
            }
            const auto cpu_state_end = Clock::now();

            MemoryStreamBuffer memory_buffer(
                pending,
                prepared.impl_->snapshot_uuid);
            std::ostream destination(&memory_buffer);
            CaptureTensorSink sink(
                *impl_, prepared.impl_->layout,
                pending,
                prepared.impl_->snapshot_uuid);
            {
                lfs::core::
                    TensorSerializationSinkScope
                        scope(sink);
                auto serialized =
                    serialize_checkpoint(
                        destination,
                        request.iteration,
                        request.strategy,
                        request.params,
                        request.bilateral_grid,
                        request.ppisp,
                        request
                            .ppisp_controller_pool,
                        request
                            .sparsity_optimizer);
                if (!serialized) {
                    throw std::runtime_error(
                        lfs::format_for_developer(
                            serialized.error()));
                }
                if (serialized->bytes !=
                        prepared.impl_
                            ->checkpoint_bytes ||
                    !sink.complete()) {
                    throw std::runtime_error(
                        "Checkpoint layout changed after preparation");
                }
                sink.finish();
            }
            const auto serialize_end = Clock::now();
            if (const auto last_event =
                    sink.last_event()) {
                require_cuda(
                    cudaEventSynchronize(last_event),
                    "wait for last snapshot D2H");
            }
            const auto pause_end = Clock::now();
            const auto capture_rss = read_rss_bytes();
            if (capture_rss >=
                prepared.impl_->baseline_rss_bytes) {
                pending->metrics.host_rss_delta_bytes =
                    std::max(
                        pending->metrics
                            .host_rss_delta_bytes,
                        capture_rss -
                            prepared.impl_
                                ->baseline_rss_bytes);
            }
            pending->metrics.host_ram_within_gate =
                pending->metrics.host_rss_delta_bytes <=
                pending->metrics.checkpoint_bytes +
                    HOST_MEMORY_GATE_HEADROOM_BYTES;

            pending->metrics.safe_point_entry_ms =
                Milliseconds(
                    capture_begin - pause_begin)
                    .count();
            pending->metrics.stream_sync_ms =
                Milliseconds(
                    sync_end - capture_begin)
                    .count();
            pending->metrics
                .additional_cpu_state_ms =
                Milliseconds(
                    cpu_state_end - sync_end)
                    .count();
            pending->metrics
                .serialize_and_issue_ms =
                Milliseconds(
                    serialize_end - cpu_state_end)
                    .count();
            pending->metrics.last_d2h_wait_ms =
                Milliseconds(
                    pause_end - serialize_end)
                    .count();
            pending->metrics.pause_ms =
                Milliseconds(
                    pause_end - pause_begin)
                    .count();
            pending->metrics
                .pause_within_rig_gate =
                pending->metrics.pause_ms <=
                pending->metrics.rig_gate_ms;
            pending->pause_end = pause_end;

            LOG_INFO(
                "Training snapshot {} iter {}: "
                "bytes={} device_bytes={} pause={:.3f}ms "
                "prepare={:.3f}ms cold_first={} "
                "(safe_entry={:.3f} sync={:.3f} cpu_state={:.3f} "
                "serialize+issue={:.3f} "
                "last_d2h_wait={:.3f}) gate={:.3f}ms "
                "raw_pinned_D2H={:.3f}GiB/s pause={} "
                "host_delta={} host_gate={}",
                pending->metrics.snapshot_uuid
                    .to_string(),
                request.iteration,
                pending->metrics.checkpoint_bytes,
                pending->metrics
                    .device_snapshot_bytes,
                pending->metrics.pause_ms,
                pending->metrics.preparation_ms,
                pending->metrics.cold_first_snapshot,
                pending->metrics
                    .safe_point_entry_ms,
                pending->metrics.stream_sync_ms,
                pending->metrics
                    .additional_cpu_state_ms,
                pending->metrics
                    .serialize_and_issue_ms,
                pending->metrics
                    .last_d2h_wait_ms,
                pending->metrics.rig_gate_ms,
                pending->metrics
                        .measured_pinned_d2h_bytes_per_second /
                    static_cast<double>(
                        1024ull * 1024 * 1024),
                pending->metrics
                        .pause_within_rig_gate
                    ? "PASS"
                    : "FAIL",
                pending->metrics.host_rss_delta_bytes,
                pending->metrics.host_ram_within_gate
                    ? "PASS"
                    : "FAIL");

            impl_->mark_issuing_complete(pending);
            prepared.impl_.reset();
            return PendingTrainingSnapshot(
                std::move(pending));
        } catch (const std::exception& error) {
            LFS_CUDA_LOG_TEARDOWN(
                cudaStreamSynchronize(
                    impl_->d2h_stream),
                impl_->d2h_stream,
                "failed training snapshot capture drain");
            pending->pause_end = Clock::now();
            {
                std::scoped_lock lock(pending->mutex);
                pending->error = std::format(
                    "Capture training snapshot failed: {}",
                    error.what());
            }
            impl_->mark_issuing_complete(pending);
            {
                std::unique_lock lock(pending->mutex);
                pending->drained_condition.wait(
                    lock,
                    [&] { return pending->drained; });
            }
            return snapshot_error(
                lfs::ErrorCode::Internal,
                pending->error,
                LFS_SOURCE_SITE_CURRENT());
        }
    }

    TrainingSnapshotServiceMetrics
    TrainingSnapshotService::metrics() const {
        std::scoped_lock lock(impl_->metrics_mutex);
        return impl_->aggregate;
    }

} // namespace lfs::training
