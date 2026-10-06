/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "training_snapshot_service.hpp"

#include "checkpoint.hpp"
#include "core/gpu_elapsed.hpp"
#include "core/host_metrics.hpp"
#include "core/logger.hpp"
#include "core/resource_messages.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"
#include "core/tensor_execution.hpp"
#include "core/tensor_label.hpp"
#include "core/tensor_readback.hpp"
#include "core/tensor_serialization_sink.hpp"
#include "core/tensor_upload.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "strategies/istrategy.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
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
#include <string>
#include <thread>
#include <utility>

#if defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace lfs::training {

    namespace {
#if defined(__APPLE__)
        constexpr std::size_t FILE_BACKED_SNAPSHOT_THRESHOLD =
            64ull * 1024 * 1024;
#endif
    }

    struct TrainingSnapshotBytes::Impl {
        std::unique_ptr<std::byte[]> bytes;
        std::FILE* file = nullptr;
        mutable void* mapping = nullptr;
        std::size_t size = 0;
    };

    TrainingSnapshotBytes::TrainingSnapshotBytes(
        const std::size_t size,
        const bool overwrite)
        : impl_(std::make_unique<Impl>()) {
        impl_->size = size;
#if defined(__APPLE__)
        if (size >= FILE_BACKED_SNAPSHOT_THRESHOLD) {
            // An unlinked spool keeps the capture out of anonymous memory; the
            // kernel can evict its clean pages under pressure. Without a usable
            // temporary file the capture falls back to heap staging.
            impl_->file = std::tmpfile();
            if (impl_->file &&
                ::ftruncate(::fileno(impl_->file), static_cast<off_t>(size)) == 0) {
                return;
            }
            LOG_WARN("Checkpoint spool unavailable ({}); staging {} bytes in memory",
                     std::strerror(errno), size);
            if (impl_->file) {
                std::fclose(impl_->file);
                impl_->file = nullptr;
            }
        }
#endif
        impl_->bytes = overwrite
                           ? std::make_unique_for_overwrite<std::byte[]>(size)
                           : std::make_unique<std::byte[]>(size);
    }

    TrainingSnapshotBytes::~TrainingSnapshotBytes() {
#if defined(__APPLE__)
        if (impl_ && impl_->mapping) {
            ::munmap(impl_->mapping, impl_->size);
        }
#endif
        if (impl_ && impl_->file) {
            std::fclose(impl_->file);
        }
    }

    std::size_t TrainingSnapshotBytes::size() const noexcept {
        return impl_->size;
    }

    std::byte* TrainingSnapshotBytes::data() noexcept {
        return impl_->bytes.get();
    }

    const std::byte* TrainingSnapshotBytes::data() const noexcept {
        return impl_->bytes.get();
    }

    bool TrainingSnapshotBytes::file_backed() const noexcept {
        return impl_->file != nullptr;
    }

    lfs::Result<std::span<const std::byte>>
    TrainingSnapshotBytes::mapped_data() const {
        if (impl_->bytes) {
            return std::span<const std::byte>(impl_->bytes.get(), impl_->size);
        }
#if defined(__APPLE__)
        if (!impl_->mapping) {
            impl_->mapping = ::mmap(
                nullptr, impl_->size, PROT_READ, MAP_SHARED,
                ::fileno(impl_->file), 0);
            if (impl_->mapping == MAP_FAILED) {
                impl_->mapping = nullptr;
                return lfs::make_error(lfs::ErrorInit{
                        .code = lfs::ErrorCode::Unavailable,
                        .domain = lfs::ErrorDomain::Training,
                        .user_message =
                            "The training snapshot could not be mapped.",
                        .detail = std::format(
                            "Checkpoint spool mapping failed: {}",
                            std::strerror(errno)),
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                    });
            }
        }
        return std::span<const std::byte>(
            static_cast<const std::byte*>(impl_->mapping), impl_->size);
#else
        return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Unavailable,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "The training snapshot could not be mapped.",
                .detail = "Checkpoint spool mapping is unavailable",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
#endif
    }

    void TrainingSnapshotBytes::write_at(
        const std::uint64_t offset,
        std::span<const std::byte> source) {
        if (offset > impl_->size ||
            source.size() > impl_->size - offset) {
            throw std::out_of_range(
                "Checkpoint spool write is out of bounds");
        }
        if (source.empty()) {
            return;
        }
        if (impl_->bytes) {
            std::memcpy(impl_->bytes.get() + offset,
                        source.data(), source.size());
            return;
        }
#if defined(__APPLE__)
        std::size_t completed = 0;
        while (completed < source.size()) {
            const auto written = ::pwrite(
                ::fileno(impl_->file), source.data() + completed,
                source.size() - completed,
                static_cast<off_t>(offset + completed));
            if (written < 0 && errno == EINTR) {
                continue;
            }
            if (written <= 0) {
                throw std::runtime_error(std::format(
                    "Checkpoint spool write failed: {}",
                    std::strerror(errno)));
            }
            completed += static_cast<std::size_t>(written);
        }
#else
        throw std::runtime_error("Checkpoint spool is unavailable");
#endif
    }

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

        class SnapshotReplanRequired final
            : public std::runtime_error {
        public:
            using std::runtime_error::runtime_error;
        };

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

        [[nodiscard]] std::string format_memory_size(
            const std::uint64_t bytes) {
            constexpr double BYTES_PER_MIB =
                1024.0 * 1024.0;
            constexpr double BYTES_PER_GIB =
                1024.0 * 1024.0 * 1024.0;
            if (bytes >=
                static_cast<std::uint64_t>(BYTES_PER_GIB)) {
                return std::format(
                    "{:.1f} GiB",
                    static_cast<double>(bytes) /
                        BYTES_PER_GIB);
            }
            return std::format(
                "{:.1f} MiB",
                static_cast<double>(bytes) /
                    BYTES_PER_MIB);
        }

        [[nodiscard]] lfs::Error snapshot_host_memory_error(
            const std::uint64_t required_bytes,
            const std::uint64_t available_bytes,
            std::string detail,
            const lfs::core::SourceSite source) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::ResourceExhausted,
                .domain = lfs::ErrorDomain::Training,
                .user_message = std::format(
                    "{}: needed {}, available {}.",
                    lfs::core::HOST_MEMORY_SAVE_ERROR_PREFIX,
                    format_memory_size(required_bytes),
                    format_memory_size(available_bytes)),
                .detail = std::move(detail),
                .detection = source,
                .fields = lfs::SmallFields{}.add(
                    "resource", "host_memory"),
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
#elif defined(__APPLE__)
            // /proc is unavailable on macOS. Reuse core's Mach resident-size
            // query instead of reporting a zero delta for every capture.
            return core::host_metrics::sample().process_rss_bytes;
#endif
            return 0;
        }

        struct HostMemoryInfo {
            std::uint64_t total_bytes = 0;
            std::uint64_t available_bytes = 0;
        };

        HostMemoryInfo read_host_memory_info() {
            HostMemoryInfo result;
#if defined(_WIN32)
            MEMORYSTATUSEX status{};
            status.dwLength = sizeof(status);
            if (GlobalMemoryStatusEx(&status)) {
                result = {
                    .total_bytes = status.ullTotalPhys,
                    .available_bytes = status.ullAvailPhys,
                };
            }
#elif defined(__APPLE__)
            if (const auto memory = core::host_metrics::memory()) {
                result = {
                    .total_bytes = memory->total_bytes,
                    .available_bytes = memory->available_bytes,
                };
            }
#elif defined(__linux__)
            std::ifstream input("/proc/meminfo");
            std::string line;
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
#endif
            const auto read_override = [](const char* name)
                -> std::optional<std::uint64_t> {
                const auto* value = std::getenv(name);
                if (!value) {
                    return std::nullopt;
                }
                unsigned long long bytes = 0;
                char trailing = '\0';
                if (std::sscanf(value, "%llu %c", &bytes, &trailing) != 1) {
                    return std::nullopt;
                }
                return static_cast<std::uint64_t>(bytes);
            };
            if (const auto total = read_override(
                    "LFS_TRAINING_SNAPSHOT_HOST_MEMORY_TOTAL_BYTES")) {
                result.total_bytes = *total;
            }
            if (const auto available = read_override(
                    "LFS_TRAINING_SNAPSHOT_HOST_MEMORY_AVAILABLE_BYTES")) {
                result.available_bytes = *available;
            }
            return result;
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
            lfs::core::Tensor source;
            const void* source_pointer = nullptr;
            lfs::core::TensorShape source_shape;
            lfs::core::DataType source_dtype =
                lfs::core::DataType::Float32;
            lfs::core::Device source_device =
                lfs::core::Device::CPU;
            std::optional<lfs::core::TensorExecutionTarget> source_stream;
            std::uint64_t source_bytes = 0;
            const void* auxiliary_source_pointer = nullptr;
            lfs::core::TensorShape auxiliary_source_shape;
            lfs::core::DataType auxiliary_source_dtype =
                lfs::core::DataType::Float32;
            lfs::core::Device auxiliary_source_device =
                lfs::core::Device::CPU;
            std::optional<lfs::core::TensorExecutionTarget> auxiliary_source_stream;
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
                std::vector<TensorLayoutWitness>& witnesses, bool retain_sources = false)
                : witnesses_(witnesses), retain_sources_(retain_sources) {}

            void write_tensor_payload(
                std::ostream& destination,
                const lfs::core::Tensor& source,
                const lfs::core::Tensor* auxiliary_source,
                const lfs::core::TensorSerializationDescriptor&
                    descriptor) override {
                validate_tensor_source(
                    source, auxiliary_source, descriptor);
                const auto position = destination.tellp();
                if (position == std::streampos(-1)) {
                    throw std::runtime_error(
                        "Cannot locate serialized tensor payload");
                }
                const auto bytes = descriptor.payload_bytes();
                witnesses_.push_back(TensorLayoutWitness{
                    .source = retain_sources_ ? source : lfs::core::Tensor{},
                    .source_pointer =
                        resolve_source_pointer(source),
                    .source_shape = source.shape(),
                    .source_dtype = source.dtype(),
                    .source_device = source.device(),
                    .source_stream = source.execution_target(),
                    .source_bytes = source.bytes(),
                    .auxiliary_source_pointer =
                        auxiliary_source
                            ? resolve_source_pointer(
                                  *auxiliary_source)
                            : nullptr,
                    .auxiliary_source_shape =
                        auxiliary_source
                            ? auxiliary_source->shape()
                            : lfs::core::TensorShape{},
                    .auxiliary_source_dtype =
                        auxiliary_source
                            ? auxiliary_source->dtype()
                            : lfs::core::DataType::Float32,
                    .auxiliary_source_device =
                        auxiliary_source
                            ? auxiliary_source->device()
                            : lfs::core::Device::CPU,
                    .auxiliary_source_stream =
                        auxiliary_source
                            ? std::make_optional(auxiliary_source->execution_target())
                            : std::nullopt,
                    .descriptor = descriptor,
                    .payload_offset =
                        static_cast<std::uint64_t>(
                            static_cast<std::streamoff>(
                                position)),
                    .payload_bytes = bytes,
                });
                if (source.device() ==
                    lfs::core::Device::GPU) {
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
                const lfs::core::Tensor* auxiliary_source,
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
                    if (auxiliary_source ||
                        source.shape() !=
                            descriptor.serialized_shape ||
                        source.dtype() != descriptor.dtype ||
                        source.bytes() !=
                            descriptor.payload_bytes()) {
                        throw std::runtime_error(
                            "Native snapshot tensor dimensions do not match serialization");
                    }
                    return;
                }
                if (source.device() !=
                        lfs::core::Device::GPU ||
                    source.ndim() != 1 ||
                    descriptor.dtype !=
                        lfs::core::DataType::Float32 ||
                    descriptor.serialized_shape.rank() != 3 ||
                    descriptor.serialized_shape[0] !=
                        descriptor.sh_primitives ||
                    descriptor.serialized_shape[1] !=
                        descriptor.sh_coefficients_rest ||
                    descriptor.serialized_shape[2] !=
                        lfs::core::kShChannels ||
                    descriptor.sh_layout_coefficients_rest <
                        descriptor.sh_coefficients_rest) {
                    throw std::runtime_error(
                        "Swizzled SH snapshot dimensions are inconsistent");
                }
                if (descriptor.encoding ==
                    lfs::core::TensorPayloadEncoding::
                        SwizzledShToCanonical) {
                    if (auxiliary_source ||
                        (source.dtype() !=
                             lfs::core::DataType::Float32 &&
                         source.dtype() !=
                             lfs::core::DataType::Float16) ||
                        source.numel() !=
                            lfs::core::sh_swizzled_float_count(
                                descriptor.sh_primitives,
                                descriptor
                                    .sh_layout_coefficients_rest)) {
                        throw std::runtime_error(
                            "Swizzled SH snapshot dimensions are inconsistent");
                    }
                    return;
                }
                if (descriptor.encoding !=
                        lfs::core::TensorPayloadEncoding::
                            QuantizedShToCanonical ||
                    source.dtype() !=
                        lfs::core::DataType::Float16 ||
                    source.numel() !=
                        lfs::core::sh_value_quant::
                            sh_value_u16_count(
                                descriptor.sh_primitives,
                                descriptor
                                    .sh_layout_coefficients_rest) ||
                    !auxiliary_source ||
                    !auxiliary_source->is_valid() ||
                    !auxiliary_source->is_contiguous() ||
                    auxiliary_source->device() !=
                        lfs::core::Device::GPU ||
                    auxiliary_source->dtype() !=
                        lfs::core::DataType::Float32 ||
                    auxiliary_source->numel() <
                        lfs::core::sh_value_quant::
                                n_bounds_for_prims(
                                    descriptor.sh_primitives) *
                            2) {
                    throw std::runtime_error(
                        "Quantized SH snapshot dimensions are inconsistent");
                }
            }

            static const void* resolve_source_pointer(
                const lfs::core::Tensor& source) {
                return lfs::core::resolve_exportable_device_ptr(
                    source);
            }

        private:
            std::vector<TensorLayoutWitness>& witnesses_;
            bool retain_sources_ = false;
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

    TrainingStepRegressionTracker::
        TrainingStepRegressionTracker(
            const std::size_t window_size)
        : window_size_(window_size) {
        if (window_size_ == 0) {
            throw std::invalid_argument(
                "Training step regression window must be non-zero");
        }
    }

    TrainingStepWindowMetrics
    TrainingStepRegressionTracker::summarize(
        const std::deque<Sample>& samples) const noexcept {
        TrainingStepWindowMetrics result;
        if (samples.empty()) {
            return result;
        }
        result.first_iteration =
            samples.front().iteration;
        result.last_iteration =
            samples.back().iteration;
        result.sample_count = samples.size();
        result.mean_ms =
            std::accumulate(
                samples.begin(), samples.end(), 0.0,
                [](const double sum,
                   const Sample& sample) {
                    return sum + sample.elapsed_ms;
                }) /
            static_cast<double>(samples.size());
        return result;
    }

    void TrainingStepRegressionTracker::observe(
        const int iteration,
        const double elapsed_ms,
        const bool topology_changed) {
        if (!(elapsed_ms >= 0.0) ||
            !std::isfinite(elapsed_ms)) {
            return;
        }
        if (topology_changed) {
            steady_run_.clear();
            if (armed_ &&
                iteration > snapshot_iteration_ &&
                !metrics_.gate_evaluated) {
                post_resume_run_.clear();
                metrics_.post_resume = {};
            }
            return;
        }

        steady_run_.push_back({
            .iteration = iteration,
            .elapsed_ms = elapsed_ms,
        });
        if (steady_run_.size() > window_size_) {
            steady_run_.pop_front();
        }
        if (steady_run_.size() == window_size_) {
            latest_steady_window_ =
                summarize(steady_run_);
        }

        if (!armed_ ||
            iteration <= snapshot_iteration_ ||
            metrics_.gate_evaluated) {
            return;
        }
        post_resume_run_.push_back({
            .iteration = iteration,
            .elapsed_ms = elapsed_ms,
        });
        metrics_.post_resume =
            summarize(post_resume_run_);
        if (post_resume_run_.size() != window_size_) {
            return;
        }
        if (metrics_.pre_snapshot.sample_count !=
                window_size_ ||
            !(metrics_.pre_snapshot.mean_ms > 0.0)) {
            return;
        }
        metrics_.regression_percent =
            (metrics_.post_resume.mean_ms /
                 metrics_.pre_snapshot.mean_ms -
             1.0) *
            100.0;
        metrics_.gate_evaluated = true;
        metrics_.within_gate =
            metrics_.regression_percent <= 10.0;
    }

    void TrainingStepRegressionTracker::arm_after_snapshot(
        const int snapshot_iteration) {
        snapshot_iteration_ = snapshot_iteration;
        post_resume_run_.clear();
        metrics_ = {};
        if (latest_steady_window_) {
            metrics_.pre_snapshot =
                *latest_steady_window_;
        }
        armed_ = true;
    }

    TrainingStepRegressionMetrics
    TrainingStepRegressionTracker::metrics() const noexcept {
        return metrics_;
    }

    void TrainingStepRegressionTracker::reset() noexcept {
        steady_run_.clear();
        latest_steady_window_.reset();
        post_resume_run_.clear();
        metrics_ = {};
        snapshot_iteration_ = 0;
        armed_ = false;
    }

    struct PreparedTrainingSnapshot::Impl {
        lfs::core::Uuid snapshot_uuid;
        int planned_iteration = 0;
        std::uint64_t baseline_rss_bytes = 0;
        std::uint64_t checkpoint_bytes = 0;
        std::uint64_t device_snapshot_bytes = 0;
        std::vector<TensorLayoutWitness> layout;
        std::shared_ptr<TrainingSnapshotBytes> staging;
        TrainingSnapshotPauseMetrics metrics;
    };

    struct PendingTrainingSnapshot::Impl {
        mutable std::mutex mutex;
        std::condition_variable drained_condition;
        std::shared_ptr<TrainingSnapshotBytes> staging;
        std::uint64_t baseline_rss_bytes = 0;
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

            bool busy = false;
            std::optional<DrainTask> task;
        };

        // Recycle at most one Metal checkpoint allocation, only after its
        // last immutable owner has released it. A new shared_ptr control block
        // prevents old weak owners from acquiring a buffer being overwritten.
        struct StagingPool {
            explicit StagingPool(const std::size_t max_retired_bytes)
                : max_retired_bytes(max_retired_bytes) {}

            std::mutex mutex;
            std::unique_ptr<TrainingSnapshotBytes> retired;
            const std::size_t max_retired_bytes;

            std::unique_ptr<TrainingSnapshotBytes> take(const std::size_t bytes) {
                std::scoped_lock lock(mutex);
                if (retired && retired->size() == bytes)
                    return std::move(retired);
                retired.reset();
                return {};
            }

            void release(std::unique_ptr<TrainingSnapshotBytes> bytes) {
                std::scoped_lock lock(mutex);
                if (!retired && !bytes->file_backed() &&
                    bytes->size() <= max_retired_bytes)
                    retired = std::move(bytes);
            }

            bool trim() {
                std::scoped_lock lock(mutex);
                const bool released = bool(retired);
                retired.reset();
                return released;
            }
        };

        std::shared_ptr<TrainingSnapshotBytes> acquire_staging(const std::size_t bytes) {
            if (!ring->prefers_recycled_host_staging())
                return std::make_shared<TrainingSnapshotBytes>(bytes);
            auto allocation = staging_pool->take(bytes);
            if (!allocation)
                allocation = std::make_unique<TrainingSnapshotBytes>(bytes, true);
            const std::weak_ptr<StagingPool> retired_pool = staging_pool;
            return std::shared_ptr<TrainingSnapshotBytes>(allocation.release(), [retired_pool](TrainingSnapshotBytes* value) {
                std::unique_ptr<TrainingSnapshotBytes> retired(value);
                if (const auto pool = retired_pool.lock())
                    pool->release(std::move(retired));
            });
        }

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
            staging_pool = std::make_shared<StagingPool>(
                config.ring_slots * config.band_bytes);
        }

        ~Impl() {
            shutdown();
        }

        void initialize_resources(
            const std::vector<TensorLayoutWitness>& layout,
            const std::span<const lfs::core::TensorExecutionTarget>
                mutating_queues) {
            if (!d2h_queue)
                d2h_queue = std::make_unique<lfs::core::TensorWorkQueue>(lfs::core::default_gpu_backend());
            ensure_device_scratch();
            calibrate_once(layout, mutating_queues);
            device_scratch = {};
            if (slots.empty()) {
                ring = std::make_unique<lfs::core::TensorReadbackRing>(d2h_queue->backend(),
                                                                       config.ring_slots, config.band_bytes, *d2h_queue, &device_scratch);
                slots.resize(config.ring_slots);
                for (size_t i = 0; i < slots.size(); ++i)
                    slots[i].pinned = ring->slot_bytes(i).data();
                // Slots own disjoint staging ranges. A single CPU copy worker
                // can hold every slot busy while the GPU is already finished,
                // extending the optimizer pause to pageable-copy throughput.
                // Keep the workers persistent and bounded with the ring.
                const auto worker_count = std::min<std::size_t>(slots.size(), 4);
                drain_threads.reserve(worker_count);
                for (std::size_t i = 0; i < worker_count; ++i) {
                    drain_threads.emplace_back([this](std::stop_token stop) {
                        drain_loop(stop);
                    });
                }
            }
        }

        void ensure_device_scratch() {
            if (device_scratch.is_valid())
                return;
            lfs::core::TensorWorkQueue::Scope scope(*d2h_queue);
            lfs::core::TensorLabelScope label("training.snapshot.device_scratch");
            device_scratch = lfs::core::Tensor::empty({config.band_bytes}, lfs::core::Device::GPU,
                                                      lfs::core::DataType::UInt8);
            device_scratch.set_name("training.snapshot.device_scratch");
        }

        void calibrate_once(
            const std::vector<TensorLayoutWitness>& layout,
            const std::span<const lfs::core::TensorExecutionTarget>
                mutating_queues) {
            std::scoped_lock lock(calibration_mutex);
            if (process_pinned_d2h_bytes_per_second > 0.0) {
                measured_bandwidth =
                    process_pinned_d2h_bytes_per_second;
                return;
            }
            std::set<lfs::core::TensorExecutionTarget> streams;
            for (const auto stream : mutating_queues) {
                if (!stream.is_default_queue()) {
                    streams.insert(stream);
                }
            }
            for (const auto& witness : layout) {
                if (witness.source_device ==
                        lfs::core::Device::GPU &&
                    witness.source_stream && !witness.source_stream->is_default_queue()) {
                    streams.insert(*witness.source_stream);
                }
                if (witness.auxiliary_source_device ==
                        lfs::core::Device::GPU &&
                    witness.auxiliary_source_stream && !witness.auxiliary_source_stream->is_default_queue()) {
                    streams.insert(
                        *witness.auxiliary_source_stream);
                }
            }
            for (const auto stream : streams) {
                stream.wait();
            }
            const auto source = std::ranges::max_element(
                layout, std::less{},
                [](const TensorLayoutWitness& witness) {
                    return witness.source_device ==
                                       lfs::core::Device::GPU &&
                                   witness.source_pointer &&
                                   witness.payload_bytes > 0
                               ? source_raw_bytes(witness)
                               : std::uint64_t{0};
                });
            if (source == layout.end() ||
                source->source_device !=
                    lfs::core::Device::GPU ||
                !source->source_pointer ||
                source->payload_bytes == 0) {
                measured_bandwidth =
                    std::numeric_limits<double>::infinity();
                return;
            }
            const auto bytes = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    std::min<std::uint64_t>(
                        config.calibration_bytes,
                        config.band_bytes),
                    source->source_device ==
                            lfs::core::Device::GPU
                        ? source_raw_bytes(*source)
                        : 0));
            if (bytes == 0) {
                throw std::runtime_error(
                    "Snapshot calibration found no CUDA source bytes");
            }
            if (!device_scratch.is_valid()) {
                throw std::runtime_error(
                    "Snapshot calibration requires device scratch");
            }

            lfs::core::TensorReadbackRing calibration(d2h_queue->backend(), 1, bytes, *d2h_queue, &device_scratch);
            calibration.enqueue(source->source, 0, bytes, 0, 0, true);
            calibration.seal(0);
            calibration.wait(0);
            calibration.release(0);
            lfs::core::GpuElapsed elapsed(d2h_queue->backend(), 2);
            if (!elapsed.mark(0, *d2h_queue))
                throw std::runtime_error("Cannot start readback calibration timer");
            // Every iteration appends to the same slot; one seal covers the batch.
            for (int i = 0; i < config.calibration_iterations; ++i)
                calibration.enqueue(source->source, 0, bytes, 0, 0, true);
            calibration.seal(0);
            if (!elapsed.mark(1, *d2h_queue) || !elapsed.wait_event(1))
                throw std::runtime_error("Cannot finish readback calibration timer");
            const auto elapsed_ms = elapsed.milliseconds(0, 1);
            if (!elapsed_ms || !(*elapsed_ms > 0.0f))
                throw std::runtime_error("Pinned D2H calibration duration is zero");
            process_pinned_d2h_bytes_per_second = static_cast<double>(bytes) * config.calibration_iterations /
                                                  (static_cast<double>(*elapsed_ms) / 1000.0);
            measured_bandwidth = process_pinned_d2h_bytes_per_second;
        }

        static std::uint64_t source_raw_bytes(
            const TensorLayoutWitness& witness) {
            return witness.source_bytes;
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

        void issue_native_to_slot(size_t slot_index, size_t pinned_offset,
                                  const lfs::core::Tensor& source, size_t source_offset, size_t bytes) {
            validate_slot_range(slot_index, pinned_offset, bytes);
            // Capture waits for the last ring fence before training resumes.
            // A direct copy already makes the host slot immutable; an extra
            // source-to-device-scratch copy adds traffic without extending
            // the snapshot lifetime. Encoded SH still uses device_scratch.
            ring->enqueue(source, source_offset, bytes, slot_index, pinned_offset);
        }

        void issue_sh_to_slot(
            const std::size_t slot_index,
            const std::size_t pinned_offset,
            const TensorLayoutWitness& witness,
            const lfs::core::Tensor& source,
            const lfs::core::Tensor* auxiliary,
            const std::uint64_t tensor_byte_offset,
            const std::size_t bytes) {
            validate_slot_range(
                slot_index, pinned_offset, bytes);
            if (!device_scratch.is_valid() ||
                tensor_byte_offset % sizeof(float) != 0 ||
                bytes % sizeof(float) != 0) {
                throw std::runtime_error(
                    "Bounded SH snapshot band is misaligned");
            }
            const auto encoding =
                witness.descriptor.encoding;
            lfs::gpu_ops::ShStorage storage = lfs::gpu_ops::ShStorage::Float32;
            if (encoding ==
                lfs::core::TensorPayloadEncoding::
                    QuantizedShToCanonical) {
                storage = lfs::gpu_ops::ShStorage::Q16;
            } else if (encoding ==
                           lfs::core::
                               TensorPayloadEncoding::
                                   SwizzledShToCanonical &&
                       witness.source_dtype ==
                           lfs::core::DataType::Float16) {
                storage = lfs::gpu_ops::ShStorage::IeeeFloat16;
            } else if (encoding !=
                       lfs::core::TensorPayloadEncoding::
                           SwizzledShToCanonical) {
                throw std::runtime_error(
                    "Unsupported SH snapshot encoding");
            }
            const lfs::core::Tensor absent;
            {
                const lfs::core::TensorExecutionTarget::Scope execution_scope(
                    *d2h_queue);
                training_sh_ops().decode_range(
                    source,
                    auxiliary != nullptr ? *auxiliary : absent,
                    device_scratch,
                    {.canonical_float_offset = tensor_byte_offset / sizeof(float),
                     .float_count = bytes / sizeof(float),
                     .primitives = witness.descriptor.sh_primitives,
                     .destination_rest = witness.descriptor.sh_coefficients_rest,
                     .layout_rest = witness.descriptor.sh_layout_coefficients_rest,
                     .storage = storage});
            }
            ring->enqueue(device_scratch, 0, bytes, slot_index, pinned_offset);
        }

        [[nodiscard]] const lfs::core::TensorFence* submit_ring_slot(
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
            const auto* completion = &ring->seal(slot_index);
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
            ring_condition.notify_all();
            return completion;
        }

        void cancel_ring_slot(
            const std::size_t slot_index) noexcept {
            if (slot_index >= slots.size()) {
                return;
            }
            try {
                d2h_queue->wait();
                ring->release(slot_index);
            } catch (const std::exception& e) { LOG_WARN("Snapshot cancel drain failed: {}", e.what()); }
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
                try {
                    ring->wait(slot_index);
                    for (const auto& segment :
                         task.segments) {
                        const auto source = std::span<const std::byte>(
                            static_cast<const std::byte*>(slot.pinned) +
                                segment.pinned_offset,
                            segment.bytes);
                        if (auto* destination =
                                task.capture->staging->data()) {
                            non_temporal_copy(
                                destination + segment.destination_offset,
                                source.data(), source.size());
                        } else {
                            task.capture->staging->write_at(
                                segment.destination_offset, source);
                        }
                    }
                    ring->release(slot_index);
                } catch (const std::exception& e) {
                    error = std::format("snapshot drain: {}", e.what());
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
            // Overwrite storage faults pages during drains, so sample the
            // completed buffer as well as the earlier pause-end observation.
            const auto rss_after_drain = read_rss_bytes();
            bool record = false;
            {
                std::scoped_lock lock(capture->mutex);
                if (capture->completion_recorded) {
                    return;
                }
                capture->completion_recorded = true;
                if (rss_after_drain >= capture->baseline_rss_bytes)
                    capture->metrics.host_rss_delta_bytes = std::max(
                        capture->metrics.host_rss_delta_bytes, rss_after_drain - capture->baseline_rss_bytes);
                capture->metrics.host_ram_within_gate =
                    capture->metrics.host_rss_delta_bytes <=
                    capture->metrics.host_staging_bytes +
                        HOST_MEMORY_GATE_HEADROOM_BYTES;
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
                // Never expose an unwritten byte from overwrite storage.
                // Header rewrites and CPU tensor stamps may overlap; their
                // union with fully drained device payloads must cover CKPT.
                std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
                ranges.reserve(capture->stamps.size());
                for (const auto& stamp : capture->stamps) {
                    if (stamp.bytes > 0 && stamp.offset <= capture->staging->size() &&
                        stamp.bytes <= capture->staging->size() - stamp.offset)
                        ranges.emplace_back(stamp.offset, stamp.offset + stamp.bytes);
                }
                std::ranges::sort(ranges);
                std::uint64_t covered = 0;
                for (const auto& [begin, end] : ranges) {
                    if (begin > covered) {
                        consistent = false;
                        break;
                    }
                    covered = std::max(covered, end);
                }
                consistent = consistent && covered == capture->staging->size();
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
                        "Snapshot byte coverage or UUID consistency proof failed";
                }
                capture->metrics.final_drain_ms =
                    Milliseconds(
                        Clock::now() -
                        capture->pause_end)
                        .count();
                capture->drained = true;
                completed = capture->metrics;
                record = capture->error.empty();
            }
            capture->drained_condition.notify_all();
            double pause_p95_ms = 0.0;
            std::size_t p95_n = 0;
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
                    aggregate.p95_n =
                        pause_samples.size();
                    pause_p95_ms =
                        aggregate.pause_p95_ms;
                    p95_n = aggregate.p95_n;
                }
            }
            if (record) {
                LOG_INFO(
                    "Training snapshot pause metric: "
                    "p95={:.3f}ms p95_n={} snapshot={} final_drain={:.3f}ms "
                    "host_delta={} host_gate={} consistency={}",
                    pause_p95_ms, p95_n, completed.snapshot_uuid.to_string(),
                    completed.final_drain_ms, completed.host_rss_delta_bytes,
                    completed.host_ram_within_gate ? "PASS" : "FAIL",
                    completed.consistency_proven ? "PASS" : "FAIL");
            }
        }

        void shutdown() {
            for (auto& worker : drain_threads)
                worker.request_stop();
            ring_condition.notify_all();
            for (auto& worker : drain_threads) {
                if (worker.joinable())
                    worker.join();
            }
            drain_threads.clear();
            if (d2h_queue) {
                try {
                    d2h_queue->wait();
                } catch (const std::exception& e) { LOG_WARN("Snapshot shutdown drain failed: {}", e.what()); }
            }
            ring.reset();
            device_scratch = {};
            slots.clear();
            d2h_queue.reset();
        }

        TrainingSnapshotServiceConfig config;
        bool initialized = false;
        double initialization_ms = 0.0;
        std::unique_ptr<lfs::core::TensorWorkQueue> d2h_queue;
        std::unique_ptr<lfs::core::TensorReadbackRing> ring;
        lfs::core::Tensor device_scratch;
        std::vector<RingSlot> slots;
        std::size_t next_slot = 0;
        std::mutex ring_mutex;
        std::condition_variable ring_condition;
        std::deque<std::size_t> drain_queue;
        std::vector<std::jthread> drain_threads;
        std::shared_ptr<StagingPool> staging_pool;
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
                    capture_->staging->write_at(
                        cursor_,
                        std::span<const std::byte>(
                            reinterpret_cast<const std::byte*>(source),
                            static_cast<std::size_t>(count)));
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
                const lfs::core::Tensor* auxiliary_source,
                const lfs::core::TensorSerializationDescriptor&
                    descriptor) override {
                if (index_ >= layout_.size()) {
                    throw std::runtime_error(
                        "Snapshot layout gained a tensor");
                }
                CountingTensorSink::validate_tensor_source(
                    source, auxiliary_source, descriptor);
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
                const auto auxiliary_matches =
                    auxiliary_source
                        ? expected
                                      .auxiliary_source_pointer ==
                                  CountingTensorSink::
                                      resolve_source_pointer(
                                          *auxiliary_source) &&
                              expected
                                      .auxiliary_source_shape ==
                                  auxiliary_source->shape() &&
                              expected
                                      .auxiliary_source_dtype ==
                                  auxiliary_source->dtype() &&
                              expected
                                      .auxiliary_source_device ==
                                  auxiliary_source->device()
                        : expected
                                  .auxiliary_source_pointer ==
                              nullptr;
                if (CountingTensorSink::
                            resolve_source_pointer(source) !=
                        expected.source_pointer ||
                    source.shape() !=
                        expected.source_shape ||
                    source.dtype() !=
                        expected.source_dtype ||
                    source.device() !=
                        expected.source_device ||
                    source.bytes() !=
                        expected.source_bytes ||
                    !auxiliary_matches ||
                    !descriptors_equal(
                        descriptor,
                        expected.descriptor) ||
                    offset != expected.payload_offset ||
                    bytes != expected.payload_bytes) {
                    throw SnapshotReplanRequired(
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
                        descriptor, expected, source,
                        auxiliary_source, offset, bytes);
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

            [[nodiscard]] const lfs::core::TensorFence*
            last_event() const noexcept {
                return last_event_;
            }

        private:
            void append_device_tensor(
                const lfs::core::
                    TensorSerializationDescriptor&
                        descriptor,
                const TensorLayoutWitness& witness,
                const lfs::core::Tensor& source,
                const lfs::core::Tensor* auxiliary,
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
                    if (descriptor.encoding !=
                        lfs::core::
                            TensorPayloadEncoding::
                                NativeContiguous) {
                        const auto row_bytes =
                            static_cast<std::uint64_t>(
                                descriptor
                                    .sh_coefficients_rest) *
                            lfs::core::kShChannels *
                            sizeof(float);
                        const auto block_bytes =
                            row_bytes *
                            static_cast<std::uint64_t>(
                                lfs::training::sh_value::
                                    kBlockSize);
                        if (block_bytes > 0 &&
                            count >= block_bytes) {
                            count -=
                                count %
                                static_cast<std::size_t>(
                                    block_bytes);
                        } else if (row_bytes > 0) {
                            count -=
                                count %
                                static_cast<std::size_t>(
                                    row_bytes);
                        } else {
                            count -=
                                count % sizeof(float);
                        }
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
                            source, tensor_offset,
                            count);
                    } else {
                        service_.issue_sh_to_slot(
                            *active_slot_,
                            active_slot_bytes_,
                            witness, source, auxiliary,
                            tensor_offset, count);
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
            const lfs::core::TensorFence* last_event_ = nullptr;
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

    void TrainingSnapshotService::
        reset_process_pinned_d2h_calibration_for_testing() {
        std::scoped_lock lock(calibration_mutex);
        process_pinned_d2h_bytes_per_second = 0.0;
    }

    lfs::Result<void> TrainingSnapshotService::initialize(
        const TrainingSnapshotCaptureRequest& request) {
        if (impl_->initialized) {
            return {};
        }
        try {
            const auto begin = Clock::now();
            std::vector<TensorLayoutWitness> layout;
            CountingStreamBuffer buffer;
            std::ostream destination(&buffer);
            CountingTensorSink sink(layout, true);
            {
                lfs::core::TensorSerializationSinkScope
                    scope(sink);
                auto serialized = serialize_checkpoint(
                    destination, request.iteration,
                    request.strategy, request.params,
                    request.bilateral_grid, request.ppisp,
                    request.ppisp_controller_pool,
                    request.sparsity_optimizer);
                if (!serialized) {
                    return lfs::Status::failure(
                        std::move(serialized)
                            .error()
                            .with_context(
                                "initialize training snapshot layout",
                                LFS_SOURCE_SITE_CURRENT()));
                }
                if (!destination ||
                    buffer.size() !=
                        serialized->bytes ||
                    buffer.size() == 0) {
                    return lfs::Status::failure(
                        snapshot_error(
                            lfs::ErrorCode::DataLoss,
                            "Snapshot initialization produced an invalid checkpoint layout",
                            LFS_SOURCE_SITE_CURRENT()));
                }
            }
            impl_->initialize_resources(
                layout, request.mutating_queues);
            impl_->initialization_ms =
                Milliseconds(Clock::now() - begin)
                    .count();
            impl_->initialized = true;
            {
                std::scoped_lock lock(
                    impl_->metrics_mutex);
                impl_->aggregate.last
                    .service_initialization_ms =
                    impl_->initialization_ms;
                impl_->aggregate.last
                    .measured_pinned_d2h_bytes_per_second =
                    impl_->measured_bandwidth;
                impl_->aggregate.last
                    .pinned_peak_bytes =
                    impl_->config.ring_slots *
                    impl_->config.band_bytes;
            }
            LOG_INFO(
                "Training snapshot service initialized off the save path: "
                "init={:.3f}ms pinned={} bytes raw_pinned_D2H={:.3f}GiB/s "
                "mutating_queues={}",
                impl_->initialization_ms,
                impl_->config.ring_slots *
                    impl_->config.band_bytes,
                impl_->measured_bandwidth /
                    static_cast<double>(
                        1024ull * 1024 * 1024),
                request.mutating_queues.size());
            return {};
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): normalize the exception into a typed snapshot error.
            return lfs::Status::failure(snapshot_error(
                lfs::ErrorCode::Internal,
                std::format(
                    "Initialize training snapshot service failed: {}",
                    error.what()),
                LFS_SOURCE_SITE_CURRENT()));
        }
    }

    lfs::Result<PreparedTrainingSnapshot>
    TrainingSnapshotService::prepare(
        const TrainingSnapshotCaptureRequest& request) {
        if (!impl_->initialized) {
            return snapshot_error(
                lfs::ErrorCode::FailedPrecondition,
                "Training snapshot service must be initialized before prepare",
                LFS_SOURCE_SITE_CURRENT());
        }
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
            if (prepared->checkpoint_bytes >
                std::numeric_limits<std::size_t>::max()) {
                return snapshot_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "Checkpoint staging exceeds address space",
                    LFS_SOURCE_SITE_CURRENT());
            }
            auto host_memory =
                read_host_memory_info();
            const auto reserve_bytes =
                request.relaxed_host_memory_gate
                    ? HOST_MEMORY_GATE_HEADROOM_BYTES
                    : std::max<std::uint64_t>(
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
            const bool file_backed_staging =
#if defined(__APPLE__)
                prepared->checkpoint_bytes >=
                FILE_BACKED_SNAPSHOT_THRESHOLD;
#else
                false;
#endif
            const auto resident_staging_bytes =
                file_backed_staging ? std::uint64_t{0}
                                    : prepared->checkpoint_bytes;
            const auto required_host_memory =
                resident_staging_bytes + reserve_bytes;
            // An idle recycled buffer must never turn a previously viable
            // save into a memory-pressure rejection. Live readers retain
            // their own buffers; the pool can only release retired storage.
            if (host_memory.available_bytes < required_host_memory &&
                impl_->staging_pool->trim())
                host_memory = read_host_memory_info();
            if (request.release_host_memory &&
                host_memory.available_bytes > 0 &&
                host_memory.available_bytes <
                    required_host_memory &&
                request.release_host_memory(
                    required_host_memory -
                    host_memory.available_bytes) > 0) {
                host_memory = read_host_memory_info();
            }
            if (host_memory.available_bytes == 0 ||
                host_memory.available_bytes <
                    required_host_memory) {
                return snapshot_host_memory_error(
                    required_host_memory,
                    host_memory.available_bytes,
                    std::format(
                        "Training snapshot {}: {} bytes available, "
                        "{} required ({} resident staging + {} reserve; "
                        "{} disk spool)",
                        request.relaxed_host_memory_gate
                            ? "rejected"
                            : "deferred",
                        host_memory.available_bytes,
                        required_host_memory,
                        resident_staging_bytes,
                        reserve_bytes,
                        file_backed_staging
                            ? prepared->checkpoint_bytes
                            : 0),
                    LFS_SOURCE_SITE_CURRENT());
            }
            prepared->staging = impl_->acquire_staging(
                static_cast<std::size_t>(prepared->checkpoint_bytes));
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
                prepared->staging->file_backed()
                    ? 0
                    : prepared->checkpoint_bytes;
            prepared->metrics.disk_staging_bytes =
                prepared->staging->file_backed()
                    ? prepared->checkpoint_bytes
                    : 0;
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
                prepared->metrics.host_staging_bytes +
                    HOST_MEMORY_GATE_HEADROOM_BYTES;
            prepared->metrics.service_initialization_ms =
                impl_->initialization_ms;
            prepared->metrics.prepare_stall_ms =
                Milliseconds(Clock::now() - begin)
                    .count();
            prepared->metrics.preparation_ms =
                prepared->metrics.prepare_stall_ms;
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
        pending->baseline_rss_bytes = prepared.impl_->baseline_rss_bytes;
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
            std::set<lfs::core::TensorExecutionTarget> streams;
            streams.insert(*impl_->d2h_queue);
            for (const auto stream :
                 request.mutating_queues) {
                if (!stream.is_default_queue()) {
                    streams.insert(stream);
                }
            }
            for (const auto& witness :
                 prepared.impl_->layout) {
                if (witness.source_device ==
                        lfs::core::Device::GPU &&
                    witness.source_stream && !witness.source_stream->is_default_queue()) {
                    streams.insert(
                        *witness.source_stream);
                }
                if (witness.auxiliary_source_device ==
                        lfs::core::Device::GPU &&
                    witness.auxiliary_source_stream && !witness.auxiliary_source_stream->is_default_queue()) {
                    streams.insert(
                        *witness.auxiliary_source_stream);
                }
            }
            for (const auto stream : streams) {
                stream.wait();
            }
            const auto sync_end = Clock::now();
            impl_->ensure_device_scratch();

            if (request.capture_additional_cpu_state) {
                auto captured =
                    request.capture_additional_cpu_state(
                        prepared.impl_->snapshot_uuid);
                if (!captured) {
                    throw std::runtime_error(
                        lfs::format_for_developer(
                            captured.error()));
                }
                pending->metrics.scng_ms =
                    captured->scng_ms;
                pending->metrics.selm_ms =
                    captured->selm_ms;
                pending->metrics.prms_ms =
                    captured->prms_ms;
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
                    const auto& error =
                        serialized.error();
                    // serialize_checkpoint() maps layout-change
                    // exceptions to a Result; restore the type.
                    if (error.code() ==
                            lfs::ErrorCode::
                                FailedPrecondition ||
                        error.detail().find(
                            "layout changed") !=
                            std::string_view::npos) {
                        throw SnapshotReplanRequired(
                            std::string(error.detail()));
                    }
                    throw std::runtime_error(
                        lfs::format_for_developer(
                            error));
                }
                if (serialized->bytes !=
                        prepared.impl_
                            ->checkpoint_bytes ||
                    !sink.complete()) {
                    throw SnapshotReplanRequired(
                        "Checkpoint layout changed after preparation");
                }
                sink.finish();
            }
            const auto serialize_end = Clock::now();
            if (const auto last_event =
                    sink.last_event()) {
                last_event->wait();
            }
            impl_->device_scratch = {};
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
                pending->metrics.host_staging_bytes +
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
            lfs::diagnostics::VramProfiler::instance().mark("training_snapshot", {}, pending->metrics.device_snapshot_bytes, pending->metrics.pause_ms);
            pending->metrics.cold_path_ms =
                pending->metrics.pause_ms +
                (pending->metrics.cold_first_snapshot
                     ? pending->metrics
                           .prepare_stall_ms
                     : 0.0);
            pending->metrics
                .pause_within_rig_gate =
                pending->metrics.pause_ms <=
                pending->metrics.rig_gate_ms;
            pending->metrics
                .cold_path_within_rig_gate =
                pending->metrics.cold_path_ms <=
                pending->metrics.rig_gate_ms;
            pending->pause_end = pause_end;
            lfs::core::Tensor::trim_memory_pool();

            LOG_INFO(
                "Training snapshot {} iter {}: "
                "bytes={} device_bytes={} pause={:.3f}ms "
                "prepare_stall={:.3f}ms cold_path={:.3f}ms "
                "cold_first={} "
                "(safe_entry={:.3f} sync={:.3f} cpu_state={:.3f} "
                "serialize+issue={:.3f} "
                "last_d2h_wait={:.3f}) gate={:.3f}ms "
                "raw_pinned_D2H={:.3f}GiB/s pause={} cold_path={} "
                "host_delta={} host_gate={}",
                pending->metrics.snapshot_uuid
                    .to_string(),
                request.iteration,
                pending->metrics.checkpoint_bytes,
                pending->metrics
                    .device_snapshot_bytes,
                pending->metrics.pause_ms,
                pending->metrics.prepare_stall_ms,
                pending->metrics.cold_path_ms,
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
                pending->metrics
                        .cold_path_within_rig_gate
                    ? "PASS"
                    : "FAIL",
                pending->metrics.host_rss_delta_bytes,
                pending->metrics.host_ram_within_gate
                    ? "PASS"
                    : "FAIL");
            LOG_INFO(
                "Training snapshot {} CPU value capture in safe point: "
                "SCNG={:.3f}ms SELM={:.3f}ms PRMS={:.3f}ms total={:.3f}ms",
                pending->metrics.snapshot_uuid
                    .to_string(),
                pending->metrics.scng_ms,
                pending->metrics.selm_ms,
                pending->metrics.prms_ms,
                pending->metrics
                    .additional_cpu_state_ms);

            impl_->mark_issuing_complete(pending);
            prepared.impl_.reset();
            return PendingTrainingSnapshot(
                std::move(pending));
        } catch (const std::exception& error) {
            try {
                impl_->d2h_queue->wait();
            } catch (const std::exception& e) { LOG_WARN("Failed snapshot drain: {}", e.what()); }
            impl_->device_scratch = {};
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
            const bool requires_replan =
                dynamic_cast<
                    const SnapshotReplanRequired*>(
                    &error) != nullptr;
            return snapshot_error(
                requires_replan
                    ? lfs::ErrorCode::
                          FailedPrecondition
                    : lfs::ErrorCode::Internal,
                pending->error,
                LFS_SOURCE_SITE_CURRENT());
        }
    }

    TrainingSnapshotServiceMetrics
    TrainingSnapshotService::metrics() const {
        std::scoped_lock lock(impl_->metrics_mutex);
        return impl_->aggregate;
    }

    void TrainingSnapshotService::
        testing_advance_completed_snapshots(
            const std::uint64_t count) {
        std::scoped_lock lock(impl_->metrics_mutex);
        impl_->aggregate.completed_snapshots += count;
    }

} // namespace lfs::training
