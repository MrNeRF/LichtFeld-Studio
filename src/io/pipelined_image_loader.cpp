/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/pipelined_image_loader.hpp"
#include "core/assert.hpp"
#include "core/error_reporter.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_image.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "pipelined_image_loader_ring.hpp"

#include <stb_image.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <semaphore>
#include <sstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <unistd.h>
#endif
#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace lfs::io {

    namespace {
        constexpr double HOST_FREE_RAM_EVICTION_RATIO = 0.10;
        constexpr double HOST_FREE_RAM_RELIEF_HYSTERESIS_RATIO = 0.05;
        constexpr std::chrono::milliseconds HOST_MEMORY_CHECK_INTERVAL{500};

        void return_freed_heap_to_os() {
#ifdef __GLIBC__
            malloc_trim(0);
#endif
        }
    } // namespace

    namespace {

        [[nodiscard]] size_t tensor_reserved_bytes(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid()) {
                return 0;
            }

            if (tensor.capacity() == 0 || tensor.ndim() == 0) {
                return tensor.bytes();
            }

            size_t row_elems = 1;
            if (tensor.ndim() > 1) {
                for (size_t dim = 1; dim < tensor.ndim(); ++dim) {
                    row_elems *= tensor.shape()[dim];
                }
            }

            return tensor.capacity() * row_elems * lfs::core::dtype_size(tensor.dtype());
        }

        void subtract_clamped(std::atomic<size_t>& value, const size_t amount) {
            if (amount == 0) {
                return;
            }

            auto current = value.load(std::memory_order_relaxed);
            while (current > 0) {
                const size_t next = current > amount ? current - amount : 0;
                if (value.compare_exchange_weak(
                        current, next, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    return;
                }
            }
        }

        [[nodiscard]] std::string legacy_message_from(const lfs::Error& error) {
            return error.user_message().empty() ? std::string(error.detail())
                                                : std::string(error.user_message());
        }

        std::filesystem::path get_temp_folder() {
#ifdef _WIN32
            const char* temp = std::getenv("TEMP");
            if (!temp)
                temp = std::getenv("TMP");
            return temp ? std::filesystem::path(temp) : std::filesystem::path("C:/Temp");
#else
            return std::filesystem::path("/tmp");
#endif
        }

        [[nodiscard]] std::uint64_t current_process_id() {
#ifdef _WIN32
            return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
            return static_cast<std::uint64_t>(getpid());
#endif
        }

        [[nodiscard]] bool process_is_alive(const std::uint64_t pid) {
#ifdef _WIN32
            HANDLE handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
            if (!handle)
                return false;
            DWORD exit_code = 0;
            const bool alive = GetExitCodeProcess(handle, &exit_code) && exit_code == STILL_ACTIVE;
            CloseHandle(handle);
            return alive;
#else
            return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
#endif
        }

        [[nodiscard]] std::filesystem::path run_spill_base() {
            return get_temp_folder() / "LichtFeld" / "pipeline_runs";
        }

        void remove_stale_run_spill_directories() {
            const auto base = run_spill_base();
            std::error_code ec;
            if (!std::filesystem::is_directory(base, ec) || ec)
                return;

            constexpr std::string_view prefix = "ppl_run_";
            for (const auto& entry : std::filesystem::directory_iterator(base, ec)) {
                if (ec)
                    break;
                if (!entry.is_directory(ec) || ec) {
                    ec.clear();
                    continue;
                }
                const auto name = entry.path().filename().string();
                if (!name.starts_with(prefix))
                    continue;

                const auto pid_start = prefix.size();
                const auto pid_end = name.find('_', pid_start);
                std::uint64_t pid = 0;
                const auto parse_end = pid_end == std::string::npos ? name.size() : pid_end;
                const auto [ptr, parse_ec] = std::from_chars(
                    name.data() + pid_start, name.data() + parse_end, pid);
                const bool parsed = parse_ec == std::errc{} && ptr == name.data() + parse_end;
                if (parsed && process_is_alive(pid))
                    continue;

                std::error_code remove_ec;
                std::filesystem::remove_all(entry.path(), remove_ec);
                if (remove_ec) {
                    LOG_WARN("[PipelinedImageLoader] Failed to remove stale run spill {}: {}",
                             lfs::core::path_to_utf8(entry.path()), remove_ec.message());
                }
            }
        }

        [[nodiscard]] bool load_params_need_processing(const LoadParams& params) {
            return params.resize_factor > 1 || params.max_width > 0 || params.undistort != nullptr;
        }

        [[nodiscard]] bool is_regular_file_no_throw(const std::filesystem::path& path) {
            if (path.empty()) {
                return false;
            }
            std::error_code ec;
            const bool exists = std::filesystem::exists(path, ec);
            if (ec || !exists) {
                return false;
            }
            const bool regular = std::filesystem::is_regular_file(path, ec);
            return !ec && regular;
        }

    } // namespace

    PipelinedImageLoader::PipelinedImageLoader(PipelinedLoaderConfig config)
        : config_(std::move(config)),
          output_queue_(std::max<size_t>(1, config_.output_queue_size)) {

        const bool cuda_run = config_.backend == lfs::core::GpuBackend::CUDA;
        if (cuda_run)
            cleanup_stale_run_spill_directories();
        config_.jpeg_batch_size = std::clamp<size_t>(config_.jpeg_batch_size, 1, 12);
        if (config_.decode_frame_ring_capacity == 0) {
            config_.decode_frame_ring_capacity = DECODE_FRAME_RING_CAPACITY;
        }
        config_.decode_frame_ring_capacity = std::clamp<size_t>(
            config_.decode_frame_ring_capacity, 4, DECODE_FRAME_RING_CAPACITY);
        if (config_.max_cache_bytes == 0)
            config_.max_cache_bytes = static_cast<size_t>(
                static_cast<double>(get_total_physical_memory()) * 0.90);
        {
            std::lock_guard<std::mutex> lock(adaptive_mutex_);
            adaptive_max_target_ = config_.decode_frame_ring_capacity - 2;
            adaptive_target_ = std::clamp<size_t>(
                config_.prefetch_count, 2, adaptive_max_target_);
        }

        ledger_.reserve(std::max(config_.prefetch_count, config_.output_queue_size) * 2);

        LOG_INFO("[PipelinedImageLoader] batch_size={}, prefetch={}, output_queue={}, ring_capacity={}, io_threads={}, cold_threads={}, 16bit_color={}",
                 config_.jpeg_batch_size,
                 config_.prefetch_count,
                 config_.output_queue_size,
                 config_.decode_frame_ring_capacity,
                 config_.io_threads,
                 config_.cold_process_threads,
                 config_.use_16bit_color);

        LOG_INFO("[PipelinedImageLoader] host compressed cache cap: {:.1f} GiB",
                 config_.max_cache_bytes / (1024.0 * 1024.0 * 1024.0));

        if (cuda_run) {
            const auto base = run_spill_base();
            std::error_code ec;
            std::filesystem::create_directories(base, ec);
            if (!ec) {
                run_spill_folder_ = base /
                                    ("ppl_run_" + std::to_string(current_process_id()) + "_" +
                                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
                std::filesystem::create_directories(run_spill_folder_, ec);
            }
            if (ec) {
                LOG_WARN("[PipelinedImageLoader] Run spill folder creation failed: {}", ec.message());
                run_spill_folder_.clear();
            } else {
                LOG_INFO("[PipelinedImageLoader] run spill folder: {}",
                         lfs::core::path_to_utf8(run_spill_folder_));
            }
        }

        running_ = true;
        const bool cuda_stage = attach_cuda_decode_stage();
        for (size_t i = 0; i < config_.io_threads; ++i) {
            io_threads_.emplace_back([this] { prefetch_thread_func(); });
        }
        if (cuda_stage) {
            start_cuda_decode_workers();
        } else {
            for (size_t i = 0; i < config_.cold_process_threads; ++i) {
                cold_process_threads_.emplace_back([this] { portable_process_thread_func(); });
            }
        }

        LOG_INFO("[PipelinedImageLoader] Started {} I/O, 1 GPU, {} cold threads",
                 config_.io_threads, config_.cold_process_threads);
    }

    PipelinedImageLoader::~PipelinedImageLoader() {
        shutdown();
    }

    void PipelinedImageLoader::shutdown() {
        if (!running_.exchange(false))
            return;

        LOG_INFO("[PipelinedImageLoader] Shutting down...");
        if (decoded_frame_ring_)
            decoded_frame_ring_->cancel();

        prefetch_queue_.signal_shutdown();
        hot_queue_.signal_shutdown();
        cold_queue_.signal_shutdown();
        output_queue_.signal_shutdown();

        for (auto& t : io_threads_) {
            if (t.joinable())
                t.join();
        }
        if (gpu_decode_thread_.joinable()) {
            gpu_decode_thread_.join();
        }
        for (auto& t : cold_process_threads_) {
            if (t.joinable())
                t.join();
        }

        // Queue and pairing entries may own tensors homed on decode_stream_.
        // Retire them while that stream is still valid; member destruction is
        // too late because shutdown destroys the stream first.
        clear();

        if (decode_queue_)
            decode_queue_->wait();
        for (const auto& queue : sidecar_queues_)
            queue->wait();
        decode_queue_.reset();
        sidecar_queues_.clear();
        release_cuda_decode_stage();

        LOG_INFO("[PipelinedImageLoader] Done: {} loaded, {} hits, {} misses",
                 stats_.total_images_loaded, stats_.hot_path_hits, stats_.cold_path_misses);
        {
            std::lock_guard<std::mutex> cache_lock(jpeg_cache_mutex_);
            LOG_INFO("[PipelinedImageLoader] compressed run cache: {} RAM entries, {:.1f} MiB RAM, {} spill entries, {:.1f} MiB spill",
                     jpeg_cache_.size(),
                     jpeg_cache_bytes_.load() / (1024.0 * 1024.0),
                     spill_cache_.size(),
                     spill_cache_bytes_ / (1024.0 * 1024.0));
        }
        cleanup_run_spill_directory();
    }

    void PipelinedImageLoader::prefetch(const std::vector<ImageRequest>& requests) {
        for (const auto& req : requests) {
            if (!running_.load(std::memory_order_acquire)) {
                return;
            }
            std::uint64_t accepted_generation = 0;
            {
                std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                if (!running_.load(std::memory_order_acquire)) {
                    return;
                }
                auto [it, inserted] = pending_pairs_.try_emplace(req.sequence_id);
                if (!inserted) {
                    continue;
                }
                it->second.primary_path = req.path;
                accepted_generation = loader_generation_.load(std::memory_order_relaxed);
                it->second.loader_generation = accepted_generation;
                it->second.mask_expected = req.mask_path.has_value() || req.extract_alpha_as_mask;
                it->second.depth_expected = req.depth_path.has_value();
                it->second.normal_expected = req.normal_path.has_value();
                accepted_sequences_.fetch_add(1, std::memory_order_relaxed);
                in_flight_.fetch_add(1, std::memory_order_acq_rel);
            }
            ImageRequest accepted_request = req;
            accepted_request.loader_generation = accepted_generation;
            (void)prefetch_queue_.push(std::move(accepted_request));
        }
    }

    void PipelinedImageLoader::prefetch(size_t sequence_id, const std::filesystem::path& path, const LoadParams& params) {
        if (!running_.load(std::memory_order_acquire)) {
            return;
        }
        ImageRequest request;
        request.sequence_id = sequence_id;
        request.loader_generation = loader_generation_.load(std::memory_order_relaxed);
        request.path = path;
        request.params = params;
        {
            std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
            if (!running_.load(std::memory_order_acquire)) {
                return;
            }
            const auto [it, inserted] = pending_pairs_.try_emplace(sequence_id);
            if (!inserted) {
                return;
            }
            it->second.primary_path = path;
            it->second.loader_generation = request.loader_generation;
            accepted_sequences_.fetch_add(1, std::memory_order_relaxed);
            in_flight_.fetch_add(1, std::memory_order_acq_rel);
        }
        (void)prefetch_queue_.push(std::move(request));
    }

    void PipelinedImageLoader::canonicalize(const std::vector<ImageRequest>& requests) {
        constexpr size_t CHUNK_SIZE = 32;
        for (size_t offset = 0; offset < requests.size(); offset += CHUNK_SIZE) {
            const size_t count = std::min(CHUNK_SIZE, requests.size() - offset);
            std::vector<ImageRequest> chunk;
            chunk.reserve(count);
            chunk.insert(chunk.end(), requests.begin() + static_cast<std::ptrdiff_t>(offset),
                         requests.begin() + static_cast<std::ptrdiff_t>(offset + count));
            prefetch(chunk);

            for (size_t i = 0; i < count; ++i) {
                auto completion = get_completion();
                if (!completion) {
                    throw std::runtime_error(legacy_message_from(completion.error()));
                }
                if (!completion->outcome) {
                    throw std::runtime_error(legacy_message_from(completion->outcome.error()));
                }
                // Destroying the completion here releases decoded GPU payloads;
                // only the final encoded run-cache/spill representation remains.
            }
        }
    }

    ReadyImage PipelinedImageLoader::get() {
        auto completion = get_completion();
        if (!completion) {
            throw std::runtime_error(legacy_message_from(completion.error()));
        }
        if (!completion->outcome) {
            throw std::runtime_error(legacy_message_from(completion->outcome.error()));
        }
        return std::move(*completion->outcome);
    }

    std::optional<ReadyImage> PipelinedImageLoader::try_get() {
        auto completion = try_get_completion();
        if (!completion) {
            return std::nullopt;
        }
        if (!completion->outcome) {
            throw std::runtime_error(legacy_message_from(completion->outcome.error()));
        }
        return std::move(*completion->outcome);
    }

    std::optional<ReadyImage> PipelinedImageLoader::try_get_for(std::chrono::milliseconds timeout) {
        auto completion = try_get_completion_for(timeout);
        if (!completion) {
            return std::nullopt;
        }
        if (!completion->outcome) {
            throw std::runtime_error(legacy_message_from(completion->outcome.error()));
        }
        return std::move(*completion->outcome);
    }

    std::optional<LoaderCompletion> PipelinedImageLoader::take_completion(
        const std::uint64_t sequence_id) {
        std::lock_guard<std::mutex> lock(ledger_mutex_);
        const auto it = ledger_.find(sequence_id);
        if (it == ledger_.end()) {
            return std::nullopt;
        }
        LoaderCompletion completion = std::move(it->second);
        ledger_.erase(it);
        if (completion.outcome) {
            release_output_ready_bytes(*completion.outcome);
        }
        subtract_clamped(in_flight_, 1);
        return completion;
    }

    lfs::Result<LoaderCompletion> PipelinedImageLoader::get_completion() {
        try {
            const auto sequence_id = output_queue_.pop();
            if (auto completion = take_completion(sequence_id)) {
                return std::move(*completion);
            }
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::IO,
                .detail = "PipelinedImageLoader completion token had no ledger entry",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        } catch (const std::runtime_error&) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Cancelled,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Warning,
                .detail = "PipelinedImageLoader shut down while waiting for a completion",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
    }

    std::optional<LoaderCompletion> PipelinedImageLoader::try_get_completion() {
        const auto sequence_id = output_queue_.try_pop();
        if (!sequence_id) {
            return std::nullopt;
        }
        return take_completion(*sequence_id);
    }

    std::optional<LoaderCompletion> PipelinedImageLoader::try_get_completion_for(
        const std::chrono::milliseconds timeout) {
        const auto sequence_id = output_queue_.try_pop_for(timeout);
        if (!sequence_id) {
            return std::nullopt;
        }
        return take_completion(*sequence_id);
    }

    size_t PipelinedImageLoader::ready_count() const {
        return output_queue_.size();
    }

    size_t PipelinedImageLoader::in_flight_count() const {
        return in_flight_.load();
    }

    size_t PipelinedImageLoader::adaptive_prefetch_target() const {
        std::lock_guard<std::mutex> lock(adaptive_mutex_);
        return adaptive_target_;
    }

    void PipelinedImageLoader::record_decode_latency(const double decode_ms) {
        if (decode_ms <= 0.0)
            return;
        constexpr double alpha = 0.2;
        std::lock_guard<std::mutex> lock(adaptive_mutex_);
        decode_latency_ema_ms_ = decode_latency_ema_ms_ <= 0.0
                                     ? decode_ms
                                     : (1.0 - alpha) * decode_latency_ema_ms_ + alpha * decode_ms;
    }

    void PipelinedImageLoader::observe_training_iteration(const double train_ms,
                                                          const double dl_wait_ms,
                                                          const std::size_t iter) {
        if (train_ms <= 0.0)
            return;
        constexpr double alpha = 0.2;
        std::lock_guard<std::mutex> lock(adaptive_mutex_);
        train_latency_ema_ms_ = train_latency_ema_ms_ <= 0.0
                                    ? train_ms
                                    : (1.0 - alpha) * train_latency_ema_ms_ + alpha * train_ms;
        adaptive_wait_sum_ms_ += std::max(0.0, dl_wait_ms);
        ++adaptive_wait_samples_;
        adaptive_occupancy_ = in_flight_.load(std::memory_order_relaxed);
        if (iter % 64 != 0)
            return;

        const double window_dl_wait_ms = adaptive_wait_samples_ > 0
                                             ? adaptive_wait_sum_ms_ /
                                                   static_cast<double>(adaptive_wait_samples_)
                                             : 0.0;
        adaptive_wait_sum_ms_ = 0.0;
        adaptive_wait_samples_ = 0;

        const auto recommended = std::clamp<size_t>(
            static_cast<size_t>(std::ceil(decode_latency_ema_ms_ /
                                          std::max(0.01, train_latency_ema_ms_))) +
                2,
            2,
            adaptive_max_target_);
        size_t next = adaptive_target_;
        if (window_dl_wait_ms > 0.05) {
            next = std::min(adaptive_max_target_, std::max(next + 1, recommended));
            adaptive_low_recommendation_windows_ = 0;
            adaptive_growth_cooldown_windows_ = 4;
        } else {
            if (adaptive_growth_cooldown_windows_ > 0) {
                --adaptive_growth_cooldown_windows_;
                adaptive_low_recommendation_windows_ = 0;
            } else if (recommended < adaptive_target_) {
                ++adaptive_low_recommendation_windows_;
                if (adaptive_low_recommendation_windows_ >= 4) {
                    next = std::min(adaptive_max_target_, std::max<size_t>(2, recommended + 1));
                    adaptive_low_recommendation_windows_ = 0;
                }
            } else {
                adaptive_low_recommendation_windows_ = 0;
            }
        }
        if (next != adaptive_target_) {
            adaptive_target_ = next;
            if (decoded_frame_ring_)
                decoded_frame_ring_->set_capacity(next + 2);
            LOG_DEBUG("[PipelinedImageLoader] adaptive prefetch target={} occupancy={} decode_ema={:.3f}ms train_ema={:.3f}ms",
                      adaptive_target_,
                      adaptive_occupancy_,
                      decode_latency_ema_ms_,
                      train_latency_ema_ms_);
        }
    }

    void PipelinedImageLoader::clear() {
        loader_generation_.fetch_add(1, std::memory_order_acq_rel);
        prefetch_queue_.clear();
        hot_queue_.clear();
        cold_queue_.clear();
        reconcile_ledger_on_shutdown();
        reset_pipeline_gpu_bytes();
        in_flight_ = 0;
    }

    void PipelinedImageLoader::reclaim_idle_decoded_frames() {
        if (decoded_frame_ring_)
            decoded_frame_ring_->reclaim_idle();
    }

    void PipelinedImageLoader::publish_loader_vram_gauges() const {
        const auto ring = decoded_frame_ring_;
        auto& profiler = lfs::diagnostics::VramProfiler::instance();
        if (!ring) {
            return;
        }
        profiler.setGauge("vram.audit.io.decoded_frame_ring.bytes",
                          static_cast<double>(ring->storage_bytes()));
    }

    PipelinedImageLoader::CacheStats PipelinedImageLoader::get_stats() const {
        CacheStats s;
        {
            // Snapshot each independently protected domain without nesting locks.
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            s = stats_;
        }
        {
            std::lock_guard<std::mutex> cache_lock(jpeg_cache_mutex_);
            s.jpeg_cache_entries = jpeg_cache_.size();
            s.jpeg_cache_bytes = jpeg_cache_bytes_.load();
            s.spill_cache_entries = spill_cache_.size();
            s.spill_cache_bytes = spill_cache_bytes_;
        }
        {
            std::lock_guard<std::mutex> pairs_lock(pending_pairs_mutex_);
            s.pending_pairs_count = pending_pairs_.size();
        }
        {
            std::lock_guard<std::mutex> tally_lock(sidecar_tally_mutex_);
            s.aggregate_sidecar_tally = sidecar_tally_;
        }
        s.accepted_sequences = accepted_sequences_.load(std::memory_order_relaxed);
        s.succeeded_sequences = succeeded_sequences_.load(std::memory_order_relaxed);
        s.failed_sequences = failed_sequences_.load(std::memory_order_relaxed);
        s.cancelled_sequences = cancelled_sequences_.load(std::memory_order_relaxed);
        s.prefetch_queue_size = prefetch_queue_.size();
        s.hot_queue_size = hot_queue_.size();
        s.cold_queue_size = cold_queue_.size();
        s.output_queue_size = output_queue_.size();
        const auto gpu_stats = get_gpu_memory_stats();
        s.output_image_bytes = gpu_stats.output_image_bytes;
        s.output_mask_bytes = gpu_stats.output_mask_bytes;
        s.output_depth_bytes = gpu_stats.output_depth_bytes;
        s.output_normal_bytes = gpu_stats.output_normal_bytes;
        s.pending_image_bytes = gpu_stats.pending_image_bytes;
        s.pending_mask_bytes = gpu_stats.pending_mask_bytes;
        s.pending_depth_bytes = gpu_stats.pending_depth_bytes;
        s.pending_normal_bytes = gpu_stats.pending_normal_bytes;
        publish_loader_vram_gauges();
        return s;
    }

    std::filesystem::path PipelinedImageLoader::run_spill_directory() const {
        return run_spill_folder_;
    }

    PipelinedImageLoader::GpuMemoryStats PipelinedImageLoader::get_gpu_memory_stats() const {
        return {
            .output_image_bytes = output_image_bytes_.load(std::memory_order_acquire),
            .output_mask_bytes = output_mask_bytes_.load(std::memory_order_acquire),
            .output_depth_bytes = output_depth_bytes_.load(std::memory_order_acquire),
            .output_normal_bytes = output_normal_bytes_.load(std::memory_order_acquire),
            .pending_image_bytes = pending_image_bytes_.load(std::memory_order_acquire),
            .pending_mask_bytes = pending_mask_bytes_.load(std::memory_order_acquire),
            .pending_depth_bytes = pending_depth_bytes_.load(std::memory_order_acquire),
            .pending_normal_bytes = pending_normal_bytes_.load(std::memory_order_acquire),
        };
    }

    std::string PipelinedImageLoader::make_cache_key(const std::filesystem::path& path, const LoadParams& params) const {
        auto key = lfs::core::path_to_utf8(path) + ":rf" + std::to_string(params.resize_factor) + "_mw" + std::to_string(params.max_width);
        if (params.undistort)
            key += "_ud";
        if (config_.use_16bit_color)
            key += "_16b";
        return key;
    }

    std::string PipelinedImageLoader::make_mask_cache_key(
        const std::filesystem::path& path,
        const LoadParams& params) const {
        auto key = lfs::core::path_to_utf8(path) +
                   ":mask_rf" + std::to_string(params.resize_factor) +
                   "_mw" + std::to_string(params.max_width);
        if (params.undistort)
            key += "_ud";
        return key;
    }

    bool PipelinedImageLoader::is_jpeg_data(const std::vector<uint8_t>& data) const {
        if (data.size() < 3)
            return false;
        return data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;
    }

    std::vector<uint8_t> PipelinedImageLoader::read_file(const std::filesystem::path& path) const {
        std::ifstream file;
        if (!lfs::core::open_file_for_read(path, std::ios::binary | std::ios::ate, file))
            throw std::runtime_error("Failed to open: " + lfs::core::path_to_utf8(path));

        const auto size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read: " + lfs::core::path_to_utf8(path));
        }
        return buffer;
    }

    std::shared_ptr<std::vector<uint8_t>> PipelinedImageLoader::load_cached_jpeg_blob(
        const std::string& cache_key) {
        return get_from_jpeg_cache(cache_key);
    }

    std::pair<int, int> PipelinedImageLoader::sidecar_target_size(
        const PrefetchedImage& item,
        const int src_w,
        const int src_h) const {
        if (!item.is_mask && item.aux_target_width > 0 && item.aux_target_height > 0)
            return {item.aux_target_width, item.aux_target_height};
        int target_w = src_w;
        int target_h = src_h;
        if (item.params.resize_factor > 1) {
            target_w = std::max(1, target_w / item.params.resize_factor);
            target_h = std::max(1, target_h / item.params.resize_factor);
        }
        const int max_w = item.params.max_width;
        if (max_w > 0 && (target_w > max_w || target_h > max_w)) {
            if (target_w > target_h) {
                target_h = std::max(1, max_w * target_h / target_w);
                target_w = max_w;
            } else {
                target_w = std::max(1, max_w * target_w / target_h);
                target_h = max_w;
            }
        }
        return {target_w, target_h};
    }

    std::shared_ptr<std::vector<uint8_t>> PipelinedImageLoader::get_from_jpeg_cache(const std::string& cache_key) {
        relieve_host_memory_pressure();
        std::filesystem::path spill_path;
        {
            std::lock_guard<std::mutex> lock(jpeg_cache_mutex_);
            if (const auto it = jpeg_cache_.find(cache_key); it != jpeg_cache_.end()) {
                it->second.last_access = std::chrono::steady_clock::now();
                return it->second.data;
            }
            if (const auto it = spill_cache_.find(cache_key); it != spill_cache_.end()) {
                it->second.last_access = std::chrono::steady_clock::now();
                spill_path = it->second.path;
            }
        }

        if (spill_path.empty())
            return nullptr;
        try {
            return std::make_shared<std::vector<uint8_t>>(read_file(spill_path));
        } catch (const std::exception& e) {
            LOG_WARN("[PipelinedImageLoader] Run spill read failed for {}: {}",
                     lfs::core::path_to_utf8(spill_path), e.what());
            std::lock_guard<std::mutex> lock(jpeg_cache_mutex_);
            if (const auto it = spill_cache_.find(cache_key); it != spill_cache_.end()) {
                spill_cache_bytes_ -= it->second.size_bytes;
                std::error_code ec;
                std::filesystem::remove(it->second.path, ec);
                spill_cache_.erase(it);
            }
            return nullptr;
        }
    }

    void PipelinedImageLoader::put_in_jpeg_cache(const std::string& cache_key, std::shared_ptr<std::vector<uint8_t>> data) {
        if (!data)
            return;
        const size_t size = data->size();
        {
            std::lock_guard<std::mutex> lock(jpeg_cache_mutex_);
            if (const auto it = jpeg_cache_.find(cache_key); it != jpeg_cache_.end()) {
                jpeg_cache_bytes_ -= it->second.size_bytes;
                jpeg_cache_.erase(it);
            }
            if (const auto it = spill_cache_.find(cache_key); it != spill_cache_.end()) {
                spill_cache_bytes_ -= it->second.size_bytes;
                std::error_code ec;
                std::filesystem::remove(it->second.path, ec);
                spill_cache_.erase(it);
            }
            evict_jpeg_cache_if_needed(size);
            if (size > config_.max_cache_bytes) {
                spill_cache_entry_locked(cache_key, data);
            } else {
                jpeg_cache_[cache_key] = JpegCacheEntry{data, std::chrono::steady_clock::now(), size};
                jpeg_cache_bytes_ += size;
            }
        }
    }

    void PipelinedImageLoader::put_in_jpeg_cache(const std::string& cache_key, std::vector<uint8_t>&& data) {
        put_in_jpeg_cache(cache_key, std::make_shared<std::vector<uint8_t>>(std::move(data)));
    }

    void PipelinedImageLoader::invalidate_cache_entry(const std::string& cache_key) {
        std::lock_guard<std::mutex> lock(jpeg_cache_mutex_);
        if (const auto it = jpeg_cache_.find(cache_key); it != jpeg_cache_.end()) {
            jpeg_cache_bytes_ -= it->second.size_bytes;
            jpeg_cache_.erase(it);
        }
        if (const auto it = spill_cache_.find(cache_key); it != spill_cache_.end()) {
            spill_cache_bytes_ -= it->second.size_bytes;
            std::error_code ec;
            std::filesystem::remove(it->second.path, ec);
            spill_cache_.erase(it);
        }
    }

    void PipelinedImageLoader::evict_jpeg_cache_if_needed(size_t required_bytes) {
        size_t target = config_.max_cache_bytes;
        const size_t available = get_available_physical_memory();
        const size_t min_free = static_cast<size_t>(get_total_physical_memory() * HOST_FREE_RAM_EVICTION_RATIO);

        if (available < min_free + required_bytes) {
            target = std::min(target, jpeg_cache_bytes_.load() / 2);
        }

        spill_least_recent_until_locked(target > required_bytes ? target - required_bytes : 0);
    }

    size_t PipelinedImageLoader::spill_least_recent_until_locked(const size_t cached_bytes_target) {
        size_t released = 0;
        while (jpeg_cache_bytes_ > cached_bytes_target && !jpeg_cache_.empty()) {
            const auto oldest = std::ranges::min_element(
                jpeg_cache_, {}, [](const auto& entry) { return entry.second.last_access; });
            const auto key = oldest->first;
            const auto data = oldest->second.data;
            jpeg_cache_bytes_ -= oldest->second.size_bytes;
            released += oldest->second.size_bytes;
            jpeg_cache_.erase(oldest);
            spill_cache_entry_locked(key, data);
        }
        return released;
    }

    size_t PipelinedImageLoader::release_host_cache(const size_t bytes) {
        if (config_.backend != lfs::core::GpuBackend::CUDA)
            return release_portable_host_cache(bytes);
        size_t released = 0;
        {
            std::lock_guard<std::mutex> lock(jpeg_cache_mutex_);
            const size_t cached = jpeg_cache_bytes_.load();
            released = spill_least_recent_until_locked(cached > bytes ? cached - bytes : 0);
        }
        if (released > 0) {
            return_freed_heap_to_os();
            LOG_INFO("[PipelinedImageLoader] Moved {:.1f} MiB of cached images from RAM to the run spill "
                     "to free host memory",
                     static_cast<double>(released) / (1024.0 * 1024.0));
        }
        return released;
    }

    void PipelinedImageLoader::relieve_host_memory_pressure() {
        const std::int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count();
        std::int64_t due = next_host_memory_check_ns_.load(std::memory_order_relaxed);
        if (now < due ||
            !next_host_memory_check_ns_.compare_exchange_strong(
                due, now + std::chrono::nanoseconds(HOST_MEMORY_CHECK_INTERVAL).count(),
                std::memory_order_relaxed)) {
            return;
        }
        if (jpeg_cache_bytes_.load(std::memory_order_relaxed) == 0)
            return;

        const size_t total = get_total_physical_memory();
        const auto min_free = static_cast<size_t>(total * HOST_FREE_RAM_EVICTION_RATIO);
        const size_t available = get_available_physical_memory();
        if (available >= min_free)
            return;
        release_host_cache(min_free - available +
                           static_cast<size_t>(total * HOST_FREE_RAM_RELIEF_HYSTERESIS_RATIO));
    }

    void PipelinedImageLoader::spill_cache_entry_locked(
        const std::string& cache_key,
        const std::shared_ptr<std::vector<uint8_t>>& data) {
        if (!data || run_spill_folder_.empty())
            return;

        const auto path = run_spill_folder_ /
                          ("blob_" + std::to_string(++spill_sequence_) + ".bin");
        std::ofstream file;
        if (!lfs::core::open_file_for_write(path, std::ios::binary, file)) {
            LOG_WARN("[PipelinedImageLoader] Failed to open run spill: {}", lfs::core::path_to_utf8(path));
            return;
        }
        file.write(reinterpret_cast<const char*>(data->data()), static_cast<std::streamsize>(data->size()));
        if (!file.good()) {
            LOG_WARN("[PipelinedImageLoader] Failed to write run spill: {}", lfs::core::path_to_utf8(path));
            return;
        }
        spill_cache_[cache_key] = SpillCacheEntry{path, std::chrono::steady_clock::now(), data->size()};
        spill_cache_bytes_ += data->size();
    }

    void PipelinedImageLoader::cleanup_run_spill_directory() {
        if (run_spill_folder_.empty())
            return;
        std::error_code ec;
        std::filesystem::remove_all(run_spill_folder_, ec);
        if (ec) {
            LOG_WARN("[PipelinedImageLoader] Failed to remove run spill {}: {}",
                     lfs::core::path_to_utf8(run_spill_folder_), ec.message());
        }
        run_spill_folder_.clear();
        std::lock_guard<std::mutex> lock(jpeg_cache_mutex_);
        spill_cache_.clear();
        spill_cache_bytes_ = 0;
    }

    void PipelinedImageLoader::cleanup_stale_run_spill_directories() {
        remove_stale_run_spill_directories();
    }

    void PipelinedImageLoader::add_output_ready_bytes(const ReadyImage& ready) {
        output_image_bytes_.fetch_add(tensor_reserved_bytes(ready.tensor), std::memory_order_acq_rel);
        if (ready.mask) {
            output_mask_bytes_.fetch_add(tensor_reserved_bytes(*ready.mask), std::memory_order_acq_rel);
        }
        if (ready.depth) {
            output_depth_bytes_.fetch_add(tensor_reserved_bytes(*ready.depth), std::memory_order_acq_rel);
        }
        if (ready.normal) {
            output_normal_bytes_.fetch_add(tensor_reserved_bytes(*ready.normal), std::memory_order_acq_rel);
        }
    }

    void PipelinedImageLoader::release_output_ready_bytes(const ReadyImage& ready) {
        subtract_clamped(output_image_bytes_, tensor_reserved_bytes(ready.tensor));
        if (ready.mask) {
            subtract_clamped(output_mask_bytes_, tensor_reserved_bytes(*ready.mask));
        }
        if (ready.depth) {
            subtract_clamped(output_depth_bytes_, tensor_reserved_bytes(*ready.depth));
        }
        if (ready.normal) {
            subtract_clamped(output_normal_bytes_, tensor_reserved_bytes(*ready.normal));
        }
    }

    void PipelinedImageLoader::publish_image_failure(
        const size_t sequence_id,
        const std::uint64_t loader_generation,
        const std::filesystem::path& path,
        std::string message,
        SidecarTally tally) {
        bool accepted = false;
        {
            std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
            if (auto it = pending_pairs_.find(sequence_id);
                it != pending_pairs_.end() &&
                it->second.loader_generation == loader_generation) {
                erase_pending_pair_locked(it);
                accepted = true;
            }
        }
        if (!accepted) {
            return;
        }

        std::string full_message = "Failed to load training image '" +
                                   lfs::core::path_to_utf8(path) + "': " + std::move(message);
        lfs::Error error = lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::DataLoss,
            .domain = lfs::ErrorDomain::IO,
            .user_message = full_message,
            .detail = full_message,
            .detection = LFS_SOURCE_SITE_CURRENT(),
            .fields = lfs::SmallFields{}.add("path", lfs::core::path_to_utf8(path)),
        });

        lfs::core::ErrorReporter::get().report(error, lfs::core::ReportChannel::OwnerLog);
        settle_completion(sequence_id, loader_generation,
                          lfs::Result<ReadyImage>(std::move(error)), tally);
    }

    void PipelinedImageLoader::fail_sidecar_locked(
        const size_t sequence_id,
        const std::uint64_t loader_generation,
        const SidecarKind kind,
        const std::filesystem::path& path,
        std::string message,
        std::unique_lock<std::mutex>& lock) {
        auto it = pending_pairs_.find(sequence_id);
        if (it == pending_pairs_.end() || it->second.loader_generation != loader_generation) {
            return;
        }

        const SidecarPolicy policy = kind == SidecarKind::Mask    ? config_.mask_policy
                                     : kind == SidecarKind::Depth ? config_.depth_policy
                                                                  : config_.normal_policy;
        auto& pair = it->second;
        if (policy == SidecarPolicy::Required) {
            const auto count = [](const bool requested, const bool delivered) {
                return SidecarCount{
                    .requested = requested ? 1u : 0u,
                    .delivered = requested && delivered ? 1u : 0u,
                    .failed = requested && !delivered ? 1u : 0u,
                };
            };
            const SidecarTally tally{
                .mask = count(pair.mask_expected || pair.mask_failed, pair.mask.has_value()),
                .depth = count(pair.depth_expected || pair.depth_failed, pair.depth.has_value()),
                .normal = count(pair.normal_expected || pair.normal_failed, pair.normal.has_value()),
            };
            const auto primary_path = pair.primary_path;
            lock.unlock();
            publish_image_failure(
                sequence_id, loader_generation, primary_path,
                "required sidecar '" + lfs::core::path_to_utf8(path) + "' failed: " +
                    std::move(message),
                tally);
            return;
        }

        switch (kind) {
        case SidecarKind::Mask:
            pair.mask_failed = true;
            pair.mask_expected = false;
            break;
        case SidecarKind::Depth:
            pair.depth_failed = true;
            pair.depth_expected = false;
            break;
        case SidecarKind::Normal:
            pair.normal_failed = true;
            pair.normal_expected = false;
            break;
        }
        try_push_ready_locked(sequence_id, it, lock);
    }

    void PipelinedImageLoader::settle_completion(
        const std::uint64_t sequence_id,
        const std::uint64_t loader_generation,
        lfs::Result<ReadyImage> outcome,
        const SidecarTally tally) {
        bool inserted = false;
        {
            std::lock_guard<std::mutex> lock(ledger_mutex_);
            if (loader_generation != loader_generation_.load(std::memory_order_acquire)) {
                if (outcome) {
                    destroy_sidecar_ready_event(outcome->depth_ready);
                    destroy_sidecar_ready_event(outcome->normal_ready);
                }
                return;
            }
            if (outcome) {
                add_output_ready_bytes(*outcome);
            }
            const auto [it, did_insert] = ledger_.emplace(
                sequence_id, LoaderCompletion{sequence_id, std::move(outcome), tally});
            inserted = did_insert;
            LFS_DEBUG_ASSERT_MSG(did_insert,
                                 "PipelinedImageLoader settled one sequence more than once");
            if (!did_insert) {
                return;
            }
            if (it->second.outcome) {
                succeeded_sequences_.fetch_add(1, std::memory_order_relaxed);
            } else if (it->second.outcome.error().code() == lfs::ErrorCode::Cancelled) {
                cancelled_sequences_.fetch_add(1, std::memory_order_relaxed);
            } else {
                failed_sequences_.fetch_add(1, std::memory_order_relaxed);
            }
            accumulate_sidecar_tally(tally);
        }
        if (inserted &&
            loader_generation == loader_generation_.load(std::memory_order_acquire)) {
            (void)output_queue_.push(sequence_id);
        }
        ledger_cv_.notify_all();
    }

    void PipelinedImageLoader::accumulate_sidecar_tally(const SidecarTally& tally) {
        const auto add = [](SidecarCount& total, const SidecarCount& delta) {
            total.requested += delta.requested;
            total.delivered += delta.delivered;
            total.failed += delta.failed;
        };
        std::lock_guard<std::mutex> lock(sidecar_tally_mutex_);
        add(sidecar_tally_.mask, tally.mask);
        add(sidecar_tally_.depth, tally.depth);
        add(sidecar_tally_.normal, tally.normal);
    }

    void PipelinedImageLoader::erase_pending_pair_locked(
        PendingPairIterator it) {
        if (it == pending_pairs_.end()) {
            return;
        }

        subtract_clamped(pending_image_bytes_, it->second.image_bytes);
        subtract_clamped(pending_mask_bytes_, it->second.mask_bytes);
        subtract_clamped(pending_depth_bytes_, it->second.depth_bytes);
        subtract_clamped(pending_normal_bytes_, it->second.normal_bytes);
        destroy_sidecar_ready_event(it->second.depth_ready);
        destroy_sidecar_ready_event(it->second.normal_ready);
        pending_pairs_.erase(it);
    }

    void PipelinedImageLoader::destroy_sidecar_ready_event(
        std::optional<lfs::core::TensorFence>& fence) {
        fence.reset();
    }

    void PipelinedImageLoader::reconcile_ledger_on_shutdown() {
        {
            std::lock_guard<std::mutex> pairs_lock(pending_pairs_mutex_);
            std::lock_guard<std::mutex> ledger_lock(ledger_mutex_);
            while (!pending_pairs_.empty()) {
                auto it = pending_pairs_.begin();
                const auto sequence_id = it->first;
                if (!ledger_.contains(sequence_id)) {
                    lfs::Error cancelled = lfs::make_error(lfs::ErrorInit{
                        .code = lfs::ErrorCode::Cancelled,
                        .domain = lfs::ErrorDomain::IO,
                        .severity = lfs::Severity::Warning,
                        .detail = "PipelinedImageLoader shut down before this request completed",
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                    });
                    ledger_.emplace(
                        sequence_id,
                        LoaderCompletion{sequence_id,
                                         lfs::Result<ReadyImage>(std::move(cancelled)),
                                         {}});
                }
                erase_pending_pair_locked(it);
            }
        }

        {
            std::lock_guard<std::mutex> lock(ledger_mutex_);
            for (auto& [sequence_id, completion] : ledger_) {
                (void)sequence_id;
                if (completion.outcome) {
                    release_output_ready_bytes(*completion.outcome);
                    destroy_sidecar_ready_event(completion.outcome->depth_ready);
                    destroy_sidecar_ready_event(completion.outcome->normal_ready);
                }
            }
            ledger_.clear();
        }
        output_queue_.clear();

        const std::uint64_t accepted = accepted_sequences_.load(std::memory_order_relaxed);
        const std::uint64_t succeeded = succeeded_sequences_.load(std::memory_order_relaxed);
        const std::uint64_t failed = failed_sequences_.load(std::memory_order_relaxed);
        const std::uint64_t previously_cancelled =
            cancelled_sequences_.load(std::memory_order_relaxed);
        const std::uint64_t unresolved =
            accepted > succeeded + failed + previously_cancelled
                ? accepted - succeeded - failed - previously_cancelled
                : 0;
        if (unresolved > 0) {
            cancelled_sequences_.fetch_add(unresolved, std::memory_order_relaxed);
        }
        const std::uint64_t cancelled =
            cancelled_sequences_.load(std::memory_order_relaxed);
        LFS_DEBUG_ASSERT_MSG(accepted == succeeded + failed + cancelled,
                             "PipelinedImageLoader shutdown: accepted != succeeded+failed+cancelled");

        SidecarTally sidecars;
        {
            std::lock_guard<std::mutex> lock(sidecar_tally_mutex_);
            sidecars = sidecar_tally_;
        }
        LOG_INFO("[PipelinedImageLoader] shutdown reconciliation: accepted={} succeeded={} failed={} cancelled={} "
                 "mask={}/{}/{} depth={}/{}/{} normal={}/{}/{}",
                 accepted, succeeded, failed, cancelled,
                 sidecars.mask.requested, sidecars.mask.delivered, sidecars.mask.failed,
                 sidecars.depth.requested, sidecars.depth.delivered, sidecars.depth.failed,
                 sidecars.normal.requested, sidecars.normal.delivered, sidecars.normal.failed);
    }

    void PipelinedImageLoader::reset_pipeline_gpu_bytes() {
        output_image_bytes_.store(0, std::memory_order_release);
        output_mask_bytes_.store(0, std::memory_order_release);
        output_depth_bytes_.store(0, std::memory_order_release);
        output_normal_bytes_.store(0, std::memory_order_release);
        pending_image_bytes_.store(0, std::memory_order_release);
        pending_mask_bytes_.store(0, std::memory_order_release);
        pending_depth_bytes_.store(0, std::memory_order_release);
        pending_normal_bytes_.store(0, std::memory_order_release);
    }

    void PipelinedImageLoader::try_push_ready_locked(
        const size_t sequence_id,
        PendingPairIterator it,
        std::unique_lock<std::mutex>& pending_lock) {
        auto& pair = it->second;
        const bool image_ready = pair.image.has_value();
        const bool mask_has_value = pair.mask.has_value();
        const bool depth_has_value = pair.depth.has_value();
        const bool normal_has_value = pair.normal.has_value();
        const bool mask_ready = !pair.mask_expected || mask_has_value;
        const bool depth_ready = !pair.depth_expected || depth_has_value;
        const bool normal_ready = !pair.normal_expected || normal_has_value;

        if (!image_ready || !mask_ready || !depth_ready || !normal_ready) {
            return;
        }

        ReadyImage ready{
            .sequence_id = sequence_id,
            .tensor = std::move(*pair.image),
            .mask = mask_has_value ? std::optional(std::move(*pair.mask)) : std::nullopt,
            .image_ready = std::move(pair.image_ready),
            .mask_ready = std::move(pair.mask_ready),
            .depth = depth_has_value ? std::optional(std::move(*pair.depth)) : std::nullopt,
            .normal = normal_has_value ? std::optional(std::move(*pair.normal)) : std::nullopt,
            .depth_ready = std::move(pair.depth_ready),
            .normal_ready = std::move(pair.normal_ready),
            .decoded_frame_leases = std::move(pair.decoded_frame_leases),
            .error = {},
        };
        const auto loader_generation = pair.loader_generation;
        const SidecarTally tally{
            .mask = {.requested = (pair.mask_expected || pair.mask_failed) ? 1u : 0u,
                     .delivered = mask_has_value ? 1u : 0u,
                     .failed = pair.mask_failed ? 1u : 0u},
            .depth = {.requested = (pair.depth_expected || pair.depth_failed) ? 1u : 0u,
                      .delivered = depth_has_value ? 1u : 0u,
                      .failed = pair.depth_failed ? 1u : 0u},
            .normal = {.requested = (pair.normal_expected || pair.normal_failed) ? 1u : 0u,
                       .delivered = normal_has_value ? 1u : 0u,
                       .failed = pair.normal_failed ? 1u : 0u},
        };
        erase_pending_pair_locked(it);
        pending_lock.unlock();

        const auto failed_sidecars = tally.mask.failed + tally.depth.failed + tally.normal.failed;
        if (failed_sidecars > 0) {
            const std::string detail =
                "PipelinedImageLoader optional sidecar degradation: mask_failed=" +
                std::to_string(tally.mask.failed) + " depth_failed=" +
                std::to_string(tally.depth.failed) + " normal_failed=" +
                std::to_string(tally.normal.failed);
            const lfs::Error degradation = lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::DataLoss,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Warning,
                .user_message = "Training image loaded without one or more optional sidecars",
                .detail = detail,
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = lfs::SmallFields{}.add(
                    "sequence", static_cast<std::uint64_t>(sequence_id)),
            });
            lfs::core::ErrorReporter::get().report(
                degradation, lfs::core::ReportChannel::OwnerLog);
        }

        settle_completion(sequence_id, loader_generation,
                          lfs::Result<ReadyImage>(std::move(ready)), tally);

        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        ++stats_.total_images_loaded;
        if (mask_has_value) {
            ++stats_.masks_loaded;
        }
        if (depth_has_value) {
            ++stats_.depths_loaded;
        }
        if (normal_has_value) {
            ++stats_.normals_loaded;
        }
    }

    std::string PipelinedImageLoader::make_sidecar_key(
        const PrefetchedImage& item,
        const SidecarCacheFormat kind) const {
        std::ostringstream key;
        key << lfs::core::path_to_utf8(item.path)
            << (kind == SidecarCacheFormat::Depth ? ":depth" : ":normal")
            << ":rf" << item.params.resize_factor
            << ":mw" << item.params.max_width
            << ":tw" << item.aux_target_width
            << ":th" << item.aux_target_height;
        if (item.undistort) {
            const auto& u = *item.undistort;
            key << ":ud"
                << ":src" << u.src_width << "x" << u.src_height
                << ":dst" << u.dst_width << "x" << u.dst_height
                << ":model" << static_cast<int>(u.model_type)
                << std::setprecision(9)
                << ":fx" << u.src_fx << ":fy" << u.src_fy
                << ":cx" << u.src_cx << ":cy" << u.src_cy
                << ":dfx" << u.dst_fx << ":dfy" << u.dst_fy
                << ":dcx" << u.dst_cx << ":dcy" << u.dst_cy
                << ":nd" << u.num_distortion;
            for (int i = 0; i < u.num_distortion; ++i) {
                key << ":d" << i << "=" << u.distortion[i];
            }
        }
        if (kind == SidecarCacheFormat::Normal) {
            key << ":srgb" << item.normal_srgb
                << ":flip" << item.normal_flip_yz
                << ":w2c" << item.normal_transform_world_to_camera
                << std::setprecision(9);
            if (item.normal_transform_world_to_camera) {
                for (const float v : item.normal_world_to_camera) {
                    key << ":" << v;
                }
            }
        }
        return key.str();
    }

    void PipelinedImageLoader::try_complete_pair(
        size_t sequence_id,
        const std::uint64_t loader_generation,
        std::optional<lfs::core::Tensor> image,
        std::optional<lfs::core::Tensor> mask,
        std::optional<lfs::core::Tensor> depth,
        std::optional<lfs::core::Tensor> normal,
        std::optional<lfs::core::TensorFence> sidecar_ready,
        std::shared_ptr<void> decoded_frame_lease) {

        // Capture each producer before publishing. Waiting belongs to the consumer,
        // otherwise a slow mask stalls subsequent work on the shared decode queue.
        // VulkanRecorderRegistry submits foreign producers and tracks access barriers;
        // Metal serializes dispatches through its shared context encoder.
        std::optional<lfs::core::TensorFence> image_ready, mask_ready;
        const bool cuda_run = config_.backend == lfs::core::GpuBackend::CUDA;
        if (image && cuda_run) {
            image_ready.emplace(lfs::core::GpuBackend::CUDA);
            image_ready->record(image->stream());
        }
        if (mask && cuda_run) {
            mask_ready.emplace(lfs::core::GpuBackend::CUDA);
            mask_ready->record(mask->stream());
        }
        std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
        auto it = pending_pairs_.find(sequence_id);
        if (it == pending_pairs_.end() || it->second.loader_generation != loader_generation) {
            return;
        }
        auto& pair = it->second;

        if (decoded_frame_lease) {
            pair.decoded_frame_leases.push_back(std::move(decoded_frame_lease));
        }

        if (image) {
            subtract_clamped(pending_image_bytes_, pair.image_bytes);
            pair.image = std::move(*image);
            pair.image_ready = std::move(image_ready);
            pair.image_bytes = tensor_reserved_bytes(*pair.image);
            pending_image_bytes_.fetch_add(pair.image_bytes, std::memory_order_acq_rel);
        }
        if (mask) {
            subtract_clamped(pending_mask_bytes_, pair.mask_bytes);
            pair.mask = std::move(*mask);
            pair.mask_ready = std::move(mask_ready);
            pair.mask_bytes = tensor_reserved_bytes(*pair.mask);
            pending_mask_bytes_.fetch_add(pair.mask_bytes, std::memory_order_acq_rel);
        }
        if (depth) {
            subtract_clamped(pending_depth_bytes_, pair.depth_bytes);
            pair.depth = std::move(*depth);
            pair.depth_bytes = tensor_reserved_bytes(*pair.depth);
            pending_depth_bytes_.fetch_add(pair.depth_bytes, std::memory_order_acq_rel);
        }
        if (normal) {
            subtract_clamped(pending_normal_bytes_, pair.normal_bytes);
            pair.normal = std::move(*normal);
            pair.normal_bytes = tensor_reserved_bytes(*pair.normal);
            pending_normal_bytes_.fetch_add(pair.normal_bytes, std::memory_order_acq_rel);
        }
        if (sidecar_ready) {
            auto& slot = depth ? pair.depth_ready : pair.normal_ready;
            slot = std::move(sidecar_ready);
        }

        try_push_ready_locked(sequence_id, it, lock);
    }

    void PipelinedImageLoader::prefetch_thread_func() {
        const lfs::core::GpuBackendScope backend(config_.backend);
        const bool cuda_run = config_.backend == lfs::core::GpuBackend::CUDA;
        while (running_) {
            ImageRequest request;
            try {
                request = prefetch_queue_.pop();
            } catch (const std::runtime_error&) {
                break;
            }

            auto fail_image_request = [&](std::string message) {
                publish_image_failure(request.sequence_id, request.loader_generation,
                                      request.path, std::move(message));
            };

            if (cuda_run && !decode_queue_) {
                fail_image_request("CUDA image processing is unavailable");
                continue;
            }

            auto enqueue_depth_request = [&] {
                if (!request.depth_path) {
                    return;
                }
                PrefetchedImage depth_result;
                depth_result.sequence_id = request.sequence_id;
                depth_result.loader_generation = request.loader_generation;
                depth_result.path = *request.depth_path;
                depth_result.params = request.params;
                depth_result.is_depth = true;
                depth_result.needs_processing = true;
                depth_result.undistort = request.undistort;
                depth_result.aux_target_width = request.aux_target_width;
                depth_result.aux_target_height = request.aux_target_height;
                if (!cuda_run) {
                    cold_queue_.push(std::move(depth_result));
                    return;
                }
                depth_result.cache_key = make_sidecar_key(depth_result, SidecarCacheFormat::Depth);
                if (auto cached = load_cached_jpeg_blob(depth_result.cache_key)) {
                    depth_result.jpeg_data = std::move(cached);
                    depth_result.is_cache_hit = true;
                    hot_queue_.push(std::move(depth_result));
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    ++stats_.hot_path_hits;
                    return;
                }
                if (!jpeg2k_cache_available_.load(std::memory_order_relaxed)) {
                    cold_queue_.push(std::move(depth_result));
                    return;
                }
                if (!is_regular_file_no_throw(*request.depth_path)) {
                    LOG_DEBUG("[PipelinedImageLoader] Skipping missing depth {}", lfs::core::path_to_utf8(*request.depth_path));
                    std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                    fail_sidecar_locked(request.sequence_id, request.loader_generation,
                                        SidecarKind::Depth,
                                        *request.depth_path, "file does not exist or is not a regular file", lock);
                    return;
                }
                cold_queue_.push(std::move(depth_result));
            };

            auto enqueue_normal_request = [&] {
                if (!request.normal_path) {
                    return;
                }
                PrefetchedImage normal_result;
                normal_result.sequence_id = request.sequence_id;
                normal_result.loader_generation = request.loader_generation;
                normal_result.path = *request.normal_path;
                normal_result.params = request.params;
                normal_result.is_normal = true;
                normal_result.normal_flip_yz = request.normal_flip_yz;
                normal_result.normal_srgb = request.normal_srgb;
                normal_result.normal_transform_world_to_camera = request.normal_transform_world_to_camera;
                normal_result.normal_world_to_camera = request.normal_world_to_camera;
                normal_result.needs_processing = true;
                normal_result.undistort = request.undistort;
                normal_result.aux_target_width = request.aux_target_width;
                normal_result.aux_target_height = request.aux_target_height;
                if (!cuda_run) {
                    cold_queue_.push(std::move(normal_result));
                    return;
                }
                normal_result.cache_key = make_sidecar_key(normal_result, SidecarCacheFormat::Normal);
                if (auto cached = load_cached_jpeg_blob(normal_result.cache_key)) {
                    normal_result.jpeg_data = std::move(cached);
                    normal_result.is_cache_hit = true;
                    hot_queue_.push(std::move(normal_result));
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    ++stats_.hot_path_hits;
                    return;
                }
                if (!jpeg2k_cache_available_.load(std::memory_order_relaxed)) {
                    cold_queue_.push(std::move(normal_result));
                    return;
                }
                if (!is_regular_file_no_throw(*request.normal_path)) {
                    LOG_DEBUG("[PipelinedImageLoader] Skipping missing normal {}", lfs::core::path_to_utf8(*request.normal_path));
                    std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                    fail_sidecar_locked(request.sequence_id, request.loader_generation,
                                        SidecarKind::Normal,
                                        *request.normal_path, "file does not exist or is not a regular file", lock);
                    return;
                }
                cold_queue_.push(std::move(normal_result));
            };

            if (!is_regular_file_no_throw(request.path)) {
                LOG_DEBUG("[PipelinedImageLoader] Skipping missing image {}", lfs::core::path_to_utf8(request.path));
                fail_image_request("file does not exist or is not a regular file");
                continue;
            }

            if (!cuda_run) {
                // Host decoding reads each source file in its worker; there is no
                // encoded run cache to consult.
                PrefetchedImage result;
                result.sequence_id = request.sequence_id;
                result.loader_generation = request.loader_generation;
                result.path = request.path;
                result.params = request.params;
                result.needs_processing = true;
                result.alpha_as_mask = request.extract_alpha_as_mask;
                result.alpha_mask_params = request.alpha_mask_params;
                result.undistort = request.undistort;
                cold_queue_.push(std::move(result));
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    ++stats_.cold_path_misses;
                }
                if (request.mask_path && !request.extract_alpha_as_mask) {
                    if (!is_regular_file_no_throw(*request.mask_path)) {
                        std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                        fail_sidecar_locked(request.sequence_id, request.loader_generation,
                                            SidecarKind::Mask,
                                            *request.mask_path, "file does not exist or is not a regular file", lock);
                    } else {
                        PrefetchedImage mask_result;
                        mask_result.sequence_id = request.sequence_id;
                        mask_result.loader_generation = request.loader_generation;
                        mask_result.path = *request.mask_path;
                        mask_result.params = request.params;
                        mask_result.is_mask = true;
                        mask_result.needs_processing = true;
                        mask_result.mask_params = request.mask_params;
                        mask_result.undistort = request.undistort;
                        cold_queue_.push(std::move(mask_result));
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        ++stats_.mask_cache_misses;
                    }
                }
                enqueue_depth_request();
                enqueue_normal_request();
                continue;
            }

            if (request.extract_alpha_as_mask) {
                const auto rgb_key = make_cache_key(request.path, request.params);
                const auto alpha_key = make_mask_cache_key(request.path, request.params);
                auto cached_rgb = get_from_jpeg_cache(rgb_key);
                auto cached_alpha = get_from_jpeg_cache(alpha_key);

                if (cached_rgb && cached_alpha) {
                    PrefetchedImage img_item;
                    img_item.sequence_id = request.sequence_id;
                    img_item.loader_generation = request.loader_generation;
                    img_item.path = request.path;
                    img_item.params = request.params;
                    img_item.cache_key = rgb_key;
                    img_item.jpeg_data = cached_rgb;
                    img_item.is_cache_hit = true;
                    hot_queue_.push(std::move(img_item));

                    PrefetchedImage mask_item;
                    mask_item.sequence_id = request.sequence_id;
                    mask_item.loader_generation = request.loader_generation;
                    mask_item.path = request.path;
                    mask_item.params = request.params;
                    mask_item.cache_key = alpha_key;
                    mask_item.jpeg_data = cached_alpha;
                    mask_item.is_mask = true;
                    mask_item.mask_params = request.alpha_mask_params;
                    mask_item.is_cache_hit = true;
                    hot_queue_.push(std::move(mask_item));

                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    ++stats_.hot_path_hits;
                } else {
                    PrefetchedImage result;
                    result.sequence_id = request.sequence_id;
                    result.loader_generation = request.loader_generation;
                    result.path = request.path;
                    result.params = request.params;
                    result.cache_key = rgb_key;
                    result.alpha_as_mask = true;
                    result.alpha_mask_params = request.alpha_mask_params;
                    result.needs_processing = true;
                    result.undistort = request.undistort;

                    cold_queue_.push(std::move(result));
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    ++stats_.cold_path_misses;
                }
                enqueue_depth_request();
                enqueue_normal_request();
                continue;
            }

            PrefetchedImage result;
            result.sequence_id = request.sequence_id;
            result.loader_generation = request.loader_generation;
            result.path = request.path;
            result.params = request.params;
            result.cache_key = make_cache_key(request.path, request.params);
            result.is_mask = false;
            result.undistort = request.undistort;

            try {
                const bool needs_requested_processing = load_params_need_processing(request.params);

                if (auto cached = load_cached_jpeg_blob(result.cache_key)) {
                    result.jpeg_data = std::move(cached);
                    result.is_cache_hit = true;
                    result.needs_processing = false;
                    hot_queue_.push(std::move(result));
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    ++stats_.hot_path_hits;
                } else {
                    result.raw_bytes = read_file(request.path);
                    result.is_original_jpeg = is_jpeg_data(result.raw_bytes);
                    result.is_cache_hit = false;

                    {
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        stats_.total_bytes_read += result.raw_bytes.size();
                    }

                    if (result.is_original_jpeg && !needs_requested_processing && nvcodec_hot_path_) {
                        // An unchanged JPEG is already a usable encoded cache
                        // entry. Avoid a decode/re-encode and its quality loss.
                        auto data = std::make_shared<std::vector<uint8_t>>(std::move(result.raw_bytes));
                        put_in_jpeg_cache(result.cache_key, data);
                        result.jpeg_data = std::move(data);
                        result.needs_processing = false;
                        result.is_cache_hit = true;
                        hot_queue_.push(std::move(result));
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        ++stats_.hot_path_hits;
                    } else {
                        result.needs_processing = true;
                        cold_queue_.push(std::move(result));
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        ++stats_.cold_path_misses;
                    }
                }
            } catch (const std::exception& e) {
                LOG_ERROR("[PipelinedImageLoader] Prefetch error {}: {}", lfs::core::path_to_utf8(request.path), e.what());
                fail_image_request(e.what());
                continue; // Skip auxiliary image processing if image failed
            }

            if (request.mask_path) {
                if (!is_regular_file_no_throw(*request.mask_path)) {
                    LOG_DEBUG("[PipelinedImageLoader] Skipping missing mask {}", lfs::core::path_to_utf8(*request.mask_path));
                    std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                    fail_sidecar_locked(request.sequence_id, request.loader_generation,
                                        SidecarKind::Mask,
                                        *request.mask_path, "file does not exist or is not a regular file", lock);
                } else {
                    PrefetchedImage mask_result;
                    mask_result.sequence_id = request.sequence_id;
                    mask_result.loader_generation = request.loader_generation;
                    mask_result.path = *request.mask_path;
                    mask_result.params = request.params;
                    mask_result.cache_key = make_mask_cache_key(*request.mask_path, request.params);
                    mask_result.is_mask = true;
                    mask_result.mask_params = request.mask_params;
                    mask_result.undistort = request.undistort;

                    try {
                        if (auto cached = get_from_jpeg_cache(mask_result.cache_key)) {
                            mask_result.jpeg_data = cached;
                            mask_result.is_cache_hit = true;
                            hot_queue_.push(std::move(mask_result));
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            ++stats_.mask_cache_hits;
                        } else {
                            mask_result.raw_bytes = read_file(*request.mask_path);
                            mask_result.is_original_jpeg = is_jpeg_data(mask_result.raw_bytes);
                            mask_result.is_cache_hit = false;

                            {
                                std::lock_guard<std::mutex> lock(stats_mutex_);
                                stats_.total_bytes_read += mask_result.raw_bytes.size();
                            }

                            mask_result.needs_processing = true;
                            cold_queue_.push(std::move(mask_result));
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            ++stats_.mask_cache_misses;
                        }
                    } catch (const std::exception& e) {
                        LOG_WARN("[PipelinedImageLoader] Mask prefetch error {}: {} - continuing without mask",
                                 lfs::core::path_to_utf8(*request.mask_path), e.what());
                        std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                        fail_sidecar_locked(request.sequence_id, request.loader_generation,
                                            SidecarKind::Mask,
                                            *request.mask_path, e.what(), lock);
                    }
                }
            }

            enqueue_depth_request();
            enqueue_normal_request();
        }
    }

    lfs::core::Tensor PipelinedImageLoader::load_image_immediate(
        const std::filesystem::path& path, const LoadParams& params) {
        if (config_.backend != lfs::core::GpuBackend::CUDA) {
            const lfs::core::GpuBackendScope backend(config_.backend);
            lfs::core::TensorUpload upload;
            auto image = decode_portable_rgb(path, params, upload);
            upload.wait();
            return image;
        }
        if (cuda_immediate_ == nullptr) {
            throw std::runtime_error("CUDA image decode is not part of this build");
        }
        return (this->*cuda_immediate_)(path, params);
    }

} // namespace lfs::io
