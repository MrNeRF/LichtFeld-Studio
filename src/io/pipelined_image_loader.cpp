/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/pipelined_image_loader.hpp"
#include "core/assert.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/error_reporter.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "cuda/image_format_kernels.cuh"
#include "diagnostics/vram_profiler.hpp"
#include "io/nvcodec_image_loader.hpp"

#include <cuda_runtime.h>
#include <stb_image.h>

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cerrno>
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

namespace lfs::io {

    struct PipelinedImageLoader::DecodedFrameRing
        : std::enable_shared_from_this<PipelinedImageLoader::DecodedFrameRing> {
        struct Slot {
            lfs::core::Tensor storage;
            bool in_use = false;
        };

        struct Lease {
            lfs::core::Tensor storage;
            std::weak_ptr<DecodedFrameRing> owner;
            size_t index = 0;

            ~Lease() {
                if (auto ring = owner.lock()) {
                    ring->release(index, storage);
                }
            }
        };

        explicit DecodedFrameRing(const size_t capacity, const std::atomic<bool>* running)
            : slots_(std::max<size_t>(1, capacity)),
              capacity_limit_(slots_.size()),
              running_(running) {}

        void release(const size_t index, const lfs::core::Tensor& storage) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (index < slots_.size() && slots_[index].in_use) {
                if (index < capacity_limit_)
                    slots_[index].storage = storage;
                else
                    slots_[index].storage = lfs::core::Tensor();
                slots_[index].in_use = false;
                if (live_count_ > 0)
                    --live_count_;
                cv_.notify_one();
            }
        }

        std::vector<std::shared_ptr<Lease>> acquire_batch(const size_t requested) {
            std::unique_lock<std::mutex> lock(mutex_);
            while (running_ && running_->load(std::memory_order_acquire)) {
                const size_t count = std::min(requested, capacity_limit_);
                const size_t free_slots = static_cast<size_t>(std::count_if(
                    slots_.begin(), slots_.begin() + capacity_limit_,
                    [](const Slot& slot) { return !slot.in_use; }));
                if (count > 0 && free_slots >= count && live_count_ + count <= capacity_limit_) {
                    std::vector<std::shared_ptr<Lease>> leases;
                    leases.reserve(count);
                    for (size_t i = 0; i < count; ++i) {
                        const auto it = std::find_if(
                            slots_.begin(), slots_.begin() + capacity_limit_,
                            [](const Slot& slot) { return !slot.in_use; });
                        it->in_use = true;
                        ++live_count_;
                        auto lease = std::make_shared<Lease>();
                        lease->storage = it->storage;
                        lease->owner = shared_from_this();
                        lease->index = static_cast<size_t>(std::distance(slots_.begin(), it));
                        leases.push_back(std::move(lease));
                    }
                    return leases;
                }
                cv_.wait(lock);
            }
            return {};
        }

        void set_capacity(const size_t capacity) {
            std::lock_guard<std::mutex> lock(mutex_);
            capacity_limit_ = std::clamp<size_t>(capacity, 1, slots_.size());
            for (size_t i = capacity_limit_; i < slots_.size(); ++i) {
                if (!slots_[i].in_use)
                    slots_[i].storage = lfs::core::Tensor();
            }
            cv_.notify_all();
        }

        void cancel() { cv_.notify_all(); }

        void reclaim_idle() {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& slot : slots_) {
                if (!slot.in_use)
                    slot.storage = lfs::core::Tensor();
            }
        }

        size_t storage_bytes() const {
            std::lock_guard<std::mutex> lock(mutex_);
            size_t total = 0;
            for (const auto& slot : slots_)
                total += slot.storage.bytes();
            return total;
        }

    private:
        std::vector<Slot> slots_;
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        size_t live_count_ = 0;
        size_t capacity_limit_ = 1;
        const std::atomic<bool>* running_ = nullptr;
    };

    namespace {

        constexpr int DEFAULT_DECODER_POOL_SIZE = 8;

    } // namespace

    namespace {

        // Cold workers scale with sidecar decode demand, but their nvimagecodec
        // decodes run GPU work on the default stream, which serializes with the
        // blocking training stream. Cap concurrency so GPU decode pressure stays
        // at pre-sidecar levels regardless of the cold pool size.
        std::counting_semaphore<>& nvcodec_decode_slots() {
            static std::counting_semaphore<> slots{2};
            return slots;
        }

        class NvcodecSlotGuard {
        public:
            NvcodecSlotGuard() { nvcodec_decode_slots().acquire(); }
            ~NvcodecSlotGuard() { nvcodec_decode_slots().release(); }
            NvcodecSlotGuard(const NvcodecSlotGuard&) = delete;
            NvcodecSlotGuard& operator=(const NvcodecSlotGuard&) = delete;
        };

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

        struct NvCodecLoaderCacheEntry {
            std::shared_ptr<NvCodecImageLoader> instance;
            size_t owner_count = 0;
        };

        [[nodiscard]] size_t normalize_nvcodec_pool_size(size_t decoder_pool_size) {
            return decoder_pool_size > 0 ? decoder_pool_size : DEFAULT_DECODER_POOL_SIZE;
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

        std::mutex& get_nvcodec_mutex() {
            static std::mutex mtx;
            return mtx;
        }

        std::unordered_map<size_t, NvCodecLoaderCacheEntry>& get_nvcodec_loader_cache() {
            static std::unordered_map<size_t, NvCodecLoaderCacheEntry> instances;
            return instances;
        }

        std::shared_ptr<NvCodecImageLoader> acquire_nvcodec_loader(size_t decoder_pool_size) {
            std::lock_guard<std::mutex> lock(get_nvcodec_mutex());
            auto& instances = get_nvcodec_loader_cache();
            const size_t requested_pool_size = normalize_nvcodec_pool_size(decoder_pool_size);

            if (auto it = instances.find(requested_pool_size);
                it != instances.end() && it->second.instance) {
                return it->second.instance;
            }

            auto instance = [&requested_pool_size] {
                NvCodecImageLoader::Options opts;
                opts.device_id = 0;
                opts.decoder_pool_size = requested_pool_size;
                opts.enable_fallback = true;
                return std::make_shared<NvCodecImageLoader>(opts);
            }();

            instances[requested_pool_size].instance = instance;
            return instance;
        }

        void retain_nvcodec_loader_cache(size_t decoder_pool_size) {
            std::lock_guard<std::mutex> lock(get_nvcodec_mutex());
            ++get_nvcodec_loader_cache()[normalize_nvcodec_pool_size(decoder_pool_size)].owner_count;
        }

        void release_nvcodec_loader_cache(size_t decoder_pool_size) {
            std::shared_ptr<NvCodecImageLoader> released_instance;

            {
                std::lock_guard<std::mutex> lock(get_nvcodec_mutex());
                auto& instances = get_nvcodec_loader_cache();
                const size_t requested_pool_size = normalize_nvcodec_pool_size(decoder_pool_size);
                const auto it = instances.find(requested_pool_size);
                if (it == instances.end() || it->second.owner_count == 0)
                    return;

                auto& entry = it->second;
                --entry.owner_count;
                if (entry.owner_count == 0) {
                    released_instance = std::move(entry.instance);
                    instances.erase(it);
                }
            }

            // Drop the cache's last reference outside the mutex so teardown does not block other callers.
            released_instance.reset();
        }

        bool is_nvcodec_available() {
            static std::once_flag flag;
            static bool available = false;
            std::call_once(flag, [] { available = NvCodecImageLoader::is_available(); });
            return available;
        }

        [[nodiscard]] bool load_params_need_processing(const LoadParams& params) {
            return params.resize_factor > 1 || params.max_width > 0 || params.undistort != nullptr;
        }

        void apply_requested_undistort(lfs::core::Tensor& tensor, const LoadParams& params) {
            if (!params.undistort)
                return;

            const bool restore_uint8 = params.output_uint8 && tensor.dtype() == lfs::core::DataType::UInt8;
            if (restore_uint8) {
                tensor = tensor.to(lfs::core::DataType::Float32) / 255.0f;
            }

            const auto scaled = lfs::core::scale_undistort_params(
                *params.undistort,
                static_cast<int>(tensor.shape()[2]),
                static_cast<int>(tensor.shape()[1]));
            tensor = lfs::core::undistort_image(tensor, scaled, nullptr);

            if (restore_uint8) {
                auto uint8_tensor = lfs::core::Tensor::empty(
                    tensor.shape(), lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                cuda::launch_float32_chw_to_uint8_chw(
                    tensor.ptr<float>(),
                    uint8_tensor.ptr<uint8_t>(),
                    tensor.shape()[1],
                    tensor.shape()[2],
                    tensor.shape()[0],
                    static_cast<cudaStream_t>(params.cuda_stream));
                tensor = std::move(uint8_tensor);
            }
        }

        lfs::core::Tensor process_mask(lfs::core::Tensor mask, const float threshold) {
            if (!mask.is_valid())
                return {};
            if (mask.dtype() == lfs::core::DataType::UInt8)
                mask = mask.to(lfs::core::DataType::Float32) / 255.0f;
            else if (mask.dtype() == lfs::core::DataType::Bool)
                mask = mask.to(lfs::core::DataType::Float32);
            mask = mask.clamp(0.0f, 1.0f);
            if (threshold > 0.0f)
                mask = mask.ge(threshold).to(lfs::core::DataType::Float32);
            return mask.contiguous();
        }

        [[nodiscard]] bool is_jpeg_file_signature(const std::filesystem::path& path) {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file))
                return false;

            std::array<uint8_t, 3> signature{};
            if (!file.read(reinterpret_cast<char*>(signature.data()),
                           static_cast<std::streamsize>(signature.size()))) {
                return false;
            }

            return signature[0] == 0xFF && signature[1] == 0xD8 && signature[2] == 0xFF;
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

        [[nodiscard]] std::string describe_current_exception(const char* fallback) {
            try {
                throw;
            } catch (const std::exception& e) {
                return e.what();
            } catch (...) {
                return fallback;
            }
        }

        [[nodiscard]] lfs::core::Tensor decode_cached_rgb_tensor(
            const std::shared_ptr<NvCodecImageLoader>& nvcodec,
            const std::shared_ptr<std::vector<uint8_t>>& jpeg_data,
            const LoadParams& params,
            const bool apply_processing) {
            auto tensor = nvcodec->load_image_from_memory_gpu(
                *jpeg_data,
                apply_processing ? params.resize_factor : 1,
                apply_processing ? params.max_width : 0,
                params.cuda_stream,
                DecodeFormat::RGB,
                params.output_uint8);
            if (!tensor.is_valid() || tensor.numel() == 0)
                return {};

            if (apply_processing)
                apply_requested_undistort(tensor, params);

            return tensor;
        }

        std::tuple<uint8_t*, int, int> load_grayscale_stb(const std::filesystem::path& path) {
            int w, h, c;
            uint8_t* const data = stbi_load(lfs::core::path_to_utf8(path).c_str(), &w, &h, &c, 1);
            return {data, w, h};
        }

        void synchronize_async_upload_before_free(cudaStream_t stream, const char* context) {
            if (!stream) {
                return;
            }
            if (const cudaError_t err = cudaStreamSynchronize(stream); err != cudaSuccess) {
                throw std::runtime_error(
                    std::string(context) + " upload sync failed: " + cudaGetErrorString(err));
            }
        }

    } // namespace

    PipelinedImageLoader::PipelinedImageLoader(PipelinedLoaderConfig config)
        : config_(std::move(config)),
          output_queue_(std::max<size_t>(1, config_.output_queue_size)) {

        cleanup_stale_run_spill_directories();
        config_.jpeg_batch_size = std::clamp<size_t>(config_.jpeg_batch_size, 1, 12);
        if (config_.max_cache_bytes == 0)
            config_.max_cache_bytes = static_cast<size_t>(
                static_cast<double>(get_total_physical_memory()) * 0.90);
        {
            std::lock_guard<std::mutex> lock(adaptive_mutex_);
            adaptive_target_ = std::clamp<size_t>(config_.prefetch_count, 2, 12);
        }

        ledger_.reserve(std::max(config_.prefetch_count, config_.output_queue_size) * 2);

        LOG_INFO("[PipelinedImageLoader] batch_size={}, prefetch={}, output_queue={}, io_threads={}, cold_threads={}, 16bit_color={}",
                 config_.jpeg_batch_size,
                 config_.prefetch_count,
                 config_.output_queue_size,
                 config_.io_threads,
                 config_.cold_process_threads,
                 config_.use_16bit_color);

        const bool nvcodec_available = is_nvcodec_available();
        LOG_INFO("[PipelinedImageLoader] host compressed cache cap: {:.1f} GiB",
                 config_.max_cache_bytes / (1024.0 * 1024.0 * 1024.0));

        {
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
        decoded_frame_ring_ = std::make_shared<DecodedFrameRing>(
            DECODE_FRAME_RING_CAPACITY, &running_);
        decoded_frame_ring_->set_capacity(adaptive_target_ + 2);
        decode_hwc_workspace_.resize(config_.jpeg_batch_size);

        for (size_t i = 0; i < config_.io_threads; ++i) {
            io_threads_.emplace_back([this] { prefetch_thread_func(); });
        }

        if (nvcodec_available) {
            // On failure fall back to the default stream rather than a bad handle.
            if (const cudaError_t err = cudaStreamCreateWithFlags(&decode_stream_, cudaStreamNonBlocking); err != cudaSuccess) {
                LOG_WARN("[PipelinedImageLoader] cudaStreamCreateWithFlags failed ({}), GPU decode falls back to default stream", cudaGetErrorString(err));
                decode_stream_ = nullptr;
            }
            gpu_decode_thread_ = std::thread([this] { gpu_batch_decode_thread_func(); });
        }

        sidecar_streams_.resize(config_.cold_process_threads, nullptr);
        for (size_t i = 0; i < config_.cold_process_threads; ++i) {
            if (const cudaError_t err = cudaStreamCreateWithFlags(&sidecar_streams_[i], cudaStreamNonBlocking);
                err != cudaSuccess) {
                LOG_WARN("[PipelinedImageLoader] sidecar stream creation failed for worker {} ({}), sidecars fall back to default stream",
                         i,
                         cudaGetErrorString(err));
                sidecar_streams_[i] = nullptr;
            }
            cold_process_threads_.emplace_back([this, i] { cold_process_thread_func(i); });
        }

        if (nvcodec_available) {
            retain_nvcodec_loader_cache(config_.decoder_pool_size);
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

        cudaDeviceSynchronize();
        if (decode_stream_) {
            lfs::core::CudaMemoryPool::instance().release_stream(decode_stream_);
            cudaStreamDestroy(decode_stream_);
            decode_stream_ = nullptr;
        }
        for (auto& stream : sidecar_streams_) {
            if (stream) {
                lfs::core::CudaMemoryPool::instance().release_stream(stream);
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
        }
        sidecar_streams_.clear();
        release_nvcodec_loader_cache(config_.decoder_pool_size);

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
            12);
        size_t next = adaptive_target_;
        if (window_dl_wait_ms > 0.05) {
            next = std::min<size_t>(12, std::max(next + 1, recommended));
            adaptive_low_recommendation_windows_ = 0;
            adaptive_growth_cooldown_windows_ = 4;
        } else {
            if (adaptive_growth_cooldown_windows_ > 0) {
                --adaptive_growth_cooldown_windows_;
                adaptive_low_recommendation_windows_ = 0;
            } else if (recommended < adaptive_target_) {
                ++adaptive_low_recommendation_windows_;
                if (adaptive_low_recommendation_windows_ >= 4) {
                    next = std::min<size_t>(12, std::max<size_t>(2, recommended + 1));
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

    lfs::core::Tensor PipelinedImageLoader::load_image_immediate(
        const std::filesystem::path& path, const LoadParams& params) {
        const auto cache_key = make_cache_key(path, params);
        const bool is_original_jpeg = is_jpeg_file_signature(path);
        const bool needs_requested_processing = load_params_need_processing(params);
        auto decode_cached_hit = [&](const std::shared_ptr<std::vector<uint8_t>>& jpeg_data) -> lfs::core::Tensor {
            if (!is_nvcodec_available())
                return {};

            try {
                auto nvcodec = acquire_nvcodec_loader(config_.decoder_pool_size);
                auto tensor = decode_cached_rgb_tensor(nvcodec, jpeg_data, params, false);
                if (tensor.is_valid() && tensor.numel() > 0)
                    return tensor;
            } catch (...) {}
            return {};
        };

        if (auto jpeg_data = load_cached_jpeg_blob(cache_key)) {
            if (auto tensor = decode_cached_hit(jpeg_data);
                tensor.is_valid() && tensor.numel() > 0) {
                return tensor;
            }
        }

        lfs::core::Tensor decoded;

        if (is_original_jpeg) {
            auto data = std::make_shared<std::vector<uint8_t>>(read_file(path));
            if (!needs_requested_processing) {
                put_in_jpeg_cache(cache_key, data);
            }

            if (is_nvcodec_available()) {
                try {
                    auto nvcodec = acquire_nvcodec_loader(config_.decoder_pool_size);
                    auto tensor = decode_cached_rgb_tensor(nvcodec, data, params, needs_requested_processing);
                    if (tensor.is_valid() && tensor.numel() > 0)
                        return tensor;
                } catch (...) {
                    LOG_DEBUG("[PipelinedImageLoader] Immediate JPEG decode fallback for {}: {}",
                              lfs::core::path_to_utf8(path),
                              describe_current_exception("non-standard nvImageCodec exception"));
                }
            }
        } else if (!config_.use_16bit_color) {
            const std::string path_str = lfs::core::path_to_utf8(path);
            int w = 0, h = 0, ch = 0;
            unsigned char* img_data = stbi_load(path_str.c_str(), &w, &h, &ch, 3);
            const bool used_stbi = (img_data != nullptr);
            if (img_data) {
                ch = 3;
            } else {
                auto [oiio_data, ow, oh, oc] = lfs::core::load_image(path, 1, 0);
                if (!oiio_data)
                    throw std::runtime_error("Failed to decode image: " + path_str);
                img_data = oiio_data;
                w = ow;
                h = oh;
                ch = oc;
            }

            const size_t H = static_cast<size_t>(h);
            const size_t W = static_cast<size_t>(w);
            const size_t C = static_cast<size_t>(ch);

            auto cpu_tensor = lfs::core::Tensor::from_blob(
                img_data, lfs::core::TensorShape({H, W, C}),
                lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
            gpu_uint8.set_name("io.image.gpu_staging");
            if (used_stbi)
                stbi_image_free(img_data);
            else
                lfs::core::free_image(img_data);

            if (params.output_uint8) {
                decoded = lfs::core::Tensor::empty(
                    lfs::core::TensorShape({C, H, W}),
                    lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                decoded.set_name("io.image.gpu_uint8");
                cuda::launch_uint8_hwc_to_uint8_chw(
                    reinterpret_cast<const uint8_t*>(gpu_uint8.data_ptr()),
                    reinterpret_cast<uint8_t*>(decoded.data_ptr()),
                    H, W, C, nullptr);
            } else {
                decoded = lfs::core::Tensor::empty(
                    lfs::core::TensorShape({C, H, W}),
                    lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                decoded.set_name("io.image.gpu_float");
                cuda::launch_uint8_hwc_to_float32_chw(
                    reinterpret_cast<const uint8_t*>(gpu_uint8.data_ptr()),
                    reinterpret_cast<float*>(decoded.data_ptr()),
                    H, W, C, nullptr);
            }

            if (is_nvcodec_available()) {
                try {
                    auto nvcodec = acquire_nvcodec_loader(config_.decoder_pool_size);
                    auto jpeg_bytes = nvcodec->encode_to_jpeg(decoded, config_.cache_jpeg_quality, nullptr);
                    auto jpeg_shared = std::make_shared<std::vector<uint8_t>>(std::move(jpeg_bytes));
                    put_in_jpeg_cache(cache_key, jpeg_shared);

                    if (needs_requested_processing) {
                        auto tensor = decode_cached_rgb_tensor(nvcodec, jpeg_shared, params, false);
                        if (tensor.is_valid() && tensor.numel() > 0)
                            return tensor;
                    }
                } catch (...) {
                    LOG_DEBUG("[PipelinedImageLoader] Immediate cache write skipped for {}: {}",
                              lfs::core::path_to_utf8(path),
                              describe_current_exception("non-standard nvImageCodec exception"));
                }
            }
        }

        if (!decoded.is_valid() || decoded.numel() == 0 || needs_requested_processing) {
            decoded = decode_file_on_cpu(path, params);
        }

        apply_requested_undistort(decoded, params);
        return decoded;
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

    lfs::core::Tensor PipelinedImageLoader::decode_file_on_cpu(
        const std::filesystem::path& path,
        const LoadParams& params) const {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;
        using lfs::core::TensorShape;

        Tensor decoded;
        Tensor gpu_staging;
        if (config_.use_16bit_color) {
            auto [img_data, width, height, channels] = lfs::core::load_image_u16(
                path, params.resize_factor, params.max_width);
            if (!img_data)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));

            const size_t H = static_cast<size_t>(height);
            const size_t W = static_cast<size_t>(width);
            const size_t C = static_cast<size_t>(channels);

            // Float16 is only a 2-byte container for the uint16 samples (no UInt16 dtype).
            auto cpu_tensor = Tensor::from_blob(
                img_data, TensorShape({H, W, C}), Device::CPU, DataType::Float16);
            gpu_staging = cpu_tensor.to(Device::CUDA);
            lfs::core::free_image(img_data);

            if (params.output_uint8) {
                decoded = Tensor::empty(TensorShape({C, H, W}), Device::CUDA, DataType::UInt8);
                cuda::launch_uint16_hwc_to_uint8_chw(
                    reinterpret_cast<const uint16_t*>(gpu_staging.data_ptr()),
                    decoded.ptr<uint8_t>(), H, W, C, nullptr);
            } else {
                decoded = Tensor::empty(TensorShape({C, H, W}), Device::CUDA, DataType::Float32);
                cuda::launch_uint16_hwc_to_float32_chw(
                    reinterpret_cast<const uint16_t*>(gpu_staging.data_ptr()),
                    decoded.ptr<float>(), H, W, C, nullptr);
            }
        } else {
            auto [img_data, width, height, channels] = lfs::core::load_image(
                path, params.resize_factor, params.max_width);
            if (!img_data)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));

            const size_t H = static_cast<size_t>(height);
            const size_t W = static_cast<size_t>(width);
            const size_t C = static_cast<size_t>(channels);

            auto cpu_tensor = Tensor::from_blob(
                img_data, TensorShape({H, W, C}), Device::CPU, DataType::UInt8);
            gpu_staging = cpu_tensor.to(Device::CUDA);
            lfs::core::free_image(img_data);

            if (params.output_uint8) {
                decoded = Tensor::empty(TensorShape({C, H, W}), Device::CUDA, DataType::UInt8);
                cuda::launch_uint8_hwc_to_uint8_chw(
                    gpu_staging.ptr<uint8_t>(), decoded.ptr<uint8_t>(), H, W, C, nullptr);
            } else {
                decoded = Tensor::empty(TensorShape({C, H, W}), Device::CUDA, DataType::Float32);
                cuda::launch_uint8_hwc_to_float32_chw(
                    gpu_staging.ptr<uint8_t>(), decoded.ptr<float>(), H, W, C, nullptr);
            }
        }

        if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(err));
        }
        return decoded;
    }

    void PipelinedImageLoader::write_derived_cache(NvCodecImageLoader& nvcodec,
                                                   const lfs::core::Tensor& tensor,
                                                   const std::string& cache_key,
                                                   void* cuda_stream) {
        if (config_.use_16bit_color && !jpeg2k_cache_available_.load(std::memory_order_relaxed))
            return;

        try {
            auto bytes = config_.use_16bit_color
                             ? nvcodec.encode_to_jpeg2k(tensor, cuda_stream)
                             : nvcodec.encode_to_jpeg(tensor, config_.cache_jpeg_quality, cuda_stream);
            put_in_jpeg_cache(cache_key, std::make_shared<std::vector<uint8_t>>(std::move(bytes)));
        } catch (...) {
            const auto message = describe_current_exception("non-standard nvImageCodec exception");
            if (config_.use_16bit_color && jpeg2k_cache_available_.exchange(false)) {
                LOG_ERROR("[PipelinedImageLoader] JPEG 2000 cache encode unavailable ({}); "
                          "16-bit training continues without a cache — every epoch re-decodes "
                          "source images. Build with the nvJPEG2000 library to enable it.",
                          message);
            } else {
                LOG_DEBUG("[PipelinedImageLoader] Derived cache write skipped for key {}: {}",
                          cache_key, message);
            }
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

    void PipelinedImageLoader::write_sidecar_cache(NvCodecImageLoader& nvcodec,
                                                   const lfs::core::Tensor& tensor,
                                                   const PrefetchedImage& item,
                                                   const SidecarCacheFormat kind,
                                                   void* cuda_stream) {
        if (!jpeg2k_cache_available_.load(std::memory_order_relaxed)) {
            return;
        }

        try {
            std::vector<uint8_t> bytes;
            // Sidecars are stored as lossless HT J2K after all resize/undistort/prior
            // transforms. Depth [H,W] is quantized as round(v*65535). Normals
            // [3,H,W] in [-1,1] are mapped to HWC [0,1] as (n+1)/2 before the same
            // u16 quantization; hot decode reverses that map with <= 1/65535 loss.
            if (kind == SidecarCacheFormat::Depth) {
                bytes = nvcodec.encode_grayscale_to_jpeg2k(tensor, cuda_stream);
            } else {
                if (tensor.ndim() != 3 || tensor.shape()[0] != 3) {
                    throw std::runtime_error("Normal sidecar cache expects [3,H,W] tensor");
                }
                const size_t height = tensor.shape()[1];
                const size_t width = tensor.shape()[2];
                auto hwc = lfs::core::Tensor::empty(
                    lfs::core::TensorShape({height, width, size_t{3}}),
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32);
                cuda::launch_normal_chw_to_jpeg2k_hwc(
                    tensor.ptr<float>(), hwc.ptr<float>(), height, width,
                    static_cast<cudaStream_t>(cuda_stream));
                bytes = nvcodec.encode_to_jpeg2k(hwc, cuda_stream);
            }
            put_in_jpeg_cache(item.cache_key, std::make_shared<std::vector<uint8_t>>(std::move(bytes)));
        } catch (...) {
            const auto message = describe_current_exception("non-standard nvImageCodec exception");
            if (jpeg2k_cache_available_.exchange(false)) {
                LOG_ERROR("[PipelinedImageLoader] JPEG 2000 sidecar cache encode unavailable ({}); "
                          "depth/normal training continues without a sidecar cache.",
                          message);
            } else {
                LOG_DEBUG("[PipelinedImageLoader] Sidecar cache write skipped for {}: {}",
                          lfs::core::path_to_utf8(item.path),
                          message);
            }
        }
    }

    lfs::core::Tensor PipelinedImageLoader::decode_cached_sidecar(NvCodecImageLoader& nvcodec,
                                                                  const PrefetchedImage& item,
                                                                  void* cuda_stream) {
        if (!item.jpeg_data) {
            return {};
        }

        auto decoded = nvcodec.decode_jpeg2k_16bit_from_memory_gpu(
            *item.jpeg_data, cuda_stream, false);
        if (item.is_depth) {
            if (!decoded.is_valid() || decoded.ndim() != 2) {
                throw std::runtime_error("Decoded depth sidecar cache is not [H,W]");
            }
            decoded.set_stream(static_cast<cudaStream_t>(cuda_stream));
            return decoded;
        }

        if (!decoded.is_valid() || decoded.ndim() != 3 || decoded.shape()[2] != 3) {
            throw std::runtime_error("Decoded normal sidecar cache is not [H,W,3]");
        }
        const size_t height = decoded.shape()[0];
        const size_t width = decoded.shape()[1];
        auto normal = lfs::core::Tensor::empty(
            lfs::core::TensorShape({size_t{3}, height, width}),
            lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);
        cuda::launch_jpeg2k_hwc_to_normal_chw(
            decoded.ptr<float>(), normal.ptr<float>(), height, width,
            static_cast<cudaStream_t>(cuda_stream));
        normal.set_stream(static_cast<cudaStream_t>(cuda_stream));
        return normal;
    }

    cudaEvent_t PipelinedImageLoader::record_sidecar_ready_event(cudaStream_t stream) {
        cudaEvent_t event = nullptr;
        if (const cudaError_t create_err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            create_err != cudaSuccess) {
            LOG_WARN("[PipelinedImageLoader] sidecar event creation failed: {}",
                     cudaGetErrorString(create_err));
            cudaStreamSynchronize(stream);
            return nullptr;
        }
        if (const cudaError_t record_err = cudaEventRecord(event, stream);
            record_err != cudaSuccess) {
            LOG_WARN("[PipelinedImageLoader] sidecar event record failed: {}",
                     cudaGetErrorString(record_err));
            cudaEventDestroy(event);
            cudaStreamSynchronize(stream);
            return nullptr;
        }
        return event;
    }

    std::pair<int, int> PipelinedImageLoader::sidecar_target_size(
        const PrefetchedImage& item,
        const int src_w,
        const int src_h) const {
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
        constexpr double HOST_FREE_RAM_EVICTION_RATIO = 0.10;
        const size_t min_free = static_cast<size_t>(get_total_physical_memory() * HOST_FREE_RAM_EVICTION_RATIO);

        if (available < min_free + required_bytes) {
            target = std::min(target, jpeg_cache_bytes_.load() / 2);
        }

        while (jpeg_cache_bytes_ + required_bytes > target && !jpeg_cache_.empty()) {
            auto oldest = jpeg_cache_.begin();
            for (auto it = jpeg_cache_.begin(); it != jpeg_cache_.end(); ++it) {
                if (it->second.last_access < oldest->second.last_access) {
                    oldest = it;
                }
            }
            const auto key = oldest->first;
            const auto data = oldest->second.data;
            jpeg_cache_bytes_ -= oldest->second.size_bytes;
            jpeg_cache_.erase(oldest);
            spill_cache_entry_locked(key, data);
        }
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
                    destroy_sidecar_ready_event(outcome->depth_ready_event);
                    destroy_sidecar_ready_event(outcome->normal_ready_event);
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
        destroy_sidecar_ready_event(it->second.depth_ready_event);
        destroy_sidecar_ready_event(it->second.normal_ready_event);
        pending_pairs_.erase(it);
    }

    void PipelinedImageLoader::destroy_sidecar_ready_event(cudaEvent_t& event) {
        if (event) {
            cudaEventDestroy(event);
            event = nullptr;
        }
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
                    destroy_sidecar_ready_event(completion.outcome->depth_ready_event);
                    destroy_sidecar_ready_event(completion.outcome->normal_ready_event);
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
            .stream = pair.stream,
            .depth = depth_has_value ? std::optional(std::move(*pair.depth)) : std::nullopt,
            .normal = normal_has_value ? std::optional(std::move(*pair.normal)) : std::nullopt,
            .depth_ready_event = pair.depth_ready_event,
            .normal_ready_event = pair.normal_ready_event,
            .decoded_frame_leases = std::move(pair.decoded_frame_leases),
            .error = {},
        };
        pair.depth_ready_event = nullptr;
        pair.normal_ready_event = nullptr;
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

    void PipelinedImageLoader::try_complete_pair(
        size_t sequence_id,
        const std::uint64_t loader_generation,
        std::optional<lfs::core::Tensor> image,
        std::optional<lfs::core::Tensor> mask,
        cudaStream_t stream,
        std::optional<lfs::core::Tensor> depth,
        std::optional<lfs::core::Tensor> normal,
        cudaEvent_t sidecar_ready_event,
        std::shared_ptr<void> decoded_frame_lease) {

        std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
        auto it = pending_pairs_.find(sequence_id);
        if (it == pending_pairs_.end() || it->second.loader_generation != loader_generation) {
            destroy_sidecar_ready_event(sidecar_ready_event);
            return;
        }
        auto& pair = it->second;

        if (decoded_frame_lease) {
            pair.decoded_frame_leases.push_back(std::move(decoded_frame_lease));
        }

        if (image) {
            subtract_clamped(pending_image_bytes_, pair.image_bytes);
            pair.image = std::move(*image);
            pair.image_bytes = tensor_reserved_bytes(*pair.image);
            pending_image_bytes_.fetch_add(pair.image_bytes, std::memory_order_acq_rel);
        }
        if (mask) {
            subtract_clamped(pending_mask_bytes_, pair.mask_bytes);
            pair.mask = std::move(*mask);
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
        if (stream)
            pair.stream = stream;
        if (sidecar_ready_event) {
            CUevent_st*& slot = depth ? pair.depth_ready_event : pair.normal_ready_event;
            if (slot) {
                cudaEventDestroy(slot);
            }
            slot = sidecar_ready_event;
        }

        try_push_ready_locked(sequence_id, it, lock);
    }

    void PipelinedImageLoader::prefetch_thread_func() {
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
                // Every source is canonicalized once for this run before it is
                // consumed from the encoded run cache. Never treat an original
                // JPEG as an already-canonical blob.
                const bool needs_requested_processing = true;

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

                    result.needs_processing = needs_requested_processing;
                    cold_queue_.push(std::move(result));
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    ++stats_.cold_path_misses;
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

    void PipelinedImageLoader::gpu_batch_decode_thread_func() {
        LFS_VRAM_SCOPE("io.image_loader");
        // Tensor ops on this thread (decode targets, format conversion) home
        // onto the decode stream so they order with the explicit-stream kernels.
        const lfs::core::CUDAStreamGuard stream_guard(decode_stream_);
        std::vector<PrefetchedImage> batch;
        batch.reserve(config_.jpeg_batch_size);

        while (running_) {
            batch.clear();
            const auto deadline = std::chrono::steady_clock::now() + config_.batch_collect_timeout;

            try {
                auto first = hot_queue_.try_pop_for(config_.output_wait_timeout);
                if (!first)
                    continue;
                batch.push_back(std::move(*first));
            } catch (const std::runtime_error&) {
                break;
            }

            const size_t batch_limit = std::clamp<size_t>(adaptive_prefetch_target(), 2, 12);
            while (batch.size() < std::min(config_.jpeg_batch_size, batch_limit)) {
                if (auto item = hot_queue_.try_pop()) {
                    batch.push_back(std::move(*item));
                    continue;
                }
                if (batch.size() >= 2)
                    break;
                const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now());
                if (remaining.count() <= 0)
                    break;
                auto item = hot_queue_.try_pop_for(std::min(remaining, std::chrono::milliseconds{1}));
                if (!item)
                    break;
                batch.push_back(std::move(*item));
            }

            if (batch.empty())
                continue;

            const auto decode_begin = std::chrono::steady_clock::now();
            try {
                auto nvcodec = acquire_nvcodec_loader(config_.decoder_pool_size);
                std::vector<bool> decoded_as_pair(batch.size(), false);

                const auto prepare_sidecar = [&](const PrefetchedImage& item,
                                                 lfs::core::Tensor decoded) {
                    if (item.is_depth) {
                        if (!decoded.is_valid() || decoded.ndim() != 2) {
                            throw std::runtime_error("Decoded depth sidecar cache is not [H,W]");
                        }
                        decoded.set_stream(decode_stream_);
                        return decoded;
                    }
                    if (!decoded.is_valid() || decoded.ndim() != 3 || decoded.shape()[2] != 3) {
                        throw std::runtime_error("Decoded normal sidecar cache is not [H,W,3]");
                    }
                    const size_t height = decoded.shape()[0];
                    const size_t width = decoded.shape()[1];
                    auto normal = lfs::core::Tensor::empty(
                        lfs::core::TensorShape({size_t{3}, height, width}),
                        lfs::core::Device::CUDA,
                        lfs::core::DataType::Float32);
                    cuda::launch_jpeg2k_hwc_to_normal_chw(
                        decoded.ptr<float>(), normal.ptr<float>(), height, width, decode_stream_);
                    normal.set_stream(decode_stream_);
                    return normal;
                };

                const auto complete_sidecar = [&](const PrefetchedImage& item,
                                                  lfs::core::Tensor sidecar) {
                    auto ready_event = record_sidecar_ready_event(decode_stream_);
                    if (item.is_depth) {
                        try_complete_pair(
                            item.sequence_id,
                            item.loader_generation,
                            std::nullopt,
                            std::nullopt,
                            nullptr,
                            std::move(sidecar),
                            std::nullopt,
                            ready_event);
                    } else {
                        try_complete_pair(
                            item.sequence_id,
                            item.loader_generation,
                            std::nullopt,
                            std::nullopt,
                            nullptr,
                            std::nullopt,
                            std::move(sidecar),
                            ready_event);
                    }
                };

                std::vector<size_t> rgb_batch_indices;
                rgb_batch_indices.reserve(batch.size());
                bool rgb_dtype_initialized = false;
                bool rgb_output_uint8 = false;
                for (size_t i = 0; i < batch.size(); ++i) {
                    if (!batch[i].is_mask && !batch[i].is_depth && !batch[i].is_normal &&
                        !batch[i].needs_processing &&
                        batch[i].jpeg_data) {
                        if (!rgb_dtype_initialized) {
                            rgb_dtype_initialized = true;
                            rgb_output_uint8 = batch[i].params.output_uint8;
                        }
                        if (batch[i].params.output_uint8 != rgb_output_uint8)
                            continue;
                        rgb_batch_indices.push_back(i);
                    }
                }
                if (!rgb_batch_indices.empty()) {
                    auto leases = decoded_frame_ring_->acquire_batch(rgb_batch_indices.size());
                    if (leases.empty())
                        throw std::runtime_error("decoded frame ring cancelled");
                    rgb_batch_indices.resize(leases.size());
                    std::vector<std::pair<const uint8_t*, size_t>> spans;
                    spans.reserve(rgb_batch_indices.size());
                    std::vector<lfs::core::Tensor*> reusable_outputs;
                    std::vector<lfs::core::Tensor*> reusable_hwc;
                    reusable_outputs.reserve(rgb_batch_indices.size());
                    reusable_hwc.reserve(rgb_batch_indices.size());
                    assert(rgb_batch_indices.size() <= decode_hwc_workspace_.size());
                    for (size_t i = 0; i < rgb_batch_indices.size(); ++i) {
                        const size_t index = rgb_batch_indices[i];
                        spans.emplace_back(batch[index].jpeg_data->data(),
                                           batch[index].jpeg_data->size());
                        reusable_outputs.push_back(&leases[i]->storage);
                        reusable_hwc.push_back(&decode_hwc_workspace_[i]);
                    }
                    auto decoded = nvcodec->decode_jpeg_batch_from_spans(
                        spans, decode_stream_, rgb_output_uint8, true,
                        &reusable_hwc, &reusable_outputs);
                    if (decoded.size() != rgb_batch_indices.size()) {
                        throw std::runtime_error("JPEG batch decode returned wrong size");
                    }
                    for (size_t j = 0; j < rgb_batch_indices.size(); ++j) {
                        const size_t index = rgb_batch_indices[j];
                        std::shared_ptr<void> lease = leases[j];
                        try_complete_pair(batch[index].sequence_id, batch[index].loader_generation,
                                          std::move(decoded[j]), std::nullopt, nullptr, std::nullopt,
                                          std::nullopt, nullptr, std::move(lease));
                        decoded_as_pair[index] = true;
                    }
                }

                for (size_t i = 0; i < batch.size(); ++i) {
                    if (decoded_as_pair[i]) {
                        continue;
                    }
                    try {
                        batch[i].params.cuda_stream = decode_stream_;
                        if (batch[i].is_depth || batch[i].is_normal) {
                            size_t pair_index = batch.size();
                            for (size_t j = i + 1; j < batch.size(); ++j) {
                                if (!decoded_as_pair[j] &&
                                    batch[j].sequence_id == batch[i].sequence_id &&
                                    batch[j].is_depth != batch[i].is_depth &&
                                    (batch[j].is_depth || batch[j].is_normal) &&
                                    batch[i].jpeg_data && batch[j].jpeg_data) {
                                    pair_index = j;
                                    break;
                                }
                            }

                            if (pair_index != batch.size()) {
                                const std::vector<std::pair<const uint8_t*, size_t>> spans{
                                    {batch[i].jpeg_data->data(), batch[i].jpeg_data->size()},
                                    {batch[pair_index].jpeg_data->data(), batch[pair_index].jpeg_data->size()}};
                                auto decoded = nvcodec->decode_jpeg2k_16bit_batch_from_spans(
                                    spans, decode_stream_, false);
                                if (decoded.size() != 2) {
                                    throw std::runtime_error("JPEG2000 sidecar pair decode returned wrong size");
                                }
                                auto first = prepare_sidecar(batch[i], std::move(decoded[0]));
                                auto second = prepare_sidecar(batch[pair_index], std::move(decoded[1]));
                                complete_sidecar(batch[i], std::move(first));
                                complete_sidecar(batch[pair_index], std::move(second));
                                decoded_as_pair[pair_index] = true;
                            } else {
                                auto sidecar = decode_cached_sidecar(*nvcodec, batch[i], decode_stream_);
                                if (!sidecar.is_valid() || sidecar.numel() == 0) {
                                    LOG_WARN("[PipelinedImageLoader] GPU sidecar decode failed for {}",
                                             lfs::core::path_to_utf8(batch[i].path));
                                    throw std::runtime_error("Invalid sidecar tensor");
                                }
                                complete_sidecar(batch[i], std::move(sidecar));
                            }
                        } else if (batch[i].is_mask) {
                            lfs::core::Tensor mask_tensor;
                            const auto& bytes = *batch[i].jpeg_data;
                            const bool is_jpeg2k = bytes.size() >= 2 && bytes[0] == 0xff && bytes[1] == 0x4f;
                            if (is_jpeg2k) {
                                auto raw = nvcodec->decode_jpeg2k_16bit_from_memory_gpu(
                                    bytes, decode_stream_, false, true);
                                // Pipeline mask sidecars are always encoded as UINT8 below.
                                // The decoder returns normalized Float32 for UINT16 streams;
                                // reinterpreting that output as raw u16 would normalize twice.
                                LFS_ASSERT_MSG(
                                    raw.dtype() == lfs::core::DataType::UInt8,
                                    "pipeline JPEG2000 mask cache must contain eight-bit samples");
                                mask_tensor = raw.to(lfs::core::DataType::Float32) / 255.0f;
                            } else {
                                mask_tensor = nvcodec->load_image_from_memory_gpu(
                                    bytes, 1, 0, decode_stream_, DecodeFormat::Grayscale);
                            }

                            if (!mask_tensor.is_valid() || mask_tensor.numel() == 0) {
                                LOG_WARN("[PipelinedImageLoader] GPU mask decode failed for {}",
                                         lfs::core::path_to_utf8(batch[i].path));
                                throw std::runtime_error("Invalid mask tensor");
                            }

                            if (batch[i].mask_params.invert)
                                mask_tensor = mask_tensor * -1.0f + 1.0f;
                            mask_tensor = process_mask(std::move(mask_tensor), batch[i].mask_params.threshold);
                            try_complete_pair(batch[i].sequence_id, batch[i].loader_generation,
                                              std::nullopt, std::move(mask_tensor), nullptr);

                        } else {
                            const bool decode_from_base_cache = batch[i].needs_processing;
                            auto tensor = nvcodec->load_image_from_memory_gpu(
                                *batch[i].jpeg_data,
                                decode_from_base_cache ? batch[i].params.resize_factor : 1,
                                decode_from_base_cache ? batch[i].params.max_width : 0,
                                batch[i].params.cuda_stream,
                                DecodeFormat::RGB,
                                batch[i].params.output_uint8);

                            if (!tensor.is_valid() || tensor.numel() == 0) {
                                LOG_WARN("[PipelinedImageLoader] GPU decode failed for {}",
                                         lfs::core::path_to_utf8(batch[i].path));
                                throw std::runtime_error("Invalid tensor");
                            }

                            if (decode_from_base_cache) {
                                apply_requested_undistort(tensor, batch[i].params);
                                write_derived_cache(*nvcodec, tensor, batch[i].cache_key,
                                                    batch[i].params.cuda_stream);
                            }

                            try_complete_pair(batch[i].sequence_id, batch[i].loader_generation,
                                              std::move(tensor), std::nullopt, nullptr);
                        }
                    } catch (...) {
                        auto& item = batch[i];
                        const auto message =
                            describe_current_exception("non-standard nvImageCodec exception");
                        if (item.is_cache_hit) {
                            const char* kind = item.is_depth    ? "depth sidecar"
                                               : item.is_normal ? "normal sidecar"
                                               : item.is_mask   ? "mask"
                                                                : "image";
                            LOG_WARN("[PipelinedImageLoader] Cached {} decode failed for {}; "
                                     "evicting cache entry and reprocessing: {}",
                                     kind,
                                     lfs::core::path_to_utf8(item.path),
                                     message);
                            invalidate_cache_entry(item.cache_key);
                        }
                        item.is_cache_hit = false;
                        item.needs_processing = true;
                        if (item.raw_bytes.empty()) {
                            try {
                                item.raw_bytes = read_file(item.path);
                            } catch (...) {
                                const auto failure_message = describe_current_exception(
                                    "failed to read sidecar for decoder fallback");
                                std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                                if (item.is_mask || item.is_depth || item.is_normal) {
                                    const auto kind = item.is_mask    ? SidecarKind::Mask
                                                      : item.is_depth ? SidecarKind::Depth
                                                                      : SidecarKind::Normal;
                                    fail_sidecar_locked(item.sequence_id, item.loader_generation,
                                                        kind, item.path,
                                                        failure_message, lock);
                                } else {
                                    lock.unlock();
                                    publish_image_failure(
                                        item.sequence_id,
                                        item.loader_generation,
                                        item.path,
                                        failure_message);
                                }
                                continue;
                            }
                        }
                        cold_queue_.push(std::move(item));
                    }
                }

                std::lock_guard<std::mutex> lock(stats_mutex_);
                ++stats_.gpu_batch_decodes;
                stats_.total_decode_calls += batch.size();

                record_decode_latency(std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - decode_begin)
                                          .count() /
                                      std::max<size_t>(1, batch.size()));

            } catch (...) {
                LOG_ERROR("[PipelinedImageLoader] Batch decode error: {}",
                          describe_current_exception("non-standard nvImageCodec exception"));
                for (auto& item : batch) {
                    item.is_cache_hit = false;
                    item.needs_processing = true;
                    if (item.raw_bytes.empty()) {
                        try {
                            item.raw_bytes = read_file(item.path);
                        } catch (...) {
                            const auto failure_message = describe_current_exception(
                                "failed to read sidecar for decoder fallback");
                            std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                            if (item.is_mask || item.is_depth || item.is_normal) {
                                const auto kind = item.is_mask    ? SidecarKind::Mask
                                                  : item.is_depth ? SidecarKind::Depth
                                                                  : SidecarKind::Normal;
                                fail_sidecar_locked(item.sequence_id, item.loader_generation,
                                                    kind, item.path,
                                                    failure_message, lock);
                            } else {
                                lock.unlock();
                                publish_image_failure(
                                    item.sequence_id,
                                    item.loader_generation,
                                    item.path,
                                    failure_message);
                            }
                            continue;
                        }
                    }
                    cold_queue_.push(std::move(item));
                }
                record_decode_latency(std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - decode_begin)
                                          .count() /
                                      std::max<size_t>(1, batch.size()));
            }
        }
    }

    void PipelinedImageLoader::cold_process_thread_func(const size_t worker_index) {
        const cudaStream_t sidecar_stream =
            worker_index < sidecar_streams_.size() ? sidecar_streams_[worker_index] : nullptr;
        while (running_) {
            PrefetchedImage item;
            try {
                item = cold_queue_.pop();
            } catch (const std::runtime_error&) {
                break;
            }

            try {
                auto nvcodec = acquire_nvcodec_loader(config_.decoder_pool_size);

                if (item.alpha_as_mask) {
                    auto [img_data, width, height, channels] = lfs::core::load_image_with_alpha(
                        item.path, item.params.resize_factor, item.params.max_width);

                    if (!img_data || channels != 4)
                        throw std::runtime_error("Failed to load RGBA image");

                    const size_t H = static_cast<size_t>(height);
                    const size_t W = static_cast<size_t>(width);

                    auto cpu_tensor = lfs::core::Tensor::from_blob(
                        img_data, lfs::core::TensorShape({H, W, 4}),
                        lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                    auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
                    lfs::core::free_image(img_data);

                    auto rgb = item.params.output_uint8
                                   ? lfs::core::Tensor::empty(
                                         lfs::core::TensorShape({3, H, W}),
                                         lfs::core::Device::CUDA, lfs::core::DataType::UInt8)
                                   : lfs::core::Tensor::empty(
                                         lfs::core::TensorShape({3, H, W}),
                                         lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                    auto alpha = lfs::core::Tensor::empty(
                        lfs::core::TensorShape({H, W}),
                        lfs::core::Device::CUDA, lfs::core::DataType::Float32);

                    if (item.params.output_uint8) {
                        cuda::launch_uint8_rgba_split_to_uint8_rgb_and_float32_alpha(
                            gpu_uint8.ptr<uint8_t>(), rgb.ptr<uint8_t>(), alpha.ptr<float>(),
                            H, W, nullptr);
                    } else {
                        cuda::launch_uint8_rgba_split_to_float32_rgb_and_alpha(
                            gpu_uint8.ptr<uint8_t>(), rgb.ptr<float>(), alpha.ptr<float>(),
                            H, W, nullptr);
                    }

                    gpu_uint8 = lfs::core::Tensor();

                    if (item.undistort) {
                        const bool restore_uint8 = item.params.output_uint8 && rgb.dtype() == lfs::core::DataType::UInt8;
                        if (restore_uint8) {
                            rgb = rgb.to(lfs::core::DataType::Float32) / 255.0f;
                        }
                        const auto scaled = lfs::core::scale_undistort_params(
                            *item.undistort, static_cast<int>(W), static_cast<int>(H));
                        rgb = lfs::core::undistort_image(rgb, scaled, nullptr);
                        alpha = lfs::core::undistort_mask(alpha, scaled, nullptr);
                        if (restore_uint8) {
                            auto uint8_rgb = lfs::core::Tensor::empty(
                                rgb.shape(), lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                            cuda::launch_float32_chw_to_uint8_chw(
                                rgb.ptr<float>(), uint8_rgb.ptr<uint8_t>(),
                                rgb.shape()[1], rgb.shape()[2], rgb.shape()[0], nullptr);
                            rgb = std::move(uint8_rgb);
                        }
                    }

                    if (is_nvcodec_available()) {
                        try {
                            auto rgb_jpeg = nvcodec->encode_to_jpeg(rgb, config_.cache_jpeg_quality, nullptr);
                            put_in_jpeg_cache(item.cache_key,
                                              std::make_shared<std::vector<uint8_t>>(std::move(rgb_jpeg)));

                            const auto alpha_key = make_mask_cache_key(item.path, item.params);
                            auto alpha_jpeg = nvcodec->encode_grayscale_to_jpeg2k(alpha, nullptr, true, true);
                            put_in_jpeg_cache(alpha_key,
                                              std::make_shared<std::vector<uint8_t>>(std::move(alpha_jpeg)));
                        } catch (...) {
                        }
                    }

                    float* const alpha_ptr = alpha.ptr<float>();
                    if (item.alpha_mask_params.invert)
                        cuda::launch_mask_invert(alpha_ptr, H, W, nullptr);
                    if (item.alpha_mask_params.threshold > 0)
                        cuda::launch_mask_threshold(alpha_ptr, H, W, item.alpha_mask_params.threshold, nullptr);
                    alpha = process_mask(std::move(alpha), item.alpha_mask_params.threshold);

                    try_complete_pair(item.sequence_id, item.loader_generation,
                                      std::move(rgb), std::move(alpha), nullptr);

                } else if (item.is_mask || item.is_depth) {
                    const cudaStream_t aux_stream = item.is_depth ? sidecar_stream : nullptr;
                    const lfs::core::CUDAStreamGuard stream_guard(aux_stream);
                    lfs::core::Tensor aux_tensor;
                    bool used_gpu = false;
                    // nvimagecodec and the uint8 path truncate to 8 bits; 16-bit
                    // depth priors must keep their precision.
                    const bool depth_16bit =
                        item.is_depth && stbi_is_16_bit(lfs::core::path_to_utf8(item.path).c_str());

                    // Depth sidecars keep their first-touch 16-bit PNG precision by
                    // decoding with stb before the processed tensor is transcoded.
                    if (is_nvcodec_available() && !item.is_depth) {
                        try {
                            const NvcodecSlotGuard slot;
                            aux_tensor = nvcodec->load_image_gpu(
                                item.path, item.params.resize_factor, item.params.max_width,
                                aux_stream, DecodeFormat::Grayscale);
                            used_gpu = true;
                        } catch (...) {
                        }
                    }

                    if (!used_gpu) {
                        int src_w = 0;
                        int src_h = 0;
                        lfs::core::Tensor gpu_gray;
                        if (depth_16bit) {
                            int channels = 0;
                            stbi_us* const gray16 = stbi_load_16(
                                lfs::core::path_to_utf8(item.path).c_str(),
                                &src_w, &src_h, &channels, 1);
                            if (!gray16)
                                throw std::runtime_error("Failed to decode 16-bit depth");
                            const lfs::core::TensorShape shape(
                                {static_cast<size_t>(src_h), static_cast<size_t>(src_w)});
                            // Float16 is only a 2-byte container for the uint16 samples (no UInt16 dtype).
                            auto cpu_tensor = lfs::core::Tensor::from_blob(
                                gray16, shape, lfs::core::Device::CPU, lfs::core::DataType::Float16);
                            auto gpu_staging = cpu_tensor.to(lfs::core::Device::CUDA, aux_stream);
                            synchronize_async_upload_before_free(aux_stream, "depth");
                            stbi_image_free(gray16);
                            gpu_gray = lfs::core::Tensor::empty(
                                shape, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                            cuda::launch_uint16_hwc_to_float32_hwc(
                                reinterpret_cast<const uint16_t*>(gpu_staging.data_ptr()),
                                gpu_gray.ptr<float>(),
                                static_cast<size_t>(src_h), static_cast<size_t>(src_w), 1,
                                aux_stream);
                        } else {
                            const auto [gray_data, w, h] = load_grayscale_stb(item.path);
                            if (!gray_data)
                                throw std::runtime_error(item.is_mask ? "Failed to decode mask" : "Failed to decode depth");
                            src_w = w;
                            src_h = h;
                            auto cpu_tensor = lfs::core::Tensor::empty(
                                lfs::core::TensorShape({static_cast<size_t>(src_h), static_cast<size_t>(src_w)}),
                                lfs::core::Device::CPU,
                                lfs::core::DataType::UInt8);
                            std::memcpy(cpu_tensor.data_ptr(), gray_data, cpu_tensor.bytes());
                            gpu_gray = cpu_tensor.to(lfs::core::Device::CUDA, aux_stream);
                            stbi_image_free(gray_data);
                        }

                        const auto [target_w, target_h] = sidecar_target_size(item, src_w, src_h);

                        if (target_w != src_w || target_h != src_h) {
                            aux_tensor = lfs::core::lanczos_resize_grayscale(gpu_gray, target_h, target_w, 2, aux_stream);
                        } else if (gpu_gray.dtype() == lfs::core::DataType::Float32) {
                            aux_tensor = std::move(gpu_gray);
                        } else {
                            aux_tensor = lfs::core::Tensor::empty(
                                lfs::core::TensorShape({static_cast<size_t>(target_h), static_cast<size_t>(target_w)}),
                                lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                            cuda::launch_uint8_hw_to_float32_hw(
                                gpu_gray.ptr<uint8_t>(), aux_tensor.ptr<float>(), target_h, target_w, aux_stream);
                        }
                    }

                    if (!aux_tensor.is_valid() || aux_tensor.ndim() != 2) {
                        throw std::runtime_error(
                            item.is_mask ? "Mask preprocessing produced an invalid tensor"
                                         : "Depth preprocessing produced an invalid tensor");
                    }
                    if (item.is_mask && aux_tensor.dtype() == lfs::core::DataType::UInt8) {
                        aux_tensor = aux_tensor.to(lfs::core::DataType::Float32) / 255.0f;
                    }
                    const size_t H = aux_tensor.shape()[0];
                    const size_t W = aux_tensor.shape()[1];

                    if (item.undistort) {
                        const auto scaled = lfs::core::scale_undistort_params(
                            *item.undistort,
                            static_cast<int>(W), static_cast<int>(H));
                        aux_tensor = lfs::core::undistort_mask(aux_tensor, scaled, aux_stream);
                    }

                    if (item.is_mask && is_nvcodec_available()) {
                        try {
                            auto jpeg_bytes = nvcodec->encode_grayscale_to_jpeg2k(aux_tensor, nullptr, true, true);
                            put_in_jpeg_cache(item.cache_key,
                                              std::make_shared<std::vector<uint8_t>>(std::move(jpeg_bytes)));
                        } catch (...) {
                            LOG_DEBUG("[PipelinedImageLoader] Mask cache encode skipped for {}: {}",
                                      lfs::core::path_to_utf8(item.path),
                                      describe_current_exception("non-standard nvImageCodec exception"));
                        }
                    }

                    if (item.is_mask) {
                        float* const mask_ptr = static_cast<float*>(aux_tensor.data_ptr());
                        if (item.mask_params.invert) {
                            cuda::launch_mask_invert(mask_ptr, H, W, aux_stream);
                        }
                        if (item.mask_params.threshold > 0) {
                            cuda::launch_mask_threshold(mask_ptr, H, W, item.mask_params.threshold, aux_stream);
                        }
                        aux_tensor = process_mask(std::move(aux_tensor), item.mask_params.threshold);
                    } else {
                        if (item.aux_target_width > 0 && item.aux_target_height > 0 &&
                            aux_tensor.ndim() == 2 &&
                            (static_cast<int>(aux_tensor.shape()[1]) != item.aux_target_width ||
                             static_cast<int>(aux_tensor.shape()[0]) != item.aux_target_height)) {
                            aux_tensor = lfs::core::lanczos_resize_grayscale(
                                aux_tensor, item.aux_target_height, item.aux_target_width, 2, aux_stream);
                        }
                        aux_tensor = aux_tensor.contiguous();
                    }
                    if (item.is_mask) {
                        try_complete_pair(item.sequence_id, item.loader_generation,
                                          std::nullopt, std::move(aux_tensor), nullptr);
                    } else {
                        write_sidecar_cache(*nvcodec, aux_tensor, item, SidecarCacheFormat::Depth, sidecar_stream);
                        auto ready_event = record_sidecar_ready_event(sidecar_stream);
                        try_complete_pair(
                            item.sequence_id,
                            item.loader_generation,
                            std::nullopt,
                            std::nullopt,
                            nullptr,
                            std::move(aux_tensor),
                            std::nullopt,
                            ready_event);
                    }

                } else if (item.is_normal) {
                    const lfs::core::CUDAStreamGuard stream_guard(sidecar_stream);
                    const std::string path_utf8 = lfs::core::path_to_utf8(item.path);
                    int src_w = 0;
                    int src_h = 0;
                    int channels = 0;
                    // nvimagecodec truncates to 8 bits; 16-bit normal priors
                    // must keep their precision, so both paths decode on CPU.
                    // The v = n*0.5 + 0.5 file encoding (with the sRGB display
                    // transform when the startup probe detected it), Y/Z flip,
                    // and world->camera rotation are all inverted in one GPU
                    // kernel; the loss re-normalizes per pixel, so resampling
                    // shrinkage is fine.
                    const bool normal_16bit = stbi_is_16_bit(path_utf8.c_str());
                    lfs::core::Tensor gpu_staging;
                    if (normal_16bit) {
                        stbi_us* const rgb16 = stbi_load_16(path_utf8.c_str(), &src_w, &src_h, &channels, 3);
                        if (!rgb16)
                            throw std::runtime_error("Failed to decode 16-bit normal map");
                        // Float16 is only a 2-byte container for the uint16 samples (no UInt16 dtype).
                        auto cpu_tensor = lfs::core::Tensor::from_blob(
                            rgb16,
                            lfs::core::TensorShape(
                                {static_cast<size_t>(src_h), static_cast<size_t>(src_w), 3}),
                            lfs::core::Device::CPU, lfs::core::DataType::Float16);
                        gpu_staging = cpu_tensor.to(lfs::core::Device::CUDA, sidecar_stream);
                        synchronize_async_upload_before_free(sidecar_stream, "normal");
                        stbi_image_free(rgb16);
                    } else {
                        stbi_uc* const rgb8 = stbi_load(path_utf8.c_str(), &src_w, &src_h, &channels, 3);
                        if (!rgb8)
                            throw std::runtime_error("Failed to decode normal map");
                        auto cpu_tensor = lfs::core::Tensor::from_blob(
                            rgb8,
                            lfs::core::TensorShape(
                                {static_cast<size_t>(src_h), static_cast<size_t>(src_w), 3}),
                            lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                        gpu_staging = cpu_tensor.to(lfs::core::Device::CUDA, sidecar_stream);
                        synchronize_async_upload_before_free(sidecar_stream, "normal");
                        stbi_image_free(rgb8);
                    }

                    cuda::NormalPriorTransform prior_transform;
                    prior_transform.srgb = item.normal_srgb;
                    prior_transform.flip_yz = item.normal_flip_yz;
                    prior_transform.world_to_camera = item.normal_transform_world_to_camera;
                    std::copy(item.normal_world_to_camera.begin(),
                              item.normal_world_to_camera.end(), prior_transform.w2c);

                    auto normal_tensor = lfs::core::Tensor::empty(
                        lfs::core::TensorShape(
                            {3, static_cast<size_t>(src_h), static_cast<size_t>(src_w)}),
                        lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                    if (normal_16bit) {
                        cuda::launch_normal_prior_u16_hwc_to_float32_chw(
                            reinterpret_cast<const uint16_t*>(gpu_staging.data_ptr()),
                            normal_tensor.ptr<float>(),
                            static_cast<size_t>(src_h), static_cast<size_t>(src_w),
                            prior_transform, sidecar_stream);
                    } else {
                        cuda::launch_normal_prior_u8_hwc_to_float32_chw(
                            gpu_staging.ptr<uint8_t>(),
                            normal_tensor.ptr<float>(),
                            static_cast<size_t>(src_h), static_cast<size_t>(src_w),
                            prior_transform, sidecar_stream);
                    }

                    const auto [target_w, target_h] = sidecar_target_size(item, src_w, src_h);
                    if (target_w != src_w || target_h != src_h) {
                        normal_tensor = lfs::core::lanczos_resize_float_chw(normal_tensor, target_h, target_w, 2, sidecar_stream);
                    }

                    if (!normal_tensor.is_valid() || normal_tensor.ndim() != 3 || normal_tensor.shape()[0] != 3) {
                        throw std::runtime_error("Normal preprocessing produced an invalid tensor");
                    }
                    if (item.undistort) {
                        const auto scaled = lfs::core::scale_undistort_params(
                            *item.undistort,
                            static_cast<int>(normal_tensor.shape()[2]),
                            static_cast<int>(normal_tensor.shape()[1]));
                        normal_tensor = lfs::core::undistort_image(normal_tensor, scaled, sidecar_stream);
                    }

                    if (item.aux_target_width > 0 && item.aux_target_height > 0 &&
                        normal_tensor.ndim() == 3 &&
                        (static_cast<int>(normal_tensor.shape()[2]) != item.aux_target_width ||
                         static_cast<int>(normal_tensor.shape()[1]) != item.aux_target_height)) {
                        normal_tensor = lfs::core::lanczos_resize_float_chw(
                            normal_tensor, item.aux_target_height, item.aux_target_width, 2, sidecar_stream);
                    }
                    normal_tensor = normal_tensor.contiguous();
                    if (!normal_tensor.is_valid() || normal_tensor.ndim() != 3 || normal_tensor.shape()[0] != 3) {
                        throw std::runtime_error("Normal preprocessing produced an invalid tensor");
                    }
                    write_sidecar_cache(*nvcodec, normal_tensor, item, SidecarCacheFormat::Normal, sidecar_stream);
                    auto ready_event = record_sidecar_ready_event(sidecar_stream);

                    try_complete_pair(
                        item.sequence_id,
                        item.loader_generation,
                        std::nullopt,
                        std::nullopt,
                        nullptr,
                        std::nullopt,
                        std::move(normal_tensor),
                        ready_event);

                } else {
                    lfs::core::Tensor decoded;
                    bool used_gpu = false;

                    if (is_nvcodec_available() && item.is_original_jpeg) {
                        try {
                            const NvcodecSlotGuard slot;
                            decoded = nvcodec->load_image_gpu(
                                item.path, item.params.resize_factor, item.params.max_width,
                                nullptr, DecodeFormat::RGB, item.params.output_uint8);
                            used_gpu = true;
                        } catch (...) {
                        }
                    }

                    if (!used_gpu) {
                        decoded = decode_file_on_cpu(item.path, item.params);
                    }

                    if (item.undistort) {
                        const bool restore_uint8 = item.params.output_uint8 && decoded.dtype() == lfs::core::DataType::UInt8;
                        if (restore_uint8) {
                            decoded = decoded.to(lfs::core::DataType::Float32) / 255.0f;
                        }
                        const auto scaled = lfs::core::scale_undistort_params(
                            *item.undistort,
                            static_cast<int>(decoded.shape()[2]),
                            static_cast<int>(decoded.shape()[1]));
                        decoded = lfs::core::undistort_image(decoded, scaled, nullptr);
                        if (restore_uint8) {
                            auto uint8_decoded = lfs::core::Tensor::empty(
                                decoded.shape(), lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                            cuda::launch_float32_chw_to_uint8_chw(
                                decoded.ptr<float>(), uint8_decoded.ptr<uint8_t>(),
                                decoded.shape()[1], decoded.shape()[2], decoded.shape()[0], nullptr);
                            decoded = std::move(uint8_decoded);
                        }
                    }

                    if (is_nvcodec_available()) {
                        write_derived_cache(*nvcodec, decoded, item.cache_key, nullptr);
                    }

                    try_complete_pair(item.sequence_id, item.loader_generation,
                                      std::move(decoded), std::nullopt, nullptr);
                }

            } catch (...) {
                const auto message = describe_current_exception("non-standard image loader exception");
                if (item.alpha_as_mask) {
                    LOG_WARN("[PipelinedImageLoader] Alpha-as-mask failed {}: {} - loading as RGB",
                             lfs::core::path_to_utf8(item.path), message);
                    try {
                        auto [img_data, width, height, channels] = lfs::core::load_image(
                            item.path, item.params.resize_factor, item.params.max_width);
                        if (!img_data)
                            throw std::runtime_error("RGB fallback also failed");

                        const size_t H = static_cast<size_t>(height);
                        const size_t W = static_cast<size_t>(width);
                        const size_t C = static_cast<size_t>(channels);

                        auto cpu_tensor = lfs::core::Tensor::from_blob(
                            img_data, lfs::core::TensorShape({H, W, C}),
                            lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                        auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
                        lfs::core::free_image(img_data);

                        auto decoded = item.params.output_uint8
                                           ? lfs::core::Tensor::empty(
                                                 lfs::core::TensorShape({C, H, W}),
                                                 lfs::core::Device::CUDA, lfs::core::DataType::UInt8)
                                           : lfs::core::Tensor::empty(
                                                 lfs::core::TensorShape({C, H, W}),
                                                 lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                        if (item.params.output_uint8) {
                            cuda::launch_uint8_hwc_to_uint8_chw(
                                gpu_uint8.ptr<uint8_t>(), decoded.ptr<uint8_t>(), H, W, C, nullptr);
                        } else {
                            cuda::launch_uint8_hwc_to_float32_chw(
                                gpu_uint8.ptr<uint8_t>(), decoded.ptr<float>(), H, W, C, nullptr);
                        }
                        {
                            std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                            if (auto it = pending_pairs_.find(item.sequence_id);
                                it != pending_pairs_.end() &&
                                it->second.loader_generation == item.loader_generation) {
                                it->second.mask_failed = true;
                                it->second.mask_expected = false;
                            }
                        }
                        try_complete_pair(item.sequence_id, item.loader_generation,
                                          std::move(decoded), std::nullopt, nullptr);
                    } catch (...) {
                        const auto fallback_message =
                            describe_current_exception("non-standard RGB fallback exception");
                        LOG_ERROR("[PipelinedImageLoader] RGB fallback also failed {}: {}",
                                  lfs::core::path_to_utf8(item.path),
                                  fallback_message);
                        publish_image_failure(item.sequence_id, item.loader_generation,
                                              item.path, fallback_message);
                    }
                } else if (item.is_mask || item.is_depth || item.is_normal) {
                    LOG_WARN("[PipelinedImageLoader] Cold process {} error {}: {} - continuing without it",
                             item.is_mask ? "mask" : (item.is_depth ? "depth" : "normal"),
                             lfs::core::path_to_utf8(item.path), message);
                    std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                    const auto kind = item.is_mask    ? SidecarKind::Mask
                                      : item.is_depth ? SidecarKind::Depth
                                                      : SidecarKind::Normal;
                    fail_sidecar_locked(item.sequence_id, item.loader_generation,
                                        kind, item.path, message, lock);
                } else {
                    LOG_ERROR("[PipelinedImageLoader] Cold process error {}: {}",
                              lfs::core::path_to_utf8(item.path), message);
                    publish_image_failure(item.sequence_id, item.loader_generation,
                                          item.path, message);
                }
            }
        }
    }

} // namespace lfs::io
