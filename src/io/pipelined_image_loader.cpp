/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/pipelined_image_loader.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "cuda/image_format_kernels.cuh"
#include "io/nvcodec_image_loader.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <random>

namespace lfs::io {

    namespace {

        constexpr int CACHE_HASH_LENGTH = 8;
        constexpr int DECODER_POOL_SIZE = 8;

        std::string generate_cache_hash() {
            static constexpr char HEX_CHARS[] = "0123456789abcdef";

            // Thread-safe: use local RNG objects to avoid data races
            thread_local std::random_device rd;
            thread_local std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 15);

            std::string hash;
            hash.reserve(CACHE_HASH_LENGTH);
            for (int i = 0; i < CACHE_HASH_LENGTH; ++i) {
                hash += HEX_CHARS[dis(gen)];
            }
            return hash;
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

        std::mutex& get_nvcodec_mutex() {
            static std::mutex mtx;
            return mtx;
        }

        std::unique_ptr<NvCodecImageLoader>& get_nvcodec_instance() {
            static std::unique_ptr<NvCodecImageLoader> instance;
            return instance;
        }

        NvCodecImageLoader& get_nvcodec_loader() {
            std::lock_guard<std::mutex> lock(get_nvcodec_mutex());
            auto& instance = get_nvcodec_instance();
            if (!instance) {
                NvCodecImageLoader::Options opts;
                opts.device_id = 0;
                opts.decoder_pool_size = DECODER_POOL_SIZE;
                opts.enable_fallback = true;
                instance = std::make_unique<NvCodecImageLoader>(opts);
            }
            return *instance;
        }

        void reset_nvcodec_loader() {
            std::lock_guard<std::mutex> lock(get_nvcodec_mutex());
            get_nvcodec_instance().reset();
        }

        bool is_nvcodec_available() {
            static std::once_flag flag;
            static bool available = false;
            std::call_once(flag, [] { available = NvCodecImageLoader::is_available(); });
            return available;
        }

    } // namespace

    PipelinedImageLoader::PipelinedImageLoader(PipelinedLoaderConfig config)
        : config_(std::move(config)) {

        LOG_INFO("[PipelinedImageLoader] batch_size={}, prefetch={}, io_threads={}, cold_threads={}",
                 config_.jpeg_batch_size, config_.prefetch_count, config_.io_threads, config_.cold_process_threads);

        if (config_.use_filesystem_cache) {
            const auto cache_base = get_temp_folder() / "LichtFeld" / "pipeline_cache";
            fs_cache_folder_ = cache_base / ("ppl_" + generate_cache_hash());

            std::error_code ec;
            std::filesystem::create_directories(fs_cache_folder_, ec);
            if (ec) {
                LOG_WARN("[PipelinedImageLoader] Cache folder creation failed: {}", ec.message());
                config_.use_filesystem_cache = false;
            }
        }

        running_ = true;

        for (size_t i = 0; i < config_.io_threads; ++i) {
            io_threads_.emplace_back([this] { prefetch_thread_func(); });
        }

        if (is_nvcodec_available()) {
            gpu_decode_thread_ = std::thread([this] { gpu_batch_decode_thread_func(); });
        }

        for (size_t i = 0; i < config_.cold_process_threads; ++i) {
            cold_process_threads_.emplace_back([this] { cold_process_thread_func(); });
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

        cudaDeviceSynchronize();
        reset_nvcodec_loader();

        if (config_.use_filesystem_cache && !fs_cache_folder_.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(fs_cache_folder_, ec);
        }

        LOG_INFO("[PipelinedImageLoader] Done: {} loaded, {} hits, {} misses",
                 stats_.total_images_loaded, stats_.hot_path_hits, stats_.cold_path_misses);
    }

    void PipelinedImageLoader::prefetch(const std::vector<ImageRequest>& requests) {
        for (const auto& req : requests) {
            prefetch_queue_.push(req);
            in_flight_.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    void PipelinedImageLoader::prefetch(size_t sequence_id, const std::filesystem::path& path, const LoadParams& params) {
        prefetch_queue_.push({sequence_id, path, params});
        in_flight_.fetch_add(1, std::memory_order_acq_rel);
    }

    ReadyImage PipelinedImageLoader::get() {
        auto result = output_queue_.pop();
        in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        return result;
    }

    std::optional<ReadyImage> PipelinedImageLoader::try_get() {
        auto result = output_queue_.try_pop();
        if (result)
            in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        return result;
    }

    std::optional<ReadyImage> PipelinedImageLoader::try_get_for(std::chrono::milliseconds timeout) {
        auto result = output_queue_.try_pop_for(timeout);
        if (result)
            in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        return result;
    }

    size_t PipelinedImageLoader::ready_count() const {
        return output_queue_.size();
    }

    size_t PipelinedImageLoader::in_flight_count() const {
        return in_flight_.load();
    }

    void PipelinedImageLoader::clear() {
        prefetch_queue_.clear();
        hot_queue_.clear();
        cold_queue_.clear();
        output_queue_.clear();
        in_flight_ = 0;
    }

    PipelinedImageLoader::CacheStats PipelinedImageLoader::get_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        CacheStats s = stats_;
        s.jpeg_cache_entries = jpeg_cache_.size();
        s.jpeg_cache_bytes = jpeg_cache_bytes_.load();
        return s;
    }

    std::string PipelinedImageLoader::make_cache_key(const std::filesystem::path& path, const LoadParams& params) const {
        return path.string() + ":rf" + std::to_string(params.resize_factor) + "_mw" + std::to_string(params.max_width);
    }

    std::filesystem::path PipelinedImageLoader::get_fs_cache_path(const std::string& cache_key) const {
        // Use the full cache_key as filename to avoid hash collisions
        // Replace path separators and colons with underscores for filesystem safety
        std::string safe_name = cache_key;
        for (char& c : safe_name) {
            if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
                c = '_';
            }
        }

        // If name is too long (>200 chars), use hash of the full key as prefix + last 100 chars
        // This ensures uniqueness while keeping reasonable filename length
        if (safe_name.length() > 200) {
            const auto hash = std::hash<std::string>{}(cache_key);
            safe_name = std::to_string(hash) + "_" + safe_name.substr(safe_name.length() - 100);
        }

        return fs_cache_folder_ / (safe_name + ".jpg");
    }

    bool PipelinedImageLoader::is_jpeg_data(const std::vector<uint8_t>& data) const {
        if (data.size() < 3)
            return false;
        return data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;
    }

    std::vector<uint8_t> PipelinedImageLoader::read_file(const std::filesystem::path& path) const {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("Failed to open: " + path.string());

        const auto size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read: " + path.string());
        }
        return buffer;
    }

    std::shared_ptr<std::vector<uint8_t>> PipelinedImageLoader::get_from_jpeg_cache(const std::string& cache_key) {
        std::lock_guard<std::mutex> lock(jpeg_cache_mutex_);
        const auto it = jpeg_cache_.find(cache_key);
        if (it == jpeg_cache_.end())
            return nullptr;
        it->second.last_access = std::chrono::steady_clock::now();
        return it->second.data;
    }

    void PipelinedImageLoader::put_in_jpeg_cache(const std::string& cache_key, std::shared_ptr<std::vector<uint8_t>> data) {
        std::lock_guard<std::mutex> lock(jpeg_cache_mutex_);
        const size_t size = data->size();
        evict_jpeg_cache_if_needed(size);
        jpeg_cache_[cache_key] = JpegCacheEntry{std::move(data), std::chrono::steady_clock::now(), size};
        jpeg_cache_bytes_ += size;
    }

    void PipelinedImageLoader::put_in_jpeg_cache(const std::string& cache_key, std::vector<uint8_t>&& data) {
        put_in_jpeg_cache(cache_key, std::make_shared<std::vector<uint8_t>>(std::move(data)));
    }

    void PipelinedImageLoader::evict_jpeg_cache_if_needed(size_t required_bytes) {
        size_t target = config_.max_cache_bytes;
        const size_t available = get_available_physical_memory();
        const size_t min_free = static_cast<size_t>(get_total_physical_memory() * config_.min_free_memory_ratio);

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
            jpeg_cache_bytes_ -= oldest->second.size_bytes;
            jpeg_cache_.erase(oldest);
        }
    }

    void PipelinedImageLoader::save_to_fs_cache(const std::string& cache_key, const std::vector<uint8_t>& data) {
        if (!config_.use_filesystem_cache)
            return;

        std::lock_guard<std::mutex> lock(fs_cache_mutex_);
        if (files_being_written_.contains(cache_key))
            return;
        files_being_written_.insert(cache_key);

        const auto path = get_fs_cache_path(cache_key);
        std::ofstream file(path, std::ios::binary);
        if (file) {
            file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
            if (!file.good()) {
                LOG_WARN("[PipelinedImageLoader] Failed to write cache file: {}", path.string());
            } else {
                file.close();
                std::ofstream done_file(path.string() + ".done");
                if (!done_file.good()) {
                    LOG_WARN("[PipelinedImageLoader] Failed to create .done marker: {}", path.string());
                }
            }
        } else {
            LOG_WARN("[PipelinedImageLoader] Failed to open cache file for writing: {}", path.string());
        }
        files_being_written_.erase(cache_key);
    }

    void PipelinedImageLoader::prefetch_thread_func() {
        while (running_) {
            ImageRequest request;
            try {
                request = prefetch_queue_.pop();
            } catch (const std::runtime_error&) {
                break;
            }

            PrefetchedImage result;
            result.sequence_id = request.sequence_id;
            result.path = request.path;
            result.params = request.params;
            result.cache_key = make_cache_key(request.path, request.params);

            try {
                if (auto cached = get_from_jpeg_cache(result.cache_key)) {
                    result.jpeg_data = cached;
                    result.is_cache_hit = true;
                    hot_queue_.push(std::move(result));
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    ++stats_.hot_path_hits;
                    continue;
                }

                if (config_.use_filesystem_cache) {
                    const auto fs_path = get_fs_cache_path(result.cache_key);
                    if (std::filesystem::exists(fs_path) && std::filesystem::exists(fs_path.string() + ".done")) {
                        auto data = std::make_shared<std::vector<uint8_t>>(read_file(fs_path));
                        put_in_jpeg_cache(result.cache_key, data);
                        result.jpeg_data = data;
                        result.is_cache_hit = true;
                        hot_queue_.push(std::move(result));
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        ++stats_.hot_path_hits;
                        continue;
                    }
                }

                result.raw_bytes = read_file(request.path);
                result.is_original_jpeg = is_jpeg_data(result.raw_bytes);
                result.is_cache_hit = false;

                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.total_bytes_read += result.raw_bytes.size();
                }

                const bool needs_resize = (request.params.resize_factor > 1 || request.params.max_width > 0);
                if (result.is_original_jpeg && !needs_resize) {
                    auto data = std::make_shared<std::vector<uint8_t>>(std::move(result.raw_bytes));
                    put_in_jpeg_cache(result.cache_key, data);
                    result.jpeg_data = data;
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
            } catch (const std::exception& e) {
                LOG_ERROR("[PipelinedImageLoader] Prefetch error {}: {}", request.path.string(), e.what());
                in_flight_.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }

    void PipelinedImageLoader::gpu_batch_decode_thread_func() {
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

            while (batch.size() < config_.jpeg_batch_size) {
                const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now());
                if (remaining.count() <= 0)
                    break;
                auto item = hot_queue_.try_pop_for(remaining);
                if (!item)
                    break;
                batch.push_back(std::move(*item));
            }

            if (batch.empty())
                continue;

            try {
                auto& nvcodec = get_nvcodec_loader();

                std::vector<std::pair<const uint8_t*, size_t>> jpeg_spans;
                jpeg_spans.reserve(batch.size());
                for (const auto& item : batch) {
                    jpeg_spans.emplace_back(item.jpeg_data->data(), item.jpeg_data->size());
                }

                auto tensors = nvcodec.batch_decode_from_spans(jpeg_spans, nullptr);

                for (size_t i = 0; i < tensors.size(); ++i) {
                    if (tensors[i].is_valid() && tensors[i].numel() > 0) {
                        output_queue_.push({batch[i].sequence_id, std::move(tensors[i]), nullptr});
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        ++stats_.total_images_loaded;
                    } else {
                        LOG_WARN("[PipelinedImageLoader] Decode failed for {}", batch[i].path.string());
                        auto& item = batch[i];
                        item.is_cache_hit = false;
                        item.needs_processing = true;
                        if (item.raw_bytes.empty()) {
                            try {
                                item.raw_bytes = read_file(item.path);
                            } catch (...) {
                                in_flight_.fetch_sub(1, std::memory_order_acq_rel);
                                continue;
                            }
                        }
                        cold_queue_.push(std::move(item));
                    }
                }

                std::lock_guard<std::mutex> lock(stats_mutex_);
                ++stats_.gpu_batch_decodes;
                stats_.total_decode_calls += batch.size();

            } catch (const std::exception& e) {
                LOG_ERROR("[PipelinedImageLoader] Batch decode error: {}", e.what());
                for (auto& item : batch) {
                    item.is_cache_hit = false;
                    item.needs_processing = true;
                    if (item.raw_bytes.empty()) {
                        try {
                            item.raw_bytes = read_file(item.path);
                        } catch (...) {
                            in_flight_.fetch_sub(1, std::memory_order_acq_rel);
                            continue;
                        }
                    }
                    cold_queue_.push(std::move(item));
                }
            }
        }
    }

    void PipelinedImageLoader::cold_process_thread_func() {
        while (running_) {
            PrefetchedImage item;
            try {
                item = cold_queue_.pop();
            } catch (const std::runtime_error&) {
                break;
            }

            try {
                lfs::core::Tensor decoded;
                auto& nvcodec = get_nvcodec_loader();
                bool used_gpu = false;

                if (is_nvcodec_available() && item.is_original_jpeg) {
                    try {
                        decoded = nvcodec.load_image_gpu(item.path, item.params.resize_factor, item.params.max_width);
                        used_gpu = true;
                    } catch (const std::exception&) {
                        // Fall back to CPU
                    }
                }

                if (!used_gpu) {
                    auto [img_data, width, height, channels] = lfs::core::load_image(
                        item.path, item.params.resize_factor, item.params.max_width);

                    if (!img_data)
                        throw std::runtime_error("Failed to decode image");

                    const size_t H = static_cast<size_t>(height);
                    const size_t W = static_cast<size_t>(width);
                    const size_t C = static_cast<size_t>(channels);

                    auto cpu_tensor = lfs::core::Tensor::from_blob(
                        img_data, lfs::core::TensorShape({H, W, C}),
                        lfs::core::Device::CPU, lfs::core::DataType::UInt8);

                    auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
                    lfs::core::free_image(img_data);

                    decoded = lfs::core::Tensor::zeros(
                        lfs::core::TensorShape({C, H, W}),
                        lfs::core::Device::CUDA, lfs::core::DataType::Float32);

                    cuda::launch_uint8_hwc_to_float32_chw(
                        reinterpret_cast<const uint8_t*>(gpu_uint8.data_ptr()),
                        reinterpret_cast<float*>(decoded.data_ptr()),
                        H, W, C, nullptr);

                    // Ensure kernel completes before returning tensor to avoid race conditions
                    cudaDeviceSynchronize();

                    gpu_uint8 = lfs::core::Tensor();
                }

                if (is_nvcodec_available()) {
                    try {
                        auto jpeg_bytes = nvcodec.encode_to_jpeg(decoded, config_.cache_jpeg_quality, nullptr);
                        put_in_jpeg_cache(item.cache_key, std::make_shared<std::vector<uint8_t>>(std::move(jpeg_bytes)));
                    } catch (const std::exception&) {
                        // Continue without caching
                    }
                }

                output_queue_.push({item.sequence_id, std::move(decoded), nullptr});

                std::lock_guard<std::mutex> lock(stats_mutex_);
                ++stats_.total_images_loaded;

            } catch (const std::exception& e) {
                LOG_ERROR("[PipelinedImageLoader] Cold process error {}: {}", item.path.string(), e.what());
                in_flight_.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }

} // namespace lfs::io
