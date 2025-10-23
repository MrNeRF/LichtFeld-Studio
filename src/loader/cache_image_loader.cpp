/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "loader/cache_image_loader.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "project/project.hpp"

#include <fstream>

// Platform-specific includes
#ifdef __linux__
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace gs::loader {

    std::size_t get_total_physical_memory() {
#ifdef __linux__
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            return info.totalram * info.mem_unit;
        }
        LOG_WARN("Failed to get total memory on Linux");
#elif defined(_WIN32)
        MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&mem_info)) {
            return mem_info.ullTotalPhys;
        }
        LOG_WARN("Failed to get total memory on Windows");
#else
        LOG_WARN("Unsupported platform for memory detection");
#endif
        // Fallback: assume 16GB
        return 16ULL * 1024 * 1024 * 1024;
    }

    std::size_t get_available_physical_memory() {
#ifdef __linux__
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            return info.freeram * info.mem_unit;
        }
        LOG_WARN("Failed to get available memory on Linux");
#elif defined(_WIN32)
        MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&mem_info)) {
            return mem_info.ullAvailPhys;
        }
        LOG_WARN("Failed to get available memory on Windows");
#else
        LOG_WARN("Unsupported platform for memory detection");
#endif
        // Fallback: assume 1GB available
        return 1ULL * 1024 * 1024 * 1024;
    }

    double get_memory_usage_ratio() {
        std::size_t total = get_total_physical_memory();
        std::size_t available = get_available_physical_memory();

        if (total == 0) {
            return 1.0; // Assume full if we can't determine
        }

        return 1.0 - (static_cast<double>(available) / static_cast<double>(total));
    }

    std::unique_ptr<CacheLoader> CacheLoader::instance_ = nullptr;
    std::once_flag CacheLoader::init_flag_;

    static bool create_done_file(const std::filesystem::path& img_path) {
        try {
            auto done_path = img_path;
            done_path += ".done";

            // Create or overwrite the .done file
            std::ofstream ofs(done_path, std::ios::trunc);
            if (!ofs) {
                return false; // Failed to open for writing
            }

            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Exception in create_done_file for {} img_path : {} ", img_path.string(), e.what());
            return false;
        } catch (...) {
            LOG_ERROR("Unknown exception in create_done_file for {}", img_path.string());
            return false;
        }
    }

    static bool does_cache_image_exists(const std::filesystem::path& img_path) {
        // Assume cached image has same base name but with ".done" marker
        auto done_path = img_path;
        done_path += ".done";

        // Both the main file and its ".done" marker must exist
        return std::filesystem::exists(img_path) &&
               std::filesystem::exists(done_path);
    }

    CacheLoader::CacheLoader(bool use_cpu_memory, bool use_fs_cache) : use_cpu_memory_(use_cpu_memory), use_fs_cache_(use_fs_cache) {
        create_new_cache_folder();
        if (min_cpu_free_memory_ratio_ < 0 || min_cpu_free_memory_ratio_ > 1) {
            LOG_WARN("min_cpu_free_memory_ratio_ is outside [0,1] interval = {}", min_cpu_free_memory_ratio_);
            min_cpu_free_memory_ratio_ = std::clamp(min_cpu_free_memory_ratio_, 0.0f, 1.0f);
        }
    }

    void CacheLoader::create_new_cache_folder() {
        if (!use_fs_cache_) {
            return;
        }

        auto cache_base = gs::management::GetLichtFeldBaseTemporaryFolder() / "cache";
        std::string unique_cache_path = LFS_CACHE_PREFIX + gs::management::generateShortHash();
        std::filesystem::path cache_folder = cache_base / unique_cache_path;

        std::error_code ec;

        // Check if parent path exists and is valid
        if (!std::filesystem::exists(cache_base.parent_path())) {
            LOG_ERROR("Cache base path does not exist: {}. Disabling filesystem cache.",
                     cache_base.parent_path().string());
            use_fs_cache_ = false;
            return;
        }

        // If cache folder already exists, wipe it
        if (std::filesystem::exists(cache_folder)) {
            LOG_DEBUG("Cache folder {} already exists, wiping it", cache_folder.string());
            std::filesystem::remove_all(cache_folder, ec);
            if (ec) {
                LOG_ERROR("Failed to wipe existing cache folder {}: {}. Disabling filesystem cache.",
                         cache_folder.string(), ec.message());
                use_fs_cache_ = false;
                return;
            }
        }

        // Create the cache folder
        std::filesystem::create_directories(cache_folder, ec);
        if (ec) {
            LOG_ERROR("Failed to create cache directory {}: {}. Disabling filesystem cache.",
                     cache_folder.string(), ec.message());
            use_fs_cache_ = false;
            return;
        }

        cache_folder_ = cache_folder;
        LOG_DEBUG("Created cache directory: {}", cache_folder.string());
    }

    void CacheLoader::reset_cache() {
        clear_cpu_cache();
        clean_cache_folders();
        create_new_cache_folder();

        cache_mode_ = CacheMode::Undetermined;
        num_expected_images_ = 0;
    }

    void CacheLoader::clean_cache_folders() {
        auto cache_base = gs::management::GetLichtFeldBaseTemporaryFolder() / "cache";
        if (!std::filesystem::exists(cache_base) || !std::filesystem::is_directory(cache_base)) {
            LOG_ERROR("Invalid base folder: {}", cache_base.string());
            return;
        }

        for (const auto& entry : std::filesystem::directory_iterator(cache_base)) {
            if (entry.is_directory()) {
                auto folder_name = entry.path().filename().string();

                if (folder_name.rfind(LFS_CACHE_PREFIX, 0) == 0) { // starts with prefix
                    if (std::filesystem::exists(entry.path() / ".lock")) {
                        LOG_DEBUG("folder: {} exists, but it is locked", entry.path().string());
                        continue;
                    }
                    std::error_code ec;
                    std::filesystem::remove_all(entry.path(), ec);
                    if (ec) {
                        LOG_ERROR("Failed to remove {}:{}", entry.path().string(), ec.message());
                        continue;
                    }
                    LOG_DEBUG("Removed folder: {}", entry.path().string());
                }
            }
        }
    }

    void CacheLoader::clear_cpu_cache() {
        std::lock_guard<std::mutex> lock(cpu_cache_mutex_);
        cpu_cache_.clear();
    }

    bool CacheLoader::has_sufficient_memory(std::size_t required_bytes) const {
        std::size_t available = get_available_physical_memory();
        std::size_t total = get_total_physical_memory();

        std::size_t min_free_bytes = std::max(
            static_cast<std::size_t>(total * min_cpu_free_memory_ratio_),
            static_cast<std::size_t>(min_cpu_free_GB_ * 1024ULL * 1024 * 1024));

        // Check if adding this image would leave sufficient free memory
        return (available > required_bytes + min_free_bytes);
    }

    void CacheLoader::evict_until_satisfied() {

        while (!cpu_cache_.empty()) {
            std::size_t available = get_available_physical_memory();
            std::size_t total = get_total_physical_memory();
            std::size_t min_free_bytes = std::max(
                static_cast<std::size_t>(total * min_cpu_free_memory_ratio_),
                static_cast<std::size_t>(min_cpu_free_GB_ * 1024ULL * 1024 * 1024));

            if (available > min_free_bytes) {
                break;
            }
            {
                std::lock_guard<std::mutex> lock(cpu_cache_mutex_);
                auto oldest = std::min_element(cpu_cache_.begin(), cpu_cache_.end(),
                                               [](const auto& a, const auto& b) {
                                                   return a.second.last_access < b.second.last_access;
                                               });
                LOG_DEBUG("Evicting cached image {} ({} bytes) from CPU cache",
                          oldest->first, oldest->second.size_bytes);
                cpu_cache_.erase(oldest);
            }
        }
    }

    void CacheLoader::evict_if_needed(std::size_t required_bytes) {
        // LRU eviction: remove least recently accessed images until we have space
        while (!cpu_cache_.empty() && !has_sufficient_memory(required_bytes)) {
            auto oldest = std::min_element(cpu_cache_.begin(), cpu_cache_.end(),
                                           [](const auto& a, const auto& b) {
                                               return a.second.last_access < b.second.last_access;
                                           });

            if (oldest != cpu_cache_.end()) {
                LOG_DEBUG("Evicting cached image {} ({} bytes) from CPU cache",
                          oldest->first, oldest->second.size_bytes);
                cpu_cache_.erase(oldest);
            } else {
                break;
            }
        }
    }

    std::size_t CacheLoader::get_cpu_cache_size() const {
        std::size_t total = 0;
        for (const auto& [key, data] : cpu_cache_) {
            total += data.size_bytes;
        }
        return total;
    }

    std::string CacheLoader::generate_cache_key(const std::filesystem::path& path, const LoadParams& params) const {
        return std::format("{}:resize_{}_maxw_{}", path.string(), params.resize_factor, params.max_width);
    }

    std::tuple<unsigned char*, int, int, int> CacheLoader::load_cached_image_from_cpu(
        const std::filesystem::path& path,
        const LoadParams& params) {

        std::string cache_key = generate_cache_key(path, params);

        // Check if image is already in CPU cache
        {
            std::lock_guard<std::mutex> lock(cpu_cache_mutex_);
            auto it = cpu_cache_.find(cache_key);
            if (it != cpu_cache_.end()) {
                // Update last access time
                it->second.last_access = std::chrono::steady_clock::now();

                // Allocate new memory and copy cached data
                auto& cached = it->second;
                // allocation should be like load_img since use call img_free
                unsigned char* img_data = static_cast<unsigned char*>(std::malloc(cached.data.size()));
                std::memcpy(img_data, cached.data.data(), cached.data.size());

                LOG_DEBUG("Loaded image {} from CPU cache", cache_key);
                return {img_data, cached.width, cached.height, cached.channels};
            }
        }

        // Image not in cache, need to load it
        bool is_image_being_loaded = false;
        {
            std::lock_guard<std::mutex> lock(cpu_cache_mutex_);
            if (image_being_loaded_cpu_.find(cache_key) != image_being_loaded_cpu_.end()) {
                is_image_being_loaded = true;
            }
            if (!is_image_being_loaded) {
                image_being_loaded_cpu_.insert(cache_key);
            }
        }

        // If another thread is loading, fall back to direct load

        if (is_image_being_loaded) {
            LOG_DEBUG("Image {} is being loaded by another thread, loading directly", cache_key);
            return load_image(path, params.resize_factor, params.max_width);
        }

        // Load the image

        auto [img_data, width, height, channels] = load_image(path, params.resize_factor, params.max_width);

        if (img_data == nullptr) {
            std::lock_guard<std::mutex> lock(cpu_cache_mutex_);
            image_being_loaded_cpu_.erase(cache_key);
            throw std::runtime_error("Failed to load image: " + path.string());
        }

        // Calculate image size
        std::size_t img_size = static_cast<std::size_t>(width) * height * channels;

        // Try to cache the image in CPU memory
        {
            std::lock_guard<std::mutex> lock(cpu_cache_mutex_);

            // Check if we have sufficient memory
            if (has_sufficient_memory(img_size)) {
                // Evict old entries if needed
                evict_if_needed(img_size);

                // Store in cache
                CachedImageData cached_data;
                cached_data.data.resize(img_size);
                std::memcpy(cached_data.data.data(), img_data, img_size);
                cached_data.width = width;
                cached_data.height = height;
                cached_data.channels = channels;
                cached_data.size_bytes = img_size;
                cached_data.last_access = std::chrono::steady_clock::now();

                cpu_cache_[cache_key] = std::move(cached_data);

                LOG_DEBUG("Cached image {} in CPU memory ({} bytes, total cache: {} bytes)",
                          cache_key, img_size, get_cpu_cache_size());
            } else {
                LOG_DEBUG("Insufficient memory to cache image {} ({} bytes required)",
                          cache_key, img_size);
            }

            image_being_loaded_cpu_.erase(cache_key);
        }

        evict_until_satisfied();

        return {img_data, width, height, channels};
    }

    std::tuple<unsigned char*, int, int, int> CacheLoader::load_cached_image_from_fs(const std::filesystem::path& path, const LoadParams& params) {
        if (cache_folder_.empty()) {
            return load_image(path, params.resize_factor, params.max_width);
        }

        std::string unique_name = std::format("rf_{}_mw_{}_", params.resize_factor, params.max_width) + path.filename().string();

        auto cache_img_path = cache_folder_ / unique_name;

        std::tuple<unsigned char*, int, int, int> result;
        if (does_cache_image_exists(cache_img_path)) {
            // Load image synchronously
            result = load_image(cache_img_path);
        } else {

            result = load_image(path, params.resize_factor, params.max_width);

            // we want only one thread to save the data - if we enter this scope either we save the image or it exists
            bool is_image_being_saved = false;
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                if (image_being_saved_.find(path.string()) != image_being_saved_.end()) {
                    is_image_being_saved = true;
                }
                if (!is_image_being_saved) {
                    image_being_saved_.insert(path.string());
                }
            }

            if (!is_image_being_saved) { // only one thread should enter this scope
                bool success = save_img_data(cache_img_path, result);

                if (!success) {
                    throw std::runtime_error("failed saving image" + path.filename().string() + " in cache folder " + cache_folder_.string());
                }
                LOG_INFO("successfully saved image {} in cache folder", cache_img_path.string());
                success = create_done_file(cache_img_path);

                if (!success) {
                    throw std::runtime_error("failed saving .done file " + path.filename().string() + " in cache folder " + cache_folder_.string());
                }
                image_being_saved_.erase(path.string());
            }
        }

        return result;
    }

    void CacheLoader::determine_cache_mode(const std::filesystem::path& path, const LoadParams& params) {
        if (cache_mode_ != CacheMode::Undetermined) {
            return; // cache where already determined
        }
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cache_mode_ != CacheMode::Undetermined) {
            return; // cache where already determined by another thread
        } else {
            if (!use_cpu_memory_ && !use_fs_cache_) {
                LOG_INFO("No caching by user");
                cache_mode_ = CacheMode::NoCache;
                return;
            }

            if (num_expected_images_ <= 0) {
                LOG_ERROR("unable to determine if full cpu caching num_expected_images_ is not initialized. skip caching.");
                cache_mode_ = CacheMode::NoCache;
                return;
            }
            clear_cpu_cache(); // cache does not suppose to be occupied

            auto [img_data, width, height, channels] = load_image(path, params.resize_factor, params.max_width);

            free_image(img_data);

            std::size_t img_size = static_cast<std::size_t>(width) * height * channels;
            std::size_t required_bytes = img_size * num_expected_images_;
            if (use_cpu_memory_ && has_sufficient_memory(required_bytes)) {
                LOG_INFO("all images can be cached in RAM. Cache Loader will be in CPU memory mode.");
                cache_mode_ = CacheMode::CPU_memory;
            } else {
                size_t giga_to_bytes = 1024ULL * 1024 * 1024;
                double required_GB = static_cast<double>(required_bytes) / static_cast<double>(giga_to_bytes);
                double available_GB = static_cast<double>(get_available_physical_memory()) / static_cast<double>(giga_to_bytes);

                if (use_cpu_memory_) {
                    LOG_INFO("not all images fit to cpu memory required_GB {:.2f}. available {:.2f}", required_GB, available_GB);
                } else {
                    LOG_INFO("skip cpu memory cache by user");
                }

                auto [org_width, org_height, org_channels] = get_image_info(path);
                if (use_fs_cache_ && (params.resize_factor > 1 || params.max_width < org_width)) {
                    LOG_INFO("Images are preprocessed. Changing Cache to FS Mode");
                    cache_mode_ = CacheMode::FileSystem;
                } else {
                    if (!use_fs_cache_) {
                        LOG_INFO("No caching by user");
                    } else {
                        LOG_INFO("Images are not preprocessed. no need for caching");
                    }
                    cache_mode_ = CacheMode::NoCache;
                }
            }
        }
    }

    std::tuple<unsigned char*, int, int, int> CacheLoader::load_cached_image(const std::filesystem::path& path, const LoadParams& params) {
        determine_cache_mode(path, params);

        if (use_cpu_memory_ && cache_mode_ == CacheMode::CPU_memory) {
            print_cache_status();
            return load_cached_image_from_cpu(path, params);
        }
        if (use_fs_cache_ && cache_mode_ == CacheMode::FileSystem) {
            return load_cached_image_from_fs(path, params);
        }

        return load_image(path, params.resize_factor, params.max_width);
    }

    void CacheLoader::print_cache_status() const {
        if (!print_cache_status_) {
            return;
        }

        std::lock_guard<std::mutex> lock(counter_mutex_);
        load_counter_++;
        size_t giga_to_bytes = 1024ULL * 1024 * 1024;
        if (load_counter_ > print_status_freq_num_) {
            load_counter_ = 0;
            double total_memory_gb = (double)get_total_physical_memory() / giga_to_bytes;
            double memory_ratio = get_memory_usage_ratio();
            double cache_ratio = (double)get_cpu_cache_size() / (double)get_total_physical_memory();

            LOG_TRACE("CacheInfo: Num images in cache {}", cpu_cache_.size());
            LOG_TRACE("CacheInfo: total memory {:.2f}GB", total_memory_gb);
            LOG_TRACE("CacheInfo: used memory {:.2f}%", 100 * memory_ratio);
            LOG_TRACE("CacheInfo: cache memory occupancy {:.2f}%", 100 * cache_ratio);
            LOG_TRACE("*****"); // seperator
        }
    }

} // namespace gs::loader