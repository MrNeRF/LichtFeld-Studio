/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace lfs::core {

    LFS_CORE_API std::tuple<int, int, int>
    get_image_info(std::filesystem::path p);
    LFS_CORE_API std::tuple<unsigned char*, int, int, int>
    load_image_with_alpha(std::filesystem::path p, int res_div = -1, int max_width = 3840);
    LFS_CORE_API std::tuple<unsigned char*, int, int, int>
    load_image_from_memory(const uint8_t* data, size_t size);

    LFS_CORE_API std::tuple<unsigned char*, int, int, int>
    load_image(std::filesystem::path p, int res_div = -1, int max_width = 3840);
    LFS_CORE_API void save_image(const std::filesystem::path& path, Tensor image);
    LFS_CORE_API void save_image(const std::filesystem::path& path,
                                 const std::vector<Tensor>& images,
                                 bool horizontal = true,
                                 int separator_width = 2);

    LFS_CORE_API bool save_img_data(const std::filesystem::path& p, const std::tuple<unsigned char*, int, int, int>& image_data);

    LFS_CORE_API void free_image(unsigned char* image);

} // namespace lfs::core

// Batch image saving functionality
namespace lfs::core::image_io {

    class LFS_CORE_API BatchImageSaver {
    public:
        static BatchImageSaver& instance();
        static BatchImageSaver* try_instance();
        static void wait_all_if_initialized();
        static size_t pending_count_if_initialized();

        // Delete copy/move constructors
        BatchImageSaver(const BatchImageSaver&) = delete;
        BatchImageSaver& operator=(const BatchImageSaver&) = delete;
        BatchImageSaver(BatchImageSaver&&) = delete;
        BatchImageSaver& operator=(BatchImageSaver&&) = delete;

        // Queue image for asynchronous saving
        void queue_save(const std::filesystem::path& path, lfs::core::Tensor image);

        // Queue multiple images for side-by-side saving
        void queue_save_multiple(const std::filesystem::path& path,
                                 const std::vector<lfs::core::Tensor>& images,
                                 bool horizontal = true,
                                 int separator_width = 2);

        // Wait for all pending saves to complete
        void wait_all();

        // Flush all pending saves and stop threads (called automatically on destruction)
        void shutdown();

        // Get number of pending saves
        size_t pending_count() const;

        // Enable/disable batch saving (useful for debugging)
        void set_enabled(bool enabled) { enabled_ = enabled; }
        bool is_enabled() const { return enabled_; }

    private:
        BatchImageSaver(size_t num_workers = 4);
        ~BatchImageSaver();

        struct SaveTask {
            std::filesystem::path path;
            std::vector<lfs::core::Tensor> images;
            bool horizontal = true;
            int separator_width = 2;
        };

        void worker_thread();
        void process_task(const SaveTask& task);
        void enqueue_task(SaveTask task);

        std::vector<std::thread> workers_;
        std::queue<SaveTask> task_queue_;
        mutable std::mutex queue_mutex_;
        std::condition_variable cv_;
        std::condition_variable cv_space_;
        std::condition_variable cv_finished_;
        std::atomic<bool> stop_{false};
        std::atomic<size_t> active_tasks_{0};
        std::atomic<bool> enabled_{true};
        size_t num_workers_;
        size_t max_pending_tasks_;
    };

    // Convenience functions that use the singleton
    inline void save_image_async(const std::filesystem::path& path, lfs::core::Tensor image) {
        BatchImageSaver::instance().queue_save(path, image);
    }

    inline void save_images_async(const std::filesystem::path& path,
                                  const std::vector<lfs::core::Tensor>& images,
                                  bool horizontal = true,
                                  int separator_width = 2) {
        BatchImageSaver::instance().queue_save_multiple(path, images, horizontal, separator_width);
    }

    inline void wait_for_pending_saves() {
        BatchImageSaver::wait_all_if_initialized();
    }

} // namespace lfs::core::image_io
