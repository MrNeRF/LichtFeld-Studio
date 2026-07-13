/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include <array>
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
    load_image_with_alpha(std::filesystem::path p, int res_div = -1, int max_width = 0);
    LFS_CORE_API std::tuple<unsigned char*, int, int, int>
    load_image_from_memory(const uint8_t* data, size_t size);

    LFS_CORE_API std::tuple<unsigned char*, int, int, int>
    load_image(std::filesystem::path p, int res_div = -1, int max_width = 0);

    LFS_CORE_API std::tuple<uint16_t*, int, int, int>
    load_image_u16(std::filesystem::path p, int res_div = -1, int max_width = 0);

    LFS_CORE_API void save_image(const std::filesystem::path& path, Tensor image);
    LFS_CORE_API void save_image_u8(const std::filesystem::path& path, Tensor image, int jpeg_quality = 95);
    LFS_CORE_API void save_image(const std::filesystem::path& path,
                                 const std::vector<Tensor>& images,
                                 bool horizontal = true,
                                 int separator_width = 2);

    LFS_CORE_API bool save_img_data(const std::filesystem::path& p, const std::tuple<unsigned char*, int, int, int>& image_data);

    LFS_CORE_API void free_image(void* image);

    // Loads channel 0 of a >8-bit image (16-bit PNG, float TIFF/EXR) at native
    // resolution as float32; integer formats are normalized to [0,1].
    // Returns {nullptr, 0, 0} for 8-bit files so callers use the uint8 path.
    LFS_CORE_API std::tuple<float*, int, int>
    load_image_gray_high_bitdepth(std::filesystem::path p);

    // Loads the first three channels of a >8-bit image at native resolution as
    // interleaved HWC float32; integer formats are normalized to [0,1].
    // Returns {nullptr, 0, 0} for 8-bit or <3-channel files.
    LFS_CORE_API std::tuple<float*, int, int>
    load_image_rgb_high_bitdepth(std::filesystem::path p);
    LFS_CORE_API void free_image_float(float* image);

    // Quantization step of the file's pixel encoding in normalized [0,1] units:
    // 1/255 for 8-bit, 1/65535 for 16-bit, 0 for float formats or on failure.
    // Header-only probe, does not decode pixels.
    LFS_CORE_API float image_quantization_step(const std::filesystem::path& p);

    // Converts a decoded normal-map prior (n in [-1,1]) between the OpenGL
    // camera convention (y up, z toward the viewer) and the OpenCV one used by
    // the rasterizer (y down, z forward) by flipping y and z in place.
    LFS_CORE_API void flip_normal_prior_yz_hwc(float* data, size_t pixel_count);
    LFS_CORE_API void flip_normal_prior_yz_chw(float* data, size_t pixel_count);
    LFS_CORE_API void transform_normal_prior_world_to_camera_hwc(
        float* data, size_t pixel_count, const std::array<float, 9>& w2c);
    LFS_CORE_API void transform_normal_prior_world_to_camera_chw(
        float* data, size_t pixel_count, const std::array<float, 9>& w2c);

    // Inverse of the sRGB display transform for [0,1] encodings. Renderers
    // (e.g. Blender's default view transform) often save normal passes through
    // it; decoding those linearly skews every normal by tens of degrees.
    LFS_CORE_API float srgb_encoding_to_linear(float v);
    // In-place sRGB->linear on already-decoded normals (n in [-1,1]).
    LFS_CORE_API void srgb_normal_prior_to_linear_chw(float* data, size_t value_count);

    struct NormalPriorSample {
        float u = 0.0f; // normalized pixel-center image coordinates in [0,1]
        float v = 0.0f;
        float r = 0.0f; // raw file-encoded channel values in [0,1]
        float g = 0.0f;
        float b = 0.0f;
    };

    // Stride-sampled pixels of a normal-map file with raw [0,1] encodings and
    // normalized coordinates; empty on failure. Dataset-level convention
    // probing decodes these under multiple gamma/frame hypotheses.
    LFS_CORE_API std::vector<NormalPriorSample> sample_normal_prior_pixels(
        const std::filesystem::path& p, size_t max_samples);

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
