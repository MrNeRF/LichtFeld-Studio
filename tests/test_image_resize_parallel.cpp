/* Test to isolate cudaErrorInvalidDevice with resized images in parallel operations */

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace lfs::core;
using namespace lfs::io;

TEST(ImageResizeParallelTest, LoadImagesWithResize) {
    // Test configuration
    const int RESIZE_FACTOR = 4;
    const int NUM_IMAGES = 4;

    // Get singleton instance with CPU cache enabled
    auto& loader = CacheLoader::getInstance(true, false); // use_cpu_memory=true, use_fs_cache=false

    // Find test images
    std::vector<std::filesystem::path> image_paths;
    const std::filesystem::path data_dir =
        std::filesystem::path(TEST_DATA_DIR) / "bicycle/images_4";

    if (std::filesystem::exists(data_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
            if (entry.path().extension() == ".JPG" || entry.path().extension() == ".jpg") {
                image_paths.push_back(entry.path());
                if (image_paths.size() >= NUM_IMAGES)
                    break;
            }
        }
    }

    std::sort(image_paths.begin(), image_paths.end());

    ASSERT_GT(image_paths.size(), 0) << "No test images found";
    LOG_INFO("Found {} test images", image_paths.size());

    // Load images with resize
    std::vector<Tensor> loaded_images;
    for (const auto& path : image_paths) {
        LoadParams params;
        params.resize_factor = RESIZE_FACTOR;
        params.max_width = 0;

        auto tensor = loader.load_cached_image(path, params);
        ASSERT_GT(tensor.numel(), 0);
        ASSERT_EQ(tensor.device(), Device::CUDA);

        LOG_INFO("Loaded {} shape: [{},{},{}]",
                 path.filename().string(),
                 tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);
        loaded_images.push_back(std::move(tensor));
    }

    LOG_INFO("All images loaded successfully");

    // Check for CUDA errors after loading
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error after loading: " << cudaGetErrorString(err);

    // Simulate what training does - parallel operations on loaded images
    const int NUM_THREADS = 4;
    std::vector<std::thread> threads;
    std::atomic<int> error_count{0};

    auto worker = [&](int thread_id, int start_idx, int end_idx) {
        // Set device for this thread
        cudaSetDevice(0);

        for (int i = start_idx; i < end_idx && i < loaded_images.size(); ++i) {
            try {
                auto& img = loaded_images[i];

                // Typical operations that might happen in training
                auto normalized = img * 2.0f - 1.0f;   // Normalize
                auto mean = normalized.mean();         // Reduce operation
                auto reshaped = normalized.view({-1}); // Reshape

                if (!std::isfinite(mean.item<float>()) || reshaped.numel() != img.numel()) {
                    error_count++;
                }

                // Check for CUDA errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    LOG_ERROR("Thread {} CUDA error at image {}: {}",
                              thread_id, i, cudaGetErrorString(err));
                    error_count++;
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Thread {} exception at image {}: {}", thread_id, i, e.what());
                error_count++;
            }
        }
    };

    // Launch parallel workers
    int images_per_thread = (loaded_images.size() + NUM_THREADS - 1) / NUM_THREADS;
    for (int t = 0; t < NUM_THREADS; ++t) {
        int start = t * images_per_thread;
        int end = std::min(start + images_per_thread, (int)loaded_images.size());
        threads.emplace_back(worker, t, start, end);
    }

    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    cudaDeviceSynchronize();

    ASSERT_EQ(error_count, 0) << "Parallel operations had errors";
    LOG_INFO("Parallel operations completed without errors");

    // Check device context after operations
    int current_device;
    cudaGetDevice(&current_device);
    LOG_INFO("Current CUDA device: {}", current_device);
    ASSERT_EQ(current_device, 0);

    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error after operations: " << cudaGetErrorString(err);
}

TEST(ImageResizeParallelTest, CompareWithNoResize) {
    auto& loader = CacheLoader::getInstance(true, false); // use_cpu_memory=true, use_fs_cache=false

    const std::filesystem::path data_dir =
        std::filesystem::path(TEST_DATA_DIR) / "bicycle/images_4";
    std::vector<std::filesystem::path> image_paths;

    if (std::filesystem::exists(data_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
            if (entry.path().extension() == ".JPG" || entry.path().extension() == ".jpg") {
                image_paths.push_back(entry.path());
                if (image_paths.size() >= 3)
                    break;
            }
        }
    }

    std::sort(image_paths.begin(), image_paths.end());

    ASSERT_FALSE(image_paths.empty()) << "No test images found in " << data_dir;
    LoadParams params;
    params.resize_factor = 1; // No resize
    params.max_width = 0;

    for (const auto& path : image_paths) {
        auto tensor = loader.load_cached_image(path, params);
        ASSERT_GT(tensor.numel(), 0);

        // Same operations
        auto normalized = tensor * 2.0f - 1.0f;
        auto mean = normalized.mean();

        EXPECT_TRUE(std::isfinite(mean.item<float>()));
        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
    }
    LOG_INFO("No-resize case works correctly");
}
