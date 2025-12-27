/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "io/pipelined_image_loader.hpp"
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <iomanip>
#include <numeric>
#include <vector>

using namespace lfs::io;

namespace {
    // Get test data path from environment or use default.
    // Set LICHTFELD_TEST_DATA_PATH to your dataset root (e.g., /media/user/data)
    std::filesystem::path get_test_data_root() {
        if (const char* env = std::getenv("LICHTFELD_TEST_DATA_PATH")) {
            return std::filesystem::path(env);
        }
        return std::filesystem::path("/media/paja/T7"); // Default fallback
    }

    // Test dataset paths (relative to test data root)
    const std::filesystem::path BOTANIC2_PATH = get_test_data_root() / "botanic2";
    const std::filesystem::path BERLIN_PATH = get_test_data_root() / "berlin/undistorted/images_4";

    // Helper: Get all image files from directory
    std::vector<std::filesystem::path> get_image_files(const std::filesystem::path& dir) {
        std::vector<std::filesystem::path> files;
        if (!std::filesystem::exists(dir)) {
            return files;
        }

        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file())
                continue;
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
        return files;
    }

    // Helper: Format throughput
    std::string format_throughput(double images_per_sec) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << images_per_sec << " img/s";
        return oss.str();
    }

    // Helper: Format time
    std::string format_time(double ms) {
        std::ostringstream oss;
        if (ms < 1.0) {
            oss << std::fixed << std::setprecision(2) << (ms * 1000) << " µs";
        } else if (ms < 1000.0) {
            oss << std::fixed << std::setprecision(2) << ms << " ms";
        } else {
            oss << std::fixed << std::setprecision(2) << (ms / 1000) << " s";
        }
        return oss.str();
    }

    struct BenchmarkResult {
        std::string name;
        size_t num_images;
        double total_time_ms;
        double images_per_sec;
        double avg_latency_ms;
        size_t hot_hits;
        size_t cold_misses;
        size_t gpu_batch_decodes;
    };

    void print_result(const BenchmarkResult& r) {
        std::cout << "  " << r.name << ":\n"
                  << "    Images:     " << r.num_images << "\n"
                  << "    Time:       " << format_time(r.total_time_ms) << "\n"
                  << "    Throughput: " << format_throughput(r.images_per_sec) << "\n"
                  << "    Avg latency:" << format_time(r.avg_latency_ms) << "\n"
                  << "    Hot hits:   " << r.hot_hits << "\n"
                  << "    Cold misses:" << r.cold_misses << "\n"
                  << "    GPU batches:" << r.gpu_batch_decodes << "\n";
    }

    // Run benchmark with given config
    BenchmarkResult run_benchmark(
        const std::string& name,
        const std::vector<std::filesystem::path>& image_files,
        const LoadParams& params,
        const PipelinedLoaderConfig& config,
        size_t max_images = 0) {

        size_t num_images = max_images > 0 ? std::min(max_images, image_files.size())
                                           : image_files.size();

        if (num_images == 0) {
            return {name, 0, 0, 0, 0, 0, 0, 0};
        }

        PipelinedImageLoader loader(config);

        // Prefetch all images
        for (size_t i = 0; i < num_images; ++i) {
            loader.prefetch(i, image_files[i], params);
        }

        // Time the retrieval
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<double> latencies;
        latencies.reserve(num_images);

        for (size_t i = 0; i < num_images; ++i) {
            auto img_start = std::chrono::high_resolution_clock::now();
            auto ready = loader.get();
            auto img_end = std::chrono::high_resolution_clock::now();

            latencies.push_back(std::chrono::duration<double, std::milli>(img_end - img_start).count());

            if (!ready.tensor.is_valid() || ready.tensor.numel() == 0) {
                LOG_WARN("Image {} failed to load", i);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        auto stats = loader.get_stats();
        double avg_latency = latencies.empty() ? 0 : std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

        return {
            name,
            num_images,
            total_time_ms,
            (num_images / total_time_ms) * 1000.0,
            avg_latency,
            stats.hot_path_hits,
            stats.cold_path_misses,
            stats.gpu_batch_decodes};
    }

} // anonymous namespace

// =============================================================================
// Basic Throughput Tests
// =============================================================================

TEST(PipelinedLoaderBenchmark, BerlinDataset_ColdCache) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available at " << BERLIN_PATH;
    }

    std::cout << "\n=== Berlin Dataset (PNG, 1000x570) - Cold Cache ===\n";
    std::cout << "Found " << files.size() << " images\n";

    LoadParams params{.resize_factor = 1, .max_width = 0};
    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 8;
    config.prefetch_count = 32;
    config.io_threads = 2;
    config.cold_process_threads = 2;

    auto result = run_benchmark("Cold cache (default config)", files, params, config, 200);
    print_result(result);

    EXPECT_GT(result.images_per_sec, 10.0) << "Throughput too low";
}

TEST(PipelinedLoaderBenchmark, BerlinDataset_WarmCache) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available at " << BERLIN_PATH;
    }

    std::cout << "\n=== Berlin Dataset (PNG, 1000x570) - Warm Cache ===\n";
    std::cout << "Found " << files.size() << " images\n";

    LoadParams params{.resize_factor = 1, .max_width = 0};
    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 8;
    config.prefetch_count = 32;
    config.io_threads = 2;
    config.cold_process_threads = 2;

    // First pass - warm the cache
    auto cold_result = run_benchmark("Pass 1 (cold)", files, params, config, 100);
    print_result(cold_result);

    // Second pass - should hit cache
    // Note: The loader is destroyed after first pass, so cache is gone
    // We need to keep the loader alive for warm cache test
    {
        PipelinedImageLoader loader(config);

        // Pass 1: warm cache
        for (size_t i = 0; i < 100 && i < files.size(); ++i) {
            loader.prefetch(i, files[i], params);
        }
        for (size_t i = 0; i < 100 && i < files.size(); ++i) {
            auto ready = loader.get();
        }

        auto stats1 = loader.get_stats();
        std::cout << "  After pass 1: " << stats1.cold_path_misses << " cold, "
                  << stats1.hot_path_hits << " hot\n";

        // Pass 2: should be all cache hits
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < 100 && i < files.size(); ++i) {
            loader.prefetch(i + 1000, files[i], params); // Different sequence IDs
        }
        for (size_t i = 0; i < 100 && i < files.size(); ++i) {
            auto ready = loader.get();
        }
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        auto stats2 = loader.get_stats();

        std::cout << "  Pass 2 (warm cache):\n"
                  << "    Time:       " << format_time(time_ms) << "\n"
                  << "    Throughput: " << format_throughput((100.0 / time_ms) * 1000) << "\n"
                  << "    New hot:    " << (stats2.hot_path_hits - stats1.hot_path_hits) << "\n"
                  << "    New cold:   " << (stats2.cold_path_misses - stats1.cold_path_misses) << "\n";

        // Warm cache should be significantly faster
        EXPECT_GT(stats2.hot_path_hits - stats1.hot_path_hits, 90)
            << "Expected most images to hit cache";
    }
}

TEST(PipelinedLoaderBenchmark, Botanic2Dataset_ColdCache) {
    auto files = get_image_files(BOTANIC2_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Botanic2 dataset not available at " << BOTANIC2_PATH;
    }

    std::cout << "\n=== Botanic2 Dataset (JPEG, 7943x5262) - Cold Cache ===\n";
    std::cout << "Found " << files.size() << " images\n";

    // Use resize_factor=4 to get reasonable size (~1986x1315)
    LoadParams params{.resize_factor = 4, .max_width = 0};
    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 8;
    config.prefetch_count = 16; // Fewer prefetch due to large images
    config.io_threads = 2;
    config.cold_process_threads = 2;

    auto result = run_benchmark("Cold cache (resize_factor=4)", files, params, config, 50);
    print_result(result);

    EXPECT_GT(result.images_per_sec, 10.0) << "Throughput too low";
}

TEST(PipelinedLoaderBenchmark, Botanic2Dataset_WarmCache_FullRes) {
    auto files = get_image_files(BOTANIC2_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Botanic2 dataset not available at " << BOTANIC2_PATH;
    }

    std::cout << "\n=== Botanic2 Full Res JPEG (no resize) - Warm Cache ===\n";

    // Original JPEGs should hit hot path directly (no resize needed)
    LoadParams params{.resize_factor = 1, .max_width = 0};
    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 4; // Smaller batches for huge images
    config.prefetch_count = 8;
    config.io_threads = 2;
    config.cold_process_threads = 1;

    {
        PipelinedImageLoader loader(config);

        // Pass 1: warm cache (original JPEG at full res)
        size_t test_count = std::min<size_t>(20, files.size());
        for (size_t i = 0; i < test_count; ++i) {
            loader.prefetch(i, files[i], params);
        }
        for (size_t i = 0; i < test_count; ++i) {
            auto ready = loader.get();
        }

        auto stats1 = loader.get_stats();
        std::cout << "  Pass 1: hot=" << stats1.hot_path_hits
                  << " cold=" << stats1.cold_path_misses << "\n";

        // For original JPEGs at full res, they should cache directly
        // So pass 2 should be all hot hits
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < test_count; ++i) {
            loader.prefetch(i + 1000, files[i], params);
        }
        for (size_t i = 0; i < test_count; ++i) {
            auto ready = loader.get();
        }
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        auto stats2 = loader.get_stats();

        std::cout << "  Pass 2 (warm): " << format_throughput((test_count / time_ms) * 1000)
                  << " (hot: " << (stats2.hot_path_hits - stats1.hot_path_hits)
                  << ", cold: " << (stats2.cold_path_misses - stats1.cold_path_misses) << ")\n";
    }
}

// =============================================================================
// Configuration Tuning Tests
// =============================================================================

TEST(PipelinedLoaderBenchmark, BatchSizeScaling) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available";
    }

    std::cout << "\n=== Batch Size Scaling (Berlin, 100 images) ===\n";

    LoadParams params{.resize_factor = 1, .max_width = 0};
    std::vector<size_t> batch_sizes = {1, 2, 4, 8, 16, 32};

    for (size_t batch_size : batch_sizes) {
        PipelinedLoaderConfig config;
        config.jpeg_batch_size = batch_size;
        config.prefetch_count = 64;
        config.io_threads = 2;
        config.cold_process_threads = 2;

        // We need fresh loader each time for fair comparison
        auto result = run_benchmark(
            "batch_size=" + std::to_string(batch_size),
            files, params, config, 100);

        std::cout << "  batch_size=" << std::setw(2) << batch_size
                  << ": " << format_throughput(result.images_per_sec)
                  << " (batches: " << result.gpu_batch_decodes << ")\n";
    }
}

TEST(PipelinedLoaderBenchmark, IOThreadScaling) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available";
    }

    std::cout << "\n=== I/O Thread Scaling (Berlin, 200 images) ===\n";

    LoadParams params{.resize_factor = 1, .max_width = 0};
    std::vector<size_t> io_threads = {1, 2, 4, 8};

    for (size_t threads : io_threads) {
        PipelinedLoaderConfig config;
        config.jpeg_batch_size = 8;
        config.prefetch_count = 64;
        config.io_threads = threads;
        config.cold_process_threads = 2;

        auto result = run_benchmark(
            "io_threads=" + std::to_string(threads),
            files, params, config, 200);

        std::cout << "  io_threads=" << threads
                  << ": " << format_throughput(result.images_per_sec) << "\n";
    }
}

TEST(PipelinedLoaderBenchmark, ColdThreadScaling) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available";
    }

    std::cout << "\n=== Cold Process Thread Scaling (Berlin PNG, 100 images) ===\n";
    std::cout << "Note: PNGs require cold path processing\n";

    LoadParams params{.resize_factor = 1, .max_width = 0};
    std::vector<size_t> cold_threads = {1, 2, 4, 8};

    for (size_t threads : cold_threads) {
        PipelinedLoaderConfig config;
        config.jpeg_batch_size = 8;
        config.prefetch_count = 64;
        config.io_threads = 2;
        config.cold_process_threads = threads;

        auto result = run_benchmark(
            "cold_threads=" + std::to_string(threads),
            files, params, config, 100);

        std::cout << "  cold_threads=" << threads
                  << ": " << format_throughput(result.images_per_sec)
                  << " (cold: " << result.cold_misses << ")\n";
    }
}

TEST(PipelinedLoaderBenchmark, PrefetchCountScaling) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available";
    }

    std::cout << "\n=== Prefetch Count Scaling (Berlin, 200 images) ===\n";

    LoadParams params{.resize_factor = 1, .max_width = 0};
    std::vector<size_t> prefetch_counts = {8, 16, 32, 64, 128};

    for (size_t prefetch : prefetch_counts) {
        PipelinedLoaderConfig config;
        config.jpeg_batch_size = 8;
        config.prefetch_count = prefetch;
        config.io_threads = 2;
        config.cold_process_threads = 2;

        auto result = run_benchmark(
            "prefetch=" + std::to_string(prefetch),
            files, params, config, 200);

        std::cout << "  prefetch=" << std::setw(3) << prefetch
                  << ": " << format_throughput(result.images_per_sec) << "\n";
    }
}

// =============================================================================
// Simulated Training Loop
// =============================================================================

TEST(PipelinedLoaderBenchmark, SimulatedTrainingLoop) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available";
    }

    std::cout << "\n=== Simulated Training Loop (Berlin, 3 epochs) ===\n";

    LoadParams params{.resize_factor = 1, .max_width = 0};
    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 8;
    config.prefetch_count = 32;
    config.io_threads = 2;
    config.cold_process_threads = 2;

    const size_t images_per_epoch = std::min<size_t>(100, files.size());
    const size_t num_epochs = 3;
    const double training_work_ms = 5.0; // Simulate 5ms of GPU training per image

    PipelinedImageLoader loader(config);

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        double pure_load_time = 0;

        // Prefetch first batch
        size_t prefetch_idx = 0;
        for (size_t i = 0; i < config.prefetch_count && prefetch_idx < images_per_epoch; ++i) {
            loader.prefetch(epoch * 10000 + prefetch_idx, files[prefetch_idx % files.size()], params);
            ++prefetch_idx;
        }

        for (size_t i = 0; i < images_per_epoch; ++i) {
            // Get image
            auto load_start = std::chrono::high_resolution_clock::now();
            auto ready = loader.get();
            auto load_end = std::chrono::high_resolution_clock::now();
            pure_load_time += std::chrono::duration<double, std::milli>(load_end - load_start).count();

            // Prefetch more
            if (prefetch_idx < images_per_epoch) {
                loader.prefetch(epoch * 10000 + prefetch_idx, files[prefetch_idx % files.size()], params);
                ++prefetch_idx;
            }

            // Simulate training work
            std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(training_work_ms * 1000)));
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_time = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();

        auto stats = loader.get_stats();
        std::cout << "  Epoch " << (epoch + 1) << ": "
                  << format_time(epoch_time) << " total, "
                  << format_time(pure_load_time) << " loading, "
                  << format_throughput((images_per_epoch / epoch_time) * 1000) << " effective, "
                  << "hot=" << stats.hot_path_hits << " cold=" << stats.cold_path_misses << "\n";

        // After first epoch, most should be cache hits
        if (epoch > 0) {
            // Note: Stats are cumulative, so we can't easily check per-epoch
            // But throughput should be better
        }
    }
}

// =============================================================================
// Minimal Worker Test
// =============================================================================

TEST(PipelinedLoaderBenchmark, MinimalWorkers) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available";
    }

    std::cout << "\n=== Minimal Worker Configuration Test ===\n";
    std::cout << "Goal: Find minimum workers for acceptable throughput\n";

    LoadParams params{.resize_factor = 1, .max_width = 0};

    struct WorkerConfig {
        size_t io_threads;
        size_t cold_threads;
    };

    std::vector<WorkerConfig> configs = {
        {1, 1}, // Absolute minimum
        {1, 2},
        {2, 1},
        {2, 2}, // Balanced
        {4, 2}, // More I/O
        {2, 4}, // More cold processing
    };

    for (const auto& wc : configs) {
        PipelinedLoaderConfig config;
        config.jpeg_batch_size = 8;
        config.prefetch_count = 32;
        config.io_threads = wc.io_threads;
        config.cold_process_threads = wc.cold_threads;

        // Test with warm cache (second pass)
        {
            PipelinedImageLoader loader(config);

            // Warm cache
            for (size_t i = 0; i < 50; ++i) {
                loader.prefetch(i, files[i], params);
            }
            for (size_t i = 0; i < 50; ++i) {
                loader.get();
            }

            // Measure warm cache performance
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < 50; ++i) {
                loader.prefetch(i + 1000, files[i], params);
            }
            for (size_t i = 0; i < 50; ++i) {
                loader.get();
            }
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            size_t total_threads = wc.io_threads + wc.cold_threads + 1; // +1 for GPU decode thread
            std::cout << "  io=" << wc.io_threads << " cold=" << wc.cold_threads
                      << " (total " << total_threads << " threads): "
                      << format_throughput((50.0 / time_ms) * 1000) << " warm\n";
        }
    }
}

// =============================================================================
// Memory Efficiency Test
// =============================================================================

TEST(PipelinedLoaderBenchmark, MemoryEfficiency) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available";
    }

    std::cout << "\n=== Memory Efficiency Test ===\n";

    LoadParams params{.resize_factor = 1, .max_width = 0};
    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 8;
    config.prefetch_count = 32;
    config.io_threads = 2;
    config.cold_process_threads = 2;
    config.max_cache_bytes = 500 * 1024 * 1024; // 500MB cache limit

    PipelinedImageLoader loader(config);

    const size_t test_images = std::min<size_t>(200, files.size());

    // Load images and track cache growth
    for (size_t i = 0; i < test_images; ++i) {
        loader.prefetch(i, files[i], params);
    }
    for (size_t i = 0; i < test_images; ++i) {
        loader.get();
    }

    auto stats = loader.get_stats();
    double cache_mb = stats.jpeg_cache_bytes / (1024.0 * 1024.0);
    double bytes_per_entry = stats.jpeg_cache_entries > 0
                                 ? static_cast<double>(stats.jpeg_cache_bytes) / stats.jpeg_cache_entries
                                 : 0;

    std::cout << "  Images loaded: " << test_images << "\n"
              << "  Cache entries: " << stats.jpeg_cache_entries << "\n"
              << "  Cache size:    " << std::fixed << std::setprecision(1) << cache_mb << " MB\n"
              << "  Bytes/entry:   " << std::fixed << std::setprecision(0) << bytes_per_entry / 1024 << " KB\n"
              << "  Hot hits:      " << stats.hot_path_hits << "\n"
              << "  Cold misses:   " << stats.cold_path_misses << "\n";

    // Berlin PNGs at 1000x570, cached as JPEG quality 95, are ~700KB
    // This is larger than pure JPEG due to PNG->JPEG conversion at high quality
    EXPECT_LT(bytes_per_entry, 1024 * 1024) << "Cache entries too large (>1MB)";
}

// =============================================================================
// Resize Factor Tests
// =============================================================================

TEST(PipelinedLoaderBenchmark, ResizeFactor_Botanic2) {
    auto files = get_image_files(BOTANIC2_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Botanic2 dataset not available";
    }

    std::cout << "\n=== Resize Factor Test (Botanic2 JPEG 7943x5262) ===\n";

    // Original dimensions
    const int orig_width = 7943;
    const int orig_height = 5262;

    PipelinedLoaderConfig config;
    config.io_threads = 2;
    config.cold_process_threads = 2;
    config.prefetch_count = 8;

    std::vector<int> resize_factors = {1, 2, 4, 8};

    for (int rf : resize_factors) {
        LoadParams params{.resize_factor = rf, .max_width = 0};
        PipelinedImageLoader loader(config);

        // Load just one image to verify dimensions
        loader.prefetch(0, files[0], params);
        auto ready = loader.get();

        ASSERT_TRUE(ready.tensor.is_valid()) << "Failed to load with resize_factor=" << rf;

        auto shape = ready.tensor.shape();
        int h = shape[1];
        int w = shape[2];

        int expected_w = orig_width / rf;
        int expected_h = orig_height / rf;

        std::cout << "  resize_factor=" << rf << ": " << w << "x" << h
                  << " (expected ~" << expected_w << "x" << expected_h << ")\n";

        // Allow small rounding differences (±1 pixel)
        EXPECT_NEAR(w, expected_w, 2) << "Width mismatch for resize_factor=" << rf;
        EXPECT_NEAR(h, expected_h, 2) << "Height mismatch for resize_factor=" << rf;
    }
}

// =============================================================================
// Max Width Tests
// =============================================================================

TEST(PipelinedLoaderBenchmark, MaxWidth_Botanic2) {
    auto files = get_image_files(BOTANIC2_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Botanic2 dataset not available";
    }

    std::cout << "\n=== Max Width Test (Botanic2 JPEG 7943x5262) ===\n";

    PipelinedLoaderConfig config;
    config.io_threads = 2;
    config.cold_process_threads = 2;
    config.prefetch_count = 8;

    std::vector<int> max_widths = {4000, 2000, 1000, 500};

    for (int mw : max_widths) {
        LoadParams params{.resize_factor = 1, .max_width = mw};
        PipelinedImageLoader loader(config);

        loader.prefetch(0, files[0], params);
        auto ready = loader.get();

        ASSERT_TRUE(ready.tensor.is_valid()) << "Failed to load with max_width=" << mw;

        auto shape = ready.tensor.shape();
        int h = shape[1];
        int w = shape[2];

        std::cout << "  max_width=" << mw << ": " << w << "x" << h << "\n";

        // Both dimensions should be <= max_width
        EXPECT_LE(w, mw) << "Width exceeds max_width=" << mw;
        EXPECT_LE(h, mw) << "Height exceeds max_width=" << mw;

        // At least one dimension should be close to max_width (aspect preserved)
        int larger = std::max(w, h);
        EXPECT_GE(larger, mw - 10) << "Image too small for max_width=" << mw;
    }
}

TEST(PipelinedLoaderBenchmark, MaxWidth_CombinedWithResize) {
    auto files = get_image_files(BOTANIC2_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Botanic2 dataset not available";
    }

    std::cout << "\n=== Max Width + Resize Factor Combined ===\n";

    PipelinedLoaderConfig config;
    config.io_threads = 2;
    config.cold_process_threads = 2;

    // resize_factor=2 gives 3971x2631, then max_width=1000 should give ~1000x662
    LoadParams params{.resize_factor = 2, .max_width = 1000};
    PipelinedImageLoader loader(config);

    loader.prefetch(0, files[0], params);
    auto ready = loader.get();

    ASSERT_TRUE(ready.tensor.is_valid());

    auto shape = ready.tensor.shape();
    int h = shape[1];
    int w = shape[2];

    std::cout << "  resize_factor=2, max_width=1000: " << w << "x" << h << "\n";

    EXPECT_LE(w, 1000);
    EXPECT_LE(h, 1000);
    EXPECT_GE(std::max(w, h), 990);
}

// =============================================================================
// Images_{2,4,8} Folder Tests
// =============================================================================

TEST(PipelinedLoaderBenchmark, ImagesFolders_Berlin) {
    const std::filesystem::path base_path = get_test_data_root() / "berlin/undistorted";

    std::cout << "\n=== Images Folder Resolution Test (Berlin) ===\n";

    struct FolderTest {
        std::string folder;
        int expected_scale;
    };

    std::vector<FolderTest> folders = {
        {"images", 1},   // 4000x2280
        {"images_2", 2}, // 2000x1140
        {"images_4", 4}, // 1000x570
        {"images_8", 8}, // 500x285
    };

    // Full resolution reference
    const int base_width = 4000;
    const int base_height = 2280;

    PipelinedLoaderConfig config;
    config.io_threads = 2;
    config.cold_process_threads = 2;

    for (const auto& ft : folders) {
        std::filesystem::path folder_path = base_path / ft.folder;
        auto files = get_image_files(folder_path);

        if (files.empty()) {
            std::cout << "  " << ft.folder << ": SKIPPED (not found)\n";
            continue;
        }

        LoadParams params{.resize_factor = 1, .max_width = 0};
        PipelinedImageLoader loader(config);

        loader.prefetch(0, files[0], params);
        auto ready = loader.get();

        ASSERT_TRUE(ready.tensor.is_valid()) << "Failed to load from " << ft.folder;

        auto shape = ready.tensor.shape();
        int h = shape[1];
        int w = shape[2];

        int expected_w = base_width / ft.expected_scale;
        int expected_h = base_height / ft.expected_scale;

        std::cout << "  " << ft.folder << ": " << w << "x" << h
                  << " (expected " << expected_w << "x" << expected_h << ")\n";

        EXPECT_NEAR(w, expected_w, 2) << "Width mismatch for " << ft.folder;
        EXPECT_NEAR(h, expected_h, 2) << "Height mismatch for " << ft.folder;
    }
}

// =============================================================================
// GPU Memory Efficiency Tests
// =============================================================================

TEST(PipelinedLoaderBenchmark, NoUnnecessaryCPURoundTrips) {
    auto files = get_image_files(BOTANIC2_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Botanic2 dataset not available";
    }

    std::cout << "\n=== GPU Memory Efficiency Test ===\n";
    std::cout << "Verifying images stay on GPU after loading\n";

    PipelinedLoaderConfig config;
    config.io_threads = 2;
    config.cold_process_threads = 2;

    // Test with resize (cold path)
    {
        LoadParams params{.resize_factor = 4, .max_width = 0};
        PipelinedImageLoader loader(config);

        loader.prefetch(0, files[0], params);
        auto ready = loader.get();

        ASSERT_TRUE(ready.tensor.is_valid());
        EXPECT_EQ(ready.tensor.device(), lfs::core::Device::CUDA)
            << "Tensor should be on GPU, not CPU";

        std::cout << "  Cold path (resize_factor=4): tensor on "
                  << (ready.tensor.device() == lfs::core::Device::CUDA ? "GPU" : "CPU")
                  << " ✓\n";
    }

    // Test full res (hot path)
    {
        LoadParams params{.resize_factor = 1, .max_width = 0};
        PipelinedImageLoader loader(config);

        loader.prefetch(0, files[0], params);
        auto ready = loader.get();

        ASSERT_TRUE(ready.tensor.is_valid());
        EXPECT_EQ(ready.tensor.device(), lfs::core::Device::CUDA)
            << "Tensor should be on GPU, not CPU";

        std::cout << "  Hot path (full res): tensor on "
                  << (ready.tensor.device() == lfs::core::Device::CUDA ? "GPU" : "CPU")
                  << " ✓\n";
    }
}

TEST(PipelinedLoaderBenchmark, TensorFormatCorrect) {
    auto files = get_image_files(BOTANIC2_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Botanic2 dataset not available";
    }

    std::cout << "\n=== Tensor Format Verification ===\n";

    PipelinedLoaderConfig config;
    LoadParams params{.resize_factor = 4, .max_width = 0};
    PipelinedImageLoader loader(config);

    loader.prefetch(0, files[0], params);
    auto ready = loader.get();

    ASSERT_TRUE(ready.tensor.is_valid());

    auto shape = ready.tensor.shape();
    EXPECT_EQ(shape.rank(), 3) << "Expected 3D tensor [C, H, W]";
    EXPECT_EQ(shape[0], 3) << "Expected 3 channels (RGB)";

    // Verify dtype is float32
    EXPECT_EQ(ready.tensor.dtype(), lfs::core::DataType::Float32)
        << "Expected float32 tensor";

    // Verify values are normalized [0, 1]
    auto cpu_tensor = ready.tensor.cpu();
    auto accessor = cpu_tensor.accessor<float, 3>();
    float min_val = 1.0f, max_val = 0.0f;

    // Sample a few values
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++) {
                float val = accessor(c, y, x);
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }
    }

    std::cout << "  Shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]\n";
    std::cout << "  DType: float32\n";
    std::cout << "  Value range: [" << min_val << ", " << max_val << "]\n";

    EXPECT_GE(min_val, 0.0f) << "Values should be >= 0";
    EXPECT_LE(max_val, 1.0f) << "Values should be <= 1";
}

// =============================================================================
// Large Image Handling (No OOM on single large images)
// =============================================================================

TEST(PipelinedLoaderBenchmark, LargeImageHandling) {
    auto files = get_image_files(BOTANIC2_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Botanic2 dataset not available";
    }

    std::cout << "\n=== Large Image Handling Test ===\n";
    std::cout << "Testing 7943x5262 JPEG (126 MB uncompressed) doesn't cause OOM\n";

    PipelinedLoaderConfig config;
    config.io_threads = 1;
    config.cold_process_threads = 1;
    config.prefetch_count = 4;

    // Load several large images at full resolution
    LoadParams params{.resize_factor = 1, .max_width = 0};
    PipelinedImageLoader loader(config);

    const size_t test_count = 10;
    for (size_t i = 0; i < test_count && i < files.size(); ++i) {
        loader.prefetch(i, files[i], params);
    }

    size_t loaded = 0;
    for (size_t i = 0; i < test_count && i < files.size(); ++i) {
        auto ready = loader.get();
        if (ready.tensor.is_valid()) {
            ++loaded;
        }
    }

    std::cout << "  Loaded " << loaded << "/" << test_count << " large images successfully\n";
    EXPECT_EQ(loaded, std::min(test_count, files.size())) << "Some images failed to load";
}

// =============================================================================
// Disk Cache Fallback Test
// =============================================================================

TEST(PipelinedLoaderBenchmark, DiskCacheFallback) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available";
    }

    std::cout << "\n=== Disk Cache Fallback Test ===\n";

    // Configure with very small memory cache to force disk usage
    PipelinedLoaderConfig config;
    config.max_cache_bytes = 10 * 1024 * 1024; // Only 10MB RAM cache
    config.use_filesystem_cache = true;
    config.io_threads = 2;
    config.cold_process_threads = 2;

    LoadParams params{.resize_factor = 1, .max_width = 0};

    // Load enough images to overflow the small cache
    const size_t overflow_count = 50; // Should exceed 10MB

    {
        PipelinedImageLoader loader(config);

        for (size_t i = 0; i < overflow_count && i < files.size(); ++i) {
            loader.prefetch(i, files[i], params);
        }

        for (size_t i = 0; i < overflow_count && i < files.size(); ++i) {
            auto ready = loader.get();
            EXPECT_TRUE(ready.tensor.is_valid()) << "Image " << i << " failed";
        }

        auto stats = loader.get_stats();
        std::cout << "  Loaded: " << overflow_count << " images\n";
        std::cout << "  Cache bytes: " << stats.jpeg_cache_bytes / (1024 * 1024) << " MB\n";
        std::cout << "  Hot hits: " << stats.hot_path_hits << "\n";
        std::cout << "  Cold misses: " << stats.cold_path_misses << "\n";
    }

    std::cout << "  Test completed without OOM ✓\n";
}

// =============================================================================
// Throughput with Various Configurations
// =============================================================================

TEST(PipelinedLoaderBenchmark, ThroughputComparison) {
    std::cout << "\n=== Throughput Comparison (various configs) ===\n";

    struct TestConfig {
        std::string name;
        std::filesystem::path path;
        int resize_factor;
        int max_width;
        size_t images;
    };

    std::vector<TestConfig> configs = {
        {"Berlin PNG (images_4, 1000x570)", BERLIN_PATH, 1, 0, 100},
        {"Botanic2 JPEG (full res, 7943x5262)", BOTANIC2_PATH, 1, 0, 20},
        {"Botanic2 JPEG (resize=4, ~1986x1315)", BOTANIC2_PATH, 4, 0, 50},
        {"Botanic2 JPEG (max_width=1000)", BOTANIC2_PATH, 1, 1000, 50},
    };

    for (const auto& tc : configs) {
        auto files = get_image_files(tc.path);
        if (files.empty()) {
            std::cout << "  " << tc.name << ": SKIPPED\n";
            continue;
        }

        size_t num = std::min(tc.images, files.size());

        PipelinedLoaderConfig config;
        config.io_threads = 2;
        config.cold_process_threads = 2;
        config.prefetch_count = 32;

        LoadParams params{.resize_factor = tc.resize_factor, .max_width = tc.max_width};
        PipelinedImageLoader loader(config);

        // Prefetch all
        for (size_t i = 0; i < num; ++i) {
            loader.prefetch(i, files[i], params);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num; ++i) {
            auto ready = loader.get();
        }
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double throughput = (num / time_ms) * 1000.0;

        std::cout << "  " << tc.name << ": " << std::fixed << std::setprecision(1)
                  << throughput << " img/s\n";

        EXPECT_GT(throughput, 5.0) << "Throughput too low for " << tc.name;
    }
}

// =============================================================================
// Detailed Performance Analysis
// =============================================================================

TEST(PipelinedLoaderBenchmark, DetailedPerfAnalysis_Botanic2) {
    auto files = get_image_files(BOTANIC2_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Botanic2 dataset not available";
    }

    std::cout << "\n=== Detailed Performance Analysis (Botanic2 with resize) ===\n";
    std::cout << "Testing cache effectiveness across multiple epochs\n\n";

    LoadParams params{.resize_factor = 4, .max_width = 0}; // ~1986x1315 output

    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 8;
    config.prefetch_count = 16;
    config.io_threads = 2;
    config.cold_process_threads = 2;

    const size_t images_per_epoch = std::min<size_t>(100, files.size());
    const size_t num_epochs = 3;

    PipelinedImageLoader loader(config);

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        // Prefetch all images for this epoch
        for (size_t i = 0; i < images_per_epoch; ++i) {
            loader.prefetch(epoch * 10000 + i, files[i], params);
        }

        // Retrieve all
        for (size_t i = 0; i < images_per_epoch; ++i) {
            auto ready = loader.get();
            ASSERT_TRUE(ready.tensor.is_valid()) << "Epoch " << epoch << " image " << i << " failed";
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_ms = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();

        auto stats = loader.get_stats();
        double throughput = (images_per_epoch / epoch_ms) * 1000.0;

        std::cout << "Epoch " << (epoch + 1) << ": " << std::fixed << std::setprecision(1)
                  << throughput << " img/s | "
                  << "hot=" << stats.hot_path_hits << " cold=" << stats.cold_path_misses << " | "
                  << "decode=" << std::setprecision(0) << stats.gpu_decode_time_ms << "ms "
                  << "(" << (stats.total_decode_calls > 0 ? stats.total_decode_calls / (stats.gpu_decode_time_ms / 1000.0) : 0) << " img/s)\n";

        // After first epoch, should be significantly faster
        if (epoch > 0) {
            EXPECT_GT(throughput, 50.0) << "Warm cache should be faster";
        }
    }
}

TEST(PipelinedLoaderBenchmark, DetailedPerfAnalysis_Berlin) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available";
    }

    std::cout << "\n=== Detailed Performance Analysis (Berlin PNG) ===\n";
    std::cout << "Small images (1000x570) - should be fast\n\n";

    LoadParams params{.resize_factor = 1, .max_width = 0};

    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 16; // Larger batches for small images
    config.prefetch_count = 32;
    config.io_threads = 2;
    config.cold_process_threads = 2;

    const size_t images_per_epoch = std::min<size_t>(200, files.size());
    const size_t num_epochs = 3;

    PipelinedImageLoader loader(config);

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < images_per_epoch; ++i) {
            loader.prefetch(epoch * 10000 + i, files[i], params);
        }

        for (size_t i = 0; i < images_per_epoch; ++i) {
            auto ready = loader.get();
            ASSERT_TRUE(ready.tensor.is_valid());
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_ms = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();

        auto stats = loader.get_stats();
        double throughput = (images_per_epoch / epoch_ms) * 1000.0;

        std::cout << "Epoch " << (epoch + 1) << ": " << std::fixed << std::setprecision(1)
                  << throughput << " img/s | "
                  << "hot=" << stats.hot_path_hits << " cold=" << stats.cold_path_misses << "\n";
    }
}

TEST(PipelinedLoaderBenchmark, OptimalConfigSearch) {
    auto files = get_image_files(BERLIN_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Berlin dataset not available";
    }

    std::cout << "\n=== Optimal Configuration Search ===\n";
    std::cout << "Testing different batch sizes and thread counts\n\n";

    LoadParams params{.resize_factor = 1, .max_width = 0};
    const size_t test_images = 100;

    struct ConfigTest {
        size_t batch_size;
        size_t io_threads;
        size_t cold_threads;
        double warm_throughput = 0;
    };

    std::vector<ConfigTest> configs = {
        {4, 1, 1},
        {8, 1, 1},
        {8, 2, 1},
        {8, 2, 2},
        {16, 1, 1},
        {16, 2, 1},
        {16, 2, 2},
        {32, 2, 2},
    };

    for (auto& ct : configs) {
        PipelinedLoaderConfig config;
        config.jpeg_batch_size = ct.batch_size;
        config.prefetch_count = std::max<size_t>(32, ct.batch_size * 4);
        config.io_threads = ct.io_threads;
        config.cold_process_threads = ct.cold_threads;

        PipelinedImageLoader loader(config);

        // Cold pass (warm cache)
        for (size_t i = 0; i < test_images; ++i) {
            loader.prefetch(i, files[i], params);
        }
        for (size_t i = 0; i < test_images; ++i) {
            loader.get();
        }

        // Warm pass (measure)
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < test_images; ++i) {
            loader.prefetch(i + 10000, files[i], params);
        }
        for (size_t i = 0; i < test_images; ++i) {
            loader.get();
        }
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        ct.warm_throughput = (test_images / time_ms) * 1000.0;

        size_t total_threads = ct.io_threads + ct.cold_threads + 1;
        std::cout << "  batch=" << std::setw(2) << ct.batch_size
                  << " io=" << ct.io_threads << " cold=" << ct.cold_threads
                  << " (total " << total_threads << " threads): "
                  << std::fixed << std::setprecision(1) << ct.warm_throughput << " img/s\n";
    }

    // Find best config
    auto best = std::max_element(configs.begin(), configs.end(),
                                 [](const ConfigTest& a, const ConfigTest& b) { return a.warm_throughput < b.warm_throughput; });

    std::cout << "\nBest: batch=" << best->batch_size << " io=" << best->io_threads
              << " cold=" << best->cold_threads << " -> " << best->warm_throughput << " img/s\n";
}

TEST(PipelinedLoaderBenchmark, DecodeTimeBreakdown) {
    auto files = get_image_files(BOTANIC2_PATH);
    if (files.empty()) {
        GTEST_SKIP() << "Botanic2 dataset not available";
    }

    std::cout << "\n=== Decode Time Breakdown (Botanic2) ===\n";
    std::cout << "Measuring where time is spent in the pipeline\n\n";

    LoadParams params{.resize_factor = 4, .max_width = 0};

    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 8;
    config.prefetch_count = 16;
    config.io_threads = 2;
    config.cold_process_threads = 2;

    const size_t test_images = 50;

    // Run twice - cold and warm
    for (int pass = 0; pass < 2; ++pass) {
        PipelinedImageLoader loader(config);

        if (pass == 1) {
            // Warm the cache first
            for (size_t i = 0; i < test_images; ++i) {
                loader.prefetch(i, files[i], params);
            }
            for (size_t i = 0; i < test_images; ++i) {
                loader.get();
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < test_images; ++i) {
            loader.prefetch(i + 10000, files[i], params);
        }
        for (size_t i = 0; i < test_images; ++i) {
            loader.get();
        }
        auto end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        auto stats = loader.get_stats();

        std::cout << (pass == 0 ? "Cold" : "Warm") << " pass (" << test_images << " images):\n";
        std::cout << "  Total time:     " << std::fixed << std::setprecision(1) << total_ms << " ms\n";
        std::cout << "  File I/O:       " << stats.file_read_time_ms << " ms ("
                  << (stats.file_read_time_ms / total_ms * 100) << "%)\n";
        std::cout << "  Cache lookup:   " << stats.cache_lookup_time_ms << " ms ("
                  << (stats.cache_lookup_time_ms / total_ms * 100) << "%)\n";
        std::cout << "  GPU decode:     " << stats.gpu_decode_time_ms << " ms ("
                  << (stats.gpu_decode_time_ms / total_ms * 100) << "%)\n";
        std::cout << "  Cold process:   " << stats.cold_process_time_ms << " ms ("
                  << (stats.cold_process_time_ms / total_ms * 100) << "%)\n";
        std::cout << "  Throughput:     " << (test_images / total_ms * 1000) << " img/s\n";

        if (stats.total_decode_calls > 0) {
            std::cout << "  Decode rate:    " << (stats.total_decode_calls / (stats.gpu_decode_time_ms / 1000.0))
                      << " img/s (GPU)\n";
        }
        std::cout << "\n";
    }
}
