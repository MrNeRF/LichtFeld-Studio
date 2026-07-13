/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/path_utils.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"

#include <OpenImageIO/imageio.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

    namespace fs = std::filesystem;

    std::string path_string(const fs::path& path) {
        return path.string();
    }

    fs::path bicycle_image_path() {
        return fs::path(PROJECT_ROOT_PATH) / "data/bicycle/images_4/_DSC8679.JPG";
    }

    struct TempFileGuard {
        fs::path path;
        ~TempFileGuard() {
            std::error_code ec;
            fs::remove(path, ec);
        }
    };

    fs::path write_depth_png_from_bicycle_content(const std::string& filename, const uint16_t variant) {
        const fs::path source_path = bicycle_image_path();
        std::unique_ptr<OIIO::ImageInput> input(OIIO::ImageInput::open(path_string(source_path)));
        if (!input) {
            throw std::runtime_error("Failed to open bicycle source: " + path_string(source_path));
        }

        const OIIO::ImageSpec source_spec = input->spec();
        std::vector<uint8_t> rgb(static_cast<size_t>(source_spec.width) * source_spec.height * 3);
        if (!input->read_image(0, 0, 0, 3, OIIO::TypeDesc::UINT8, rgb.data())) {
            throw std::runtime_error("Failed to read bicycle source: " + input->geterror());
        }
        input->close();

        std::vector<uint16_t> depth(static_cast<size_t>(source_spec.width) * source_spec.height);
        for (int y = 0; y < source_spec.height; ++y) {
            for (int x = 0; x < source_spec.width; ++x) {
                const size_t rgb_idx = (static_cast<size_t>(y) * source_spec.width + x) * 3;
                const uint32_t luma =
                    static_cast<uint32_t>(rgb[rgb_idx + 0]) * 77u +
                    static_cast<uint32_t>(rgb[rgb_idx + 1]) * 150u +
                    static_cast<uint32_t>(rgb[rgb_idx + 2]) * 29u;
                const uint16_t base = static_cast<uint16_t>((luma >> 8u) * 257u);
                depth[static_cast<size_t>(y) * source_spec.width + x] =
                    static_cast<uint16_t>(base ^ variant ^
                                          static_cast<uint16_t>((x * 17 + y * 31) & 0xffu));
            }
        }
        depth.front() = 0u;
        depth.back() = 65535u;

        const fs::path output_path =
            fs::temp_directory_path() / filename;
        std::unique_ptr<OIIO::ImageOutput> output(OIIO::ImageOutput::create(path_string(output_path)));
        if (!output) {
            throw std::runtime_error("Failed to create depth PNG output: " + OIIO::geterror());
        }

        OIIO::ImageSpec spec(source_spec.width, source_spec.height, 1, OIIO::TypeDesc::UINT16);
        spec.attribute("png:compressionLevel", 1);
        if (!output->open(path_string(output_path), spec)) {
            throw std::runtime_error("Failed to open depth PNG output: " + output->geterror());
        }
        if (!output->write_image(OIIO::TypeDesc::UINT16, depth.data())) {
            throw std::runtime_error("Failed to write depth PNG output: " + output->geterror());
        }
        output->close();
        return output_path;
    }

    fs::path write_depth_png_from_bicycle_content() {
        return write_depth_png_from_bicycle_content("lfs_pipelined_sidecar_jpeg2k_depth16.png", 0u);
    }

    fs::path pipeline_cache_dir() {
        return fs::path("/tmp") / "LichtFeld" / "pipeline_cache" / "ppl_j2k_unified_v1";
    }

    void clear_pipeline_cache_dir() {
        std::error_code ec;
        fs::remove_all(pipeline_cache_dir(), ec);
    }

    std::vector<fs::path> list_pipeline_cache_dirs() {
        const fs::path base = fs::path("/tmp") / "LichtFeld" / "pipeline_cache";
        std::vector<fs::path> dirs;
        std::error_code ec;
        if (!fs::is_directory(base, ec) || ec) {
            return dirs;
        }
        for (const auto& entry : fs::directory_iterator(base, ec)) {
            if (ec) {
                break;
            }
            if (!entry.is_directory(ec) || ec) {
                ec.clear();
                continue;
            }
            const auto name = entry.path().filename().string();
            if (name.starts_with("ppl_")) {
                dirs.push_back(entry.path());
            }
        }
        return dirs;
    }

    fs::path find_new_pipeline_cache_dir(const std::vector<fs::path>& before) {
        std::unordered_set<std::string> before_set;
        for (const auto& dir : before) {
            before_set.insert(dir.string());
        }

        for (const auto& dir : list_pipeline_cache_dirs()) {
            if (!before_set.contains(dir.string())) {
                return dir;
            }
        }
        throw std::runtime_error("Failed to find new pipelined loader cache directory");
    }

    std::vector<fs::path> corrupt_cache_payloads(const fs::path& cache_dir) {
        std::vector<fs::path> corrupted;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(cache_dir, ec)) {
            if (ec) {
                break;
            }
            if (!entry.is_regular_file(ec) || ec || entry.path().extension() != ".jpg") {
                ec.clear();
                continue;
            }

            std::fstream file(entry.path(), std::ios::binary | std::ios::in | std::ios::out);
            if (!file) {
                continue;
            }
            file.seekp(0, std::ios::beg);
            const std::array<char, 16> poison{};
            file.write(poison.data(), static_cast<std::streamsize>(poison.size()));
            if (file.good()) {
                corrupted.push_back(entry.path());
            }
        }
        return corrupted;
    }

    lfs::io::ReadyImage wait_for_ready(lfs::io::PipelinedImageLoader& loader) {
        auto ready = loader.try_get_for(std::chrono::seconds(20));
        if (!ready) {
            throw std::runtime_error("Timed out waiting for pipelined loader output");
        }
        if (ready->depth_ready_event) {
            const cudaError_t wait_status = cudaEventSynchronize(ready->depth_ready_event);
            if (wait_status != cudaSuccess) {
                throw std::runtime_error(std::string("depth event wait failed: ") +
                                         cudaGetErrorString(wait_status));
            }
            cudaEventDestroy(ready->depth_ready_event);
            ready->depth_ready_event = nullptr;
        }
        return std::move(*ready);
    }

} // namespace

TEST(PipelinedImageLoaderSidecarJpeg2k, DepthFirstTouchTranscodesThenHotDecodes) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GT(device_count, 0);
    ASSERT_TRUE(lfs::io::NvCodecImageLoader::is_available());
    clear_pipeline_cache_dir();

    const TempFileGuard depth_file{write_depth_png_from_bicycle_content()};
    const fs::path& depth_path = depth_file.path;

    lfs::io::PipelinedLoaderConfig config;
    config.io_threads = 1;
    config.cold_process_threads = 1;
    config.jpeg_batch_size = 2;
    config.decoder_pool_size = 2;
    config.output_queue_size = 2;
    config.prefetch_count = 2;
    config.use_filesystem_cache = true;
    config.max_cache_bytes = 512ULL * 1024ULL * 1024ULL;

    lfs::io::PipelinedImageLoader loader(config);

    lfs::io::ImageRequest first;
    first.sequence_id = 1;
    first.path = bicycle_image_path();
    first.depth_path = depth_path;
    first.params.output_uint8 = false;

    loader.prefetch(std::vector<lfs::io::ImageRequest>{first});
    auto first_ready = wait_for_ready(loader);
    ASSERT_TRUE(first_ready.depth.has_value());
    const auto stats_after_first = loader.get_stats();
    EXPECT_GE(stats_after_first.jpeg_cache_entries, size_t{2});
    EXPECT_GT(stats_after_first.jpeg_cache_bytes, size_t{0});
    std::error_code remove_ec;
    fs::remove(depth_path, remove_ec);
    ASSERT_FALSE(fs::exists(depth_path));

    lfs::io::ImageRequest second = first;
    second.sequence_id = 2;
    loader.prefetch(std::vector<lfs::io::ImageRequest>{second});
    auto second_ready = wait_for_ready(loader);
    ASSERT_TRUE(second_ready.depth.has_value());

    const auto stats_after_second = loader.get_stats();
    EXPECT_GT(stats_after_second.hot_path_hits, stats_after_first.hot_path_hits);

    const std::vector<float> first_values = first_ready.depth->cpu().to_vector();
    const std::vector<float> second_values = second_ready.depth->cpu().to_vector();
    ASSERT_EQ(second_values.size(), first_values.size());

    constexpr float kTolerance = 1.0f / 65535.0f;
    for (size_t i = 0; i < first_values.size(); ++i) {
        ASSERT_LE(std::abs(second_values[i] - first_values[i]), kTolerance)
            << "depth mismatch at element " << i;
    }
}

TEST(PipelinedImageLoaderSidecarJpeg2k, CorruptFilesystemSidecarFallsBackAndRepairs) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GT(device_count, 0);
    ASSERT_TRUE(lfs::io::NvCodecImageLoader::is_available());
    clear_pipeline_cache_dir();

    const TempFileGuard depth_a{
        write_depth_png_from_bicycle_content("lfs_pipelined_sidecar_jpeg2k_depth16_a.png", 0u)};
    const TempFileGuard depth_b{
        write_depth_png_from_bicycle_content("lfs_pipelined_sidecar_jpeg2k_depth16_b.png", 0x1357u)};

    lfs::io::PipelinedLoaderConfig config;
    config.io_threads = 1;
    config.cold_process_threads = 1;
    config.jpeg_batch_size = 2;
    config.decoder_pool_size = 2;
    config.output_queue_size = 2;
    config.prefetch_count = 2;
    config.use_filesystem_cache = true;
    config.max_cache_bytes = 1;

    const auto cache_dirs_before = list_pipeline_cache_dirs();
    lfs::io::PipelinedImageLoader loader(config);
    const fs::path cache_dir = find_new_pipeline_cache_dir(cache_dirs_before);

    lfs::io::ImageRequest first;
    first.sequence_id = 1;
    first.path = bicycle_image_path();
    first.depth_path = depth_a.path;
    first.params.output_uint8 = false;

    loader.prefetch(std::vector<lfs::io::ImageRequest>{first});
    auto first_ready = wait_for_ready(loader);
    ASSERT_TRUE(first_ready.depth.has_value());
    const std::vector<float> first_values = first_ready.depth->cpu().to_vector();

    const auto corrupted = corrupt_cache_payloads(cache_dir);
    ASSERT_GE(corrupted.size(), size_t{1});

    lfs::io::ImageRequest evicting_request = first;
    evicting_request.sequence_id = 2;
    evicting_request.depth_path = depth_b.path;
    loader.prefetch(std::vector<lfs::io::ImageRequest>{evicting_request});
    auto evicting_ready = wait_for_ready(loader);
    ASSERT_TRUE(evicting_ready.depth.has_value());

    lfs::io::ImageRequest repaired_request = first;
    repaired_request.sequence_id = 3;
    loader.prefetch(std::vector<lfs::io::ImageRequest>{repaired_request});
    auto repaired_ready = wait_for_ready(loader);
    ASSERT_TRUE(repaired_ready.depth.has_value());

    const std::vector<float> repaired_values = repaired_ready.depth->cpu().to_vector();
    ASSERT_EQ(repaired_values.size(), first_values.size());
    constexpr float kTolerance = 1.0f / 65535.0f;
    for (size_t i = 0; i < first_values.size(); ++i) {
        ASSERT_LE(std::abs(repaired_values[i] - first_values[i]), kTolerance)
            << "repaired depth mismatch at element " << i;
    }

    std::error_code remove_ec;
    fs::remove(depth_a.path, remove_ec);
    ASSERT_FALSE(fs::exists(depth_a.path));

    lfs::io::ImageRequest hot_after_repair_request = first;
    hot_after_repair_request.sequence_id = 4;
    loader.prefetch(std::vector<lfs::io::ImageRequest>{hot_after_repair_request});
    auto hot_after_repair_ready = wait_for_ready(loader);
    ASSERT_TRUE(hot_after_repair_ready.depth.has_value());
}
