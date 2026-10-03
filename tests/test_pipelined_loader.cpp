/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/tensor_serialization.hpp"
#include "cuda_backend_test.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include "licht_test_support.hpp"
#include "training/dataset.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace lfs::core;
using namespace lfs::io;

namespace {

    class PipelinedImageLoaderTest : public lfs::test::CudaBackendTest {
    protected:
        static std::uint64_t process_id() {
#ifdef _WIN32
            return static_cast<std::uint64_t>(_getpid());
#else
            return static_cast<std::uint64_t>(getpid());
#endif
        }

        static void SetUpTestSuite() {
            if (!gpu_backend_available(GpuBackend::CUDA)) {
                GTEST_SKIP() << "Image decoding requires an available CUDA device";
            }
            image_path_ = std::filesystem::path(TEST_DATA_DIR) /
                          "bicycle/images_4/_DSC8744.JPG";
            ASSERT_TRUE(std::filesystem::is_regular_file(image_path_)) << image_path_;
            mask_path_ = std::filesystem::temp_directory_path() /
                         ("lfs_pipelined_loader_bicycle_mask_" +
                          std::to_string(process_id()) + ".png");
            if (!std::filesystem::is_regular_file(mask_path_)) {
                auto [pixels, width, height, channels] = lfs::core::load_image(image_path_);
                const bool valid = pixels != nullptr && width > 0 && height > 0 && channels > 0;
                lfs::core::Tensor mask;
                if (valid) {
                    mask = lfs::core::Tensor::empty(
                        {static_cast<size_t>(height), static_cast<size_t>(width), size_t{1}},
                        lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                    auto* target = mask.ptr<uint8_t>();
                    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
                        const size_t x = i % static_cast<size_t>(width);
                        target[i] = width > 1
                                        ? static_cast<uint8_t>((255u * x) /
                                                               static_cast<size_t>(width - 1))
                                        : uint8_t{0};
                    }
                }
                if (pixels)
                    lfs::core::free_image(pixels);
                ASSERT_TRUE(valid) << image_path_;
                ASSERT_NO_THROW(lfs::core::save_image(mask_path_, std::move(mask)));
            }
            ASSERT_TRUE(std::filesystem::is_regular_file(mask_path_)) << mask_path_;
        }

        static void TearDownTestSuite() {
            std::error_code ec;
            std::filesystem::remove(mask_path_, ec);
        }

        void SetUp() override {
            LFS_CUDA_BACKEND_OR_RETURN();
            ASSERT_TRUE(std::filesystem::is_regular_file(image_path_)) << image_path_;
            ASSERT_TRUE(std::filesystem::is_regular_file(mask_path_)) << mask_path_;
        }

        static PipelinedLoaderConfig config() {
            PipelinedLoaderConfig result;
            result.jpeg_batch_size = 2;
            result.prefetch_count = 4;
            result.output_queue_size = 4;
            result.decoder_pool_size = 2;
            result.io_threads = 1;
            result.cold_process_threads = 1;
            result.max_cache_bytes = 64 * 1024 * 1024;
            return result;
        }

        ImageRequest request(const size_t sequence_id,
                             const int max_width,
                             const bool with_mask = true) const {
            ImageRequest result;
            result.sequence_id = sequence_id;
            result.path = image_path_;
            result.params.resize_factor = 1;
            result.params.max_width = max_width;
            if (with_mask) {
                result.mask_path = mask_path_;
            }
            return result;
        }

        static std::filesystem::path image_path_;
        static std::filesystem::path mask_path_;
    };

    std::filesystem::path PipelinedImageLoaderTest::image_path_;
    std::filesystem::path PipelinedImageLoaderTest::mask_path_;

    std::vector<float> mask_values(const ReadyImage& ready) {
        EXPECT_TRUE(ready.mask.has_value());
        if (!ready.mask) {
            return {};
        }
        return ready.mask->cpu().to_vector();
    }

} // namespace

TEST_F(PipelinedImageLoaderTest, LoadsRealImageAndMaskWithExpectedContract) {
    PipelinedImageLoader loader(config());
    loader.prefetch({request(7, 128)});
    const auto ready = loader.get();

    EXPECT_TRUE(ready.error.empty()) << ready.error;
    EXPECT_EQ(ready.sequence_id, 7u);
    ASSERT_TRUE(ready.tensor.is_valid());
    EXPECT_EQ(gpu_backend_of(ready.tensor), GpuBackend::CUDA);
    EXPECT_EQ(ready.tensor.device(), Device::GPU);
    EXPECT_EQ(ready.tensor.dtype(), DataType::Float32);
    ASSERT_EQ(ready.tensor.shape().rank(), 3u);
    EXPECT_EQ(ready.tensor.shape()[0], 3u);
    EXPECT_LE(std::max(ready.tensor.shape()[1], ready.tensor.shape()[2]), 128u);

    ASSERT_TRUE(ready.mask.has_value());
    EXPECT_EQ(ready.mask->device(), Device::GPU);
    EXPECT_EQ(ready.mask->dtype(), DataType::Float32);
    EXPECT_EQ(ready.mask->shape(),
              TensorShape({ready.tensor.shape()[1], ready.tensor.shape()[2]}));
    EXPECT_GE(ready.mask->min().item<float>(), 0.0f);
    EXPECT_LE(ready.mask->max().item<float>(), 1.0f);
}

TEST_F(PipelinedImageLoaderTest, OriginalJpegUsesDirectDecodeWithoutColdReencoding) {
    for (const bool high_precision : {false, true}) {
        SCOPED_TRACE(high_precision);
        auto settings = config();
        settings.use_16bit_color = high_precision;
        PipelinedImageLoader loader(settings);
        auto input = request(0, 0, false);
        input.params.resize_factor = 1;
        input.params.output_uint8 = !high_precision;
        loader.prefetch({input});
        const auto ready = loader.get();
        ASSERT_TRUE(ready.tensor.is_valid());
        EXPECT_EQ(ready.tensor.dtype(), high_precision ? DataType::Float32 : DataType::UInt8);
        const auto stats = loader.get_stats();
        EXPECT_EQ(stats.cold_path_misses, 0u);
        EXPECT_EQ(stats.cpu_decode_calls, 0u);
        EXPECT_EQ(stats.hot_path_hits, 1u);
        input.sequence_id = 1;
        loader.prefetch({input});
        const auto repeated = loader.get();
        EXPECT_EQ(ready.tensor.to(DataType::Float32).cpu().to_vector(),
                  repeated.tensor.to(DataType::Float32).cpu().to_vector());
    }
}

TEST_F(PipelinedImageLoaderTest, UndistortedTrainingImageIsIdenticalForImmediateColdAndHit) {
    const auto [width, height, channels] = lfs::core::get_image_info(image_path_);
    ASSERT_GT(width, 0);
    ASSERT_GT(height, 0);
    ASSERT_GE(channels, 3);
    const float focal = 0.9765625f * static_cast<float>(width);
    const auto undistort = lfs::core::compute_undistort_params(
        focal, focal, 0.5f * width, 0.5f * height, width, height,
        Tensor::from_vector({0.10f, -0.33f, 0.85f}, {3}, Device::CPU),
        Tensor::from_vector({0.001f, -0.001f}, {2}, Device::CPU),
        CameraModelType::PINHOLE, 0.0f);

    LoadParams params;
    params.resize_factor = 1;
    params.max_width = 128;
    params.output_uint8 = true;
    params.undistort = &undistort;

    PipelinedImageLoader loader(config());
    const auto immediate = loader.load_image_immediate(image_path_, params);
    ASSERT_TRUE(immediate.is_valid());

    ImageRequest request;
    request.sequence_id = 1;
    request.path = image_path_;
    request.params = params;
    request.undistort = &undistort;
    loader.prefetch({request});
    const auto cold = loader.get();
    ASSERT_TRUE(cold.error.empty()) << cold.error;

    request.sequence_id = 2;
    loader.prefetch({request});
    const auto hit = loader.get();
    ASSERT_TRUE(hit.error.empty()) << hit.error;

    const auto expect_identical = [](const Tensor& lhs, const Tensor& rhs) {
        ASSERT_EQ(lhs.shape(), rhs.shape());
        ASSERT_EQ(lhs.dtype(), rhs.dtype());
        const auto lhs_cpu = lhs.cpu().contiguous();
        const auto rhs_cpu = rhs.cpu().contiguous();
        ASSERT_EQ(lhs_cpu.bytes(), rhs_cpu.bytes());
        EXPECT_EQ(std::memcmp(lhs_cpu.data_ptr(), rhs_cpu.data_ptr(), lhs_cpu.bytes()), 0);
    };
    expect_identical(immediate, cold.tensor);
    expect_identical(cold.tensor, hit.tensor);
}

// Fails if a release frees nothing, spills a newer image before the oldest, or
// loses pixels on the way through the spill.
TEST_F(PipelinedImageLoaderTest, ReleaseHostCacheSpillsLeastRecentImagesFirst) {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(image_path_.parent_path())) {
        if (entry.path().extension() == ".JPG")
            paths.push_back(entry.path());
    }
    std::ranges::sort(paths);
    ASSERT_GE(paths.size(), 3u);
    paths.resize(3);

    PipelinedImageLoader loader(config());
    const auto load = [&loader](const size_t sequence_id, const std::filesystem::path& path) {
        ImageRequest input;
        input.sequence_id = sequence_id;
        input.path = path;
        input.params.resize_factor = 1;
        input.params.max_width = 0;
        input.params.output_uint8 = true;
        loader.prefetch({input});
        const auto ready = loader.get();
        EXPECT_TRUE(ready.error.empty()) << ready.error;
        return ready.tensor.to(DataType::Float32).cpu().to_vector();
    };

    std::vector<std::vector<float>> first_pass;
    for (size_t i = 0; i < paths.size(); ++i)
        first_pass.push_back(load(i, paths[i]));

    const auto cached = loader.get_stats();
    ASSERT_EQ(cached.jpeg_cache_entries, paths.size());
    const auto oldest_bytes = static_cast<size_t>(std::filesystem::file_size(paths.front()));

    EXPECT_EQ(loader.release_host_cache(1), oldest_bytes);
    const auto after_one = loader.get_stats();
    EXPECT_EQ(after_one.jpeg_cache_entries, paths.size() - 1);
    EXPECT_EQ(after_one.jpeg_cache_bytes, cached.jpeg_cache_bytes - oldest_bytes);
    EXPECT_EQ(after_one.spill_cache_entries, cached.spill_cache_entries + 1);

    EXPECT_EQ(loader.release_host_cache(cached.jpeg_cache_bytes), after_one.jpeg_cache_bytes);
    const auto after_all = loader.get_stats();
    EXPECT_EQ(after_all.jpeg_cache_entries, 0u);
    EXPECT_EQ(after_all.jpeg_cache_bytes, 0u);
    EXPECT_EQ(after_all.spill_cache_entries, cached.spill_cache_entries + paths.size());
    EXPECT_EQ(loader.release_host_cache(1), 0u);

    for (size_t i = 0; i < paths.size(); ++i)
        EXPECT_EQ(load(paths.size() + i, paths[i]), first_pass[i]) << paths[i];
}

TEST_F(PipelinedImageLoaderTest, TrainingStartupOnlyPrefetchesBoundedBatch) {
    const auto empty = Tensor::zeros({0}, Device::CPU);
    const auto camera = std::make_shared<Camera>(
        Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
        1.f, 1.f, .5f, .5f, empty, empty, CameraModelType::PINHOLE,
        image_path_.filename().string(), image_path_, std::filesystem::path{}, 1, 1, 0);
    for (const size_t count : {128u, 5114u}) {
        SCOPED_TRACE(count);
        std::vector<std::shared_ptr<Camera>> cameras(count, camera);
        lfs::training::DatasetConfig dataset_config;
        dataset_config.resize_factor = 1;
        dataset_config.max_width = 32;
        auto dataset = std::make_shared<lfs::training::CameraDataset>(std::move(cameras), dataset_config);
        const auto started = std::chrono::steady_clock::now();
        lfs::training::PipelinedDataLoader<lfs::training::InfiniteRandomSampler> loader(
            dataset, lfs::training::InfiniteRandomSampler(count, 42), config());
        const auto stats = loader.get_stats();
        EXPECT_LE(stats.accepted_sequences, config().prefetch_count);
        EXPECT_GT(stats.accepted_sequences, 0u);
        auto first = loader.next();
        ASSERT_TRUE(first);
        EXPECT_TRUE(first->data.image.is_valid());
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "Startup for " << count << " cameras: " << elapsed << " s; accepted " << stats.accepted_sequences << " requests\n";
    }
}

TEST_F(PipelinedImageLoaderTest, PngMaxWidthUsesOneDecode) {
    PipelinedImageLoader loader(config());
    const auto before = loader.get_stats().cpu_decode_calls;

    LoadParams params;
    params.max_width = 16;
    params.output_uint8 = true;
    const auto tensor = loader.load_image_immediate(mask_path_, params);

    const auto after = loader.get_stats().cpu_decode_calls;
    EXPECT_EQ(after - before, 1U);
    ASSERT_TRUE(tensor.is_valid());
    ASSERT_EQ(tensor.shape().rank(), 3U);
    EXPECT_LE(std::max(tensor.shape()[1], tensor.shape()[2]), 16U);
}

TEST_F(PipelinedImageLoaderTest, ImmediateCacheHitDoesNotRepeatResize) {
    PipelinedImageLoader loader(config());
    auto input = request(0, 0, false);
    input.params.resize_factor = 2;
    input.params.output_uint8 = true;

    loader.prefetch({input});
    const auto cold = loader.get();
    ASSERT_TRUE(cold.error.empty()) << cold.error;
    ASSERT_TRUE(cold.tensor.is_valid());
    ASSERT_EQ(loader.get_stats().jpeg_cache_entries, 1U);

    input.sequence_id = 1;
    loader.prefetch({input});
    const auto hot = loader.get();
    ASSERT_TRUE(hot.error.empty()) << hot.error;

    const auto immediate = loader.load_image_immediate(input.path, input.params);
    ASSERT_TRUE(immediate.is_valid());
    EXPECT_EQ(hot.tensor.shape(), cold.tensor.shape());
    EXPECT_EQ(immediate.shape(), hot.tensor.shape());
    EXPECT_EQ(immediate.to(DataType::Float32).cpu().to_vector(),
              hot.tensor.to(DataType::Float32).cpu().to_vector());
}

TEST_F(PipelinedImageLoaderTest, ImmediateCacheHitDoesNotRepeatUndistortion) {
    const lfs::test::licht::TemporaryDirectory temp("lfs-immediate-undistort");
    constexpr int width = 128;
    constexpr int height = 96;
    std::vector<uint8_t> pixels(width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t offset = static_cast<size_t>(y * width + x) * 3;
            pixels[offset] = static_cast<uint8_t>((x * 255) / (width - 1));
            pixels[offset + 1] = static_cast<uint8_t>((y * 255) / (height - 1));
            pixels[offset + 2] = static_cast<uint8_t>((x + y) & 0xff);
        }
    }
    const auto image_path = temp.path / "distorted.png";
    ASSERT_TRUE(save_png(image_path, pixels.data(), width, height, 3, 8, 1));

    UndistortParams undistort{};
    undistort.src_width = undistort.dst_width = width;
    undistort.src_height = undistort.dst_height = height;
    undistort.src_fx = undistort.dst_fx = 100.0f;
    undistort.src_fy = undistort.dst_fy = 100.0f;
    undistort.src_cx = undistort.dst_cx = width / 2.0f;
    undistort.src_cy = undistort.dst_cy = height / 2.0f;
    undistort.model_type = CameraModelType::PINHOLE;
    undistort.distortion[0] = 0.08f;
    undistort.distortion[1] = -0.02f;
    undistort.distortion[3] = 0.001f;
    undistort.distortion[4] = -0.0015f;
    undistort.num_distortion = 5;

    PipelinedImageLoader loader(config());
    ImageRequest request{};
    request.sequence_id = 0;
    request.path = image_path;
    request.params.resize_factor = 1;
    request.params.output_uint8 = true;
    request.params.undistort = &undistort;
    request.undistort = &undistort;

    loader.prefetch({request});
    const auto cold = loader.get();
    ASSERT_TRUE(cold.error.empty()) << cold.error;
    ASSERT_EQ(loader.get_stats().jpeg_cache_entries, 1U);

    request.sequence_id = 1;
    loader.prefetch({request});
    const auto hot = loader.get();
    ASSERT_TRUE(hot.error.empty()) << hot.error;

    const auto immediate = loader.load_image_immediate(image_path, request.params);
    ASSERT_TRUE(immediate.is_valid());
    EXPECT_EQ(hot.tensor.shape(), cold.tensor.shape());
    EXPECT_EQ(immediate.shape(), hot.tensor.shape());
    EXPECT_EQ(immediate.to(DataType::Float32).cpu().to_vector(),
              hot.tensor.to(DataType::Float32).cpu().to_vector());
}

// The single-path prefetch overload must carry the undistortion like a full request; otherwise the
// cold path caches distorted pixels under the undistorted key and every later hit returns them.
TEST_F(PipelinedImageLoaderTest, PathPrefetchUndistortsLikeFullRequest) {
    const lfs::test::licht::TemporaryDirectory temp("lfs-path-prefetch-undistort");
    constexpr int width = 128;
    constexpr int height = 96;
    std::vector<uint8_t> pixels(width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t offset = static_cast<size_t>(y * width + x) * 3;
            pixels[offset] = static_cast<uint8_t>((x * 255) / (width - 1));
            pixels[offset + 1] = static_cast<uint8_t>((y * 255) / (height - 1));
            pixels[offset + 2] = static_cast<uint8_t>((x + y) & 0xff);
        }
    }
    const auto image_path = temp.path / "distorted.png";
    ASSERT_TRUE(save_png(image_path, pixels.data(), width, height, 3, 8, 1));

    UndistortParams undistort{};
    undistort.src_width = undistort.dst_width = width;
    undistort.src_height = undistort.dst_height = height;
    undistort.src_fx = undistort.dst_fx = 100.0f;
    undistort.src_fy = undistort.dst_fy = 100.0f;
    undistort.src_cx = undistort.dst_cx = width / 2.0f;
    undistort.src_cy = undistort.dst_cy = height / 2.0f;
    undistort.model_type = CameraModelType::PINHOLE;
    undistort.distortion[0] = 0.08f;
    undistort.num_distortion = 1;

    LoadParams params;
    params.resize_factor = 1;
    params.output_uint8 = true;
    params.undistort = &undistort;

    PipelinedImageLoader path_loader(config());
    path_loader.prefetch(0, image_path, params);
    const auto from_path = path_loader.get();
    ASSERT_TRUE(from_path.error.empty()) << from_path.error;

    PipelinedImageLoader request_loader(config());
    ImageRequest request{};
    request.path = image_path;
    request.params = params;
    request.undistort = &undistort;
    request_loader.prefetch({request});
    const auto from_request = request_loader.get();
    ASSERT_TRUE(from_request.error.empty()) << from_request.error;

    EXPECT_EQ(from_path.tensor.to(DataType::Float32).cpu().to_vector(),
              from_request.tensor.to(DataType::Float32).cpu().to_vector());
}

TEST_F(PipelinedImageLoaderTest, ResizeAndMaxWidthKeepImageAndMaskAligned) {
    PipelinedImageLoader loader(config());
    loader.prefetch({request(1, 256)});
    const auto large = loader.get();
    loader.prefetch({request(2, 96)});
    const auto small = loader.get();

    ASSERT_TRUE(large.mask.has_value());
    ASSERT_TRUE(small.mask.has_value());
    EXPECT_EQ(large.mask->shape(),
              TensorShape({large.tensor.shape()[1], large.tensor.shape()[2]}));
    EXPECT_EQ(small.mask->shape(),
              TensorShape({small.tensor.shape()[1], small.tensor.shape()[2]}));
    EXPECT_LE(std::max(small.tensor.shape()[1], small.tensor.shape()[2]), 96u);
    EXPECT_LT(small.tensor.shape()[1] * small.tensor.shape()[2],
              large.tensor.shape()[1] * large.tensor.shape()[2]);
}

TEST_F(PipelinedImageLoaderTest, MaskCacheHitPreservesInvertAndThresholdSemantics) {
    PipelinedImageLoader loader(config());

    constexpr float threshold = 0.25f;
    auto threshold_request = request(1, 0);
    threshold_request.mask_params.threshold = threshold;
    loader.prefetch({threshold_request});
    const auto thresholded_cold = loader.get();

    threshold_request.sequence_id = 2;
    loader.prefetch({threshold_request});
    const auto thresholded_hot = loader.get();

    auto normal_request = request(3, 0);
    loader.prefetch({normal_request});
    const auto normal = loader.get();

    auto inverted_request = request(4, 0);
    inverted_request.mask_params.invert = true;
    loader.prefetch({inverted_request});
    const auto inverted = loader.get();

    const auto normal_values = mask_values(normal);
    const auto inverted_values = mask_values(inverted);
    const auto thresholded_cold_values = mask_values(thresholded_cold);
    const auto thresholded_hot_values = mask_values(thresholded_hot);
    ASSERT_EQ(inverted_values.size(), normal_values.size());
    ASSERT_EQ(thresholded_cold_values.size(), normal_values.size());
    ASSERT_EQ(thresholded_hot_values.size(), normal_values.size());

    size_t zeros = 0;
    size_t ones = 0;
    for (size_t i = 0; i < normal_values.size(); ++i) {
        // UINT8 lossless J2K quantization is 1/255.
        EXPECT_NEAR(inverted_values[i], 1.0f - normal_values[i], 0.01f);
        const float expected = normal_values[i] >= threshold ? 1.0f : 0.0f;
        EXPECT_FLOAT_EQ(thresholded_cold_values[i], expected);
        EXPECT_FLOAT_EQ(thresholded_hot_values[i], expected);
        zeros += thresholded_hot_values[i] == 0.0f;
        ones += thresholded_hot_values[i] == 1.0f;
    }
    EXPECT_GT(zeros, 0u);
    EXPECT_GT(ones, 0u);
}

TEST_F(PipelinedImageLoaderTest, MultipleRequestsPreserveIdsAndOptionalMask) {
    PipelinedImageLoader loader(config());
    loader.prefetch({request(11, 96, false), request(12, 96, true)});

    std::map<size_t, bool> mask_by_sequence;
    for (int i = 0; i < 2; ++i) {
        const auto ready = loader.get();
        EXPECT_TRUE(ready.error.empty()) << ready.error;
        ASSERT_TRUE(ready.tensor.is_valid());
        mask_by_sequence.emplace(ready.sequence_id, ready.mask.has_value());
    }

    EXPECT_EQ(mask_by_sequence,
              (std::map<size_t, bool>{{11u, false}, {12u, true}}));
}

class PipelinedMaskUndistortTest : public lfs::test::CudaBackendTest,
                                   public ::testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(PipelinedMaskUndistortTest, ProcessesEntireOutputMaskOnColdAndRepeatedLoad) {
    const auto [alpha_mask, enlarge] = GetParam();
    const lfs::test::licht::TemporaryDirectory temp("lfs-mask-undistort");
    constexpr int src_w = 128;
    constexpr int src_h = 96;
    const int dst_w = enlarge ? src_w * 2 : src_w / 2;
    const int dst_h = enlarge ? src_h * 2 : src_h / 2;
    const auto image_path = temp.path / "image.png";
    const auto mask_path = temp.path / "mask.png";
    const std::vector<uint8_t> rgba(src_w * src_h * 4, 64);
    const std::vector<uint8_t> mask(src_w * src_h, 64);
    ASSERT_TRUE(save_png(image_path, rgba.data(), src_w, src_h, 4, 8, 1));
    ASSERT_TRUE(save_png(mask_path, mask.data(), src_w, src_h, 1, 8, 1));

    UndistortParams undistort{};
    undistort.src_width = src_w;
    undistort.src_height = src_h;
    undistort.dst_width = dst_w;
    undistort.dst_height = dst_h;
    undistort.src_fx = undistort.src_fy = src_w;
    undistort.src_cx = src_w / 2.0f;
    undistort.src_cy = src_h / 2.0f;
    undistort.dst_fx = undistort.dst_fy = static_cast<float>(dst_w);
    undistort.dst_cx = dst_w / 2.0f;
    undistort.dst_cy = dst_h / 2.0f;
    undistort.model_type = CameraModelType::PINHOLE;

    PipelinedLoaderConfig config;
    config.jpeg_batch_size = 1;
    config.prefetch_count = 1;
    config.output_queue_size = 1;
    config.decoder_pool_size = 1;
    config.io_threads = 1;
    config.cold_process_threads = 1;
    PipelinedImageLoader loader(config);
    ImageRequest request{};
    request.path = image_path;
    request.params.resize_factor = 1;
    request.params.max_width = 0;
    request.params.undistort = &undistort;
    request.undistort = &undistort;
    request.extract_alpha_as_mask = alpha_mask;
    request.mask_params = {.invert = true, .threshold = 0.5f};
    request.alpha_mask_params = request.mask_params;
    if (!alpha_mask)
        request.mask_path = mask_path;

    for (size_t sequence = 0; sequence < 2; ++sequence) {
        SCOPED_TRACE(sequence);
        request.sequence_id = sequence;
        loader.prefetch({request});
        const auto ready = loader.try_get_for(std::chrono::seconds(20));
        ASSERT_TRUE(ready.has_value());
        ASSERT_TRUE(ready->error.empty()) << ready->error;
        ASSERT_TRUE(ready->mask.has_value());
        EXPECT_EQ(ready->tensor.shape(), TensorShape({3, static_cast<size_t>(dst_h), static_cast<size_t>(dst_w)}));
        EXPECT_EQ(ready->mask->shape(), TensorShape({static_cast<size_t>(dst_h), static_cast<size_t>(dst_w)}));
        // All source values are below 0.5, including undistortion's zero border.
        // Invert then threshold must keep every output pixel, including the
        // tail beyond the source pixel count when undistortion enlarges it.
        const auto values = ready->mask->cpu().to_vector();
        ASSERT_FALSE(values.empty());
        EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](const float value) { return value == 1.0f; }));
    }
}

INSTANTIATE_TEST_SUITE_P(AlphaAndSidecar, PipelinedMaskUndistortTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool()));

TEST(SidecarResumeSampler, DeterministicCameraStreamContinuesAtCheckpointOffset) {
    constexpr std::uint64_t seed = 0x4c46535f73616d70ULL;
    constexpr size_t dataset_size = 17;
    constexpr size_t checkpoint_iteration = 31;

    lfs::training::InfiniteRandomSampler uninterrupted(dataset_size, seed, 0);
    lfs::training::InfiniteRandomSampler resumed(dataset_size, seed, checkpoint_iteration);

    for (size_t i = 0; i < 40; ++i) {
        const auto expected = uninterrupted.next(1);
        ASSERT_TRUE(expected.has_value());
        if (i < checkpoint_iteration)
            continue;
        const auto actual = resumed.next(1);
        ASSERT_TRUE(actual.has_value());
        EXPECT_EQ(*actual, *expected) << "camera stream diverged at post-resume sample " << i;
    }
}
