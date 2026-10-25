/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/parameters.hpp"
#include "io/cuda/image_sharpen.hpp"
#include "io/pipelined_image_loader.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace lfs::core;
using namespace lfs::io;

namespace {

    // Not multiples of the kernel tiles, so partial tiles and clamped borders are covered.
    constexpr int W = 37, H = 29;
    constexpr std::array kAllKernels{SharpenKernel::Usm, SharpenKernel::Laplacian4, SharpenKernel::Laplacian8};

    // [3,H,W] CUDA image from 0..255 levels: UInt8, or Float32 on the 16-bit grid like a decoded 16-bit image.
    template <typename Levels>
    Tensor make_image(const int width, const int height, Levels levels, const DataType dtype) {
        std::vector<float> values(static_cast<size_t>(3) * width * height);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const float level = levels(x, y, c);
                    values[(static_cast<size_t>(c) * height + y) * width + x] =
                        dtype == DataType::UInt8 ? level
                                                 : static_cast<float>(static_cast<int>(level) * 257) * (1.0f / 65535.0f);
                }
        auto image = Tensor::from_vector(values, {3, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CUDA);
        return dtype == DataType::UInt8 ? image.to(DataType::UInt8) : image;
    }

    std::vector<float> values_of(const Tensor& image) {
        return image.to(DataType::Float32).cpu().contiguous().to_vector();
    }

    void expect_identical(const Tensor& lhs, const Tensor& rhs) {
        ASSERT_EQ(lhs.shape(), rhs.shape());
        ASSERT_EQ(lhs.dtype(), rhs.dtype());
        const auto a = lhs.cpu().contiguous();
        const auto b = rhs.cpu().contiguous();
        EXPECT_EQ(std::memcmp(a.data_ptr(), b.data_ptr(), a.bytes()), 0);
    }

    size_t at(const int x, const int y, const int c, const int width = W, const int height = H) {
        return (static_cast<size_t>(c) * height + y) * width + x;
    }

} // namespace

TEST(ImageSharpenTest, FlatImageIsUnchanged) {
    const auto flat = [](int, int, const int c) { return std::array{200.0f, 100.0f, 30.0f}[c]; };
    for (const auto dtype : {DataType::UInt8, DataType::Float32}) {
        const auto image = make_image(W, H, flat, dtype);
        for (const auto kernel : kAllKernels) {
            SCOPED_TRACE(std::string(sharpen_kernel_name(kernel)) + (dtype == DataType::UInt8 ? " u8" : " f32"));
            const auto sharpened = sharpen_image(image, {.kernel = kernel, .amount = 3.0f, .radius = 2.0f}, true, nullptr);
            expect_identical(sharpened, image);
        }
    }
}

// At amount 1 on gray input the Laplacian variants are the 0/-1/5 and -1/9 kernels.
TEST(ImageSharpenTest, LaplacianMatchesTextbookKernelsOnGrayInterior) {
    const auto level = [](const int x, const int y) { return static_cast<float>(64 + (x * 37 + y * 23 + x * y) % 128); };
    const auto image = make_image(W, H, [&](const int x, const int y, int) { return level(x, y); }, DataType::UInt8);
    for (const bool eight : {false, true}) {
        SCOPED_TRACE(eight ? "laplacian8" : "laplacian4");
        const auto kernel = eight ? SharpenKernel::Laplacian8 : SharpenKernel::Laplacian4;
        const auto out = sharpen_image(image, {.kernel = kernel, .amount = 1.0f}, true, nullptr).cpu().to_vector_uint8();
        size_t mismatches = 0;
        for (int y = 1; y + 1 < H; ++y)
            for (int x = 1; x + 1 < W; ++x) {
                float expected = 5.0f * level(x, y) - level(x - 1, y) - level(x + 1, y) - level(x, y - 1) - level(x, y + 1);
                if (eight)
                    expected += 4.0f * level(x, y) - level(x - 1, y - 1) - level(x + 1, y - 1) - level(x - 1, y + 1) -
                                level(x + 1, y + 1);
                expected = std::clamp(expected, 0.0f, 255.0f);
                for (int c = 0; c < 3; ++c)
                    mismatches += out[at(x, y, c)] != static_cast<uint8_t>(expected);
            }
        EXPECT_EQ(mismatches, 0u);
    }
}

TEST(ImageSharpenTest, UsmPushesAStepEdgeApartAndAmountZeroIsANoOp) {
    const int edge = W / 2;
    const auto image = make_image(W, H, [edge](const int x, int, int) { return x < edge ? 60.0f : 180.0f; }, DataType::UInt8);
    const auto out = sharpen_image(image, {.amount = 1.0f, .radius = 1.5f}, true, nullptr).cpu().to_vector_uint8();
    for (int y = 0; y < H; ++y)
        for (int c = 0; c < 3; ++c) {
            EXPECT_LT(out[at(edge - 1, y, c)], 60);
            EXPECT_GT(out[at(edge, y, c)], 180);
            EXPECT_EQ(out[at(0, y, c)], 60);
            EXPECT_EQ(out[at(W - 1, y, c)], 180);
        }
    expect_identical(sharpen_image(image, {.amount = 0.0f, .radius = 1.5f}, true, nullptr), image);
}

TEST(ImageSharpenTest, ThresholdLeavesSmallDifferencesUnchanged) {
    // A +-2 level ripple on the left (|d| about 0.016) and a 100 level edge at x = 30 (|d| about 0.1).
    const auto image = make_image(
        W, H, [](const int x, const int y, int) { return x < 24 ? ((x + y) % 2 ? 102.0f : 98.0f) : (x < 30 ? 80.0f : 180.0f); },
        DataType::UInt8);
    const auto original = values_of(image);
    const auto sharpen = [&](const float threshold) {
        return values_of(sharpen_image(image, {.kernel = SharpenKernel::Laplacian4, .amount = 1.0f, .threshold = threshold}, true, nullptr));
    };
    const auto thresholded = sharpen(0.05f);
    const auto unthresholded = sharpen(0.0f);
    for (int y = 0; y < H; ++y) {
        EXPECT_EQ(thresholded[at(10, y, 0)], original[at(10, y, 0)]);
        EXPECT_NE(unthresholded[at(10, y, 0)], original[at(10, y, 0)]);
        EXPECT_NE(thresholded[at(29, y, 0)], original[at(29, y, 0)]);
        EXPECT_NE(thresholded[at(30, y, 0)], original[at(30, y, 0)]);
    }
}

TEST(ImageSharpenTest, ByteAndFloatImagesAgreeAndFloatStaysOnThe16BitGrid) {
    const auto color = [](const int x, const int y, const int c) { return static_cast<float>((x * 29 + y * 53 + c * 71 + x * y * 3) % 256); };
    const auto bytes = make_image(W, H, color, DataType::UInt8);
    const auto floats = make_image(W, H, color, DataType::Float32);
    for (const auto kernel : kAllKernels) {
        SCOPED_TRACE(sharpen_kernel_name(kernel));
        const SharpenParams params{.kernel = kernel, .amount = 1.0f, .radius = 2.0f};
        const auto from_bytes = sharpen_image(bytes, params, true, nullptr).cpu().to_vector_uint8();
        const auto from_floats = sharpen_image(floats, params, true, nullptr).cpu().to_vector();
        float difference = 0.0f;
        size_t off_grid = 0;
        for (size_t i = 0; i < from_floats.size(); ++i) {
            difference = std::max(difference, std::abs(from_bytes[i] / 255.0f - from_floats[i]));
            // A 16-bit cache round trip must return the same float.
            const auto level = static_cast<uint16_t>(std::clamp(from_floats[i] * 65535.0f, 0.0f, 65535.0f) + 0.5f);
            off_grid += static_cast<float>(level) * (1.0f / 65535.0f) != from_floats[i];
        }
        EXPECT_LE(difference, 1.0f / 255.0f);
        EXPECT_EQ(off_grid, 0u);
    }
}

namespace {

    class ImageSharpenLoaderTest : public ::testing::Test {
    protected:
        static constexpr int kWidth = 96, kHeight = 64;

        static void SetUpTestSuite() {
            directory_ = std::filesystem::temp_directory_path() /
                         ("lfs_image_sharpen_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::filesystem::create_directories(directory_);
            std::vector<uint8_t> bytes(static_cast<size_t>(kWidth) * kHeight * 3);
            std::vector<uint16_t> words(bytes.size());
            for (int y = 0; y < kHeight; ++y)
                for (int x = 0; x < kWidth; ++x)
                    for (int c = 0; c < 3; ++c) {
                        const double level = 128.0 + 70.0 * std::sin(0.35 * x + c) * std::cos(0.25 * y) + (x < kWidth / 2 ? -30.0 : 30.0);
                        const size_t i = (static_cast<size_t>(y) * kWidth + x) * 3 + c;
                        bytes[i] = static_cast<uint8_t>(std::clamp(level, 0.0, 255.0));
                        words[i] = static_cast<uint16_t>(std::clamp(level * 257.0, 0.0, 65535.0));
                    }
            ASSERT_TRUE(save_png(png8(), bytes.data(), kWidth, kHeight, 3, 8, 1));
            ASSERT_TRUE(save_png(png16(), words.data(), kWidth, kHeight, 3, 16, 1));
            save_image_u8(jpeg(), Tensor::from_blob(bytes.data(), {static_cast<size_t>(kHeight), static_cast<size_t>(kWidth), 3}, Device::CPU, DataType::UInt8), 95);
            ASSERT_TRUE(std::filesystem::is_regular_file(jpeg()));
        }

        static void TearDownTestSuite() {
            std::error_code ec;
            std::filesystem::remove_all(directory_, ec);
        }

        static std::filesystem::path png8() { return directory_ / "pattern8.png"; }
        static std::filesystem::path png16() { return directory_ / "pattern16.png"; }
        static std::filesystem::path jpeg() { return directory_ / "pattern.jpg"; }

        static PipelinedLoaderConfig config(const bool sixteen_bit = false) {
            PipelinedLoaderConfig result;
            result.jpeg_batch_size = 2;
            result.prefetch_count = 4;
            result.output_queue_size = 4;
            result.decoder_pool_size = 2;
            result.io_threads = 1;
            result.cold_process_threads = 1;
            result.max_cache_bytes = 64 * 1024 * 1024;
            result.use_16bit_color = sixteen_bit;
            return result;
        }

        static ReadyImage load(PipelinedImageLoader& loader, const size_t sequence_id,
                               const std::filesystem::path& path, const LoadParams& params) {
            ImageRequest request;
            request.sequence_id = sequence_id;
            request.path = path;
            request.params = params;
            loader.prefetch({request});
            auto ready = loader.get();
            EXPECT_TRUE(ready.error.empty()) << ready.error;
            return ready;
        }

        static std::filesystem::path directory_;
    };

    std::filesystem::path ImageSharpenLoaderTest::directory_;

} // namespace

// The cold load is the kernel applied to the unsharpened load, and the cached copy returns it again.
TEST_F(ImageSharpenLoaderTest, ColdLoadIsTheKernelAppliedToThePlainLoadAndCacheHitsMatchIt) {
    struct Case {
        std::filesystem::path path;
        bool sixteen_bit;
    };
    for (const auto& [path, sixteen_bit] : {Case{png8(), false}, Case{jpeg(), false}, Case{png16(), true}}) {
        SCOPED_TRACE(path.filename().string());
        PipelinedImageLoader loader(config(sixteen_bit));
        LoadParams plain;
        plain.output_uint8 = !sixteen_bit;
        LoadParams sharpened = plain;
        sharpened.sharpen = {.amount = 1.5f, .radius = 1.2f, .threshold = 0.01f};

        const auto expected = sharpen_image(loader.load_image_immediate(path, plain), sharpened.sharpen, true, nullptr);
        const auto cold = load(loader, 0, path, sharpened);
        const auto hit = load(loader, 1, path, sharpened);
        const auto stats = loader.get_stats();
        EXPECT_EQ(stats.cold_path_misses, 1u);
        EXPECT_EQ(stats.hot_path_hits, 1u);
        expect_identical(cold.tensor, expected);

        if (sixteen_bit) {
            expect_identical(hit.tensor, cold.tensor);
            continue;
        }
        const auto cold_values = values_of(cold.tensor);
        const auto hit_values = values_of(hit.tensor);
        ASSERT_EQ(cold_values.size(), hit_values.size());
        double squared = 0.0;
        for (size_t i = 0; i < cold_values.size(); ++i)
            squared += (cold_values[i] - hit_values[i]) * (cold_values[i] - hit_values[i]);
        const double mse = squared / static_cast<double>(cold_values.size());
        // JPEG q95 of these sharpened patterns measures about 43 dB.
        EXPECT_GT(10.0 * std::log10(255.0 * 255.0 / std::max(mse, 1e-12)), 40.0);
    }
}

// Off keeps the key and the raw JPEG shortcut; each setting that changes the result gets its own entry.
TEST_F(ImageSharpenLoaderTest, SharpenSettingsKeyTheCacheOnlyWhenOn) {
    PipelinedImageLoader loader(config());
    LoadParams params;
    params.output_uint8 = true;
    size_t sequence = 0;
    const auto expect_load = [&](const SharpenParams& sharpen, const size_t cold, const size_t hot) {
        params.sharpen = sharpen;
        load(loader, sequence++, jpeg(), params);
        const auto stats = loader.get_stats();
        EXPECT_EQ(stats.cold_path_misses, cold) << sequence;
        EXPECT_EQ(stats.hot_path_hits, hot) << sequence;
    };
    expect_load({}, 0, 1);
    expect_load({.kernel = SharpenKernel::Laplacian8, .amount = 0.0f, .radius = 3.0f, .threshold = 0.5f}, 0, 2);
    EXPECT_EQ(loader.get_stats().cpu_decode_calls, 0u);

    expect_load({.amount = 1.0f}, 1, 2);
    expect_load({.amount = 1.0f}, 1, 3);
    expect_load({.amount = 2.0f}, 2, 3);
    expect_load({.amount = 1.0f, .radius = 2.0f}, 3, 3);
    expect_load({.amount = 1.0f, .threshold = 0.1f}, 4, 3);
    expect_load({.kernel = SharpenKernel::Laplacian4, .amount = 1.0f}, 5, 3);
    expect_load({.kernel = SharpenKernel::Laplacian4, .amount = 1.0f, .radius = 5.0f}, 5, 4);
    EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 6u);
}

TEST(ImageSharpenParamsTest, LoadingParamsRoundTripAndOlderSettingsLoadWithSharpeningOff) {
    param::LoadingParams params;
    params.sharpen = {.kernel = SharpenKernel::Laplacian8, .amount = 1.25f, .radius = 2.5f, .threshold = 0.02f};
    EXPECT_EQ(param::LoadingParams::from_json(params.to_json()).sharpen, params.sharpen);

    auto older = params.to_json();
    for (const auto* key : {"sharpen_kernel", "sharpen_amount", "sharpen_radius", "sharpen_threshold"})
        older.erase(key);
    EXPECT_EQ(param::LoadingParams::from_json(older).sharpen, SharpenParams{});

    // Stored settings with an unknown kernel load with the default; a user config is rejected.
    auto unknown = params.to_json();
    unknown["sharpen_kernel"] = "box";
    EXPECT_EQ(param::LoadingParams::from_json(unknown).sharpen.kernel, SharpenKernel::Usm);
    const auto config_path = std::filesystem::temp_directory_path() / "lfs_image_sharpen_unknown_kernel.json";
    std::ofstream(config_path) << nlohmann::json{{"dataset", {{"loading_params", unknown}}}}.dump();
    const auto imported = param::read_training_parameters_from_json(config_path);
    std::error_code ec;
    std::filesystem::remove(config_path, ec);
    ASSERT_FALSE(imported.has_value());
    EXPECT_NE(imported.error().detail().find("sharpen_kernel"), std::string::npos);
}
