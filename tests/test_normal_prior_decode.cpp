/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "io/pipelined_image_loader.hpp"
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <stb_image.h>
#include <tuple>
#include <vector>

#include "core/image_io.hpp"

namespace {

    float srgb_from_linear(const float v) {
        if (v <= 0.0031308f) {
            return v * 12.92f;
        }
        return 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    }

} // namespace

TEST(NormalPriorDecodeTest, SrgbEncodingToLinearRoundTrips) {
    for (const float v : {0.0f, 0.01f, 0.25f, 0.5f, 0.7354f, 0.9f, 1.0f}) {
        EXPECT_NEAR(lfs::core::srgb_encoding_to_linear(srgb_from_linear(v)), v, 1e-5f);
    }
    // The value that misled the linear decode: sRGB(0.5) encodes a zero
    // normal component but decodes linearly to 0.459.
    EXPECT_NEAR(lfs::core::srgb_encoding_to_linear(0.7354f), 0.5f, 1e-3f);
}

TEST(NormalPriorDecodeTest, SrgbNormalPriorToLinearChwFixesUnitNorm) {
    // Axis-aligned unit normals written through the sRGB display transform,
    // decoded with the (wrong) linear assumption first.
    const std::array<std::array<float, 3>, 3> normals = {{{0.f, 0.f, 1.f},
                                                          {0.f, 1.f, 0.f},
                                                          {-1.f, 0.f, 0.f}}};
    const size_t pixel_count = normals.size();
    std::vector<float> chw(pixel_count * 3);
    for (size_t i = 0; i < pixel_count; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            const float encoded = srgb_from_linear(normals[i][c] * 0.5f + 0.5f);
            chw[c * pixel_count + i] = encoded * 2.0f - 1.0f; // linear decode of sRGB data
        }
    }

    for (size_t i = 0; i < pixel_count; ++i) {
        const float x = chw[i];
        const float y = chw[pixel_count + i];
        const float z = chw[2 * pixel_count + i];
        EXPECT_GT(std::abs(std::sqrt(x * x + y * y + z * z) - 1.0f), 0.15f)
            << "linear decode of sRGB data should be visibly non-unit";
    }

    lfs::core::srgb_normal_prior_to_linear_chw(chw.data(), chw.size());

    for (size_t i = 0; i < pixel_count; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(chw[c * pixel_count + i], normals[i][c], 2e-3f);
        }
    }
}

namespace {
    constexpr int normal_width = 64;
    constexpr int normal_height = 64;
    constexpr size_t normal_pixels = normal_width * normal_height;

    enum class NormalEncoding { Png8,
                                Png16,
                                Jpeg8,
                                Tiff8,
                                Tiff16,
                                TiffFloat };

    class PipelinedNormalDecode : public ::testing::TestWithParam<std::tuple<NormalEncoding, bool>> {
    protected:
        void SetUp() override {
            directory_ = std::filesystem::temp_directory_path() /
                         ("lfs_normal_decode_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::filesystem::create_directories(directory_);
        }
        void TearDown() override {
            std::error_code ec;
            std::filesystem::remove_all(directory_, ec);
        }

        // Independent uncompressed RGB TIFF with native sample precision.
        void write_tiff(const std::filesystem::path& path, int bits, bool floating,
                        const std::vector<uint32_t>& samples) {
            std::ofstream file(path, std::ios::binary);
            const auto put = [&](uint32_t value, int bytes) {
                for (int i = 0; i < bytes; ++i) {
                    file.put(static_cast<char>(value & 255));
                    value >>= 8;
                }
            };
            file.write("II", 2);
            put(42, 2);
            put(8, 4);
            put(10, 2);
            const auto tag = [&](uint16_t id, uint16_t type, uint32_t count, uint32_t value) {
                put(id, 2);
                put(type, 2);
                put(count, 4);
                put(value, 4);
            };
            tag(256, 4, 1, normal_width);
            tag(257, 4, 1, normal_height);
            tag(258, 3, 3, 134); // RGB bits per sample
            tag(259, 3, 1, 1);
            tag(262, 3, 1, 2);
            tag(273, 4, 1, 146);
            tag(277, 3, 1, 3);
            tag(278, 4, 1, normal_height);
            tag(279, 4, 1, samples.size() * (bits / 8));
            tag(339, 3, 3, 140); // sample formats
            put(0, 4);
            for (int i = 0; i < 3; ++i)
                put(bits, 2);
            for (int i = 0; i < 3; ++i)
                put(floating ? 3 : 1, 2);
            for (auto value : samples)
                put(value, bits / 8);
            ASSERT_TRUE(file.good());
        }
        std::filesystem::path directory_;
    };
} // namespace

TEST_P(PipelinedNormalDecode, PreservesSamplesTransformsAndCache) {
    using namespace lfs::core;
    const auto [encoding, transform] = GetParam();
    const bool png = encoding == NormalEncoding::Png8 || encoding == NormalEncoding::Png16;
    const bool jpeg = encoding == NormalEncoding::Jpeg8;
    const bool floating = encoding == NormalEncoding::TiffFloat;
    const int bits = floating ? 32 : (encoding == NormalEncoding::Png16 || encoding == NormalEncoding::Tiff16 ? 16 : 8);
    const auto rgb_path = directory_ / "image.png";
    const auto normal_path = directory_ / (png ? "normal.PNG" : jpeg     ? "normal.jpg"
                                                            : bits == 16 ? "normal.tif"
                                                                         : "normal.TIFF");
    const std::array<uint16_t, 8> pattern{0, 1, 257, 12345, 32768, 48001, 65534, 65535};
    std::vector<uint8_t> u8;
    std::vector<uint16_t> u16;
    std::vector<uint32_t> samples;
    std::vector<float> encoded;
    for (size_t i = 0; i < normal_pixels * 3; ++i) {
        const auto value = pattern[i % pattern.size()];
        u8.push_back(static_cast<uint8_t>(value >> 8));
        u16.push_back(value);
        const float v = floating ? (static_cast<float>(i % 96) + 0.125f) / 97.0f : bits == 16 ? value / 65535.0f
                                                                                              : u8.back() / 255.0f;
        encoded.push_back(v);
        samples.push_back(floating ? std::bit_cast<uint32_t>(v) : bits == 16 ? value
                                                                             : u8.back());
    }
    ASSERT_TRUE(save_png(rgb_path, u8.data(), normal_width, normal_height, 3, 8, 1));
    if (png) {
        ASSERT_TRUE(save_png(normal_path, bits == 16 ? static_cast<void*>(u16.data()) : static_cast<void*>(u8.data()), normal_width, normal_height, 3, bits, 1));
    } else if (jpeg) {
        ASSERT_TRUE(save_img_data(normal_path, {u8.data(), normal_width, normal_height, 3}));
        int w, h, c;
        auto* decoded = stbi_load(normal_path.string().c_str(), &w, &h, &c, 3);
        ASSERT_NE(decoded, nullptr);
        for (size_t i = 0; i < encoded.size(); ++i)
            encoded[i] = decoded[i] / 255.0f;
        stbi_image_free(decoded);
    } else {
        write_tiff(normal_path, bits, floating, samples);
    }
    std::vector<float> expected(normal_pixels * 3);
    const std::array<float, 9> rotation{0, -1, 0, 1, 0, 0, 0, 0, 1};
    for (size_t i = 0; i < normal_pixels; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            const float v = encoded[i * 3 + c];
            expected[c * normal_pixels + i] = (transform ? srgb_encoding_to_linear(v) : v) * 2.0f - 1.0f;
        }
    }
    if (transform) {
        flip_normal_prior_yz_chw(expected.data(), normal_pixels);
        transform_normal_prior_world_to_camera_chw(expected.data(), normal_pixels, rotation);
    }
    lfs::io::PipelinedLoaderConfig config;
    config.io_threads = 1;
    config.cold_process_threads = 1;
    config.decoder_pool_size = 1;
    config.output_queue_size = 2;
    config.prefetch_count = 2;
    lfs::io::PipelinedImageLoader loader(config);
    lfs::io::ImageRequest request;
    request.path = rgb_path;
    request.normal_path = normal_path;
    request.sequence_id = 1;
    request.normal_srgb = transform;
    request.normal_flip_yz = transform;
    request.normal_transform_world_to_camera = transform;
    request.normal_world_to_camera = rotation;
    request.aux_target_width = transform ? normal_width / 2 : normal_width;
    request.aux_target_height = transform ? normal_height / 2 : normal_height;
    const auto reference = resize_normal_prior(Tensor::from_vector(expected, {3, normal_height, normal_width}, Device::CUDA),
                                               request.aux_target_height, request.aux_target_width)
                               .cpu()
                               .to_vector();
    for (int pass = 0; pass < 2; ++pass) {
        SCOPED_TRACE(pass);
        loader.prefetch({request});
        auto ready = loader.try_get_for(std::chrono::seconds(20));
        ASSERT_TRUE(ready.has_value());
        ASSERT_TRUE(ready->normal.has_value());
        if (ready->normal_ready_event) {
            ASSERT_EQ(cudaEventSynchronize(ready->normal_ready_event), cudaSuccess);
            ASSERT_EQ(cudaEventDestroy(ready->normal_ready_event), cudaSuccess);
            ready->normal_ready_event = nullptr;
        }
        const auto values = ready->normal->cpu().to_vector();
        ASSERT_EQ(values.size(), reference.size());
        for (size_t i = 0; i < values.size(); ++i)
            ASSERT_NEAR(values[i], reference[i], pass == 0 ? 2e-6f : 4e-5f) << "sample " << i;
        // A second successful read must come from the cache.
        std::filesystem::remove(normal_path);
        ++request.sequence_id;
    }
    EXPECT_GT(loader.get_stats().hot_path_hits, 0u);
    EXPECT_EQ(loader.get_stats().aggregate_sidecar_tally.normal.failed, 0u);
}

INSTANTIATE_TEST_SUITE_P(Formats, PipelinedNormalDecode,
                         ::testing::Combine(::testing::Values(NormalEncoding::Png8, NormalEncoding::Png16, NormalEncoding::Jpeg8,
                                                              NormalEncoding::Tiff8, NormalEncoding::Tiff16, NormalEncoding::TiffFloat),
                                            ::testing::Bool()));
