/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Saving a GPU image quantizes it on its device. The bytes, and so the files, must be the host path's on every
// backend; this test runs on whichever GPU backend the test binary selects.

#include "core/image_io.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

    std::filesystem::path unique_temp_path(const std::string_view stem) {
        static std::atomic_uint64_t sequence{0};
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() /
               ("lfs_image_save_device_" + std::string(stem) + "_" + std::to_string(timestamp) + "_" +
                std::to_string(sequence.fetch_add(1)) + ".png");
    }

    std::vector<char> read_file(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }

} // namespace

// Every float that clamp(0, 1) * 255 + 0.5 turns into a byte must give the byte the host path gives: at each of the
// 257 boundaries between two bytes (where a multiply and add fused into one rounding could tip a value over), at the
// ends of the range and past them, and for NaN and the infinities. The PNG files the two paths write must be the
// same bytes, for [C,H,W] planes and for an [H,W,C] image.
TEST(ImageSaveDevice, GpuImageQuantizesLikeTheHostAtEveryByteBoundary) {
    std::vector<float> values;
    for (int k = 0; k <= 256; ++k) {
        float x = (static_cast<float>(k) - 0.5f) / 255.0f;
        for (int step = 0; step < 4; ++step)
            x = std::nextafter(x, -1.0f);
        for (int step = 0; step < 9; ++step, x = std::nextafter(x, 2.0f))
            values.push_back(x);
    }
    for (const float v : {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 1e-40f, -1e-40f, 0.999999f, 1.000001f,
                          std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::lowest()})
        values.push_back(v);
    for (std::size_t i = 0; values.size() < 3 * 64 * 61; ++i)
        values.push_back(static_cast<float>((i * 2654435761u) % 100003u) / 90000.0f - 0.05f);
    const std::size_t width = 61, height = values.size() / 3 / width + 1;
    values.resize(3 * width * height, 0.5f);

    const auto save_both = [&](const lfs::core::TensorShape& shape, const std::string_view what) {
        const auto host_path = unique_temp_path(std::string(what) + "_host");
        const auto device_path = unique_temp_path(std::string(what) + "_device");
        lfs::core::save_image(host_path, lfs::core::Tensor::from_vector(values, shape, lfs::core::Device::CPU));
        lfs::core::save_image(device_path, lfs::core::Tensor::from_vector(values, shape, lfs::core::Device::GPU));
        const auto host = read_file(host_path);
        const auto device = read_file(device_path);
        std::error_code ec;
        std::filesystem::remove(host_path, ec);
        std::filesystem::remove(device_path, ec);
        ASSERT_FALSE(host.empty()) << what;
        EXPECT_EQ(device.size(), host.size()) << what;
        EXPECT_TRUE(device == host) << what << ": the PNG files differ";
    };
    save_both(lfs::core::TensorShape{3, height, width}, "planes");
    save_both(lfs::core::TensorShape{height, width, 3}, "interleaved");

    // The bytes themselves, against the host path's arithmetic written out.
    const auto expected_byte = [](const float v) {
        const float clamped = std::isnan(v) ? 0.0f : std::clamp(v, 0.0f, 1.0f);
        volatile float scaled = clamped * 255.0f;
        return static_cast<std::uint8_t>(scaled + 0.5f);
    };
    const auto path = unique_temp_path("bytes");
    lfs::core::save_image(path, lfs::core::Tensor::from_vector(values, lfs::core::TensorShape{3, height, width},
                                                               lfs::core::Device::GPU));
    auto [decoded, w, h, channels] = lfs::core::load_image(path);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    ASSERT_NE(decoded, nullptr);
    ASSERT_EQ(channels, 3);
    ASSERT_EQ(static_cast<std::size_t>(w), width);
    ASSERT_EQ(static_cast<std::size_t>(h), height);
    const std::size_t plane = width * height;
    std::size_t mismatches = 0;
    for (std::size_t pixel = 0; pixel < plane; ++pixel)
        for (std::size_t c = 0; c < 3; ++c)
            mismatches += decoded[pixel * 3 + c] != expected_byte(values[c * plane + pixel]);
    lfs::core::free_image(decoded);
    EXPECT_EQ(mismatches, 0u);
}
