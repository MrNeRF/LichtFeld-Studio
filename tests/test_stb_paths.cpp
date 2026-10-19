/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/native_fs.hpp"
#include "core/stb_file.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <memory>

TEST(StbPaths, FileReadersDecodeBeyondTheWindowsFilenameBuffer) {
    namespace fs = lfs::core::native_fs;
    namespace stb = lfs::core::stb_file;
#ifdef LFS_REQUIRE_LONG_PATHS_DISABLED
    ASSERT_EQ(fs::process_long_paths_enabled(), std::optional<bool>(false));
#endif
    const auto root = fs::temp_directory_path() /
                      ("lfs-stb-path-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        fs::path root;
        ~Cleanup() {
            std::error_code ignored;
            fs::remove_all(root, ignored);
        }
    } cleanup{root};
    auto directory = root;
    while (directory.native().size() < 1200)
        directory /= fs::path(u8"component_\u6e2c\u8a66_01234567890123456789");
    fs::create_directories(directory);
    const auto rgb = directory / "rgb.ppm";
    {
        std::ofstream output(lfs::core::native_fs::native_path(rgb), std::ios::binary);
        output << "P6\n1 1\n255\n";
        output.put(char(255));
        output.put(0);
        output.put(0);
        ASSERT_TRUE(output.good());
    }
    int w = 0, h = 0, channels = 0;
    ASSERT_TRUE(stb::stb_info(rgb, &w, &h, &channels));
    EXPECT_EQ(w, 1);
    EXPECT_EQ(h, 1);
    EXPECT_EQ(channels, 3);
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(stb::stb_load(rgb, &w, &h, &channels, 3), stbi_image_free);
    ASSERT_NE(pixels, nullptr);
    EXPECT_EQ(pixels.get()[0], 255);
    EXPECT_EQ(pixels.get()[1], 0);
    EXPECT_EQ(pixels.get()[2], 0);
    EXPECT_FALSE(stb::stb_is_16_bit(rgb));
    const auto depth = directory / "depth.pgm";
    {
        std::ofstream output(lfs::core::native_fs::native_path(depth), std::ios::binary);
        output << "P5\n1 1\n65535\n";
        output.put(char(255));
        output.put(char(255));
        ASSERT_TRUE(output.good());
    }
    EXPECT_TRUE(stb::stb_is_16_bit(depth));
    std::unique_ptr<stbi_us, decltype(&stbi_image_free)> samples(stb::stb_load_16(depth, &w, &h, &channels, 1), stbi_image_free);
    ASSERT_NE(samples, nullptr);
    EXPECT_EQ(samples.get()[0], 65535);
}
