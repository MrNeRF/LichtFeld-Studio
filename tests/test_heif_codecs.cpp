/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_codecs.hpp"
#include "core/image_heif.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
namespace {
    namespace codec = lfs::core::image_codecs;
    const std::filesystem::path fixtures = LFS_HEIF_FIXTURES;
    TEST(ProductionHeif, PrimaryRgbAlphaRotationAndPrecisionMatchIndependentDecode) {
        for (const auto name : {"rgb8", "rgba8", "rotated", "mirrored", "rgb10", "rgb12", "pq"}) {
            SCOPED_TRACE(name);
            codec::Probe info;
            codec::Image image;
            std::string error;
            const auto path = fixtures / (std::string(name) + ".heic");
            ASSERT_TRUE(codec::probe(path, info, error)) << error;
            ASSERT_TRUE(codec::decode(path, image, error)) << error;
            EXPECT_EQ(image.width, info.width);
            EXPECT_EQ(image.height, info.height);
            EXPECT_EQ(image.channels, std::string_view(name) == "rgba8" ? 4 : 3);
            EXPECT_EQ(image.width, std::string_view(name) == "rotated" ? 24 : std::string_view(name) == "rgb8" ? 128
                                                                                                               : 32);
            EXPECT_EQ(image.height, std::string_view(name) == "rotated" ? 32 : std::string_view(name) == "rgb8" ? 96
                                                                                                                : 24);
            std::ifstream file(fixtures / (std::string(name) + ".reference"), std::ios::binary);
            std::vector<uint8_t> expected((std::istreambuf_iterator<char>(file)), {});
            ASSERT_EQ(expected.size(), image.data.size());
            if (std::string_view(name) == "rgb12" || std::string_view(name) == "rgb10" || std::string_view(name) == "pq") {
                ASSERT_EQ(image.sample_type, codec::SampleType::UInt16);
                bool precision = false;
                for (size_t i = 0; i < expected.size(); i += 2) {
                    const int bits = std::string_view(name) == "rgb10" ? 10 : 12;
                    const auto sample = (uint16_t(expected[i]) | (uint16_t(expected[i + 1]) << 8)) >> (16 - bits);
                    const uint32_t maximum = (1u << bits) - 1;
                    const auto reference = (uint32_t(sample) * 65535 + maximum / 2) / maximum;
                    uint16_t actual;
                    std::memcpy(&actual, image.data.data() + i, 2);
                    ASSERT_LE(std::abs(int(actual) - int(reference)), 1) << i;
                    precision |= actual % 257 != 0;
                }
                EXPECT_TRUE(precision);
            } else {
                for (size_t i = 0; i < expected.size(); ++i)
                    ASSERT_LE(std::abs(int(expected[i]) - image.data[i]), 1) << i;
            }
        }
    }
    TEST(ProductionHeif, DisplayP3MatchesIndependentMatrixAndHdrIsNotSilentlySdr) {
        codec::Image raw, display;
        std::string error;
        ASSERT_TRUE(codec::decode_heif(fixtures / "p3.heic", raw, error)) << error;
        ASSERT_TRUE(codec::decode_heif(fixtures / "p3.heic", display, error, true)) << error;
        EXPECT_NE(raw.data, display.data);
        const auto linear = [](double v) { return v <= .04045 ? v / 12.92 : std::pow((v + .055) / 1.055, 2.4); };
        const auto encoded = [](double v) {v=std::clamp(v,0.,1.);return v<=.0031308?v*12.92:1.055*std::pow(v,1./2.4)-.055; };
        for (size_t i = 0; i < raw.data.size(); i += 3) {
            const double r = linear(raw.data[i] / 255.), g = linear(raw.data[i + 1] / 255.), b = linear(raw.data[i + 2] / 255.);
            const std::array<double, 3> reference{encoded(1.224940176 * r - .224940176 * g), encoded(-.042056955 * r + 1.042056955 * g), encoded(-.019637555 * r - .078636046 * g + 1.098273601 * b)};
            for (size_t c = 0; c < 3; ++c)
                EXPECT_NEAR(display.data[i + c], std::lround(reference[c] * 255), 2) << i << ":" << c;
        }
        ASSERT_TRUE(codec::decode_heif(fixtures / "pq.heic", raw, error)) << error;
        EXPECT_FALSE(codec::decode_heif(fixtures / "pq.heic", display, error, true));
        EXPECT_NE(error.find("SDR"), std::string::npos);
    }
    TEST(ProductionHeif, IccProfilesAndPrimaryItemAreExplicit) {
        codec::Image image, raw;
        std::string error;
        ASSERT_TRUE(codec::decode_heif(fixtures / "icc.heic", image, error, true)) << error;
        ASSERT_TRUE(codec::decode_heif(fixtures / "icc.heic", raw, error)) << error;
        ASSERT_EQ(image.data.size(), raw.data.size());
        for (size_t i = 0; i < image.data.size(); ++i)
            EXPECT_NEAR(image.data[i], raw.data[i], 1);
        ASSERT_TRUE(codec::decode_heif(fixtures / "badicc.heic", raw, error)) << error;
        EXPECT_FALSE(codec::decode_heif(fixtures / "badicc.heic", image, error, true));
        EXPECT_NE(error.find("profile"), std::string::npos);
        codec::Probe info;
        ASSERT_TRUE(codec::probe(fixtures / "multi.heic", info, error)) << error;
        EXPECT_EQ(info.width, 16);
        EXPECT_EQ(info.height, 16);
        ASSERT_TRUE(codec::decode(fixtures / "multi.heic", image, error)) << error;
        EXPECT_EQ(image.width, 16);
        EXPECT_EQ(image.height, 16);
        ASSERT_GE(image.data.size(), 3);
        EXPECT_NEAR(image.data[0], 200, 1);
        EXPECT_NEAR(image.data[1], 80, 1);
        EXPECT_NEAR(image.data[2], 30, 1);
    }
    TEST(ProductionHeif, PremultipliedDisplayAlphaAndConcurrentReads) {
        codec::Image raw, display;
        std::string error;
        ASSERT_TRUE(codec::decode_heif(fixtures / "premultiplied.heic", raw, error)) << error;
        ASSERT_TRUE(codec::decode_heif(fixtures / "premultiplied.heic", display, error, true)) << error;
        ASSERT_EQ(raw.channels, 4);
        ASSERT_EQ(raw.data.size(), display.data.size());
        for (size_t i = 0; i < raw.data.size(); i += 4) {
            const int alpha = raw.data[i + 3];
            EXPECT_EQ(display.data[i + 3], alpha);
            for (size_t c = 0; c < 3; ++c) {
                const int expected = alpha ? std::min(255, (int(raw.data[i + c]) * 255 + alpha / 2) / alpha) : 0;
                EXPECT_NEAR(display.data[i + c], expected, 1);
            }
        }
        std::vector<std::future<bool>> reads;
        for (int i = 0; i < 8; ++i)
            reads.push_back(std::async(std::launch::async, [] {
                codec::Image image;
                std::string diagnostic;
                return codec::decode_heif(fixtures / "p3.heic", image, diagnostic, true) && image.width == 32 && image.height == 24;
            }));
        for (auto& read : reads)
            EXPECT_TRUE(read.get());
    }
    TEST(ProductionHeif, ExtensionAliasesUnicodeAndTruncation) {
        const auto root = std::filesystem::temp_directory_path() / ("lfs-heif-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(root);
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() {
                std::error_code e;
                std::filesystem::remove_all(path, e);
            }
        } cleanup{root};
        const auto path = root / std::filesystem::path(u8"é-日本語.HEIF");
        std::filesystem::copy_file(fixtures / "rgb8.heic", path);
        codec::Image image;
        std::string error;
        ASSERT_TRUE(codec::decode(path, image, error)) << error;
        std::filesystem::resize_file(path, 32);
        codec::Probe info;
        EXPECT_FALSE(codec::probe(path, info, error));
        EXPECT_FALSE(error.empty());
        EXPECT_FALSE(codec::decode(path, image, error));
        EXPECT_FALSE(error.empty());
        EXPECT_TRUE(image.data.empty());
    }
} // namespace
