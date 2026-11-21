/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_io.hpp"
#include "core/tensor.hpp"
#include "io/masked_image_fill.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::io::MaskedImageFill;

    // fill_masked_image's rule, written as directly as possible.
    bool keeps(float value, const MaskedImageFill& policy) {
        if (policy.invert)
            value = 1.0f - value;
        if (policy.keep_high_band)
            return value > 250.5f / 255.0f;
        if (policy.threshold > 0.0f)
            return value >= policy.threshold;
        return value > 0.0f;
    }

    float normalized(const uint8_t value) { return static_cast<float>(value) / 255.0f; }
    float normalized(const uint16_t value) { return static_cast<float>(value) / 65535.0f; }
    float normalized(const float value) { return value; }

    // Every ignored pixel scans its whole clipped window for the kept pixel of least squared
    // distance, then least linear index; else the first kept pixel; else zero.
    template <typename T>
    std::vector<T> brute_force(std::vector<T> pixels, const int width, const int height, const int channels,
                               const bool planar, const std::vector<uint8_t>& keep) {
        const size_t count = size_t(width) * height;
        const auto offset = [&](const size_t pixel, const int c) {
            return planar ? c * count + pixel : pixel * channels + c;
        };
        const auto source = pixels;
        std::optional<size_t> first;
        for (size_t i = 0; i < count && !first; ++i)
            if (keep[i])
                first = i;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const size_t pixel = size_t(y) * width + x;
                if (keep[pixel])
                    continue;
                std::optional<size_t> nearest;
                int best = 0;
                for (int qy = std::max(0, y - 16); qy <= std::min(height - 1, y + 16); ++qy)
                    for (int qx = std::max(0, x - 16); qx <= std::min(width - 1, x + 16); ++qx) {
                        const size_t candidate = size_t(qy) * width + qx;
                        const int distance = (qx - x) * (qx - x) + (qy - y) * (qy - y);
                        if (keep[candidate] && (!nearest || distance < best || (distance == best && candidate < *nearest))) {
                            nearest = candidate;
                            best = distance;
                        }
                    }
                if (!nearest)
                    nearest = first;
                for (int c = 0; c < 3; ++c)
                    pixels[offset(pixel, c)] = nearest ? source[offset(*nearest, c)] : T{};
            }
        return pixels;
    }

    template <typename T>
    T random_sample(std::mt19937& random) {
        if constexpr (std::is_same_v<T, float>)
            return std::uniform_real_distribution<float>(-0.25f, 1.25f)(random);
        else
            return static_cast<T>(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<T>::max())(random));
    }

    template <typename T>
    std::vector<T> random_pixels(const size_t count, std::mt19937& random) {
        std::vector<T> pixels(count);
        for (auto& value : pixels)
            value = random_sample<T>(random);
        return pixels;
    }

    template <typename T>
    void expect_same_samples(const std::vector<T>& actual, const std::vector<T>& expected) {
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i)
            ASSERT_EQ(std::memcmp(&actual[i], &expected[i], sizeof(T)), 0) << "sample " << i;
    }

    Tensor host_bytes(const std::vector<uint8_t>& values, lfs::core::TensorShape shape) {
        auto tensor = Tensor::empty(std::move(shape), Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
        return tensor;
    }

    // A byte mask on the source grid, kept with the given probability, with in-between values.
    std::vector<uint8_t> random_mask(const size_t count, const double kept, std::mt19937& random) {
        std::vector<uint8_t> mask(count);
        std::bernoulli_distribution keep(kept);
        std::uniform_int_distribution<int> soft(0, 255);
        for (auto& value : mask)
            value = static_cast<uint8_t>(keep(random) ? std::max(soft(random), 128) : std::min(soft(random), 127));
        return mask;
    }

    template <typename T>
    void check_mask_fill(const int width, const int height, const int channels, const bool planar,
                         const double kept, const uint32_t seed) {
        SCOPED_TRACE(::testing::Message() << width << "x" << height << " channels=" << channels << " planar=" << planar
                                          << " kept=" << kept << " sample bytes=" << sizeof(T));
        std::mt19937 random(seed);
        const size_t count = size_t(width) * height;
        const auto mask = random_mask(count, kept, random);
        const auto pixels = random_pixels<T>(count * channels, random);
        const MaskedImageFill policy{.mask = host_bytes(mask, {size_t(height), size_t(width)}), .threshold = 0.5f};
        std::vector<uint8_t> keep(count);
        for (size_t i = 0; i < count; ++i)
            keep[i] = keeps(normalized(mask[i]), policy);
        auto actual = pixels;
        lfs::io::fill_masked_image_host(actual.data(), width, height, channels, planar, policy);
        expect_same_samples(actual, brute_force(pixels, width, height, channels, planar, keep));
    }

    TEST(MaskedImageFillHost, MatchesBruteForceAcrossSampleTypesLayoutsAndDensities) {
        uint32_t seed = 1;
        for (const auto [width, height] : {std::pair{37, 29}, std::pair{64, 48}, std::pair{5, 3}, std::pair{1, 1},
                                           std::pair{41, 1}, std::pair{1, 41}})
            for (const int channels : {3, 4})
                for (const bool planar : {false, true})
                    for (const double kept : {0.003, 0.02, 0.2, 0.6, 0.97}) {
                        check_mask_fill<uint8_t>(width, height, channels, planar, kept, ++seed);
                        check_mask_fill<uint16_t>(width, height, channels, planar, kept, ++seed);
                        check_mask_fill<float>(width, height, channels, planar, kept, ++seed);
                    }
    }

    // Sparse kept pixels on a lattice make many equally near candidates.
    TEST(MaskedImageFillHost, EquallyNearKeptPixelsResolveToTheLowerIndex) {
        constexpr int width = 47, height = 39;
        for (const int step : {2, 3, 4, 6, 10, 33}) {
            SCOPED_TRACE(step);
            std::vector<uint8_t> mask(size_t(width) * height, 0), keep(mask.size(), 0);
            for (int y = 1; y < height; y += step)
                for (int x = 2; x < width; x += step)
                    mask[size_t(y) * width + x] = keep[size_t(y) * width + x] = 255;
            std::vector<uint8_t> pixels(mask.size() * 3);
            for (size_t i = 0; i < pixels.size(); ++i)
                pixels[i] = static_cast<uint8_t>(i * 7 + i / 3);
            const MaskedImageFill policy{.mask = host_bytes(mask, {height, width})};
            auto actual = pixels;
            lfs::io::fill_masked_image_host(actual.data(), width, height, 3, false, policy);
            expect_same_samples(actual, brute_force(pixels, width, height, 3, false, keep));
        }
        // Left beats right and above beats below; among distance 5, row 3 comes before row 6.
        constexpr int size = 9;
        const auto pixel = [](const int x, const int y) { return size_t(y) * size + x; };
        const auto source_of = [&](const std::vector<std::pair<int, int>>& kept, const int x, const int y) {
            std::vector<uint8_t> mask(size * size, 0);
            for (const auto& [kx, ky] : kept)
                mask[pixel(kx, ky)] = 255;
            std::vector<uint8_t> image(size * size * 3);
            for (size_t i = 0; i < size * size; ++i)
                image[i * 3] = static_cast<uint8_t>(i);
            lfs::io::fill_masked_image_host(image.data(), size, size, 3, false, {.mask = host_bytes(mask, {size, size})});
            return image[pixel(x, y) * 3];
        };
        EXPECT_EQ(source_of({{2, 4}, {6, 4}}, 4, 4), pixel(2, 4));
        EXPECT_EQ(source_of({{6, 4}, {2, 4}}, 4, 4), pixel(2, 4));
        EXPECT_EQ(source_of({{4, 6}, {4, 2}}, 4, 4), pixel(4, 2));
        EXPECT_EQ(source_of({{3, 6}, {6, 3}}, 4, 4), pixel(6, 3));
        EXPECT_EQ(source_of({{5, 5}, {3, 5}, {5, 3}, {3, 3}}, 4, 4), pixel(3, 3));
        // A diagonal neighbour is nearer than a straight one two pixels away.
        EXPECT_EQ(source_of({{4, 2}, {5, 5}}, 4, 4), pixel(5, 5));
    }

    TEST(MaskedImageFillHost, BordersFullMasksAndUnreachablePixels) {
        constexpr int width = 80, height = 60;
        std::mt19937 random(7);
        const auto pixels = random_pixels<uint8_t>(size_t(width) * height * 4, random);
        const auto run = [&](const std::vector<uint8_t>& mask) {
            auto actual = pixels;
            std::vector<uint8_t> keep(mask.size());
            for (size_t i = 0; i < mask.size(); ++i)
                keep[i] = mask[i] != 0;
            lfs::io::fill_masked_image_host(actual.data(), width, height, 4, false,
                                            {.mask = host_bytes(mask, {height, width})});
            expect_same_samples(actual, brute_force(pixels, width, height, 4, false, keep));
            return actual;
        };
        // Masked borders around a kept centre; then a kept corner far from most pixels, whose
        // first kept pixel serves every pixel outside the window.
        std::vector<uint8_t> mask(size_t(width) * height, 0);
        for (int y = 20; y < 40; ++y)
            for (int x = 25; x < 55; ++x)
                mask[size_t(y) * width + x] = 255;
        run(mask);
        std::fill(mask.begin(), mask.end(), 0);
        for (int y = 50; y < height; ++y)
            for (int x = 70; x < width; ++x)
                mask[size_t(y) * width + x] = 255;
        const auto corner = run(mask);
        const size_t first = size_t(50) * width + 70;
        for (int c = 0; c < 3; ++c)
            EXPECT_EQ(corner[c], pixels[first * 4 + c]);
        // Nothing kept: RGB becomes zero and alpha is untouched.
        std::fill(mask.begin(), mask.end(), 0);
        const auto empty = run(mask);
        for (size_t i = 0; i < size_t(width) * height; ++i) {
            EXPECT_EQ(empty[i * 4] | empty[i * 4 + 1] | empty[i * 4 + 2], 0);
            EXPECT_EQ(empty[i * 4 + 3], pixels[i * 4 + 3]);
        }
        // Nothing ignored: unchanged.
        std::fill(mask.begin(), mask.end(), 1);
        EXPECT_EQ(run(mask), pixels);
    }

    template <typename T>
    void check_alpha_fill(const std::vector<T>& alpha_values, const MaskedImageFill& policy) {
        SCOPED_TRACE(::testing::Message() << "invert=" << policy.invert << " threshold=" << policy.threshold
                                          << " high_band=" << policy.keep_high_band << " sample bytes=" << sizeof(T));
        const int width = 23, height = static_cast<int>((alpha_values.size() + width - 1) / width);
        const size_t count = size_t(width) * height;
        std::mt19937 random(static_cast<uint32_t>(alpha_values.size()));
        auto pixels = random_pixels<T>(count * 4, random);
        std::vector<uint8_t> keep(count);
        for (size_t i = 0; i < count; ++i) {
            pixels[i * 4 + 3] = alpha_values[i % alpha_values.size()];
            keep[i] = keeps(normalized(pixels[i * 4 + 3]), policy);
        }
        auto actual = pixels;
        lfs::io::fill_masked_image_host(actual.data(), width, height, 4, false, policy);
        expect_same_samples(actual, brute_force(pixels, width, height, 4, false, keep));
    }

    TEST(MaskedImageFillHost, AlphaInvertThresholdsAndHighBandClassifyLikeTheKernel) {
        std::vector<uint8_t> bytes(256);
        for (int k = 0; k < 256; ++k)
            bytes[k] = static_cast<uint8_t>((k * 113) % 256);
        std::vector<uint16_t> words;
        for (const uint32_t center : {0u, 1u, 13107u, 32767u, 32768u, 64224u, 64381u, 65535u})
            for (int delta = -3; delta <= 3; ++delta)
                words.push_back(static_cast<uint16_t>(std::clamp<int64_t>(int64_t{center} + delta, 0, 65535)));
        const float subnormal = std::numeric_limits<float>::denorm_min();
        const std::vector<float> floats{0.f, -0.f, subnormal, -subnormal, 1e-30f, 0.2f, 0.49999997f, 0.5f,
                                        0.98235291f, 0.98235297f, 1.f, 1.5f, -2.f,
                                        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                        std::numeric_limits<float>::quiet_NaN(), 0.75f, 0.3f};
        for (const bool invert : {false, true})
            // 0.2 equals 51 / 255; the subnormal cutoff excludes only zero and below.
            for (const float threshold : {0.f, 0.2f, 0.5f, 1.f, subnormal})
                for (const bool high_band : {false, true}) {
                    const MaskedImageFill policy{.alpha = true, .invert = invert, .threshold = threshold, .keep_high_band = high_band};
                    check_alpha_fill(bytes, policy);
                    check_alpha_fill(words, policy);
                    check_alpha_fill(floats, policy);
                }
    }

    TEST(MaskedImageFillHost, FloatMasksKeepSubnormalAndNonFiniteOrder) {
        constexpr int width = 19, height = 17;
        const float subnormal = std::numeric_limits<float>::denorm_min();
        const std::vector<float> values{0.f, -0.f, subnormal, 2 * subnormal, -subnormal, 0.5f, 1.f, 2.f,
                                        std::numeric_limits<float>::quiet_NaN(),
                                        std::numeric_limits<float>::infinity()};
        std::vector<float> mask(size_t(width) * height);
        for (size_t i = 0; i < mask.size(); ++i)
            mask[i] = values[(i * 7) % values.size()];
        std::mt19937 random(11);
        const auto pixels = random_pixels<float>(mask.size() * 3, random);
        for (const bool invert : {false, true})
            for (const float threshold : {0.f, subnormal, 2 * subnormal, 0.5f, std::numeric_limits<float>::infinity()}) {
                SCOPED_TRACE(::testing::Message() << "invert=" << invert << " threshold=" << threshold);
                const MaskedImageFill policy{.mask = Tensor::from_vector(mask, {height, width}, Device::CPU),
                                             .invert = invert,
                                             .threshold = threshold};
                std::vector<uint8_t> keep(mask.size());
                for (size_t i = 0; i < mask.size(); ++i)
                    keep[i] = keeps(mask[i], policy);
                auto actual = pixels;
                lfs::io::fill_masked_image_host(actual.data(), width, height, 3, true, policy);
                expect_same_samples(actual, brute_force(pixels, width, height, 3, true, keep));
            }
    }

    TEST(MaskedImageFillHost, OutputGridMasksExpandNearestAndOtherGridsKeepTheImage) {
        constexpr int width = 53, height = 37;
        std::mt19937 random(5);
        const auto pixels = random_pixels<uint16_t>(size_t(width) * height * 3, random);
        for (const auto [resize, max_width] : {std::pair{2, 0}, std::pair{8, 0}, std::pair{1, 20}, std::pair{4, 9}}) {
            SCOPED_TRACE(::testing::Message() << "resize=" << resize << " max_width=" << max_width);
            const auto [mask_width, mask_height] = lfs::core::resized_image_dimensions(width, height, resize, max_width);
            const auto mask = random_mask(size_t(mask_width) * mask_height, 0.4, random);
            const MaskedImageFill policy{.mask = host_bytes(mask, {size_t(mask_height), size_t(mask_width)}),
                                         .threshold = 0.5f,
                                         .resize_factor = resize,
                                         .max_width = max_width};
            std::vector<uint8_t> keep(size_t(width) * height);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const int my = height > 1 ? std::min(y * mask_height / height, mask_height - 1) : 0;
                    const int mx = width > 1 ? std::min(x * mask_width / width, mask_width - 1) : 0;
                    keep[size_t(y) * width + x] = keeps(normalized(mask[size_t(my) * mask_width + mx]), policy);
                }
            EXPECT_TRUE(lfs::io::masked_image_fill_applies(policy, width, height));
            auto actual = pixels;
            lfs::io::fill_masked_image_host(actual.data(), width, height, 3, false, policy);
            expect_same_samples(actual, brute_force(pixels, width, height, 3, false, keep));
        }
        const MaskedImageFill other{.mask = host_bytes(std::vector<uint8_t>(size_t(width + 1) * height, 0), {height, width + 1})};
        EXPECT_FALSE(lfs::io::masked_image_fill_applies(other, width, height));
        EXPECT_FALSE(lfs::io::masked_image_fill_applies({}, width, height));
        auto unchanged = pixels;
        lfs::io::fill_masked_image_host(unchanged.data(), width, height, 3, false, other);
        lfs::io::fill_masked_image_host(unchanged.data(), width, height, 3, false, {});
        EXPECT_EQ(unchanged, pixels);
    }

    TEST(MaskedImageFillHost, MaskShapesAndDtypesConvertAsOnCuda) {
        constexpr int width = 31, height = 22;
        std::mt19937 random(3);
        const auto pixels = random_pixels<uint8_t>(size_t(width) * height * 3, random);
        std::vector<float> values(size_t(width) * height);
        for (auto& value : values)
            value = std::uniform_int_distribution<int>(0, 3)(random) == 0 ? 1.f : 0.f;
        const auto floats = Tensor::from_vector(values, {height, width}, Device::CPU);
        std::vector<uint8_t> keep(values.size());
        for (size_t i = 0; i < values.size(); ++i)
            keep[i] = values[i] > 0.f;
        const auto expected = brute_force(pixels, width, height, 3, false, keep);
        for (const auto& mask : {floats, floats.unsqueeze(0), floats.unsqueeze(2), floats.to(DataType::Bool),
                                 floats.to(DataType::Int32), floats.to(DataType::Float16)}) {
            SCOPED_TRACE(::testing::Message() << "shape rank=" << mask.ndim() << " dtype=" << static_cast<int>(mask.dtype()));
            auto actual = pixels;
            lfs::io::fill_masked_image_host(actual.data(), width, height, 3, false, {.mask = mask});
            EXPECT_EQ(actual, expected);
        }
    }

    TEST(MaskedImageFillHost, TensorImagesMatchTheirSamplesAndBadLayoutsThrow) {
        constexpr int width = 17, height = 13;
        std::mt19937 random(9);
        const auto mask = random_mask(size_t(width) * height, 0.3, random);
        const MaskedImageFill policy{.mask = host_bytes(mask, {height, width}), .threshold = 0.5f};
        const auto words = random_pixels<uint16_t>(size_t(width) * height * 4, random);
        auto expected = words;
        lfs::io::fill_masked_image_host(expected.data(), width, height, 4, true, policy);
        // Float16 is the 16-bit sample container of CUDA staging images.
        auto tensor = Tensor::empty({4, height, width}, Device::CPU, DataType::Float16);
        std::memcpy(tensor.data_ptr(), words.data(), words.size() * sizeof(uint16_t));
        lfs::io::fill_masked_image_host(tensor, true, policy);
        std::vector<uint16_t> actual(words.size());
        std::memcpy(actual.data(), tensor.data_ptr(), actual.size() * sizeof(uint16_t));
        EXPECT_EQ(actual, expected);

        const auto bytes = random_pixels<uint8_t>(size_t(width) * height * 3, random);
        auto byte_tensor = host_bytes(bytes, {height, width, 3});
        lfs::io::fill_masked_image_host(byte_tensor, false, policy);
        auto byte_expected = bytes;
        lfs::io::fill_masked_image_host(byte_expected.data(), width, height, 3, false, policy);
        EXPECT_EQ(byte_tensor.to_vector_uint8(), byte_expected);

        std::vector<uint8_t> two(size_t(width) * height * 2);
        EXPECT_THROW(lfs::io::fill_masked_image_host(two.data(), width, height, 2, false, policy), std::invalid_argument);
        std::vector<uint8_t> rgb(size_t(width) * height * 3);
        EXPECT_THROW(lfs::io::fill_masked_image_host(rgb.data(), width, height, 3, false, {.alpha = true}),
                     std::invalid_argument);
        auto ints = Tensor::zeros({height, width, 3}, Device::CPU, DataType::Int32);
        EXPECT_THROW(lfs::io::fill_masked_image_host(ints, false, policy), std::invalid_argument);
    }
} // namespace
