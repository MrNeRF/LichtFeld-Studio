/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/mask_normalization.hpp"
#include "core/image_loader.hpp"
#include "io/cache_image_loader.hpp"
#include "licht_test_support.hpp"
#include "mask_test_utils.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace {
    using namespace lfs::core;
    constexpr size_t width = 64;
    constexpr size_t height = 48;

    class CameraMaskThreshold : public ::testing::Test {
    protected:
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
                GTEST_SKIP() << "CUDA unavailable";
            lfs::io::CacheLoader::getInstance(false);
            set_image_loader([](const ImageLoadParams& p) {
                return lfs::io::CacheLoader::getInstance().load_cached_image(
                    p.path, {.resize_factor = p.resize_factor, .max_width = p.max_width, .cuda_stream = p.stream, .output_uint8 = p.output_uint8, .mask_grayscale = p.mask_grayscale});
            });
        }

        std::shared_ptr<Camera> camera(const std::filesystem::path& image = {},
                                       const std::filesystem::path& mask = {}, bool warped = false) {
            auto radial = warped ? Tensor::from_vector({-.1f, .02f}, {2}, Device::CPU) : Tensor{};
            auto result = std::make_shared<Camera>(
                Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
                55.f, 55.f, width / 2.f, height / 2.f, radial, Tensor{}, CameraModelType::PINHOLE,
                "frame.png", image, mask, width, height, 0);
            if (warped)
                result->prepare_undistortion();
            return result;
        }
    };
} // namespace

// The reference is host double division rounded once to float, independent of
// CUDA fast-math and of the production byte conversion.
TEST_F(CameraMaskThreshold, ExhaustiveByteNormalizationAndCutoffs) {
    std::vector<float> bytes(width * height), reference(bytes.size());
    std::vector<float> cutoffs{.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f, .25f, .75f, 1.f};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<float>(i % 256);
        reference[i] = lfs::test::mask::exact_ratio(static_cast<uint32_t>(i % 256), 255);
    }
    for (int k = 0; k < 256; ++k) {
        const float value = lfs::test::mask::exact_ratio(static_cast<uint32_t>(k), 255);
        cutoffs.push_back(std::nextafter(value, -std::numeric_limits<float>::infinity()));
        cutoffs.push_back(value);
        cutoffs.push_back(std::nextafter(value, std::numeric_limits<float>::infinity()));
    }
    std::sort(cutoffs.begin(), cutoffs.end());
    cutoffs.erase(std::unique(cutoffs.begin(), cutoffs.end()), cutoffs.end());
    for (const bool warped : {false, true}) {
        auto actual_camera = camera({}, {}, warped);
        auto reference_camera = camera({}, {}, warped);
        actual_camera->set_mask_tensor(Tensor::from_vector(bytes, {height, width}, Device::CPU).to(DataType::UInt8));
        reference_camera->set_mask_tensor(Tensor::from_vector(reference, {height, width}, Device::CPU));
        for (const bool invert : {false, true}) {
            SCOPED_TRACE(warped);
            SCOPED_TRACE(invert);
            const auto expected_tensor = reference_camera->load_and_get_mask(1, 0, invert, 0.f, false);
            const auto expected = expected_tensor.cpu().to_vector();
            const auto actual_tensor = actual_camera->load_and_get_mask(1, 0, invert, 0.f, false);
            const auto actual = actual_tensor.cpu().to_vector();
            size_t normalization_errors = 0, classification_errors = 0, camera_errors = 0;
            for (size_t i = 0; i < actual.size(); ++i)
                normalization_errors += std::bit_cast<uint32_t>(actual[i]) != std::bit_cast<uint32_t>(expected[i]);
            for (float threshold : cutoffs) {
                // ge covers the entire float cutoff domain. Camera's documented
                // threshold-zero request uses its separate binary-evaluation default.
                const auto classified = lfs::core::cuda::threshold_mask(actual_tensor, threshold).cpu().to_vector_uint8();
                for (size_t i = 0; i < classified.size(); ++i) {
                    classification_errors += classified[i] != (expected[i] >= threshold);
                }
                if (threshold > 0.f && threshold <= 1.f) {
                    const auto result = actual_camera->load_and_get_mask(1, 0, invert, threshold, true).cpu().to_vector_uint8();
                    for (size_t i = 0; i < result.size(); ++i) {
                        camera_errors += result[i] != (expected[i] >= threshold);
                    }
                }
            }
            EXPECT_EQ(normalization_errors, 0u);
            EXPECT_EQ(classification_errors, 0u);
            EXPECT_EQ(camera_errors, 0u);
        }
    }
}
