/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "io/nvcodec_image_loader.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

    std::vector<uint8_t> read_test_jpeg(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

} // namespace

TEST(NvCodecSentinelValidatorTest, SkippedMemberIsRetriedOrFailsHardBeforeReturn) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const auto path = std::filesystem::path(PROJECT_ROOT_PATH) /
                      "data/bicycle/images_4/_DSC8739.JPG";
    if (!std::filesystem::is_regular_file(path)) {
        GTEST_SKIP() << "bicycle dataset is absent: " << path;
    }
    const std::vector<uint8_t> jpeg = read_test_jpeg(path);
    ASSERT_FALSE(jpeg.empty());
    const std::vector<std::pair<const uint8_t*, size_t>> spans{
        {jpeg.data(), jpeg.size()},
        {jpeg.data(), jpeg.size()}};

    lfs::io::NvCodecImageLoader::Options retry_options;
    retry_options.decoder_pool_size = 1;
    retry_options.sentinel_validation_test_seam.skipped_member = 1;
    std::unique_ptr<lfs::io::NvCodecImageLoader> retry_loader;
    try {
        retry_loader = std::make_unique<lfs::io::NvCodecImageLoader>(retry_options);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nvImageCodec unavailable: " << e.what();
    }

    // The seam sends the primary decode for member 1 to a discard buffer. The
    // real destination can become a varied image only if sentinel validation
    // detects it and the non-HW CUDA retry overwrites it.
    const auto recovered = retry_loader->decode_jpeg_batch_from_spans(
        spans, nullptr, true, true);
    ASSERT_EQ(recovered.size(), spans.size());
    ASSERT_TRUE(recovered[1].is_valid());
    EXPECT_EQ(recovered[1].device(), lfs::core::Device::GPU);
    EXPECT_EQ(recovered[1].dtype(), lfs::core::DataType::UInt8);
    const auto recovered_values = recovered[1].cpu().to_vector();
    ASSERT_FALSE(recovered_values.empty());
    const auto [min_it, max_it] = std::minmax_element(
        recovered_values.begin(), recovered_values.end());
    EXPECT_LT(*min_it, *max_it);
    // Both members decode the same bytes, so the retried member must reproduce
    // member 0 exactly; a leftover sentinel pattern would differ everywhere.
    ASSERT_TRUE(recovered[0].is_valid());
    EXPECT_EQ(recovered[0].cpu().to_vector(), recovered_values);

    lfs::io::NvCodecImageLoader::Options fail_options;
    fail_options.decoder_pool_size = 1;
    fail_options.sentinel_validation_test_seam.skipped_member = 1;
    fail_options.sentinel_validation_test_seam.skip_cuda_retry = true;
    std::unique_ptr<lfs::io::NvCodecImageLoader> fail_loader;
    try {
        fail_loader = std::make_unique<lfs::io::NvCodecImageLoader>(fail_options);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nvImageCodec unavailable for hard-failure case: " << e.what();
    }

    bool returned_batch = false;
    try {
        (void)fail_loader->decode_jpeg_batch_from_spans(
            spans, nullptr, true, true);
        returned_batch = true;
    } catch (const std::runtime_error& e) {
        EXPECT_NE(
            std::string(e.what()).find(
                "destination remained wholly unchanged after one CUDA retry"),
            std::string::npos);
    }
    EXPECT_FALSE(returned_batch) << "a twice-unwritten destination must never be returned";
}
