/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "io/nvcodec_image_loader.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
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

    std::vector<std::filesystem::path> find_dataset_jpegs(const size_t count) {
        std::vector<std::filesystem::path> found;
        const auto data_root = std::filesystem::path(PROJECT_ROOT_PATH) / "data";
        std::error_code ec;
        for (const auto& dataset : std::filesystem::directory_iterator(data_root, ec)) {
            const auto images = dataset.path() / "images_4";
            if (!std::filesystem::is_directory(images, ec)) {
                continue;
            }
            for (const auto& entry : std::filesystem::directory_iterator(images, ec)) {
                auto ext = entry.path().extension().string();
                std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
                if (ext == ".jpg" || ext == ".jpeg") {
                    found.push_back(entry.path());
                }
            }
            if (found.size() >= count) {
                std::ranges::sort(found);
                found.resize(count);
                return found;
            }
            found.clear();
        }
        return {};
    }

} // namespace

class NvCodecSentinelValidatorTest : public lfs::test::CudaBackendTest {};
class NvCodecBatchDecodeTest : public lfs::test::CudaBackendTest {};

TEST_F(NvCodecBatchDecodeTest, PlanarUint8MatchesInterleavedDecode) {
    const auto paths = find_dataset_jpegs(3);
    if (paths.size() != 3) {
        GTEST_SKIP() << "no dataset with images_4 JPEGs under data/";
    }
    std::vector<std::vector<uint8_t>> jpegs;
    std::vector<std::pair<const uint8_t*, size_t>> spans;
    for (const auto& path : paths) {
        jpegs.push_back(read_test_jpeg(path));
        ASSERT_FALSE(jpegs.back().empty()) << path;
    }
    for (const auto& jpeg : jpegs) {
        spans.emplace_back(jpeg.data(), jpeg.size());
    }

    std::unique_ptr<lfs::io::NvCodecImageLoader> loader;
    try {
        lfs::io::NvCodecImageLoader::Options options;
        options.decoder_pool_size = 1;
        loader = std::make_unique<lfs::io::NvCodecImageLoader>(options);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nvImageCodec unavailable: " << e.what();
    }

    // The uint8 request decodes planar RGB into the CHW destination; the float
    // request decodes interleaved RGB and transposes. A wrong plane stride or
    // channel order in the planar descriptor changes nearly every byte.
    const auto planar = loader->decode_jpeg_batch_from_spans(spans, nullptr, true, true);
    const auto interleaved = loader->decode_jpeg_batch_from_spans(spans, nullptr, false, true);
    ASSERT_EQ(planar.size(), spans.size());
    ASSERT_EQ(interleaved.size(), spans.size());
    for (size_t i = 0; i < spans.size(); ++i) {
        ASSERT_EQ(planar[i].dtype(), lfs::core::DataType::UInt8);
        ASSERT_EQ(interleaved[i].dtype(), lfs::core::DataType::Float32);
        ASSERT_EQ(planar[i].shape(), interleaved[i].shape());
        ASSERT_EQ(planar[i].shape()[0], 3u);
        const auto bytes = planar[i].cpu().to_vector();
        const auto values = interleaved[i].cpu().to_vector();
        ASSERT_EQ(bytes.size(), values.size());
        size_t mismatches = 0;
        for (size_t k = 0; k < bytes.size(); ++k) {
            if (std::lround(values[k] * 255.0f) != std::lround(bytes[k])) {
                ++mismatches;
            }
        }
        EXPECT_EQ(mismatches, 0u) << paths[i];
    }
}

TEST_F(NvCodecSentinelValidatorTest, SkippedMemberIsRetriedOrFailsHardBeforeReturn) {
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
