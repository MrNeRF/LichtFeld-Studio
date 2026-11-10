/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_codecs.hpp"
#include "core/image_io.hpp"
#include "core/path_utils.hpp"
#include "core/uuid.hpp"
#include "gui/rmlui/rml_preview_texture.hpp"
#include <array>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>

using lfs::vis::gui::PreviewTextureLoader;
using lfs::vis::gui::PreviewTexturePixels;
using lfs::vis::gui::PreviewTextureStatus;

namespace {
    auto decode(const std::filesystem::path& path, int max_size, bool overlay) {
        return lfs::vis::gui::decodePreviewTexture(path, max_size, false, overlay);
    }
    auto decodeMask(const std::shared_ptr<const std::vector<std::byte>>& encoded, int max_size) {
        return lfs::vis::gui::decodePreviewTexture({}, max_size, false, true, encoded);
    }

    // Handles are opaque to the loader; the upload callback stands in for a renderer.
    PreviewTextureStatus loadAndSettle(PreviewTextureLoader& loader, Rml::TextureHandle texture,
                                       const std::string& source, bool accept_upload) {
        const auto request = lfs::vis::gui::parsePreviewTextureRequest(source);
        EXPECT_TRUE(request);
        if (!request)
            return PreviewTextureStatus::Unknown;
        loader.load(texture, source, *request);
        EXPECT_EQ(loader.status(source), PreviewTextureStatus::Pending);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (loader.status(source) == PreviewTextureStatus::Pending && std::chrono::steady_clock::now() < deadline) {
            loader.uploadReady([accept_upload](Rml::TextureHandle, const Rml::String&, PreviewTexturePixels&) {
                return accept_upload;
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return loader.status(source);
    }
} // namespace

class PreviewOverlayDecode : public ::testing::Test {
protected:
    void SetUp() override {
        directory = std::filesystem::temp_directory_path() / ("lfs-preview-" + lfs::core::generate_uuid_v4().to_string());
        ASSERT_TRUE(std::filesystem::create_directory(directory));
    }
    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
    std::filesystem::path directory;
};

TEST_F(PreviewOverlayDecode, ExpiredMemoryProposalIsNotADecodeFailure) {
    const auto expired = decodeMask({}, 64);
    EXPECT_TRUE(expired.expired);
    EXPECT_TRUE(expired.pixels.empty());
    const auto corrupt = decodeMask(
        std::make_shared<const std::vector<std::byte>>(3, std::byte{0}), 64);
    EXPECT_FALSE(corrupt.expired);
    EXPECT_TRUE(corrupt.pixels.empty());
}

TEST_F(PreviewOverlayDecode, GrayscaleCoverageIsPremultipliedButOrdinaryPhotoStaysOpaque) {
    const auto path = directory / "gray.png";
    const std::array<unsigned char, 3> source{0, 128, 255};
    ASSERT_TRUE(lfs::core::save_png(path, source.data(), 3, 1, 1, 8, 1));
    const auto mask = decode(path, 64, true);
    ASSERT_EQ(mask.pixels.size(), 12u);
    for (std::size_t i = 0; i < source.size(); ++i)
        for (std::size_t c = 0; c < 4; ++c)
            EXPECT_EQ(mask.pixels[i * 4 + c], source[i]);
    const auto photo = decode(path, 64, false);
    ASSERT_EQ(photo.pixels.size(), 12u);
    for (std::size_t i = 0; i < source.size(); ++i)
        EXPECT_EQ(photo.pixels[i * 4 + 3], 255);
    const auto small = decode(path, 1, true);
    EXPECT_EQ(small.width, 1);
    EXPECT_EQ(small.height, 1);
    EXPECT_EQ(small.pixels, (std::vector<Rml::byte>{128, 128, 128, 128}));
    const auto encoded = std::make_shared<const std::vector<std::byte>>(
        lfs::core::image_codecs::encode_png(source, 3, 1, 1));
    ASSERT_FALSE(encoded->empty());
    const auto memory_mask = decodeMask(encoded, 64);
    EXPECT_EQ(memory_mask.width, mask.width);
    EXPECT_EQ(memory_mask.height, mask.height);
    EXPECT_EQ(memory_mask.pixels, mask.pixels);
    EXPECT_EQ(decodeMask(encoded, 1).pixels, small.pixels);
}

TEST_F(PreviewOverlayDecode, RgbaDownscaleDoesNotBleedTransparentColor) {
    const auto path = directory / "rgba.png";
    const std::array<unsigned char, 8> source{255, 0, 0, 0, 0, 0, 255, 128};
    ASSERT_TRUE(lfs::core::save_png(path, source.data(), 2, 1, 4, 8, 1));
    const auto original = decode(path, 64, true);
    EXPECT_EQ(original.pixels, (std::vector<Rml::byte>{0, 0, 0, 0, 0, 0, 128, 128}));
    const auto small = decode(path, 1, true);
    EXPECT_EQ(small.pixels, (std::vector<Rml::byte>{0, 0, 64, 64}));
}

TEST_F(PreviewOverlayDecode, GrayAlphaRetainsItsAlphaInsteadOfUsingGrayAsCoverage) {
    const auto path = directory / "gray-alpha.png";
    const std::array<unsigned char, 4> source{200, 0, 100, 128};
    ASSERT_TRUE(lfs::core::save_png(path, source.data(), 2, 1, 2, 8, 1));
    const auto decoded = decode(path, 64, true);
    EXPECT_EQ(decoded.pixels, (std::vector<Rml::byte>{0, 0, 0, 0, 50, 50, 50, 128}));
}

TEST_F(PreviewOverlayDecode, StatusFollowsLiveSourceAndRetirement) {
    const auto path = directory / "status.png";
    const std::array<unsigned char, 1> pixel{255};
    ASSERT_TRUE(lfs::core::save_png(path, pixel.data(), 1, 1, 1, 8, 1));
    const std::string source = "preview://path=" + lfs::core::path_to_utf8(path) + "&mode=mask&thumb=8&rev=1";

    PreviewTextureLoader loader;
    EXPECT_EQ(loader.status(source), PreviewTextureStatus::Unknown);
    EXPECT_EQ(loadAndSettle(loader, 1, source, true), PreviewTextureStatus::Ready);
    // A retired handle for the same source must not clear the live texture's status.
    EXPECT_EQ(loadAndSettle(loader, 2, source, false), PreviewTextureStatus::Error);
    loader.cancel(1);
    EXPECT_EQ(loader.status(source), PreviewTextureStatus::Error);
    loader.cancel(2);
    EXPECT_EQ(loader.status(source), PreviewTextureStatus::Unknown);
}

TEST_F(PreviewOverlayDecode, SameSourceHasIndependentRendererReadiness) {
    const auto path = directory / "shared.png";
    const std::array<unsigned char, 1> pixel{255};
    ASSERT_TRUE(lfs::core::save_png(path, pixel.data(), 1, 1, 1, 8, 1));
    const std::string source = "preview://path=" + lfs::core::path_to_utf8(path) + "&mode=mask&thumb=8&rev=1";

    PreviewTextureLoader main;
    PreviewTextureLoader external;
    EXPECT_EQ(loadAndSettle(main, 1, source, true), PreviewTextureStatus::Ready);
    EXPECT_EQ(external.status(source), PreviewTextureStatus::Unknown);
    EXPECT_EQ(loadAndSettle(external, 1, source, false), PreviewTextureStatus::Error);
    EXPECT_EQ(main.status(source), PreviewTextureStatus::Ready);
}

TEST_F(PreviewOverlayDecode, ExpiredMemoryPayloadReturnsToUnknown) {
    const std::string source = "preview://memory=proposal-1&mode=mask&thumb=8";
    PreviewTextureLoader loader;
    loader.setMemoryProvider([](std::string_view) { return std::weak_ptr<const std::vector<std::byte>>{}; });
    EXPECT_EQ(loadAndSettle(loader, 1, source, true), PreviewTextureStatus::Unknown);
}

TEST(PreviewTextureRequest, MemorySourcesAreBoundedMaskOverlays) {
    using lfs::vis::gui::parsePreviewTextureRequest;
    const auto mask = parsePreviewTextureRequest("preview://memory=a%2Fb&mode=mask&thumb=64");
    ASSERT_TRUE(mask);
    EXPECT_EQ(mask->memory_key, "a/b");
    EXPECT_TRUE(mask->mask_overlay);
    EXPECT_EQ(mask->max_size, 64);
    EXPECT_FALSE(parsePreviewTextureRequest("preview://memory=a&thumb=64"));
    EXPECT_FALSE(parsePreviewTextureRequest("preview://memory=a&mode=mask"));
    EXPECT_FALSE(parsePreviewTextureRequest("preview://memory=a&path=b&mode=mask&thumb=64"));
    EXPECT_FALSE(parsePreviewTextureRequest("preview://memory=a&kind=licht&mode=mask&thumb=64"));
}
