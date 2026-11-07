/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_codecs.hpp"
#include "core/image_io.hpp"
#include "core/uuid.hpp"
#include "gui/rmlui/rmlui_vk_backend.hpp"
#include <array>
#include <filesystem>
#include <gtest/gtest.h>

using lfs::vis::gui::PreviewTextureStatus;

namespace {
    // Exercise the public renderer ABI, including Windows' DLL boundary.
#ifdef _MSC_VER
    __declspec(noinline)
#else
    __attribute__((noinline))
#endif
    void
    releaseTexture(Rml::RenderInterface& renderer, Rml::TextureHandle texture) {
        renderer.ReleaseTexture(texture);
    }
} // namespace

class RenderInterfaceVKTestAccess {
public:
    static auto decode(const std::filesystem::path& path, int max_size, bool overlay) {
        return RenderInterface_VK::DecodePreviewTexture(path, max_size, false, overlay);
    }
    static auto decodeMask(const std::shared_ptr<const std::vector<std::byte>>& encoded, int max_size) {
        return RenderInterface_VK::DecodePreviewTexture({}, max_size, false, true, encoded);
    }

    static void checkRendererLocalStatus() {
        RenderInterface_VK main;
        RenderInterface_VK external;
        RenderInterface_VK::texture_data_t main_texture, external_texture;
        const std::string source = "preview://path=same-file&rev=1";
        main_texture.m_preview_status = PreviewTextureStatus::Pending;
        external_texture.m_preview_status = PreviewTextureStatus::Ready;
        main.m_preview_textures_by_source[source] = &main_texture;
        external.m_preview_textures_by_source[source] = &external_texture;
        EXPECT_EQ(main.previewTextureStatus(source), PreviewTextureStatus::Pending);
        EXPECT_EQ(external.previewTextureStatus(source), PreviewTextureStatus::Ready);
        main.m_preview_textures_by_source.clear();
        external.m_preview_textures_by_source.clear();
    }

    static void checkPreviewStatusLifetime() {
        RenderInterface_VK backend;
        RenderInterface_VK::texture_data_t texture;
        texture.m_preview_source = "preview://path=fixture&rev=1";
        texture.m_preview_status = PreviewTextureStatus::Pending;
        EXPECT_EQ(backend.previewTextureStatus(texture.m_preview_source), PreviewTextureStatus::Unknown);
        backend.m_preview_textures_by_source[texture.m_preview_source] = &texture;
        EXPECT_EQ(backend.previewTextureStatus(texture.m_preview_source), PreviewTextureStatus::Pending);
        texture.m_preview_status = PreviewTextureStatus::Ready;
        auto retired = texture;
        releaseTexture(backend, reinterpret_cast<Rml::TextureHandle>(&retired));
        EXPECT_EQ(backend.previewTextureStatus(texture.m_preview_source), PreviewTextureStatus::Ready);
        texture.m_preview_status = PreviewTextureStatus::Error;
        EXPECT_EQ(backend.previewTextureStatus(texture.m_preview_source), PreviewTextureStatus::Error);
        releaseTexture(backend, reinterpret_cast<Rml::TextureHandle>(&texture));
        EXPECT_EQ(backend.previewTextureStatus(texture.m_preview_source), PreviewTextureStatus::Unknown);
        // These two inert fixtures own no GPU resources and live on this stack.
        for (auto& pending : backend.m_pending_for_deletion_textures_by_frames)
            pending.clear();
    }
};

TEST(PreviewTextureStatus, FollowsLiveSourceAndRetirement) {
    RenderInterfaceVKTestAccess::checkPreviewStatusLifetime();
}

TEST(PreviewTextureStatus, SameSourceHasIndependentRendererReadiness) {
    RenderInterfaceVKTestAccess::checkRendererLocalStatus();
}

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
    const auto expired = RenderInterfaceVKTestAccess::decodeMask({}, 64);
    EXPECT_TRUE(expired.expired);
    EXPECT_TRUE(expired.pixels.empty());
    const auto corrupt = RenderInterfaceVKTestAccess::decodeMask(
        std::make_shared<const std::vector<std::byte>>(3, std::byte{0}), 64);
    EXPECT_FALSE(corrupt.expired);
    EXPECT_TRUE(corrupt.pixels.empty());
}

TEST_F(PreviewOverlayDecode, GrayscaleCoverageIsPremultipliedButOrdinaryPhotoStaysOpaque) {
    const auto path = directory / "gray.png";
    const std::array<unsigned char, 3> source{0, 128, 255};
    ASSERT_TRUE(lfs::core::save_png(path, source.data(), 3, 1, 1, 8, 1));
    const auto mask = RenderInterfaceVKTestAccess::decode(path, 64, true);
    ASSERT_EQ(mask.pixels.size(), 12u);
    for (std::size_t i = 0; i < source.size(); ++i)
        for (std::size_t c = 0; c < 4; ++c)
            EXPECT_EQ(mask.pixels[i * 4 + c], source[i]);
    const auto photo = RenderInterfaceVKTestAccess::decode(path, 64, false);
    ASSERT_EQ(photo.pixels.size(), 12u);
    for (std::size_t i = 0; i < source.size(); ++i)
        EXPECT_EQ(photo.pixels[i * 4 + 3], 255);
    const auto small = RenderInterfaceVKTestAccess::decode(path, 1, true);
    EXPECT_EQ(small.width, 1);
    EXPECT_EQ(small.height, 1);
    EXPECT_EQ(small.pixels, (std::vector<Rml::byte>{128, 128, 128, 128}));
    const auto encoded = std::make_shared<const std::vector<std::byte>>(
        lfs::core::image_codecs::encode_png(source, 3, 1, 1));
    ASSERT_FALSE(encoded->empty());
    const auto memory_mask = RenderInterfaceVKTestAccess::decodeMask(encoded, 64);
    EXPECT_EQ(memory_mask.width, mask.width);
    EXPECT_EQ(memory_mask.height, mask.height);
    EXPECT_EQ(memory_mask.pixels, mask.pixels);
    EXPECT_EQ(RenderInterfaceVKTestAccess::decodeMask(encoded, 1).pixels, small.pixels);
}

TEST_F(PreviewOverlayDecode, RgbaDownscaleDoesNotBleedTransparentColor) {
    const auto path = directory / "rgba.png";
    const std::array<unsigned char, 8> source{255, 0, 0, 0, 0, 0, 255, 128};
    ASSERT_TRUE(lfs::core::save_png(path, source.data(), 2, 1, 4, 8, 1));
    const auto original = RenderInterfaceVKTestAccess::decode(path, 64, true);
    EXPECT_EQ(original.pixels, (std::vector<Rml::byte>{0, 0, 0, 0, 0, 0, 128, 128}));
    const auto small = RenderInterfaceVKTestAccess::decode(path, 1, true);
    EXPECT_EQ(small.pixels, (std::vector<Rml::byte>{0, 0, 64, 64}));
}

TEST_F(PreviewOverlayDecode, GrayAlphaRetainsItsAlphaInsteadOfUsingGrayAsCoverage) {
    const auto path = directory / "gray-alpha.png";
    const std::array<unsigned char, 4> source{200, 0, 100, 128};
    ASSERT_TRUE(lfs::core::save_png(path, source.data(), 2, 1, 2, 8, 1));
    const auto decoded = RenderInterfaceVKTestAccess::decode(path, 64, true);
    EXPECT_EQ(decoded.pixels, (std::vector<Rml::byte>{0, 0, 0, 0, 50, 50, 50, 128}));
}
