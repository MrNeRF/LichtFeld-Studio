/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor.hpp"
#include "gui/ui_texture.hpp"
#include "window/graphics_context.hpp"
#include <array>
#include <gtest/gtest.h>

TEST(UiTextureTensor, FullRegionAndLinearUploadsPreserveTensorDtypes) {
    using namespace lfs::core;
    using namespace lfs::vis;
    auto graphics = createGraphicsContext();
    if (!graphics->initializeHeadless())
        GTEST_SKIP() << graphics->lastError();
    gui::connectUiTextureGraphics(graphics.get());
    struct Disconnect {
        ~Disconnect() { gui::connectUiTextureGraphics(nullptr); }
    } disconnect;
    gui::UiTexture texture;
    std::array<uint8_t, 16> pixels{0, 1, 2, 255, 10, 20, 30, 255, 2, 4, 6, 255, 100, 200, 250, 255};
    ASSERT_TRUE(texture.upload(pixels.data(), 2, 2, 4));
    ASSERT_TRUE(texture.valid());
    auto image = texture.image();
    ASSERT_TRUE(image);
    EXPECT_EQ(image->dtype(), DataType::UInt8);
    for (size_t i = 0; i < pixels.size(); ++i)
        EXPECT_EQ(image->ptr<uint8_t>()[i], pixels[i]);
    std::array<uint8_t, 4> region{50, 60, 70, 80};
    ASSERT_TRUE(texture.uploadRegion(region.data(), 2, 2, 1, 0, 1, 1, 4));
    image = texture.image();
    for (size_t i = 0; i < region.size(); ++i)
        EXPECT_EQ(image->ptr<uint8_t>()[4 + i], region[i]);
    auto linear = Tensor::full({2, 2, 4}, 0.125f, Device::CPU);
    auto uploaded = texture.uploadLinearRgba(linear);
    ASSERT_TRUE(uploaded) << uploaded.error().detail();
    image = texture.image();
    ASSERT_TRUE(image);
    EXPECT_EQ(image->dtype(), DataType::Float32);
    for (size_t i = 0; i < image->numel(); ++i)
        EXPECT_FLOAT_EQ(image->ptr<float>()[i], 0.125f);
    EXPECT_EQ(gui::uiTextureImage(texture.textureId()).get(), image.get());
    EXPECT_FALSE(texture.rmlSrcUrl(2, 2).empty());
    texture.reset();
    EXPECT_FALSE(texture.valid());
    EXPECT_TRUE(graphics->waitIdle());
}
