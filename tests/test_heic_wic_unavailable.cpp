/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <array>
#include <gtest/gtest.h>
#include <objbase.h>
#include <wincodec.h>
#include <windows.h>
#include <wrl/client.h>

namespace {
    HRESULT WINAPI missing_wic_factory(REFCLSID, LPUNKNOWN, DWORD, REFIID, LPVOID* output) {
        *output = nullptr;
        return REGDB_E_CLASSNOTREG;
    }
} // namespace

// Compile the actual private adapter with an unavailable COM factory. Keep the
// substitution in this test translation unit: no runtime hook, global provider
// override, registry edits or installed-extension changes enter production.
#define CoCreateInstance missing_wic_factory
#define read_heic_system read_heic_system_without_wic
#include "../src/core/image_heic_system.cpp"
#undef read_heic_system
#undef CoCreateInstance

TEST(ProductionHeif, MissingWicFactoryFailsWithoutChangingOutput) {
    namespace codec = lfs::core::image_codecs;
    const std::filesystem::path path = "unopened.heic";
    const std::array<uint8_t, 4> bytes{0, 0, 0, 0};
    const codec::Probe expected{2, 2, 3, codec::SampleType::UInt8};
    for (const bool memory : {false, true}) {
        for (const bool probe : {false, true}) {
            codec::Image image{1, 1, 3, codec::SampleType::UInt8, {17, 23, 41}};
            std::string error;
            EXPECT_FALSE(codec::read_heic_system_without_wic(memory ? nullptr : &path, bytes, expected,
                                                             8, 1, false, probe ? nullptr : &image, error));
            EXPECT_NE(error.find("provider factory failed"), std::string::npos) << error;
            EXPECT_NE(error.find("2147746132"), std::string::npos) << error;
            EXPECT_EQ(image.width, 1);
            EXPECT_EQ(image.height, 1);
            EXPECT_EQ(image.data, (std::vector<uint8_t>{17, 23, 41}));
        }
    }
}
