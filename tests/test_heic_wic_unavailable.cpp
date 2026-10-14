/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <array>
#include <gtest/gtest.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <wincodec.h>
#include <windows.h>
#include <wrl/client.h>

namespace {
    unsigned enumerations = 0;
    HRESULT WINAPI missing_hevc_decoder(GUID category, UINT32, const MFT_REGISTER_TYPE_INFO* input,
                                        const MFT_REGISTER_TYPE_INFO*, IMFActivate*** output, UINT32* count) {
        EXPECT_EQ(category, MFT_CATEGORY_VIDEO_DECODER);
        EXPECT_EQ(input->guidSubtype, MFVideoFormat_HEVC);
        ++enumerations;
        *output = nullptr;
        *count = 0;
        return S_OK;
    }
} // namespace

// Compile the actual adapter with WIC installed but no HEVC MFT. The
// substitution is confined to this translation unit, without registry edits.
#define MFTEnumEx             missing_hevc_decoder
#define read_heic_system      read_heic_system_without_hevc
#define heic_system_available heic_system_available_without_hevc
#include "../src/core/image_heic_system.cpp"
#undef heic_system_available
#undef read_heic_system
#undef MFTEnumEx

TEST(ProductionHeif, MissingHevcDecoderFailsBeforeWicProbeAndPixelCopy) {
    namespace codec = lfs::core::image_codecs;
    const std::filesystem::path path = "unopened.heic";
    const std::array<uint8_t, 4> bytes{};
    const codec::Probe expected{2, 2, 3, codec::SampleType::UInt8};
    for (const bool memory : {false, true}) {
        for (const bool probe : {false, true}) {
            codec::Image image{1, 1, 3, codec::SampleType::UInt8, {17, 23, 41}};
            std::string error;
            EXPECT_FALSE(codec::read_heic_system_without_hevc(memory ? nullptr : &path, bytes, expected,
                                                              8, 1, false, probe ? nullptr : &image, error));
            EXPECT_NE(error.find("provider unavailable"), std::string::npos) << error;
            EXPECT_NE(error.find("HEVC Video Extensions"), std::string::npos) << error;
            EXPECT_NE(error.find("HEIF Image Extensions"), std::string::npos) << error;
            EXPECT_NE(error.find("0xC00D5212"), std::string::npos) << error;
            EXPECT_EQ(image.width, 1);
            EXPECT_EQ(image.height, 1);
            EXPECT_EQ(image.data, (std::vector<uint8_t>{17, 23, 41}));
        }
    }
    EXPECT_EQ(enumerations, 1u);
    std::string error;
    EXPECT_FALSE(codec::wic_status(WINCODEC_ERR_COMPONENTNOTFOUND, "open", error));
    EXPECT_NE(error.find("provider unavailable"), std::string::npos);
    EXPECT_NE(error.find("0x88982F50"), std::string::npos);
}
