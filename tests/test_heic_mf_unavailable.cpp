/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <gtest/gtest.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <wincodec.h>
#include <windows.h>
#include <wrl/client.h>

namespace {
    unsigned loads = 0;
    HMODULE WINAPI missing_mf_module(LPCWSTR name, HANDLE, DWORD flags) {
        EXPECT_STREQ(name, L"mfplat.dll");
        EXPECT_EQ(flags, DWORD(LOAD_LIBRARY_SEARCH_SYSTEM32));
        ++loads;
        SetLastError(ERROR_MOD_NOT_FOUND);
        return nullptr;
    }
} // namespace

// Compile the real adapter with Media Foundation absent, without changing the OS.
#define LoadLibraryExW        missing_mf_module
#define read_heic_system      read_heic_system_without_mf
#define heic_system_available heic_system_available_without_mf
#include "../src/core/image_heic_system.cpp"
#undef heic_system_available
#undef read_heic_system
#undef LoadLibraryExW

TEST(ProductionHeif, MissingMediaFoundationIsALazyUnavailableProvider) {
    namespace codec = lfs::core::image_codecs;
    EXPECT_EQ(loads, 0u);
    std::string error;
    for (int i = 0; i < 2; ++i) {
        EXPECT_FALSE(codec::heic_system_available_without_mf(error));
        EXPECT_NE(error.find("provider unavailable"), std::string::npos);
        EXPECT_NE(error.find("Media Feature Pack"), std::string::npos);
        EXPECT_NE(error.find("0x8007007E"), std::string::npos);
    }
    EXPECT_EQ(loads, 1u);
    const std::filesystem::path path = "unopened.heic";
    const codec::Probe expected{2, 2, 3, codec::SampleType::UInt8};
    codec::Image image{1, 1, 3, codec::SampleType::UInt8, {17, 23, 41}};
    EXPECT_FALSE(codec::read_heic_system_without_mf(&path, {}, expected, 8, 1, false, nullptr, error));
    EXPECT_FALSE(codec::read_heic_system_without_mf(nullptr, {}, expected, 8, 1, false, &image, error));
    EXPECT_EQ(image.data, (std::vector<uint8_t>{17, 23, 41}));
    EXPECT_EQ(loads, 1u);
}
