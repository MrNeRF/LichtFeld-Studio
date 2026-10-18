/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Offline fixture generator. The application exposes no JXL writer here.
#include <array>
#include <cstdlib>
#include <fstream>
#include <jxl/encode.h>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3)
        return 2;
    const bool wide = argc == 3 && std::atoi(argv[2]) == 16;
    const bool floating = argc == 3 && std::atoi(argv[2]) == 32;
    const auto check = [](JxlEncoderStatus status) {
        if (status != JXL_ENC_SUCCESS)
            throw std::runtime_error("Fixture encoding failed");
    };
    std::unique_ptr<JxlEncoder, decltype(&JxlEncoderDestroy)> encoder(JxlEncoderCreate(nullptr), JxlEncoderDestroy);
    if (!encoder)
        return 1;
    JxlBasicInfo info;
    JxlEncoderInitBasicInfo(&info);
    info.xsize = info.ysize = 2;
    info.num_color_channels = 3;
    info.bits_per_sample = floating ? 32 : wide ? 16
                                                : 8;
    info.exponent_bits_per_sample = floating ? 8 : 0;
    info.uses_original_profile = JXL_TRUE;
    check(JxlEncoderSetBasicInfo(encoder.get(), &info));
    JxlColorEncoding color;
    if (floating)
        JxlColorEncodingSetToLinearSRGB(&color, JXL_FALSE);
    else
        JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
    check(JxlEncoderSetColorEncoding(encoder.get(), &color));
    auto* frame = JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);
    check(JxlEncoderSetFrameLossless(frame, JXL_TRUE));
    const std::array<uint8_t, 12> pixels{255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0};
    const std::array<uint16_t, 12> pixels16{65535, 32768, 0, 65535, 32768, 0, 65535, 32768, 0, 65535, 32768, 0};
    const std::array<float, 12> pixels32{2.f, .5f, -.25f, 2.f, .5f, -.25f, 2.f, .5f, -.25f, 2.f, .5f, -.25f};
    const JxlPixelFormat format{3, floating ? JXL_TYPE_FLOAT : wide ? JXL_TYPE_UINT16
                                                                    : JXL_TYPE_UINT8,
                                JXL_NATIVE_ENDIAN, 0};
    const void* samples = floating ? static_cast<const void*>(pixels32.data()) : wide ? static_cast<const void*>(pixels16.data())
                                                                                      : pixels.data();
    check(JxlEncoderAddImageFrame(frame, &format, samples, floating ? sizeof(pixels32) : wide ? sizeof(pixels16)
                                                                                              : sizeof(pixels)));
    JxlEncoderCloseInput(encoder.get());
    std::vector<uint8_t> bytes(4096);
    auto* next = bytes.data();
    size_t available = bytes.size();
    check(JxlEncoderProcessOutput(encoder.get(), &next, &available));
    std::ofstream output(argv[1], std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size() - available);
    return output ? 0 : 1;
}
