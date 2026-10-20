// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <core/image_jxl.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <jxl/decode.h>
#include <media/image_output.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
    void require(bool ok, const char* reason) {
        if (!ok)
            throw std::runtime_error(reason);
    }
    std::vector<uint8_t> read(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(file), {}};
    }
} // namespace
int main() {
    using namespace lfs::media;
    namespace codecs = lfs::core::image_codecs;
    const auto path = std::filesystem::temp_directory_path() /
                      ("lfs-jxl-u16-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".jxl");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{path};
    try {
        // Unaligned, padded two-row input; thresholds, negatives, clipping and zero alpha.
        const std::array<float, 24> samples{-.1f, .0031308f, .5f, .5f, .25f, 1.1f, .75f, 1.f,
                                            .2f, .4f, .6f, 0.f, .125f, .25f, .5f, .25f,
                                            0.f, 1.f, .1f, 1.f, .3f, .2f, .1f, .75f};
        for (int channels : {3, 4})
            for (bool premultiplied : {false, true}) {
                if (channels == 3 && premultiplied)
                    continue;
                constexpr int width = 3, height = 2;
                const size_t stride = width * channels * sizeof(float) + 7;
                std::vector<uint8_t> storage(1 + stride * height, 0xa5);
                for (int i = 0; i < width * height; ++i)
                    for (int c = 0; c < channels; ++c) {
                        const auto value = samples[i * 4 + c];
                        std::memcpy(storage.data() + 1 + (i / width) * stride + ((i % width) * channels + c) * sizeof(float), &value, sizeof(value));
                    }
                const auto original = storage;
                FrameView frame{{width, height, stride, channels == 4 ? FramePixelFormat::RGBAFloat32 : FramePixelFormat::RGBFloat32, {ColorTransfer::Linear, ColorPrimaries::Bt709, channels == 3 ? AlphaMode::None : premultiplied ? AlphaMode::Premultiplied
                                                                                                                                                                                                                                     : AlphaMode::Independent}},
                                {},
                                std::span(storage).subspan(1)};
                for (auto primaries : {ColorPrimaries::Bt709, ColorPrimaries::Bt2020})
                    for (auto transfer : {ColorTransfer::Linear, ColorTransfer::Srgb}) {
                        frame.layout.color.primaries = primaries;
                        JxlOutputOptions options;
                        options.precision = JxlPrecision::UInt16;
                        options.transfer = transfer;
                        options.effort = 1;
                        options.overwrite = true;
                        const auto result = ImageOutput::writeJxl(path, frame, options);
                        require(bool(result), result ? "" : std::string(result.error().detail()).c_str());
                        const auto encoded = read(path);
                        std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> decoder(JxlDecoderCreate(nullptr), JxlDecoderDestroy);
                        require(bool(decoder), "independent decoder allocation");
                        require(JxlDecoderSubscribeEvents(decoder.get(), JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS, "decoder subscription");
                        JxlDecoderSetInput(decoder.get(), encoded.data(), encoded.size());
                        JxlDecoderCloseInput(decoder.get());
                        std::vector<uint16_t> direct(width * height * channels);
                        const JxlPixelFormat format{uint32_t(channels), JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0};
                        bool complete = false;
                        for (;;) {
                            const auto status = JxlDecoderProcessInput(decoder.get());
                            if (status == JXL_DEC_SUCCESS)
                                break;
                            if (status == JXL_DEC_BASIC_INFO) {
                                JxlBasicInfo info{};
                                require(JxlDecoderGetBasicInfo(decoder.get(), &info) == JXL_DEC_SUCCESS && info.bits_per_sample == 16 && info.exponent_bits_per_sample == 0, "UInt16 codestream precision");
                            } else if (status == JXL_DEC_COLOR_ENCODING) {
                                JxlColorEncoding color{};
                                require(JxlDecoderGetColorAsEncodedProfile(decoder.get(), JXL_COLOR_PROFILE_TARGET_DATA, &color) == JXL_DEC_SUCCESS, "encoded color profile");
                                require(color.transfer_function == (transfer == ColorTransfer::Srgb ? JXL_TRANSFER_FUNCTION_SRGB : JXL_TRANSFER_FUNCTION_LINEAR), "encoded transfer");
                                require(color.primaries == (primaries == ColorPrimaries::Bt2020 ? JXL_PRIMARIES_2100 : JXL_PRIMARIES_SRGB), "encoded primaries");
                            } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
                                require(JxlDecoderSetImageOutBuffer(decoder.get(), &format, direct.data(), direct.size() * 2) == JXL_DEC_SUCCESS, "output buffer");
                            } else if (status == JXL_DEC_FULL_IMAGE)
                                complete = true;
                            else
                                require(false, "independent decode failed");
                        }
                        require(complete, "complete independent image");
                        codecs::Image native;
                        std::string error;
                        codecs::JxlDecodeLimits limits;
                        limits.color_policy = codecs::JxlColorPolicy::Native;
                        require(codecs::decode_jxl(encoded, native, error, false, limits), error.c_str());
                        require(native.data.size() == direct.size() * 2 && std::memcmp(native.data.data(), direct.data(), native.data.size()) == 0, "native and independent codes match exactly");
                        for (int i = 0; i < width * height; ++i)
                            for (int c = 0; c < channels; ++c) {
                                double value = samples[i * 4 + c];
                                if (premultiplied && c < 3)
                                    value = samples[i * 4 + 3] > 0 ? double(samples[i * 4 + c] / samples[i * 4 + 3]) : 0;
                                value = std::clamp(value, 0., 1.);
                                if (c < 3 && transfer == ColorTransfer::Srgb)
                                    value = value <= double(.0031308f) ? 12.92 * value : 1.055 * std::pow(value, 1. / 2.4) - .055;
                                const auto expected = std::lround(value * 65535.);
                                require(std::abs(long(direct[i * channels + c]) - expected) <= (c == 3 ? 0 : 1), "independent quantization reference");
                            }
                        require(storage == original, "borrowed source unchanged");
                        options.cancelled = [] { return true; };
                        require(!ImageOutput::writeJxl(path, frame, options) && read(path) == encoded, "cancel preserves destination");
                        options.cancelled = {};
                        options.precision = JxlPrecision::Float32;
                        options.transfer = ColorTransfer::Linear;
                        require(!ImageOutput::writeJxl(path, frame, options) && read(path) == encoded, "Float32 effort1 still rejected");
                    }
            }
        std::cout << "PASS JXL UInt16: profiles, independent/native codes, padded input, alpha, cancellation, Float32 guard\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
