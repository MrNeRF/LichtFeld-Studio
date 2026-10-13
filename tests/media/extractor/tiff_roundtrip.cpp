// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <core/image_codecs.hpp>
#include <core/image_io.hpp>
#include <cstring>
#include <iostream>
#include <media/image_output.hpp>
#include <memory>

int main() {
    using namespace lfs::media;
    const auto path = std::filesystem::temp_directory_path() /
                      ("lfs-tiff-roundtrip-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tif");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{path};
    const std::array<float, 8> samples{.125f, .25f, .5f, .375f, .35f, .3f, .2f, .75f};
    std::array<uint8_t, sizeof(samples)> bytes{};
    std::memcpy(bytes.data(), samples.data(), bytes.size());
    for (auto primaries : {ColorPrimaries::Bt709, ColorPrimaries::Bt2020}) {
        for (auto transfer : {ColorTransfer::Linear, ColorTransfer::Srgb, ColorTransfer::Bt709}) {
            FrameView frame{{2, 1, bytes.size(), FramePixelFormat::RGBAFloat32, {ColorTransfer::Linear, primaries, AlphaMode::Independent}}, {}, bytes};
            Tiff16OutputOptions options;
            options.transfer = transfer;
            options.overwrite = true;
            const auto result = ImageOutput::writeTiff16(path, frame, options);
            if (!result) {
                std::cerr << result.error().detail();
                return 1;
            }
            auto [rgb, width, height, channels] = lfs::core::load_image_u16(path);
            std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> rgb_owner(rgb, lfs::core::free_image);
            auto [rgba, aw, ah, ac] = lfs::core::load_image_with_alpha_u16(path);
            std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> rgba_owner(rgba, lfs::core::free_image);
            auto [rgb8, bw, bh, bc] = lfs::core::load_image(path);
            std::unique_ptr<uint8_t, decltype(&lfs::core::free_image)> rgb8_owner(rgb8, lfs::core::free_image);
            auto [raw, rw, rh, rc] = lfs::core::load_image_float(path);
            std::unique_ptr<float, decltype(&lfs::core::free_image_float)> raw_owner(raw, lfs::core::free_image_float);
            if (!rgb || !rgba || !rgb8 || !raw || width != 2 || height != 1 || channels != 3 || aw != 2 || ah != 1 || ac != 4 || bw != 2 || bh != 1 || bc != 3 || rw != 2 || rh != 1 || rc != 4)
                return 2;
            for (size_t pixel = 0; pixel < 2; ++pixel) {
                std::array<double, 3> linear{samples[pixel * 4], samples[pixel * 4 + 1], samples[pixel * 4 + 2]};
                if (primaries == ColorPrimaries::Bt2020) {
                    // Independent double reference, RGB->XYZ->RGB D65 matrices.
                    const auto v = linear;
                    const double x = .6369580483 * v[0] + .1446169036 * v[1] + .1688809752 * v[2];
                    const double y = .2627002120 * v[0] + .6779980715 * v[1] + .0593017165 * v[2];
                    const double z = .0280726930 * v[1] + 1.0609850577 * v[2];
                    linear = {3.2409699419 * x - 1.5373831776 * y - .4986107603 * z,
                              -.9692436363 * x + 1.8759675015 * y + .0415550574 * z,
                              .0556300797 * x - .2039769589 * y + 1.0569715142 * z};
                }
                for (size_t c = 0; c < 3; ++c) {
                    const double v = std::clamp(linear[c], 0., 1.);
                    const double encoded = v <= .0031308 ? 12.92 * v : 1.055 * std::pow(v, 1. / 2.4) - .055;
                    const auto expected = std::lround(encoded * 65535.);
                    if (std::abs(long(rgb[pixel * 3 + c]) - expected) > 4 || rgba[pixel * 4 + c] != rgb[pixel * 3 + c] ||
                        std::abs(int(rgb8[pixel * 3 + c]) - int(std::lround(encoded * 255.))) > 1)
                        return 3;
                    const double source = samples[pixel * 4 + c];
                    const double stored = transfer == ColorTransfer::Linear ? source : transfer == ColorTransfer::Srgb ? (source <= .0031308 ? 12.92 * source : 1.055 * std::pow(source, 1. / 2.4) - .055)
                                                                                                                       : (source < .018 ? 4.5 * source : 1.099 * std::pow(source, .45) - .099);
                    if (std::abs(raw[pixel * 4 + c] - stored) > 2. / 65535.)
                        return 4;
                }
                if (rgba[pixel * 4 + 3] != std::lround(samples[pixel * 4 + 3] * 65535.))
                    return 5;
            }
        }
    }
    // External/untagged TIFFs retain raw legacy values; an unsupported explicit
    // LichtFeld contract must fail instead of silently assuming sRGB.
    const std::array<uint16_t, 3> legacy{1234, 23456, 54321};
    const auto legacy_bytes = std::span(reinterpret_cast<const uint8_t*>(legacy.data()), sizeof(legacy));
    if (!lfs::core::image_codecs::write_tiff_u16(path, legacy_bytes, 1, 1, 3, 6, std::nullopt, {.overwrite = true}))
        return 6;
    auto [unchanged, lw, lh, lc] = lfs::core::load_image_u16(path);
    std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> legacy_owner(unchanged, lfs::core::free_image);
    if (!unchanged || lw != 1 || lh != 1 || lc != 3 || !std::equal(legacy.begin(), legacy.end(), unchanged))
        return 7;
    if (!lfs::core::image_codecs::write_tiff_u16(path, legacy_bytes, 1, 1, 3, 6,
                                                 R"({"lfsEncoding":"tiff16-sdr-v99"})", {.overwrite = true}))
        return 8;
    try {
        auto [unexpected, uw, uh, uc] = lfs::core::load_image_u16(path);
        lfs::core::free_image(unexpected);
        return 9;
    } catch (const std::runtime_error&) {
        return 0;
    }
}
