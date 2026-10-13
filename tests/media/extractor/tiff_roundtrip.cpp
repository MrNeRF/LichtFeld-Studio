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
#include <tiffio.h>

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
    const std::array<float, 40> samples{
        .125f, .25f, .5f, .375f, .35f, .3f, .2f, .75f,
        .000001f, .0001f, .001f, 1.f,
        .0031307f, .0031308f, .0031309f, .5f,
        .01799f, .018f, .01801f, 1.f,
        1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, .5f,
        0.f, 0.f, 1.f, 0.f, -.1f, 1.1f, .4f, 1.f,
        1.f, 1.f, 1.f, 1.f};
    constexpr int pixel_count = int(samples.size() / 4);
    auto png_path = path;
    png_path.replace_extension(".png");
    Cleanup png_cleanup{png_path};
    std::array<uint8_t, sizeof(samples)> bytes{};
    std::memcpy(bytes.data(), samples.data(), bytes.size());
    for (auto primaries : {ColorPrimaries::Bt709, ColorPrimaries::Bt2020}) {
        for (auto transfer : {ColorTransfer::Linear, ColorTransfer::Srgb, ColorTransfer::Bt709}) {
            FrameView frame{{pixel_count, 1, bytes.size(), FramePixelFormat::RGBAFloat32, {ColorTransfer::Linear, primaries, AlphaMode::Independent}}, {}, bytes};
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
            if (!rgb || !rgba || !rgb8 || !raw || width != pixel_count || height != 1 || channels != 3 || aw != pixel_count || ah != 1 || ac != 4 || bw != pixel_count || bh != 1 || bc != 3 || rw != pixel_count || rh != 1 || rc != 4)
                return 2;
            for (size_t pixel = 0; pixel < pixel_count; ++pixel) {
                std::array<double, 3> linear{};
                for (size_t c = 0; c < 3; ++c) {
                    const double value = raw[pixel * 4 + c];
                    // Start from stored quantized values, then use an independent
                    // double EOTF: source quantization near black is amplified by sRGB.
                    linear[c] = transfer == ColorTransfer::Linear ? value : transfer == ColorTransfer::Srgb ? (value <= .04045 ? value / 12.92 : std::pow((value + .055) / 1.055, 2.4))
                                                                                                            : (value < .081 ? value / 4.5 : std::pow((value + .099) / 1.099, 1. / .45));
                }
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
                    const double source = std::clamp(double(samples[pixel * 4 + c]), 0., 1.);
                    // Match the float API's representable branch boundaries;
                    // BT.709's rounded constants have a discontinuity at .018.
                    const double stored = transfer == ColorTransfer::Linear ? source : transfer == ColorTransfer::Srgb ? (source <= double(.0031308f) ? 12.92 * source : 1.055 * std::pow(source, 1. / 2.4) - .055)
                                                                                                                       : (source < double(.018f) ? 4.5 * source : 1.099 * std::pow(source, .45) - .099);
                    if (std::abs(raw[pixel * 4 + c] - stored) > 2. / 65535.)
                        return 4;
                }
                if (rgba[pixel * 4 + 3] != std::lround(samples[pixel * 4 + 3] * 65535.))
                    return 5;
            }
            if (!lfs::core::save_png(png_path, rgba, width, height, 4, 16, 6))
                return 10;
            for (const auto geometry : {std::array{2, 0}, std::array{1, 3}}) {
                auto [a, w, h, c] = lfs::core::load_image_u16(path, geometry[0], geometry[1]);
                std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> owner_a(a, lfs::core::free_image);
                auto [b, pw, ph, pc] = lfs::core::load_image_u16(png_path, geometry[0], geometry[1]);
                std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> owner_b(b, lfs::core::free_image);
                if (!a || !b || w != pw || h != ph || c != pc || !std::equal(a, a + w * h * c, b))
                    return 11;
                auto [aa, wa, ha, ca] = lfs::core::load_image_with_alpha_u16(path, geometry[0], geometry[1]);
                std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> owner_aa(aa, lfs::core::free_image);
                auto [ba, pwa, pha, pca] = lfs::core::load_image_with_alpha_u16(png_path, geometry[0], geometry[1]);
                std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> owner_ba(ba, lfs::core::free_image);
                if (!aa || !ba || wa != pwa || ha != pha || ca != pca || !std::equal(aa, aa + wa * ha * ca, ba))
                    return 12;
            }
        }
    }
    // Only a supported, well-formed encoding opts into color management.
    const std::array<uint16_t, 3> legacy{1234, 23456, 54321};
    const auto legacy_bytes = std::span(reinterpret_cast<const uint8_t*>(legacy.data()), sizeof(legacy));
    if (!lfs::core::image_codecs::write_tiff_u16(path, legacy_bytes, 1, 1, 3, 6, std::nullopt, {.overwrite = true}))
        return 6;
    auto [unchanged, lw, lh, lc] = lfs::core::load_image_u16(path);
    std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> legacy_owner(unchanged, lfs::core::free_image);
    if (!unchanged || lw != 1 || lh != 1 || lc != 3 || !std::equal(legacy.begin(), legacy.end(), unchanged))
        return 7;
    std::vector<uint16_t> table(65536);
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = uint16_t(i);
    lfs::core::image_codecs::TiffColorTags tags{
        .chromaticities = std::array<float, 8>{.64f, .33f, .30f, .60f, .15f, .06f, .3127f, .3290f},
        .transfer_function = table};
    const std::string valid = R"({"lfsEncoding":"tiff16-sdr-v1","lfsTransfer":"linear","lfsPrimaries":"bt709","lfsUnits":"relative","lfsAlpha":"none"})";
    const auto rejects = [&]() {
        try {
            auto [unexpected, uw, uh, uc] = lfs::core::load_image_u16(path);
            lfs::core::free_image(unexpected);
            return false;
        } catch (const std::runtime_error&) { return true; }
    };
    for (const auto description : {
             R"({"lfsEncoding":"tiff16-sdr-v99"})",
             R"({"lfsEncoding":"tiff16-sdr-v1",)",
             R"({"lfsTransfer":"linear","lfsPrimaries":"bt709","lfsUnits":"relative"})",
             R"({"lfsEncoding":7,"lfsTransfer":"linear"})",
             R"(unstructured description containing "lfsTransfer" and "lfsEncoding")",
             R"([{"lfsEncoding":"tiff16-sdr-v1"}])"}) {
        if (!lfs::core::image_codecs::write_tiff_u16(path, legacy_bytes, 1, 1, 3, 6, description, {.overwrite = true}, tags))
            return 19;
        auto [foreign, fw, fh, fc] = lfs::core::load_image_u16(path);
        std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> foreign_owner(foreign, lfs::core::free_image);
        if (!foreign || fw != 1 || fh != 1 || fc != 3 || !std::equal(legacy.begin(), legacy.end(), foreign))
            return 20;
    }
    // Inject oversized descriptions through libtiff, independently of the
    // export API's bounded metadata contract. Even an apparent owned marker
    // inside malformed/oversized text must preserve foreign TIFF pixels.
    for (const auto prefix : {"external description ", "{\"lfsEncoding\":\"tiff16-sdr-v1\",\"padding\":\""}) {
        if (!lfs::core::image_codecs::write_tiff_u16(path, legacy_bytes, 1, 1, 3, 6, std::nullopt, {.overwrite = true}))
            return 22;
        auto* tiff = TIFFOpen(path.string().c_str(), "r+");
        if (!tiff)
            return 23;
        std::string description = prefix;
        description.append(1024 * 1024, 'x');
        const bool written = TIFFSetField(tiff, TIFFTAG_IMAGEDESCRIPTION, description.c_str()) && TIFFRewriteDirectory(tiff);
        TIFFClose(tiff);
        if (!written)
            return 24;
        auto [foreign, fw, fh, fc] = lfs::core::load_image_u16(path);
        std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> owner(foreign, lfs::core::free_image);
        if (!foreign || fw != 1 || fh != 1 || fc != 3 || !std::equal(legacy.begin(), legacy.end(), foreign))
            return 25;
    }
    for (const auto description : {
             R"({"lfsEncoding":"tiff16-sdr-v1"})",
             R"({"lfsEncoding":"tiff16-sdr-v1","lfsTransfer":"srgb","lfsPrimaries":"bt709","lfsUnits":"relative","lfsAlpha":"none"})",
             R"({"lfsEncoding":"tiff16-sdr-v1","lfsTransfer":"linear","lfsPrimaries":"bt2020","lfsUnits":"relative","lfsAlpha":"none"})",
             R"({"lfsEncoding":"tiff16-sdr-v1","lfsTransfer":"linear","lfsPrimaries":"bt709","lfsUnits":"relative","lfsAlpha":"independent"})",
             R"({"lfsEncoding":"tiff16-sdr-v1","lfsTransfer":"linear","lfsTransfer":"linear","lfsPrimaries":"bt709","lfsUnits":"relative","lfsAlpha":"none"})"}) {
        if (!lfs::core::image_codecs::write_tiff_u16(path, legacy_bytes, 1, 1, 3, 6, description, {.overwrite = true}, tags))
            return 13;
        if (!rejects()) {
            std::cerr << "Accepted inconsistent metadata: " << description;
            return 14;
        }
    }
    if (!lfs::core::image_codecs::write_tiff_u16(path, legacy_bytes, 1, 1, 3, 6, valid, {.overwrite = true}))
        return 15;
    if (!rejects())
        return 16;
    table[65000] = 0;
    if (!lfs::core::image_codecs::write_tiff_u16(path, legacy_bytes, 1, 1, 3, 6, valid, {.overwrite = true}, tags))
        return 17;
    if (!rejects())
        return 18;
    for (const auto delta : {-1, 1}) {
        table[65000] = uint16_t(65000 + delta);
        if (!lfs::core::image_codecs::write_tiff_u16(path, legacy_bytes, 1, 1, 3, 6, valid, {.overwrite = true}, tags) || rejects())
            return 21;
    }
    return 0;
}
