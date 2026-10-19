// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Deliberately include only the installed public media API and link lfs_media.
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <media/image_output.hpp>

int main() {
    using namespace lfs::media;
    const auto path = std::filesystem::temp_directory_path() /
                      ("lfs-public-exr-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".exr");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    } cleanup{path};
    const std::array<float, 3> samples{-.25f, .125f, 2.f};
    std::array<uint8_t, sizeof(samples)> pixels{};
    std::memcpy(pixels.data(), samples.data(), pixels.size());
    FrameView view{{1, 1, pixels.size(), FramePixelFormat::RGBFloat32, {ColorTransfer::Linear, ColorPrimaries::Bt709, AlphaMode::None}}, {}, pixels};
    view.info.origin = FrameOrigin::External;
    auto surface = FrameSurface::copyOf(view);
    if (!surface)
        return 1;
    ExrOutputOptions options;
    options.precision = ExrPrecision::Float;
    auto result = ImageOutput::writeExr(path, surface->view(), options);
    if (!result) {
        std::cerr << result.error().detail();
        return 2;
    }
    auto collision = ImageOutput::writeExr(path, view, options);
    if (collision || collision.error().code() != lfs::ErrorCode::AlreadyExists || std::filesystem::file_size(path) <= 100)
        return 3;
    auto tiff_path = path;
    tiff_path.replace_extension(".tif");
    Cleanup tiff_cleanup{tiff_path};
    Tiff16OutputOptions tiff_options;
    auto tiff_result = ImageOutput::writeTiff16(tiff_path, surface->view(), tiff_options);
    if (!tiff_result) {
        std::cerr << tiff_result.error().detail();
        return 4;
    }
    auto tiff_collision = ImageOutput::writeTiff16(tiff_path, view, tiff_options);
    if (tiff_collision || tiff_collision.error().code() != lfs::ErrorCode::AlreadyExists || std::filesystem::file_size(tiff_path) <= 100)
        return 5;
    auto jxl_path = path;
    jxl_path.replace_extension(".jxl");
    Cleanup jxl_cleanup{jxl_path};
    const JxlOutputOptions legacy_jxl_options{true, 1.f, 4, false, {}};
    static_cast<void>(legacy_jxl_options);
    auto jxl_result = ImageOutput::writeJxl(jxl_path, surface->view());
    if (!jxl_result) {
        std::cerr << jxl_result.error().detail();
        return 6;
    }
    auto jxl_collision = ImageOutput::writeJxl(jxl_path, surface->view());
    if (jxl_collision || jxl_collision.error().code() != lfs::ErrorCode::AlreadyExists || std::filesystem::file_size(jxl_path) <= 10)
        return 7;
    auto avif_path = path;
    avif_path.replace_extension(".avif");
    Cleanup avif_cleanup{avif_path};
    auto avif_result = ImageOutput::writeAvif(avif_path, surface->view());
    if (!avif_result) {
        std::cerr << avif_result.error().detail();
        return 8;
    }
    auto avif_collision = ImageOutput::writeAvif(avif_path, view);
    return !avif_collision && avif_collision.error().code() == lfs::ErrorCode::AlreadyExists && std::filesystem::file_size(avif_path) > 10 ? 0 : 9;
}
