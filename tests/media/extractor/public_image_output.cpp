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
    return !collision && collision.error().code() == lfs::ErrorCode::AlreadyExists && std::filesystem::file_size(path) > 100 ? 0 : 3;
}
