// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "io/video/video_encoder.hpp"
#include "media/video_player.hpp"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace {
    void require(bool value, const char* message) {
        if (!value)
            throw std::runtime_error(message);
    }
    struct CudaWriter final : lfs::media::VideoEncodeWriter {
        std::uint8_t luma = 32;
        lfs::Result<void> write(const lfs::media::VideoEncodeTarget& target) override {
            require(target.backend == lfs::media::VideoEncodeBackend::Cuda &&
                        target.layout == lfs::media::VideoEncodeLayout::NV12,
                    "native session writer requires NV12 CUDA planes");
            for (int plane = 0; plane < 2; ++plane) {
                const auto& pixels = target.planes[plane];
                // FFmpeg owns its CUDA context. Upload through the same UVA
                // transfer path used by Studio's producer, rather than use a
                // runtime memset on an allocation from another context.
                const std::vector<std::uint8_t> source(pixels.width * pixels.height, plane == 0 ? luma : 128);
                const auto status = cudaMemcpy2D(pixels.data, pixels.row_stride, source.data(), pixels.width,
                                                 pixels.width, pixels.height, cudaMemcpyHostToDevice);
                if (status != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorString(status));
            }
            const auto status = cudaDeviceSynchronize();
            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));
            return {};
        }
    };
} // namespace

nlohmann::json runNativeVideoContracts(const nlohmann::json& request) {
    using namespace lfs;
    if (request.at("operation") == "native-encode-session") {
        media::VideoEncodeSession session;
        media::VideoEncodeOptions options{.width = 320, .height = 240, .framerate = 10, .preferred_backend = media::VideoEncodeBackend::Cuda};
        auto opened = session.open(core::utf8_to_path(request.at("output").get<std::string>()), options);
        if (!opened)
            throw std::runtime_error(std::string(opened.error().detail()));
        require(session.backend() == media::VideoEncodeBackend::Cuda, "native session silently fell back from NVENC");
        CudaWriter writer;
        for (int index = 0; index < 12; ++index) {
            writer.luma = static_cast<std::uint8_t>(32 + index * 16);
            auto written = session.writeFrame(writer);
            if (!written)
                throw std::runtime_error(std::string(written.error().detail()));
        }
        auto closed = session.close();
        if (!closed)
            throw std::runtime_error(std::string(closed.error().detail()));
        return {{"success", true}, {"backend", "nvenc"}, {"frames", 12}};
    }
    if (request.at("operation") == "native-preview") {
        io::VideoPlayer player;
        require(player.open(core::utf8_to_path(request.at("input").get<std::string>())), "native player open");
        require(player.hardwareDecodeActive(), "native player silently fell back to software");
        player.seek(.3);
        player.seek(.1);
        require(player.takeError().empty(), "native player seek");
        const auto* pixels = player.currentFrameData();
        const auto size = static_cast<std::size_t>(player.width()) * player.height() * player.currentFrameChannels();
        require(pixels && size, "native player frame");
        nlohmann::json result{{"success", true}, {"hardware_decode", true}, {"time", player.currentTime()}, {"size", {player.width(), player.height()}}, {"pixels", std::vector<std::uint8_t>(pixels, pixels + size)}};
        player.close();
        require(!player.isOpen() && !player.hardwareDecodeActive() && !player.currentFrameData(), "native player close");
        return result;
    }
    io::video::VideoExportOptions options;
    options.preset = io::video::VideoPreset::CUSTOM;
    options.width = 320;
    options.height = 240;
    options.framerate = 10;
    options.crf = 18;
    io::video::VideoEncoder encoder;
    auto opened = encoder.open(core::utf8_to_path(request.at("output").get<std::string>()), options);
    if (!opened)
        throw std::runtime_error(opened.error());
    require(encoder.backend() == media::VideoEncodeBackend::Cuda, "native encoder silently fell back to software");
    // Exercise all existing Studio producers through the shared session.
    std::vector<std::uint8_t> rgba(options.width * options.height * 4, 64);
    auto first = encoder.writeFrame(rgba, options.width, options.height);
    if (!first)
        throw std::runtime_error(first.error());
    for (int index = 1; index < 3; ++index) {
        auto tensor = core::Tensor::ones({240, 320, 3}, index == 1 ? core::Device::GPU : core::Device::CPU)
                          .mul(static_cast<float>(64 + index * 64) / 255.0f);
        auto written = encoder.writeFrame(tensor);
        if (!written)
            throw std::runtime_error(written.error());
    }
    auto closed = encoder.close();
    if (!closed)
        throw std::runtime_error(closed.error());
    require(!encoder.isOpen() && encoder.close().has_value(), "native encoder close");
    return {{"success", true}, {"backend", "nvenc"}, {"frames", 3}};
}
