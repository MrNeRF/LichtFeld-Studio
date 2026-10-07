// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "io/video/color_convert.cuh"
#include "io/video/video_encoder.hpp"
#include "media/video_player.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace {
    void require(bool value, const char* message) {
        if (!value)
            throw std::runtime_error(message);
    }
    struct CudaStream {
        cudaStream_t value = nullptr;
        CudaStream() {
            const auto status = cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking);
            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));
        }
        ~CudaStream() { cudaStreamDestroy(value); }
    };
    nlohmann::json conversionContracts() {
        using namespace lfs::core;
        CudaStream stream;
        std::size_t checked = 0;
        for (const auto [width, height] : std::array{std::pair{2, 2}, std::pair{34, 18}, std::pair{320, 240}}) {
            std::vector<float> rgb(static_cast<std::size_t>(width) * height * 3);
            std::uint32_t seed = 3003;
            for (auto& value : rgb) {
                seed = 1664525 * seed + 1013904223;
                value = static_cast<float>(seed & 65535) / 32768.0f - 0.5f;
            }
            // Probe both sides of every byte rounding boundary, plus nonfinite
            // values. This catches fused multiply/add changing quantization.
            std::vector<float> edges{-1.0f, 0.0f, 1.0f, 2.0f,
                                     std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity()};
            for (int byte = 0; byte < 255; ++byte) {
                const float boundary = (byte + .5f) / 255.0f;
                edges.push_back(std::nextafter(boundary, 0.0f));
                edges.push_back(boundary);
                edges.push_back(std::nextafter(boundary, 1.0f));
            }
            std::copy_n(edges.begin(), std::min(edges.size(), rgb.size()), rgb.begin());
            const auto byte = [](float value) {
                return std::floor(std::clamp(value, 0.0f, 1.0f) * 255.0f + .5f);
            };
            const auto output_byte = [](float value) {
                return std::isnan(value) ? std::uint8_t{0} : static_cast<std::uint8_t>(value);
            };
            std::array<std::vector<std::uint8_t>, 3> expected{
                std::vector<std::uint8_t>(width * height),
                std::vector<std::uint8_t>(width * height / 4),
                std::vector<std::uint8_t>(width * height / 4)};
            for (int row = 0; row < height; row += 2) {
                for (int col = 0; col < width; col += 2) {
                    std::array<float, 3> sum{};
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            const auto pixel = (row + dy) * width + col + dx;
                            const auto r = byte(rgb[3 * pixel]), g = byte(rgb[3 * pixel + 1]), b = byte(rgb[3 * pixel + 2]);
                            expected[0][pixel] = output_byte(std::floor((66 * r + 129 * g + 25 * b + 128) / 256) + 16);
                            sum[0] += r;
                            sum[1] += g;
                            sum[2] += b;
                        }
                    }
                    const auto r = std::floor(sum[0] / 4), g = std::floor(sum[1] / 4), b = std::floor(sum[2] / 4);
                    const auto chroma = (row / 2) * (width / 2) + col / 2;
                    expected[1][chroma] = output_byte(std::clamp(std::floor((-38 * r - 74 * g + 112 * b + 128) / 256) + 128, 0.0f, 255.0f));
                    expected[2][chroma] = output_byte(std::clamp(std::floor((112 * r - 94 * g - 18 * b + 128) / 256) + 128, 0.0f, 255.0f));
                }
            }
            auto input = Tensor::from_blob(rgb.data(), {static_cast<std::size_t>(height), static_cast<std::size_t>(width), 3},
                                           Device::CPU, DataType::Float32)
                             .gpu();
            require(cudaDeviceSynchronize() == cudaSuccess, "conversion input preparation");
            input.set_stream(stream.value);
            // Tail sentinels make incomplete blocks and the smallest extent
            // exercise bounds as well as the byte-exact color contract.
            std::array<Tensor, 3> output;
            for (int plane = 0; plane < 3; ++plane) {
                output[plane] = Tensor::empty_like(input, {expected[plane].size() + 64}, DataType::UInt8);
                output[plane].set_stream(stream.value);
                require(cudaMemsetAsync(output[plane].data_ptr(), 0xcd, output[plane].bytes(), stream.value) == cudaSuccess,
                        "conversion sentinel initialization");
            }
            lfs::io::video::rgbToYuv420pCuda(input.ptr<float>(), output[0].ptr<std::uint8_t>(),
                                             output[1].ptr<std::uint8_t>(), output[2].ptr<std::uint8_t>(),
                                             width, height, stream.value);
            require(cudaStreamSynchronize(stream.value) == cudaSuccess, "conversion stream completion");
            for (int plane = 0; plane < 3; ++plane) {
                const auto actual = output[plane].to_vector_uint8();
                require(std::equal(expected[plane].begin(), expected[plane].end(), actual.begin()),
                        "CUDA conversion differs from scalar Studio color reference");
                require(std::all_of(actual.begin() + expected[plane].size(), actual.end(), [](auto value) { return value == 0xcd; }),
                        "CUDA conversion wrote outside plane bounds");
                checked += expected[plane].size();
            }
        }
        return {{"success", true}, {"checked_bytes", checked}, {"extents", 3}, {"nondefault_stream", true}};
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
    if (request.at("operation") == "native-conversion")
        return conversionContracts();
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
        if (index == 1) {
            // A cropped view must be materialized before the flat CUDA kernel;
            // the adjacent black half detects an incorrect source row stride.
            tensor = core::Tensor::cat({tensor, core::Tensor::zeros_like(tensor)}, 1).slice(1, 0, 320);
            require(!tensor.is_contiguous(), "native encoder exercises strided RGB input");
        }
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
