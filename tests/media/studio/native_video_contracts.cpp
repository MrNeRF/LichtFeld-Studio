// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "core/tensor_color.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "io/video/color_convert.cuh"
#include "io/video/video_encoder.hpp"
#include "media/cuda_frame_ffmpeg.hpp"
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
    nlohmann::json semiplanarColorContracts() {
        using namespace lfs::core;
        constexpr int width = 32, height = 8, pitch = width * 2 + 16;
        CudaStream stream;
        std::vector<uint16_t> y(pitch / 2 * height, 0xffff), uv(pitch / 2 * height / 2, 0xffff);
        const std::array<int, 9> codes{0, 64, 65, 66, 67, 128, 512, 940, 1023};
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
                y[row * pitch / 2 + col] = static_cast<uint16_t>((codes[col % codes.size()] << 6) | 63);
        for (int row = 0; row < height / 2; ++row)
            for (int col = 0; col < width; col += 2) {
                uv[row * pitch / 2 + col] = static_cast<uint16_t>(((row == 0 ? 512 : 64 + col * 20) << 6) | 63);
                uv[row * pitch / 2 + col + 1] = static_cast<uint16_t>(((row == 0 ? 512 : 940 - col * 20) << 6) | 63);
            }
        auto dy = Tensor::empty({y.size() * 2}, Device::GPU, DataType::UInt8);
        auto duv = Tensor::empty({uv.size() * 2}, Device::GPU, DataType::UInt8);
        auto rgb = Tensor::empty({width * height * 3 + 64}, Device::GPU, DataType::UInt8);
        require(cudaMemcpy(dy.data_ptr(), y.data(), y.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess, "P010 luma upload");
        require(cudaMemcpy(duv.data_ptr(), uv.data(), uv.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess, "P010 chroma upload");
        AVCUDADeviceContext cuda{};
        cuda.cuda_ctx = reinterpret_cast<CUcontext>(uintptr_t{1});
        AVHWDeviceContext device{};
        device.type = AV_HWDEVICE_TYPE_CUDA;
        device.hwctx = &cuda;
        AVHWFramesContext frames{};
        frames.format = AV_PIX_FMT_CUDA;
        frames.sw_format = AV_PIX_FMT_P010LE;
        frames.device_ctx = &device;
        AVBufferRef buffer{};
        buffer.data = reinterpret_cast<uint8_t*>(&frames);
        AVFrame frame{};
        frame.format = AV_PIX_FMT_CUDA;
        frame.width = width;
        frame.height = height;
        frame.hw_frames_ctx = &buffer;
        frame.data[0] = static_cast<uint8_t*>(dy.data_ptr());
        frame.data[1] = static_cast<uint8_t*>(duv.data_ptr());
        frame.linesize[0] = frame.linesize[1] = pitch;
        frame.chroma_location = AVCHROMA_LOC_LEFT;
        int checked = 0;
        for (const auto matrix : {AVCOL_SPC_SMPTE170M, AVCOL_SPC_BT709, AVCOL_SPC_BT2020_NCL}) {
            frame.colorspace = matrix;
            const double kr = matrix == AVCOL_SPC_BT709 ? .2126 : matrix == AVCOL_SPC_BT2020_NCL ? .2627
                                                                                                 : .299;
            const double kb = matrix == AVCOL_SPC_BT709 ? .0722 : matrix == AVCOL_SPC_BT2020_NCL ? .0593
                                                                                                 : .114;
            for (const bool full : {false, true}) {
                frame.color_range = full ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
                auto view = lfs::media::detail::cudaFrameView(&frame);
                require(view.has_value() && view->layout == lfs::media::detail::CudaVideoLayout::P010, "P010 descriptor");
                require(cudaMemset(rgb.data_ptr(), 0xcd, rgb.bytes()) == cudaSuccess, "P010 output sentinel");
                lfs::io::video::semiplanarToRgbCuda(frame.data[0], frame.data[1], static_cast<uint8_t*>(rgb.data_ptr()),
                                                    width, height, pitch, pitch, true, view->color, stream.value);
                require(cudaStreamSynchronize(stream.value) == cudaSuccess, "P010 nondefault stream completion");
                const auto actual = rgb.to_vector_uint8();
                for (int row = 0; row < height; ++row)
                    for (int col = 0; col < width; ++col) {
                        const double yy = ((y[row * pitch / 2 + col] >> 6) - (full ? 0.0 : 64.0)) / (full ? 1023.0 : 876.0);
                        const double sy = (row + .5) / 2 - .5;
                        const int first = static_cast<int>(std::floor(sy));
                        const double weight = sy - first;
                        const auto chroma = [&](int channel) {
                            const auto at = [&](int r) { return uv[std::clamp(r, 0, height / 2 - 1) * pitch / 2 + (col / 2) * 2 + channel] >> 6; };
                            return (at(first) * (1 - weight) + at(first + 1) * weight - 512.0) / (full ? 1023.0 : 896.0);
                        };
                        const double u = chroma(0), v = chroma(1);
                        const std::array<double, 3> expected{yy + 2 * (1 - kr) * v,
                                                             yy - 2 * (1 - kb) * kb / (1 - kr - kb) * u - 2 * (1 - kr) * kr / (1 - kr - kb) * v,
                                                             yy + 2 * (1 - kb) * u};
                        for (int channel = 0; channel < 3; ++channel) {
                            const int byte = static_cast<int>(std::floor(std::clamp(expected[channel], 0.0, 1.0) * 255 + .5));
                            require(std::abs(int(actual[(row * width + col) * 3 + channel]) - byte) <= 1, "P010 differs from scalar reference");
                            ++checked;
                        }
                    }
                if (!full)
                    require(actual[4 * 3] == 1 && actual[4 * 3 + 1] == 1 && actual[4 * 3 + 2] == 1,
                            "P010 lower two meaningful bits must survive until RGB quantization");
                require(std::all_of(actual.begin() + width * height * 3, actual.end(), [](auto b) { return b == 0xcd; }), "P010 output bounds");
            }
        }
        frame.linesize[0] = width;
        require(!lfs::media::detail::cudaFrameView(&frame), "P010 rejects byte pitch sized for NV12");
        frame.linesize[0] = pitch + 1;
        require(!lfs::media::detail::cudaFrameView(&frame), "P010 rejects odd UINT16 pitch");
        frame.linesize[0] = pitch;
        frame.colorspace = AVCOL_SPC_BT2020_CL;
        require(!lfs::media::detail::cudaFrameColorSupported(&frame, AVCOL_SPC_BT709), "constant-luminance matrix uses fallback");
        frame.colorspace = AVCOL_SPC_BT709;
        frame.chroma_location = AVCHROMA_LOC_CENTER;
        require(!lfs::media::detail::cudaFrameView(&frame), "unsupported chroma location uses fallback");
        frame.chroma_location = AVCHROMA_LOC_LEFT;
        frames.sw_format = AV_PIX_FMT_YUV420P10LE;
        require(!lfs::media::detail::cudaFrameView(&frame), "planar ten-bit storage must not be read as P010");
        return {{"success", true}, {"checked_components", checked}, {"nondefault_stream", true}};
    }
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
            CUDAStreamGuard execution(stream.value);
            std::array<Tensor, 3> output;
            Yuv420Planes planes;
            for (int plane = 0; plane < 3; ++plane) {
                output[plane] = Tensor::empty_like(input, {expected[plane].size() + 64}, DataType::UInt8);
                output[plane].set_stream(stream.value);
                require(cudaMemsetAsync(output[plane].data_ptr(), 0xcd, output[plane].bytes(), stream.value) == cudaSuccess, "conversion sentinel initialization");
            }
            planes.y = output[0].slice(0, 0, width * height).reshape({height, width});
            planes.u = output[1].slice(0, 0, width * height / 4).reshape({height / 2, width / 2});
            planes.v = output[2].slice(0, 0, width * height / 4).reshape({height / 2, width / 2});
            require(rgb_to_yuv420p_into(input, planes).has_value(), "tensor conversion dispatch");
            const auto host = Tensor::from_blob(rgb.data(), {static_cast<size_t>(height), static_cast<size_t>(width), 3}, Device::CPU, DataType::Float32);
            auto cpu = rgb_to_yuv420p(host);
            require(cpu.has_value(), "CPU conversion dispatch");
            require(cpu->y.to_vector_uint8() == expected[0] && cpu->u.to_vector_uint8() == expected[1] && cpu->v.to_vector_uint8() == expected[2], "CPU conversion differs from scalar reference");
#if LFS_TENSOR_VULKAN
            {
                GpuBackendScope backend(GpuBackend::Vulkan);
                auto vk = rgb_to_yuv420p(host.gpu());
                require(vk.has_value(), "Vulkan conversion dispatch");
                require(vk->y.to_vector_uint8() == expected[0] && vk->u.to_vector_uint8() == expected[1] && vk->v.to_vector_uint8() == expected[2], "Vulkan conversion differs from scalar reference");
            }
#endif
            require(cudaStreamSynchronize(stream.value) == cudaSuccess, "conversion stream completion");
            for (int plane = 0; plane < 3; ++plane) {
                const auto actual = output[plane].to_vector_uint8();
                require(std::equal(expected[plane].begin(), expected[plane].end(), actual.begin()),
                        "CUDA conversion differs from scalar Studio color reference");
                require(actual.size() == expected[plane].size() + 64, "conversion output storage extent");
                require(std::all_of(actual.begin() + expected[plane].size(), actual.end(), [](auto value) { return value == 0xcd; }), "tensor conversion wrote outside visible plane bounds");
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
    if (request.at("operation") == "native-yuv-color")
        return semiplanarColorContracts();
    if (request.at("operation") == "native-conversion")
        return conversionContracts();
    if (request.at("operation") == "native-encode-session") {
        media::VideoEncodeSession session;
        media::VideoEncodeOptions options{.width = 320, .height = 240, .framerate = 10, .preferred_backend = media::VideoEncodeBackend::Cuda};
        options.width = request.value("width", options.width);
        options.height = request.value("height", options.height);
        options.spherical_video = request.value("spherical", false);
        if (request.value("hevc", false))
            options.codec = media::VideoEncodeCodec::HEVC;
        auto opened = session.open(core::utf8_to_path(request.at("output").get<std::string>()), options);
        if (!opened)
            throw std::runtime_error(std::string(opened.error().detail()));
        require(session.backend() == media::VideoEncodeBackend::Cuda, "native session silently fell back from NVENC");
        CudaWriter writer;
        const int frames = request.value("frames", 12);
        for (int index = 0; index < frames; ++index) {
            writer.luma = static_cast<std::uint8_t>(32 + index * 16);
            auto written = session.writeFrame(writer);
            if (!written)
                throw std::runtime_error(std::string(written.error().detail()));
        }
        auto closed = session.close();
        if (!closed)
            throw std::runtime_error(std::string(closed.error().detail()));
        return {{"success", true}, {"backend", "nvenc"}, {"frames", frames}};
    }
    if (request.at("operation") == "native-preview") {
        struct DecodeObservation {
            bool hardware = false;
            core::LogLevel previous = core::Logger::get().level();
            core::LogHandlerToken token;
            DecodeObservation() : token(core::Logger::get().add_log_handler([this](auto, const auto&, std::string_view message) { if(message.starts_with("VideoPlayer: NVDEC decoder:")) hardware=true; })) { core::Logger::get().set_level(core::LogLevel::Info); }
            ~DecodeObservation() {
                core::Logger::get().remove_log_handler(token);
                core::Logger::get().set_level(previous);
            }
        } observed;
        io::VideoPlayer player;
        require(player.open(core::utf8_to_path(request.at("input").get<std::string>())), "native player open");
        require(observed.hardware, "native player silently fell back to software");
        player.seek(.3);
        player.seek(.1);
        require(player.takeError().empty(), "native player seek");
        const auto* pixels = player.currentFrameData();
        const auto size = static_cast<std::size_t>(player.width()) * player.height() * player.currentFrameChannels();
        require(pixels && size, "native player frame");
        nlohmann::json result{{"success", true}, {"hardware_decode", true}, {"time", player.currentTime()}, {"size", {player.width(), player.height()}}, {"pixels", std::vector<std::uint8_t>(pixels, pixels + size)}};
        return result;
    }
    io::video::VideoExportOptions options;
    options.preset = io::video::VideoPreset::CUSTOM;
    options.width = 320;
    options.height = 240;
    options.framerate = 10;
    options.crf = 18;
    if (request.value("hevc", false))
        options.codec = media::VideoEncodeCodec::HEVC;
    CudaStream stream;
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
    for (int index = 1; index < 12; ++index) {
        auto tensor = core::Tensor::ones({240, 320, 3}, index == 11 ? core::Device::CPU : core::Device::GPU)
                          .mul(static_cast<float>(64 + index * 12) / 255.0f);
        if (index % 2 == 1) {
            // A cropped view must be materialized before the flat CUDA kernel;
            // the adjacent black half detects an incorrect source row stride.
            tensor = core::Tensor::cat({tensor, core::Tensor::zeros_like(tensor)}, 1).slice(1, 0, 320);
            require(!tensor.is_contiguous(), "native encoder exercises strided RGB input");
        }
        // Reuse planes for several frames, then change streams and return to
        // the default stream. Distinct decoded luma detects stale/racing reuse.
        const auto execution_stream = index >= 4 && index < 8 ? stream.value : nullptr;
        if (execution_stream) {
            require(cudaDeviceSynchronize() == cudaSuccess, "encoder input preparation");
            tensor.set_stream(execution_stream);
        }
        core::CUDAStreamGuard execution(execution_stream);
        auto written = encoder.writeFrame(tensor);
        if (!written)
            throw std::runtime_error(written.error());
    }
    auto closed = encoder.close();
    if (!closed)
        throw std::runtime_error(closed.error());
    require(!encoder.isOpen() && encoder.close().has_value(), "native encoder close");
    return {{"success", true}, {"backend", "nvenc"}, {"frames", 12}};
}
