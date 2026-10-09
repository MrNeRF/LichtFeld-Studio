// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/crash_handler.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/image_codecs.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_color.hpp"
#include "core/tensor_upload.hpp"
#include "io/image_output.hpp"
#include "io/video/video_encoder.hpp"
#include "media/video_color.hpp"
#include "media/video_player.hpp"
#include "visualizer/rendering/float_color_readback.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
namespace lfs::io {
    std::unique_ptr<media::detail::LinearVideoRenderer> createLinearTensorRenderer();
}
namespace {
    void videoOutputContracts(lfs::core::Device device, bool require_videotoolbox) {
        using namespace lfs;
        const auto path = std::filesystem::temp_directory_path() /
                          ("lfs-video-reuse-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".mp4");
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() {
                std::error_code ignored;
                std::filesystem::remove(path, ignored);
            }
        } cleanup{path};
        io::video::VideoEncoder encoder;
        for (const auto width : {320, 160}) {
            io::video::VideoExportOptions options;
            options.preset = io::video::VideoPreset::CUSTOM;
            options.width = width;
            options.height = 96;
            options.framerate = 10;
            const auto opened = encoder.open(path, options);
            if (!opened)
                throw std::runtime_error(opened.error());
            if (require_videotoolbox && encoder.backend() != media::VideoEncodeBackend::VideoToolbox)
                throw std::runtime_error("VideoToolbox contract silently fell back to software");
            std::cout << "video encoder backend=" << static_cast<int>(encoder.backend())
                      << ", extent=" << width << "x96\n";
            for (int index = 0; index < 12; ++index) {
                // Alternate CPU/GPU producers and strided views; reopening at a
                // different extent must also discard the previous plane cache.
                // Different channels expose layout mistakes; zero padding makes
                // an incorrectly packed read of this strided view observable.
                std::vector<float> values(static_cast<size_t>(96 * width * 2 * 3), 0.f);
                for (int row = 0; row < 96; ++row)
                    for (int column = 0; column < width; ++column)
                        for (int channel = 0; channel < 3; ++channel)
                            values[(row * width * 2 + column) * 3 + channel] =
                                (32 + index * 16 + (channel - 1) * 16) / 255.f;
                auto frame = core::Tensor::from_vector(values, {96, static_cast<size_t>(width) * 2, 3},
                                                       index < 6 ? device : core::Device::CPU)
                                 .slice(1, 0, width);
                const auto written = encoder.writeFrame(frame);
                if (!written)
                    throw std::runtime_error(written.error());
            }
            const auto closed = encoder.close();
            if (!closed || encoder.isOpen() || !encoder.close())
                throw std::runtime_error("Video producer did not close idempotently");
            io::VideoPlayer player;
            if (!player.open(path) || player.width() != width || player.height() != 96)
                throw std::runtime_error("Reopened video has incorrect extent");
            for (int index = 0; index < 12; ++index) {
                player.seek(index / 10.);
                const auto* pixels = player.currentFrameData();
                const auto count = static_cast<size_t>(width) * 96 * player.currentFrameChannels();
                if (!pixels || !count || !player.takeError().empty())
                    throw std::runtime_error("Cannot decode reused video planes");
                double error = 0;
                for (size_t pixel = 0; pixel < static_cast<size_t>(width) * 96; ++pixel)
                    for (int channel = 0; channel < 3; ++channel)
                        error += std::abs(int(pixels[pixel * player.currentFrameChannels() + channel]) - (32 + index * 16 + (channel - 1) * 16));
                if (error / (width * 96 * 3) >= 6)
                    throw std::runtime_error("Reused video planes contain stale or incorrect pixels: frame=" +
                                             std::to_string(index) + ", time=" + std::to_string(player.currentTime()) +
                                             ", first=" + std::to_string(pixels[0]) + ", mean error=" +
                                             std::to_string(error / (width * 96 * 3)));
            }
        }
    }

    void imageOutputContracts(lfs::core::Device device) {
        using namespace lfs;
        const auto path = std::filesystem::temp_directory_path() /
                          ("lfs-tensor-image-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".exr");
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() {
                std::error_code ignored;
                std::filesystem::remove(path, ignored);
            }
        } cleanup{path};
        const std::vector<float> values{-.25f, .125f, 2.f, .5f, .5f, 1.f, .25f, .25f};
        const auto host = core::Tensor::from_vector(values, {1, 2, 4}, core::Device::CPU);
        const media::FrameColor color{media::ColorTransfer::Linear, media::ColorPrimaries::Bt709, media::AlphaMode::Independent};
        media::FrameInfo info;
        info.origin = media::FrameOrigin::Rendered;
        for (const auto dtype : {core::DataType::Float32, core::DataType::Float16}) {
            const auto source = host.to(device).to(dtype);
            const auto captured = vis::linearFloatColorReadback(source, {.transparent = true});
            if (!captured)
                throw Exception(captured.error());
            const auto* linear = captured->ptr<float>();
            for (size_t pixel = 0; pixel < 2; ++pixel) {
                for (size_t channel = 0; channel < 3; ++channel) {
                    const double straight = values[pixel * 4 + channel] / values[pixel * 4 + 3];
                    const double expected = straight <= .04045 ? straight / 12.92 : std::pow((straight + .055) / 1.055, 2.4);
                    if (std::abs(linear[pixel * 4 + channel] - expected) > 2e-5)
                        throw std::runtime_error("Unquantized raster capture changes signed/high-range color or applies alpha twice");
                }
                if (linear[pixel * 4 + 3] != values[pixel * 4 + 3])
                    throw std::runtime_error("Unquantized raster capture changes coverage");
            }
            for (const auto precision : {media::ExrPrecision::Half, media::ExrPrecision::Float}) {
                media::ExrOutputOptions options;
                options.precision = precision;
                options.overwrite = true;
                auto result = io::writeExrImage(path, source, color, options, info);
                if (!result)
                    throw Exception(result.error());
                core::image_codecs::Image decoded;
                std::string error;
                if (!core::image_codecs::decode(path, decoded, error) || decoded.width != 2 || decoded.height != 1 ||
                    decoded.channels != 4 || decoded.sample_type != core::image_codecs::SampleType::Float32)
                    throw std::runtime_error("Tensor EXR did not retain dimensions/channels/float precision: " + error);
                for (size_t pixel = 0; pixel < 2; ++pixel)
                    for (size_t c = 0; c < 4; ++c) {
                        float actual;
                        std::memcpy(&actual, decoded.data.data() + (pixel * 4 + c) * sizeof(float), sizeof(float));
                        const auto expected = values[pixel * 4 + c] * (c == 3 ? 1.f : values[pixel * 4 + 3]);
                        if (actual != expected)
                            throw std::runtime_error("Tensor EXR loses signed/high-range samples or premultiplies alpha incorrectly");
                    }
            }
        }
        const auto collision = io::writeExrImage(path, host, color);
        if (collision || collision.error().code() != ErrorCode::AlreadyExists)
            throw std::runtime_error("Tensor EXR collision policy differs from shared writer");
        media::ExrOutputOptions cancelled;
        cancelled.overwrite = true;
        cancelled.cancelled = [] { return true; };
        const auto stopped = io::writeExrImage(path, host, color, cancelled);
        if (stopped || stopped.error().code() != ErrorCode::Cancelled)
            throw std::runtime_error("Tensor EXR did not cancel before readback");
        const auto invalid = io::writeExrImage(path, host.to(core::DataType::UInt8), color);
        if (invalid || invalid.error().code() != ErrorCode::InvalidArgument)
            throw std::runtime_error("Tensor EXR accepted a quantized byte image");
        auto display = color;
        display.transfer = media::ColorTransfer::Srgb;
        const auto rejected = io::writeExrImage(path, host, display);
        if (rejected || rejected.error().code() != ErrorCode::Unsupported)
            throw std::runtime_error("Tensor EXR silently relabelled display color as linear");
    }
} // namespace
int main(int argc, char** argv) {
    using namespace lfs;
    using namespace media::detail;
    try {
        const std::string_view name = argc > 1 ? argv[1] : "metal";
        const auto backend = name == "cuda" ? core::GpuBackend::CUDA : name == "vulkan" ? core::GpuBackend::Vulkan
                                                                                        : core::GpuBackend::Metal;
        const bool cpu = name == "cpu";
        const bool require_videotoolbox = argc > 2 && std::string_view(argv[2]) == "--require-videotoolbox";
        std::optional<core::GpuBackendScope> scope;
        if (!cpu) {
            const auto selected = core::set_default_gpu_backend(backend);
            if (!selected)
                throw std::runtime_error(std::string(selected.error().detail()));
            if (!core::gpu_backend_available(backend)) {
                std::cout << name << ": device unavailable\n";
                return 77;
            }
            scope.emplace(backend);
        }
        struct Shutdown {
            ~Shutdown() { core::teardown_gpu_before_exit(); }
        } shutdown;
        imageOutputContracts(cpu ? core::Device::CPU : core::Device::GPU);
        // CUDA builds select NVENC by default even for a CPU source. Keep the
        // CPU-only contract independent of hardware; GPU profiles also exercise
        // CPU sources above, while non-CUDA builds cover the CPU producer here.
        if (!cpu || !LFS_HAS_CUDA || require_videotoolbox)
            videoOutputContracts(cpu ? core::Device::CPU : core::Device::GPU, require_videotoolbox);
        auto renderer = cpu ? nullptr : io::createLinearTensorRenderer();
        if (!cpu && !renderer)
            throw std::runtime_error("Linear tensor op unavailable on an available GPU backend");
        const auto make_bytes = [](const std::vector<uint8_t>& source, core::Device device) {
            auto bytes = core::Tensor::empty({source.size()}, device, core::DataType::UInt8);
            if (device == core::Device::CPU) {
                std::memcpy(bytes.ptr<uint8_t>(), source.data(), source.size());
            } else {
                core::TensorUpload upload;
                upload.enqueue(bytes, std::as_bytes(std::span(source)), nullptr);
                upload.wait();
            }
            return bytes;
        };
        const auto convert = [&](const std::vector<uint8_t>& source, const VideoColorParameters& color, int width, int height, std::vector<uint8_t>& output) -> media::SinkResult {
            if (renderer)
                return renderer->convert(source, color, width, height, output);
            auto bytes = make_bytes(source, core::Device::CPU);
            auto result = core::video_to_linear_rgb(bytes, color, width, height);
            if (!result)
                return media::SinkResult::failure(std::move(result).error());
            if (result->device() != core::Device::CPU || result->dtype() != core::DataType::Float32 ||
                result->size(0) != size_t(height) || result->size(1) != size_t(width) || result->size(2) != 3 || result->bytes() != output.size())
                throw std::runtime_error("Tensor video output violates shape/dtype/device contract");
            std::memcpy(output.data(), result->ptr<float>(), output.size());
            return {};
        };
        VideoColorParameters color{};
        color.width = 8;
        color.height = 4;
        color.rgb = 1;
        for (uint32_t c = 0; c < 3; ++c) {
            color.component[c] = {c, 24, 3, 0, 8, 8, 4};
            color.scale[c] = 1.f / 255.f;
            color.decode[c * 3 + c] = 1;
        }
        std::vector<uint8_t> source(8 * 4 * 3);
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 8; ++x)
                for (int c = 0; c < 3; ++c)
                    source[(y * 8 + x) * 3 + c] = x % 4 == 0 || x % 4 == 3 ? 255 : 0;
        std::vector<uint8_t> output(2 * 1 * 12);
        auto result = convert(source, color, 2, 1, output);
        if (!result)
            throw std::runtime_error(std::string(result.error().detail()));
        for (size_t i = 0; i < output.size(); i += 4) {
            float value;
            std::memcpy(&value, output.data() + i, 4);
            if (std::abs(value - .5f) > 1e-6)
                throw std::runtime_error("GPU reduction aliases instead of averaging the linear checkerboard");
        }
        // Interleaved 4:2:0 with varied luma/chroma, limited range and BT709.
        // The CPU oracle below evaluates analytic code/range/matrix/transfer
        // equations independently of the sampler and tensor implementation.
        source.resize(48);
        color.rgb = 0;
        color.transfer = 2;
        color.component[0] = {0, 8, 1, 0, 8, 8, 4};
        color.component[1] = {32, 8, 2, 0, 8, 4, 2};
        color.component[2] = {33, 8, 2, 0, 8, 4, 2};
        color.chroma_x_scale = color.chroma_y_scale = .5f;
        color.chroma_y_offset = -.25f;
        const std::array matrix{1.f, 0.f, 1.5748f, 1.f, -0.187324273f, -0.468124273f, 1.f, 1.8556f, 0.f};
        std::copy(matrix.begin(), matrix.end(), color.decode);
        color.scale[0] = 1.f / 219;
        color.bias[0] = 16.f / 219;
        color.scale[1] = color.scale[2] = 1.f / 224;
        color.bias[1] = color.bias[2] = 128.f / 224;
        for (int i = 0; i < 32; ++i)
            source[i] = 16 + (i * 19) % 220;
        for (int i = 0; i < 8; ++i) {
            source[32 + 2 * i] = 40 + (i * 23) % 190;
            source[33 + 2 * i] = 40 + (i * 31) % 190;
        }
        output.resize(8 * 4 * 12);
        result = convert(source, color, 8, 4, output);
        if (!result)
            throw std::runtime_error(std::string(result.error().detail()));
        const auto chroma = [&](int channel, int x, int y) {
            const double sx = x * .5, sy = y * .5 - .25;
            const int ix = static_cast<int>(std::floor(sx)), iy = static_cast<int>(std::floor(sy));
            const double fx = sx - ix, fy = sy - iy;
            const auto code = [&](int xx, int yy) { return double(source[32 + std::clamp(yy, 0, 1) * 8 + std::clamp(xx, 0, 3) * 2 + channel]); };
            return ((1 - fx) * code(ix, iy) + fx * code(ix + 1, iy)) * (1 - fy) + ((1 - fx) * code(ix, iy + 1) + fx * code(ix + 1, iy + 1)) * fy;
        };
        double maximum_error = 0;
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 8; ++x) {
                const double l = (source[y * 8 + x] - 16.0) / 219, u = (chroma(0, x, y) - 128) / 224, v = (chroma(1, x, y) - 128) / 224;
                const std::array encoded{l + 1.5748 * v, l - 2 * .0722 * (1 - .0722) / .7152 * u - 2 * .2126 * (1 - .2126) / .7152 * v, l + 1.8556 * u};
                for (int c = 0; c < 3; ++c) {
                    float actual;
                    std::memcpy(&actual, output.data() + ((y * 8 + x) * 3 + c) * 4, 4);
                    const double value = encoded[c], expected = value < .081 ? value / 4.5 : std::pow((value + .099) / 1.099, 1 / .45);
                    maximum_error = std::max(maximum_error, std::abs(actual - expected));
                }
            }
        if (maximum_error > 3e-6)
            throw std::runtime_error("GPU 4:2:0 differs from the independent linear reference: " + std::to_string(maximum_error));
        // Packed 10-bit RGB must retain high shifted components (X2RGB10).
        color = {};
        color.width = color.height = 1;
        color.rgb = 1;
        for (uint32_t c = 0; c < 3; ++c) {
            color.component[c] = {0, 4, 4, (2u - c) * 10, 10, 1, 1};
            color.scale[c] = 1.f / 1023;
            color.decode[c * 3 + c] = 1;
        }
        const uint32_t packed = (1023u << 20) | (512u << 10) | 17u;
        source.resize(4);
        std::memcpy(source.data(), &packed, 4);
        output.resize(12);
        result = convert(source, color, 1, 1, output);
        if (!result)
            throw std::runtime_error(std::string(result.error().detail()));
        const std::array expected{1.f, 512.f / 1023, 17.f / 1023};
        for (size_t c = 0; c < 3; ++c) {
            float actual;
            std::memcpy(&actual, output.data() + c * 4, 4);
            if (std::abs(actual - expected[c]) > 1e-6)
                throw std::runtime_error("Packed RGB10 loses shifted components");
        }
        auto bytes = make_bytes(source, cpu ? core::Device::CPU : core::Device::GPU);
        auto tensor = core::video_to_linear_rgb(bytes, color, 1, 1);
        if (!tensor || tensor->device() != bytes.device() || core::gpu_backend_of(*tensor) != core::gpu_backend_of(bytes))
            throw std::runtime_error("Tensor video op does not preserve input device/backend");
        const auto reject = [&](const auto& input, const VideoColorParameters& invalid, uint32_t width = 1) {
            const auto rejected = core::video_to_linear_rgb(input, invalid, width, 1);
            if (rejected || rejected.error().code() != ErrorCode::InvalidArgument)
                throw std::runtime_error("Invalid tensor video layout was not rejected before dispatch");
        };
        auto invalid = color;
        invalid.component[2].offset = uint32_t(source.size());
        reject(bytes, invalid);
        invalid = color;
        invalid.component[1].shift = 32;
        reject(bytes, invalid);
        invalid = color;
        invalid.component[0].pitch = 0;
        reject(bytes, invalid);
        invalid = color;
        invalid.decode[0] = std::numeric_limits<float>::quiet_NaN();
        reject(bytes, invalid);
        reject(bytes, color, UINT32_MAX);
        auto wrong_dtype = core::Tensor::empty({source.size()}, core::Device::CPU, core::DataType::Float32);
        reject(wrong_dtype, color);
        std::cout << name << ": linear video contracts passed, maximum error=" << maximum_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
