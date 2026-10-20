// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/color_transfer.h"
#include "core/crash_handler.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/image_codecs.hpp"
#include "core/image_io.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_color.hpp"
#include "core/tensor_upload.hpp"
#include "io/cache_image_loader.hpp"
#include "io/image_output.hpp"
#include "io/pipelined_image_loader.hpp"
#include "io/video/video_encoder.hpp"
#include "media/media_probe.hpp"
#include "media/video_color.hpp"
#include "media/video_player.hpp"
#include "visualizer/rendering/float_color_readback.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
namespace lfs::io {
    std::unique_ptr<media::detail::LinearVideoRenderer> createLinearTensorRenderer();
}
namespace {
    void tiffPipelineContracts() {
        using namespace lfs;
        constexpr int width = 16, height = 8, channels = 4;
        const auto path = std::filesystem::temp_directory_path() /
                          ("lfs-tiff-pipeline-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tif");
        auto png = path;
        png.replace_extension(".png");
        auto& cache = io::CacheLoader::getInstance(true);
        struct Cleanup {
            std::filesystem::path tiff, png;
            io::CacheLoader& cache;
            ~Cleanup() {
                cache.reset_cache();
                std::error_code error;
                std::filesystem::remove(tiff, error);
                std::filesystem::remove(png, error);
            }
        } cleanup{path, png, cache};
        std::vector<float> samples(width * height * channels);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const size_t offset = (y * width + x) * channels;
                samples[offset] = .05f + .5f * x / (width - 1);
                samples[offset + 1] = .1f + .3f * y / (height - 1);
                samples[offset + 2] = .2f + .1f * ((x + y) % 5) / 4;
                samples[offset + 3] = .375f;
            }
        const auto compare = [](const core::Tensor& actual, const core::Tensor& reference) {
            auto a = actual.to(core::Device::CPU).contiguous();
            auto b = reference.to(core::Device::CPU).contiguous();
            if (a.shape() != b.shape() || a.dtype() != b.dtype() || a.bytes() != b.bytes() ||
                std::memcmp(a.data_ptr(), b.data_ptr(), a.bytes()))
                throw std::runtime_error("TIFF pipeline differs from equivalent canonical PNG pixels");
        };
        for (auto primaries : {media::ColorPrimaries::Bt709, media::ColorPrimaries::Bt2020})
            for (auto transfer : {media::ColorTransfer::Linear, media::ColorTransfer::Srgb, media::ColorTransfer::Bt709}) {
                media::FrameView frame{{width, height, width * channels * sizeof(float), media::FramePixelFormat::RGBAFloat32, {media::ColorTransfer::Linear, primaries, media::AlphaMode::Independent}}, {}, {reinterpret_cast<const uint8_t*>(samples.data()), samples.size() * sizeof(float)}};
                media::Tiff16OutputOptions options;
                options.transfer = transfer;
                options.overwrite = true;
                const auto written = media::ImageOutput::writeTiff16(path, frame, options);
                if (!written)
                    throw std::runtime_error(std::string(written.error().detail()));
                auto [rgba, w, h, c] = core::load_image_with_alpha_u16(path);
                const std::unique_ptr<uint16_t, decltype(&core::free_image)> owned(rgba, core::free_image);
                if (!rgba || w != width || h != height || c != channels || !core::save_png(png, rgba, w, h, c, 16, 6))
                    throw std::runtime_error("TIFF pipeline canonical PNG fixture failed");
                cache.reset_cache();
                cache.update_cache_params(true, 2, 0.f, 0.f, false, 1);
                for (const auto params : {
                         io::LoadParams{},
                         io::LoadParams{.decode_16bit = true},
                         io::LoadParams{.resize_factor = 2, .output_uint8 = true, .decode_16bit = true},
                         io::LoadParams{.max_width = 5, .decode_16bit = true}}) {
                    const auto reference = io::load_rgb_image_cpu_decoded(png, params);
                    const auto direct = io::load_rgb_image_cpu_decoded(path, params);
                    compare(direct, reference);
                    const auto cold = cache.load_cached_image(path, params);
                    const auto warm = cache.load_cached_image(path, params);
                    if (cache.get_cache_mode() != io::CacheLoader::CacheMode::CPU_memory)
                        throw std::runtime_error("TIFF pipeline did not exercise the memory cache");
                    compare(cold, reference);
                    compare(warm, reference);
                    if (params.decode_16bit && params.resize_factor == 1 && params.max_width == 0) {
                        auto values = warm.to(core::Device::CPU).contiguous();
                        for (size_t channel = 0; channel < 3; ++channel)
                            for (size_t pixel = 0; pixel < width * height; ++pixel)
                                if (std::abs(values.ptr<float>()[channel * width * height + pixel] - rgba[pixel * channels + channel] / 65535.f) > 2.f / 65535)
                                    throw std::runtime_error("TIFF cache silently quantized a decode_16bit request to RGB8");
                    }
                }
                const auto resized_rgba = io::load_rgba_image_cpu_decoded(path, 2, 5, nullptr, true);
                compare(resized_rgba, io::load_rgba_image_cpu_decoded(png, 2, 5, nullptr, true));
                auto alpha_values = resized_rgba.to(core::Device::CPU).contiguous();
                if (alpha_values.shape()[0] != 4)
                    throw std::runtime_error("TIFF RGBA resize lost alpha channel");
                const size_t plane = alpha_values.shape()[1] * alpha_values.shape()[2];
                for (size_t pixel = 0; pixel < plane; ++pixel)
                    if (std::abs(alpha_values.ptr<float>()[3 * plane + pixel] - std::lround(.375 * 65535) / 65535.f) > 2.f / 65535)
                        throw std::runtime_error("TIFF RGBA resize changed constant alpha");
            }
    }
    void videoOutputContracts(lfs::core::Device device, bool require_videotoolbox,
                              lfs::media::VideoEncodeCodec codec = lfs::media::VideoEncodeCodec::H264) {
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
        // Saturated patches distinguish BT.601 from BT.709; neutral ramps alone
        // cannot detect a wrong matrix or an inferred HD color interpretation.
        constexpr std::array<std::array<int, 3>, 12> colors{{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {0, 255, 255}, {255, 0, 255}, {0, 0, 0}, {255, 255, 255}, {32, 64, 96}, {64, 96, 128}, {128, 160, 192}, {192, 224, 240}}};
        io::video::VideoEncoder encoder;
        for (const auto width : {320, 160}) {
            io::video::VideoExportOptions options;
            options.preset = io::video::VideoPreset::CUSTOM;
            options.width = width;
            options.height = 96;
            options.framerate = 10;
            options.codec = codec;
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
                                colors[index][channel] / 255.f;
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
            const auto description = media::MediaProbe::inspect(path);
            if (!description || !description->selected_video_stream)
                throw std::runtime_error("Cannot inspect encoded color interpretation");
            const auto& stream = description->streams.at(*description->selected_video_stream);
            if (codec == media::VideoEncodeCodec::AV1 && stream.codec != "av1")
                throw std::runtime_error("AV1 Tensor export changed codec");
            const auto& color = stream.color;
            if (color.matrix != "smpte170m" || color.range != "tv" || color.primaries || color.transfer)
                throw std::runtime_error("Encoded color tags do not describe the supplied BT.601 limited-range planes");
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
                        error += std::abs(int(pixels[pixel * player.currentFrameChannels() + channel]) - colors[index][channel]);
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

// Uses an ordinary 16-bit PNG; this regression does not depend on TIFF export.
void imageCachePrecisionContracts() {
    using namespace lfs;
    constexpr size_t width = 8, height = 4, channels = 3;
    std::vector<uint16_t> samples(width * height * channels);
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i] = static_cast<uint16_t>(1 + (i * 683) % 65535);
    const auto path = std::filesystem::temp_directory_path() /
                      ("lfs-cache-precision-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".png");
    auto& cache = io::CacheLoader::getInstance(true);
    struct Cleanup {
        std::filesystem::path path;
        io::CacheLoader& cache;
        ~Cleanup() {
            cache.reset_cache();
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{path, cache};
    if (!core::save_png(path, samples.data(), width, height, channels, 16, 6))
        throw std::runtime_error("Cannot write the cache precision PNG fixture");
    cache.reset_cache();
    cache.update_cache_params(true, 2, 0.f, 0.f, false, 1);
    for (const auto params : {io::LoadParams{}, io::LoadParams{.decode_16bit = true},
                              io::LoadParams{.resize_factor = 2, .decode_16bit = true},
                              io::LoadParams{.max_width = 3, .output_uint8 = true, .decode_16bit = true}}) {
        const auto reference = io::load_rgb_image_cpu_decoded(path, params, params.decode_16bit).cpu().contiguous();
        for (const auto actual : {io::load_rgb_image_cpu_decoded(path, params),
                                  cache.load_cached_image(path, params), cache.load_cached_image(path, params)}) {
            const auto host = actual.cpu().contiguous();
            if (host.shape() != reference.shape() || host.dtype() != reference.dtype() || host.bytes() != reference.bytes() ||
                std::memcmp(host.data_ptr(), reference.data_ptr(), host.bytes()))
                throw std::runtime_error("Direct/cold/warm image load lost the requested decode precision");
            if (params.decode_16bit && params.resize_factor == 1 && params.max_width == 0)
                for (size_t c = 0; c < channels; ++c)
                    for (size_t pixel = 0; pixel < width * height; ++pixel)
                        if (std::abs(host.ptr<float>()[c * width * height + pixel] - samples[pixel * channels + c] / 65535.f) > 2.f / 65535.f)
                            throw std::runtime_error("Sixteen-bit image cache request was quantized to RGB8");
        }
        if (cache.get_cache_mode() != io::CacheLoader::CacheMode::CPU_memory)
            throw std::runtime_error("Precision regression did not exercise the decoded memory cache");
    }
}

void heifCacheAndQueueContracts() {
    using namespace lfs;
    auto& cache = io::CacheLoader::getInstance(true);
    struct Reset {
        io::CacheLoader& cache;
        ~Reset() { cache.reset_cache(); }
    } reset{cache};
    std::vector<const char*> images{"rgb12.avif", "p3.avif"};
    for (const auto name : {"system-p3icc.heic", "system-rgb10.heic"}) {
        core::image_codecs::Image native;
        std::string error;
        if (core::image_codecs::decode(std::filesystem::path(LFS_HEIF_FIXTURES) / name, native, error))
            images.push_back(name);
        else {
#ifdef __APPLE__
            throw std::runtime_error(std::string("Qualified Apple HEIC consumer failed: ") + error);
#elif defined(_WIN32)
            if (std::string_view(name) == "system-p3icc.heic" &&
                error.find("provider unavailable") == std::string::npos)
                throw std::runtime_error(std::string("Qualified Windows HEIC consumer failed: ") + error);
#endif
            std::cout << "Native HEIC consumer unavailable for " << name << ": " << error << '\n';
        }
    }
    for (const auto name : images) {
        const auto path = std::filesystem::path(LFS_HEIF_FIXTURES) / name;
        cache.reset_cache();
        cache.update_cache_params(true, 2, 0.f, 0.f, false, 1);
        io::PipelinedLoaderConfig config;
        config.backend = core::default_gpu_backend();
        config.io_threads = 1;
        config.cold_process_threads = 1;
        config.output_queue_size = 2;
        config.prefetch_count = 2;
        config.use_16bit_color = true;
        io::PipelinedImageLoader queue(config);
        size_t sequence = 0;
        for (const auto params : {io::LoadParams{}, io::LoadParams{.decode_16bit = true}, io::LoadParams{.resize_factor = 2, .decode_16bit = true}, io::LoadParams{.max_width = 5, .output_uint8 = true, .decode_16bit = true}}) {
            const auto reference = io::load_rgb_image_cpu_decoded(path, params).cpu().contiguous();
            const auto compare = [&](const core::Tensor& actual) {
                const auto host = actual.cpu().contiguous();
                if (host.shape() != reference.shape() || host.dtype() != reference.dtype() || host.bytes() != reference.bytes() || std::memcmp(host.data_ptr(), reference.data_ptr(), host.bytes()))
                    throw std::runtime_error("HEIF cache/async loader changed color, shape or precision");
            };
            compare(cache.load_cached_image(path, params));
            compare(cache.load_cached_image(path, params));
            if (cache.get_cache_mode() != io::CacheLoader::CacheMode::CPU_memory)
                throw std::runtime_error("HEIF test did not exercise decoded memory cache");
            // Request precision explicitly; preserve the queue's existing run-level default.
            auto queued = params;
            queued.decode_16bit = true;
            const auto queued_reference = io::load_rgb_image_cpu_decoded(path, queued).cpu().contiguous();
            for (int repeat = 0; repeat < 2; ++repeat) {
                queue.prefetch(sequence, path, queued);
                const auto completion = queue.try_get_completion_for(std::chrono::seconds(10));
                if (!completion || !completion->outcome || completion->sequence != sequence)
                    throw std::runtime_error("HEIF asynchronous completion failed");
                const auto host = completion->outcome->tensor.cpu().contiguous();
                bool equal = host.shape() == queued_reference.shape() && host.dtype() == queued_reference.dtype() && host.bytes() == queued_reference.bytes();
                if (equal && host.dtype() == core::DataType::Float32 && repeat == 1 &&
                    config.backend == core::GpuBackend::CUDA && (queued.resize_factor != 1 || queued.max_width != 0)) {
                    // The existing derived JPEG 2000 cache clamps/rounds resized floats to UINT16.
                    // Original samples and every cold decode still require byte equality.
                    for (size_t i = 0; i < host.numel(); ++i)
                        equal = equal && std::abs(host.ptr<float>()[i] - std::clamp(queued_reference.ptr<float>()[i], 0.f, 1.f)) <= 0.5f / 65535.f + 1e-7f;
                } else if (equal) {
                    equal = std::memcmp(host.data_ptr(), queued_reference.data_ptr(), host.bytes()) == 0;
                }
                if (!equal)
                    throw std::runtime_error(std::string("HEIF training queue changed samples: ") + name + " sequence=" + std::to_string(sequence));
                ++sequence;
            }
        }
        queue.shutdown();
    }
}

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
        if (!cpu) {
            imageCachePrecisionContracts();
            heifCacheAndQueueContracts();
        }
        imageOutputContracts(cpu ? core::Device::CPU : core::Device::GPU);
        if (!cpu)
            tiffPipelineContracts();
        // CUDA builds select NVENC by default even for a CPU source. Keep the
        // CPU-only contract independent of hardware; GPU profiles also exercise
        // CPU sources above, while non-CUDA builds cover the CPU producer here.
        if (!cpu || !LFS_HAS_CUDA || require_videotoolbox) {
            const auto device = cpu ? core::Device::CPU : core::Device::GPU;
            videoOutputContracts(device, require_videotoolbox);
            videoOutputContracts(device, false, media::VideoEncodeCodec::AV1);
        }
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
        // Preserve the normalized EOTF used by the existing HDR tonemapper.
        // Freeze its previous float expression over every 14-bit input code.
        for (int code = 0; code <= 16383; ++code) {
            const float value = code / 16383.f;
            const float p = std::pow(std::fmax(value, 0.f), 1.f / (2523.f / 4096 * 128));
            const float previous = std::pow(std::fmax(p - 3424.f / 4096, 0.f) /
                                                (2413.f / 4096 * 32 - 2392.f / 4096 * 32 * p),
                                            1.f / (2610.f / 4096 / 4));
            if (core::color::pqToLinear(value) != previous)
                throw std::runtime_error("Shared PQ EOTF changes the existing normalized HDR curve");
        }
        // Planar RGB16 and shifted semiplanar 10-bit 4:2:0, in both byte
        // orders. Independent analytic values cover native/reduced/enlarged
        // extents; the resize oracle operates on linear luminance, not codes.
        for (const unsigned transfer : {3u, 4u}) {
            for (const bool rgb : {false, true}) {
                for (const bool big_endian : {false, true}) {
                    VideoColorParameters pq{};
                    pq.width = 4;
                    pq.height = 2;
                    pq.rgb = rgb;
                    pq.big_endian = big_endian;
                    pq.transfer = transfer;
                    std::vector<uint8_t> samples(rgb ? 48 : 24);
                    const auto store = [&](size_t offset, unsigned code) {
                        samples[offset] = uint8_t(big_endian ? code >> 8 : code);
                        samples[offset + 1] = uint8_t(big_endian ? code : code >> 8);
                    };
                    if (rgb) {
                        for (unsigned c = 0; c < 3; ++c) {
                            pq.component[c] = {c * 16, 8, 2, 0, 16, 4, 2};
                            pq.scale[c] = 1.f / 65535;
                            pq.decode[c * 3 + c] = 1;
                            for (unsigned y = 0; y < 2; ++y)
                                for (unsigned x = 0; x < 4; ++x)
                                    store(c * 16 + y * 8 + x * 2, 5000 + x * 12000 + c * 3000);
                        }
                    } else {
                        pq.component[0] = {0, 8, 2, 6, 10, 4, 2};
                        pq.component[1] = {16, 8, 4, 6, 10, 2, 1};
                        pq.component[2] = {18, 8, 4, 6, 10, 2, 1};
                        pq.chroma_x_scale = pq.chroma_y_scale = .5f;
                        pq.chroma_y_offset = -.25f;
                        const std::array matrix{1.f, 0.f, 1.4746f, 1.f, -.164553127f, -.571353127f, 1.f, 1.8814f, 0.f};
                        std::copy(matrix.begin(), matrix.end(), pq.decode);
                        pq.scale[0] = 1.f / 876;
                        pq.bias[0] = 64.f / 876;
                        pq.scale[1] = pq.scale[2] = 1.f / 896;
                        pq.bias[1] = pq.bias[2] = 512.f / 896;
                        for (unsigned y = 0; y < 2; ++y)
                            for (unsigned x = 0; x < 4; ++x)
                                store(y * 8 + x * 2, (64 + x * 280) << 6);
                        for (unsigned offset = 16; offset < 24; offset += 2)
                            store(offset, 512 << 6);
                    }
                    const auto reference = [&](int x, int c) {
                        const double code = rgb ? (5000 + std::clamp(x, 0, 3) * 12000 + c * 3000) / 65535.
                                                : std::clamp(x, 0, 3) * 280 / 876.;
                        if (transfer == 4)
                            return code <= .5 ? code * code / 3 : (std::exp((code - .55991073) / .17883277) + .28466892) / 12;
                        const double p = std::pow(code, 32. / 2523);
                        return 10000 * std::pow(std::max(p - 3424. / 4096, 0.) /
                                                    (2413. / 128 - 2392. / 128 * p),
                                                16384. / 2610);
                    };
                    for (const int width : {2, 4, 8}) {
                        const int height = width / 2;
                        std::vector<uint8_t> converted(width * height * 12);
                        const auto converted_result = convert(samples, pq, width, height, converted);
                        if (!converted_result)
                            throw std::runtime_error(std::string(converted_result.error().detail()));
                        for (int y = 0; y < height; ++y)
                            for (int x = 0; x < width; ++x)
                                for (int c = 0; c < 3; ++c) {
                                    const double sx = (x + .5) * 4 / width - .5;
                                    const int ix = int(std::floor(sx));
                                    const double fraction = sx - ix;
                                    const double expected_value = width == 2 ? (reference(x * 2, c) + reference(x * 2 + 1, c)) / 2
                                                                             : reference(ix, c) * (1 - fraction) + reference(ix + 1, c) * fraction;
                                    float actual;
                                    std::memcpy(&actual, converted.data() + ((y * width + x) * 3 + c) * 4, 4);
                                    if (std::abs(actual - expected_value) > (transfer == 4 ? std::max(1e-7, expected_value * 2e-6) : std::max(5e-5, expected_value * 2e-4)))
                                        throw std::runtime_error("HDR endian/layout/linear resize differs from independent reference");
                                }
                    }
                }
            }
        }
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
        double maximum_error = 0;
        for (const unsigned transfer : {2u, 3u, 4u}) {
            color.transfer = transfer;
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

            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 8; ++x) {
                    const double l = (source[y * 8 + x] - 16.0) / 219, u = (chroma(0, x, y) - 128) / 224, v = (chroma(1, x, y) - 128) / 224;
                    const std::array encoded{l + 1.5748 * v, l - 2 * .0722 * (1 - .0722) / .7152 * u - 2 * .2126 * (1 - .2126) / .7152 * v, l + 1.8556 * u};
                    for (int c = 0; c < 3; ++c) {
                        float actual;
                        std::memcpy(&actual, output.data() + ((y * 8 + x) * 3 + c) * 4, 4);
                        const double value = encoded[c];
                        const double p = std::pow(std::clamp(value, 0., 1.), 32. / 2523);
                        const double bounded = std::clamp(value, 0., 1.);
                        const double expected = transfer == 4 ? (bounded <= .5 ? bounded * bounded / 3 : (std::exp((bounded - .55991073) / .17883277) + .28466892) / 12) : transfer == 3 ? 10000 * std::pow(std::max(p - 3424. / 4096, 0.) / (2413. / 128 - 2392. / 128 * p), 16384. / 2610)
                                                                                                                                                                       : value < .081    ? value / 4.5
                                                                                                                                                                                         : std::pow((value + .099) / 1.099, 1 / .45);
                        if (std::abs(actual - expected) > (transfer == 3 ? std::max(5e-5, expected * 2e-4) : 3e-6))
                            throw std::runtime_error("Varied 4:2:0 chroma differs from independent SDR/PQ/HLG reference");
                        if (transfer == 2)
                            maximum_error = std::max(maximum_error, std::abs(actual - expected));
                    }
                }
            if (maximum_error > 3e-6)
                throw std::runtime_error("GPU 4:2:0 differs from the independent linear reference: " + std::to_string(maximum_error));
        }
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
        // PQ uses an independent double-precision EOTF reference and absolute
        // luminance; it must not reuse the SDR normalized-output contract.
        color.transfer = 3;
        result = convert(source, color, 1, 1, output);
        if (!result)
            throw std::runtime_error(std::string(result.error().detail()));
        for (size_t c = 0; c < 3; ++c) {
            const double p = std::pow(double(expected[c]), 32.0 / 2523.0);
            const double nits = 10000 * std::pow(std::max(p - 3424.0 / 4096, 0.0) / (2413.0 / 128 - 2392.0 / 128 * p), 16384.0 / 2610);
            float actual;
            std::memcpy(&actual, output.data() + c * 4, 4);
            if (std::abs(actual - nits) > std::max(5e-5, nits * 2e-4))
                throw std::runtime_error("PQ conversion differs from absolute-luminance reference");
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
