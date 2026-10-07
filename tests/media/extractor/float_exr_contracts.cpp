// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/image_codecs.hpp"
#include "media/media_ingest.hpp"
#include "media/video_frame_extractor.hpp"
#include <OpenEXR/openexr.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace {
    void require(bool ok, const char* message) {
        if (!ok)
            throw std::runtime_error(message);
    }
    std::vector<uint8_t> bytes(const std::filesystem::path& path) {
        std::ifstream f(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    }
    void preserved(const std::filesystem::path& path, const std::vector<uint8_t>& expected) {
        require(bytes(path) == expected, "failed/cancelled EXR preserves previous target");
        for (const auto& entry : std::filesystem::directory_iterator(path.parent_path()))
            require(entry.path().extension() != ".tmp", "owned EXR temporaries are cleaned up");
    }
} // namespace
int runFloatExrUnitContracts() {
    using namespace lfs::media;
    const auto directory = std::filesystem::temp_directory_path() / ("lfs-float-exr-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code e;
            std::filesystem::remove_all(path, e);
        }
    } cleanup{directory};
    // Unaligned source and odd stride exercise byte-safe copy/codec input.
    std::vector<uint8_t> storage(27, 0x7f);
    const std::array<float, 3> a{-.25f, 1.5f, .1234567f}, b{.5f, 2.f, .0001f};
    std::memcpy(storage.data() + 1, a.data(), 12);
    std::memcpy(storage.data() + 14, b.data(), 12);
    FrameView view{{1, 2, 13, FramePixelFormat::RGBFloat32, {ColorTransfer::Linear, ColorPrimaries::Bt709, AlphaMode::None}}, {}, std::span<const uint8_t>(storage).subspan(1)};
    view.info.origin = FrameOrigin::Rendered;
    view.info.output_timestamp = Timestamp{3, {1, 24}};
    auto owned = FrameSurface::copyOf(view);
    require(owned.has_value() && owned->view().pixels[12] == 0, "float snapshot retains visible rows and zeroes padding");
    MemoryFrameSink limited(24);
    require(limited.begin({}).has_value() && limited.write(view).error().code() == lfs::ErrorCode::ResourceExhausted, "float payload includes padded row budget");
    MemoryFrameSink memory(25);
    require(memory.begin({}).has_value() && memory.write(view).has_value(), "float snapshot exact budget accepted");
    storage.assign(storage.size(), 0);
    view = owned->view();
    require(view.info.origin == FrameOrigin::Rendered && !view.info.source_timestamp && view.info.output_timestamp->ticks == 3, "rendered ownership and output timing independent of source PTS");
    auto invalid = view;
    invalid.layout.row_stride = 11;
    require(!invalid.requiredBytes(), "float row size validated");
    invalid = view;
    invalid.layout.row_stride = std::numeric_limits<size_t>::max();
    require(!invalid.requiredBytes(), "float stride overflow rejected");

    for (const auto precision : {ExrPrecision::Half, ExrPrecision::Float}) {
        for (const auto compression : {ExrCompression::None, ExrCompression::ZIP}) {
            const auto path = directory / (std::to_string(static_cast<int>(precision)) + std::to_string(static_cast<int>(compression)) + ".exr");
            ExrOutputOptions options{precision, compression};
            options.provenance = "external renderer test";
            const auto result = ImageOutput::writeExr(path, view, options);
            require(result.has_value(), "float EXR written for both sample types/compressions");
            lfs::core::image_codecs::Image image;
            std::string error;
            require(lfs::core::image_codecs::decode(path, image, error), "EXR round-trip through production reader");
            require(image.width == 1 && image.height == 2 && image.sample_type == lfs::core::image_codecs::SampleType::Float32, "EXR dimensions and actual sample type");
            for (int y = 0; y < 2; ++y)
                for (int c = 0; c < 3; ++c) {
                    float actual = 0;
                    std::memcpy(&actual, image.data.data() + (y * image.channels + c) * sizeof(float), sizeof(float));
                    const float expected = (y ? b : a)[c];
                    require(std::abs(actual - expected) <= (precision == ExrPrecision::Float ? 1e-7f : .0001f), "EXR preserves negative/above-one samples with declared quantization");
                }
            exr_context_t context = nullptr;
            const auto name = path.string();
            require(exr_start_read(&context, name.c_str(), nullptr) == EXR_ERR_SUCCESS, "read EXR header");
            const char* text = nullptr;
            int32_t length = 0;
            require(exr_attr_get_string(context, 0, "lfsOrigin", &length, &text) == EXR_ERR_SUCCESS && std::string(text) == "rendered", "origin metadata written");
            require(exr_attr_get_string(context, 0, "lfsOutputTimestamp", &length, &text) == EXR_ERR_SUCCESS && std::string(text) == "3/1/24", "output timeline is rational");
            const exr_attribute_t* attr = nullptr;
            require(exr_get_attribute_by_name(context, 0, "lfsSourceTimestamp", &attr) != EXR_ERR_SUCCESS, "no fake source PTS for rendered frame");
            (void)exr_finish(&context);
            require(ImageOutput::writeExr(path, view, options).error().code() == lfs::ErrorCode::AlreadyExists, "default EXR never overwrites");
        }
    }

    const std::array<float, 8> rgba{2.f, -1.f, .5f, .5f, .5f, 1.f, 2.f, 0.f};
    FrameView alpha{{2, 1, 32, FramePixelFormat::RGBAFloat32, {ColorTransfer::Linear, ColorPrimaries::Bt2020, AlphaMode::Independent}}, {}, {reinterpret_cast<const uint8_t*>(rgba.data()), sizeof(rgba)}};
    auto path = directory / "alpha.exr";
    ExrOutputOptions options;
    options.precision = ExrPrecision::Float;
    require(ImageOutput::writeExr(path, alpha, options).has_value(), "straight RGBA EXR accepted");
    lfs::core::image_codecs::Image image;
    std::string error;
    require(lfs::core::image_codecs::decode(path, image, error) && image.channels == 4, "RGBA EXR reader retains alpha");
    const std::array<float, 8> expected_alpha{1.f, -.5f, .25f, .5f, 0.f, 0.f, 0.f, 0.f};
    require(image.data == std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(expected_alpha.data()), reinterpret_cast<const uint8_t*>(expected_alpha.data()) + sizeof(expected_alpha)), "EXR uses premultiplied alpha, including transparent pixel");
    alpha.layout.color.alpha = AlphaMode::Premultiplied;
    require(ImageOutput::writeExr(directory / "premult.exr", alpha, options).has_value(), "premultiplied alpha accepted without reapplication");
    auto invalid_alpha = rgba;
    auto invalid_alpha_view = alpha;
    invalid_alpha_view.pixels = {reinterpret_cast<const uint8_t*>(invalid_alpha.data()), sizeof(invalid_alpha)};
    const auto alpha_previous = bytes(path);
    options.overwrite = true;
    for (const float value : {-0.01f, 1.01f, std::numeric_limits<float>::quiet_NaN()}) {
        invalid_alpha[3] = value;
        require(ImageOutput::writeExr(path, invalid_alpha_view, options).error().code() == lfs::ErrorCode::InvalidArgument, "invalid alpha is rejected before commit");
        preserved(path, alpha_previous);
    }

    path = directory / "previous.exr";
    {
        std::ofstream f(path, std::ios::binary);
        f << "previous target";
    }
    const auto previous = bytes(path);
    lfs::io::VideoFrameExtractor legacy;
    lfs::io::VideoFrameExtractor::Params legacy_exr;
    legacy_exr.format = lfs::io::ImageFormat::EXR;
    legacy_exr.output_dir = directory / "legacy-exr";
    std::string legacy_error;
    require(!legacy.extract(legacy_exr, legacy_error) && legacy.lastError()->code() == lfs::ErrorCode::Unsupported &&
                !std::filesystem::exists(legacy_exr.output_dir),
            "legacy RGB8 EXR request must not write JPEG bytes or invent EXR metadata");
    options.overwrite = true;
    std::array<float, 3> bad{.2f, .3f, .4f};
    FrameView bad_view{{1, 1, 12, FramePixelFormat::RGBFloat32, {ColorTransfer::Linear, ColorPrimaries::Bt709, AlphaMode::None}}, {}, {reinterpret_cast<const uint8_t*>(bad.data()), sizeof(bad)}};
    for (const float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), 65505.f}) {
        bad[0] = value;
        options.precision = ExrPrecision::Half;
        require(ImageOutput::writeExr(path, bad_view, options).error().code() == lfs::ErrorCode::InvalidArgument, "nonfinite/half overflow is explicit error");
        preserved(path, previous);
    }
    bad[0] = 100000.f;
    options.precision = ExrPrecision::Float;
    require(ImageOutput::writeExr(directory / "large.exr", bad_view, options).has_value(), "FLOAT finite range is preserved");
    bad_view.layout.color.transfer = ColorTransfer::Srgb;
    require(ImageOutput::writeExr(path, bad_view, options).error().code() == lfs::ErrorCode::Unsupported, "writer never silently linearizes input");
    preserved(path, previous);
    bad_view.layout.color.transfer = ColorTransfer::Linear;
    bad_view.layout.color.primaries = ColorPrimaries::Unspecified;
    require(ImageOutput::writeExr(path, bad_view, options).error().code() == lfs::ErrorCode::Unsupported, "unknown primaries rejected");
    preserved(path, previous);
    options.precision = static_cast<ExrPrecision>(-1);
    require(ImageOutput::writeExr(path, view, options).error().code() == lfs::ErrorCode::InvalidArgument, "invalid precision rejected");
    preserved(path, previous);
    options.precision = ExrPrecision::Float;
    {
        const int width = static_cast<int>((64ULL * 1024 * 1024) / (12 * 16)) + 1;
        std::vector<uint8_t> payload(static_cast<size_t>(width) * 12 * 16);
        FrameView oversized{{width, 16, static_cast<size_t>(width) * 12, FramePixelFormat::RGBFloat32, view.layout.color}, {}, payload};
        require(ImageOutput::writeExr(path, oversized, options).error().code() == lfs::ErrorCode::ResourceExhausted, "EXR chunk budget rejected before output side effects");
        preserved(path, previous);
    }
    FileFrameSinkOptions invalid_options;
    invalid_options.output_directory = directory / "invalid-options";
    invalid_options.format = FrameFileFormat::EXR;
    invalid_options.exr.compression = static_cast<ExrCompression>(-1);
    FileFrameSink invalid_sink(invalid_options);
    require(invalid_sink.begin({}).error().code() == lfs::ErrorCode::InvalidArgument && !std::filesystem::exists(invalid_options.output_directory), "direct EXR sink validates before directory creation");
    auto invalid_info = view;
    invalid_info.info.output_timestamp->time_base.denominator = 0;
    require(ImageOutput::writeExr(path, invalid_info, options).error().code() == lfs::ErrorCode::InvalidArgument, "invalid timestamp cannot enter EXR header");
    preserved(path, previous);
    std::vector<float> large(4 * 33 * 3, .123f);
    FrameView chunks{{4, 33, 48, FramePixelFormat::RGBFloat32, view.layout.color}, {}, {reinterpret_cast<const uint8_t*>(large.data()), large.size() * sizeof(float)}};
    for (const int stop : {1, 2, 3, 4, 5}) {
        int count = 0;
        options.cancelled = [&] { return ++count == stop; };
        require(ImageOutput::writeExr(path, chunks, options).error().code() == lfs::ErrorCode::Cancelled, "cancellation checked before, during and after chunks");
        preserved(path, previous);
    }
    options.cancelled = []() -> bool { throw 7; };
    require(ImageOutput::writeExr(path, view, options).error().code() == lfs::ErrorCode::Internal, "non-standard producer callback failure is structured");
    preserved(path, previous);
    options.cancelled = {};
    require(ImageOutput::writeExr(path, view, options).has_value(), "explicit successful replacement accepted");
    require(bytes(path) != previous, "successful EXR replaced target");
    auto missing = ImageOutput::writeExr(directory / "missing" / "out.exr", view);
    require(!missing && missing.error().native().has_value(), "filesystem failure retains native diagnostic");
    FileFrameSink rgb_files({directory / "png"});
    require(rgb_files.begin({}).has_value() && rgb_files.write(view).error().code() == lfs::ErrorCode::Unsupported, "PNG sink does not interpret float bytes as RGB8");
    // Two producers must never replace each other's completed target by default.
    const auto race = directory / "race.exr";
    bool first = false, second = false;
    std::thread t1([&] { first = ImageOutput::writeExr(race, view).has_value(); });
    std::thread t2([&] { second = ImageOutput::writeExr(race, view).has_value(); });
    t1.join();
    t2.join();
    require(first != second, "atomic no-replace commit admits exactly one producer");
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        require(entry.path().extension() != ".tmp", "all EXR temporaries cleaned");
    return 0;
}
