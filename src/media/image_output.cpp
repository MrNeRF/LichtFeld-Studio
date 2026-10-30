// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/image_output.hpp"
#include "core/atomic_file.hpp"
#include "core/color_transfer.h"
#include "core/image_avif.hpp"
#include "core/image_codecs.hpp"
#include "core/image_exr.hpp"
#include "core/image_jxl.hpp"
#include "core/path_utils.hpp"
#include "media/media_options.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <format>
#include <limits>
#include <nlohmann/json.hpp>
#include <vector>

namespace lfs::media {
    namespace {
        Error outputError(ErrorCode code, std::string message, int native = 0, const char* provider = "ImageOutput", const char* operation = "ImageOutput.writeExr") {
            return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT(), .fields = SmallFields{}.add("operation", std::string_view(operation)), .native = NativeError{ErrorDomain::IO, native, provider}});
        }
        struct WriteFailure {
            Error error;
        };
        void cancel(const ExrOutputOptions& options) {
            if (options.cancelled && options.cancelled())
                throw WriteFailure{outputError(ErrorCode::Cancelled, "EXR write cancelled")};
        }
        int64_t writeBytes(void* userdata, const void* buffer, uint64_t size, uint64_t offset) noexcept {
            auto* stream = static_cast<FILE*>(userdata);
            if (offset > uint64_t(std::numeric_limits<int64_t>::max()) || size > uint64_t(std::numeric_limits<int64_t>::max()) || size > std::numeric_limits<size_t>::max())
                return -1;
#ifdef _WIN32
            if (_fseeki64(stream, int64_t(offset), SEEK_SET))
                return -1;
#else
            if (offset > uint64_t(std::numeric_limits<off_t>::max()) || fseeko(stream, off_t(offset), SEEK_SET))
                return -1;
#endif
            return std::fwrite(buffer, 1, size_t(size), stream) == size ? int64_t(size) : -1;
        }
        std::string timestampText(const Timestamp& value) {
            return std::format("{}/{}/{}", value.ticks, value.time_base.numerator, value.time_base.denominator);
        }
        template <class Fn>
        SinkResult outputBoundary(Fn&& fn, const char* operation = "ImageOutput.writeExr") {
            try {
                return fn();
            } catch (const WriteFailure& failure) { return SinkResult::failure(failure.error); } catch (const std::bad_alloc&) {
                return SinkResult::failure(outputError(ErrorCode::ResourceExhausted, "Image output allocation failed (bad_alloc=true)", 0, "ImageOutput", operation));
            } catch (const std::exception& error) {
                // LFS-CENSUS-OK(empty-catch): outputError returns a core Internal error retaining the exception text; temporary cleanup preserves the target.
                return SinkResult::failure(outputError(ErrorCode::Internal, std::format("Image output failed (exception={})", error.what()), 0, "ImageOutput", operation));
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): callback failures become structured errors.
                return SinkResult::failure(outputError(ErrorCode::Internal, "Image output failed (nonstandard_exception=true)", 0, "ImageOutput", operation));
            }
        }
        template <class Sample, class Fail>
        SinkResult quantizeSdr(const FrameView& frame, ColorTransfer transfer, uint16_t maximum,
                               const std::function<bool()>& cancelled, std::vector<Sample>& pixels, Fail&& fail) {
            const auto& layout = frame.layout;
            const bool rgba = layout.format == FramePixelFormat::RGBAFloat32;
            const size_t channels = rgba ? 4 : 3;
            const auto encode = [&](float v) {
                v = std::clamp(v, 0.0f, 1.0f);
                return transfer == ColorTransfer::Srgb    ? core::color::linearToSrgb(v)
                       : transfer == ColorTransfer::Bt709 ? core::color::linearToBt709(v)
                                                          : v;
            };
            const auto quantize = [&](float v) { return static_cast<Sample>(std::floor(std::clamp(double(v), 0.0, 1.0) * double(maximum) + 0.5)); };
            for (int y = 0; y < layout.height; ++y) {
                if (cancelled && cancelled())
                    return fail(ErrorCode::Cancelled, "SDR conversion cancelled");
                for (int x = 0; x < layout.width; ++x) {
                    std::array<float, 4> sample{0, 0, 0, 1};
                    std::memcpy(sample.data(), frame.pixels.data() + y * layout.row_stride + x * channels * 4, channels * 4);
                    for (size_t c = 0; c < channels; ++c)
                        if (!std::isfinite(sample[c]))
                            return fail(ErrorCode::InvalidArgument, "SDR cannot encode non-finite samples");
                    if (rgba && (sample[3] < 0 || sample[3] > 1))
                        return fail(ErrorCode::InvalidArgument, "SDR alpha must be in [0,1]");
                    const auto offset = (static_cast<size_t>(y) * layout.width + x) * channels;
                    for (size_t c = 0; c < 3; ++c) {
                        const float straight = rgba && layout.color.alpha == AlphaMode::Premultiplied
                                                   ? (sample[3] > 0 ? sample[c] / sample[3] : 0)
                                                   : sample[c];
                        if (!std::isfinite(straight))
                            return fail(ErrorCode::InvalidArgument, "SDR unassociated sample is not finite");
                        pixels[offset + c] = quantize(encode(straight));
                    }
                    if (rgba)
                        pixels[offset + 3] = quantize(sample[3]);
                }
            }
            return {};
        }
    } // namespace
    SinkResult validateAvifOutputOptions(const AvifOutputOptions& options) {
        if (auto valid = core::image_codecs::validate_avif_encode_options({.lossless = options.lossless, .quality = options.quality, .speed = options.speed, .bit_depth = options.bit_depth}); !valid)
            return valid;
        if (!core::image_codecs::avif_encoder_available())
            return SinkResult::failure(outputError(ErrorCode::Unavailable, "Built-in AOM AVIF encoder unavailable", 0, "AVIF", "ImageOutput.writeAvif"));
        return {};
    }
    SinkResult ImageOutput::writeAvif(const std::filesystem::path& path, const FrameView& frame, const AvifOutputOptions& options, std::string_view xmp) {
        return outputBoundary([&]() -> SinkResult {
            const auto fail = [](ErrorCode code, std::string message) { return SinkResult::failure(outputError(code, std::move(message), 0, "AVIF", "ImageOutput.writeAvif")); };
            if (auto valid = validateAvifOutputOptions(options); !valid)
                return valid;
            if (xmp.size() > 32 * 1024 * 1024 || xmp.find('\0') != std::string_view::npos)
                return fail(ErrorCode::InvalidArgument, "Invalid AVIF XMP packet");
            if (auto required = frame.requiredBytes(); !required)
                return SinkResult::failure(required.error());
            const auto& layout = frame.layout;
            const bool bytes = layout.format == FramePixelFormat::RGB8;
            const bool rgba = layout.format == FramePixelFormat::RGBAFloat32;
            if (layout.color.units != ColorUnits::Relative ||
                (bytes ? options.bit_depth != 8 || layout.color.transfer != ColorTransfer::Srgb || layout.color.primaries != ColorPrimaries::Bt709
                       : (layout.format != FramePixelFormat::RGBFloat32 && !rgba) || layout.color.transfer != ColorTransfer::Linear ||
                             (layout.color.primaries != ColorPrimaries::Bt709 && layout.color.primaries != ColorPrimaries::Bt2020)) ||
                (!rgba && layout.color.alpha != AlphaMode::None) ||
                (rgba && layout.color.alpha != AlphaMode::Independent && layout.color.alpha != AlphaMode::Premultiplied))
                return fail(ErrorCode::Unsupported, "AVIF requires declared relative SDR sRGB RGB8 or linear BT709/BT2020 float RGB/RGBA with matching alpha");
            const size_t channels = rgba ? 4 : 3;
            if (size_t(layout.width) > SIZE_MAX / size_t(layout.height) / channels / 2)
                return fail(ErrorCode::ResourceExhausted, "AVIF packed size overflows");
            namespace codecs = core::image_codecs;
            codecs::AvifEncodeOptions encoding{.lossless = options.lossless, .quality = options.quality, .speed = options.speed, .bit_depth = options.bit_depth, .primaries = layout.color.primaries == ColorPrimaries::Bt2020 ? codecs::AvifPrimaries::Bt2020 : codecs::AvifPrimaries::Bt709};
            if (size_t(layout.height) > encoding.pixels / size_t(layout.width))
                return fail(ErrorCode::ResourceExhausted, "AVIF pixel limit exceeded before conversion");
            std::vector<uint16_t> converted;
            std::vector<uint8_t> packed;
            auto pixels = frame.pixels;
            size_t stride = layout.row_stride;
            if (!bytes) {
                if (options.bit_depth == 8) {
                    packed.resize(size_t(layout.width) * layout.height * channels);
                    if (auto result = quantizeSdr(frame, ColorTransfer::Srgb, 255, options.cancelled, packed, fail); !result)
                        return result;
                    pixels = packed;
                } else {
                    converted.resize(size_t(layout.width) * layout.height * channels);
                    if (auto result = quantizeSdr(frame, ColorTransfer::Srgb, uint16_t((1u << options.bit_depth) - 1), options.cancelled, converted, fail); !result)
                        return result;
                    pixels = {reinterpret_cast<const uint8_t*>(converted.data()), converted.size() * 2};
                }
                stride = size_t(layout.width) * channels * (options.bit_depth == 8 ? 1 : 2);
            }
            return codecs::write_avif(path, pixels, layout.width, layout.height, int(channels), stride, encoding,
                                      {.overwrite = options.overwrite, .durable = false, .create_directories = false, .cancelled = options.cancelled}, xmp);
        },
                              "ImageOutput.writeAvif");
    }
    SinkResult validateJxlOutputOptions(const JxlOutputOptions& options) {
        if (outputOptionName(options.precision) == "unknown" ||
            (options.transfer != ColorTransfer::Linear && options.transfer != ColorTransfer::Srgb) ||
            (options.precision == JxlPrecision::Float32 && options.transfer != ColorTransfer::Linear))
            return SinkResult::failure(outputError(ErrorCode::InvalidArgument, "JXL UInt16 allows linear/sRGB transfer; Float32 requires linear transfer", 0, "JXL", "ImageOutput.writeJxl"));
        return core::image_codecs::validate_jxl_encode_options({.lossless = options.lossless, .distance = options.distance, .effort = options.effort});
    }
    SinkResult validateJxlOutputOptions(const JxlOutputOptions& options, FramePixelFormat format) {
        if (auto valid = validateJxlOutputOptions(options); !valid)
            return valid;
        if (format == FramePixelFormat::RGB8 && options.precision == JxlPrecision::UInt16)
            return SinkResult::failure(outputError(ErrorCode::Unsupported, "JXL UInt16 conversion requires linear Float32 input", 0, "JXL", "ImageOutput.writeJxl"));
        return core::image_codecs::validate_jxl_encode_options({.lossless = options.lossless, .distance = options.distance, .effort = options.effort},
                                                               format == FramePixelFormat::RGB8 ? core::image_codecs::SampleType::UInt8 : options.precision == JxlPrecision::UInt16 ? core::image_codecs::SampleType::UInt16
                                                                                                                                                                                    : core::image_codecs::SampleType::Float32);
    }
    SinkResult ImageOutput::writeJxl(const std::filesystem::path& path, const FrameView& frame, const JxlOutputOptions& options, std::string_view xmp) {
        return outputBoundary([&]() -> SinkResult {
            if (auto valid = validateJxlOutputOptions(options, frame.layout.format); !valid)
                return valid;
            if (auto required = frame.requiredBytes(); !required)
                return SinkResult::failure(required.error());
            const auto& layout = frame.layout;
            const bool rgb8 = layout.format == FramePixelFormat::RGB8;
            const bool rgba = layout.format == FramePixelFormat::RGBAFloat32;
            if (layout.color.units != ColorUnits::Relative ||
                (rgb8 ? layout.color.transfer != ColorTransfer::Srgb || layout.color.primaries != ColorPrimaries::Bt709
                      : (layout.format != FramePixelFormat::RGBFloat32 && !rgba) || layout.color.transfer != ColorTransfer::Linear ||
                            (layout.color.primaries != ColorPrimaries::Bt709 && layout.color.primaries != ColorPrimaries::Bt2020)) ||
                (!rgba && layout.color.alpha != AlphaMode::None) ||
                (rgba && layout.color.alpha != AlphaMode::Independent && layout.color.alpha != AlphaMode::Premultiplied))
                return SinkResult::failure(outputError(ErrorCode::Unsupported, "JXL requires declared relative SDR sRGB RGB8 or linear BT709/BT2020 float RGB/RGBA with matching alpha", 0, "JXL", "ImageOutput.writeJxl"));
            namespace codecs = core::image_codecs;
            codecs::JxlEncodeOptions encoding{.lossless = options.lossless, .distance = options.distance, .effort = options.effort, .color = rgb8 ? codecs::JxlEncodingColor::SRGB : layout.color.primaries == ColorPrimaries::Bt2020 ? codecs::JxlEncodingColor::LinearBT2020
                                                                                                                                                                                                                                      : codecs::JxlEncodingColor::LinearSRGB,
                                              .premultiplied_alpha = rgba && layout.color.alpha == AlphaMode::Premultiplied};
            std::vector<uint16_t> converted;
            auto pixels = frame.pixels;
            size_t stride = layout.row_stride;
            if (options.precision == JxlPrecision::UInt16) {
                const auto fail = [](ErrorCode code, std::string message) { return SinkResult::failure(outputError(code, std::move(message), 0, "JXL", "ImageOutput.writeJxl")); };
                const size_t channels = rgba ? 4 : 3;
                if (size_t(layout.height) > encoding.pixels / size_t(layout.width) ||
                    size_t(layout.width) > SIZE_MAX / size_t(layout.height) / channels / sizeof(uint16_t))
                    return fail(ErrorCode::ResourceExhausted, "JXL UInt16 pixel limit or packed extent exceeded before conversion");
                if (options.cancelled && options.cancelled())
                    return fail(ErrorCode::Cancelled, "JXL UInt16 conversion cancelled");
                converted.resize(size_t(layout.width) * layout.height * channels);
                if (auto result = quantizeSdr(frame, options.transfer, 65535, options.cancelled, converted, fail); !result)
                    return result;
                pixels = {reinterpret_cast<const uint8_t*>(converted.data()), converted.size() * sizeof(uint16_t)};
                stride = size_t(layout.width) * channels * sizeof(uint16_t);
                encoding.premultiplied_alpha = false;
                if (options.transfer == ColorTransfer::Srgb)
                    encoding.color = layout.color.primaries == ColorPrimaries::Bt2020 ? codecs::JxlEncodingColor::SRGBBT2020 : codecs::JxlEncodingColor::SRGB;
            }
            return codecs::write_jxl(path, pixels, layout.width, layout.height, rgba ? 4 : 3,
                                     rgb8 ? codecs::SampleType::UInt8 : options.precision == JxlPrecision::UInt16 ? codecs::SampleType::UInt16
                                                                                                                  : codecs::SampleType::Float32,
                                     stride, encoding,
                                     {.overwrite = options.overwrite, .durable = false, .create_directories = false, .cancelled = options.cancelled}, xmp);
        },
                              "ImageOutput.writeJxl");
    }
    SinkResult validateExrOutputOptions(const ExrOutputOptions& options) {
        if (outputOptionName(options.precision) == "unknown" || outputOptionName(options.compression) == "unknown" ||
            options.provenance.find('\0') != std::string::npos || options.provenance.size() > 1024 * 1024)
            return SinkResult::failure(outputError(ErrorCode::InvalidArgument, std::format("Invalid EXR options (precision={}, compression={}, provenance_bytes={})", int(options.precision), int(options.compression), options.provenance.size())));
        return {};
    }
    SinkResult validateTiff16OutputOptions(const Tiff16OutputOptions& options) {
        if ((options.transfer != ColorTransfer::Linear && options.transfer != ColorTransfer::Srgb && options.transfer != ColorTransfer::Bt709) ||
            options.provenance.find('\0') != std::string::npos || options.provenance.size() > 1024 * 1024)
            return SinkResult::failure(outputError(ErrorCode::InvalidArgument, "Invalid TIFF16 transfer or provenance", 0, "TIFF", "ImageOutput.writeTiff16"));
        return {};
    }
    SinkResult ImageOutput::writeTiff16(const std::filesystem::path& target, const FrameView& frame, const Tiff16OutputOptions& options) {
        return writeTiff16(target, frame, options, {});
    }
    SinkResult ImageOutput::writeTiff16(const std::filesystem::path& target, const FrameView& frame, const Tiff16OutputOptions& options, std::string_view xmp) {
        return outputBoundary([&]() -> SinkResult {
            const auto fail = [](ErrorCode code, std::string message) -> SinkResult {
                return SinkResult::failure(outputError(code, std::move(message), 0, "TIFF", "ImageOutput.writeTiff16"));
            };
            if (auto valid = validateTiff16OutputOptions(options); !valid)
                return valid;
            if (xmp.size() > 32 * 1024 * 1024 || xmp.find('\0') != std::string_view::npos)
                return fail(ErrorCode::InvalidArgument, "Invalid TIFF16 XMP packet");
            const auto required = frame.requiredBytes();
            if (!required)
                return SinkResult::failure(required.error());
            const auto& layout = frame.layout;
            const bool rgba = layout.format == FramePixelFormat::RGBAFloat32;
            if ((!rgba && layout.format != FramePixelFormat::RGBFloat32) || layout.color.transfer != ColorTransfer::Linear ||
                layout.color.units != ColorUnits::Relative ||
                (layout.color.primaries != ColorPrimaries::Bt709 && layout.color.primaries != ColorPrimaries::Bt2020))
                return fail(ErrorCode::Unsupported, "TIFF16 requires relative linear RGB/RGBA and declared BT709/BT2020 primaries; HDR units require an explicit conversion");
            if ((!rgba && layout.color.alpha != AlphaMode::None) ||
                (rgba && layout.color.alpha != AlphaMode::Independent && layout.color.alpha != AlphaMode::Premultiplied))
                return fail(ErrorCode::InvalidArgument, "TIFF16 alpha must match frame layout");
            const auto valid_time = [](const std::optional<Timestamp>& t) { return !t || (t->time_base.numerator > 0 && t->time_base.denominator > 0); };
            if (!valid_time(frame.info.source_timestamp) || !valid_time(frame.info.output_timestamp) ||
                (frame.info.source_component_depth && *frame.info.source_component_depth <= 0) ||
                (frame.info.origin != FrameOrigin::Unspecified && frame.info.origin != FrameOrigin::Decoded && frame.info.origin != FrameOrigin::Rendered && frame.info.origin != FrameOrigin::External))
                return fail(ErrorCode::InvalidArgument, "Invalid TIFF16 frame timing, precision or origin");
            const size_t channels = rgba ? 4 : 3;
            if (static_cast<size_t>(layout.width) > std::numeric_limits<size_t>::max() / layout.height / channels / 2)
                return fail(ErrorCode::InvalidArgument, "TIFF16 packed size overflows");
            std::vector<uint16_t> pixels(static_cast<size_t>(layout.width) * layout.height * channels);
            if (auto converted = quantizeSdr(frame, options.transfer, 65535, options.cancelled, pixels, fail); !converted)
                return converted;
            const auto lookup = core::image_codecs::tiff_transfer_function(outputOptionName(options.transfer));
            nlohmann::json description{{"lfsTransfer", outputOptionName(options.transfer)}, {"lfsPrimaries", outputOptionName(layout.color.primaries)}, {"lfsUnits", "relative"}, {"lfsAlpha", rgba ? "independent" : "none"}, {"lfsProvenance", options.provenance}, {"lfsOrigin", static_cast<int>(frame.info.origin)}, {"lfsQuantization", "clamp-round-half-up-uint16"}};
            description["lfsEncoding"] = core::image_codecs::tiff_sdr_encoding;
            if (frame.info.source_timestamp)
                description["lfsSourceTimestamp"] = timestampText(*frame.info.source_timestamp);
            if (frame.info.output_timestamp)
                description["lfsOutputTimestamp"] = timestampText(*frame.info.output_timestamp);
            if (frame.info.source_component_depth)
                description["lfsSourceComponentDepth"] = *frame.info.source_component_depth;
            core::image_codecs::TiffColorTags tags{
                .chromaticities = layout.color.primaries == ColorPrimaries::Bt709 ? std::array<float, 8>{.64f, .33f, .30f, .60f, .15f, .06f, .3127f, .3290f}
                                                                                  : std::array<float, 8>{.708f, .292f, .170f, .797f, .131f, .046f, .3127f, .3290f},
                .transfer_function = lookup};
            return core::image_codecs::write_tiff_u16(target, {reinterpret_cast<const uint8_t*>(pixels.data()), pixels.size() * 2},
                                                      layout.width, layout.height, int(channels), static_cast<size_t>(layout.width) * channels * 2,
                                                      description.dump(-1, ' ', true),
                                                      {.overwrite = options.overwrite, .durable = false, .create_directories = false, .cancelled = options.cancelled}, tags, xmp);
        },
                              "ImageOutput.writeTiff16");
    }
    SinkResult ImageOutput::writeExr(const std::filesystem::path& target, const FrameView& frame, const ExrOutputOptions& options) {
        return writeExr(target, frame, options, {});
    }
    SinkResult ImageOutput::writeExr(const std::filesystem::path& target, const FrameView& frame, const ExrOutputOptions& options, const std::string& xmp) {
        return outputBoundary([&]() -> SinkResult {
            if (auto valid = validateExrOutputOptions(options); !valid)
                return valid;
            if (xmp.size() > 32 * 1024 * 1024 || xmp.find('\0') != std::string::npos)
                return SinkResult::failure(outputError(ErrorCode::InvalidArgument, "Invalid EXR XMP packet"));
            auto required = frame.requiredBytes();
            if (!required)
                return SinkResult::failure(required.error());
            const auto& layout = frame.layout;
            const bool rgba = layout.format == FramePixelFormat::RGBAFloat32;
            if (outputOptionName(layout.color.units) == "unknown")
                return SinkResult::failure(outputError(ErrorCode::InvalidArgument, "Unknown EXR sample units"));
            if (layout.format != FramePixelFormat::RGBFloat32 && !rgba)
                return SinkResult::failure(outputError(ErrorCode::Unsupported, std::format("EXR requires float RGB/RGBA (format={})", int(layout.format))));
            if (layout.color.transfer != ColorTransfer::Linear || (layout.color.primaries != ColorPrimaries::Bt709 && layout.color.primaries != ColorPrimaries::Bt2020))
                return SinkResult::failure(outputError(ErrorCode::Unsupported, std::format("EXR requires linear samples and BT709/BT2020 primaries (transfer={}, primaries={})", int(layout.color.transfer), int(layout.color.primaries))));
            if ((!rgba && layout.color.alpha != AlphaMode::None) || (rgba && layout.color.alpha != AlphaMode::Independent && layout.color.alpha != AlphaMode::Premultiplied))
                return SinkResult::failure(outputError(ErrorCode::InvalidArgument, std::format("EXR alpha must match layout (rgba={}, alpha={})", rgba, int(layout.color.alpha))));
            if (target.empty() || target.filename().empty() || target.native().find(std::filesystem::path::value_type{}) != std::filesystem::path::string_type::npos ||
                (frame.info.origin != FrameOrigin::Unspecified && frame.info.origin != FrameOrigin::Decoded && frame.info.origin != FrameOrigin::Rendered && frame.info.origin != FrameOrigin::External))
                return SinkResult::failure(outputError(ErrorCode::InvalidArgument, std::format("Invalid EXR path/options/origin (path_bytes={}, provenance_bytes={}, precision={}, compression={}, origin={})", target.native().size(), options.provenance.size(), int(options.precision), int(options.compression), int(frame.info.origin))));
            const auto valid_time = [](const std::optional<Timestamp>& value) { return !value || (value->time_base.numerator > 0 && value->time_base.denominator > 0); };
            if (!valid_time(frame.info.source_timestamp) || !valid_time(frame.info.output_timestamp) || (frame.info.source_component_depth && *frame.info.source_component_depth <= 0))
                return SinkResult::failure(outputError(ErrorCode::InvalidArgument, std::format("Invalid EXR timing/precision (source_time_valid={}, output_time_valid={}, depth={})", valid_time(frame.info.source_timestamp), valid_time(frame.info.output_timestamp), frame.info.source_component_depth.value_or(-1))));
            std::vector<std::pair<std::string, std::string>> text{{"lfsTransfer", "linear"}, {"lfsUnits", std::string(outputOptionName(layout.color.units))}, {"lfsAlpha", rgba ? "premultiplied" : "none"}};
            if (!options.provenance.empty())
                text.emplace_back("lfsProvenance", options.provenance);
            if (frame.info.origin != FrameOrigin::Unspecified)
                text.emplace_back("lfsOrigin", frame.info.origin == FrameOrigin::Decoded ? "decoded" : frame.info.origin == FrameOrigin::Rendered ? "rendered"
                                                                                                                                                  : "external");
            if (frame.info.source_timestamp)
                text.emplace_back("lfsSourceTimestamp", timestampText(*frame.info.source_timestamp));
            if (frame.info.output_timestamp)
                text.emplace_back("lfsOutputTimestamp", timestampText(*frame.info.output_timestamp));
            if (frame.info.source_component_depth)
                text.emplace_back("lfsSourceComponentDepth", std::to_string(*frame.info.source_component_depth));
            namespace codecs = core::image_codecs;
            codecs::ExrEncodeOptions encoding{
                .precision = options.precision == ExrPrecision::Half ? codecs::ExrSampleType::Half : codecs::ExrSampleType::Float,
                .compression = options.compression == ExrCompression::ZIP ? codecs::ExrCompression::Zip : codecs::ExrCompression::None,
                .premultiply_alpha = rgba && layout.color.alpha == AlphaMode::Independent,
                .chromaticities = layout.color.primaries == ColorPrimaries::Bt709 ? std::array<float, 8>{.64f, .33f, .30f, .60f, .15f, .06f, .3127f, .3290f} : std::array<float, 8>{.708f, .292f, .170f, .797f, .131f, .046f, .3127f, .3290f},
                .text = text,
                .cancelled = options.cancelled};
            cancel(options);
            const auto name = core::path_to_utf8(target);
            auto status = core::writeFileAtomically(target, [&](FILE* stream) -> Status {
                auto encoded = codecs::encode_exr({layout.width, layout.height, rgba ? 4 : 3, layout.row_stride, frame.pixels}, {name, stream, writeBytes}, encoding, xmp);
                if (!encoded)
                    return Status::failure(std::move(encoded).error().with_context("ImageOutput.writeExr", LFS_SOURCE_SITE_CURRENT()));
                return {};
            },
                                                    {.overwrite = options.overwrite, .durable = false, .create_directories = false, .cancelled = options.cancelled});
            if (!status)
                return SinkResult::failure(std::move(status).error());
            return {};
        });
    }
} // namespace lfs::media
