/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "image_jxl.hpp"
#include "core/color_transfer.h"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

#include <jxl/cms.h>
#include <jxl/color_encoding.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/resizable_parallel_runner.h>
#include <lcms2.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <thread>

namespace lfs::core::image_codecs {
    namespace {
        bool fail(std::string& error, const char* reason) {
            error = std::string("JXL: ") + reason;
            return false;
        }

        struct CodecMemory {
            struct alignas(std::max_align_t) Header {
                size_t bytes;
            };
            size_t limit;
            std::atomic<size_t> live{0};
            std::atomic<bool> exhausted{false};
            static void* allocate(void* opaque, size_t bytes) noexcept {
                auto& memory = *static_cast<CodecMemory*>(opaque);
                size_t previous = memory.live.load();
                for (;;) {
                    if (bytes > memory.limit - previous || bytes > SIZE_MAX - sizeof(Header)) {
                        memory.exhausted = true;
                        return nullptr;
                    }
                    if (memory.live.compare_exchange_weak(previous, previous + bytes))
                        break;
                }
                auto* block = static_cast<Header*>(std::malloc(sizeof(Header) + bytes));
                if (!block) {
                    memory.live.fetch_sub(bytes);
                    memory.exhausted = true;
                    return nullptr;
                }
                block->bytes = bytes;
                return block + 1;
            }
            static void release(void* opaque, void* pointer) noexcept {
                if (!pointer)
                    return;
                auto* block = static_cast<Header*>(pointer) - 1;
                static_cast<CodecMemory*>(opaque)->live.fetch_sub(block->bytes);
                std::free(block);
            }
        };

        // Borrow memory directly; file input uses the same decoder event loop
        // with a bounded sliding window, never a whole-file staging copy.
        struct Input {
            std::ifstream file;
            std::span<const uint8_t> borrowed;
            std::vector<uint8_t> window;
            size_t window_size = 0;
            uint64_t remaining = 0;
            bool memory = false, supplied = false;
            bool opened = false;
            explicit Input(std::span<const uint8_t> bytes) : borrowed(bytes), memory(true) {}
            explicit Input(const std::filesystem::path& path) {
                if (lfs::core::open_file_for_read(path, std::ios::binary | std::ios::ate, file) && file.tellg() >= 0) {
                    opened = true;
                    remaining = static_cast<uint64_t>(file.tellg());
                    file.seekg(0);
                }
            }
            bool refill(JxlDecoder* decoder, const size_t limit, std::string& error) {
                const size_t retained = supplied ? JxlDecoderReleaseInput(decoder) : 0;
                if (memory) {
                    if (supplied || borrowed.empty())
                        return fail(error, "incomplete image buffer");
                    supplied = true;
                    if (JxlDecoderSetInput(decoder, borrowed.data(), borrowed.size()) != JXL_DEC_SUCCESS)
                        return fail(error, "cannot set input buffer");
                    JxlDecoderCloseInput(decoder);
                    return true;
                }
                if (!opened)
                    return fail(error, "cannot open source file");
                if (!remaining && !supplied)
                    return fail(error, "empty source file");
                if (!file || !remaining)
                    return fail(error, "cannot read complete source");
                if (retained > window_size)
                    return fail(error, "invalid buffered input state");
                if (retained)
                    std::memmove(window.data(), window.data() + window_size - retained, retained);
                if (window.empty() || retained == window.size()) {
                    const size_t next = std::min(limit, window.empty() ? size_t{64} << 10 : window.size() * 2);
                    if (next <= retained)
                        return fail(error, "buffered input byte limit exhausted");
                    window.resize(next);
                }
                const size_t count = static_cast<size_t>(std::min<uint64_t>(remaining, window.size() - retained));
                if (!file.read(reinterpret_cast<char*>(window.data() + retained), static_cast<std::streamsize>(count)))
                    return fail(error, "source truncated while reading");
                remaining -= count;
                window_size = retained + count;
                supplied = true;
                if (JxlDecoderSetInput(decoder, window.data(), window_size) != JXL_DEC_SUCCESS)
                    return fail(error, "cannot set file input");
                if (!remaining)
                    JxlDecoderCloseInput(decoder);
                return true;
            }
        };

        size_t sample_bytes(SampleType type) {
            return type == SampleType::Float32 ? sizeof(float) : type == SampleType::UInt16 ? sizeof(uint16_t)
                                                                                            : 1;
        }

        struct WorkerLease {
            static inline std::atomic<size_t> used{0};
            size_t count = 0;
            void acquire(size_t desired) {
                const size_t capacity = std::max(1u, std::thread::hardware_concurrency());
                size_t current = used.load();
                do {
                    count = std::min(desired, capacity - std::min(current, capacity));
                } while (!used.compare_exchange_weak(current, current + count));
            }
            ~WorkerLease() { used.fetch_sub(count); }
        };

        Status encode_error(ErrorCode code, std::string detail) {
            return Status::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = "JXL encode: " + std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()}));
        }

        template <class Writer>
        Status encode_image(std::span<const uint8_t> input, int width, int height, int channels,
                            SampleType type, size_t stride, const JxlEncodeOptions& options, std::string_view xmp, Writer&& write) {
            try {
                const auto cancelled = [&] { return options.cancelled && options.cancelled(); };
                if (cancelled())
                    return encode_error(ErrorCode::Cancelled, "cancelled before encoding");
                if (auto valid = validate_jxl_encode_options(options, type); !valid)
                    return valid;
                if (xmp.size() > 32u * 1024 * 1024 || xmp.find('\0') != std::string_view::npos)
                    return encode_error(ErrorCode::InvalidArgument, "XMP exceeds 32 MiB or contains NUL");
                const bool floating = type == SampleType::Float32;
                const bool alpha = channels == 2 || channels == 4;
                if (width <= 0 || height <= 0 || channels < 1 || channels > 4 ||
                    (type != SampleType::UInt8 && type != SampleType::UInt16 && !floating) ||
                    (options.premultiplied_alpha && (!floating || !alpha)))
                    return encode_error(ErrorCode::InvalidArgument, "invalid layout, color or encoder options");
                const size_t sample = sample_bytes(type);
                if (size_t(height) > options.pixels / size_t(width) ||
                    size_t(width) > SIZE_MAX / size_t(channels) / sample / size_t(height))
                    return encode_error(ErrorCode::ResourceExhausted, "pixel limit or packed extent exceeded");
                const size_t row = size_t(width) * channels * sample;
                if (stride < row || (size_t(height - 1) && stride > (SIZE_MAX - row) / size_t(height - 1)) ||
                    input.size() < size_t(height - 1) * stride + row)
                    return encode_error(ErrorCode::InvalidArgument, "invalid row stride or truncated pixel buffer");
                const size_t extent = row * size_t(height);
                // Packed borrowed input needs no additional whole-image copy.
                std::unique_ptr<uint8_t[]> packed;
                if (stride != row || options.premultiplied_alpha)
                    packed.reset(new uint8_t[extent]);
                if (floating || packed) {
                    for (int y = 0; y < height; ++y) {
                        if (cancelled())
                            return encode_error(ErrorCode::Cancelled, "cancelled while preparing rows");
                        const auto* source = input.data() + size_t(y) * stride;
                        if (!floating) {
                            std::memcpy(packed.get() + size_t(y) * row, source, row);
                            continue;
                        }
                        for (int x = 0; x < width; ++x) {
                            float values[4];
                            const size_t offset = size_t(x) * channels * sample;
                            std::memcpy(values, source + offset, channels * sample);
                            for (int c = 0; c < channels; ++c)
                                if (!std::isfinite(values[c]))
                                    return encode_error(ErrorCode::InvalidArgument, "nonfinite input sample");
                            const float a = alpha ? values[channels - 1] : 1.f;
                            if (alpha && (a < 0 || a > 1))
                                return encode_error(ErrorCode::InvalidArgument, "alpha must be in [0,1]");
                            if (options.premultiplied_alpha) {
                                for (int c = 0; c < channels - 1; ++c) {
                                    values[c] = a > 0 ? values[c] / a : 0;
                                    if (!std::isfinite(values[c]))
                                        return encode_error(ErrorCode::InvalidArgument, "nonfinite unassociated sample");
                                }
                            }
                            if (packed)
                                std::memcpy(packed.get() + size_t(y) * row + offset, values, channels * sample);
                        }
                    }
                }
                const uint8_t* pixels = packed ? packed.get() : input.data();
                CodecMemory memory{options.encoder_bytes};
                const JxlMemoryManager manager{&memory, CodecMemory::allocate, CodecMemory::release};
                WorkerLease workers;
                std::unique_ptr<void, decltype(&JxlResizableParallelRunnerDestroy)> runner(JxlResizableParallelRunnerCreate(nullptr), JxlResizableParallelRunnerDestroy);
                std::unique_ptr<JxlEncoder, decltype(&JxlEncoderDestroy)> encoder(JxlEncoderCreate(&manager), JxlEncoderDestroy);
                if (!encoder || !runner)
                    return encode_error(ErrorCode::ResourceExhausted, "cannot allocate encoder or runner");
                const auto* cms = JxlGetDefaultCms();
                if (!cms)
                    return encode_error(ErrorCode::Unavailable, "color management is unavailable");
                JxlEncoderSetCms(encoder.get(), *cms);
                const auto check = [&](JxlEncoderStatus status, const char* operation = "configuration") -> Status {
                    if (memory.exhausted)
                        return encode_error(ErrorCode::ResourceExhausted, "encoder allocation limit exhausted");
                    if (status != JXL_ENC_SUCCESS)
                        return encode_error(ErrorCode::Internal, std::string("provider failure during ") + operation + " (code=" + std::to_string(JxlEncoderGetError(encoder.get())) + ")");
                    return {};
                };
                workers.acquire(std::min<size_t>(JxlResizableParallelRunnerSuggestThreads(width, height), options.max_threads ? options.max_threads - 1 : size_t{8}));
                JxlResizableParallelRunnerSetThreads(runner.get(), workers.count);
                if (auto result = check(JxlEncoderSetParallelRunner(encoder.get(), JxlResizableParallelRunner, runner.get())); !result)
                    return result;
                JxlBasicInfo info;
                JxlEncoderInitBasicInfo(&info);
                info.xsize = width;
                info.ysize = height;
                info.bits_per_sample = sample * 8;
                info.exponent_bits_per_sample = floating ? 8 : 0;
                info.uses_original_profile = options.lossless;
                info.num_color_channels = channels <= 2 ? 1 : 3;
                info.num_extra_channels = alpha ? 1 : 0;
                info.alpha_bits = alpha ? info.bits_per_sample : 0;
                info.alpha_exponent_bits = alpha ? info.exponent_bits_per_sample : 0;
                if (auto result = check(JxlEncoderSetBasicInfo(encoder.get(), &info)); !result)
                    return result;
                JxlColorEncoding color;
                if (options.color == JxlEncodingColor::SRGB || options.color == JxlEncodingColor::SRGBBT2020)
                    JxlColorEncodingSetToSRGB(&color, channels <= 2);
                else
                    JxlColorEncodingSetToLinearSRGB(&color, channels <= 2);
                if (options.color == JxlEncodingColor::LinearBT2020 || options.color == JxlEncodingColor::SRGBBT2020)
                    color.primaries = JXL_PRIMARIES_2100;
                if (auto result = check(JxlEncoderSetColorEncoding(encoder.get(), &color)); !result)
                    return result;
                auto* frame = JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);
                if (!frame)
                    return encode_error(ErrorCode::ResourceExhausted, "cannot allocate frame settings");
                for (auto status : {JxlEncoderSetFrameLossless(frame, options.lossless),
                                    JxlEncoderSetFrameDistance(frame, options.lossless ? 0.f : options.distance),
                                    JxlEncoderFrameSettingsSetOption(frame, JXL_ENC_FRAME_SETTING_EFFORT, options.effort)})
                    if (auto result = check(status); !result)
                        return result;
                if (alpha) {
                    // Distance >=10 otherwise resamples alpha despite distance 0.
                    for (auto status : {JxlEncoderSetExtraChannelDistance(frame, 0, 0.f),
                                        JxlEncoderFrameSettingsSetOption(frame, JXL_ENC_FRAME_SETTING_RESAMPLING, 1),
                                        JxlEncoderFrameSettingsSetOption(frame, JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING, 1)})
                        if (auto result = check(status); !result)
                            return result;
                }
                if (floating && options.lossless)
                    // The default adaptive predictor in libjxl 0.12 can reject
                    // finite IEEE32 patterns with mixed signs. Left prediction
                    // keeps the full bit pattern without a lossy fallback.
                    if (auto result = check(JxlEncoderFrameSettingsSetOption(frame, JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 1)); !result)
                        return result;
                if (!xmp.empty()) {
                    if (auto result = check(JxlEncoderUseContainer(encoder.get(), JXL_TRUE), "metadata container"); !result)
                        return result;
                    if (auto result = check(JxlEncoderUseBoxes(encoder.get()), "metadata configuration"); !result)
                        return result;
                    const JxlBoxType box{'x', 'm', 'l', ' '};
                    if (auto result = check(JxlEncoderAddBox(encoder.get(), box, reinterpret_cast<const uint8_t*>(xmp.data()), xmp.size(), JXL_FALSE), "XMP submission"); !result)
                        return result;
                }
                const JxlPixelFormat format{static_cast<uint32_t>(channels), floating ? JXL_TYPE_FLOAT : type == SampleType::UInt16 ? JXL_TYPE_UINT16
                                                                                                                                    : JXL_TYPE_UINT8,
                                            JXL_NATIVE_ENDIAN, 0};
                if (auto result = check(JxlEncoderAddImageFrame(frame, &format, pixels, extent), "pixel submission"); !result)
                    return result;
                JxlEncoderCloseInput(encoder.get());
                std::array<uint8_t, 64 * 1024> buffer;
                size_t total = 0;
                const size_t output_limit = options.max_output_bytes ? options.max_output_bytes : size_t{256} * 1024 * 1024;
                for (;;) {
                    if (cancelled())
                        return encode_error(ErrorCode::Cancelled, "cancelled while producing output");
                    uint8_t* next = buffer.data();
                    size_t available = buffer.size();
                    const auto status = JxlEncoderProcessOutput(encoder.get(), &next, &available);
                    if (memory.exhausted)
                        return encode_error(ErrorCode::ResourceExhausted, "encoder allocation limit exhausted");
                    const size_t count = buffer.size() - available;
                    if (count > output_limit - total)
                        return encode_error(ErrorCode::ResourceExhausted, "encoded output byte limit exhausted");
                    if (count) {
                        if (auto result = write(std::span<const uint8_t>(buffer.data(), count)); !result)
                            return result;
                        total += count;
                    }
                    if (status == JXL_ENC_SUCCESS)
                        return cancelled() ? encode_error(ErrorCode::Cancelled, "cancelled before completing output") : Status{};
                    if (status != JXL_ENC_NEED_MORE_OUTPUT)
                        return check(status, "output production");
                }
            } catch (const std::bad_alloc&) {
                return encode_error(ErrorCode::ResourceExhausted, "allocation failed");
            } catch (const std::exception& exception) {
                // LFS-CENSUS-OK(empty-catch): retain encoder/callback diagnostics at the synchronous Status boundary.
                return encode_error(ErrorCode::Internal, exception.what());
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): no callback exception escapes the synchronous encoding boundary.
                return encode_error(ErrorCode::Internal, "nonstandard callback exception");
            }
        }

        bool validate_sdr_icc(std::span<const uint8_t> bytes, std::string& error) {
            using Profile = std::unique_ptr<void, decltype(&cmsCloseProfile)>;
            Profile profile(cmsOpenProfileFromMem(bytes.data(), static_cast<cmsUInt32Number>(bytes.size())), cmsCloseProfile);
            if (!profile)
                return fail(error, "invalid ICC profile");
            if (const auto* cicp = static_cast<const cmsVideoSignalType*>(cmsReadTag(profile.get(), cmsSigcicpTag));
                cicp && (cicp->TransferCharacteristics == 16 || cicp->TransferCharacteristics == 18))
                return fail(error, "HDR display conversion requires an explicit tone-mapping policy");
            const cmsTagSignature tags[]{cmsSigRedTRCTag, cmsSigGreenTRCTag, cmsSigBlueTRCTag, cmsSigGrayTRCTag};
            for (auto tag : tags) {
                const auto* curve = static_cast<const cmsToneCurve*>(cmsReadTag(profile.get(), tag));
                if (!curve)
                    continue;
                double pq_error = 0., hlg_error = 0.;
                for (int i = 1; i < 32; ++i) {
                    const float x = i / 32.f;
                    const float y = cmsEvalToneCurveFloat(curve, x);
                    if (!std::isfinite(y))
                        return fail(error, "invalid ICC transfer curve");
                    pq_error = std::max(pq_error, double(std::abs(y - lfs::core::color::pqToLinear(x))));
                    hlg_error = std::max(hlg_error, double(std::abs(y - lfs::core::color::hlgToSceneLinear(x))));
                }
                if (pq_error < .003 || hlg_error < .003)
                    return fail(error, "HDR display conversion requires an explicit tone-mapping policy");
            }
            return true;
        }

        bool header_layout(const JxlBasicInfo& basic, Probe& info, const JxlDecodeLimits& limits, std::string& error, bool native_output = true) {
            if (!basic.xsize || !basic.ysize || basic.xsize > INT_MAX || basic.ysize > INT_MAX ||
                basic.ysize > limits.pixels / basic.xsize)
                return fail(error, "image dimensions exceed pixel limit");
            if (basic.have_animation)
                return fail(error, "animation is not supported by the still-image loader");
            if (basic.num_color_channels != 1 && basic.num_color_channels != 3)
                return fail(error, "unsupported color channel layout");
            if (!basic.bits_per_sample || basic.bits_per_sample > 32 || basic.exponent_bits_per_sample > 8 ||
                (!basic.exponent_bits_per_sample && basic.bits_per_sample > 16) ||
                basic.alpha_bits > 32 || basic.alpha_exponent_bits > 8 ||
                (!basic.alpha_exponent_bits && basic.alpha_bits > 16))
                return fail(error, "unsupported sample precision");
            const auto type = basic.exponent_bits_per_sample || basic.alpha_exponent_bits ? SampleType::Float32
                              : std::max(basic.bits_per_sample, basic.alpha_bits) > 8     ? SampleType::UInt16
                                                                                          : SampleType::UInt8;
            info = {static_cast<int>(basic.xsize), static_cast<int>(basic.ysize),
                    static_cast<int>(basic.num_color_channels + (basic.alpha_bits ? 1 : 0)), type};
            const size_t pixels = size_t(info.width) * info.height;
            if (native_output && pixels > limits.output_bytes / (info.channels * sample_bytes(type)))
                return fail(error, "decoded output byte limit exhausted");
            return true;
        }

        bool set_color(JxlDecoder* decoder, const JxlBasicInfo& basic, bool display,
                       const JxlDecodeLimits& limits, std::string& error) {
            JxlColorEncoding original{};
            const bool structured = JxlDecoderGetColorAsEncodedProfile(decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &original) == JXL_DEC_SUCCESS;
            size_t profile_bytes = 0;
            if (JxlDecoderGetICCProfileSize(decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &profile_bytes) != JXL_DEC_SUCCESS ||
                profile_bytes > limits.icc_bytes)
                return fail(error, "unsupported or oversized color profile");
            std::vector<uint8_t> profile;
            const bool convert = display || limits.color_policy == JxlColorPolicy::SRGB;
            if (!structured && convert) {
                profile.resize(profile_bytes);
                if (profile.empty() || JxlDecoderGetColorAsICCProfile(decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, profile.data(), profile.size()) != JXL_DEC_SUCCESS)
                    return fail(error, "invalid color profile");
                if (!validate_sdr_icc(profile, error))
                    return false;
            }
            if (convert) {
                if (basic.intensity_target > 255.f || (structured && (original.transfer_function == JXL_TRANSFER_FUNCTION_PQ || original.transfer_function == JXL_TRANSFER_FUNCTION_HLG)))
                    return fail(error, "HDR display conversion requires an explicit tone-mapping policy");
                JxlColorEncoding srgb{};
                srgb.color_space = basic.num_color_channels == 1 ? JXL_COLOR_SPACE_GRAY : JXL_COLOR_SPACE_RGB;
                srgb.white_point = JXL_WHITE_POINT_D65;
                srgb.primaries = JXL_PRIMARIES_SRGB;
                srgb.transfer_function = JXL_TRANSFER_FUNCTION_SRGB;
                srgb.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
                if (JxlDecoderSetOutputColorProfile(decoder, &srgb, nullptr, 0) != JXL_DEC_SUCCESS)
                    return fail(error, "cannot convert display color profile");
            }
            return true;
        }

        bool run(Input& input, Image& image, Probe& info, std::string& error,
                 bool probe_only, bool display, const JxlDecodeLimits& limits, DecodeTarget* requested = nullptr) {
            if ((input.memory ? input.borrowed.size() : input.remaining) > limits.encoded_bytes)
                return fail(error, "encoded input byte limit exhausted");
            CodecMemory memory{limits.decoder_bytes};
            const JxlMemoryManager manager{&memory, CodecMemory::allocate, CodecMemory::release};
            WorkerLease workers;
            std::unique_ptr<void, decltype(&JxlResizableParallelRunnerDestroy)> runner(JxlResizableParallelRunnerCreate(nullptr), JxlResizableParallelRunnerDestroy);
            std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> decoder(JxlDecoderCreate(&manager), JxlDecoderDestroy);
            if (!decoder)
                return fail(error, "decoder allocation limit exhausted");
            const int events = JXL_DEC_BASIC_INFO | (probe_only ? 0 : JXL_DEC_COLOR_ENCODING | JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE | JXL_DEC_BOX);
            const auto* cms = JxlGetDefaultCms();
            if (!cms || JxlDecoderSubscribeEvents(decoder.get(), events) != JXL_DEC_SUCCESS ||
                JxlDecoderSetCoalescing(decoder.get(), JXL_TRUE) != JXL_DEC_SUCCESS ||
                JxlDecoderSetKeepOrientation(decoder.get(), JXL_FALSE) != JXL_DEC_SUCCESS ||
                JxlDecoderSetUnpremultiplyAlpha(decoder.get(), JXL_TRUE) != JXL_DEC_SUCCESS ||
                JxlDecoderSetRenderSpotcolors(decoder.get(), JXL_FALSE) != JXL_DEC_SUCCESS ||
                ((display || limits.color_policy == JxlColorPolicy::SRGB) && JxlDecoderSetCms(decoder.get(), *cms) != JXL_DEC_SUCCESS) ||
                !runner || JxlDecoderSetParallelRunner(decoder.get(), JxlResizableParallelRunner, runner.get()) != JXL_DEC_SUCCESS)
                return fail(error, "cannot configure decoder");
            JxlBasicInfo basic{};
            Image candidate;
            std::unique_ptr<uint8_t[]> staging;
            uint8_t* pixels = nullptr;
            size_t output_bytes = 0;
            unsigned frames = 0, images = 0, boxes = 0;
            bool header = false, color = false;
            for (;;) {
                const auto status = JxlDecoderProcessInput(decoder.get());
                if (memory.exhausted)
                    return fail(error, "decoder allocation limit exhausted");
                switch (status) {
                case JXL_DEC_NEED_MORE_INPUT:
                    if (!input.refill(decoder.get(), limits.buffered_input_bytes, error))
                        return false;
                    break;
                case JXL_DEC_BASIC_INFO:
                    if (JxlDecoderGetBasicInfo(decoder.get(), &basic) != JXL_DEC_SUCCESS || !header_layout(basic, info, limits, error, !requested))
                        return error.empty() ? fail(error, "invalid basic information") : false;
                    if (!probe_only) {
                        const size_t suggested = JxlResizableParallelRunnerSuggestThreads(basic.xsize, basic.ysize);
                        const size_t cap = limits.max_threads ? limits.max_threads - 1 : size_t{8};
                        workers.acquire(std::min(suggested, cap));
                        JxlResizableParallelRunnerSetThreads(runner.get(), workers.count);
                    }
                    if (requested) {
                        const int channels = requested->channels ? requested->channels : info.channels;
                        if (channels < 1 || channels > 4 || (channels <= 2 && basic.num_color_channels == 3) ||
                            (requested->sample_type != SampleType::UInt8 && requested->sample_type != SampleType::UInt16 && requested->sample_type != SampleType::Float32))
                            return fail(error, "unsupported target layout");
                        info.channels = channels;
                        info.sample_type = requested->sample_type;
                        if (size_t(info.width) * info.height > limits.output_bytes / (size_t(info.channels) * sample_bytes(info.sample_type)))
                            return fail(error, "decoded output byte limit exhausted");
                    }
                    if (basic.num_extra_channels != (basic.alpha_bits ? 1u : 0u))
                        return fail(error, "auxiliary channels require an explicit import policy");
                    header = true;
                    if (probe_only)
                        return true;
                    break;
                case JXL_DEC_COLOR_ENCODING:
                    if (!header || !set_color(decoder.get(), basic, display, limits, error))
                        return false;
                    color = true;
                    break;
                case JXL_DEC_FRAME: {
                    JxlFrameHeader frame{};
                    if (++frames != 1 || JxlDecoderGetFrameHeader(decoder.get(), &frame) != JXL_DEC_SUCCESS || !frame.is_last)
                        return fail(error, "additional image frames are unsupported");
                    break;
                }
                case JXL_DEC_NEED_IMAGE_OUT_BUFFER: {
                    if (!header || !color || frames != 1)
                        return fail(error, "invalid image event order");
                    const JxlPixelFormat format{static_cast<uint32_t>(info.channels),
                                                info.sample_type == SampleType::Float32 ? JXL_TYPE_FLOAT : info.sample_type == SampleType::UInt16 ? JXL_TYPE_UINT16
                                                                                                                                                  : JXL_TYPE_UINT8,
                                                JXL_NATIVE_ENDIAN, 0};
                    size_t bytes = 0;
                    const size_t expected = size_t(info.width) * info.height * info.channels * sample_bytes(info.sample_type);
                    if (JxlDecoderImageOutBufferSize(decoder.get(), &format, &bytes) != JXL_DEC_SUCCESS || bytes != expected || bytes > limits.output_bytes)
                        return fail(error, "inconsistent output buffer size");
                    candidate = {info.width, info.height, info.channels, info.sample_type, {}};
                    if (requested) {
                        staging.reset(new uint8_t[bytes]);
                        pixels = staging.get();
                    } else {
                        candidate.data.resize(bytes);
                        pixels = candidate.data.data();
                    }
                    output_bytes = bytes;
                    if (JxlDecoderSetImageOutBuffer(decoder.get(), &format, pixels, bytes) != JXL_DEC_SUCCESS)
                        return fail(error, "cannot set output buffer");
                    break;
                }
                case JXL_DEC_FULL_IMAGE:
                    if (++images != 1)
                        return fail(error, "additional images are unsupported");
                    break;
                case JXL_DEC_BOX:
                    if (++boxes > 1024)
                        return fail(error, "container box limit exhausted");
                    break;
                case JXL_DEC_SUCCESS:
                    if (!header || !color || frames != 1 || images != 1 || !output_bytes)
                        return fail(error, "incomplete still image");
                    if ((!input.memory && input.remaining) || JxlDecoderReleaseInput(decoder.get()))
                        return fail(error, "unprocessed trailing input");
                    if (candidate.sample_type == SampleType::Float32)
                        for (size_t offset = 0; offset < output_bytes; offset += sizeof(float)) {
                            float value;
                            std::memcpy(&value, pixels + offset, sizeof(value));
                            if (!std::isfinite(value))
                                return fail(error, "nonfinite decoded sample");
                        }
                    if (requested) {
                        void* destination = requested->data ? requested->data : requested->allocate ? requested->allocate(output_bytes, requested->user)
                                                                                                    : std::malloc(output_bytes);
                        if (!destination)
                            return fail(error, "cannot allocate target");
                        std::memcpy(destination, staging.get(), output_bytes);
                        requested->data = destination;
                    } else {
                        image = std::move(candidate);
                    }
                    return true;
                default:
                    if (status == JXL_DEC_ERROR && images == 1)
                        return fail(error, "malformed trailing input after complete image");
                    error = "JXL: malformed or incomplete codestream (status=" + std::to_string(status) + ")";
                    return false;
                }
            }
        }

        template <class Function>
        bool guarded(std::string& error, Function&& function) {
            error.clear();
            try {
                return function();
            } catch (const std::bad_alloc&) {
                return fail(error, "allocation failed");
            } catch (const std::exception& exception) {
                // LFS-CENSUS-OK(empty-catch): This synchronous bool/error codec
                // boundary preserves the exception diagnostic for its caller.
                error = std::string("JXL: ") + exception.what();
                return false;
            }
        }

        template <class Source>
        bool target_decode(Source source, DecodeTarget& target, Probe& result, std::string& error, bool display, const JxlDecodeLimits& limits) {
            return guarded(error, [&] {
                Image image;
                Input input(source);
                Probe info;
                if (!run(input, image, info, error, false, display, limits, &target))
                    return false;
                result = info;
                return true;
            });
        }
    } // namespace

    bool is_jxl(std::span<const uint8_t> bytes) {
        if (bytes.empty())
            return false;
        const auto signature = JxlSignatureCheck(bytes.data(), bytes.size());
        return signature == JXL_SIG_CODESTREAM || signature == JXL_SIG_CONTAINER;
    }

    namespace {
        enum class MetadataCarrier { None,
                                     Exif,
                                     Xml };
    } // namespace

    static SourceImageMetadata scan_jxl_metadata(Input& input, const MetadataLimits& exif_limits, const XmpLimits* xml_limits,
                                                 bool collect_exif, MetadataCarrier& damaged) {
        SourceImageMetadata result;
        result.exif.scope = "jxl-container-exif";
        result.exif.status = MetadataStatus::Absent;
        if (xml_limits) {
            result.xmp.scope = "jxl-container-xml";
            result.xmp.status = MetadataStatus::Absent;
        }
        bool exif_stopped = !collect_exif, xml_stopped = !xml_limits;
        const auto failure = [&](MetadataStatus status, const char* message) {
            // A carrier stopped at its own quota retains that completed snapshot.
            if (!exif_stopped) {
                result.exif.status = status;
                result.exif.diagnostic = message;
            }
            if (!xml_stopped) {
                result.xmp.status = status;
                result.xmp.diagnostic = message;
            }
        };
        if (collect_exif && (!exif_limits.block_bytes || exif_limits.block_bytes > 16u << 20 ||
                             !exif_limits.total_bytes || exif_limits.total_bytes > 64u << 20 ||
                             !exif_limits.blocks || exif_limits.blocks > 4096)) {
            result.exif.status = MetadataStatus::Limited;
            result.exif.diagnostic = "Invalid JPEG XL EXIF collection limits";
            exif_stopped = true;
            if (xml_stopped)
                return result;
        }
        if (!input.opened) {
            failure(MetadataStatus::IoError, "JXL: cannot open metadata source");
            return result;
        }
        std::array<uint8_t, 12> signature{};
        input.file.read(reinterpret_cast<char*>(signature.data()), signature.size());
        const auto signature_size = static_cast<size_t>(input.file.gcount());
        if (input.file.bad()) {
            failure(MetadataStatus::IoError, "JXL: cannot read metadata source");
            return result;
        }
        if (JxlSignatureCheck(signature.data(), signature_size) == JXL_SIG_CODESTREAM)
            return result;
        input.file.clear();
        input.file.seekg(0);
        if (input.remaining > (uint64_t{2} << 30)) {
            failure(MetadataStatus::Limited, "JXL: metadata encoded-input limit exhausted");
            return result;
        }
        CodecMemory memory{64u << 20};
        const JxlMemoryManager manager{&memory, CodecMemory::allocate, CodecMemory::release};
        // Output storage must outlive the decoder even on an exceptional exit.
        std::vector<uint8_t> pending;
        std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> decoder(JxlDecoderCreate(&manager), JxlDecoderDestroy);
        if (!decoder) {
            failure(MetadataStatus::Limited, "JXL: metadata decoder allocation limit exhausted");
            return result;
        }
        if (JxlDecoderSubscribeEvents(decoder.get(), JXL_DEC_BOX | JXL_DEC_BOX_COMPLETE) != JXL_DEC_SUCCESS) {
            failure(MetadataStatus::Malformed, "JXL: cannot configure box collection");
            return result;
        }
        // Configure before processing: libjxl starts each box decoder before emitting its box event.
        const bool decompress = JxlDecoderSetDecompressBoxes(decoder.get(), JXL_TRUE) == JXL_DEC_SUCCESS;
        using Carrier = MetadataCarrier;
        Carrier active = Carrier::None;
        size_t exif_retained = 0, xml_retained = 0, boxes = 0, budget = 0, written = 0, supplied = 0;
        bool compressed = false;
        uint64_t ordinal = 0;
        const auto carrier_failure = [&](Carrier carrier, MetadataStatus status, const char* message, bool stop) {
            if (carrier == Carrier::Exif) {
                result.exif.status = status;
                result.exif.diagnostic = message;
                exif_stopped = stop;
            } else {
                result.xmp.status = status;
                result.xmp.diagnostic = message;
                xml_stopped = stop;
            }
        };
        const auto finish = [&] {
            pending.resize(written);
            if (active == Carrier::Exif) {
                ExifBlock block;
                block.envelope = ExifEnvelope::JxlOffset;
                block.jxl_box = JxlBox{ordinal, compressed};
                block.bytes = std::move(pending);
                exif_retained += written;
                result.exif.blocks.push_back(std::move(block));
                if (result.exif.status == MetadataStatus::Absent)
                    result.exif.status = MetadataStatus::Collected;
            } else {
                XmpBlock block;
                block.envelope = XmpEnvelope::JxlXml;
                block.jxl_box = JxlBox{ordinal, compressed};
                block.bytes = std::move(pending);
                xml_retained += written;
                result.xmp.blocks.push_back(std::move(block));
                if (result.xmp.status == MetadataStatus::Absent)
                    result.xmp.status = MetadataStatus::Collected;
            }
            active = Carrier::None;
        };
        const auto release_output = [&] {
            const auto unused = JxlDecoderReleaseBoxBuffer(decoder.get());
            written += supplied - unused;
            supplied = 0;
            if (written <= budget)
                return true;
            carrier_failure(active, MetadataStatus::Limited,
                            active == Carrier::Exif ? "JXL: EXIF decoded-byte budget exhausted" : "JXL: XMP decoded-byte budget exhausted", true);
            active = Carrier::None;
            return false;
        };
        for (;;) {
            const auto status = JxlDecoderProcessInput(decoder.get());
            if (memory.exhausted) {
                failure(MetadataStatus::Limited, "JXL: metadata decoder allocation limit exhausted");
                return result;
            }
            if (status == JXL_DEC_NEED_MORE_INPUT) {
                std::string error;
                if (!input.refill(decoder.get(), 4u << 20, error)) {
                    failure(input.file.bad() ? MetadataStatus::IoError : error == "JXL: buffered input byte limit exhausted" ? MetadataStatus::Limited
                                                                                                                             : MetadataStatus::Malformed,
                            error.c_str());
                    return result;
                }
            } else if (status == JXL_DEC_BOX_COMPLETE) {
                // Commit before parsing the next header: it may be malformed.
                // Completion events are not new boxes and must not change ordinals.
                if (active != Carrier::None && release_output())
                    finish();
                if (exif_stopped && xml_stopped)
                    return result;
            } else if (status == JXL_DEC_BOX || status == JXL_DEC_SUCCESS) {
                if (active != Carrier::None && release_output())
                    finish();
                if (status == JXL_DEC_SUCCESS || (exif_stopped && xml_stopped))
                    return result;
                if (boxes++ >= 4096) {
                    failure(MetadataStatus::Limited, "JXL: metadata box enumeration limit exhausted");
                    return result;
                }
                JxlBoxType raw{}, type{};
                if (JxlDecoderGetBoxType(decoder.get(), raw, JXL_FALSE) != JXL_DEC_SUCCESS ||
                    JxlDecoderGetBoxType(decoder.get(), type, JXL_TRUE) != JXL_DEC_SUCCESS) {
                    failure(MetadataStatus::Malformed, "JXL: invalid metadata box type");
                    return result;
                }
                const Carrier carrier = !std::memcmp(type, "Exif", 4) ? Carrier::Exif : !std::memcmp(type, "xml ", 4) ? Carrier::Xml
                                                                                                                      : Carrier::None;
                if (carrier == Carrier::None || (carrier == Carrier::Exif ? exif_stopped : xml_stopped))
                    continue;
                const bool exif = carrier == Carrier::Exif;
                const auto count_limit = exif ? exif_limits.blocks : xml_limits->blocks;
                if ((exif ? result.exif.blocks.size() : result.xmp.blocks.size()) >= count_limit) {
                    carrier_failure(carrier, MetadataStatus::Limited,
                                    exif ? "JXL: EXIF block count budget exhausted" : "JXL: XMP block count budget exhausted", true);
                    if (exif_stopped && xml_stopped)
                        return result;
                    continue;
                }
                compressed = !std::memcmp(raw, "brob", 4);
                ordinal = boxes - 1;
                budget = exif ? std::min(exif_limits.block_bytes, exif_limits.total_bytes - exif_retained)
                              : std::min(xml_limits->block_bytes, xml_limits->total_bytes - xml_retained);
                if (compressed && !exif)
                    budget = std::min(budget, xml_limits->inflated_bytes);
                uint64_t contents = 0;
                if (JxlDecoderGetBoxSizeContents(decoder.get(), &contents) != JXL_DEC_SUCCESS) {
                    failure(MetadataStatus::Malformed, "JXL: invalid metadata box size");
                    return result;
                }
                if (!compressed && contents > budget) {
                    carrier_failure(carrier, MetadataStatus::Limited,
                                    exif ? "JXL: EXIF collection byte budget exhausted" : "JXL: XMP collection byte budget exhausted", false);
                    continue;
                }
                if (compressed && !decompress) {
                    carrier_failure(carrier, MetadataStatus::Unsupported, "JXL: compressed metadata is unsupported by this provider", true);
                    if (exif_stopped && xml_stopped)
                        return result;
                    continue;
                }
                written = 0;
                pending.resize(std::min(size_t{4096}, compressed ? budget + 1 : static_cast<size_t>(std::min<uint64_t>(contents, budget)) + 1));
                supplied = pending.size();
                if (JxlDecoderSetBoxBuffer(decoder.get(), pending.data(), supplied) != JXL_DEC_SUCCESS) {
                    failure(MetadataStatus::Malformed, "JXL: cannot set metadata box buffer");
                    return result;
                }
                active = carrier;
            } else if (status == JXL_DEC_BOX_NEED_MORE_OUTPUT && active != Carrier::None) {
                if (!release_output()) {
                    if (exif_stopped && xml_stopped)
                        return result;
                    continue;
                }
                const auto next = std::min(budget + 1, std::max(pending.size() + 1, pending.size() * 2));
                pending.resize(next);
                supplied = next - written;
                if (JxlDecoderSetBoxBuffer(decoder.get(), pending.data() + written, supplied) != JXL_DEC_SUCCESS) {
                    failure(MetadataStatus::Malformed, "JXL: cannot extend metadata box buffer");
                    return result;
                }
            } else {
                if (status == JXL_DEC_ERROR && active != Carrier::None && compressed) {
                    carrier_failure(active, MetadataStatus::Malformed, "JXL: invalid or incomplete compressed metadata box", true);
                    if (active == Carrier::Exif ? !xml_stopped : !exif_stopped)
                        damaged = active;
                    return result;
                }
                failure(MetadataStatus::Malformed, "JXL: invalid or incomplete metadata container");
                return result;
            }
        }
    }

    SourceImageMetadata read_jxl_metadata(const std::filesystem::path& path, const MetadataLimits& exif_limits, const XmpLimits* xml_limits) {
        Input input(path);
        const auto encoded_bytes = input.remaining;
        MetadataCarrier damaged = MetadataCarrier::None;
        auto result = scan_jxl_metadata(input, exif_limits, xml_limits, true, damaged);
        if (damaged == MetadataCarrier::None)
            return result;

        // A fresh decoder can skip the damaged carrier without decompressing it.
        // Retry only the other carrier, once, with the same limits and open file.
        input.file.clear();
        input.file.seekg(0);
        if (!input.file) {
            if (damaged == MetadataCarrier::Exif) {
                result.xmp.status = MetadataStatus::IoError;
                result.xmp.diagnostic = "JXL: cannot rewind metadata source";
            } else {
                result.exif.status = MetadataStatus::IoError;
                result.exif.diagnostic = "JXL: cannot rewind metadata source";
            }
            return result;
        }
        input.remaining = encoded_bytes;
        input.window_size = 0;
        input.supplied = false;
        MetadataCarrier ignored = MetadataCarrier::None;
        if (damaged == MetadataCarrier::Exif) {
            // Release the superseded payloads before collecting replacements.
            result.xmp = {};
            result.xmp = scan_jxl_metadata(input, exif_limits, xml_limits, false, ignored).xmp;
        } else {
            result.exif = {};
            result.exif = scan_jxl_metadata(input, exif_limits, nullptr, true, ignored).exif;
        }
        return result;
    }

    uint32_t jxl_encoder_version() { return JxlEncoderVersion(); }

    Status validate_jxl_encode_options(const JxlEncodeOptions& options) {
        if (!std::isfinite(options.distance) || options.distance <= 0 || options.distance > 25 || options.effort < 1 || options.effort > 9 ||
            (options.color != JxlEncodingColor::SRGB && options.color != JxlEncodingColor::LinearSRGB && options.color != JxlEncodingColor::LinearBT2020 && options.color != JxlEncodingColor::SRGBBT2020))
            return encode_error(ErrorCode::InvalidArgument, "invalid color, distance (must be in (0,25]) or effort (must be in [1,9])");
        return {};
    }

    Status validate_jxl_encode_options(const JxlEncodeOptions& options, SampleType type) {
        if (auto valid = validate_jxl_encode_options(options); !valid)
            return valid;
        if (type == SampleType::Float32 && options.lossless && options.effort < 4)
            return encode_error(ErrorCode::InvalidArgument, "lossless Float32 requires effort in [4,9] with libjxl 0.12; efforts 1-3 remain available for integer or lossy input");
        return {};
    }

    Status encode_jxl_memory(std::span<const uint8_t> input, int width, int height, int channels, SampleType type,
                             size_t stride, std::vector<uint8_t>& output, const JxlEncodeOptions& options, std::string_view xmp) {
        std::vector<uint8_t> candidate;
        auto status = encode_image(input, width, height, channels, type, stride, options, xmp, [&](std::span<const uint8_t> bytes) -> Status {
            candidate.insert(candidate.end(), bytes.begin(), bytes.end());
            return {};
        });
        if (status)
            output = std::move(candidate);
        return status;
    }

    Status write_jxl(const std::filesystem::path& path, std::span<const uint8_t> input, int width, int height,
                     int channels, SampleType type, size_t stride, const JxlEncodeOptions& options, const AtomicFileOptions& publication, std::string_view xmp) {
        try {
            if (auto valid = validate_jxl_encode_options(options, type); !valid)
                return valid;
            auto encoding = options;
            if (!encoding.max_output_bytes) {
                constexpr size_t minimum = size_t{256} * 1024 * 1024;
                constexpr size_t overhead = 1024 * 1024;
                encoding.max_output_bytes = minimum;
                // Packed raw extent, not stride padding. Invalid views are rejected
                // by encode_image; arithmetic here must remain safe before that.
                const size_t sample = sample_bytes(type);
                if (width > 0 && height > 0 && channels >= 1 && channels <= 4 && sample &&
                    size_t(width) <= (SIZE_MAX - overhead) / 2 / size_t(height) / size_t(channels) / sample) {
                    const size_t raw = size_t(width) * size_t(height) * size_t(channels) * sample;
                    encoding.max_output_bytes = std::max(minimum, raw * 2 + overhead);
                }
            }
            encoding.cancelled = [a = options.cancelled, b = publication.cancelled] { return (a && a()) || (b && b()); };
            return writeFileAtomically(path, [&](FILE* stream) { return encode_image(input, width, height, channels, type, stride, encoding, xmp, [&](std::span<const uint8_t> bytes) -> Status {
                                                                     if (std::fwrite(bytes.data(), 1, bytes.size(), stream) != bytes.size()) {
                                                                         const int native = errno;
                                                                         return Status::failure(make_error({.code = ErrorCode::Unavailable, .domain = ErrorDomain::IO, .detail = "Cannot write JXL output (errno=" + std::to_string(native) + ")", .detection = LFS_SOURCE_SITE_CURRENT(), .native = NativeError{ErrorDomain::IO, native, "stdio"}}));
                                                                     }
                                                                     return {};
                                                                 }); }, publication);
        } catch (const std::bad_alloc&) {
            return encode_error(ErrorCode::ResourceExhausted, "allocation failed while preparing publication");
        } catch (const std::exception& exception) {
            // LFS-CENSUS-OK(empty-catch): preserve publication setup diagnostics at the Status boundary.
            return encode_error(ErrorCode::Internal, exception.what());
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): report callback setup exceptions without changing the destination.
            return encode_error(ErrorCode::Internal, "nonstandard publication exception");
        }
    }

    bool probe_jxl(const std::filesystem::path& path, Probe& result, std::string& error, const JxlDecodeLimits& limits) {
        return guarded(error, [&] {
            Input input(path);
            Image ignored;
            Probe candidate;
            if (!run(input, ignored, candidate, error, true, false, limits))
                return false;
            result = candidate;
            return true;
        });
    }

    bool decode_jxl(const std::filesystem::path& path, Image& result, std::string& error, bool display, const JxlDecodeLimits& limits) {
        return guarded(error, [&] { Input input(path); Probe info; return run(input, result, info, error, false, display, limits); });
    }
    bool decode_jxl(std::span<const uint8_t> bytes, Image& result, std::string& error, bool display, const JxlDecodeLimits& limits) {
        return guarded(error, [&] { Input input(bytes); Probe info; return run(input, result, info, error, false, display, limits); });
    }
    bool decode_jxl_to_buffer(const std::filesystem::path& path, DecodeTarget& target, Probe& result, std::string& error, bool display, const JxlDecodeLimits& limits) {
        return target_decode(path, target, result, error, display, limits);
    }
    bool decode_jxl_to_buffer(std::span<const uint8_t> bytes, DecodeTarget& target, Probe& result, std::string& error, bool display, const JxlDecodeLimits& limits) {
        return target_decode(bytes, target, result, error, display, limits);
    }
} // namespace lfs::core::image_codecs
