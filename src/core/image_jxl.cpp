/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "image_jxl.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

#include <jxl/cms.h>
#include <jxl/color_encoding.h>
#include <jxl/decode.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <new>

namespace lfs::core::image_codecs {
    namespace {
        bool fail(std::string& error, const char* reason) {
            error = std::string("JXL: ") + reason;
            return false;
        }

        struct DecoderMemory {
            struct alignas(std::max_align_t) Header {
                size_t bytes;
            };
            size_t limit, live = 0;
            bool exhausted = false;
            static void* allocate(void* opaque, size_t bytes) noexcept {
                auto& memory = *static_cast<DecoderMemory*>(opaque);
                if (bytes > memory.limit - memory.live || bytes > SIZE_MAX - sizeof(Header)) {
                    memory.exhausted = true;
                    return nullptr;
                }
                auto* block = static_cast<Header*>(std::malloc(sizeof(Header) + bytes));
                if (!block) {
                    memory.exhausted = true;
                    return nullptr;
                }
                block->bytes = bytes;
                memory.live += bytes;
                return block + 1;
            }
            static void release(void* opaque, void* pointer) noexcept {
                if (!pointer)
                    return;
                auto* block = static_cast<Header*>(pointer) - 1;
                static_cast<DecoderMemory*>(opaque)->live -= block->bytes;
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
            explicit Input(std::span<const uint8_t> bytes) : borrowed(bytes), memory(true) {}
            explicit Input(const std::filesystem::path& path) {
                if (lfs::core::open_file_for_read(path, std::ios::binary | std::ios::ate, file) && file.tellg() >= 0) {
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

        bool header_layout(const JxlBasicInfo& basic, Probe& info, const JxlDecodeLimits& limits, std::string& error) {
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
            if (pixels > limits.output_bytes / (info.channels * sample_bytes(type)))
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
            if (!structured) {
                profile.resize(profile_bytes);
                if (profile.empty() || JxlDecoderGetColorAsICCProfile(decoder, JXL_COLOR_PROFILE_TARGET_ORIGINAL, profile.data(), profile.size()) != JXL_DEC_SUCCESS)
                    return fail(error, "invalid color profile");
            }
            if (display) {
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
                 bool probe_only, bool display, const JxlDecodeLimits& limits, bool float_output = false) {
            if ((input.memory ? input.borrowed.size() : input.remaining) > limits.encoded_bytes)
                return fail(error, "encoded input byte limit exhausted");
            DecoderMemory memory{limits.decoder_bytes};
            const JxlMemoryManager manager{&memory, DecoderMemory::allocate, DecoderMemory::release};
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
                (display && JxlDecoderSetCms(decoder.get(), *cms) != JXL_DEC_SUCCESS))
                return fail(error, "cannot configure decoder");
            JxlBasicInfo basic{};
            Image candidate;
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
                    if (JxlDecoderGetBasicInfo(decoder.get(), &basic) != JXL_DEC_SUCCESS || !header_layout(basic, info, limits, error))
                        return error.empty() ? fail(error, "invalid basic information") : false;
                    if (float_output) {
                        info.sample_type = SampleType::Float32;
                        if (size_t(info.width) * info.height > limits.output_bytes / (size_t(info.channels) * sizeof(float)))
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
                    candidate = {info.width, info.height, info.channels, info.sample_type, std::vector<uint8_t>(bytes)};
                    if (JxlDecoderSetImageOutBuffer(decoder.get(), &format, candidate.data.data(), candidate.data.size()) != JXL_DEC_SUCCESS)
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
                    if (!header || !color || frames != 1 || images != 1 || candidate.data.empty())
                        return fail(error, "incomplete still image");
                    if ((!input.memory && input.remaining) || JxlDecoderReleaseInput(decoder.get()))
                        return fail(error, "unprocessed trailing input");
                    if (candidate.sample_type == SampleType::Float32)
                        for (size_t offset = 0; offset < candidate.data.size(); offset += sizeof(float)) {
                            float value;
                            std::memcpy(&value, candidate.data.data() + offset, sizeof(value));
                            if (!std::isfinite(value))
                                return fail(error, "nonfinite decoded sample");
                        }
                    image = std::move(candidate);
                    return true;
                default:
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

        bool copy_target(const Image& image, DecodeTarget& target, Probe& result,
                         const JxlDecodeLimits& limits, std::string& error) {
            const int channels = target.channels ? target.channels : image.channels;
            if (channels < 1 || channels > 4 || (channels <= 2 && image.channels >= 3) ||
                (target.sample_type != SampleType::UInt8 && target.sample_type != SampleType::UInt16 && target.sample_type != SampleType::Float32))
                return fail(error, "unsupported target channel count");
            const size_t pixels = size_t(image.width) * image.height;
            const size_t stride = sample_bytes(target.sample_type);
            if (pixels > limits.output_bytes / (channels * stride))
                return fail(error, "target byte limit exhausted");
            const size_t bytes = pixels * channels * stride;
            void* destination = target.data ? target.data : target.allocate ? target.allocate(bytes, target.user)
                                                                            : std::malloc(bytes);
            if (!destination)
                return fail(error, "cannot allocate target");
            for (size_t pixel = 0; pixel < pixels; ++pixel)
                for (int channel = 0; channel < channels; ++channel) {
                    const bool alpha = (channels == 2 || channels == 4) && channel == channels - 1;
                    const bool source_alpha = image.channels == 2 || image.channels == 4;
                    const int source_color = image.channels <= 2 ? 1 : 3;
                    const size_t index = pixel * image.channels + (alpha ? image.channels - 1 : source_color == 1 ? 0
                                                                                                                  : std::min(channel, 2));
                    double value = 1.;
                    if (!alpha || source_alpha) {
                        if (image.sample_type == SampleType::UInt8)
                            value = image.data[index] / 255.;
                        else if (image.sample_type == SampleType::UInt16) {
                            uint16_t sample;
                            std::memcpy(&sample, image.data.data() + index * sizeof(sample), sizeof(sample));
                            value = sample / 65535.;
                        } else {
                            float sample;
                            std::memcpy(&sample, image.data.data() + index * sizeof(sample), sizeof(sample));
                            value = sample;
                        }
                    }
                    auto* output = static_cast<uint8_t*>(destination) + (pixel * channels + channel) * stride;
                    if (target.sample_type == SampleType::Float32) {
                        const float sample = static_cast<float>(value);
                        std::memcpy(output, &sample, sizeof(sample));
                    } else if (target.sample_type == SampleType::UInt16) {
                        const auto sample = static_cast<uint16_t>(std::floor(std::clamp(value, 0., 1.) * 65535. + .5));
                        std::memcpy(output, &sample, sizeof(sample));
                    } else
                        *output = static_cast<uint8_t>(std::floor(std::clamp(value, 0., 1.) * 255. + .5));
                }
            target.data = destination;
            result = {image.width, image.height, channels, target.sample_type};
            return true;
        }

        template <class Source>
        bool target_decode(Source source, DecodeTarget& target, Probe& result, std::string& error, bool display, const JxlDecodeLimits& limits) {
            return guarded(error, [&] {
                Image image;
                Input input(source);
                Probe info;
                return run(input, image, info, error, false, display, limits, target.sample_type == SampleType::Float32) && copy_target(image, target, result, limits, error);
            });
        }
    } // namespace

    bool is_jxl(std::span<const uint8_t> bytes) {
        if (bytes.empty())
            return false;
        const auto signature = JxlSignatureCheck(bytes.data(), bytes.size());
        return signature == JXL_SIG_CODESTREAM || signature == JXL_SIG_CONTAINER;
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
