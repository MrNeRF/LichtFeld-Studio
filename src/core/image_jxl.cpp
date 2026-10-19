/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "image_jxl.hpp"
#include "core/color_transfer.h"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

#include <jxl/cms.h>
#include <jxl/color_encoding.h>
#include <jxl/decode.h>
#include <jxl/resizable_parallel_runner.h>
#include <lcms2.h>

#include <algorithm>
#include <atomic>
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

        struct DecoderMemory {
            struct alignas(std::max_align_t) Header {
                size_t bytes;
            };
            size_t limit;
            std::atomic<size_t> live{0};
            std::atomic<bool> exhausted{false};
            static void* allocate(void* opaque, size_t bytes) noexcept {
                auto& memory = *static_cast<DecoderMemory*>(opaque);
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
                static_cast<DecoderMemory*>(opaque)->live.fetch_sub(block->bytes);
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
            DecoderMemory memory{limits.decoder_bytes};
            const JxlMemoryManager manager{&memory, DecoderMemory::allocate, DecoderMemory::release};
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
