/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "image_avif.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <exception>
#include <libheif/heif.h>
#include <memory>
#include <stdexcept>

namespace lfs::core::image_codecs {
    namespace {
        Status failure(ErrorCode code, std::string detail) {
            return Status::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = "AVIF: " + detail, .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
        const heif_encoder_descriptor* descriptor() {
            const heif_encoder_descriptor* result = nullptr;
            return heif_get_encoder_descriptors(heif_compression_AV1, "aom", &result, 1) == 1 ? result : nullptr;
        }
        template <class Writer>
        Status encode(std::span<const uint8_t> input, int width, int height, int channels, size_t stride,
                      const AvifEncodeOptions& options, std::string_view xmp, Writer&& write) {
            try {
                const auto cancelled = [&] { return options.cancelled && options.cancelled(); };
                if (cancelled())
                    return failure(ErrorCode::Cancelled, "cancelled before encoding");
                if (auto valid = validate_avif_encode_options(options); !valid)
                    return valid;
                if (xmp.size() > 32u * 1024 * 1024 || xmp.find('\0') != std::string_view::npos)
                    return failure(ErrorCode::InvalidArgument, "XMP exceeds 32 MiB or contains NUL");
                if (width <= 0 || height <= 0 || channels < 1 || channels > 4)
                    return failure(ErrorCode::InvalidArgument, "invalid dimensions or channel count");
                const size_t sample = options.bit_depth == 8 ? 1 : 2;
                if (size_t(height) > options.pixels / size_t(width) || size_t(width) > SIZE_MAX / size_t(channels) / sample / size_t(height))
                    return failure(ErrorCode::ResourceExhausted, "pixel limit or image extent exceeded");
                const size_t row = size_t(width) * channels * sample;
                if (stride < row || (height > 1 && stride > (SIZE_MAX - row) / size_t(height - 1)) || input.size() < size_t(height - 1) * stride + row)
                    return failure(ErrorCode::InvalidArgument, "invalid row stride or truncated input");
                const auto* provider = descriptor();
                if (!provider)
                    return failure(ErrorCode::Unavailable, "built-in AOM encoder unavailable");
                std::unique_ptr<heif_context, decltype(&heif_context_free)> context(heif_context_alloc(), heif_context_free);
                if (!context)
                    return failure(ErrorCode::ResourceExhausted, "cannot allocate context");
                const auto check = [](heif_error error) {
                    return error.code == heif_error_Ok ? Status{} : failure(error.code == heif_error_Memory_allocation_error ? ErrorCode::ResourceExhausted : ErrorCode::Internal, "provider error " + std::to_string(error.code) + "/" + std::to_string(error.subcode));
                };
                heif_encoder* raw_encoder = nullptr;
                if (auto status = check(heif_context_get_encoder(context.get(), provider, &raw_encoder)); !status)
                    return status;
                std::unique_ptr<heif_encoder, decltype(&heif_encoder_release)> encoder(raw_encoder, heif_encoder_release);
                for (auto status : {heif_encoder_set_lossless(encoder.get(), options.lossless),
                                    heif_encoder_set_lossy_quality(encoder.get(), options.quality),
                                    heif_encoder_set_parameter_integer(encoder.get(), "speed", options.speed),
                                    heif_encoder_set_parameter_integer(encoder.get(), "threads", options.threads),
                                    heif_encoder_set_parameter_string(encoder.get(), "chroma", "444"),
                                    heif_encoder_set_parameter_boolean(encoder.get(), "lossless-alpha", true)})
                    if (auto result = check(status); !result)
                        return result;
                const bool gray = channels <= 2;
                const bool alpha = channels == 2 || channels == 4;
                heif_image* raw_image = nullptr;
                if (auto status = check(heif_image_create(width, height, gray ? heif_colorspace_monochrome : heif_colorspace_RGB,
                                                          gray ? heif_chroma_monochrome : heif_chroma_444, &raw_image));
                    !status)
                    return status;
                std::unique_ptr<heif_image, decltype(&heif_image_release)> image(raw_image, heif_image_release);
                const std::array<heif_channel, 4> planes = gray ? std::array{heif_channel_Y, heif_channel_Alpha, heif_channel_Y, heif_channel_Y}
                                                                : std::array{heif_channel_R, heif_channel_G, heif_channel_B, heif_channel_Alpha};
                const uint16_t maximum = static_cast<uint16_t>((1u << options.bit_depth) - 1);
                for (int c = 0; c < channels; ++c) {
                    if (auto status = check(heif_image_add_plane(image.get(), planes[c], width, height, options.bit_depth)); !status)
                        return status;
                    size_t destination_stride = 0;
                    auto* destination = heif_image_get_plane2(image.get(), planes[c], &destination_stride);
                    if (!destination || destination_stride < size_t(width) * sample)
                        return failure(ErrorCode::Internal, "invalid provider plane");
                    for (int y = 0; y < height; ++y) {
                        if (cancelled())
                            return failure(ErrorCode::Cancelled, "cancelled while preparing planes");
                        for (int x = 0; x < width; ++x) {
                            const auto* value = input.data() + size_t(y) * stride + (size_t(x) * channels + c) * sample;
                            uint16_t code = *value;
                            if (sample == 2)
                                std::memcpy(&code, value, 2);
                            if (code > maximum)
                                return failure(ErrorCode::InvalidArgument, "sample exceeds declared bit depth");
                            auto* target = destination + size_t(y) * destination_stride + size_t(x) * sample;
                            if (sample == 1)
                                *target = static_cast<uint8_t>(code);
                            else
                                std::memcpy(target, &code, sample);
                        }
                    }
                }
                std::unique_ptr<heif_color_profile_nclx, decltype(&heif_nclx_color_profile_free)> color(heif_nclx_color_profile_alloc(), heif_nclx_color_profile_free);
                if (!color)
                    return failure(ErrorCode::ResourceExhausted, "cannot allocate color profile");
                for (auto status : {heif_nclx_color_profile_set_color_primaries(color.get(), options.primaries == AvifPrimaries::Bt709 ? 1 : 9),
                                    heif_nclx_color_profile_set_transfer_characteristics(color.get(), 13),
                                    heif_nclx_color_profile_set_matrix_coefficients(color.get(), gray ? 1 : options.lossless                        ? 0
                                                                                                        : options.primaries == AvifPrimaries::Bt709 ? 1
                                                                                                                                                    : 9)})
                    if (auto result = check(status); !result)
                        return result;
                color->full_range_flag = 1;
                if (auto status = check(heif_image_set_nclx_color_profile(image.get(), color.get())); !status)
                    return status;
                heif_image_set_premultiplied_alpha(image.get(), false);
                std::unique_ptr<heif_encoding_options, decltype(&heif_encoding_options_free)> settings(heif_encoding_options_alloc(), heif_encoding_options_free);
                if (!settings)
                    return failure(ErrorCode::ResourceExhausted, "cannot allocate encoding options");
                settings->save_alpha_channel = alpha;
                settings->output_nclx_profile = color.get();
                if (cancelled())
                    return failure(ErrorCode::Cancelled, "cancelled before provider submission");
                heif_image_handle* raw_encoded = nullptr;
                const auto encoded = heif_context_encode_image(context.get(), image.get(), encoder.get(), settings.get(), xmp.empty() ? nullptr : &raw_encoded);
                std::unique_ptr<heif_image_handle, decltype(&heif_image_handle_release)> handle(raw_encoded, heif_image_handle_release);
                if (auto status = check(encoded); !status)
                    return status;
                if (!xmp.empty()) {
                    if (cancelled())
                        return failure(ErrorCode::Cancelled, "cancelled before metadata insertion");
                    if (!handle)
                        return failure(ErrorCode::Internal, "provider returned no encoded image handle");
                    if (auto status = check(heif_context_add_XMP_metadata(context.get(), handle.get(), xmp.data(), int(xmp.size()))); !status)
                        return status;
                }
                if (cancelled())
                    return failure(ErrorCode::Cancelled, "cancelled after provider encoding");
                struct Output {
                    Writer& write;
                    const AvifEncodeOptions& options;
                    size_t total = 0;
                    Status result;
                    std::exception_ptr exception;
                } output{write, options};
                heif_writer writer{1, [](heif_context*, const void* data, size_t bytes, void* user) -> heif_error {
                                       auto& out = *static_cast<Output*>(user);
                                       try {
                                           if (!out.result || out.exception)
                                               return {heif_error_Encoding_error, heif_suberror_Unspecified, "AVIF output already failed"};
                                           if (out.options.cancelled && out.options.cancelled())
                                               out.result = failure(ErrorCode::Cancelled, "cancelled while serializing");
                                           else if (bytes > out.options.max_output_bytes - out.total)
                                               out.result = failure(ErrorCode::ResourceExhausted, "encoded output limit exceeded");
                                           else {
                                               out.result = out.write(std::span<const uint8_t>(static_cast<const uint8_t*>(data), bytes));
                                               if (out.result)
                                                   out.total += bytes;
                                           }
                                       } catch (...) {
                                           // LFS-CENSUS-OK(empty-catch): retain without allocating and rethrow after leaving the C ABI.
                                           out.exception = std::current_exception();
                                           return {heif_error_Encoding_error, heif_suberror_Unspecified, "AVIF output callback exception"};
                                       }
                                       return out.result ? heif_error{heif_error_Ok, heif_suberror_Unspecified, "Success"}
                                                         : heif_error{heif_error_Encoding_error, heif_suberror_Unspecified, "AVIF output failed"};
                                   }};
                const auto status = heif_context_write(context.get(), &writer, &output);
                if (output.exception)
                    std::rethrow_exception(output.exception);
                if (!output.result)
                    return std::move(output.result);
                if (auto result = check(status); !result)
                    return result;
                return cancelled() ? failure(ErrorCode::Cancelled, "cancelled before completion") : Status{};
            } catch (const std::bad_alloc&) {
                return failure(ErrorCode::ResourceExhausted, "allocation failed");
            } catch (const std::exception& exception) {
                // LFS-CENSUS-OK(empty-catch): retain callback/provider diagnostics at the synchronous Status boundary.
                return failure(ErrorCode::Internal, exception.what());
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): convert synchronous provider/callback exceptions to the typed API.
                return failure(ErrorCode::Internal, "nonstandard encoding callback or provider exception");
            }
        }
    } // namespace
    bool avif_encoder_available() { return descriptor() != nullptr; }
    std::string avif_encoder_version() {
        const auto* provider = descriptor();
        return provider ? std::string(heif_get_version()) + ":" + heif_encoder_descriptor_get_name(provider) : std::string{};
    }
    Status validate_avif_encode_options(const AvifEncodeOptions& options) {
        if ((options.bit_depth != 8 && options.bit_depth != 10 && options.bit_depth != 12) || options.quality < 0 || options.quality > 100 || options.speed < 0 || options.speed > 9 || options.threads < 1 || options.threads > 8 ||
            (options.primaries != AvifPrimaries::Bt709 && options.primaries != AvifPrimaries::Bt2020))
            return failure(ErrorCode::InvalidArgument, "invalid depth, quality, speed or thread count");
        return {};
    }
    Status encode_avif_memory(std::span<const uint8_t> data, int width, int height, int channels, size_t stride,
                              std::vector<uint8_t>& output, const AvifEncodeOptions& options, std::string_view xmp) {
        std::vector<uint8_t> candidate;
        auto result = encode(data, width, height, channels, stride, options, xmp, [&](std::span<const uint8_t> bytes) -> Status {
            candidate.insert(candidate.end(), bytes.begin(), bytes.end());
            return {};
        });
        if (result)
            output = std::move(candidate);
        return result;
    }
    Status write_avif(const std::filesystem::path& path, std::span<const uint8_t> data, int width, int height,
                      int channels, size_t stride, const AvifEncodeOptions& options, const AtomicFileOptions& publication, std::string_view xmp) {
        try {
            auto encoding = options;
            encoding.cancelled = [a = options.cancelled, b = publication.cancelled] { return (a && a()) || (b && b()); };
            return writeFileAtomically(path, [&](FILE* stream) { return encode(data, width, height, channels, stride, encoding, xmp, [&](std::span<const uint8_t> bytes) -> Status {
                                                                     if (std::fwrite(bytes.data(), 1, bytes.size(), stream) != bytes.size()) {
                                                                         const int native = errno;
                                                                         return Status::failure(make_error({.code = ErrorCode::Unavailable, .domain = ErrorDomain::IO, .detail = "Cannot write AVIF output (errno=" + std::to_string(native) + ")", .detection = LFS_SOURCE_SITE_CURRENT(), .native = NativeError{ErrorDomain::IO, native, "stdio"}}));
                                                                     }
                                                                     return {};
                                                                 }); }, publication);
        } catch (const std::bad_alloc&) {
            return failure(ErrorCode::ResourceExhausted, "allocation failed while preparing publication");
        } catch (const std::exception& exception) {
            // LFS-CENSUS-OK(empty-catch): preserve publication setup diagnostics at the Status boundary.
            return failure(ErrorCode::Internal, exception.what());
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): publication setup must not leak callback exceptions from the Status API.
            return failure(ErrorCode::Internal, "nonstandard publication setup exception");
        }
    }
} // namespace lfs::core::image_codecs
