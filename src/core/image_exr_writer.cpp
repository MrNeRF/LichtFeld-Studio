/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "image_exr.hpp"
#include <OpenEXR/openexr.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>

namespace lfs::core::image_codecs {
    namespace {
        Error encodeError(ErrorCode code, std::string detail, int native = 0) {
            return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT(), .native = NativeError{ErrorDomain::IO, native, "OpenEXRCore"}});
        }
        struct StreamState {
            const ExrWriteStream& stream;
            std::string diagnostic;
        };
        int64_t writeBytes(exr_const_context_t, void* userdata, const void* bytes, uint64_t size, uint64_t offset, exr_stream_error_func_ptr_t) noexcept {
            const auto& state = *static_cast<StreamState*>(userdata);
            return state.stream.write(state.stream.user, bytes, size, offset);
        }
        void diagnostic(exr_const_context_t context, exr_result_t, const char* message) noexcept {
            void* user = nullptr;
            exr_get_user_data(context, &user);
            if (user && message) {
                auto& state = *static_cast<StreamState*>(user);
                try {
                    state.diagnostic = message;
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): the result code still reports the
                    // failure if allocation of the diagnostic text fails.
                    state.diagnostic.clear();
                }
            }
        }
        struct Context {
            exr_context_t value = nullptr;
            ~Context() {
                if (value)
                    exr_finish(&value);
            }
        };
        struct Pipeline {
            exr_encode_pipeline_t value = EXR_ENCODE_PIPELINE_INITIALIZER;
            exr_const_context_t context = nullptr;
            ~Pipeline() {
                if (context)
                    exr_encoding_destroy(context, &value);
            }
        };
        void check(exr_result_t status, const StreamState& state) {
            if (status != EXR_ERR_SUCCESS)
                throw Exception(encodeError(ErrorCode::Unavailable,
                                            std::format("EXR encoding failed (status={}, name={}, detail={})",
                                                        int(status), state.stream.name, state.diagnostic.empty() ? exr_get_error_code_as_string(status) : state.diagnostic),
                                            int(status)));
        }
        void cancel(const ExrEncodeOptions& options) {
            if (options.cancelled && options.cancelled())
                throw Exception(encodeError(ErrorCode::Cancelled, "EXR encoding cancelled (cancelled=true)"));
        }
    } // namespace

    Result<void> encode_exr(const ExrEncodeView& view, const ExrWriteStream& stream, const ExrEncodeOptions& options) {
        try {
            const auto row_bytes = view.width > 0 && (view.channels == 3 || view.channels == 4) ? size_t(view.width) * view.channels * sizeof(float) : 0;
            const auto rows = view.height > 0 ? size_t(view.height - 1) : 0;
            if (!row_bytes || view.height <= 0 || row_bytes > size_t(std::numeric_limits<int32_t>::max()) ||
                view.row_stride < row_bytes || (rows && view.row_stride > (std::numeric_limits<size_t>::max() - row_bytes) / rows) ||
                view.pixels.size() < rows * view.row_stride + row_bytes || !stream.write || stream.name.empty() || stream.name.find('\0') != std::string_view::npos)
                return Result<void>::failure(encodeError(ErrorCode::InvalidArgument, std::format("Invalid EXR view/stream (width={}, height={}, channels={}, stride={}, payload={}, writer={})", view.width, view.height, view.channels, view.row_stride, view.pixels.size(), stream.write != nullptr)));
            if ((options.precision != ExrSampleType::Half && options.precision != ExrSampleType::Float) ||
                (options.compression != ExrCompression::None && options.compression != ExrCompression::Zip) ||
                (options.premultiply_alpha && view.channels != 4))
                return Result<void>::failure(encodeError(ErrorCode::InvalidArgument, std::format("Invalid EXR options (precision={}, compression={}, premultiply={}, channels={})", int(options.precision), int(options.compression), options.premultiply_alpha, view.channels)));
            const auto chunk_rows = std::min(view.height, options.compression == ExrCompression::Zip ? 16 : 1);
            if (row_bytes * size_t(chunk_rows) > 64ULL * 1024 * 1024)
                return Result<void>::failure(encodeError(ErrorCode::ResourceExhausted, std::format("EXR chunk exceeds budget (bytes={}, limit={})", row_bytes * size_t(chunk_rows), 64ULL * 1024 * 1024)));
            for (const auto component : options.chromaticities)
                if (!std::isfinite(component))
                    return Result<void>::failure(encodeError(ErrorCode::InvalidArgument, std::format("EXR chromaticity must be finite (value={})", component)));
            constexpr size_t metadata_limit = 1024 * 1024 + 4096;
            if (options.text.size() > 32)
                return Result<void>::failure(encodeError(ErrorCode::InvalidArgument, std::format("EXR text attribute count exceeds budget (count={}, limit=32)", options.text.size())));
            size_t metadata_bytes = 0;
            std::vector<std::string_view> keys;
            for (const auto& [key, value] : options.text) {
                if (key.empty() || key.size() > 255 || key.find('\0') != std::string::npos || value.find('\0') != std::string::npos || value.size() > metadata_limit - metadata_bytes)
                    return Result<void>::failure(encodeError(ErrorCode::InvalidArgument, std::format("Invalid EXR text metadata (key_bytes={}, value_bytes={}, used_bytes={}, limit={})", key.size(), value.size(), metadata_bytes, metadata_limit)));
                metadata_bytes += value.size();
                constexpr std::array<std::string_view, 9> reserved{"channels", "compression", "dataWindow", "displayWindow", "lineOrder", "pixelAspectRatio", "screenWindowCenter", "screenWindowWidth", "chromaticities"};
                if (std::find(reserved.begin(), reserved.end(), key) != reserved.end() || std::find(keys.begin(), keys.end(), key) != keys.end())
                    return Result<void>::failure(encodeError(ErrorCode::InvalidArgument, std::format("EXR text metadata has reserved or duplicate key (key={})", key)));
                keys.push_back(key);
            }
            cancel(options);
            StreamState state{stream, {}};
            Context context;
            exr_context_initializer_t init = EXR_DEFAULT_CONTEXT_INITIALIZER;
            init.user_data = &state;
            init.write_fn = writeBytes;
            init.error_handler_fn = diagnostic;
            const std::string name(stream.name);
            check(exr_start_write(&context.value, name.c_str(), EXR_WRITE_FILE_DIRECTLY, &init), state);
            int part = -1;
            check(exr_add_part(context.value, nullptr, EXR_STORAGE_SCANLINE, &part), state);
            check(exr_initialize_required_attr_simple(context.value, part, view.width, view.height,
                                                      options.compression == ExrCompression::Zip ? EXR_COMPRESSION_ZIP : EXR_COMPRESSION_NONE),
                  state);
            constexpr std::array<const char*, 4> names{"R", "G", "B", "A"};
            for (int c = 0; c < view.channels; ++c)
                check(exr_add_channel(context.value, part, names[c], options.precision == ExrSampleType::Half ? EXR_PIXEL_HALF : EXR_PIXEL_FLOAT, EXR_PERCEPTUALLY_LOGARITHMIC, 1, 1), state);
            const auto& c = options.chromaticities;
            const exr_attr_chromaticities_t chroma{c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]};
            check(exr_attr_set_chromaticities(context.value, part, "chromaticities", &chroma), state);
            for (const auto& [key, value] : options.text)
                check(exr_attr_set_string(context.value, part, key.c_str(), value.c_str()), state);
            check(exr_write_header(context.value), state);
            Pipeline pipeline{.context = context.value};
            std::vector<float> scratch;
            bool initialized = false;
            for (int y = 0; y < view.height;) {
                cancel(options);
                exr_chunk_info_t chunk{};
                check(exr_write_scanline_chunk_info(context.value, part, y, &chunk), state);
                const auto* input = view.pixels.data() + size_t(y) * view.row_stride;
                const bool repack = options.premultiply_alpha || view.row_stride > size_t(std::numeric_limits<int32_t>::max()) ||
                                    view.row_stride % alignof(float) != 0 || reinterpret_cast<uintptr_t>(input) % alignof(float) != 0;
                if (repack)
                    scratch.resize(size_t(chunk.height) * view.width * view.channels);
                for (int row = 0; row < chunk.height; ++row)
                    for (int x = 0; x < view.width; ++x) {
                        std::array<float, 4> values{};
                        std::memcpy(values.data(), input + size_t(row) * view.row_stride + size_t(x) * view.channels * sizeof(float), view.channels * sizeof(float));
                        for (int channel = 0; channel < view.channels; ++channel)
                            if (!std::isfinite(values[channel]))
                                throw Exception(encodeError(ErrorCode::InvalidArgument, std::format("EXR sample must be finite (x={}, y={}, channel={}, value={})", x, y + row, channel, values[channel])));
                        if (view.channels == 4 && (values[3] < 0.f || values[3] > 1.f))
                            throw Exception(encodeError(ErrorCode::InvalidArgument, std::format("EXR alpha outside [0,1] (x={}, y={}, alpha={})", x, y + row, values[3])));
                        if (options.premultiply_alpha)
                            for (int channel = 0; channel < 3; ++channel)
                                values[channel] *= values[3];
                        if (options.precision == ExrSampleType::Half)
                            for (int channel = 0; channel < view.channels; ++channel)
                                if (std::abs(values[channel]) > 65504.f)
                                    throw Exception(encodeError(ErrorCode::InvalidArgument, std::format("EXR HALF sample overflow (x={}, y={}, channel={}, value={}, limit=65504); select FLOAT", x, y + row, channel, values[channel])));
                        if (repack)
                            std::memcpy(scratch.data() + (size_t(row) * view.width + x) * view.channels, values.data(), view.channels * sizeof(float));
                    }
                if (initialized)
                    check(exr_encoding_update(context.value, part, &chunk, &pipeline.value), state);
                else {
                    check(exr_encoding_initialize(context.value, part, &chunk, &pipeline.value), state);
                    initialized = true;
                }
                for (int channel_index = 0; channel_index < pipeline.value.channel_count; ++channel_index) {
                    auto& channel = pipeline.value.channels[channel_index];
                    const auto found = std::find_if(names.begin(), names.end(), [&](const auto* n) { return std::strcmp(n, channel.channel_name) == 0; });
                    if (found == names.end())
                        throw Exception(encodeError(ErrorCode::Internal, std::format("Unexpected EXR channel (name={})", channel.channel_name)));
                    channel.user_data_type = EXR_PIXEL_FLOAT;
                    channel.user_bytes_per_element = sizeof(float);
                    channel.user_pixel_stride = view.channels * sizeof(float);
                    channel.user_line_stride = int32_t(repack ? row_bytes : view.row_stride);
                    channel.encode_from_ptr = (repack ? reinterpret_cast<const uint8_t*>(scratch.data()) : input) + size_t(found - names.begin()) * sizeof(float);
                }
                check(exr_encoding_choose_default_routines(context.value, part, &pipeline.value), state);
                check(exr_encoding_run(context.value, part, &pipeline.value), state);
                y += chunk.height;
            }
            check(exr_encoding_destroy(context.value, &pipeline.value), state);
            pipeline.context = nullptr;
            check(exr_finish(&context.value), state);
            return {};
        } catch (const Exception& error) {
            return Result<void>::failure(error.error());
        } catch (const std::bad_alloc&) {
            return Result<void>::failure(encodeError(ErrorCode::ResourceExhausted, "EXR encoding allocation failed (bad_alloc=true)"));
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): encodeError constructs a core Internal error retaining the exception text; the caller never commits failed encoding.
            return Result<void>::failure(encodeError(ErrorCode::Internal, std::format("EXR encoding failed (exception={})", error.what())));
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): callback failures become structured errors.
            return Result<void>::failure(encodeError(ErrorCode::Internal, "EXR encoding failed (nonstandard_exception=true)"));
        }
    }
} // namespace lfs::core::image_codecs
