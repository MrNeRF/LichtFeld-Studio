/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "image_exr.hpp"
#include "core/path_utils.hpp"

#include <OpenEXR/openexr.h>
#include <array>
#include <cstring>
#include <limits>

namespace lfs::core::image_codecs {
    namespace {
        // Context and pipeline own all OpenEXR allocations, including on errors.
        struct Context {
            exr_context_t value = nullptr;
            ~Context() {
                if (value)
                    exr_finish(&value);
            }
        };

        struct Pipeline {
            exr_decode_pipeline_t value = EXR_DECODE_PIPELINE_INITIALIZER;
            exr_const_context_t context;
            ~Pipeline() { exr_decoding_destroy(context, &value); }
        };

        bool check(exr_result_t status, std::string& error) {
            if (status == EXR_ERR_SUCCESS) {
                error.clear();
                return true;
            }
            const auto detail = std::move(error);
            error = std::string("EXR: ") + exr_get_error_code_as_string(status);
            if (!detail.empty())
                error += ": " + detail;
            return false;
        }

        struct Header {
            exr_attr_box2i_t window{};
            exr_storage_t storage{};
            int width = 0;
            int height = 0;
            bool grayscale = false;
            int source_channels = 0;
            std::array<int, 4> channels{-1, -1, -1, -1};
        };

        bool open(const std::filesystem::path& path, Context& context, Header& header,
                  std::string& error) {
            exr_context_initializer_t init = EXR_DEFAULT_CONTEXT_INITIALIZER;
            // Keep the library's useful details without printing or throwing through C.
            init.user_data = &error;
            init.error_handler_fn = [](exr_const_context_t ctxt, exr_result_t, const char* message) noexcept {
                void* user = nullptr;
                if (message && exr_get_user_data(ctxt, &user) == EXR_ERR_SUCCESS && user) {
                    try {
                        *static_cast<std::string*>(user) = message;
                    } catch (...) {
                        // Fall back to the result code if storing the detail fails.
                        static_cast<std::string*>(user)->clear();
                    }
                }
            };
            const auto name = path_to_utf8(path);
            if (!check(exr_start_read(&context.value, name.c_str(), &init), error) ||
                !check(exr_get_storage(context.value, 0, &header.storage), error) ||
                !check(exr_get_data_window(context.value, 0, &header.window), error))
                return false;
            if (header.storage != EXR_STORAGE_SCANLINE && header.storage != EXR_STORAGE_TILED) {
                error = "EXR: deep images are not supported";
                return false;
            }
            const auto width = int64_t(header.window.max.x) - header.window.min.x + 1;
            const auto height = int64_t(header.window.max.y) - header.window.min.y + 1;
            // OpenEXR's output line stride is int32_t. Validate before any allocation.
            if (width <= 0 || height <= 0 || width > int64_t(std::numeric_limits<int32_t>::max() / (4 * sizeof(float))) ||
                height > std::numeric_limits<int>::max() ||
                uint64_t(width) * uint64_t(height) > std::numeric_limits<size_t>::max() / (4 * sizeof(float))) {
                error = "EXR: invalid or oversized data window";
                return false;
            }
            header.width = static_cast<int>(width);
            header.height = static_cast<int>(height);
            const exr_attr_chlist_t* channels = nullptr;
            if (!check(exr_get_channels(context.value, 0, &channels), error))
                return false;
            // Preserve LoadEXR's contract: a single channel is replicated to RGBA;
            // otherwise select the unlayered RGB channels and optional alpha.
            header.grayscale = channels->num_channels == 1;
            constexpr std::array<const char*, 4> names{"R", "G", "B", "A"};
            for (int i = 0; i < channels->num_channels; ++i) {
                for (int c = 0; c < 4; ++c) {
                    if (std::strcmp(channels->entries[i].name.str, names[c]) == 0)
                        header.channels[c] = i;
                }
            }
            if (header.grayscale)
                header.channels = {0, -1, -1, -1};
            if (!header.grayscale && (header.channels[0] < 0 || header.channels[1] < 0 || header.channels[2] < 0)) {
                error = "EXR: RGB channels not found";
                return false;
            }
            // Auxiliary channels such as Z do not make an RGB camera have alpha.
            header.source_channels = header.grayscale ? 1 : header.channels[3] >= 0 ? 4
                                                                                    : 3;
            for (const int index : header.channels) {
                if (index >= 0 && (channels->entries[index].x_sampling != 1 || channels->entries[index].y_sampling != 1)) {
                    error = "EXR: subsampled RGB/alpha channels are not supported";
                    return false;
                }
            }
            return true;
        }
    } // namespace

    bool probe_exr(const std::filesystem::path& path, Probe& result, std::string& error) {
        error.clear();
        Context context;
        Header header;
        if (!open(path, context, header, error))
            return false;
        // Header only: do not read the chunk table or decompress pixels for camera import.
        result = {header.width, header.height, header.source_channels, SampleType::Float32};
        return true;
    }

    bool decode_exr(const std::filesystem::path& path, Image& result, std::string& error) {
        error.clear();
        Context context;
        Header header;
        if (!open(path, context, header, error))
            return false;
        std::error_code file_error;
        const auto file_size = std::filesystem::file_size(path, file_error);
        if (file_error) {
            error = "EXR: unable to read file size: " + file_error.message();
            return false;
        }
        int32_t chunk_count = 0;
        uint64_t table_offset = 0;
        if (!check(exr_get_chunk_count(context.value, 0, &chunk_count), error) ||
            !check(exr_get_chunk_table_offset(context.value, 0, &table_offset), error))
            return false;
        // A real file must contain its offset table. Check before the library or
        // the output buffer allocates from dimensions in a potentially damaged header.
        // Do not impose a compression-ratio limit: uniform HDR images compress well.
        if (chunk_count <= 0 || table_offset > file_size ||
            uint64_t(chunk_count) > (file_size - table_offset) / sizeof(uint64_t)) {
            error = "EXR: data window requires a chunk table larger than the file";
            return false;
        }
        Image image;
        image.width = header.width;
        image.height = header.height;
        image.channels = 4;
        image.sample_type = SampleType::Float32;
        float* pixels = nullptr;

        Pipeline pipeline{EXR_DECODE_PIPELINE_INITIALIZER, context.value};
        bool initialized = false;
        auto decode_chunk = [&](const exr_chunk_info_t& chunk, int x, int y) {
            if (x < 0 || y < 0 || chunk.width <= 0 || chunk.height <= 0 ||
                int64_t(x) + chunk.width > header.width || int64_t(y) + chunk.height > header.height) {
                error = "EXR: chunk outside data window";
                return false;
            }
            if (chunk.data_offset > file_size || chunk.packed_size > file_size - chunk.data_offset ||
                (chunk.compression == EXR_COMPRESSION_NONE && chunk.packed_size != chunk.unpacked_size)) {
                error = "EXR: chunk size is inconsistent with the file or data window";
                return false;
            }
            // Delay full-image allocation until real pixel chunk metadata is available.
            if (!pixels) {
                image.data.resize(size_t(header.width) * header.height * 4 * sizeof(float));
                pixels = reinterpret_cast<float*>(image.data.data());
                for (size_t i = 0; i < image.data.size() / sizeof(float); i += 4)
                    pixels[i + 3] = 1.0f;
            }
            const auto status = initialized
                                    ? exr_decoding_update(context.value, 0, &chunk, &pipeline.value)
                                    : exr_decoding_initialize(context.value, 0, &chunk, &pipeline.value);
            if (!check(status, error))
                return false;
            initialized = true;
            for (int i = 0; i < pipeline.value.channel_count; ++i) {
                auto& channel = pipeline.value.channels[i];
                channel.decode_to_ptr = nullptr;
                for (int c = 0; c < 4; ++c) {
                    if (header.channels[c] == i) {
                        channel.user_data_type = EXR_PIXEL_FLOAT;
                        channel.user_bytes_per_element = sizeof(float);
                        channel.user_pixel_stride = 4 * sizeof(float);
                        channel.user_line_stride = static_cast<int32_t>(size_t(header.width) * 4 * sizeof(float));
                        channel.decode_to_ptr = reinterpret_cast<uint8_t*>(pixels + (size_t(y) * header.width + x) * 4 + c);
                    }
                }
            }
            return check(exr_decoding_choose_default_routines(context.value, 0, &pipeline.value), error) &&
                   check(exr_decoding_run(context.value, 0, &pipeline.value), error);
        };

        if (header.storage == EXR_STORAGE_SCANLINE) {
            int32_t lines = 0;
            if (!check(exr_get_scanlines_per_chunk(context.value, 0, &lines), error))
                return false;
            if (lines <= 0) {
                error = "EXR: invalid scanline chunk size";
                return false;
            }
            for (int64_t y = header.window.min.y; y <= header.window.max.y; y += lines) {
                exr_chunk_info_t chunk{};
                if (!check(exr_read_scanline_chunk_info(context.value, 0, static_cast<int>(y), &chunk), error) ||
                    !decode_chunk(chunk, 0, static_cast<int>(y - header.window.min.y)))
                    return false;
            }
        } else {
            int32_t tile_width = 0, tile_height = 0;
            if (!check(exr_get_tile_sizes(context.value, 0, 0, 0, &tile_width, &tile_height), error))
                return false;
            if (tile_width <= 0 || tile_height <= 0) {
                error = "EXR: invalid tile size";
                return false;
            }
            // As with TinyEXR, use the full-resolution level of mip/ripmap images.
            for (int64_t y = 0, ty = 0; y < header.height; y += tile_height, ++ty) {
                for (int64_t x = 0, tx = 0; x < header.width; x += tile_width, ++tx) {
                    exr_chunk_info_t chunk{};
                    if (!check(exr_read_tile_chunk_info(context.value, 0, static_cast<int>(tx), static_cast<int>(ty), 0, 0, &chunk), error) ||
                        !decode_chunk(chunk, static_cast<int>(x), static_cast<int>(y)))
                        return false;
                }
            }
        }
        if (header.grayscale) {
            for (size_t i = 0; i < image.data.size() / sizeof(float); i += 4)
                pixels[i + 1] = pixels[i + 2] = pixels[i + 3] = pixels[i];
        }
        result = std::move(image);
        return true;
    }
} // namespace lfs::core::image_codecs
