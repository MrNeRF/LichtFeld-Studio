/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "parsers/jpeg.h"
#include <array>
#include <nvimgcodec.h>
#include <string.h>
#include <vector>

#include "exif_orientation.h"
#include "imgproc/exception.h"
#include "log_ext.h"

#include "parsers/byte_io.h"
#include "parsers/exif.h"

namespace nvimgcodec {

    using jpeg_marker_t = std::array<uint8_t, 2>;
    using jpeg_exif_header_t = std::array<uint8_t, 6>;

    namespace {

        constexpr jpeg_marker_t sos_marker = {0xff, 0xda};
        constexpr jpeg_marker_t soi_marker = {0xff, 0xd8};
        constexpr jpeg_marker_t eoi_marker = {0xff, 0xd9};
        constexpr jpeg_marker_t app1_marker = {0xff, 0xe1};
        constexpr jpeg_marker_t app14_marker = {0xff, 0xee};

        constexpr jpeg_exif_header_t exif_header = {'E', 'x', 'i', 'f', 0, 0};

        bool IsValidMarker(const jpeg_marker_t& marker) {
            return marker[0] == 0xff && marker[1] != 0x00;
        }

        bool IsSofMarker(const jpeg_marker_t& marker) {
            // According to https://www.w3.org/Graphics/JPEG/itu-t81.pdf table B.1 Marker code assignments
            // SOF markers are from range 0xFFC0-0xFFCF, excluding 0xFFC4, 0xFFC8 and 0xFFCC.
            if (!IsValidMarker(marker) || marker[1] < 0xc0 || marker[1] > 0xcf)
                return false;
            return marker[1] != 0xc4 && marker[1] != 0xc8 && marker[1] != 0xcc;
        }

        nvimgcodecSampleDataType_t precision_to_sample_type(int precision) {
            if (precision <= 8)
                return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            else if (precision <= 16)
                return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
            else
                return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
        }

        nvimgcodecChromaSubsampling_t chroma_subsampling_from_factors(
            int ncomponents, uint8_t yh, uint8_t yv, uint8_t uh, uint8_t uv, uint8_t vh, uint8_t vv) {
            if (ncomponents == 1)
                return NVIMGCODEC_SAMPLING_GRAY;

            if (ncomponents == 3) {
                uint8_t minh = std::min(yh, std::min(uh, vh));
                uint8_t minv = std::min(yv, std::min(uv, vv));

                if (minh == 0 || minv == 0)
                    return NVIMGCODEC_SAMPLING_UNSUPPORTED;

                if (yh % minh || uh % minh || vh % minh || yv % minv || uv % minv || vv % minv)
                    return NVIMGCODEC_SAMPLING_UNSUPPORTED; // non-integer factors
                yh /= minh;
                uh /= minh;
                vh /= minh;
                yv /= minv;
                uv /= minv;
                vv /= minv;

                if (uh != vh || uv != vv)
                    return NVIMGCODEC_SAMPLING_UNSUPPORTED; // in chroma subsamplings we support chroma should have same factors

                if (uh != 1 || uv != 1)
                    return NVIMGCODEC_SAMPLING_UNSUPPORTED; // U/V should be 1x1

                if (yh == 1 && yv == 1)
                    return NVIMGCODEC_SAMPLING_444;
                else if (yh == 2 && yv == 1)
                    return NVIMGCODEC_SAMPLING_422;
                else if (yh == 2 && yv == 2)
                    return NVIMGCODEC_SAMPLING_420;
                else if (yh == 1 && yv == 2)
                    return NVIMGCODEC_SAMPLING_440;
                else if (yh == 4 && yv == 1)
                    return NVIMGCODEC_SAMPLING_411;
                else if (yh == 4 && yv == 2)
                    return NVIMGCODEC_SAMPLING_410;
                else if (yh == 2 && yv == 4)
                    return NVIMGCODEC_SAMPLING_410V;
            }
            return NVIMGCODEC_SAMPLING_UNSUPPORTED;
        }

        nvimgcodecStatus_t GetCodeStreamInfoImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream) {
            if (codestream_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO) {
                NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected structure type");
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }
            strcpy(codestream_info->codec_name, "jpeg");
            codestream_info->num_images = 1;

            // Calculate bitstream size - for JPEG, it's the entire file size
            code_stream->io_stream->size(code_stream->io_stream->instance, &(codestream_info->size));

            return NVIMGCODEC_STATUS_SUCCESS;
        }

    } // namespace

    JPEGParserPlugin::JPEGParserPlugin(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework), parser_desc_{NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC, sizeof(nvimgcodecParserDesc_t), nullptr, this, plugin_id_, "jpeg", static_can_parse, static_create, Parser::static_destroy, Parser::static_get_codestream_info, Parser::static_get_image_info} {
    }

    nvimgcodecParserDesc_t* JPEGParserPlugin::getParserDesc() {
        return &parser_desc_;
    }

    nvimgcodecStatus_t JPEGParserPlugin::canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream) {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg_parser_can_parse");
            CHECK_NULL(result);
            CHECK_NULL(code_stream);
            nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
            size_t length = 0;
            io_stream->size(io_stream->instance, &length);
            if (length < sizeof(jpeg_marker_t)) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Code stream is too short to be a JPEG");
                *result = 0;
                return NVIMGCODEC_STATUS_SUCCESS;
            }
            io_stream->seek(io_stream->instance, 0, SEEK_SET);
            auto signature = ReadValue<jpeg_marker_t>(io_stream);
            *result = (signature == soi_marker);
        } catch (const std::runtime_error& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if code stream can be parsed - " << e.what());
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t JPEGParserPlugin::static_can_parse(void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream) {
        try {
            CHECK_NULL(instance);
            auto handle = reinterpret_cast<JPEGParserPlugin*>(instance);
            return handle->canParse(result, code_stream);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
        }
    }

    JPEGParserPlugin::Parser::Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework)
        : plugin_id_(plugin_id), framework_(framework) {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg_parser_destroy");
    }

    nvimgcodecStatus_t JPEGParserPlugin::create(nvimgcodecParser_t* parser) {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg_parser_create");
            CHECK_NULL(parser);
            *parser = reinterpret_cast<nvimgcodecParser_t>(new JPEGParserPlugin::Parser(plugin_id_, framework_));
        } catch (const std::runtime_error& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create jpeg parser - " << e.what());
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t JPEGParserPlugin::static_create(void* instance, nvimgcodecParser_t* parser) {
        try {
            CHECK_NULL(instance);
            auto handle = reinterpret_cast<JPEGParserPlugin*>(instance);
            handle->create(parser);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t JPEGParserPlugin::Parser::static_destroy(nvimgcodecParser_t parser) {
        try {
            CHECK_NULL(parser);
            auto handle = reinterpret_cast<JPEGParserPlugin::Parser*>(parser);
            delete handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t JPEGParserPlugin::Parser::getCodeStreamInfo(nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream) {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg_parser_get_codestream_info");
            CHECK_NULL(code_stream);
            CHECK_NULL(codestream_info);
            return GetCodeStreamInfoImpl(plugin_id_, framework_, codestream_info, code_stream);
        } catch (const std::runtime_error& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream info from jpeg stream - " << e.what());
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }
    }

    nvimgcodecStatus_t JPEGParserPlugin::Parser::getImageInfo(nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream) {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg_parser_get_image_info");
        try {
            CHECK_NULL(code_stream);
            CHECK_NULL(image_info);
            size_t size = 0;
            nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
            io_stream->size(io_stream->instance, &size);
            io_stream->seek(io_stream->instance, 0, SEEK_SET);

            std::array<uint8_t, 2> signature;
            size_t read_nbytes = 0;
            io_stream->read(io_stream->instance, &read_nbytes, &signature[0], signature.size());
            if (read_nbytes != signature.size()) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected end-of-stream");
                return NVIMGCODEC_STATUS_BAD_CODESTREAM;
            }
            if (signature != soi_marker) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected signature");
                return NVIMGCODEC_STATUS_BAD_CODESTREAM;
            }

            bool read_shape = false, read_orientation = false, read_app14 = false;
            uint16_t height = 0, width = 0;
            uint8_t num_components = 0;
            uint8_t precision = 8;
            nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
            int adobe_transform = -1;
            nvimgcodecChromaSubsampling_t subsampling = NVIMGCODEC_SAMPLING_NONE;
            jpeg_marker_t sof_marker = {};
            while (!read_shape || !read_orientation || !read_app14) {
                jpeg_marker_t marker{};
                marker[0] = ReadValue<uint8_t>(io_stream);
                // https://www.w3.org/Graphics/JPEG/itu-t81.pdf section B.1.1.2 Markers
                // Any marker may optionally be preceded by any number of fill bytes,
                // which are bytes assigned code '\xFF'
                do {
                    marker[1] = ReadValue<uint8_t>(io_stream);
                } while (marker[1] == 0xff);

                if (!IsValidMarker(marker)) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Invalid marker");
                    return NVIMGCODEC_STATUS_BAD_CODESTREAM;
                }
                if (marker == sos_marker)
                    break;

                uint16_t size = ReadValueBE<uint16_t>(io_stream);
                ptrdiff_t offset = 0;
                auto res = io_stream->tell(io_stream->instance, &offset);
                if (res != NVIMGCODEC_STATUS_SUCCESS)
                    return res;
                ptrdiff_t next_marker_offset = offset - 2 + size;
                if (IsSofMarker(marker)) {
                    sof_marker = marker;
                    precision = ReadValue<uint8_t>(io_stream);
                    height = ReadValueBE<uint16_t>(io_stream);
                    width = ReadValueBE<uint16_t>(io_stream);
                    num_components = ReadValue<uint8_t>(io_stream);

                    if (num_components > 4)
                        return NVIMGCODEC_STATUS_BAD_CODESTREAM; // should not happen
                    std::array<std::pair<uint8_t, uint8_t>, 4> sampling_factors;
                    for (int c = 0; c < num_components; c++) {
                        io_stream->skip(io_stream->instance, 1); // component_id
                        auto temp = ReadValue<uint8_t>(io_stream);
                        auto horizontal_sampling_factor = temp >> 4;
                        auto vertical_sampling_factor = temp & 0x0F;
                        sampling_factors[c] = {horizontal_sampling_factor, vertical_sampling_factor};
                        io_stream->skip(io_stream->instance, 1); // quantization table selector
                    }
                    uint8_t yh = num_components > 0 ? sampling_factors[0].first : 0;
                    uint8_t yv = num_components > 0 ? sampling_factors[0].second : 0;
                    uint8_t uh = num_components > 1 ? sampling_factors[1].first : 0;
                    uint8_t uv = num_components > 1 ? sampling_factors[1].second : 0;
                    uint8_t vh = num_components > 2 ? sampling_factors[2].first : 0;
                    uint8_t vv = num_components > 2 ? sampling_factors[2].second : 0;
                    subsampling = chroma_subsampling_from_factors(num_components, yh, yv, uh, uv, vh, vv);

                    read_shape = true;
                } else if (marker == app1_marker && ReadValue<jpeg_exif_header_t>(io_stream) == exif_header) {
                    std::vector<uint8_t> exif_block(size - 8);
                    io_stream->read(io_stream->instance, &read_nbytes, exif_block.data(), exif_block.size());
                    if (read_nbytes != exif_block.size()) {
                        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Failed to read EXIF block");
                        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
                    }

                    cv::ExifReader reader;
                    if (!reader.parseExif(exif_block.data(), exif_block.size()))
                        continue;
                    auto entry = reader.getTag(cv::ORIENTATION);
                    if (entry.tag != cv::INVALID_TAG) {
                        orientation = FromExifOrientation(static_cast<ExifOrientation>(entry.field_u16));
                        read_orientation = true;
                    }
                } else if (marker == app14_marker) {
                    constexpr uint16_t app14_data_len = 14;
                    constexpr std::array<uint8_t, 5> adobe_signature = {0x41, 0x64, 0x6F, 0x62, 0x65}; /// Adobe in ASCII
                    auto signature = ReadValue<std::array<uint8_t, 5>>(io_stream);
                    if (size == app14_data_len && signature == adobe_signature) {
                        io_stream->skip(io_stream->instance, 2 + 2 + 2); ////version, flags0, flags1
                        adobe_transform = ReadValue<uint8_t>(io_stream);
                    }
                }
                io_stream->seek(io_stream->instance, next_marker_offset, SEEK_SET);
            }
            if (!read_shape) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Failed to read image dimensions");
                return NVIMGCODEC_STATUS_BAD_CODESTREAM;
            }

            image_info->struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
            image_info->orientation = orientation;
            image_info->chroma_subsampling = subsampling;
            strcpy(image_info->codec_name, "jpeg");
            switch (num_components) {
            case 1:
                image_info->color_spec = NVIMGCODEC_COLORSPEC_GRAY;
                image_info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
                break;
            case 4:
                if (adobe_transform == 2) {
                    image_info->color_spec = NVIMGCODEC_COLORSPEC_YCCK;
                    image_info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_YCCK;
                } else {
                    image_info->color_spec = NVIMGCODEC_COLORSPEC_CMYK;
                    image_info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_CMYK;
                }
                break;
            case 3:
                // assume that 3 channels is always going to be YCbCr
                image_info->color_spec = NVIMGCODEC_COLORSPEC_SYCC;
                image_info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_YCC;
                break;
            default:
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected number of channels" << num_components);
                return NVIMGCODEC_STATUS_BAD_CODESTREAM;
            }

            image_info->num_planes = num_components;
            auto sample_type = precision_to_sample_type(precision);
            for (int p = 0; p < num_components; p++) {
                image_info->plane_info[p].width = width;
                image_info->plane_info[p].height = height;
                image_info->plane_info[p].num_channels = 1;
                image_info->plane_info[p].sample_type = sample_type;
                image_info->plane_info[p].precision = precision;
            }

            nvimgcodecJpegImageInfo_t* jpeg_image_info = reinterpret_cast<nvimgcodecJpegImageInfo_t*>(image_info->struct_next);
            while (jpeg_image_info && jpeg_image_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO)
                jpeg_image_info = reinterpret_cast<nvimgcodecJpegImageInfo_t*>(jpeg_image_info->struct_next);
            if (jpeg_image_info && jpeg_image_info->struct_type == NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO) {
                jpeg_image_info->encoding = NVIMGCODEC_JPEG_ENCODING_UNKNOWN;
                if (sof_marker[1] >= 0xc0 && sof_marker[1] <= 0xcf)
                    jpeg_image_info->encoding = static_cast<nvimgcodecJpegEncoding_t>(sof_marker[1]);
            }

        } catch (const UnexpectedEndOfStream& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from jpeg stream - " << e.what());
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        } catch (const std::runtime_error& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from jpeg stream - " << e.what());
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }

        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t JPEGParserPlugin::Parser::static_get_codestream_info(
        nvimgcodecParser_t parser, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream) {
        try {
            CHECK_NULL(parser);
            auto handle = reinterpret_cast<JPEGParserPlugin::Parser*>(parser);
            return handle->getCodeStreamInfo(codestream_info, code_stream);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
        }
    }

    nvimgcodecStatus_t JPEGParserPlugin::Parser::static_get_image_info(
        nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream) {
        try {
            CHECK_NULL(parser);
            auto handle = reinterpret_cast<JPEGParserPlugin::Parser*>(parser);
            return handle->getImageInfo(image_info, code_stream);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
        }
    }

    class JpegParserExtension {
    public:
        explicit JpegParserExtension(const nvimgcodecFrameworkDesc_t* framework)
            : framework_(framework), jpeg_parser_plugin_(framework) {
            framework->registerParser(framework->instance, jpeg_parser_plugin_.getParserDesc(), NVIMGCODEC_PRIORITY_NORMAL);
        }
        ~JpegParserExtension() { framework_->unregisterParser(framework_->instance, jpeg_parser_plugin_.getParserDesc()); }

        static nvimgcodecStatus_t jpeg_parser_extension_create(
            void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework) {
            try {
                CHECK_NULL(framework)
                NVIMGCODEC_LOG_TRACE(framework, "jpeg_parser_ext", "jpeg_parser_extension_create");
                CHECK_NULL(extension)
                *extension = reinterpret_cast<nvimgcodecExtension_t>(new JpegParserExtension(framework));
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }
            return NVIMGCODEC_STATUS_SUCCESS;
        }

        static nvimgcodecStatus_t jpeg_parser_extension_destroy(nvimgcodecExtension_t extension) {
            try {
                CHECK_NULL(extension)
                auto ext_handle = reinterpret_cast<nvimgcodec::JpegParserExtension*>(extension);
                NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "jpeg_parser_ext", "jpeg_parser_extension_destroy");
                delete ext_handle;
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }
            return NVIMGCODEC_STATUS_SUCCESS;
        }

    private:
        const nvimgcodecFrameworkDesc_t* framework_;
        JPEGParserPlugin jpeg_parser_plugin_;
    };

    // clang-format off
nvimgcodecExtensionDesc_t jpeg_parser_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "jpeg_parser_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_VER,

    JpegParserExtension::jpeg_parser_extension_create,
    JpegParserExtension::jpeg_parser_extension_destroy
};
    // clang-format on

    nvimgcodecStatus_t get_jpeg_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc) {
        if (ext_desc == nullptr) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        *ext_desc = jpeg_parser_extension;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

} // namespace nvimgcodec
