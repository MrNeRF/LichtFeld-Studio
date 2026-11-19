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
#include "type_convert.h"

nvjpegOutputFormat_t nvimgcodec_to_nvjpeg_format(nvimgcodecSampleFormat_t nvimgcodec_format, nvimgcodecSampleDataType_t nvimgcodec_data_type)
{
    switch (nvimgcodec_format) {
    case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        return NVJPEG_OUTPUT_UNCHANGED;
    case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:{
        if (nvimgcodec_data_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16) {
            return NVJPEG_OUTPUT_UNCHANGEDI_U16;
        } else {
            return NVJPEG_OUTPUT_RGBI;
        }
    }

    case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        return NVJPEG_OUTPUT_RGB;
    case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        return NVJPEG_OUTPUT_RGBI;
    case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        return NVJPEG_OUTPUT_BGR;
    case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        return NVJPEG_OUTPUT_BGRI;
    case NVIMGCODEC_SAMPLEFORMAT_I_Y:
    case NVIMGCODEC_SAMPLEFORMAT_P_Y: {
        if (nvimgcodec_data_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16) {
            return NVJPEG_OUTPUT_UNCHANGEDI_U16;
        } else {
            return NVJPEG_OUTPUT_Y;
        }
    }
    case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
    case NVIMGCODEC_SAMPLEFORMAT_I_YUV:
        return NVJPEG_OUTPUT_YUV;
    default:
        return NVJPEG_OUTPUT_UNCHANGED;
    }
}

nvjpegExifOrientation_t nvimgcodec_to_nvjpeg_orientation(nvimgcodecOrientation_t orientation)
{
    if (orientation.rotated == 0 && !orientation.flip_x && !orientation.flip_y) {
        return NVJPEG_ORIENTATION_NORMAL;
    } else if (orientation.rotated == 0 && orientation.flip_x && !orientation.flip_y) {
        return NVJPEG_ORIENTATION_FLIP_HORIZONTAL;
    } else if (orientation.rotated == 180 && !orientation.flip_x && !orientation.flip_y) {
        return NVJPEG_ORIENTATION_ROTATE_180;
    } else if (orientation.rotated == 0 && !orientation.flip_x && orientation.flip_y) {
        return NVJPEG_ORIENTATION_FLIP_VERTICAL;
    } else if (orientation.rotated == 90 && !orientation.flip_x && orientation.flip_y) {
        return NVJPEG_ORIENTATION_TRANSPOSE;
    } else if (orientation.rotated == 270 && !orientation.flip_x && !orientation.flip_y) {
        return NVJPEG_ORIENTATION_ROTATE_90;
    } else if (orientation.rotated == 270 && !orientation.flip_x && orientation.flip_y) {
        return NVJPEG_ORIENTATION_TRANSVERSE;
    } else if (orientation.rotated == 90 && !orientation.flip_x && !orientation.flip_y) {
        return NVJPEG_ORIENTATION_ROTATE_270;
    } else {
        return NVJPEG_ORIENTATION_UNKNOWN;
    }
}

nvjpegChromaSubsampling_t nvimgcodec_to_nvjpeg_css(nvimgcodecChromaSubsampling_t nvimgcodec_css)
{
    switch (nvimgcodec_css) {
    case NVIMGCODEC_SAMPLING_UNSUPPORTED:
        return NVJPEG_CSS_UNKNOWN;
    case NVIMGCODEC_SAMPLING_444:
        return NVJPEG_CSS_444;
    case NVIMGCODEC_SAMPLING_422:
        return NVJPEG_CSS_422;
    case NVIMGCODEC_SAMPLING_420:
        return NVJPEG_CSS_420;
    case NVIMGCODEC_SAMPLING_440:
        return NVJPEG_CSS_440;
    case NVIMGCODEC_SAMPLING_411:
        return NVJPEG_CSS_411;
    case NVIMGCODEC_SAMPLING_410:
        return NVJPEG_CSS_410;
    case NVIMGCODEC_SAMPLING_GRAY:
        return NVJPEG_CSS_GRAY;
    case NVIMGCODEC_SAMPLING_410V:
        return NVJPEG_CSS_410V;
    default:
        return NVJPEG_CSS_UNKNOWN;
    }
}

nvjpegJpegEncoding_t nvimgcodec_to_nvjpeg_encoding(nvimgcodecJpegEncoding_t nvimgcodec_encoding)
{
    switch (nvimgcodec_encoding) {
    case NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT:
        return NVJPEG_ENCODING_BASELINE_DCT;
    case NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN:
        return NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN;
    case NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN:
        return NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;
    default:
        return NVJPEG_ENCODING_UNKNOWN;
    }
}

nvimgcodecJpegEncoding_t nvjpeg_to_nvimgcodec_encoding(nvjpegJpegEncoding_t nvjpeg_encoding)
{
    switch (nvjpeg_encoding) {
    case NVJPEG_ENCODING_BASELINE_DCT:
        return NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT;
    case NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN:
        return NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN;
    case NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN:
        return NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;
    default:
        return NVIMGCODEC_JPEG_ENCODING_UNKNOWN;
    }
}

nvimgcodecSampleDataType_t precision_to_sample_type(int precision)
{
    return precision == 8 ? NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 : NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
}

nvimgcodecChromaSubsampling_t nvjpeg_to_nvimgcodec_css(nvjpegChromaSubsampling_t nvjpeg_css)
{
    switch (nvjpeg_css) {
    case NVJPEG_CSS_UNKNOWN:
        return NVIMGCODEC_SAMPLING_NONE;
    case NVJPEG_CSS_444:
        return NVIMGCODEC_SAMPLING_444;
    case NVJPEG_CSS_422:
        return NVIMGCODEC_SAMPLING_422;
    case NVJPEG_CSS_420:
        return NVIMGCODEC_SAMPLING_420;
    case NVJPEG_CSS_440:
        return NVIMGCODEC_SAMPLING_440;
    case NVJPEG_CSS_411:
        return NVIMGCODEC_SAMPLING_411;
    case NVJPEG_CSS_410:
        return NVIMGCODEC_SAMPLING_410;
    case NVJPEG_CSS_GRAY:
        return NVIMGCODEC_SAMPLING_GRAY;
    case NVJPEG_CSS_410V:
        return NVIMGCODEC_SAMPLING_410V;
    default:
        return NVIMGCODEC_SAMPLING_UNSUPPORTED;
    }
}

nvimgcodecOrientation_t exif_to_nvimgcodec_orientation(nvjpegExifOrientation_t exif_orientation)
{
    switch (exif_orientation) {
    case NVJPEG_ORIENTATION_NORMAL:
        return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
    case NVJPEG_ORIENTATION_FLIP_HORIZONTAL:
        return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, true, false};
    case NVJPEG_ORIENTATION_ROTATE_180:
        return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 180, false, false};
    case NVJPEG_ORIENTATION_FLIP_VERTICAL:
        return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, true};
    case NVJPEG_ORIENTATION_TRANSPOSE:
        return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 90, false, true};
    case NVJPEG_ORIENTATION_ROTATE_90:
        return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 270, false, false};
    case NVJPEG_ORIENTATION_TRANSVERSE:
        return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 270, false, true};
    case NVJPEG_ORIENTATION_ROTATE_270:
        return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 90, false, false};
    default:
        return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
    }
}
