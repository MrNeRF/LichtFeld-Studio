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

#pragma once

#include <nvimgcodec.h>
#include <nvjpeg.h>

nvjpegOutputFormat_t nvimgcodec_to_nvjpeg_format(nvimgcodecSampleFormat_t nvimgcodec_format, nvimgcodecSampleDataType_t nvimgcodec_data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UNKNOWN);
nvjpegExifOrientation_t nvimgcodec_to_nvjpeg_orientation(nvimgcodecOrientation_t orientation);
nvjpegChromaSubsampling_t nvimgcodec_to_nvjpeg_css(nvimgcodecChromaSubsampling_t nvimgcodec_css);
nvjpegJpegEncoding_t nvimgcodec_to_nvjpeg_encoding(nvimgcodecJpegEncoding_t nvimgcodec_encoding);

nvimgcodecSampleDataType_t precision_to_sample_type(int precision);
nvimgcodecChromaSubsampling_t nvjpeg_to_nvimgcodec_css(nvjpegChromaSubsampling_t nvjpeg_css);
nvimgcodecOrientation_t exif_to_nvimgcodec_orientation(nvjpegExifOrientation_t exif_orientation);
nvimgcodecJpegEncoding_t nvjpeg_to_nvimgcodec_encoding(nvjpegJpegEncoding_t nvjpeg_encoding);
