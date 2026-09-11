/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <sstream>
#include <string>

#include <nvimgcodec.h>

#include "imgproc/type_utils.h"

namespace nvimgcodec {

    namespace {

        inline void log_image_info_check_warning(
            const nvimgcodecFrameworkDesc_t* framework,
            const char* plugin_id,
            const char* msg) {
            nvimgcodecDebugMessageData_t data{
                NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSAGE_DATA,
                sizeof(nvimgcodecDebugMessageData_t),
                nullptr,
                msg,
                0,
                nullptr,
                plugin_id,
                NVIMGCODEC_VER};
            framework->log(framework->instance, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING,
                           NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, &data);
        }

    } // anonymous namespace

    // Returns true iff every plane in `image_info` agrees with plane 0 on both `sample_type`
    // and `precision`. On the first mismatch, logs an warning through the framework and returns
    // false (no further fields are checked once one fails). The plugin_id identifies which
    // codec produced the message in the log stream.
    //
    // Intended for canDecode / canEncode implementations whose codecs do not handle planes
    // with mixed sample types or mixed precisions; the caller is responsible for OR-ing
    // NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED into its status on a false return.
    inline bool check_planes_consistency(
        const nvimgcodecFrameworkDesc_t* framework,
        const char* plugin_id,
        const nvimgcodecImageInfo_t& image_info) {
        if (image_info.num_planes < 2) {
            return true;
        }
        const auto type0 = image_info.plane_info[0].sample_type;
        const auto precision0 = ResolvePrecision(image_info.plane_info[0].precision, type0);
        for (uint32_t p = 1; p < image_info.num_planes; ++p) {
            if (image_info.plane_info[p].sample_type != type0) {
                log_image_info_check_warning(framework, plugin_id,
                                             "All planes must have the same sample type");
                return false;
            }
            if (ResolvePrecision(image_info.plane_info[p].precision, image_info.plane_info[p].sample_type) != precision0) {
                log_image_info_check_warning(framework, plugin_id,
                                             "All planes must have the same precision");
                return false;
            }
        }
        return true;
    }

    // If `image_info` declares a per-plane precision that differs from the full bitdepth
    // of the sample data type, log a one-shot warning saying this encoder will ignore the
    // override and encode at the dtype's full precision. The plugin_id is what the log
    // system uses to identify which encoder produced the message.
    //
    // Intended for encoder extensions that do not honor `plane_info.precision`. Bypassed
    // silently for full-precision (precision == 0 or precision == dtype bitdepth) inputs.
    inline void warn_if_custom_precision_unsupported(
        const nvimgcodecFrameworkDesc_t* framework,
        const char* plugin_id,
        const nvimgcodecImageInfo_t& image_info) {
        if (image_info.num_planes == 0) {
            return;
        }
        const uint8_t dtype_bitdepth = static_cast<uint8_t>(TypeSize(image_info.plane_info[0].sample_type) * 8);
        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            const uint8_t pr = image_info.plane_info[p].precision;
            if (pr != 0 && pr != dtype_bitdepth) {
                std::stringstream ss;
                ss << "Custom plane_info.precision (" << static_cast<int>(pr)
                   << " bits) is not supported; encoding will use the full type precision ("
                   << static_cast<int>(dtype_bitdepth) << " bits).";
                const std::string msg = ss.str();
                log_image_info_check_warning(framework, plugin_id, msg.c_str());
                return;
            }
        }
    }

} // namespace nvimgcodec
