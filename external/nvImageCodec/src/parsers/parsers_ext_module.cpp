/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "imgproc/exception.h"
#include "log_ext.h"
#include "parsers/jpeg.h"
#include "parsers/jpeg2k.h"
#include <nvimgcodec.h>

namespace nvimgcodec {

    class ParsersExtension {
    public:
        explicit ParsersExtension(const nvimgcodecFrameworkDesc_t* framework)
            : framework_(framework), jpeg_parser_plugin_(framework), jpeg2k_parser_plugin_(framework) {
            framework->registerParser(framework->instance, jpeg_parser_plugin_.getParserDesc(), NVIMGCODEC_PRIORITY_NORMAL);
            framework->registerParser(framework->instance, jpeg2k_parser_plugin_.getParserDesc(), NVIMGCODEC_PRIORITY_NORMAL);
        }
        ~ParsersExtension() {
            framework_->unregisterParser(framework_->instance, jpeg_parser_plugin_.getParserDesc());
            framework_->unregisterParser(framework_->instance, jpeg2k_parser_plugin_.getParserDesc());
        }

        static nvimgcodecStatus_t parsers_extension_create(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework) {
            try {
                CHECK_NULL(framework)
                NVIMGCODEC_LOG_TRACE(framework, "nvimgcodec_builtin_parsers", "parsers_extension_create");
                CHECK_NULL(extension)
                *extension = reinterpret_cast<nvimgcodecExtension_t>(new ParsersExtension(framework));
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }
            return NVIMGCODEC_STATUS_SUCCESS;
        }

        static nvimgcodecStatus_t parsers_extension_destroy(nvimgcodecExtension_t extension) {
            try {
                CHECK_NULL(extension)
                auto ext_handle = reinterpret_cast<nvimgcodec::ParsersExtension*>(extension);
                NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "nvimgcodec_builtin_parsers", "parsers_extension_destroy");
                delete ext_handle;
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }
            return NVIMGCODEC_STATUS_SUCCESS;
        }

    private:
        const nvimgcodecFrameworkDesc_t* framework_;
        JPEGParserPlugin jpeg_parser_plugin_;
        JPEG2KParserPlugin jpeg2k_parser_plugin_;
    };

    // clang-format off
nvimgcodecExtensionDesc_t parsers_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "nvimgcodec_builtin_parsers",
    NVIMGCODEC_VER,
    NVIMGCODEC_VER, 

    ParsersExtension::parsers_extension_create,
    ParsersExtension::parsers_extension_destroy
};
    // clang-format on

    nvimgcodecStatus_t get_parsers_extension_desc(nvimgcodecExtensionDesc_t* ext_desc) {
        if (ext_desc == nullptr) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        *ext_desc = parsers_extension;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

} // namespace nvimgcodec
