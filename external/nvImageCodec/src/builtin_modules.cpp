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

#include "builtin_modules.h"
#include "imgproc/exception.h"
#include "parsers/parsers_ext_module.h"
#include <vector>

namespace nvimgcodec {

    const std::vector<nvimgcodecExtensionDesc_t>& get_builtin_modules() {
        static std::vector<nvimgcodecExtensionDesc_t> builtin_modules_vec;
        if (builtin_modules_vec.empty()) {
            builtin_modules_vec.push_back({NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr});
            nvimgcodecStatus_t ret = get_parsers_extension_desc(&builtin_modules_vec.back());
            if (ret != NVIMGCODEC_STATUS_SUCCESS)
                throw Exception(INTERNAL_ERROR, "Failed to load parsers extension");
        }
        return builtin_modules_vec;
    }

} // namespace nvimgcodec
