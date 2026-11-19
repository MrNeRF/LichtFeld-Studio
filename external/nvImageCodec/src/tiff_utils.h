/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <type_traits>

namespace nvimgcodec {

/**
 * @brief Find TIFF info extension in struct_next chain.
 * @param info Pointer to the CodeStreamInfo structure.
 * @return Pointer to the TIFF extension if found, nullptr otherwise.
 */
template <typename T>
inline auto findInfoTiffExt(T* info) -> std::conditional_t<std::is_const_v<T>,
    const nvimgcodecCodeStreamInfoTiffExt_t*, nvimgcodecCodeStreamInfoTiffExt_t*> {
    if (!info) return nullptr;
    void* ext_ptr = info->struct_next;
    while (ext_ptr != nullptr) {
        auto* base = static_cast<std::conditional_t<std::is_const_v<T>,
            const nvimgcodecCodeStreamInfoTiffExt_t*, nvimgcodecCodeStreamInfoTiffExt_t*>>(ext_ptr);
        if (base->struct_type == NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO) {
            return base;
        }
        ext_ptr = base->struct_next;
    }
    return nullptr;
}

} // namespace nvimgcodec
