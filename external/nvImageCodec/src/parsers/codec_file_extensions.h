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

#include <nvimgcodec.h>
#include <cstddef>
#include <initializer_list>

namespace nvimgcodec {

// registerFileExtension / unregisterFileExtension were appended to
// nvimgcodecFrameworkDesc_t. Read struct_size before those pointers so an
// extension built against this header does not load them from an older, shorter
// descriptor.
// The threshold is deliberately the end of unregisterFileExtension, not
// sizeof(nvimgcodecFrameworkDesc_t), so appending future members does not
// disable this API. Register also requires unregisterFileExtension to be in
// the struct: both members shipped together, and a register-only descriptor
// is treated as not having the API.
inline bool hasFileExtensionApi(const nvimgcodecFrameworkDesc_t* framework)
{
    return framework != nullptr &&
           framework->struct_size >= offsetof(nvimgcodecFrameworkDesc_t, unregisterFileExtension) +
                                         sizeof(framework->unregisterFileExtension);
}

template <typename Extensions>
inline void registerCodecFileExtensions(
    const nvimgcodecFrameworkDesc_t* framework, const char* codec, const Extensions& file_extensions, int is_default = 0)
{
    if (!hasFileExtensionApi(framework) || framework->registerFileExtension == nullptr || codec == nullptr) {
        return;
    }
    for (const char* ext : file_extensions) {
        framework->registerFileExtension(framework->instance, codec, ext, is_default);
    }
}

template <typename Extensions>
inline void unregisterCodecFileExtensions(
    const nvimgcodecFrameworkDesc_t* framework, const char* codec, const Extensions& file_extensions)
{
    if (!hasFileExtensionApi(framework) || framework->unregisterFileExtension == nullptr || codec == nullptr) {
        return;
    }
    for (const char* ext : file_extensions) {
        framework->unregisterFileExtension(framework->instance, codec, ext);
    }
}

inline void registerCodecFileExtensions(const nvimgcodecFrameworkDesc_t* framework, const char* codec,
    std::initializer_list<const char*> file_extensions, int is_default = 0)
{
    registerCodecFileExtensions<std::initializer_list<const char*>>(framework, codec, file_extensions, is_default);
}

inline void unregisterCodecFileExtensions(
    const nvimgcodecFrameworkDesc_t* framework, const char* codec, std::initializer_list<const char*> file_extensions)
{
    unregisterCodecFileExtensions<std::initializer_list<const char*>>(framework, codec, file_extensions);
}

} // namespace nvimgcodec
