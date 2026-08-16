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

#include "nvimgcodec_director.h"
#include "builtin_modules.h"
#include "code_stream.h"
#include "directory_scaner.h"
#include "environment.h"
#include "image_generic_decoder.h"
#include "image_generic_encoder.h"
#include "iostream_factory.h"
#include "library_loader.h"

namespace nvimgcodec {

    NvImgCodecDirector::NvImgCodecDirector(const nvimgcodecInstanceCreateInfo_t* create_info)
        : logger_("nvimgcodec"), default_debug_messenger_manager_(&logger_, create_info), codec_registry_(&logger_), plugin_framework_(&logger_, &codec_registry_, std::move(std::make_unique<Environment>()), std::move(std::make_unique<DirectoryScaner>()), std::move(std::make_unique<LibraryLoader>()), create_info->extension_modules_path ? create_info->extension_modules_path : "") {
        if (create_info->load_builtin_modules) {
            for (auto builtin_ext : get_builtin_modules())
                plugin_framework_.registerExtension(nullptr, &builtin_ext);
        }

        if (create_info->load_extension_modules) {
            plugin_framework_.discoverAndLoadExtModules();
        }
    }

    NvImgCodecDirector::~NvImgCodecDirector() {
    }

    std::unique_ptr<CodeStream> NvImgCodecDirector::createCodeStream() {
        return std::make_unique<CodeStream>(&codec_registry_, std::make_unique<IoStreamFactory>());
    }

    std::unique_ptr<ImageGenericDecoder> NvImgCodecDirector::createGenericDecoder(
        const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        return std::make_unique<ImageGenericDecoder>(&logger_, &codec_registry_, exec_params, options);
    }

    std::unique_ptr<ImageGenericEncoder> NvImgCodecDirector::createGenericEncoder(
        const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        return std::make_unique<ImageGenericEncoder>(&logger_, &codec_registry_, exec_params, options);
    }

    void NvImgCodecDirector::registerDebugMessenger(IDebugMessenger* messenger) {
        logger_.registerDebugMessenger(messenger);
    }

    void NvImgCodecDirector::unregisterDebugMessenger(IDebugMessenger* messenger) {
        logger_.unregisterDebugMessenger(messenger);
    }

} // namespace nvimgcodec
