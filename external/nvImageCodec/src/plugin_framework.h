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

#include "idirectory_scaner.h"
#include "ilibrary_loader.h"
#include <filesystem>
#include <map>
#include <nvimgcodec.h>
#include <string>
#include <vector>

namespace nvimgcodec {

    class ICodecRegistry;
    class ICodec;
    class IEnvironment;
    class ILogger;

    std::string GetDefaultExtensionsPath();
    char GetPathSeparator();
    // Build the default extensions search path from a libnvimgcodec dynamic-library
    // directory. Returns "<so_dir>/extensions".
    // Exposed for unit tests.
    std::string BuildDefaultExtensionsPathFromSoDir(const std::filesystem::path& so_dir);
    class PluginFramework {
    public:
        explicit PluginFramework(ILogger* logger, ICodecRegistry* codec_registry, std::unique_ptr<IEnvironment> env,
                                 std::unique_ptr<IDirectoryScaner> directory_scaner, std::unique_ptr<ILibraryLoader> library_loader,
                                 const std::string& extensions_path);
        ~PluginFramework();
        nvimgcodecStatus_t registerExtension(nvimgcodecExtension_t* extension, const nvimgcodecExtensionDesc_t* extension_desc);
        nvimgcodecStatus_t unregisterExtension(nvimgcodecExtension_t extension);
        void unregisterAllExtensions();

        void discoverAndLoadExtModules();
        void loadExtModule(const std::string& modulePath);

    private:
        struct Module {
            std::string path_;
            ILibraryLoader::LibraryHandle lib_handle_;
            nvimgcodecExtensionModuleEntryFunc_t extension_entry_;
        };

        struct Extension {
            nvimgcodecExtension_t handle_;
            nvimgcodecExtensionDesc_t desc_;
            Module module_;
        };

        nvimgcodecStatus_t registerExtension(
            nvimgcodecExtension_t* extension, const nvimgcodecExtensionDesc_t* extension_desc, const Module& module);
        nvimgcodecStatus_t unregisterExtension(std::map<std::string, Extension>::const_iterator it);

        ICodec* ensureExistsAndRetrieveCodec(const char* codec_name);

        nvimgcodecStatus_t registerEncoder(const nvimgcodecEncoderDesc_t* desc, float priority);
        nvimgcodecStatus_t unregisterEncoder(const nvimgcodecEncoderDesc_t* desc);
        nvimgcodecStatus_t registerDecoder(const nvimgcodecDecoderDesc_t* desc, float priority);
        nvimgcodecStatus_t unregisterDecoder(const nvimgcodecDecoderDesc_t* desc);
        nvimgcodecStatus_t registerParser(const nvimgcodecParserDesc_t* desc, float priority);
        nvimgcodecStatus_t unregisterParser(const nvimgcodecParserDesc_t* desc);

        nvimgcodecStatus_t log(const nvimgcodecDebugMessageSeverity_t message_severity, const nvimgcodecDebugMessageCategory_t message_category,
                               const nvimgcodecDebugMessageData_t* callback_data);

        static nvimgcodecStatus_t static_register_encoder(void* instance, const nvimgcodecEncoderDesc_t* desc, float priority);
        static nvimgcodecStatus_t static_unregister_encoder(void* instance, const nvimgcodecEncoderDesc_t* desc);
        static nvimgcodecStatus_t static_register_decoder(void* instance, const nvimgcodecDecoderDesc_t* desc, float priority);
        static nvimgcodecStatus_t static_unregister_decoder(void* instance, const nvimgcodecDecoderDesc_t* desc);
        static nvimgcodecStatus_t static_register_parser(void* instance, const nvimgcodecParserDesc_t* desc, float priority);
        static nvimgcodecStatus_t static_unregister_parser(void* instance, const nvimgcodecParserDesc_t* desc);

        static nvimgcodecStatus_t static_log(void* instance, const nvimgcodecDebugMessageSeverity_t message_severity,
                                             const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* callback_data);

        ILogger* logger_;
        std::unique_ptr<IEnvironment> env_;
        std::unique_ptr<IDirectoryScaner> directory_scaner_;
        std::unique_ptr<ILibraryLoader> library_loader_;
        std::map<std::string, Extension> extensions_;
        nvimgcodecFrameworkDesc_t framework_desc_;
        ICodecRegistry* codec_registry_;
        std::vector<std::string> extension_paths_;
    };
} // namespace nvimgcodec
