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
#include <atomic>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "idirectory_scaner.h"
#include "ilibrary_loader.h"

namespace nvimgcodec {

class PluginFramework;

} // namespace nvimgcodec

struct nvimgcodecExtension
{
    nvimgcodec::PluginFramework* framework_;
};

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
class PluginFramework
{
  public:
    explicit PluginFramework(ILogger* logger, ICodecRegistry* codec_registry, std::unique_ptr<IEnvironment> env,
        std::unique_ptr<IDirectoryScaner> directory_scaner, std::unique_ptr<ILibraryLoader> library_loader,
        const std::string& extensions_path, std::atomic_size_t* active_clients = nullptr,
        std::recursive_mutex* registry_mutex = nullptr);
    ~PluginFramework();
    nvimgcodecStatus_t registerExtension(nvimgcodecExtension_t* extension, const nvimgcodecExtensionDesc_t* extension_desc);
    nvimgcodecStatus_t unregisterExtension(nvimgcodecExtension_t extension);
    void unregisterAllExtensions();
    nvimgcodecStatus_t getExtensions(nvimgcodecExtension_t* extensions, int* count) const;
    nvimgcodecStatus_t getExtensionDesc(nvimgcodecExtension_t extension, const nvimgcodecExtensionDesc_t** extension_desc) const;
    const nvimgcodecFrameworkDesc_t* getFrameworkDesc() const;

    void discoverAndLoadExtModules();
    void loadExtModule(const std::string& modulePath);

  private:
    struct Module
    {
        std::string path_;
        ILibraryLoader::LibraryHandle lib_handle_;
        nvimgcodecExtensionModuleEntryFunc_t extension_entry_;
    };

    struct Extension
    {
        std::unique_ptr<::nvimgcodecExtension> public_handle_;
        nvimgcodecExtension_t plugin_handle_;
        nvimgcodecExtensionDesc_t desc_;
        Module module_;
        bool plugin_destroyed_ = false;
    };

    nvimgcodecStatus_t registerExtension(
        nvimgcodecExtension_t* extension, const nvimgcodecExtensionDesc_t* extension_desc, const Module& module);
    nvimgcodecStatus_t unregisterExtension(std::map<std::string, Extension>::iterator it);
    bool hasActiveClients() const;

    ICodec* ensureExistsAndRetrieveCodec(const char* codec_name);

    nvimgcodecStatus_t registerEncoder(const nvimgcodecEncoderDesc_t* desc, float priority);
    nvimgcodecStatus_t unregisterEncoder(const nvimgcodecEncoderDesc_t* desc);
    nvimgcodecStatus_t registerDecoder(const nvimgcodecDecoderDesc_t* desc, float priority);
    nvimgcodecStatus_t unregisterDecoder(const nvimgcodecDecoderDesc_t* desc);
    nvimgcodecStatus_t registerParser(const nvimgcodecParserDesc_t* desc, float priority);
    nvimgcodecStatus_t unregisterParser(const nvimgcodecParserDesc_t* desc);
    nvimgcodecStatus_t registerFileExtension(const char* codec, const char* file_extension, int is_default);
    nvimgcodecStatus_t unregisterFileExtension(const char* codec, const char* file_extension);
    nvimgcodecStatus_t getCodecNames(char* codec_names, int* count, const char* file_extension) const;
    nvimgcodecStatus_t getFileExtensions(char* file_extensions, int* count, const char* codec_name) const;
    nvimgcodecStatus_t getDecoderDescs(
        const nvimgcodecDecoderDesc_t** decoder_descs, int* count, const char* codec_name, nvimgcodecBackendKind_t backend_kind) const;
    nvimgcodecStatus_t getEncoderDescs(
        const nvimgcodecEncoderDesc_t** encoder_descs, int* count, const char* codec_name, nvimgcodecBackendKind_t backend_kind) const;
    nvimgcodecStatus_t getParserDescs(const nvimgcodecParserDesc_t** parser_descs, int* count, const char* codec_name) const;

    nvimgcodecStatus_t log(const nvimgcodecDebugMessageSeverity_t message_severity, const nvimgcodecDebugMessageCategory_t message_category,
        const nvimgcodecDebugMessageData_t* callback_data);

    static nvimgcodecStatus_t static_register_encoder(void* instance, const nvimgcodecEncoderDesc_t* desc, float priority);
    static nvimgcodecStatus_t static_unregister_encoder(void* instance, const nvimgcodecEncoderDesc_t* desc);
    static nvimgcodecStatus_t static_register_decoder(void* instance, const nvimgcodecDecoderDesc_t* desc, float priority);
    static nvimgcodecStatus_t static_unregister_decoder(void* instance, const nvimgcodecDecoderDesc_t* desc);
    static nvimgcodecStatus_t static_register_parser(void* instance, const nvimgcodecParserDesc_t* desc, float priority);
    static nvimgcodecStatus_t static_unregister_parser(void* instance, const nvimgcodecParserDesc_t* desc);
    static nvimgcodecStatus_t static_register_file_extension(
        void* instance, const char* codec, const char* file_extension, int is_default);
    static nvimgcodecStatus_t static_unregister_file_extension(void* instance, const char* codec, const char* file_extension);
    static nvimgcodecStatus_t static_get_codec_names(
        void* instance, char* codec_names, int* count, const char* file_extension);
    static nvimgcodecStatus_t static_get_file_extensions(
        void* instance, char* file_extensions, int* count, const char* codec_name);
    static nvimgcodecStatus_t static_get_decoder_descs(void* instance, const nvimgcodecDecoderDesc_t** decoder_descs, int* count,
        const char* codec_name, nvimgcodecBackendKind_t backend_kind);
    static nvimgcodecStatus_t static_get_encoder_descs(void* instance, const nvimgcodecEncoderDesc_t** encoder_descs, int* count,
        const char* codec_name, nvimgcodecBackendKind_t backend_kind);
    static nvimgcodecStatus_t static_get_parser_descs(
        void* instance, const nvimgcodecParserDesc_t** parser_descs, int* count, const char* codec_name);

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
    std::atomic_size_t* active_clients_;
    std::recursive_mutex owned_registry_mutex_;
    std::recursive_mutex* registry_mutex_;
};
} // namespace nvimgcodec
