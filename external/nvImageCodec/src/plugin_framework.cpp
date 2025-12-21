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

#include "plugin_framework.h"

#include <algorithm>
#include <cstring>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <iostream>
#include <nvimgcodec_version.h>
#include <sstream>

#include "codec.h"
#include "codec_registry.h"
#include "ienvironment.h"
#include "image_decoder_factory.h"
#include "image_encoder.h"
#include "image_encoder_factory.h"
#include "image_parser_factory.h"
#include "log.h"

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
#include <dlfcn.h>
#endif

#define CUDART_MAJOR_VERSION (CUDART_VERSION / 1000)
namespace fs = std::filesystem;

namespace nvimgcodec {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)

    std::string GetDefaultExtensionsPath() {
        Dl_info info;
        if (dladdr((const void*)GetDefaultExtensionsPath, &info)) {
            const fs::path lib_dir = fs::path(info.dli_fname).parent_path();

            // Installed layout: lib64/../extensions
            fs::path installed_path = lib_dir;
            if (installed_path.filename() == "lib64") {
                installed_path = installed_path.parent_path();
            }
            installed_path /= "extensions";

            // Portable layout: ./extensions (priority)
            const fs::path portable_path = lib_dir / "extensions";
            return portable_path.string() + ":" + installed_path.string();
        }
        return "/opt/nvidia/nvimgcodec_cuda" + std::to_string(CUDART_MAJOR_VERSION) + "/extensions";
    }

    char GetPathSeparator() {
        return ':';
    }

#elif defined(_WIN32) || defined(_WIN64)

    std::string GetDefaultExtensionsPath() {
        char dll_path[MAX_PATH];
        HMODULE hm = NULL;

        if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCSTR)&GetDefaultExtensionsPath, &hm) != 0) {
            if (GetModuleFileName(hm, dll_path, sizeof(dll_path)) != 0) {
                const fs::path lib_dir = fs::path(dll_path).parent_path();

                // Installed layout: bin/../extensions
                fs::path installed_path = lib_dir;
                if (installed_path.filename() == "bin") {
                    installed_path = installed_path.parent_path();
                }
                installed_path /= "extensions";

                // Portable layout: ./extensions (priority)
                const fs::path portable_path = lib_dir / "extensions";
                return portable_path.string() + ";" + installed_path.string();
            }
        }
        return "C:/Program Files/NVIDIA nvImageCodec/v" + std::to_string(NVIMGCODEC_VER_MAJOR) + "." +
               std::to_string(NVIMGCODEC_VER_MINOR) + "/" + std::to_string(CUDART_MAJOR_VERSION) + "/extensions";
    }

    char GetPathSeparator() {
        return ';';
    }

#endif

    PluginFramework::PluginFramework(ILogger* logger, ICodecRegistry* codec_registry, std::unique_ptr<IEnvironment> env,
                                     std::unique_ptr<IDirectoryScaner> directory_scaner, std::unique_ptr<ILibraryLoader> library_loader, const std::string& extensions_path)
        : logger_(logger),
          env_(std::move(env)),
          directory_scaner_(std::move(directory_scaner)),
          library_loader_(std::move(library_loader)),
          framework_desc_{NVIMGCODEC_STRUCTURE_TYPE_FRAMEWORK_DESC, sizeof(nvimgcodecFrameworkDesc_t), nullptr, this, "nvImageCodec", NVIMGCODEC_VER,
                          CUDART_VERSION, &static_log, &static_register_encoder, &static_unregister_encoder, &static_register_decoder,
                          &static_unregister_decoder, &static_register_parser, &static_unregister_parser},
          codec_registry_(codec_registry),
          extension_paths_{} {

        std::string effective_ext_path = extensions_path;
        if (effective_ext_path.empty()) {
            std::string env_extensions_path = env_->getVariable("NVIMGCODEC_EXTENSIONS_PATH");
            effective_ext_path = env_extensions_path.empty() ? GetDefaultExtensionsPath() : std::string(env_extensions_path);
        }
        std::stringstream ss(effective_ext_path);
        std::string current_path;
        while (getline(ss, current_path, GetPathSeparator())) {
            NVIMGCODEC_LOG_DEBUG(logger_, "Using extension path [" << current_path << "]");
            extension_paths_.push_back(current_path);
        }
    }

    PluginFramework::~PluginFramework() {
        unregisterAllExtensions();
    }

    nvimgcodecStatus_t PluginFramework::static_register_encoder(void* instance, const nvimgcodecEncoderDesc_t* desc, float priority) {
        PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
        return handle->registerEncoder(desc, priority);
    }

    nvimgcodecStatus_t PluginFramework::static_register_decoder(void* instance, const nvimgcodecDecoderDesc_t* desc, float priority) {
        PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
        return handle->registerDecoder(desc, priority);
    }

    nvimgcodecStatus_t PluginFramework::static_register_parser(void* instance, const nvimgcodecParserDesc_t* desc, float priority) {
        PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
        return handle->registerParser(desc, priority);
    }

    nvimgcodecStatus_t PluginFramework::static_unregister_encoder(void* instance, const nvimgcodecEncoderDesc_t* desc) {
        PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
        return handle->unregisterEncoder(desc);
    }

    nvimgcodecStatus_t PluginFramework::static_unregister_decoder(void* instance, const nvimgcodecDecoderDesc_t* desc) {
        PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
        return handle->unregisterDecoder(desc);
    }

    nvimgcodecStatus_t PluginFramework::static_unregister_parser(void* instance, const nvimgcodecParserDesc_t* desc) {
        PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
        return handle->unregisterParser(desc);
    }

    nvimgcodecStatus_t PluginFramework::static_log(void* instance, const nvimgcodecDebugMessageSeverity_t message_severity,
                                                   const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* data) {
        PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
        return handle->log(message_severity, message_category, data);
    }

    nvimgcodecStatus_t PluginFramework::registerExtension(
        nvimgcodecExtension_t* extension, const nvimgcodecExtensionDesc_t* extension_desc, const Module& module) {
        if (extension_desc == nullptr) {
            NVIMGCODEC_LOG_ERROR(logger_, "Extension description cannot be null");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        if (extension_desc->id == nullptr) {
            NVIMGCODEC_LOG_ERROR(logger_, "Extension id cannot be null");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        if (extension_desc->create == nullptr) {
            NVIMGCODEC_LOG_ERROR(logger_, "Could not find  'create' function in extension");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        if (extension_desc->destroy == nullptr) {
            NVIMGCODEC_LOG_ERROR(logger_, "Could not find  'destroy' function in extension");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        if (extension_desc->ext_api_version > NVIMGCODEC_VER) {
            NVIMGCODEC_LOG_WARNING(logger_, "Could not register extension "
                                                << extension_desc->id << " version:" << NVIMGCODEC_STREAM_VER(extension_desc->version)
                                                << " Extension API version: " << NVIMGCODEC_STREAM_VER(extension_desc->ext_api_version)
                                                << " newer than framework API version: " << NVIMGCODEC_VER);
            return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
        }
        auto it = extensions_.find(extension_desc->id);
        if (it != extensions_.end()) {
            std::string reg_ext_path = !it->second.module_.path_.empty() ? " (" + it->second.module_.path_ + ")" : " (API)";
            if (it->second.desc_.version == extension_desc->version) {
                NVIMGCODEC_LOG_WARNING(logger_, "Could not register extension "
                                                    << extension_desc->id << " version:" << NVIMGCODEC_STREAM_VER(extension_desc->version)
                                                    << " Extension with the same id and version already registered" << reg_ext_path);
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            } else if (it->second.desc_.version > extension_desc->version) {
                NVIMGCODEC_LOG_WARNING(
                    logger_, "Could not register extension "
                                 << extension_desc->id << " version:" << NVIMGCODEC_STREAM_VER(extension_desc->version)
                                 << " Extension with the same id and newer version: " << NVIMGCODEC_STREAM_VER(it->second.desc_.version)
                                 << " already registered" << reg_ext_path);
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            } else if (it->second.desc_.version < extension_desc->version) {
                NVIMGCODEC_LOG_WARNING(
                    logger_, "Extension with the same id:" << extension_desc->id
                                                           << " and older version: " << NVIMGCODEC_STREAM_VER(it->second.desc_.version)
                                                           << " already registered and will be unregistered" << reg_ext_path);
                unregisterExtension(it);
            }
        }

        NVIMGCODEC_LOG_INFO(
            logger_, "Registering extension " << extension_desc->id << " version:" << NVIMGCODEC_STREAM_VER(extension_desc->version));
        PluginFramework::Extension internal_extension;
        internal_extension.desc_ = *extension_desc;
        internal_extension.module_ = module;
        nvimgcodecStatus_t status =
            internal_extension.desc_.create(internal_extension.desc_.instance, &internal_extension.handle_, &framework_desc_);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(logger_, "Could not create extension");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        if (extension)
            *extension = internal_extension.handle_;

        extensions_.emplace(extension_desc->id, internal_extension);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t PluginFramework::registerExtension(nvimgcodecExtension_t* extension, const nvimgcodecExtensionDesc_t* extension_desc) {
        Module module;
        module.lib_handle_ = nullptr;
        module.extension_entry_ = nullptr;

        return registerExtension(extension, extension_desc, module);
    }

    nvimgcodecStatus_t PluginFramework::unregisterExtension(std::map<std::string, Extension>::const_iterator it) {
        NVIMGCODEC_LOG_INFO(
            logger_, "Unregistering extension " << it->second.desc_.id << " version:" << NVIMGCODEC_STREAM_VER(it->second.desc_.version));
        it->second.desc_.destroy(it->second.handle_);

        if (it->second.module_.lib_handle_ != nullptr) {
            NVIMGCODEC_LOG_INFO(logger_, "Unloading extension module:" << it->second.module_.path_);
            library_loader_->unloadLibrary(it->second.module_.lib_handle_);
        }
        extensions_.erase(it);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t PluginFramework::unregisterExtension(nvimgcodecExtension_t extension) {
        auto it = std::find_if(extensions_.begin(), extensions_.end(), [&](const auto& e) -> bool { return e.second.handle_ == extension; });
        if (it == extensions_.end()) {
            NVIMGCODEC_LOG_WARNING(logger_, "Could not find extension to unregister ");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        return unregisterExtension(it);
    }

    void PluginFramework::unregisterAllExtensions() {
        while (!extensions_.empty()) {
            unregisterExtension(extensions_.begin());
        }
    }

    bool is_extension_disabled(fs::path dir_entry_path) {
        return dir_entry_path.filename().string().front() == '~';
    }

    void PluginFramework::discoverAndLoadExtModules() {
        for (const auto& dir : extension_paths_) {

            if (!directory_scaner_->exists(dir)) {
                NVIMGCODEC_LOG_DEBUG(logger_, "Plugin dir does not exists [" << dir << "]");
                continue;
            }
            directory_scaner_->start(dir);
            while (directory_scaner_->hasMore()) {
                fs::path dir_entry_path = directory_scaner_->next();
                auto status = directory_scaner_->symlinkStatus(dir_entry_path);
                if (fs::is_regular_file(status) || fs::is_symlink(status)) {
                    if (is_extension_disabled(dir_entry_path)) {
                        continue;
                    }
                    const std::string module_path(dir_entry_path.string());
                    loadExtModule(module_path);
                }
            }
        }
    }

    void PluginFramework::loadExtModule(const std::string& modulePath) {
        NVIMGCODEC_LOG_INFO(logger_, "Loading extension module: " << modulePath);
        PluginFramework::Module module;
        module.path_ = modulePath;
        try {
            module.lib_handle_ = library_loader_->loadLibrary(modulePath);
            if (!module.lib_handle_) {
#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
                throw std::runtime_error(dlerror());
#elif defined(_WIN32) || defined(_WIN64)
                DWORD err = ::GetLastError();
                throw std::runtime_error(std::string("Failed to load library: '#" + std::to_string(err) + "'"));
#endif
            }
        } catch (const std::runtime_error& e) {
            NVIMGCODEC_LOG_ERROR(logger_, "Could not load extension module library. Error: " << e.what());
            return;
        } catch (...) {
            NVIMGCODEC_LOG_ERROR(logger_, "Could not load extension module library: " << modulePath);
            return;
        }

        NVIMGCODEC_LOG_TRACE(logger_, "Getting extension module entry func");
        try {
            module.extension_entry_ = reinterpret_cast<nvimgcodecExtensionModuleEntryFunc_t>(
                library_loader_->getFuncAddress(module.lib_handle_, "nvimgcodecExtensionModuleEntry"));

        } catch (...) {
            NVIMGCODEC_LOG_ERROR(logger_, "Could not find extension module entry function: " << modulePath);
            NVIMGCODEC_LOG_INFO(logger_, "Unloading extension module:" << modulePath);
            library_loader_->unloadLibrary(module.lib_handle_);

            return;
        }
        nvimgcodecExtensionDesc_t extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        nvimgcodecStatus_t status = module.extension_entry_(&extension_desc);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(logger_, "Could not get extension module description");
            NVIMGCODEC_LOG_INFO(logger_, "Unloading extension module:" << modulePath);
            library_loader_->unloadLibrary(module.lib_handle_);

            return;
        }
        nvimgcodecExtension_t extension;
        status = registerExtension(&extension, &extension_desc, module);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_INFO(logger_, "Unloading extension module:" << modulePath);
            library_loader_->unloadLibrary(module.lib_handle_);
        }
    }

    ICodec* PluginFramework::ensureExistsAndRetrieveCodec(const char* codec_name) {
        ICodec* codec = codec_registry_->getCodecByName(codec_name);
        if (codec == nullptr) {
            NVIMGCODEC_LOG_INFO(logger_, "Codec " << codec_name << " not yet registered, registering for first time");
            std::unique_ptr<Codec> new_codec = std::make_unique<Codec>(logger_, codec_name);
            codec_registry_->registerCodec(std::move(new_codec));
            codec = codec_registry_->getCodecByName(codec_name);
        }
        return codec;
    }

    nvimgcodecStatus_t PluginFramework::registerEncoder(const nvimgcodecEncoderDesc_t* desc, float priority) {
        NVIMGCODEC_LOG_INFO(logger_, "Framework is registering encoder (id:" << desc->id << " codec:" << desc->codec << ")");
        ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
        std::unique_ptr<IImageEncoderFactory> encoder_factory = std::make_unique<ImageEncoderFactory>(desc);
        codec->registerEncoderFactory(std::move(encoder_factory), priority);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t PluginFramework::unregisterEncoder(const nvimgcodecEncoderDesc_t* desc) {
        NVIMGCODEC_LOG_INFO(logger_, "Framework is unregistering encoder (id:" << desc->id << " codec:" << desc->codec << ")");
        ICodec* codec = codec_registry_->getCodecByName(desc->codec);
        if (codec == nullptr) {
            NVIMGCODEC_LOG_WARNING(logger_, "Codec " << desc->codec << " not registered");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        codec->unregisterEncoderFactory(desc->id);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t PluginFramework::registerDecoder(const nvimgcodecDecoderDesc_t* desc, float priority) {
        NVIMGCODEC_LOG_INFO(logger_, "Framework is registering decoder (id:" << desc->id << " codec:" << desc->codec << ")");
        ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
        std::unique_ptr<IImageDecoderFactory> decoder_factory = std::make_unique<ImageDecoderFactory>(desc);
        codec->registerDecoderFactory(std::move(decoder_factory), priority);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t PluginFramework::unregisterDecoder(const nvimgcodecDecoderDesc_t* desc) {
        NVIMGCODEC_LOG_INFO(logger_, "Framework is unregistering decoder (id:" << desc->id << " codec:" << desc->codec << ")");
        ICodec* codec = codec_registry_->getCodecByName(desc->codec);
        if (codec == nullptr) {
            NVIMGCODEC_LOG_WARNING(logger_, "Codec " << desc->codec << " not registered");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        codec->unregisterDecoderFactory(desc->id);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t PluginFramework::registerParser(const nvimgcodecParserDesc_t* desc, float priority) {
        NVIMGCODEC_LOG_INFO(logger_, "Framework is registering parser (id:" << desc->id << " codec:" << desc->codec << ")");
        ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
        std::unique_ptr<IImageParserFactory> parser_factory = std::make_unique<ImageParserFactory>(desc);
        codec->registerParserFactory(std::move(parser_factory), priority);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t PluginFramework::unregisterParser(const nvimgcodecParserDesc_t* desc) {
        NVIMGCODEC_LOG_INFO(logger_, "Framework is unregistering parser (id:" << desc->id << " codec:" << desc->codec << ")");
        ICodec* codec = codec_registry_->getCodecByName(desc->codec);
        if (codec == nullptr) {
            NVIMGCODEC_LOG_WARNING(logger_, "Codec " << desc->codec << " not registered");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        codec->unregisterParserFactory(desc->id);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t PluginFramework::log(const nvimgcodecDebugMessageSeverity_t message_severity,
                                            const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* data) {
        logger_->log(message_severity, message_category, data);
        return NVIMGCODEC_STATUS_SUCCESS;
    }
} // namespace nvimgcodec
