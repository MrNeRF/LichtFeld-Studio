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

#include <cuda_runtime_api.h>
#include <nvimgcodec_version.h>
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "codec.h"
#include "codec_registry.h"
#include "ienvironment.h"
#include "image_decoder_factory.h"
#include "image_encoder.h"
#include "image_encoder_factory.h"
#include "image_parser_factory.h"
#include "log.h"
#include "file_extension_utils.h"

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
    #include <dlfcn.h>
#endif

#define CUDART_MAJOR_VERSION (CUDART_VERSION / 1000)
namespace fs = std::filesystem;

namespace nvimgcodec {

namespace {

template <typename T>
nvimgcodecStatus_t fillList(const std::vector<T>& values, T* output, int* count, ILogger* logger)
{
    if (count == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (values.size() > static_cast<size_t>(INT_MAX)) {
        return NVIMGCODEC_STATUS_INTERNAL_ERROR;
    }
    if (output == nullptr) {
        *count = static_cast<int>(values.size());
        return NVIMGCODEC_STATUS_SUCCESS;
    }
    if (*count < 0) {
        NVIMGCODEC_LOG_ERROR(logger, "Requested list size must not be negative");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (static_cast<size_t>(*count) > values.size()) {
        NVIMGCODEC_LOG_ERROR(logger, "Requested " << *count << " list entries, but only " << values.size() << " are registered");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    std::copy_n(values.begin(), *count, output);
    return NVIMGCODEC_STATUS_SUCCESS;
}

template <size_t SlotSize>
nvimgcodecStatus_t fillNameList(const std::vector<std::string>& values, char* output, int* count, ILogger* logger)
{
    if (count == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (values.size() > static_cast<size_t>(INT_MAX)) {
        return NVIMGCODEC_STATUS_INTERNAL_ERROR;
    }
    if (output == nullptr) {
        *count = static_cast<int>(values.size());
        return NVIMGCODEC_STATUS_SUCCESS;
    }
    if (*count < 0) {
        NVIMGCODEC_LOG_ERROR(logger, "Requested list size must not be negative");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (static_cast<size_t>(*count) > values.size()) {
        NVIMGCODEC_LOG_ERROR(logger, "Requested " << *count << " list entries, but only " << values.size() << " are registered");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    for (int i = 0; i < *count; ++i) {
        if (values[i].size() >= SlotSize) {
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
    }
    for (int i = 0; i < *count; ++i) {
        std::memcpy(output + static_cast<size_t>(i) * SlotSize, values[i].c_str(), values[i].size() + 1);
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

bool isValidBackendKindFilter(nvimgcodecBackendKind_t backend_kind)
{
    return backend_kind == NVIMGCODEC_BACKEND_KIND_ANY || backend_kind == NVIMGCODEC_BACKEND_KIND_CPU_ONLY ||
           backend_kind == NVIMGCODEC_BACKEND_KIND_GPU_ONLY || backend_kind == NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU ||
           backend_kind == NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY;
}

template <typename Desc>
bool isValidComponentDesc(const Desc* desc, nvimgcodecStructureType_t structure_type)
{
    return desc != nullptr && desc->struct_type == structure_type && desc->struct_size == sizeof(Desc) && desc->id != nullptr &&
           desc->id[0] != '\0' && desc->codec != nullptr && desc->codec[0] != '\0' &&
           std::strlen(desc->codec) < NVIMGCODEC_MAX_CODEC_NAME_SIZE;
}

bool hasRequiredCallbacks(const nvimgcodecDecoderDesc_t* desc)
{
    return desc->create != nullptr && desc->destroy != nullptr && desc->canDecode != nullptr &&
           (desc->decode != nullptr || desc->decodeBatch != nullptr);
}

bool hasRequiredCallbacks(const nvimgcodecEncoderDesc_t* desc)
{
    return desc->create != nullptr && desc->destroy != nullptr && desc->canEncode != nullptr && desc->encode != nullptr;
}

bool hasRequiredCallbacks(const nvimgcodecParserDesc_t* desc)
{
    return desc->canParse != nullptr && desc->create != nullptr && desc->destroy != nullptr && desc->getCodeStreamInfo != nullptr &&
           desc->getImageInfo != nullptr;
}

template <typename Callable>
nvimgcodecStatus_t callNoexcept(Callable&& callable) noexcept
{
    try {
        return callable();
    } catch (...) {
        return NVIMGCODEC_STATUS_INTERNAL_ERROR;
    }
}

} // namespace

char GetPathSeparator()
{
#if defined(_WIN32) || defined(_WIN64)
    return ';';
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
    return ':';
#else
#error "Unsupported platform: nvimgcodec only builds on Linux and Windows"
#endif
}

std::string BuildDefaultExtensionsPathFromSoDir(const fs::path& so_dir)
{
    // Default extension discovery follows the directory containing the loaded
    // nvImageCodec runtime library:
    //
    //   Package type                 libnvimgcodec lives in           extensions live in
    //   ---------------------------- -------------------------------- -----------------------------
    //   Linux deb/rpm               <prefix>/lib*/libnvimgcodec/<cuda>/ <so_dir>/extensions/
    //   Linux conda package          $CONDA_PREFIX/lib/libnvimgcodec/<cuda>/ <so_dir>/extensions/
    //   Linux Python wheel           <pkgdir>/                        <pkgdir>/extensions/
    //   Windows MSI installer        <prefix>/bin/<cuda>/             <so_dir>/extensions/
    //   Windows conda package        $CONDA_PREFIX/Library/bin/<cuda>/ <so_dir>/extensions/
    //   Windows Python wheel         <pkgdir>/                        <pkgdir>/extensions/
    return (so_dir / "extensions").generic_string();
}

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)

std::string GetDefaultExtensionsPath()
{
    Dl_info info;
    if (dladdr((const void*)GetDefaultExtensionsPath, &info)) {
        return BuildDefaultExtensionsPathFromSoDir(fs::path(info.dli_fname).parent_path());
    }
    return NVIMGCODEC_DEFAULT_EXTENSIONS_PATH;
}

#elif defined(_WIN32) || defined(_WIN64)

std::string GetDefaultExtensionsPath()
{
    char dll_path[MAX_PATH];
    HMODULE hm = NULL;

    if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR)&GetDefaultExtensionsPath, &hm) != 0) {
        if (GetModuleFileName(hm, dll_path, sizeof(dll_path)) != 0) {
            return BuildDefaultExtensionsPathFromSoDir(fs::path(dll_path).parent_path());
        }
    }

    return NVIMGCODEC_DEFAULT_EXTENSIONS_PATH;
}

#else
#error "Unsupported platform: nvimgcodec only builds on Linux and Windows"
#endif

PluginFramework::PluginFramework(ILogger* logger, ICodecRegistry* codec_registry, std::unique_ptr<IEnvironment> env,
    std::unique_ptr<IDirectoryScaner> directory_scaner, std::unique_ptr<ILibraryLoader> library_loader, const std::string& extensions_path,
    std::atomic_size_t* active_clients, std::recursive_mutex* registry_mutex)
    : logger_(logger)
    , env_(std::move(env))
    , directory_scaner_(std::move(directory_scaner))
    , library_loader_(std::move(library_loader))
    , framework_desc_{NVIMGCODEC_STRUCTURE_TYPE_FRAMEWORK_DESC, sizeof(nvimgcodecFrameworkDesc_t), nullptr, this, "nvImageCodec", NVIMGCODEC_VER,
          CUDART_VERSION, &static_log, &static_register_encoder, &static_unregister_encoder, &static_register_decoder,
          &static_unregister_decoder, &static_register_parser, &static_unregister_parser, &static_register_file_extension,
          &static_unregister_file_extension, &static_get_codec_names, &static_get_file_extensions,
          &static_get_decoder_descs, &static_get_encoder_descs, &static_get_parser_descs}
    , codec_registry_(codec_registry)
    , extension_paths_{}
    , active_clients_(active_clients)
    , registry_mutex_(registry_mutex ? registry_mutex : &owned_registry_mutex_)
{

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

PluginFramework::~PluginFramework()
{
    unregisterAllExtensions();
}

nvimgcodecStatus_t PluginFramework::static_register_encoder(void* instance, const nvimgcodecEncoderDesc_t* desc, float priority)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return callNoexcept([&] { return handle->registerEncoder(desc, priority); });
}

nvimgcodecStatus_t PluginFramework::static_register_decoder(void* instance, const nvimgcodecDecoderDesc_t* desc, float priority)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return callNoexcept([&] { return handle->registerDecoder(desc, priority); });
}

nvimgcodecStatus_t PluginFramework::static_register_parser(void* instance, const nvimgcodecParserDesc_t* desc, float priority)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return callNoexcept([&] { return handle->registerParser(desc, priority); });
}

nvimgcodecStatus_t PluginFramework::static_unregister_encoder(void* instance, const nvimgcodecEncoderDesc_t* desc)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return callNoexcept([&] { return handle->unregisterEncoder(desc); });
}

nvimgcodecStatus_t PluginFramework::static_unregister_decoder(void* instance, const nvimgcodecDecoderDesc_t* desc)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return callNoexcept([&] { return handle->unregisterDecoder(desc); });
}

nvimgcodecStatus_t PluginFramework::static_unregister_parser(void* instance, const nvimgcodecParserDesc_t* desc)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return callNoexcept([&] { return handle->unregisterParser(desc); });
}

nvimgcodecStatus_t PluginFramework::static_get_codec_names(
    void* instance, char* codec_names, int* count, const char* file_extension)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return callNoexcept(
        [&] { return reinterpret_cast<PluginFramework*>(instance)->getCodecNames(codec_names, count, file_extension); });
}

nvimgcodecStatus_t PluginFramework::static_get_file_extensions(
    void* instance, char* file_extensions, int* count, const char* codec_name)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return callNoexcept(
        [&] { return reinterpret_cast<PluginFramework*>(instance)->getFileExtensions(file_extensions, count, codec_name); });
}

nvimgcodecStatus_t PluginFramework::static_get_decoder_descs(
    void* instance, const nvimgcodecDecoderDesc_t** decoder_descs, int* count, const char* codec_name, nvimgcodecBackendKind_t backend_kind)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return callNoexcept(
        [&] { return reinterpret_cast<PluginFramework*>(instance)->getDecoderDescs(decoder_descs, count, codec_name, backend_kind); });
}

nvimgcodecStatus_t PluginFramework::static_get_encoder_descs(
    void* instance, const nvimgcodecEncoderDesc_t** encoder_descs, int* count, const char* codec_name, nvimgcodecBackendKind_t backend_kind)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return callNoexcept(
        [&] { return reinterpret_cast<PluginFramework*>(instance)->getEncoderDescs(encoder_descs, count, codec_name, backend_kind); });
}

nvimgcodecStatus_t PluginFramework::static_get_parser_descs(
    void* instance, const nvimgcodecParserDesc_t** parser_descs, int* count, const char* codec_name)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return callNoexcept([&] { return reinterpret_cast<PluginFramework*>(instance)->getParserDescs(parser_descs, count, codec_name); });
}

nvimgcodecStatus_t PluginFramework::static_register_file_extension(
    void* instance, const char* codec, const char* file_extension, int is_default)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->registerFileExtension(codec, file_extension, is_default);
}

nvimgcodecStatus_t PluginFramework::static_unregister_file_extension(void* instance, const char* codec, const char* file_extension)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->unregisterFileExtension(codec, file_extension);
}

nvimgcodecStatus_t PluginFramework::static_log(void* instance, const nvimgcodecDebugMessageSeverity_t message_severity,
    const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* data)
{
    if (instance == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return callNoexcept([&] { return handle->log(message_severity, message_category, data); });
}

nvimgcodecStatus_t PluginFramework::registerExtension(
    nvimgcodecExtension_t* extension, const nvimgcodecExtensionDesc_t* extension_desc, const Module& module)
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
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

    if (hasActiveClients()) {
        NVIMGCODEC_LOG_WARNING(logger_, "Cannot register an extension while decoders, encoders, or code streams are active");
        return NVIMGCODEC_STATUS_RESOURCE_IN_USE;
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
            nvimgcodecStatus_t unregister_status = unregisterExtension(it);
            if (unregister_status != NVIMGCODEC_STATUS_SUCCESS) {
                return unregister_status;
            }
        }
    }

    NVIMGCODEC_LOG_INFO(
        logger_, "Registering extension " << extension_desc->id << " version:" << NVIMGCODEC_STREAM_VER(extension_desc->version));
    PluginFramework::Extension internal_extension;
    internal_extension.public_handle_ = std::make_unique<::nvimgcodecExtension>();
    internal_extension.public_handle_->framework_ = this;
    internal_extension.plugin_handle_ = nullptr;
    internal_extension.desc_ = *extension_desc;
    internal_extension.module_ = module;
    nvimgcodecStatus_t status = callNoexcept([&] {
        return internal_extension.desc_.create(internal_extension.desc_.instance, &internal_extension.plugin_handle_, &framework_desc_);
    });
    if (status != NVIMGCODEC_STATUS_SUCCESS) {
        NVIMGCODEC_LOG_ERROR(logger_, "Could not create extension");
        return status;
    }

    auto registered = extensions_.emplace(extension_desc->id, std::move(internal_extension)).first;
    if (extension)
        *extension = registered->second.public_handle_.get();
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PluginFramework::registerExtension(nvimgcodecExtension_t* extension, const nvimgcodecExtensionDesc_t* extension_desc)
{
    Module module;
    module.lib_handle_ = nullptr;
    module.extension_entry_ = nullptr;

    return registerExtension(extension, extension_desc, module);
}

bool PluginFramework::hasActiveClients() const
{
    return active_clients_ != nullptr && active_clients_->load(std::memory_order_acquire) != 0;
}

nvimgcodecStatus_t PluginFramework::unregisterExtension(std::map<std::string, Extension>::iterator it)
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
    if (hasActiveClients()) {
        NVIMGCODEC_LOG_WARNING(logger_, "Cannot unregister an extension while decoders, encoders, or code streams are active");
        return NVIMGCODEC_STATUS_RESOURCE_IN_USE;
    }
    NVIMGCODEC_LOG_INFO(
        logger_, "Unregistering extension " << it->second.desc_.id << " version:" << NVIMGCODEC_STREAM_VER(it->second.desc_.version));
    if (!it->second.plugin_destroyed_) {
        nvimgcodecStatus_t status = callNoexcept([&] { return it->second.desc_.destroy(it->second.plugin_handle_); });
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(logger_, "Could not destroy extension " << it->second.desc_.id);
            return status;
        }
        it->second.plugin_destroyed_ = true;
        it->second.plugin_handle_ = nullptr;
    }

    if (it->second.module_.lib_handle_ != nullptr) {
        NVIMGCODEC_LOG_INFO(logger_, "Unloading extension module:" << it->second.module_.path_);
        nvimgcodecStatus_t status = callNoexcept([&] {
            library_loader_->unloadLibrary(it->second.module_.lib_handle_);
            return NVIMGCODEC_STATUS_SUCCESS;
        });
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(logger_, "Could not unload extension module " << it->second.module_.path_);
            return status;
        }
        it->second.module_.lib_handle_ = nullptr;
    }
    extensions_.erase(it);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PluginFramework::unregisterExtension(nvimgcodecExtension_t extension)
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
    auto it = std::find_if(
        extensions_.begin(), extensions_.end(), [&](const auto& e) -> bool { return e.second.public_handle_.get() == extension; });
    if (it == extensions_.end()) {
        NVIMGCODEC_LOG_WARNING(logger_, "Could not find extension to unregister ");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    return unregisterExtension(it);
}

void PluginFramework::unregisterAllExtensions()
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
    while (!extensions_.empty()) {
        auto it = extensions_.begin();
        if (unregisterExtension(it) != NVIMGCODEC_STATUS_SUCCESS) {
            // Keep the module loaded rather than unloading code whose extension failed
            // to destroy. Erasing the record prevents an infinite teardown loop.
            extensions_.erase(it);
        }
    }
}

nvimgcodecStatus_t PluginFramework::getExtensions(nvimgcodecExtension_t* extensions, int* count) const
{
    std::vector<nvimgcodecExtension_t> handles;
    handles.reserve(extensions_.size());
    for (const auto& entry : extensions_) {
        handles.push_back(entry.second.public_handle_.get());
    }
    return fillList(handles, extensions, count, logger_);
}

nvimgcodecStatus_t PluginFramework::getExtensionDesc(
    nvimgcodecExtension_t extension, const nvimgcodecExtensionDesc_t** extension_desc) const
{
    if (extension == nullptr || extension_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    auto it = std::find_if(
        extensions_.begin(), extensions_.end(), [&](const auto& entry) { return entry.second.public_handle_.get() == extension; });
    if (it == extensions_.end()) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    *extension_desc = &it->second.desc_;
    return NVIMGCODEC_STATUS_SUCCESS;
}

const nvimgcodecFrameworkDesc_t* PluginFramework::getFrameworkDesc() const
{
    return &framework_desc_;
}

nvimgcodecStatus_t PluginFramework::getCodecNames(char* codec_names, int* count, const char* file_extension) const
{
    std::vector<std::string> names;
    if (file_extension != nullptr) {
        names = codec_registry_->getCodecNames(normalizeFileExtension(file_extension));
    } else {
        names.reserve(codec_registry_->getCodecsCount());
        for (size_t i = 0; i < codec_registry_->getCodecsCount(); ++i) {
            ICodec* codec = codec_registry_->getCodecByIndex(i);
            if (codec->hasRegistrations()) {
                names.push_back(codec->name());
            }
        }
    }
    return fillNameList<NVIMGCODEC_MAX_CODEC_NAME_SIZE>(names, codec_names, count, logger_);
}

nvimgcodecStatus_t PluginFramework::getFileExtensions(
    char* file_extensions, int* count, const char* codec_name) const
{
    std::vector<std::string> extensions =
        codec_name != nullptr ? codec_registry_->getFileExtensions(codec_name) : codec_registry_->getFileExtensions();
    return fillNameList<NVIMGCODEC_MAX_FILE_EXTENSION_SIZE>(extensions, file_extensions, count, logger_);
}

nvimgcodecStatus_t PluginFramework::getDecoderDescs(
    const nvimgcodecDecoderDesc_t** decoder_descs, int* count, const char* codec_name, nvimgcodecBackendKind_t backend_kind) const
{
    if (!isValidBackendKindFilter(backend_kind)) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    std::vector<const nvimgcodecDecoderDesc_t*> descs;
    auto append_codec = [&](ICodec* codec) {
        for (IImageDecoderFactory* factory : codec->getDecoderFactories()) {
            if (factory != nullptr && (backend_kind == NVIMGCODEC_BACKEND_KIND_ANY || factory->getBackendKind() == backend_kind)) {
                descs.push_back(factory->getDecoderDesc());
            }
        }
    };

    if (codec_name != nullptr) {
        if (codec_name[0] == '\0') {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        ICodec* codec = codec_registry_->getCodecByName(codec_name);
        if (codec != nullptr) {
            append_codec(codec);
        }
    } else {
        for (size_t i = 0; i < codec_registry_->getCodecsCount(); ++i) {
            append_codec(codec_registry_->getCodecByIndex(i));
        }
    }
    return fillList(descs, decoder_descs, count, logger_);
}

nvimgcodecStatus_t PluginFramework::getEncoderDescs(
    const nvimgcodecEncoderDesc_t** encoder_descs, int* count, const char* codec_name, nvimgcodecBackendKind_t backend_kind) const
{
    if (!isValidBackendKindFilter(backend_kind)) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    std::vector<const nvimgcodecEncoderDesc_t*> descs;
    auto append_codec = [&](ICodec* codec) {
        for (IImageEncoderFactory* factory : codec->getEncoderFactories()) {
            if (factory != nullptr && (backend_kind == NVIMGCODEC_BACKEND_KIND_ANY || factory->getBackendKind() == backend_kind)) {
                descs.push_back(factory->getEncoderDesc());
            }
        }
    };

    if (codec_name != nullptr) {
        if (codec_name[0] == '\0') {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        ICodec* codec = codec_registry_->getCodecByName(codec_name);
        if (codec != nullptr) {
            append_codec(codec);
        }
    } else {
        for (size_t i = 0; i < codec_registry_->getCodecsCount(); ++i) {
            append_codec(codec_registry_->getCodecByIndex(i));
        }
    }
    return fillList(descs, encoder_descs, count, logger_);
}

nvimgcodecStatus_t PluginFramework::getParserDescs(const nvimgcodecParserDesc_t** parser_descs, int* count, const char* codec_name) const
{
    std::vector<const nvimgcodecParserDesc_t*> descs;
    auto append_codec = [&](ICodec* codec) {
        for (IImageParserFactory* factory : codec->getParserFactories()) {
            if (factory != nullptr) {
                descs.push_back(factory->getParserDesc());
            }
        }
    };

    if (codec_name != nullptr) {
        if (codec_name[0] == '\0') {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        ICodec* codec = codec_registry_->getCodecByName(codec_name);
        if (codec != nullptr) {
            append_codec(codec);
        }
    } else {
        for (size_t i = 0; i < codec_registry_->getCodecsCount(); ++i) {
            append_codec(codec_registry_->getCodecByIndex(i));
        }
    }
    return fillList(descs, parser_descs, count, logger_);
}

bool is_extension_disabled(fs::path dir_entry_path)
{
    const std::string filename = dir_entry_path.filename().string();
    return !filename.empty() && filename.front() == '~';
}

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool is_loadable_extension_module(fs::path dir_entry_path)
{
#if defined(_WIN32) || defined(_WIN64)
    return lowercase(dir_entry_path.extension().string()) == ".dll";
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
    const std::string filename = lowercase(dir_entry_path.filename().string());
    const auto so_pos = filename.rfind(".so");
    if (so_pos == std::string::npos) {
        return false;
    }

    const auto suffix_pos = so_pos + 3;
    if (suffix_pos == filename.size()) {
        return true;
    }
    if (filename[suffix_pos] != '.') {
        return false;
    }

    bool expect_digit = true;
    for (auto it = filename.begin() + suffix_pos + 1; it != filename.end(); ++it) {
        if (*it == '.') {
            if (expect_digit) {
                return false;
            }
            expect_digit = true;
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(*it))) {
            return false;
        }
        expect_digit = false;
    }
    return !expect_digit;
#else
#error "Unsupported platform: nvimgcodec only builds on Linux and Windows"
#endif
}

void PluginFramework::discoverAndLoadExtModules()
{
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
                // Extension install dirs may contain import/static libraries; do
                // not log load errors for files that are known non-modules.
                if (!is_loadable_extension_module(dir_entry_path)) {
                    NVIMGCODEC_LOG_DEBUG(logger_, "Skipping non-extension module [" << dir_entry_path.string() << "]");
                    continue;
                }
                if (is_extension_disabled(dir_entry_path)) {
                    continue;
                }
                const std::string module_path(dir_entry_path.string());
                loadExtModule(module_path);
            }
        }
    }
}

void PluginFramework::loadExtModule(const std::string& modulePath)
{
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
    } catch (const std::exception& e) {
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
        if (!module.extension_entry_) {
            throw std::runtime_error("nvimgcodecExtensionModuleEntry was not found");
        }

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

ICodec* PluginFramework::ensureExistsAndRetrieveCodec(const char* codec_name)
{
    ICodec* codec = codec_registry_->getCodecByName(codec_name);
    if (codec == nullptr) {
        NVIMGCODEC_LOG_INFO(logger_, "Codec " << codec_name << " not yet registered, registering for first time");
        std::unique_ptr<Codec> new_codec = std::make_unique<Codec>(logger_, codec_name);
        codec_registry_->registerCodec(std::move(new_codec));
        codec = codec_registry_->getCodecByName(codec_name);
    }
    return codec;
}

nvimgcodecStatus_t PluginFramework::registerEncoder(const nvimgcodecEncoderDesc_t* desc, float priority)
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
    if (!isValidComponentDesc(desc, NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC) || !hasRequiredCallbacks(desc) ||
        !isValidBackendKindFilter(desc->backend_kind) || desc->backend_kind == NVIMGCODEC_BACKEND_KIND_ANY || !std::isfinite(priority)) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (hasActiveClients()) {
        return NVIMGCODEC_STATUS_RESOURCE_IN_USE;
    }
    NVIMGCODEC_LOG_INFO(logger_, "Framework is registering encoder (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
    for (IImageEncoderFactory* factory : codec->getEncoderFactories()) {
        if (factory->getEncoderId() == desc->id) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
    }
    std::unique_ptr<IImageEncoderFactory> encoder_factory = std::make_unique<ImageEncoderFactory>(desc);
    codec->registerEncoderFactory(std::move(encoder_factory), priority);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PluginFramework::unregisterEncoder(const nvimgcodecEncoderDesc_t* desc)
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
    if (!isValidComponentDesc(desc, NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC)) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (hasActiveClients()) {
        return NVIMGCODEC_STATUS_RESOURCE_IN_USE;
    }
    NVIMGCODEC_LOG_INFO(logger_, "Framework is unregistering encoder (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = codec_registry_->getCodecByName(desc->codec);
    if (codec == nullptr) {
        NVIMGCODEC_LOG_WARNING(logger_, "Codec " << desc->codec << " not registered");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return codec->unregisterEncoderFactory(desc) ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_INVALID_PARAMETER;
}

nvimgcodecStatus_t PluginFramework::registerDecoder(const nvimgcodecDecoderDesc_t* desc, float priority)
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
    if (!isValidComponentDesc(desc, NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC) || !hasRequiredCallbacks(desc) ||
        !isValidBackendKindFilter(desc->backend_kind) || desc->backend_kind == NVIMGCODEC_BACKEND_KIND_ANY || !std::isfinite(priority)) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (hasActiveClients()) {
        return NVIMGCODEC_STATUS_RESOURCE_IN_USE;
    }
    NVIMGCODEC_LOG_INFO(logger_, "Framework is registering decoder (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
    for (IImageDecoderFactory* factory : codec->getDecoderFactories()) {
        if (factory->getDecoderId() == desc->id) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
    }
    std::unique_ptr<IImageDecoderFactory> decoder_factory = std::make_unique<ImageDecoderFactory>(desc);
    codec->registerDecoderFactory(std::move(decoder_factory), priority);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PluginFramework::unregisterDecoder(const nvimgcodecDecoderDesc_t* desc)
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
    if (!isValidComponentDesc(desc, NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC)) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (hasActiveClients()) {
        return NVIMGCODEC_STATUS_RESOURCE_IN_USE;
    }
    NVIMGCODEC_LOG_INFO(logger_, "Framework is unregistering decoder (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = codec_registry_->getCodecByName(desc->codec);
    if (codec == nullptr) {
        NVIMGCODEC_LOG_WARNING(logger_, "Codec " << desc->codec << " not registered");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return codec->unregisterDecoderFactory(desc) ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_INVALID_PARAMETER;
}

nvimgcodecStatus_t PluginFramework::registerParser(const nvimgcodecParserDesc_t* desc, float priority)
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
    if (!isValidComponentDesc(desc, NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC) || !hasRequiredCallbacks(desc) || !std::isfinite(priority)) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (hasActiveClients()) {
        return NVIMGCODEC_STATUS_RESOURCE_IN_USE;
    }
    NVIMGCODEC_LOG_INFO(logger_, "Framework is registering parser (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
    for (IImageParserFactory* factory : codec->getParserFactories()) {
        if (factory->getParserId() == desc->id) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
    }
    std::unique_ptr<IImageParserFactory> parser_factory = std::make_unique<ImageParserFactory>(desc);
    codec->registerParserFactory(std::move(parser_factory), priority);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PluginFramework::unregisterParser(const nvimgcodecParserDesc_t* desc)
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
    if (!isValidComponentDesc(desc, NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC)) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (hasActiveClients()) {
        return NVIMGCODEC_STATUS_RESOURCE_IN_USE;
    }
    NVIMGCODEC_LOG_INFO(logger_, "Framework is unregistering parser (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = codec_registry_->getCodecByName(desc->codec);
    if (codec == nullptr) {
        NVIMGCODEC_LOG_WARNING(logger_, "Codec " << desc->codec << " not registered");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return codec->unregisterParserFactory(desc) ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_INVALID_PARAMETER;
}

nvimgcodecStatus_t PluginFramework::registerFileExtension(const char* codec, const char* file_extension, int is_default)
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
    if (codec == nullptr || codec[0] == '\0' || file_extension == nullptr) {
        NVIMGCODEC_LOG_ERROR(logger_, "Codec name and file extension are required to register a file extension");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (!codecNameFitsCApiSlot(codec)) {
        NVIMGCODEC_LOG_ERROR(logger_, "Codec name exceeds NVIMGCODEC_MAX_CODEC_NAME_SIZE");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    std::string normalized = normalizeFileExtension(file_extension);
    if (!fileExtensionFitsCApiSlot(normalized)) {
        NVIMGCODEC_LOG_ERROR(logger_, "File extension is empty or exceeds NVIMGCODEC_MAX_FILE_EXTENSION_SIZE");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (hasActiveClients()) {
        NVIMGCODEC_LOG_WARNING(logger_, "Cannot register a file extension while decoders, encoders, or code streams are active");
        return NVIMGCODEC_STATUS_RESOURCE_IN_USE;
    }
    NVIMGCODEC_LOG_INFO(logger_, "Framework is registering file extension " << normalized << " for codec " << codec
                                                                            << (is_default ? " (default)" : ""));
    ICodec* codec_obj = ensureExistsAndRetrieveCodec(codec);
    codec_registry_->registerFileExtension(normalized, codec_obj->name(), is_default != 0);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PluginFramework::unregisterFileExtension(const char* codec, const char* file_extension)
{
    std::lock_guard<std::recursive_mutex> registry_lock(*registry_mutex_);
    if (codec == nullptr || codec[0] == '\0' || file_extension == nullptr) {
        NVIMGCODEC_LOG_ERROR(logger_, "Codec name and file extension are required to unregister a file extension");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    std::string normalized = normalizeFileExtension(file_extension);
    if (normalized.empty()) {
        NVIMGCODEC_LOG_ERROR(logger_, "File extension cannot be empty");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (hasActiveClients()) {
        NVIMGCODEC_LOG_WARNING(logger_, "Cannot unregister a file extension while decoders, encoders, or code streams are active");
        return NVIMGCODEC_STATUS_RESOURCE_IN_USE;
    }
    NVIMGCODEC_LOG_INFO(logger_, "Framework is unregistering file extension " << normalized << " for codec " << codec);
    ICodec* codec_obj = codec_registry_->getCodecByName(codec);
    if (codec_obj == nullptr) {
        NVIMGCODEC_LOG_WARNING(logger_, "Codec " << codec << " not registered");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    codec_registry_->unregisterFileExtension(normalized, codec_obj->name());
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PluginFramework::log(const nvimgcodecDebugMessageSeverity_t message_severity,
    const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* data)
{
    logger_->log(message_severity, message_category, data);
    return NVIMGCODEC_STATUS_SUCCESS;
}
} // namespace nvimgcodec
