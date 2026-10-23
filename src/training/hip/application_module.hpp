/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
#include "application_dispatch_api.h"
#include "core/tensor/backend/hip/external_access.hpp"
#include "core/tensor/backend/hip/architecture_runtime.hpp"
#include <filesystem>
#include <stdexcept>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
namespace lfs::training::hip {
class ApplicationModule {
#ifdef _WIN32
    HMODULE handle_ = nullptr;
#else
    void* handle_ = nullptr;
#endif
public:
    explicit ApplicationModule(const std::filesystem::path& path) {
        if (!path.is_absolute()) throw std::invalid_argument("HIP application module requires an absolute path");
#ifdef _WIN32
        handle_ = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle_) throw std::runtime_error("Cannot load HIP application training module");
        try {
            if (symbol<decltype(&lfs_hip_train_abi)>("lfs_hip_train_abi")() != 1)
                throw std::runtime_error("HIP application training ABI mismatch");
            auto install=symbol<decltype(&lfs_hip_application_dispatch_install)>("lfs_hip_application_dispatch_install");
            const LfsHipApplicationDispatch hooks{sizeof(LfsHipApplicationDispatch),nullptr,
                core::internal::hip_external_call_begin,core::internal::hip_external_call_end,
                core::internal::hip_tensor_scratch_allocate,core::internal::hip_tensor_scratch_release};
            LfsHipError error{};
            if(install(&hooks,&error))throw std::runtime_error(std::string("HIP dispatch hooks: ")+error.message);
            const auto utf8=path.u8string();
            std::fprintf(stderr,"[info] HIP training module loaded: %s\n",reinterpret_cast<const char*>(utf8.c_str()));
        } catch (...) { unload(); throw; }
    }
    ApplicationModule(const ApplicationModule&) = delete;
    ApplicationModule& operator=(const ApplicationModule&) = delete;
    ~ApplicationModule() { unload(); }
    template<class F> F symbol(const char* name) const {
#ifdef _WIN32
        auto value = GetProcAddress(handle_, name);
#else
        auto value = dlsym(handle_, name);
#endif
        if (!value) throw std::runtime_error(std::string("Missing HIP application entry point: ")+name);
        return reinterpret_cast<F>(value);
    }
private:
    void unload() noexcept {
        if (!handle_) return;
#ifdef _WIN32
        FreeLibrary(handle_);
#else
        dlclose(handle_);
#endif
        handle_ = nullptr;
    }
};
// Weak cache only: the Tensor context owns code lifetime through its module
// retention seam. The module itself must never own the Tensor context.
inline std::shared_ptr<ApplicationModule> cached_application_module(const std::filesystem::path& path) {
    static std::mutex mutex;
    static std::unordered_map<std::filesystem::path,std::weak_ptr<ApplicationModule>> cache;
    std::lock_guard lock(mutex);
    const auto key = path.lexically_normal();
    auto& entry = cache[key];
    if (auto existing = entry.lock()) return existing;
    auto module = std::make_shared<ApplicationModule>(key);
    entry = module;
    return module;
}
inline std::shared_ptr<ApplicationModule> cached_profile_module(const std::filesystem::path& preferred,
                                                               const std::filesystem::path& generic) {
    if(preferred==generic)return cached_application_module(generic);
    static std::mutex mutex;
    static std::unordered_set<std::filesystem::path> failed;
    static std::unordered_map<std::filesystem::path,std::weak_ptr<ApplicationModule>> validated;
    std::lock_guard lock(mutex);
    if(auto existing=validated[preferred].lock())return existing;
    if(!failed.contains(preferred)) {
        try {
            auto module=cached_application_module(preferred);
            const char* targets=module->symbol<decltype(&lfs_hip_train_targets)>("lfs_hip_train_targets")();
            if(!targets || !lfs_hip_has_architecture(targets,preferred.parent_path().filename().string()))
                throw std::runtime_error("HIP profile GPU architecture does not match its package directory");
            auto baseline=cached_application_module(generic);
            const char* profile_id=module->symbol<decltype(&lfs_hip_train_set_id)>("lfs_hip_train_set_id")();
            const char* generic_id=baseline->symbol<decltype(&lfs_hip_train_set_id)>("lfs_hip_train_set_id")();
            if(!profile_id || !generic_id || !std::strcmp(profile_id,"standalone-unversioned") || std::strcmp(profile_id,generic_id))
                throw std::runtime_error("HIP architecture profile belongs to a different module build");
            validated[preferred]=module;
            const auto architecture=preferred.parent_path().filename().string();
            std::fprintf(stderr,"[info] HIP architecture profile selected: %s\n",architecture.c_str());
            return module;
        }
        catch(const std::exception& error) {
            failed.insert(preferred);
            std::fprintf(stderr,"[warning] HIP architecture profile unavailable (%s); falling back to generic training module\n",error.what());
        }
    }
    return cached_application_module(generic);
}
}
