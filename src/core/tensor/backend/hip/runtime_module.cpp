/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "runtime_module.hpp"

#include <stdexcept>
#include <string>

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

namespace lfs::core::internal {
    namespace {
        std::string loader_error() {
#ifdef _WIN32
            return "Windows loader error " + std::to_string(GetLastError());
#else
            const auto* message = dlerror();
            return message ? message : "unknown loader error";
#endif
        }
    } // namespace

    HipRuntimeModule::HipRuntimeModule(const std::filesystem::path& path, const std::filesystem::path& startup_dependency) {
        if (!path.is_absolute())
            throw std::invalid_argument("HIP runtime module requires an absolute path");
#ifdef _WIN32
        constexpr auto flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
        HMODULE startup = nullptr;
        if (!startup_dependency.empty()) {
            if (!startup_dependency.is_absolute())
                throw std::invalid_argument("HIP startup dependency requires an absolute path");
            // HIPRTC must be initialized before the combined module registers
            // device images when the executable is in another directory.
            startup = LoadLibraryExW(startup_dependency.c_str(), nullptr, flags);
            if (!startup)
                throw std::runtime_error("Cannot load HIP startup dependency: " + loader_error());
        }
        module_ = LoadLibraryExW(path.c_str(), nullptr, flags);
        const auto error = GetLastError();
        if (startup)
            FreeLibrary(startup); // The module's import owns the successful dependency.
        SetLastError(error);
#else
        if (!startup_dependency.empty())
            throw std::invalid_argument("Explicit HIP startup dependencies are Windows-only");
        module_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!module_)
            throw std::runtime_error("Cannot load HIP runtime module: " + loader_error());
        try {
            api_.runtime_abi = reinterpret_cast<decltype(api_.runtime_abi)>(symbol("lfs_hip_runtime_abi"));
            const auto version = api_.runtime_abi();
            if (version != LFS_HIP_RUNTIME_ABI_VERSION)
                throw std::runtime_error("HIP runtime ABI mismatch: expected " +
                                         std::to_string(LFS_HIP_RUNTIME_ABI_VERSION) + ", got " + std::to_string(version));
#define LFS_HIP_RUNTIME_EXPORT(Name) api_.Name = reinterpret_cast<decltype(api_.Name)>(symbol("lfs_hip_" #Name));
#include "runtime_exports.def"
#undef LFS_HIP_RUNTIME_EXPORT
        } catch (...) {
            close();
            throw;
        }
    }

    HipRuntimeModule::~HipRuntimeModule() { close(); }

    void HipRuntimeModule::close() noexcept {
        if (!module_)
            return;
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(module_));
#else
        dlclose(module_);
#endif
        module_ = nullptr;
    }

    void* HipRuntimeModule::symbol(const char* name) const {
#ifdef _WIN32
        auto* address = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module_), name));
#else
        auto* address = dlsym(module_, name);
#endif
        if (!address)
            throw std::runtime_error(std::string("Missing HIP runtime export: ") + name);
        return address;
    }

} // namespace lfs::core::internal
