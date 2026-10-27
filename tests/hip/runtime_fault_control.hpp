/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <filesystem>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
namespace lfs::test {
    class FaultControl {
    public:
        explicit FaultControl(const std::filesystem::path& path) {
#ifdef _WIN32
            module_ = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (module_) {
                arm = reinterpret_cast<decltype(arm)>(GetProcAddress(module_, "lfs_hip_test_fail_once"));
                resources = reinterpret_cast<decltype(resources)>(GetProcAddress(module_, "lfs_hip_test_live_resources"));
            }
#else
            module_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (module_) {
                arm = reinterpret_cast<decltype(arm)>(dlsym(module_, "lfs_hip_test_fail_once"));
                resources = reinterpret_cast<decltype(resources)>(dlsym(module_, "lfs_hip_test_live_resources"));
            }
#endif
            if (!arm || !resources) {
                close();
                throw std::runtime_error("Missing explicit fault module hooks");
            }
        }
        FaultControl(const FaultControl&) = delete;
        FaultControl& operator=(const FaultControl&) = delete;
        ~FaultControl() { close(); }
        void (*arm)(int) = nullptr;
        int (*resources)() = nullptr;

    private:
        void close() {
#ifdef _WIN32
            if (module_)
                FreeLibrary(module_);
#else
            if (module_)
                dlclose(module_);
#endif
            module_ = nullptr;
        }
#ifdef _WIN32
        HMODULE module_ = nullptr;
#else
        void* module_ = nullptr;
#endif
    };
} // namespace lfs::test
