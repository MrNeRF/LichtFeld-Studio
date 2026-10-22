/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "architecture_runtime.hpp"
#include <filesystem>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lfs::core::internal {
inline std::filesystem::path hip_executable_directory() {
#ifdef _WIN32
    std::wstring path(32768, L'\0');
    const auto length=GetModuleFileNameW(nullptr,path.data(),static_cast<DWORD>(path.size()));
    if(!length || length>=path.size())return {};
    path.resize(length);
    return std::filesystem::path(path).parent_path();
#else
    std::string path(32768,'\0');
    const auto length=readlink("/proc/self/exe",path.data(),path.size());
    if(length<=0 || static_cast<size_t>(length)>=path.size())return {};
    path.resize(length);
    return std::filesystem::path(path).parent_path();
#endif
}
inline std::filesystem::path hip_packaged_module(const std::filesystem::path& compiled,
                                                const std::filesystem::path& executable_directory) {
    if(!executable_directory.is_absolute())return compiled;
    const auto packaged=executable_directory/"hip"/compiled.filename();
    std::error_code error;
    return std::filesystem::is_regular_file(packaged,error)?packaged:compiled;
}
inline std::filesystem::path hip_profile_module(const std::filesystem::path& generic,
                                               std::string_view architecture) {
    architecture=lfs_hip_base_architecture(architecture);
    if(!architecture.starts_with("gfx") || architecture.size()<4 || architecture.size()>16)return generic;
    for(char digit:architecture.substr(3))
        if(!((digit>='0' && digit<='9') || (digit>='a' && digit<='f')))return generic;
    const auto profile=generic.parent_path()/"profiles"/std::string(architecture)/generic.filename();
    std::error_code error;
    return std::filesystem::is_regular_file(profile,error)?profile:generic;
}
} // namespace lfs::core::internal
