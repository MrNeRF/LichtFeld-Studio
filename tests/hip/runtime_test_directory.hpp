/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <filesystem>
#include <stdexcept>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

struct RuntimeDirectory {
#ifdef _WIN32
    DLL_DIRECTORY_COOKIE cookie = nullptr;
#endif
    explicit RuntimeDirectory(const std::filesystem::path& path) {
#ifdef _WIN32
        if (!path.empty()) {
            cookie = AddDllDirectory(path.c_str());
            if (!cookie)
                throw std::runtime_error("Cannot add explicit test SDK directory");
        }
#else
        (void)path;
#endif
    }
    RuntimeDirectory(const RuntimeDirectory&) = delete;
    RuntimeDirectory& operator=(const RuntimeDirectory&) = delete;
    ~RuntimeDirectory() {
#ifdef _WIN32
        if (cookie)
            RemoveDllDirectory(cookie);
#endif
    }
};
